"""
insitu_experiment.py  (OPTIONAL STEP F2 - GPU, about 30-60 minutes)
===================================================================
Only if fathomnet_download.py produced enough FathomNet jellyfish (>= ~100 crops).

Question: is the near-perfect bag-vs-jellyfish score real, or does it come from the two classes
being photographed by different cameras? FathomNet jellyfish are filmed by ROVs, like our
TrashCan bags, so with them BOTH classes come from the same kind of camera.

We train the same binary ResNet-50 recipe as train.py (ImageNet weights, standard augmentation,
class-weighted loss, AdamW + cosine, early stopping on validation macro-F1, seeds 813-815) with
three different jellyfish sources, always with the SAME TrashCan bags:
    kaggle  : Kaggle clear-water jellyfish   (= our main models; their checkpoints are reused)
    insitu  : FathomNet ROV jellyfish
    both    : Kaggle + FathomNet jellyfish
and the same three recipes with a CLIP linear probe (cheap). Every model is tested on:
    test vs Kaggle jellies   (TrashCan test bags + Kaggle test jellyfish)
    test vs in-situ jellies  (TrashCan test bags + FathomNet test jellyfish)
    bag_clear / jelly_rov    (counter groups of the class x background benchmark)
    ROV background only      (share called "plastic bag" when there is NO object)

Run:    python insitu_experiment.py
Output: outputs/extra/table_insitu.md / .tex, outputs/extra/insitu_runs.csv
"""

import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import common as C
import extra_common as X
from robustness import load_checkpoint


def datasets():
    df = C.load_splits()
    f = pd.read_csv(X.FATHOM_CSV)
    f = f.assign(source="insitu", binary_id=1, fine_label="medusozoa_insitu", is_aug=False)
    bags = df[df.source == "trashcan"]
    kag = df[df.source == "jelly"]
    out = {}
    for split in ["train", "val", "test"]:
        b = bags[bags.split == split]
        out[("kaggle", split)] = pd.concat([b, kag[kag.split == split]])
        out[("insitu", split)] = pd.concat([b, f[f.split == split]])
        out[("both", split)] = pd.concat([b, kag[kag.split == split], f[f.split == split]])
    return out


def test_sets(D):
    g = pd.read_csv(X.GROUP_CSV) if X.GROUP_CSV.exists() else pd.DataFrame(columns=["set", "group", "path", "y_bin"])
    g = g[g.set == "gtest"].rename(columns={"y_bin": "binary_id"})
    sets = {"test vs Kaggle jellies": D[("kaggle", "test")], "test vs in-situ jellies": D[("insitu", "test")]}
    for grp in ["bag_clear", "jelly_rov", "bgonly_rov"]:
        sub = g[g.group == grp]
        if len(sub):
            sets[grp] = sub.assign(binary_id=sub.binary_id.clip(lower=0))   # background-only: label unused
    return sets


def score(probs, df, name):
    pred = probs.argmax(1)
    y = df.binary_id.values.astype(int)
    if name == "bgonly_rov":
        return float((pred == 0).mean())                 # share called "plastic bag" with no object
    if name in ("bag_clear", "jelly_rov"):
        return float((pred == y).mean())
    from sklearn.metrics import f1_score
    return float(f1_score(y, pred, average="macro"))


def train_resnet(D, src, seed, epochs=15, patience=5, quick=False):
    C.seed_everything(seed)
    tr, va = D[(src, "train")], D[(src, "val")]
    if quick:
        tr, va = tr.groupby("binary_id").head(16), va.groupby("binary_id").head(8); epochs = 1
    lab = "binary_id"
    loader = C.make_loader(tr, lab, C.build_transform("standard"), 32, shuffle=True)
    vloader = C.make_loader(va, lab, C.build_transform("none"), 64)
    model = C.build_model("resnet50", 2, pretrained=True).to(C.DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = torch.amp.GradScaler(enabled=C.DEVICE.type == "cuda")
    crit = nn.CrossEntropyLoss(weight=C.class_weights(tr[lab].values, 2).to(C.DEVICE))
    best, best_state, bad = -1, None, 0
    for ep in range(epochs):
        C.run_epoch(model, loader, crit, opt, scaler); sched.step()
        y, p = C.predict(model, vloader)
        f1 = C.compute_metrics(y, p, C.BINARY_CLASSES)["macro_f1"]
        print(f"   [{src} s{seed}] epoch {ep + 1:02d} val macro-F1 {f1:.3f}")
        if f1 > best:
            best, bad = f1, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model.eval()


def eval_torch(model, sets):
    tf = C.build_transform("none")
    return {n: score(C.predict(model, C.make_loader(d, "binary_id", tf, 64))[1], d, n) for n, d in sets.items()}


def clip_runs(D, sets):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score
    from clip_eval import ClipEncoder
    enc = ClipEncoder("openai/clip-vit-base-patch32")
    cache = {}

    def emb(df):
        key = tuple(df.path)
        if key not in cache:
            cache[key] = enc.encode_images(df.path.tolist()).numpy()
        return cache[key]

    rows = []
    for src in ["kaggle", "insitu", "both"]:
        tr, va = D[(src, "train")], D[(src, "val")]
        best = (-1, None)
        for c in [0.1, 1.0, 10.0, 100.0]:
            clf = LogisticRegression(C=c, max_iter=5000, class_weight="balanced").fit(emb(tr), tr.binary_id)
            f1 = f1_score(va.binary_id, clf.predict(emb(va)), average="macro")
            if f1 > best[0]:
                best = (f1, clf)
        rows.append({"model": "CLIP linear probe", "jelly source": src, "seed": C.SEED,
                     **{n: score(best[1].predict_proba(emb(d)), d, n) for n, d in sets.items()}})
        print(f"   CLIP probe [{src}] {rows[-1]}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="*", type=int, default=[813, 814, 815])
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no_clip", action="store_true")
    args = ap.parse_args()
    if not X.FATHOM_CSV.exists():
        raise SystemExit("FathomNet crops not found - run fathomnet_download.py first (or skip this optional step).")
    X.make_extra_dirs()
    D = datasets()
    sets = test_sets(D)
    print({k: len(v) for k, v in D.items()})
    rows = []
    for seed in args.seeds:
        ck = C.CKPT_DIR / f"binary_resnet50_pt_ft_aug-standard_s{seed}.pt"   # the main model = 'kaggle' source
        if ck.exists():
            model, _ = load_checkpoint(ck)
            rows.append({"model": "ResNet-50", "jelly source": "kaggle", "seed": seed, **eval_torch(model, sets)})
            print("   ", rows[-1])
        for src in ["insitu", "both"]:
            model = train_resnet(D, src, seed, quick=args.quick)
            rows.append({"model": "ResNet-50", "jelly source": src, "seed": seed, **eval_torch(model, sets)})
            print("   ", rows[-1])
            del model; torch.cuda.empty_cache() if torch.cuda.is_available() else None
        if args.quick:
            break
    if not args.no_clip:
        rows += clip_runs(D, sets)
    res = pd.DataFrame(rows)
    res.to_csv(X.EXTRA_DIR / "insitu_runs.csv", index=False)
    cols = [c for c in res.columns if c not in ("model", "jelly source", "seed")]
    g = res.groupby(["model", "jelly source"])
    mean, std, n = g[cols].mean(), g[cols].std(), g.size()
    tab = pd.DataFrame({"model": [i[0] for i in mean.index], "trained with jellyfish from": [i[1] for i in mean.index],
                        "seeds": n.values})
    for c in cols:
        label = {"bag_clear": "bag on clear bg (acc)", "jelly_rov": "jelly on ROV bg (acc)",
                 "bgonly_rov": "ROV bg only → 'bag'"}.get(c, c + " (macro-F1)")
        tab[label] = [f"{m:.3f}" if pd.isna(s) else f"{m:.3f} ± {s:.3f}" for m, s in zip(mean[c], std[c])]
    from summarize import to_latex, to_markdown
    (X.EXTRA_DIR / "table_insitu.md").write_text(to_markdown(tab), encoding="utf-8")
    (X.EXTRA_DIR / "table_insitu.tex").write_text(
        to_latex(tab, "Same-camera test with FathomNet in-situ jellyfish.").replace("→", r"$\rightarrow$"), encoding="utf-8")
    print(to_markdown(tab))


if __name__ == "__main__":
    main()
