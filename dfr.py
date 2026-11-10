"""
dfr.py  (NEW STEP B - GPU, about 20-40 minutes for all checkpoints)
===================================================================
For every trained deep model (checkpoint) this script:

 1. Extracts the "penultimate features" (the numbers that go into the last layer) for the
    train / val / test images, the group benchmark (groups.csv) and the unseen objects.
 2. ERM scores: puts the features through the model's OWN last layer -> the normal predictions
    ("ERM" = empirical risk minimisation = ordinary training). Also saves two OOD scores:
       energy  = logsumexp(logits)                       (Liu et al., NeurIPS 2020)
       kNN     = -distance to 10th nearest training image (Sun et al., ICML 2022)
 3. DFR (Deep Feature Reweighting, Kirichenko et al., ICLR 2023): keeps the whole network
    frozen and re-trains ONLY the last layer (logistic regression) on a small, GROUP-BALANCED
    set built from the VALIDATION split: real bags, bags on grey, real jellyfish, jellyfish on
    grey and jellyfish on ROV backgrounds. The bright-blue background is NOT in this set, so the
    blue-background test stays a test on an unseen background. Takes seconds per model.

Everything is written to outputs/preds/<run>__erm.csv and <run>__dfr.csv for stats.py.

Run:   python dfr.py                       (main + underwater models, all seeds)
       python dfr.py --all                 (every checkpoint, incl. imbalance/ablation/crossdata runs)
       python dfr.py --runs binary_resnet50_pt_ft_aug-standard_s813
"""

import argparse
import gc
import re

import numpy as np
import pandas as pd
import torch

import common as C
import extra_common as X
from robustness import load_checkpoint

MAIN_PATTERN = re.compile(r"^(binary|fine)_(tinycnn|resnet50|efficientnet_b0|vit_b_16|vgg16)_(pt|scratch)_ft_"
                          r"aug-(standard|underwater)_s\d+$")


def head_module(model, name):
    if name == "tinycnn" or name.startswith("resnet"):
        return model.fc
    if name in ("efficientnet_b0", "vgg16"):
        return model.classifier[1] if name == "efficientnet_b0" else model.classifier[6]
    if name == "vit_b_16":
        return model.heads.head
    raise ValueError(name)


@torch.no_grad()
def features_and_logits(model, head, paths, batch=64):
    """Run the model once; a hook on the last layer catches its INPUT (= the features)."""
    store = {}
    h = head.register_forward_pre_hook(lambda m, inp: store.__setitem__("f", inp[0].detach()))
    feats, logits = [], []
    for i in range(0, len(paths), batch):
        x = X.to_model_input(X.load_uint8(paths[i:i + batch]))
        with torch.autocast(device_type=C.DEVICE.type, enabled=C.DEVICE.type == "cuda"):
            out = model(x)
        logits.append(out.float().cpu()); feats.append(store["f"].float().cpu())
    h.remove()
    return torch.cat(feats).numpy(), torch.cat(logits).numpy()


def process(ckpt, table, quick=False):
    run = ckpt.stem
    model, ck = load_checkpoint(ckpt)
    task, arch = ck["task"], ck["model"]
    n_classes = ck["num_classes"]
    head = head_module(model, arch)
    feats, logits = features_and_logits(model, head, table.path.tolist())
    probs = C.softmax_np(logits)
    is_train = (table.set == "train").values
    knn = X.knn_ood_score(feats[is_train], feats)
    energy = torch.logsumexp(torch.as_tensor(logits), dim=1).numpy()
    ev = ~is_train
    meta = {"family": "erm", "arch": arch, "aug": ck["args"]["aug"], "seed": ck["args"]["seed"], "run": run}
    p_erm = X.save_preds(f"{run}__erm", table[ev], probs[ev], task,
                         extra_cols={"energy": energy[ev], "knn": knn[ev]}, meta=meta)
    print(f"  [{run}] ERM"); X.quick_report(p_erm)

    # ---- DFR: last-layer retraining on the group-balanced validation set ----
    lab = "y_fine" if task == "fine" else "y_bin"
    rw = (table.set == "gval") & table.group.isin(X.DFR_GROUPS) & (table[lab] >= 0)
    if rw.sum() < 20:
        print("  not enough group images for DFR - run group_benchmark.py first"); return
    sc, clf, c_best = X.fit_group_balanced_head(feats[rw.values], table.loc[rw, lab].values.astype(int),
                                                table.loc[rw, "group"].values, table.loc[rw, "cluster"].values)
    p_dfr = X.full_proba(clf, sc.transform(feats), n_classes)
    meta.update({"family": "dfr", "dfr_C": c_best})
    p2 = X.save_preds(f"{run}__dfr", table[ev], p_dfr[ev], task,
                      extra_cols={"knn": knn[ev]}, meta=meta)
    print(f"  [{run}] DFR (C={c_best})"); X.quick_report(p2)
    del model; gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--skip_existing", action="store_true")
    args = ap.parse_args()
    C.seed_everything(); X.make_extra_dirs()
    table = X.eval_table(include_train=True)
    ckpts = sorted(C.CKPT_DIR.glob("*.pt"))
    if args.runs:
        ckpts = [c for c in ckpts if c.stem in args.runs]
    elif args.quick:
        ckpts = [c for c in ckpts if c.stem.startswith("QUICK_")][:2]
        table = table.groupby(["set", "group", "y_fine"]).head(6).reset_index(drop=True)
    elif not args.all:
        ckpts = [c for c in ckpts if MAIN_PATTERN.match(c.stem)]
    else:
        ckpts = [c for c in ckpts if not c.stem.startswith("QUICK_")]
    print(f"{len(ckpts)} checkpoints, {len(table)} images per checkpoint")
    print("Loading images into memory once ...")
    X.load_uint8(table.path.tolist())
    for ck in ckpts:
        if args.skip_existing and (X.PRED_DIR / f"{ck.stem}__dfr.csv").exists():
            print(f"[skip] {ck.stem}"); continue
        process(ck, table, args.quick)
    print(f"Done. Predictions in {X.PRED_DIR}")


if __name__ == "__main__":
    main()
