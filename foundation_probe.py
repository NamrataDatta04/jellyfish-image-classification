"""
foundation_probe.py  (NEW STEP C - GPU, about 10-20 minutes)
============================================================
Foundation models WITHOUT fine-tuning, scored on exactly the same images as the CNNs/ViTs:

  CLIP ViT-B/32 (Radford et al., 2021)  - zero-shot (text prompts, no training)
                                         - linear probe (logistic regression on frozen embeddings)
                                         - linear probe + DFR-style group-balanced training
  DINOv2 ViT-B/14 (Oquab et al., 2024)  - linear probe
                                         - linear probe + group-balanced training

Why: our fine-tuned models collapse when the background changes, while CLIP zero-shot did not.
The literature reports exactly this pattern - fine-tuning raises in-distribution accuracy but can
destroy the robustness of the pretrained features (Kumar et al., ICLR 2022; Wortsman et al.,
CVPR 2022). This script tests that claim on our task with a self-supervised (DINOv2) and a
language-supervised (CLIP) backbone.

The probe's regularisation C is chosen on the normal VALIDATION split (macro-F1), exactly like
clip_eval.py. The group-balanced probe uses the same validation group set as dfr.py.

Run:    python foundation_probe.py                 (both tasks, CLIP + DINOv2)
        python foundation_probe.py --models clip   (only CLIP)
        python foundation_probe.py --clip_model openai/clip-vit-large-patch14   (bigger CLIP, optional)
Output: outputs/preds/<task>_<model>_<variant>.csv
"""

import argparse

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

import common as C
import extra_common as X
from clip_eval import as_tensor   # handles both tensor and Hugging Face output objects

CLASS_TEXT = {
    "fine": ["plastic bag", "barrel jellyfish", "blue jellyfish", "compass jellyfish",
             "lion's mane jellyfish", "mauve stinger jellyfish", "moon jellyfish"],
    "binary": ["plastic bag", "jellyfish"],
}
PROMPTS = {"photo": "a photo of a {}.", "underwater": "a photo of a {} underwater in the sea."}


class Backbone:
    """Turns image paths into L2-normalised embeddings with a frozen pretrained model."""

    def __init__(self, kind, model_id):
        from transformers import AutoImageProcessor, AutoModel, CLIPModel, CLIPProcessor
        self.kind = kind
        if kind == "clip":
            self.processor = CLIPProcessor.from_pretrained(model_id)
            self.model = CLIPModel.from_pretrained(model_id).to(C.DEVICE).eval()
            self.logit_scale = self.model.logit_scale.exp().item()
        else:
            self.processor = AutoImageProcessor.from_pretrained(model_id)
            self.model = AutoModel.from_pretrained(model_id).to(C.DEVICE).eval()

    @torch.no_grad()
    def images(self, paths, batch=48):
        out = []
        for i in range(0, len(paths), batch):
            imgs = [C.open_rgb(p) for p in paths[i:i + batch]]
            inp = self.processor(images=imgs, return_tensors="pt").to(C.DEVICE)
            if self.kind == "clip":
                f = as_tensor(self.model.get_image_features(**inp))
            else:
                o = self.model(**inp)
                f = torch.cat([o.pooler_output, o.last_hidden_state[:, 1:].mean(1)], dim=1)  # CLS + mean patch
            out.append(F.normalize(f.float(), dim=-1).cpu())
        return torch.cat(out).numpy()

    @torch.no_grad()
    def texts(self, texts):
        inp = self.processor(text=texts, return_tensors="pt", padding=True).to(C.DEVICE)
        f = as_tensor(self.model.get_text_features(**inp))
        return F.normalize(f.float(), dim=-1).cpu().numpy()


def linear_probe(Xtr, ytr, Xva, yva):
    best = (-1, None, None)
    for c in [0.1, 1.0, 10.0, 100.0]:
        clf = LogisticRegression(C=c, max_iter=5000, class_weight="balanced").fit(Xtr, ytr)
        f1 = f1_score(yva, clf.predict(Xva), average="macro")
        if f1 > best[0]:
            best = (f1, clf, c)
    return best


def run_backbone(kind, model_id, table, tasks):
    print(f"\n=== {kind}: {model_id} ===")
    bb = Backbone(kind, model_id)
    feats = bb.images(table.path.tolist())
    short = "clip" if kind == "clip" and "base-patch32" in model_id else model_id.split("/")[-1].replace("-", "_")
    is_tr, is_va = (table.set == "train").values, (table.set == "val").values
    ev = (table.set != "train").values
    knn = X.knn_ood_score(feats[is_tr], feats)
    for task in tasks:
        lab = "y_fine" if task == "fine" else "y_bin"
        n_cls = len(C.class_names(task))
        meta = {"arch": short, "aug": "none", "seed": C.SEED}
        # zero-shot (CLIP only): prompt style chosen on the validation split
        if kind == "clip":
            best = (-1, None, None)
            for key, tpl in PROMPTS.items():
                T = bb.texts([tpl.format(c) for c in CLASS_TEXT[task]])
                pz = C.softmax_np(bb.logit_scale * feats @ T.T)
                f1 = f1_score(table.loc[is_va, lab], pz[is_va].argmax(1), average="macro")
                if f1 > best[0]:
                    best = (f1, pz, key)
            p = X.save_preds(f"{task}_{short}__zeroshot", table[ev], best[1][ev], task,
                             extra_cols={"knn": knn[ev]}, meta={**meta, "family": "zeroshot", "prompt": best[2]})
            print(f"  [{task}] {short} zero-shot (prompt '{best[2]}')"); X.quick_report(p)
        # linear probe on the normal training split
        f1, clf, c = linear_probe(feats[is_tr], table.loc[is_tr, lab].values, feats[is_va], table.loc[is_va, lab].values)
        p = X.save_preds(f"{task}_{short}__probe", table[ev], X.full_proba(clf, feats[ev], n_cls), task,
                         extra_cols={"knn": knn[ev]}, meta={**meta, "family": "probe", "C": c})
        print(f"  [{task}] {short} linear probe (C={c}, val F1 {f1:.3f})"); X.quick_report(p)
        # group-balanced probe (same recipe as DFR)
        rw = (table.set == "gval") & table.group.isin(X.DFR_GROUPS) & (table[lab] >= 0)
        if rw.sum() >= 20:
            sc, clf2, c2 = X.fit_group_balanced_head(feats[rw.values], table.loc[rw, lab].values.astype(int),
                                                     table.loc[rw, "group"].values, table.loc[rw, "cluster"].values)
            p = X.save_preds(f"{task}_{short}__probe_dfr", table[ev], X.full_proba(clf2, sc.transform(feats[ev]), n_cls),
                             task, extra_cols={"knn": knn[ev]}, meta={**meta, "family": "probe_dfr", "C": c2})
            print(f"  [{task}] {short} group-balanced probe (C={c2})"); X.quick_report(p)
    del bb; torch.cuda.empty_cache() if torch.cuda.is_available() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=["clip", "dinov2"], choices=["clip", "dinov2"])
    ap.add_argument("--clip_model", default="openai/clip-vit-base-patch32")
    ap.add_argument("--dino_model", default="facebook/dinov2-base")
    ap.add_argument("--tasks", nargs="*", default=["binary", "fine"])
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    C.seed_everything(); X.make_extra_dirs()
    table = X.eval_table(include_train=True)
    if args.quick:
        table = table.groupby(["set", "group", "y_fine"]).head(6).reset_index(drop=True)
    if "clip" in args.models:
        run_backbone("clip", args.clip_model, table, args.tasks)
    if "dinov2" in args.models:
        run_backbone("dinov2", args.dino_model, table, args.tasks)
    print(f"Done. Predictions in {X.PRED_DIR}")


if __name__ == "__main__":
    main()
