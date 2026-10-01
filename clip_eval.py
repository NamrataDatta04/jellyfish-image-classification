"""
clip_eval.py  (STEP 4)
======================
Multimodal (image + text) baseline with CLIP, written in the same style as the Week 8 CLIP lab.

  (a) Zero-shot: NO training. CLIP compares each image with sentences such as
      "a photo of a moon jellyfish" and picks the most similar sentence.
      We try 3 sentence styles + their average (prompt-sensitivity study) and choose
      the style on the VALIDATION set.
  (b) Linear probe: CLIP image embeddings (512 numbers per image) + logistic regression
      trained on our training set, with C chosen on the validation set.

Both are tested on the clean test set and on all robustness sets.

Run:  python clip_eval.py --task fine      (and again with --task binary)
"""

import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from transformers import CLIPProcessor, CLIPModel

import common as C

# The words used for each class (same order as common.FINE_CLASSES / BINARY_CLASSES)
CLASS_TEXT = {
    "fine": ["plastic bag", "barrel jellyfish", "blue jellyfish", "compass jellyfish",
             "lion's mane jellyfish", "mauve stinger jellyfish", "moon jellyfish"],
    "binary": ["plastic bag", "jellyfish"],
}
PROMPT_SETS = {
    "short": "{}",
    "photo": "a photo of a {}.",
    "underwater": "a photo of a {} underwater in the sea.",
}


def as_tensor(features):
    """Return the embedding tensor from a tensor or a Hugging Face output object (function from the lab)."""
    if torch.is_tensor(features):
        return features
    for key in ["pooler_output", "image_embeds", "text_embeds"]:
        if getattr(features, key, None) is not None:
            return getattr(features, key)
    if isinstance(features, (tuple, list)):
        return features[0]
    raise TypeError(type(features))


class ClipEncoder:
    """Loads CLIP once and turns images / sentences into normalised embedding vectors."""

    def __init__(self, model_id):
        self.processor = CLIPProcessor.from_pretrained(model_id)
        self.model = CLIPModel.from_pretrained(model_id).to(C.DEVICE).eval()
        self.logit_scale = self.model.logit_scale.exp().item()

    @torch.no_grad()
    def encode_images(self, paths, pre=None, batch_size=64):
        feats = []
        for i in range(0, len(paths), batch_size):
            imgs = [C.open_rgb(p, pre) for p in paths[i:i + batch_size]]
            inputs = self.processor(images=imgs, return_tensors="pt").to(C.DEVICE)
            f = as_tensor(self.model.get_image_features(**inputs))
            feats.append(F.normalize(f.float(), dim=-1).cpu())
        return torch.cat(feats)

    @torch.no_grad()
    def encode_texts(self, texts):
        inputs = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(C.DEVICE)
        f = as_tensor(self.model.get_text_features(**inputs))
        return F.normalize(f.float(), dim=-1).cpu()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["fine", "binary"], default="fine")
    parser.add_argument("--bag_source", choices=C.BAG_SOURCES, default=C.DEFAULT_BAG_SOURCE)
    parser.add_argument("--clip_model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    C.seed_everything()
    C.make_dirs()
    (C.OUT_DIR / "robustness").mkdir(exist_ok=True)
    names, lab = C.class_names(args.task), C.label_column(args.task)
    tag = "QUICK_" if args.quick else ""
    src = "" if args.bag_source == C.DEFAULT_BAG_SOURCE else f"_bags-{args.bag_source}"

    df = C.load_splits()
    if args.quick:
        df = df.groupby(["split", "fine_label", "source"]).head(6)
    enc = ClipEncoder(args.clip_model)

    # ---- 1. image embeddings for every set (computed once, then reused) ----
    print("Encoding images with CLIP ...")
    emb, ys = {}, {}
    for split in ["train", "val"]:
        rows = C.select_rows(df, split, args.bag_source)
        emb[split], ys[split] = enc.encode_images(rows["path"].tolist()), rows[lab].values
    eval_sets = C.evaluation_sets(df, args.bag_source)
    for set_name, (rows, pre) in eval_sets.items():
        emb[set_name], ys[set_name] = enc.encode_images(rows["path"].tolist(), pre), rows[lab].values

    # ---- 2. (a) zero-shot with each prompt style ----
    text_emb = {k: enc.encode_texts([t.format(c) for c in CLASS_TEXT[args.task]]) for k, t in PROMPT_SETS.items()}
    text_emb["ensemble"] = F.normalize(torch.stack(list(text_emb.values())).mean(0), dim=-1)

    def zero_shot_probs(image_emb, key):
        return (enc.logit_scale * image_emb @ text_emb[key].T).softmax(dim=1).numpy()

    prompt_rows = []
    for key in text_emb:
        val_f1 = f1_score(ys["val"], zero_shot_probs(emb["val"], key).argmax(1), average="macro")
        test_m = C.compute_metrics(ys["test"], zero_shot_probs(emb["test"], key), names)
        prompt_rows.append({"task": args.task, "prompt_set": key, "val_macro_f1": val_f1,
                            "test_accuracy": test_m["accuracy"], "test_macro_f1": test_m["macro_f1"]})
        print(f"Zero-shot [{key:10s}] val F1 {val_f1:.3f} | test acc {test_m['accuracy']:.3f} F1 {test_m['macro_f1']:.3f}")
    prompt_df = pd.DataFrame(prompt_rows)
    prompt_df.to_csv(C.OUT_DIR / f"{tag}clip_prompt_sensitivity_{args.task}{src}.csv", index=False)
    best_prompt = prompt_df.sort_values("val_macro_f1", ascending=False).iloc[0]["prompt_set"]

    # ---- 3. (b) linear probe, C chosen on validation ----
    best = {"f1": -1}
    for c in [0.1, 1.0, 10.0, 100.0]:
        clf = LogisticRegression(C=c, max_iter=5000, class_weight="balanced").fit(emb["train"].numpy(), ys["train"])
        f1 = f1_score(ys["val"], clf.predict(emb["val"].numpy()), average="macro")
        print(f"Linear probe C={c:<6} val F1 {f1:.3f}")
        if f1 > best["f1"]:
            best = {"f1": f1, "clf": clf, "C": c}

    # ---- 4. test both methods on every evaluation set ----
    methods = {
        f"clip_zeroshot_{best_prompt}": (lambda e: zero_shot_probs(e, best_prompt), None),
        "clip_linear_probe": (lambda e: best["clf"].predict_proba(e.numpy()), best["f1"]),
    }
    rob_rows = []
    for label, (prob_fn, val_f1) in methods.items():
        for set_name in eval_sets:
            probs = prob_fn(emb[set_name])
            m = C.compute_metrics(ys[set_name], probs, names)
            rob_rows.append({"run": f"{args.task}_{label}{src}", "task": args.task, "model": label, "aug": "none",
                             "bag_source": args.bag_source, "seed": C.SEED, "eval_set": set_name,
                             "level": int(set_name[-1]) if set_name.startswith("test_uw") else 0,
                             "accuracy": m["accuracy"], "macro_f1": m.get("macro_f1"),
                             "plastic_recall": m["plastic_recall"], "mean_confidence": m["mean_confidence"],
                             "pred_plastic_rate": float((probs.argmax(1) == C.PLASTIC_ID).mean()),
                             "ood_auroc": C.ood_auroc(prob_fn(emb["test"]), probs) if set_name == "ood_other" else None})
        probs = prob_fn(emb["test"])
        test_m = C.compute_metrics(ys["test"], probs, names)
        run = f"{tag}{args.task}_{label}{src}"
        C.save_json({"run": run, "method": "clip", "task": args.task, "model": label, "pretrained": True,
                     "freeze": True, "aug": "none", "seed": C.SEED, "train_frac": 1.0,
                     "bag_source": args.bag_source, "imbalance": "weighted",
                     "clip_model": args.clip_model, "best_val_macro_f1": val_f1,
                     "hyperparameters": {"C": best["C"]} if "probe" in label else {"prompt_set": best_prompt},
                     "test": test_m,
                     "test_predictions": {"path": eval_sets["test"][0]["path"].tolist(),
                                          "y_true": ys["test"].tolist(), "y_pred": probs.argmax(1).tolist()}},
                    C.RESULTS_DIR / f"{run}.json")
        print(f"TEST {label}: acc {test_m['accuracy']:.3f} macro-F1 {test_m['macro_f1']:.3f}")
    pd.DataFrame(rob_rows).to_csv(C.OUT_DIR / "robustness" / f"{tag}clip_{args.task}{src}.csv", index=False)


if __name__ == "__main__":
    main()
