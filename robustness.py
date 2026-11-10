"""
robustness.py  (STEP 5)
=======================
Stress-tests every trained deep model (every checkpoint in outputs/checkpoints).

  test            the normal test set (for reference)
  test_tc         jellyfish + REAL underwater bags (TrashCan)    |  cross-dataset:
  test_syn        jellyfish + SYNTHETIC bags on land (Kaggle)     |  does it work on the other bag source?
  test_uw1/3/5    test images made to look further underwater (colour cast + haze + blur)
  test_bright     test images made darker (0.5, 0.75) or brighter (1.25, 1.5).
                  TrashCan bags are dark and jellyfish photos are bright, so a model that learned
                  "dark = bag" will change its answers here.
  test_noise      Gaussian pixel noise (sensor noise, particles in the water)
  test_fgsm       FGSM adversarial attack (Goodfellow et al., 2015): tiny pixel changes chosen to fool the model
  sup_orig/gray/blue  background swap: the SAME bag on its own background, grey, bright blue.
                  If "plastic recall" drops on bright blue, the model uses the background as a shortcut.
  ood_other       objects never seen in training (bottles, cans, containers, fish, starfish):
                  does the model at least show low confidence? (OOD AUROC, Hendrycks & Gimpel 2017)

Run:  python robustness.py                                  (all checkpoints)
      python robustness.py --runs fine_resnet50_pt_ft_aug-standard_s813
Output: outputs/robustness/deep_<run>.csv   (summarize.py turns these into tables and plots)
"""

import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

import common as C


class GaussianNoise:
    """Add Gaussian noise (sigma on a 0-1 pixel scale). A fixed seed gives the same noise every run."""

    def __init__(self, sigma, seed=C.SEED):
        self.sigma, self.rng = sigma, np.random.default_rng(seed)

    def __call__(self, img):
        x = np.asarray(img, dtype=np.float32) / 255.0
        x = np.clip(x + self.rng.normal(0, self.sigma, x.shape), 0, 1)
        return Image.fromarray((x * 255).astype(np.uint8))


def load_checkpoint(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = C.build_model(ck["model"], ck["num_classes"], pretrained=False)
    model.load_state_dict(ck["state_dict"])
    return model.to(C.DEVICE).eval(), ck


def fgsm_predict(model, loader, eps):
    """FGSM: move every pixel by eps (in 0-1 pixel units) in the direction that increases the loss most."""
    mean = torch.tensor(C.MEAN, device=C.DEVICE).view(1, 3, 1, 1)
    std = torch.tensor(C.STD, device=C.DEVICE).view(1, 3, 1, 1)
    ys, probs = [], []
    for x, y in loader:
        x, y = x.to(C.DEVICE), y.to(C.DEVICE)
        pix = (x * std + mean).detach().requires_grad_(True)          # back to 0-1 pixels
        loss = F.cross_entropy(model((pix - mean) / std), y)
        grad, = torch.autograd.grad(loss, pix)
        adv = (pix + eps * grad.sign()).clamp(0, 1).detach()
        with torch.no_grad():
            probs.append(F.softmax(model((adv - mean) / std).float(), 1).cpu().numpy())
        ys.append(y.cpu().numpy())
    return np.concatenate(ys), np.concatenate(probs)


def make_row(run, ck, set_name, level, y, probs, names, extra=None):
    m = C.compute_metrics(y, probs, names)
    a = ck["args"]
    r = {"run": run, "task": ck["task"], "model": ck["model"], "aug": a["aug"], "freeze": a["freeze"],
         "pretrained": not a["scratch"], "bag_source": a.get("bag_source", C.DEFAULT_BAG_SOURCE),
         "imbalance": a.get("imbalance", C.DEFAULT_IMBALANCE), "seed": a["seed"],
         "eval_set": set_name, "level": level, "accuracy": m["accuracy"], "macro_f1": m.get("macro_f1"),
         "plastic_recall": m["plastic_recall"], "mean_confidence": m["mean_confidence"],
         "pred_plastic_rate": float((probs.argmax(1) == C.PLASTIC_ID).mean())}
    r.update(extra or {})
    return r


def evaluate_run(ckpt_path, df):
    run = ckpt_path.stem
    model, ck = load_checkpoint(ckpt_path)
    names, lab = C.class_names(ck["task"]), C.label_column(ck["task"])
    tf = C.build_transform("none")
    sets = C.evaluation_sets(df, ck["args"].get("bag_source", C.DEFAULT_BAG_SOURCE))
    test = sets["test"][0]
    _, p_test = C.predict(model, C.make_loader(test, lab, tf, 64))
    rows = []
    for set_name, (sub, pre) in sets.items():
        y, p = C.predict(model, C.make_loader(sub, lab, tf, 64, pre_transform=pre))
        level = int(set_name[-1]) if set_name.startswith("test_uw") else 0
        extra = {"ood_auroc": C.ood_auroc(p_test, p)} if set_name == "ood_other" else {}
        rows.append(make_row(run, ck, set_name, level, y, p, names, extra))
    for factor in [0.5, 0.75, 1.25, 1.5]:
        y, p = C.predict(model, C.make_loader(test, lab, tf, 64, pre_transform=C.FixedBrightness(factor)))
        rows.append(make_row(run, ck, "test_bright", factor, y, p, names))
    for sigma in [0.05, 0.1, 0.2]:
        y, p = C.predict(model, C.make_loader(test, lab, tf, 64, pre_transform=GaussianNoise(sigma)))
        rows.append(make_row(run, ck, "test_noise", sigma, y, p, names))
    for eps in [1, 2, 4, 8]:
        y, p = fgsm_predict(model, C.make_loader(test, lab, tf, 32), eps / 255)
        rows.append(make_row(run, ck, "test_fgsm", eps, y, p, names))
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="*", default=None, help="checkpoint names (default: all)")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--skip_existing", action="store_true")
    args = parser.parse_args()
    C.seed_everything()
    out_dir = C.OUT_DIR / "robustness"
    out_dir.mkdir(parents=True, exist_ok=True)
    df = C.load_splits()
    if args.quick:
        df = df.groupby(["split", "fine_label", "source"]).head(6)
    ckpts = sorted(C.CKPT_DIR.glob("*.pt"))
    if args.runs:
        ckpts = [c for c in ckpts if c.stem in args.runs]
    for ck in ckpts:
        out = out_dir / f"deep_{ck.stem}.csv"
        if args.skip_existing and out.exists():
            print(f"[skip] {ck.stem}"); continue
        print(f"Robustness tests for {ck.stem} ...")
        res = evaluate_run(ck, df)
        res.to_csv(out, index=False)
        print(res[["eval_set", "level", "accuracy", "macro_f1", "plastic_recall"]].to_string(index=False))


if __name__ == "__main__":
    main()
