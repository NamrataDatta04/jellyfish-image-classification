"""
gradcam.py  (STEP 6)
--------------------
Explainability with Grad-CAM (Selvaraju et al., ICCV 2017).
Grad-CAM shows which part of the image made the model choose a class.

Two analyses:
  1. Quantitative "does the model look at the bag?" test on the background-swap set.
     TrashCan gives the outline (mask) of every bag, so we can measure:
       - energy-in-mask: share of the Grad-CAM heat that falls on the bag (higher = better)
       - pointing game:  is the hottest pixel on the bag? (hit rate, higher = better)
     for the same TrashCan bag on its own deep-sea background, on grey, and on bright blue.
  2. Qualitative heat-map figures for the report (bags and jellyfish, including mistakes).

Run:  python gradcam.py                          (all checkpoints with seed 813)
      python gradcam.py --runs fine_resnet50_pt_ft_aug-standard_s813
"""

import argparse
import gc  # GRADCAM_MEMORY_PATCH

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt

import common as C
from robustness import load_checkpoint


class GradCAM:
    """
    How Grad-CAM works, in 4 steps:
      1. run the image through the model and remember the feature maps of one late layer;
      2. compute how much the score of the chosen class changes when each feature map changes (gradients);
      3. average those gradients -> one importance number per feature map;
      4. heat-map = ReLU(sum of feature maps x importance), resized to the image size (0 = unused, 1 = most used).
    """

    def __init__(self, model, layer):
        self.model = model
        self.acts = self.grads = None
        self.handle = layer.register_forward_hook(self._save)

    def _save(self, module, inputs, output):
        self.acts = output
        if output.requires_grad:  # only when we ask for gradients (not inside torch.no_grad)
            output.register_hook(lambda g: setattr(self, "grads", g))

    def __call__(self, x, target=None):
        """x: batch of normalised images. target: class ids (None = predicted class)."""
        x = x.to(C.DEVICE).clone().requires_grad_(True)  # makes gradients flow even for frozen models
        logits = self.model(x)
        if target is None:
            target = logits.argmax(1)
        target = torch.as_tensor(target, device=C.DEVICE).view(-1)
        self.model.zero_grad()
        logits.gather(1, target.view(-1, 1)).sum().backward()
        A, G = self.acts, self.grads
        if A.dim() == 3:  # ViT tokens (B, 1+N, D) -> drop class token -> (B, D, h, w)
            n = int((A.shape[1] - 1) ** 0.5)
            A = A[:, 1:].reshape(A.shape[0], n, n, -1).permute(0, 3, 1, 2)
            G = G[:, 1:].reshape(G.shape[0], n, n, -1).permute(0, 3, 1, 2)
        weights = G.mean(dim=(2, 3), keepdim=True)            # importance of each channel
        cam = F.relu((weights * A).sum(1, keepdim=True))       # weighted sum of feature maps
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
        cam = cam / (cam.flatten(1).max(1)[0].view(-1, 1, 1) + 1e-8)  # 0..1 per image
        out = (cam.detach().cpu().numpy(), F.softmax(logits.detach().float(), 1).cpu().numpy())
        self.acts = self.grads = None   # free the GPU memory held by the last forward pass
        return out


def load_batch(paths, tf):
    return torch.stack([tf(C.open_rgb(p)) for p in paths])


def mask_array(path):
    m = Image.open(path).convert("L").resize((C.IMG_SIZE, C.IMG_SIZE), Image.NEAREST)
    return np.asarray(m) > 127


def overlay(ax, img_tensor, cam, title):
    ax.imshow(C.denorm(img_tensor))
    ax.imshow(cam, cmap="jet", alpha=0.45)
    ax.set_title(title, fontsize=8)
    ax.axis("off")


def analyse_run(ckpt_path, df, n_fig=4, batch=4):
    run = ckpt_path.stem
    model, ck = load_checkpoint(ckpt_path)
    for p_ in model.parameters():
        p_.requires_grad_(False)   # only the input needs gradients -> much less memory
    names = C.class_names(ck["task"])
    cammer = GradCAM(model, C.gradcam_layer(model, ck["model"]))
    tf = C.build_transform("none")
    out_fig = C.FIG_DIR / "gradcam"
    out_fig.mkdir(parents=True, exist_ok=True)

    # ---- 1. background-swap: where does the model look when it says "plastic bag"? ----
    rows = []
    sup = df[df.split.str.startswith("sup_")]
    for bg in ["sup_orig", "sup_gray", "sup_blue"]:
        part = sup[sup.split == bg]
        for i in range(0, len(part), batch):
            chunk = part.iloc[i:i + batch]
            cams, probs = cammer(load_batch(chunk["path"], tf), target=[C.PLASTIC_ID] * len(chunk))
            for cam, prob, (_, r) in zip(cams, probs, chunk.iterrows()):
                mask = mask_array(r["mask_path"])
                peak = np.unravel_index(cam.argmax(), cam.shape)
                empty = cam.sum() == 0  # Grad-CAM found no positive evidence -> metrics undefined
                rows.append({"run": run, "task": ck["task"], "model": ck["model"], "aug": ck["args"]["aug"],
                             "bag_source": ck["args"].get("bag_source", C.DEFAULT_BAG_SOURCE),
                             "background": bg.replace("sup_", ""), "bag": r["mask_path"],
                             "energy_in_mask": np.nan if empty else float((cam * mask).sum() / cam.sum()),
                             "mask_area": float(mask.mean()),
                             "pointing_hit": np.nan if empty else float(mask[peak]),
                             "p_plastic": float(prob[C.PLASTIC_ID]),
                             "pred_plastic": bool(prob.argmax() == C.PLASTIC_ID)})
    res = pd.DataFrame(rows)

    # Figure: same bags on the three backgrounds
    groups = sup["mask_path"].unique()[:n_fig]          # one row of the figure per bag
    fig, axes = plt.subplots(n_fig, 3, figsize=(8, 2.7 * n_fig))
    for r_i, g in enumerate(groups):
        for c_i, bg in enumerate(["sup_orig", "sup_gray", "sup_blue"]):
            path = sup[(sup.mask_path == g) & (sup.split == bg)]["path"].iloc[0]
            x = load_batch([path], tf)
            cam, prob = cammer(x, target=[C.PLASTIC_ID])
            overlay(axes[r_i, c_i], x[0], cam[0],
                    f"{bg[4:]} | P(plastic)={prob[0, C.PLASTIC_ID]:.2f}\npred: {names[prob[0].argmax()]}")
    fig.suptitle(f"Grad-CAM for 'plastic_bag' on three backgrounds\n{run}", fontsize=9)
    fig.tight_layout(); fig.savefig(out_fig / f"{run}_background_swap.png", dpi=130); plt.close(fig)

    # ---- 2. test images: one per class + some mistakes ----
    test = C.select_rows(df, "test", ck["args"].get("bag_source", C.DEFAULT_BAG_SOURCE))
    lab = C.label_column(ck["task"])
    picks = test.groupby("fine_label").head(1)
    _, p_test = C.predict(model, C.make_loader(test, lab, tf, 32))
    pred = p_test.argmax(1)
    wrong = test[pred != test[lab].values].head(4)
    show = pd.concat([picks, wrong])
    cols = 4
    n_rows = int(np.ceil(len(show) / cols))
    fig, axes = plt.subplots(n_rows, cols, figsize=(10, 2.8 * n_rows))
    axes = np.array(axes).reshape(-1)
    for ax in axes:
        ax.axis("off")
    for k, (_, r) in enumerate(show.iterrows()):
        x = load_batch([r["path"]], tf)
        cam, prob = cammer(x)
        tag = "MISTAKE " if k >= len(picks) else ""
        overlay(axes[k], x[0], cam[0], f"{tag}true: {names[r[lab]]}\npred: {names[prob[0].argmax()]} ({prob[0].max():.2f})")
    fig.suptitle(f"Grad-CAM on test images (predicted class)\n{run}", fontsize=9)
    fig.tight_layout(); fig.savefig(out_fig / f"{run}_test_examples.png", dpi=130); plt.close(fig)
    cammer.handle.remove()
    return res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="*", default=None)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    C.seed_everything()
    df = C.load_splits()
    if args.quick:
        sup_bags = df[df.split == "sup_orig"]["mask_path"].head(8)
        df = pd.concat([df[df.split == "test"].groupby(["fine_label", "source"]).head(4), df[df.mask_path.isin(sup_bags)]])
    ckpts = sorted(C.CKPT_DIR.glob("*.pt"))
    if args.runs:
        ckpts = [c for c in ckpts if c.stem in args.runs]
    else:  # seed-813 models only; QUICK_ smoke-test models only in --quick mode
        ckpts = [c for c in ckpts if c.stem.endswith(f"_s{C.SEED}") and c.stem.startswith("QUICK_") == args.quick]
    all_res = []
    for ck in ckpts:
        print(f"Grad-CAM for {ck.stem} ...")
        all_res.append(analyse_run(ck, df))
        gc.collect(); torch.cuda.empty_cache()   # release this model before loading the next one
    if not all_res:
        print("No checkpoints found."); return
    res = pd.concat(all_res)
    res.to_csv(C.OUT_DIR / "gradcam_background_swap_per_image.csv", index=False)
    summary = res.groupby(["run", "background"]).agg(
        energy_in_mask=("energy_in_mask", "mean"), pointing_hit_rate=("pointing_hit", "mean"),
        mask_area=("mask_area", "mean"),  # energy_in_mask above this = better than a uniform heat-map
        plastic_recall=("pred_plastic", "mean"), mean_p_plastic=("p_plastic", "mean"),
        n=("bag", "count")).reset_index()
    summary.to_csv(C.OUT_DIR / "gradcam_background_swap_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
