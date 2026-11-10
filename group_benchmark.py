"""
group_benchmark.py  (NEW STEP A - no GPU needed, about 5-10 minutes)
====================================================================
Builds the "class x background" benchmark used to measure SHORTCUT LEARNING.

Why: in our data every plastic bag comes from deep-sea robot (ROV) video and every jellyfish
from bright, clear-water photos. A model can therefore score ~100% by recognising the
CAMERA / BACKGROUND instead of the object. To test this we build, for the validation AND
the test split (never mixing videos or duplicate groups between them):

  bag_rov      real TrashCan bag on its own ROV background          (= what the model was trained on)
  bag_gray     the same bag cut out with its outline and put on grey
  bag_clear    the same bag on a bright-blue, aquarium-like background (= sup_blue)
  jelly_clear  real jellyfish photo                                   (= what the model was trained on)
  jelly_gray   the same jellyfish cut out and put on grey
  jelly_rov    the same jellyfish put on a REAL ROV background (from a bag crop of the same split
               with the bag removed)
  bgonly_rov   ROV background with the bag removed (no object at all)
  bgonly_clear jellyfish photo with the jellyfish removed (no object at all)
  jelly_insitu (optional) FathomNet jellyfish filmed by ROVs, if fathomnet_download.py was run

Bags are cut out with the TrashCan polygon outlines. Jellyfish have no outlines, so we segment
them automatically: the photo border gives the water colours, pixels far from those colours are the
animal (Otsu threshold + clean-up). A quality check rejects masks that are too big or leak into the
background; those photos are skipped for the pasted groups and counted (reasons in jelly_mask_log.csv). A quality-check figure is saved so you can SEE the masks.
Objects are removed with a simple "normalised convolution" fill (blur of the surrounding pixels).
The SAME soft-edge pasting is used for bags and jellyfish, so pasting artefacts cannot tell the
classes apart.

It also runs the "low-level cue test" that explains why our underwater augmentation failed:
a simple logistic regression on 12 low-level numbers (colour, brightness, sharpness, saturation)
tries to tell bags from jellyfish, with and without the augmentation applied to BOTH classes.

Run:     python group_benchmark.py
Output:  outputs/groups.csv, outputs/crops/groups/..., outputs/extra/fig_group_examples.png,
         outputs/extra/fig_jelly_mask_check.png, outputs/extra/lowlevel_cue_test.csv
"""

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageFilter
from scipy import ndimage
from skimage.color import rgb2lab
from skimage.filters import threshold_otsu

import common as C
import extra_common as X

GRAY = (128, 128, 128)


# -----------------------------------------------------------------------------
# Masks
# -----------------------------------------------------------------------------
def bag_masks(bags):
    """Re-create the TrashCan polygon mask of every bag crop (same rules as prepare_data.py)."""
    import prepare_data as P
    tc_root = C.data_paths()["trashcan"]
    objects, _ = P.load_trashcan(tc_root)
    by_key = {f"{o['video']}_{o['ann_id']}": o for o in objects}
    masks = {}
    for _, r in bags.iterrows():
        o = by_key.get(Path(r.path).stem)
        if o is None:
            continue
        box = P.clean_box(o)
        if box is None:
            continue
        m = P.object_mask({"frame": str(tc_root / o["split_dir"] / o["file_name"]), "seg": o["seg"], "box": box})
        if np.asarray(m).mean() >= 255 * 0.05:          # same rule as the background-swap set
            masks[r.path] = m
    return masks


def jelly_mask(img):
    """
    Automatic jellyfish outline for a clear-water photo (no outlines exist for this dataset).
    1. the outer 8-pixel frame of the photo = water; k-means finds its 4 main colours (water is often a gradient)
    2. distance of every pixel to the NEAREST water colour (Lab space) -> Otsu threshold -> animal pixels
    3. clean up: opening, closing, fill holes; keep the largest piece (+ other big pieces away from the border)
    4. QUALITY CHECK - the mask is rejected (photo skipped for the pasted groups) when it covers more than
       50% of the photo, or reaches more than 8% of a ring 10 px inside the border (it has 'leaked' into the
       background). Returns (mask as PIL 'L' image or None, area fraction, reason).
    """
    from sklearn.cluster import KMeans
    x = np.asarray(img.convert("RGB").resize((224, 224)), dtype=np.float32) / 255.0
    lab = rgb2lab(x)
    b = 8
    border = np.concatenate([lab[:b].reshape(-1, 3), lab[-b:].reshape(-1, 3),
                             lab[:, :b].reshape(-1, 3), lab[:, -b:].reshape(-1, 3)])
    water = KMeans(4, n_init=3, random_state=0).fit(border).cluster_centers_
    dist = np.min(np.linalg.norm(lab[:, :, None, :] - water[None, None], axis=3), axis=2)
    dist = ndimage.gaussian_filter(dist, 1.5)
    try:
        t = threshold_otsu(dist)
    except ValueError:
        return None, 0.0, "flat image"
    m = dist > max(t, 10.0)
    m = ndimage.binary_opening(m, iterations=2)
    m = ndimage.binary_closing(m, iterations=3)
    m = ndimage.binary_fill_holes(m)
    lbl, n = ndimage.label(m)
    if n == 0:
        return None, 0.0, "nothing found"
    sizes = ndimage.sum(m, lbl, range(1, n + 1))
    order = np.argsort(-sizes)
    keep = np.zeros_like(m)
    for rank, i in enumerate(order):
        comp = lbl == (i + 1)
        if rank > 0 and sizes[i] < 0.3 * sizes[order[0]]:
            break
        if rank > 0 and np.concatenate([comp[0], comp[-1], comp[:, 0], comp[:, -1]]).mean() > 0.02:
            continue
        keep |= comp
    area = float(keep.mean())
    k = 10
    ring = np.concatenate([keep[k], keep[-k - 1], keep[k:-k, k], keep[k:-k, -k - 1]]).mean()
    if area < 0.03:
        return None, area, "too small"
    if area > 0.50:
        return None, area, "too big"
    if ring > 0.08:
        return None, area, "leaks into background"
    return Image.fromarray((keep * 255).astype(np.uint8)), area, "ok"


# -----------------------------------------------------------------------------
# Image operations
# -----------------------------------------------------------------------------
def remove_object(img, mask, grow=4, sigma=6.0, rounds=3):
    """Fill the object area with the surrounding background (normalised convolution, a simple inpainting)."""
    x = np.asarray(img.convert("RGB"), dtype=np.float32)
    m = np.asarray(mask.resize(img.size, Image.NEAREST)) > 127
    m = ndimage.binary_dilation(m, iterations=grow)
    known = (~m).astype(np.float32)
    out = x.copy()
    for k in range(rounds):                              # repeat with wider blur so big holes are filled too
        s = sigma * (2 ** k)
        num = np.stack([ndimage.gaussian_filter(out[..., c] * known, s) for c in range(3)], -1)
        den = ndimage.gaussian_filter(known, s)[..., None]
        fill = num / np.maximum(den, 1e-6)
        out[m] = fill[m]
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def paste(obj_img, mask, background):
    """Soft-edged paste (the same Gaussian blur of the outline as in prepare_data.py)."""
    bg = background.convert("RGB").resize(obj_img.size, Image.BILINEAR)
    soft = mask.resize(obj_img.size, Image.NEAREST).filter(ImageFilter.GaussianBlur(1))
    return Image.composite(obj_img.convert("RGB"), bg, soft)


def plain(size, color):
    return Image.new("RGB", size, color)


# -----------------------------------------------------------------------------
# Building the groups for one split
# -----------------------------------------------------------------------------
def build_split(df, split, masks, rng, out_rows, mask_log, max_jelly=None):
    tag = "gval" if split == "val" else "gtest"
    root = X.GROUP_DIR / tag
    for g in ["bag_rov", "bag_gray", "bag_clear", "jelly_clear", "jelly_gray", "jelly_rov",
              "bgonly_rov", "bgonly_clear", "mask"]:
        (root / g).mkdir(parents=True, exist_ok=True)

    def add(path, group, y_bin, y_fine, cluster, src, mask_path=""):
        out_rows.append({"path": str(path), "set": tag, "group": group, "y_bin": y_bin, "y_fine": y_fine,
                         "cluster": str(cluster), "src_path": src, "mask_path": str(mask_path)})

    # ---- bags ----
    bags = C.select_rows(df, split, "trashcan")
    bags = bags[bags.source == "trashcan"]
    backgrounds = []                                     # ROV backgrounds of THIS split (bag removed)
    for _, r in bags.iterrows():
        if r.path not in masks:
            continue
        img = C.open_rgb(r.path)
        m = masks[r.path].resize(img.size, Image.NEAREST)
        stem = Path(r.path).stem
        mp = root / "mask" / f"{stem}.png"; m.save(mp)
        add(r.path, "bag_rov", 0, 0, r.group, r.path, mp)
        for g, bg in [("bag_gray", plain(img.size, GRAY)), ("bag_clear", Image.fromarray(
                (C.bright_blue_background(img.size[1], img.size[0], rng) * 255).astype(np.uint8)))]:
            p = root / g / f"{stem}.jpg"; paste(img, m, bg).save(p, quality=95)
            add(p, g, 0, 0, r.group, r.path, mp)
        bgo = remove_object(img, m)
        p = root / "bgonly_rov" / f"{stem}.jpg"; bgo.save(p, quality=95)
        add(p, "bgonly_rov", -1, -1, r.group, r.path, mp)
        backgrounds.append((bgo, r.group))

    # ---- jellyfish ----
    jel = C.select_rows(df, split, "trashcan")
    jel = jel[jel.source == "jelly"]
    if max_jelly:
        jel = jel.head(max_jelly)
    for i, (_, r) in enumerate(jel.iterrows()):
        img = C.open_rgb(r.path).resize((224, 224), Image.BILINEAR)
        m, area, why = jelly_mask(img)
        mask_log.append({"split": split, "path": r.path, "fine_label": r.fine_label, "area": area,
                         "ok": m is not None, "reason": why})
        add(r.path, "jelly_clear", 1, r.fine_id, r.group, r.path)       # the real photo is always kept
        if m is None:
            continue
        stem = f"{r.fine_label}_{Path(r.path).stem}_{i}"
        mp = root / "mask" / f"{stem}.png"; m.save(mp)
        p = root / "jelly_gray" / f"{stem}.jpg"; paste(img, m, plain(img.size, GRAY)).save(p, quality=95)
        add(p, "jelly_gray", 1, r.fine_id, r.group, r.path, mp)
        if backgrounds:
            bg, bg_vid = backgrounds[int(rng.integers(len(backgrounds)))]
            p = root / "jelly_rov" / f"{stem}.jpg"; paste(img, m, bg).save(p, quality=95)
            add(p, "jelly_rov", 1, r.fine_id, r.group, r.path, mp)
        p = root / "bgonly_clear" / f"{stem}.jpg"; remove_object(img, m, grow=8).save(p, quality=95)
        add(p, "bgonly_clear", -1, -1, r.group, r.path, mp)


def add_fathomnet(out_rows):
    """FathomNet in-situ jellyfish (optional): val part -> gval, test part -> gtest, group 'jelly_insitu'."""
    if not X.FATHOM_CSV.exists():
        print("FathomNet jellyfish not found (optional) - skipping the jelly_insitu group.")
        return
    f = pd.read_csv(X.FATHOM_CSV)
    for _, r in f[f.split.isin(["val", "test"])].iterrows():
        out_rows.append({"path": r.path, "set": "gval" if r.split == "val" else "gtest", "group": "jelly_insitu",
                         "y_bin": 1, "y_fine": -1, "cluster": str(r.group), "src_path": r.path, "mask_path": ""})
    print(f"Added FathomNet in-situ jellyfish: {f.split.value_counts().to_dict()}")


# -----------------------------------------------------------------------------
# Figures
# -----------------------------------------------------------------------------
def figure_examples(g):
    import matplotlib.pyplot as plt
    order = ["bag_rov", "bag_gray", "bag_clear", "bgonly_rov", "jelly_clear", "jelly_gray", "jelly_rov", "bgonly_clear"]
    t = g[g.set == "gtest"]
    fig, axes = plt.subplots(3, len(order), figsize=(2.0 * len(order), 6.4))
    for c, grp in enumerate(order):
        rows = t[t.group == grp]
        for r_ in range(3):
            ax = axes[r_, c]; ax.axis("off")
            if r_ < len(rows):
                ax.imshow(Image.open(rows.iloc[r_ * max(1, len(rows) // 3)].path).convert("RGB").resize((160, 160)))
            if r_ == 0:
                ax.set_title(grp, fontsize=9)
    fig.suptitle("Class x background benchmark (test split). Real groups: bag_rov, jelly_clear. "
                 "Counter groups: bag_clear, jelly_rov.", fontsize=9)
    fig.tight_layout(); fig.savefig(X.EXTRA_DIR / "fig_group_examples.png", dpi=130); plt.close(fig)


def figure_masks(g):
    import matplotlib.pyplot as plt
    t = g[(g.set == "gtest") & (g.group == "jelly_gray")].sample(frac=1, random_state=1).head(14)
    if not len(t):
        return
    fig, axes = plt.subplots(2, len(t), figsize=(1.6 * len(t), 3.4))
    axes = np.array(axes).reshape(2, -1)
    for i, (_, r) in enumerate(t.iterrows()):
        axes[0, i].imshow(Image.open(r.src_path).convert("RGB").resize((160, 160))); axes[0, i].axis("off")
        x = np.asarray(Image.open(r.src_path).convert("RGB").resize((160, 160)), dtype=np.float32) / 255
        mk = np.asarray(Image.open(r.mask_path).resize((160, 160))) > 127
        x[~mk] *= 0.25
        axes[1, i].imshow(x); axes[1, i].axis("off")
    fig.suptitle("Quality check: automatic jellyfish masks (top: photo, bottom: kept area, background darkened)", fontsize=9)
    fig.tight_layout(); fig.savefig(X.EXTRA_DIR / "fig_jelly_mask_check.png", dpi=130); plt.close(fig)


# -----------------------------------------------------------------------------
# Low-level cue test: why the underwater augmentation could not remove the shortcut
# -----------------------------------------------------------------------------
def lowlevel_features(img):
    """12 numbers a 'source detector' could use: colour mean/std, brightness, sharpness, saturation."""
    x = np.asarray(img.convert("RGB").resize((224, 224), Image.BILINEAR), dtype=np.float32) / 255.0
    grey = x.mean(2)
    lap = ndimage.laplace(grey)
    hsv = np.asarray(img.convert("RGB").resize((224, 224)).convert("HSV"), dtype=np.float32) / 255.0
    return np.concatenate([x.mean((0, 1)), x.std((0, 1)),
                           [grey.mean(), grey.std(), np.log(lap.var() + 1e-8), np.abs(lap).mean(),
                            hsv[..., 1].mean(), hsv[..., 1].std()]])


def lowlevel_cue_test(df, seed=C.SEED):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    random.seed(seed); np.random.seed(seed)
    tr = C.select_rows(df, "train", "trashcan"); te = C.select_rows(df, "test", "trashcan")
    tr = tr[~tr.is_aug]                                   # unique photos only
    rows = []
    aug = C.DomainAugment(p=0.5)
    for cond in ["no augmentation", "underwater augmentation on BOTH classes"]:
        def feats(rows_):
            out = []
            for p in rows_.path:
                img = C.open_rgb(p)
                if cond != "no augmentation":
                    img = aug(img)
                out.append(lowlevel_features(img))
            return np.array(out)
        Xtr, Xte = feats(tr), feats(te)
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced"))
        clf.fit(Xtr, tr.binary_id.values)
        p = clf.predict_proba(Xte)[:, 1]
        acc = ((p > 0.5).astype(int) == te.binary_id.values).mean()
        auc = roc_auc_score(te.binary_id.values, p)
        rows.append({"condition": cond, "test_accuracy": acc, "test_auc": auc})
        print(f"  low-level cue classifier [{cond}]: accuracy {acc:.3f}, AUC {auc:.3f}")
    out = pd.DataFrame(rows)
    out.to_csv(X.EXTRA_DIR / "lowlevel_cue_test.csv", index=False)
    return out


# -----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="only a few images (to check the code)")
    ap.add_argument("--skip_cue_test", action="store_true")
    args = ap.parse_args()
    C.seed_everything(); X.make_extra_dirs()
    rng = np.random.default_rng(C.SEED)
    df = C.load_splits()

    bags = df[(df.source == "trashcan") & df.split.isin(["val", "test"])]
    if args.quick:
        bags = bags.groupby("split").head(6)
    print(f"1. Re-creating TrashCan bag outlines for {len(bags)} val/test bags ...")
    masks = bag_masks(bags)
    print(f"   outlines found for {len(masks)} bags")

    import shutil
    for old in ["gval", "gtest"]:                       # start clean, so no image from an older run is left over
        shutil.rmtree(X.GROUP_DIR / old, ignore_errors=True)
    rows, mask_log = [], []
    for split in ["val", "test"]:
        print(f"2. Building the groups for the {split} split ...")
        build_split(df[df.path.isin(bags.path) | (df.source == "jelly")], split, masks, rng, rows, mask_log,
                    max_jelly=6 if args.quick else None)
    add_fathomnet(rows)
    g = pd.DataFrame(rows)
    g.to_csv(X.GROUP_CSV, index=False)
    ml = pd.DataFrame(mask_log)
    ml.to_csv(X.EXTRA_DIR / "jelly_mask_log.csv", index=False)

    print("\nImages per group:")
    print(g.groupby(["set", "group"]).size().unstack(0).fillna(0).astype(int).to_string())
    print(f"\nJellyfish outline found for {ml.ok.mean():.0%} of jellyfish photos "
          f"(per species: {ml.groupby('fine_label').ok.mean().round(2).to_dict()})")
    print("Reasons:", ml.reason.value_counts().to_dict())
    figure_examples(g); figure_masks(g)
    if not args.skip_cue_test:
        print("3. Low-level cue test (explains the failed underwater augmentation) ...")
        lowlevel_cue_test(df if not args.quick else df.groupby(["split", "source"]).head(20))
    print(f"\nSaved {X.GROUP_CSV} and figures in {X.EXTRA_DIR}. LOOK at fig_jelly_mask_check.png.")


if __name__ == "__main__":
    main()
