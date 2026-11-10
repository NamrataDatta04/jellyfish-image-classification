"""
prepare_data.py  (STEP 1)
-------------------------
Builds a clean, leak-free train/validation/test split from the RAW Kaggle folders
and creates the extra test sets used in the robustness study.

What it does:
  1. Reads all jellyfish images (6 species) from archive_jelly/Train_Test_Valid.
  2. Finds duplicates and near-duplicates. The jellyfish data contains many
     "aug-*" copies of the same photo, and some provided test images are copies
     of training images. We put copies in the same "group" and split by group,
     so the same photo can never be in both train and test.
  3. Samples plastic bag images (Plastic Bag + Garbage Bag folders).
  4. Samples paper bags as an out-of-distribution (OOD) test set.
  5. Builds the background-swap test set ("SUP"): the same bag, cut out with its
     COCO mask, placed on (a) its original land background, (b) plain grey,
     (c) a synthetic underwater background.
  6. Saves outputs/splits.csv, outputs/data_summary.json and data figures.

Run:  python prepare_data.py            (or: python prepare_data.py --data_root "D:/.../Task")
"""

import argparse
import json
import re
from collections import Counter

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter
import matplotlib.pyplot as plt

import common as C

IMG_EXT = {".jpg", ".jpeg", ".png"}


def list_images(folder):
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMG_EXT)


# ---------------------------------------------------------------------------
# Duplicate detection
# ---------------------------------------------------------------------------
def average_hash(path, size=16):
    """256-bit 'average hash': small grey thumbnail, 1 if pixel > mean. Similar images -> similar bits."""
    a = np.asarray(Image.open(path).convert("L").resize((size, size)), dtype=np.float32)
    return (a > a.mean()).flatten().astype(np.int32)


def hamming_matrix(hashes):
    """Number of different bits between every pair of hashes."""
    H = np.stack(hashes)
    return H @ (1 - H).T + (1 - H) @ H.T


class UnionFind:
    """Keeps track of which images belong to the same duplicate group."""

    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, i):
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, i, j):
        self.parent[self.find(i)] = self.find(j)


def build_jelly_table(jelly_root):
    rows = []
    for split_dir in ["Train", "valid", "test"]:
        for folder, cls in C.JELLY_FOLDERS.items():
            for p in list_images(jelly_root / split_dir / folder):
                m = re.match(r"aug-\d+-(.+)", p.name)
                base = m.group(1) if m else p.name
                rows.append({"path": str(p), "source": "jelly", "fine_label": cls,
                             "orig_split": split_dir, "is_aug": bool(m),
                             "name_key": f"{split_dir}/{cls}/{base.rsplit('.', 1)[0]}"})
    return pd.DataFrame(rows)


def group_jellyfish(df):
    """Give every jellyfish image a duplicate-group id and return a small report."""
    n = len(df)
    uf = UnionFind(n)
    # (a) aug-<k>-<name> files are copies of <name>
    first_of_key = {}
    for i, k in enumerate(df["name_key"]):
        if k in first_of_key:
            uf.union(i, first_of_key[k])
        else:
            first_of_key[k] = i
    # (b) images that look almost the same (small hash distance)
    print("Hashing jellyfish images to find near-duplicates ...")
    D = hamming_matrix([average_hash(p) for p in df["path"]])
    close = np.argwhere(np.triu(D <= C.DUP_HASH_BITS, 1))
    for i, j in close:
        uf.union(i, j)
    df["group"] = [f"jelly_{uf.find(i)}" for i in range(n)]

    # Report numbers for the Data section of the report
    labels = df["fine_label"].values
    splits = df["orig_split"].values
    provided_eval = np.where(splits != "Train")[0]
    train_idx = np.where(splits == "Train")[0]
    leaked = [i for i in provided_eval if (D[i, train_idx] <= C.DUP_HASH_BITS).any()]
    report = {
        "jelly_images_total": int(n),
        "jelly_aug_copies": int(df["is_aug"].sum()),
        "near_duplicate_pairs": int(len(close)),
        "exact_duplicate_pairs(hash distance 0)": int((np.triu(D == 0, 1)).sum()),
        "cross_class_duplicate_pairs": int(sum(labels[i] != labels[j] for i, j in close)),
        "unique_groups": int(df["group"].nunique()),
        "provided_valid_test_images": int(len(provided_eval)),
        "provided_valid_test_images_with_copy_in_train": int(len(leaked)),
    }
    return df, report


# ---------------------------------------------------------------------------
# Group-aware stratified split
# ---------------------------------------------------------------------------
def group_split(df, rng):
    """Split groups (not single images) into train/val/test, separately for each class."""
    df = df.copy()
    df["split"] = ""
    group_class = df.groupby("group")["fine_label"].agg(lambda s: Counter(s).most_common(1)[0][0])
    for cls in sorted(group_class.unique()):
        groups = group_class[group_class == cls].index.to_numpy()
        rng.shuffle(groups)
        n_train = int(round(C.SPLIT_FRACTIONS[0] * len(groups)))
        n_val = int(round(C.SPLIT_FRACTIONS[1] * len(groups)))
        # fractions are over UNIQUE photos (groups), not over copies
        for k, g in enumerate(groups):
            s = "train" if k < n_train else ("val" if k < n_train + n_val else "test")
            df.loc[df["group"] == g, "split"] = s
    # In val/test keep only ONE image per group (prefer the original, not an aug copy),
    # so near-identical copies cannot inflate the scores.
    ev = df[df["split"].isin(["val", "test"])].sort_values("is_aug")
    keep = ev.drop_duplicates("group").index
    drop = ev.index.difference(keep)
    return df.drop(index=drop)


# ---------------------------------------------------------------------------
# Background-swap test set (SUP) from COCO masks
# ---------------------------------------------------------------------------
def polygon_mask(segmentation, w, h):
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)
    for poly in segmentation:
        if len(poly) >= 6:
            draw.polygon(list(map(float, poly)), fill=255)
    return mask


def build_sup_set(coco_root, rng):
    """Same bag object on three backgrounds: original, grey, synthetic underwater."""
    coco = json.load(open(coco_root / "coco_instances.json"))
    cat_name = {c["id"]: c["name"] for c in coco["categories"]}
    anns = {}
    for a in coco["annotations"]:
        anns.setdefault(a["image_id"], []).append(a)
    # keep images with exactly ONE bag that is plastic or garbage (both are plastic bags)
    candidates = [im for im in coco["images"]
                  if len(anns.get(im["id"], [])) == 1
                  and cat_name[anns[im["id"]][0]["category_id"]] in ("plastic", "garbage")
                  and isinstance(anns[im["id"]][0]["segmentation"], list)]
    rng.shuffle(candidates)
    rows = []
    for d in ["orig", "gray", "water", "mask"]:
        (C.SUP_DIR / d).mkdir(parents=True, exist_ok=True)
    for im in candidates[:C.N_SUP]:
        a = anns[im["id"]][0]
        img = Image.open(coco_root / "images_raw" / im["file_name"]).convert("RGB")
        w, h = img.size
        mask = polygon_mask(a["segmentation"], w, h)
        if np.asarray(mask).mean() < 255 * 0.02:   # skip tiny / broken masks
            continue
        soft = mask.filter(ImageFilter.GaussianBlur(1))  # soft edge so the cut is not too sharp
        stem = f"{im['id']:08d}"
        grey = Image.new("RGB", (w, h), (128, 128, 128))
        water = Image.fromarray((C.synthetic_water_background(h, w, rng) * 255).astype(np.uint8))
        bag_uw = C.underwater_effect(img, rng.uniform(1.0, 3.0))  # the bag itself also looks underwater
        versions = {"orig": img,
                    "gray": Image.composite(img, grey, soft),
                    "water": Image.composite(bag_uw, water, soft)}
        mask.save(C.SUP_DIR / "mask" / f"{stem}.png")
        for v, pic in versions.items():
            out = C.SUP_DIR / v / f"{stem}.jpg"
            pic.save(out, quality=95)
            rows.append({"path": str(out), "source": "sup", "fine_label": "plastic_bag",
                         "split": f"sup_{v}", "group": f"sup_{stem}",
                         "bag_type": cat_name[a["category_id"]],
                         "mask_path": str(C.SUP_DIR / "mask" / f"{stem}.png")})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figures for the Data section
# ---------------------------------------------------------------------------
def figure_class_counts(df):
    main = df[df["split"].isin(["train", "val", "test"])]
    table = main.groupby(["fine_label", "split"]).size().unstack(fill_value=0)[["train", "val", "test"]]
    ax = table.plot(kind="bar", stacked=True, figsize=(8, 4))
    ax.set_ylabel("images"); ax.set_title("Images per class and split (after removing duplicates)")
    plt.xticks(rotation=30, ha="right"); plt.tight_layout()
    plt.savefig(C.FIG_DIR / "data_class_counts.png", dpi=150); plt.close()
    return table


def figure_samples(df, rng, k=6):
    fig, axes = plt.subplots(len(C.FINE_CLASSES), k, figsize=(2 * k, 2 * len(C.FINE_CLASSES)))
    for r, cls in enumerate(C.FINE_CLASSES):
        rows = df[(df.fine_label == cls) & (df.split == "train") & (~df.is_aug.astype(bool))]
        pick = rows.sample(min(k, len(rows)), random_state=int(rng.integers(1e6)))
        for c in range(k):
            ax = axes[r, c]; ax.axis("off")
            if c < len(pick):
                ax.imshow(Image.open(pick.iloc[c]["path"]).convert("RGB"))
            if c == 0:
                ax.set_title(cls, fontsize=9, loc="left")
    plt.tight_layout(); plt.savefig(C.FIG_DIR / "data_samples.png", dpi=120); plt.close()


def figure_underwater_demo(df):
    test = df[df.split == "test"]
    picks = [test[test.source == "bag"].iloc[0], test[test.source == "jelly"].iloc[0]]
    fig, axes = plt.subplots(2, 6, figsize=(12, 5))
    for r, row in enumerate(picks):
        img = Image.open(row["path"]).convert("RGB")
        for s in range(6):
            axes[r, s].imshow(C.underwater_effect(img, s)); axes[r, s].axis("off")
            axes[r, s].set_title(f"severity {s}", fontsize=9)
    plt.suptitle("Simulated underwater effect used for augmentation and robustness tests")
    plt.tight_layout(); plt.savefig(C.FIG_DIR / "data_underwater_severity.png", dpi=130); plt.close()


def figure_sup_examples(sup, k=4):
    stems = sup["group"].unique()[:k]
    fig, axes = plt.subplots(k, 4, figsize=(9, 2.3 * k))
    for r, g in enumerate(stems):
        rows = sup[sup.group == g].set_index("split")
        panels = [("sup_orig", "original background"), ("sup_gray", "grey background"),
                  ("sup_water", "synthetic underwater")]
        for c, (s, title) in enumerate(panels):
            axes[r, c].imshow(Image.open(rows.loc[s, "path"])); axes[r, c].set_title(title, fontsize=9)
        axes[r, 3].imshow(Image.open(rows.iloc[0]["mask_path"]), cmap="gray"); axes[r, 3].set_title("COCO mask", fontsize=9)
        for ax in axes[r]:
            ax.axis("off")
    plt.tight_layout(); plt.savefig(C.FIG_DIR / "data_background_swap_examples.png", dpi=120); plt.close()


# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default=None, help="Task folder with archive_jelly and archive_Plastic")
    parser.add_argument("--seed", type=int, default=C.SEED)
    args = parser.parse_args()

    C.seed_everything(args.seed)
    rng = np.random.default_rng(args.seed)
    C.make_dirs()
    P = C.data_paths(args.data_root)
    for k, v in P.items():
        if not v.exists():
            raise FileNotFoundError(f"Missing {k} folder: {v}")

    # 1-2. Jellyfish + duplicate groups
    jelly, dup_report = group_jellyfish(build_jelly_table(P["jelly"]))
    print(json.dumps(dup_report, indent=1))

    # 3. Plastic bags (half shopping bags, half garbage bags)
    bag_rows = []
    for folder, bag_type in [("Plastic Bag Images", "plastic"), ("Garbage Bag Images", "garbage")]:
        files = list_images(P["bags"] / folder)
        for i in rng.choice(len(files), C.N_BAG_PER_FOLDER, replace=False):
            bag_rows.append({"path": str(files[i]), "source": "bag", "fine_label": "plastic_bag",
                             "bag_type": bag_type, "orig_split": "none", "is_aug": False,
                             "group": f"bag_{bag_type}_{files[i].stem}"})
    bags = pd.DataFrame(bag_rows)

    # Split jellyfish and bags together (each class separately, by group)
    main = group_split(pd.concat([jelly, bags], ignore_index=True), rng)

    # 4. Paper bags = OOD test only
    paper_files = list_images(P["bags"] / "Paper Bag Images")
    ood = pd.DataFrame([{"path": str(paper_files[i]), "source": "paper", "fine_label": "paper_bag",
                         "split": "ood_paper", "group": f"paper_{paper_files[i].stem}", "bag_type": "paper"}
                        for i in rng.choice(len(paper_files), C.N_OOD_PAPER, replace=False)])

    # 5. Background-swap set
    print("Building background-swap (SUP) test set from COCO masks ...")
    sup = build_sup_set(P["coco"], rng)

    # Label ids for both tasks
    df = pd.concat([main, ood, sup], ignore_index=True)
    df["is_aug"] = df["is_aug"].astype("boolean").fillna(False).astype(bool)
    fine_map = {c: i for i, c in enumerate(C.FINE_CLASSES)}
    df["fine_id"] = df["fine_label"].map(fine_map).fillna(-1).astype(int)   # paper bag -> -1 (no class)
    df["binary_id"] = np.where(df["fine_label"] == "plastic_bag", 0,
                               np.where(df["fine_id"] >= 0, 1, -1))
    df.to_csv(C.SPLITS_CSV, index=False)

    # 6. Summary + figures
    table = figure_class_counts(df)
    figure_samples(df, rng)
    figure_underwater_demo(df)
    figure_sup_examples(sup)
    summary = {"duplicates": dup_report,
               "counts_per_class_and_split": table.to_dict(),
               "split_sizes": df["split"].value_counts().to_dict(),
               "train_aug_copies_kept": int(df[(df.split == "train") & df.is_aug].shape[0])}
    C.save_json(summary, C.OUT_DIR / "data_summary.json")
    print(table)
    print("Split sizes:", df["split"].value_counts().to_dict())
    print(f"Saved {C.SPLITS_CSV} and figures in {C.FIG_DIR}")


if __name__ == "__main__":
    main()
