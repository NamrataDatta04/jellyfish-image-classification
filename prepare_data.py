"""
prepare_data.py  (STEP 1)
=========================
Turns the three RAW Kaggle downloads into one clean, leak-free data table
(outputs/splits.csv) that every other script uses. Nothing is copied by hand:
running this script again gives exactly the same result.

Raw data (inside the Task folder):
  archive_jelly/Train_Test_Valid/{Train,valid,test}/<species>/   jellyfish photos, 6 species
  archive_trashcan/.../instance_version/                           TrashCan 1.0 (underwater trash, video frames + COCO boxes)
  archive_Plastic/Bag Classes/Bag Classes/<type> Bag Images/       synthetic bags pasted on land photos

What it does, step by step:
  A. JELLYFISH  - list all photos; find copies and near-copies (the dataset has "aug-*"
                  copies and some photos appear in both its train and test folders);
                  put copies in the same "group" so a photo can never be in train AND test.
  B. TRASHCAN   - cut every "trash_bag" object out of the video frames (a "crop"), after
                  CLEANING: remove the telemetry text / watermark bands, drop tiny crops,
                  drop near-identical crops from consecutive frames of the same video.
                  Each video is one group, because frames of one video look almost the same.
                  (TrashCan's own train/val split shares 113 of 115 videos, so we do NOT use it.)
  C. SYNTHETIC  - take 450 synthetic bags (225 "plastic" + 225 "garbage") for the
                  cross-dataset comparison (train on one bag source, test on the other).
  D. SPLIT      - 60% train / 20% validation / 20% test, by GROUP and separately for each class.
  E. EXTRA TEST SETS
                - background swap: TrashCan test bags cut out with their outline (mask) and
                  pasted on (1) their own background, (2) grey, (3) bright blue like the jelly photos;
                - out-of-distribution: other TrashCan objects the model never learns
                  (bottles, cans, containers, fish, starfish).
  F. CLASS BALANCE CHECK - counts per class and split, imbalance ratio, figures.

Run:  python prepare_data.py            (or:  python prepare_data.py --data_root "D:/.../Task")
"""

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter
import matplotlib.pyplot as plt

import common as C

IMG_EXT = {".jpg", ".jpeg", ".png"}


def list_images(folder):
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMG_EXT)


# ---------------------------------------------------------------------------
# Tools for finding copies of the same photo
# ---------------------------------------------------------------------------
def average_hash(img, size=16):
    """
    A 256-bit fingerprint of a picture: shrink to 16x16 grey, then 1 if a pixel is brighter
    than the average, else 0. Two versions of the same photo get almost the same bits.
    """
    a = np.asarray(img.convert("L").resize((size, size)), dtype=np.float32)
    return (a > a.mean()).flatten().astype(np.int32)


def hamming_matrix(hashes):
    """Number of different bits between every pair of fingerprints."""
    H = np.stack(hashes)
    return H @ (1 - H).T + (1 - H) @ H.T


class UnionFind:
    """Keeps track of which items belong together (in the same duplicate group)."""

    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, i):
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, i, j):
        self.parent[self.find(i)] = self.find(j)


# ---------------------------------------------------------------------------
# A. JELLYFISH
# ---------------------------------------------------------------------------
def build_jelly_table(jelly_root):
    """
    One row per jellyfish photo. We read only Train_Test_Valid/{Train,valid,test}, because the
    six species folders at the top of archive_jelly are exact copies of Train_Test_Valid/Train.
    """
    rows = []
    for split_dir in ["Train", "valid", "test"]:
        for folder, cls in C.JELLY_FOLDERS.items():
            for p in list_images(jelly_root / split_dir / folder):
                m = re.match(r"aug-\d+-(.+)", p.name)   # "aug-98-05.jpg" is a copy of "05.jpg"
                base = m.group(1) if m else p.name
                rows.append({"path": str(p), "source": "jelly", "fine_label": cls,
                             "orig_split": split_dir, "is_aug": bool(m),
                             "name_key": f"{split_dir}/{cls}/{base.rsplit('.', 1)[0]}"})
    return pd.DataFrame(rows)


def group_jellyfish(df):
    """Give every jellyfish photo a group id (copies share a group) and report what was found."""
    n = len(df)
    uf = UnionFind(n)
    # (a) "aug-<k>-<name>" files are copies of "<name>"
    first = {}
    for i, k in enumerate(df["name_key"]):
        if k in first:
            uf.union(i, first[k])
        else:
            first[k] = i
    # (b) photos that look almost the same (small fingerprint distance)
    print("  Hashing jellyfish photos to find near-duplicates ...")
    D = hamming_matrix([average_hash(Image.open(p)) for p in df["path"]])
    close = np.argwhere(np.triu(D <= C.DUP_HASH_BITS, 1))
    for i, j in close:
        uf.union(i, j)
    df["group"] = [f"jelly_{uf.find(i)}" for i in range(n)]

    labels, splits = df["fine_label"].values, df["orig_split"].values
    provided_eval = np.where(splits != "Train")[0]
    train_idx = np.where(splits == "Train")[0]
    leaked = [i for i in provided_eval if (D[i, train_idx] <= C.DUP_HASH_BITS).any()]
    report = {
        "images_total": int(n),
        "aug_copies": int(df["is_aug"].sum()),
        "near_duplicate_pairs": int(len(close)),
        "exact_duplicate_pairs": int(np.triu(D == 0, 1).sum()),
        "duplicate_pairs_with_different_species_label": int(sum(labels[i] != labels[j] for i, j in close)),
        "unique_photos(groups)": int(df["group"].nunique()),
        "provided_valid_test_images": int(len(provided_eval)),
        "provided_valid_test_images_with_copy_in_provided_train": int(len(leaked)),
    }
    return df, report


# ---------------------------------------------------------------------------
# B. TRASHCAN
# ---------------------------------------------------------------------------
def video_of(file_name):
    """'vid_000143_frame0000012.jpg' -> 'vid_000143'"""
    return re.match(r"(vid_\d+)", file_name).group(1)


def load_trashcan(tc_root):
    """Read both TrashCan annotation files into one list of objects (we re-split later by video)."""
    objects, frames_per_split = [], {}
    for split in ["train", "val"]:
        j = json.load(open(tc_root / f"instances_{split}_trashcan.json"))
        cats = {c["id"]: c["name"] for c in j["categories"]}
        ims = {i["id"]: i for i in j["images"]}
        frames_per_split[split] = {video_of(i["file_name"]) for i in ims.values()}
        for a in j["annotations"]:
            im = ims[a["image_id"]]
            objects.append({"split_dir": split, "file_name": im["file_name"], "W": im["width"], "H": im["height"],
                            "cls": cats[a["category_id"]], "bbox": a["bbox"], "seg": a["segmentation"],
                            "ann_id": f"{split}{a['id']}", "video": video_of(im["file_name"])})
    shared = frames_per_split["train"] & frames_per_split["val"]
    return objects, {"official_train_videos": len(frames_per_split["train"]),
                     "official_val_videos": len(frames_per_split["val"]),
                     "videos_in_both_official_train_and_val": len(shared)}


def clean_box(obj):
    """
    CLEANING RULE 1 + 2.
    The frames carry telemetry text at the top and bottom and a '(c) JAMSTEC' watermark,
    and 480x360 frames also have black bars. We keep only the clean middle band of the frame
    (common.TC_CLEAN_BAND) and cut the box to that band. The box is dropped if it becomes
    smaller than TC_MIN_SIDE pixels or loses more than 40% of its area.
    Returns the cleaned box (x0, y0, x1, y1) with a small margin, or None.
    """
    W, H = obj["W"], obj["H"]
    lo, hi = C.TC_CLEAN_BAND.get(H, (0.12, 0.80))
    x, y, w, h = obj["bbox"]
    x0, x1 = max(x, 0.01 * W), min(x + w, 0.99 * W)
    y0, y1 = max(y, lo * H), min(y + h, hi * H)
    cw, ch = x1 - x0, y1 - y0
    if cw < C.TC_MIN_SIDE or ch < C.TC_MIN_SIDE or cw * ch < C.TC_MIN_KEEP * w * h:
        return None
    mx, my = C.TC_MARGIN * cw, C.TC_MARGIN * ch   # a little context around the object
    return (max(x0 - mx, 0.01 * W), max(y0 - my, lo * H), min(x1 + mx, 0.99 * W), min(y1 + my, hi * H))


def crop_objects(objects, tc_root, wanted_classes, out_subdir, max_per_class=None, allowed_videos=None, rng=None):
    """
    Cut out objects of the wanted classes, clean them, remove near-copies inside each video
    (CLEANING RULE 3), and save each crop as a JPEG. Returns (table of crops, cleaning log).
    """
    out_dir = C.CROP_DIR / out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    cand = [o for o in objects if o["cls"] in wanted_classes
            and (allowed_videos is None or o["video"] in allowed_videos)]
    log = Counter(boxes=len(cand))
    rows, seen = [], defaultdict(list)   # seen[(video, class)] = fingerprints already kept
    for o in sorted(cand, key=lambda o: (o["video"], o["file_name"], o["ann_id"])):
        box = clean_box(o)
        if box is None:
            log["dropped_text_band_or_too_small"] += 1
            continue
        crop = Image.open(tc_root / o["split_dir"] / o["file_name"]).convert("RGB").crop(tuple(map(int, box)))
        fp = average_hash(crop)
        key = (o["video"], o["cls"])
        if any((fp != g).sum() <= C.TC_DUP_BITS for g in seen[key]):
            log["dropped_near_copy_same_video"] += 1
            continue
        seen[key].append(fp)
        path = out_dir / f"{o['video']}_{o['ann_id']}.jpg"
        crop.save(path, quality=95)
        rows.append({"path": str(path), "video": o["video"], "obj_class": o["cls"], "box": list(box),
                     "frame": str(tc_root / o["split_dir"] / o["file_name"]), "seg": o["seg"]})
    df = pd.DataFrame(rows)
    if max_per_class and len(df):   # keep a random sample of at most max_per_class crops per class
        df = df.sample(frac=1, random_state=int(rng.integers(1e6))).groupby("obj_class").head(max_per_class)
    log["kept"] = len(df)
    return df, dict(log)


def split_videos(bags, rng):
    """
    Assign whole VIDEOS to train/val/test so that about 60/20/20 of the CROPS land in each split.
    (Videos have very different numbers of crops, so we fill the split that is furthest below its target.)
    """
    counts = bags.groupby("video").size().to_dict()          # {video: number of crops}
    vids = list(counts)
    rng.shuffle(vids)
    vids = sorted(vids, key=lambda v: -counts[v])          # big videos first, ties in random order
    target = np.array(C.SPLIT_FRACTIONS) * sum(counts.values())
    filled = np.zeros(3)
    assign = {}
    for v in vids:
        k = int(np.argmax(target - filled))                  # the split that needs crops most
        assign[v] = ["train", "val", "test"][k]
        filled[k] += counts[v]
    return bags["video"].map(assign)


# ---------------------------------------------------------------------------
# D. GROUP-AWARE SPLIT FOR JELLYFISH AND SYNTHETIC BAGS
# ---------------------------------------------------------------------------
def group_split(df, rng):
    """Split GROUPS (not single photos) 60/20/20, separately for every class."""
    df = df.copy()
    df["split"] = ""
    group_class = df.groupby("group")["fine_label"].agg(lambda s: Counter(s).most_common(1)[0][0])
    for cls in sorted(group_class.unique()):
        groups = group_class[group_class == cls].index.to_numpy()
        rng.shuffle(groups)
        n_train = int(round(C.SPLIT_FRACTIONS[0] * len(groups)))
        n_val = int(round(C.SPLIT_FRACTIONS[1] * len(groups)))
        for k, g in enumerate(groups):
            df.loc[df["group"] == g, "split"] = "train" if k < n_train else ("val" if k < n_train + n_val else "test")
    # In val/test keep ONE photo per group (prefer the original over an aug copy), so copies cannot inflate scores.
    ev = df[df["split"].isin(["val", "test"])].sort_values("is_aug")
    drop = ev.index.difference(ev.drop_duplicates("group").index)
    return df.drop(index=drop)


# ---------------------------------------------------------------------------
# E. BACKGROUND-SWAP TEST SET
# ---------------------------------------------------------------------------
def object_mask(row):
    """The object's outline (COCO polygon in frame coordinates) drawn as a white mask inside the crop."""
    frame = Image.open(row["frame"])
    mask = Image.new("L", frame.size, 0)
    draw = ImageDraw.Draw(mask)
    for poly in row["seg"] if isinstance(row["seg"], list) else []:
        if len(poly) >= 6:
            draw.polygon(list(map(float, poly)), fill=255)
    return mask.crop(tuple(map(int, row["box"])))


def build_background_swap(test_bags, rng):
    """Same bag, three backgrounds: its own deep-sea background, grey, bright blue (like jellyfish photos)."""
    rows = []
    for d in ["orig", "gray", "blue", "mask"]:
        (C.CROP_DIR / "swap" / d).mkdir(parents=True, exist_ok=True)
    for _, r in test_bags.iterrows():
        img = Image.open(r["path"]).convert("RGB")
        mask = object_mask(r)
        if np.asarray(mask).mean() < 255 * 0.05:     # outline missing or tiny -> skip
            continue
        w, h = img.size
        soft = mask.filter(ImageFilter.GaussianBlur(1))
        grey = Image.fromarray((C.plain_background(h, w, (128, 128, 128)) * 255).astype(np.uint8))
        blue = Image.fromarray((C.bright_blue_background(h, w, rng) * 255).astype(np.uint8))
        stem = Path(r["path"]).stem
        mask_path = C.CROP_DIR / "swap" / "mask" / f"{stem}.png"
        mask.save(mask_path)
        for v, pic in {"orig": img, "gray": Image.composite(img, grey, soft),
                       "blue": Image.composite(img, blue, soft)}.items():
            out = C.CROP_DIR / "swap" / v / f"{stem}.jpg"
            pic.save(out, quality=95)
            rows.append({"path": str(out), "source": "swap", "fine_label": "plastic_bag", "split": f"sup_{v}",
                         "group": r["video"], "video": r["video"], "mask_path": str(mask_path), "is_aug": False})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# F. CLASS BALANCE CHECK + FIGURES
# ---------------------------------------------------------------------------
def balance_check(df):
    """Counts per class and split for both tasks (main setting: TrashCan bags), and the imbalance ratio."""
    out = {}
    for task in ["binary", "fine"]:
        lab, names = C.label_column(task), C.class_names(task)
        out[task] = {}
        for split in ["train", "val", "test"]:
            rows = C.select_rows(df, split, "trashcan")
            out[task][split] = C.imbalance_report(rows, lab, names)
        train = C.select_rows(df, "train", "trashcan")
        out[task]["train_unique_photos_only"] = C.imbalance_report(train[~train.is_aug], lab, names)
    return out


def figure_balance(df, bal):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), gridspec_kw={"width_ratios": [1, 2.6]})
    for ax, task in zip(axes, ["binary", "fine"]):
        names = C.class_names(task)
        x = np.arange(len(names))
        for k, (split, colour) in enumerate([("train", "tab:blue"), ("val", "tab:orange"), ("test", "tab:green")]):
            counts = [bal[task][split]["counts"][n] for n in names]
            ax.bar(x + (k - 1) * 0.27, counts, 0.27, label=split, color=colour)
        uniq = [bal[task]["train_unique_photos_only"]["counts"][n] for n in names]
        ax.scatter(x - 0.27, uniq, marker="_", s=250, color="k", label="train, unique photos only")
        ax.set_xticks(x, names, rotation=30, ha="right")
        ax.set_ylabel("images")
        ax.set_title(f"{task} task — train imbalance ratio {bal[task]['train']['imbalance_ratio']:.1f} : 1")
        ax.legend(fontsize=8)
    fig.suptitle("Class balance (plastic bags = TrashCan crops)")
    fig.tight_layout(); fig.savefig(C.FIG_DIR / "data_class_balance.png", dpi=150); plt.close(fig)


def figure_samples(df, rng, k=6):
    groups = [("plastic_bag (TrashCan)", df[(df.source == "trashcan") & (df.split == "train")])]
    groups += [(c, df[(df.fine_label == c) & (df.split == "train") & (~df.is_aug)]) for c in C.FINE_CLASSES[1:]]
    groups += [("plastic_bag (synthetic)", df[(df.source == "synthetic") & (df.split == "train")])]
    fig, axes = plt.subplots(len(groups), k, figsize=(2 * k, 2 * len(groups)))
    for r, (title, rows) in enumerate(groups):
        pick = rows.sample(min(k, len(rows)), random_state=int(rng.integers(1e6)))
        for c in range(k):
            ax = axes[r, c]; ax.axis("off")
            if c < len(pick):
                ax.imshow(Image.open(pick.iloc[c]["path"]).convert("RGB"))
            if c == 0:
                ax.set_title(title, fontsize=9, loc="left")
    fig.tight_layout(); fig.savefig(C.FIG_DIR / "data_samples.png", dpi=110); plt.close(fig)


def figure_cleaning(objects, tc_root):
    """One TrashCan frame with the clean band and the bag boxes kept / dropped — explains the cleaning rules."""
    bags = [o for o in objects if o["cls"] == C.TC_BAG_CLASS]
    frames = defaultdict(list)
    for o in bags:
        frames[(o["split_dir"], o["file_name"])].append(o)
    picks = sorted(frames.items(), key=lambda kv: -len(kv[1]))[:2]
    fig, axes = plt.subplots(1, len(picks), figsize=(6 * len(picks), 4))
    for ax, ((split, name), objs) in zip(np.atleast_1d(axes), picks):
        im = Image.open(tc_root / split / name).convert("RGB")
        W, H = im.size
        lo, hi = C.TC_CLEAN_BAND.get(H, (0.12, 0.80))
        ax.imshow(im)
        ax.axhspan(0, lo * H, color="red", alpha=0.25); ax.axhspan(hi * H, H, color="red", alpha=0.25)
        for o in objs:
            x, y, w, h = o["bbox"]
            keep = clean_box(o) is not None
            ax.add_patch(plt.Rectangle((x, y), w, h, fill=False, lw=2, color="lime" if keep else "red"))
        ax.set_title(f"{name}\nred band = text removed, green = kept, red box = dropped", fontsize=8)
        ax.axis("off")
    fig.tight_layout(); fig.savefig(C.FIG_DIR / "data_trashcan_cleaning.png", dpi=130); plt.close(fig)


def figure_underwater_demo(df):
    test = df[df.split == "test"]
    picks = [test[test.source == "trashcan"].iloc[0], test[test.source == "jelly"].iloc[0]]
    fig, axes = plt.subplots(2, 6, figsize=(12, 5))
    for r, row in enumerate(picks):
        img = Image.open(row["path"]).convert("RGB").resize((224, 224))
        for s in range(6):
            axes[r, s].imshow(C.underwater_effect(img, s)); axes[r, s].axis("off")
            axes[r, s].set_title(f"severity {s}", fontsize=9)
    fig.suptitle("Simulated underwater effect (robustness test)")
    fig.tight_layout(); fig.savefig(C.FIG_DIR / "data_underwater_severity.png", dpi=120); plt.close(fig)


def figure_swap(swap, k=4):
    groups = swap.drop_duplicates("video")["mask_path"].tolist()[:k]   # examples from different videos
    fig, axes = plt.subplots(len(groups), 4, figsize=(9, 2.3 * len(groups)))
    axes = np.atleast_2d(axes)
    for r, m in enumerate(groups):
        rows = swap[swap.mask_path == m].set_index("split")
        for c, (s, title) in enumerate([("sup_orig", "own background"), ("sup_gray", "grey"),
                                        ("sup_blue", "bright blue (like jelly photos)")]):
            axes[r, c].imshow(Image.open(rows.loc[s, "path"])); axes[r, c].set_title(title, fontsize=8)
        axes[r, 3].imshow(Image.open(m), cmap="gray"); axes[r, 3].set_title("TrashCan outline", fontsize=8)
        for ax in axes[r]:
            ax.axis("off")
    fig.tight_layout(); fig.savefig(C.FIG_DIR / "data_background_swap_examples.png", dpi=120); plt.close(fig)


# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default=None, help="Task folder that holds the archive_* folders")
    parser.add_argument("--seed", type=int, default=C.SEED)
    args = parser.parse_args()

    C.seed_everything(args.seed)
    rng = np.random.default_rng(args.seed)
    C.make_dirs()
    P = C.data_paths(args.data_root)
    for k, v in P.items():
        if not v.exists():
            raise FileNotFoundError(f"Missing {k} folder: {v}")
    summary = {}

    # ---- A. jellyfish ----
    print("A. Jellyfish")
    jelly, summary["jellyfish_duplicates"] = group_jellyfish(build_jelly_table(P["jelly"]))
    print(json.dumps(summary["jellyfish_duplicates"], indent=1))

    # ---- B. TrashCan bags ----
    print("B. TrashCan: cutting out and cleaning trash_bag objects ...")
    objects, summary["trashcan_official_split"] = load_trashcan(P["trashcan"])
    bags, summary["trashcan_bag_cleaning"] = crop_objects(objects, P["trashcan"], {C.TC_BAG_CLASS}, "trashcan_bags")
    bags["split"] = split_videos(bags, rng)
    bags = bags.assign(source="trashcan", fine_label="plastic_bag", group=bags["video"], is_aug=False)
    summary["trashcan_bag_cleaning"]["videos"] = int(bags["video"].nunique())
    summary["trashcan_bag_videos_per_split"] = bags.groupby("split")["video"].nunique().to_dict()
    print(summary["trashcan_official_split"], summary["trashcan_bag_cleaning"])

    # ---- C. synthetic bags (for the cross-dataset comparison) ----
    print("C. Synthetic bags")
    synth_rows = []
    for folder, kind in [("Plastic Bag Images", "plastic"), ("Garbage Bag Images", "garbage")]:
        files = list_images(P["synthetic_bags"] / folder)
        for i in rng.choice(len(files), C.N_SYNTH_PER_FOLDER, replace=False):
            synth_rows.append({"path": str(files[i]), "source": "synthetic", "fine_label": "plastic_bag",
                               "obj_class": f"synthetic_{kind}", "is_aug": False,
                               "group": f"synth_{kind}_{files[i].stem}"})

    # ---- D. split jellyfish + synthetic by group (TrashCan already split by video) ----
    print("D. Splitting 60/20/20 by group")
    js = group_split(pd.concat([jelly, pd.DataFrame(synth_rows)], ignore_index=True), rng)
    main = pd.concat([js, bags.drop(columns=["seg", "box", "frame"])], ignore_index=True)

    # ---- E. extra test sets ----
    print("E. Background-swap and out-of-distribution test sets")
    swap = build_background_swap(bags[bags.split == "test"], rng)
    train_videos = set(bags.loc[bags.split == "train", "video"])
    other_videos = {o["video"] for o in objects} - train_videos       # never use videos seen in training
    ood, summary["ood_cleaning"] = crop_objects(objects, P["trashcan"], set(C.TC_OOD_CLASSES), "ood",
                                                max_per_class=C.TC_OOD_PER_CLASS, allowed_videos=other_videos, rng=rng)
    ood = ood.drop(columns=["seg", "box", "frame"]).assign(source="ood", fine_label="unknown_object",
                                                          split="ood_other", group=ood["video"], is_aug=False)
    summary["ood_counts"] = ood["obj_class"].value_counts().to_dict()

    # ---- labels for both tasks ----
    df = pd.concat([main, swap, ood], ignore_index=True)
    df["is_aug"] = df["is_aug"].astype("boolean").fillna(False).astype(bool)
    fine_map = {c: i for i, c in enumerate(C.FINE_CLASSES)}
    df["fine_id"] = df["fine_label"].map(fine_map).fillna(-1).astype(int)          # unknown objects -> -1
    df["binary_id"] = np.where(df["fine_label"] == "plastic_bag", 0, np.where(df["fine_id"] >= 0, 1, -1))
    df.to_csv(C.SPLITS_CSV, index=False)

    # ---- F. class balance + figures ----
    print("F. Class balance check")
    summary["class_balance"] = balance_check(df)
    summary["split_sizes"] = df["split"].value_counts().to_dict()
    figure_balance(df, summary["class_balance"])
    figure_samples(df, rng)
    figure_cleaning(objects, P["trashcan"])
    figure_underwater_demo(df)
    if len(swap):
        figure_swap(swap)
    C.save_json(summary, C.OUT_DIR / "data_summary.json")
    for task in ["binary", "fine"]:
        b = summary["class_balance"][task]["train"]
        print(f"  {task}: train counts {b['counts']} -> imbalance ratio {b['imbalance_ratio']:.2f}:1")
    print("  Split sizes:", summary["split_sizes"])
    print(f"Saved {C.SPLITS_CSV}, {C.OUT_DIR / 'data_summary.json'} and figures in {C.FIG_DIR}")


if __name__ == "__main__":
    main()
