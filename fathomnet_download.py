"""
fathomnet_download.py  (OPTIONAL STEP F1 - internet, no GPU, about 15-40 minutes)
=================================================================================
Downloads IN-SITU jellyfish (Medusozoa) filmed by underwater robots (ROVs) from FathomNet
(Katija et al., Scientific Reports 2022; MBARI), cuts every jellyfish out of the frame and
saves the crops in  Task/archive_fathomnet/crops/ , with a table outputs/fathomnet/fathomnet_jelly.csv.

Why: our Kaggle jellyfish are clear-water photos, our bags are ROV video. FathomNet jellyfish are
ROV images too, so they let us test bag vs jellyfish with the SAME kind of camera.

Licence: FathomNet images are CC BY-NC-ND 4.0 (annotations CC BY-ND 4.0); using them to train and
test models for non-commercial research is allowed, but we must NOT redistribute the images.
So the repository contains only this script, never the crops.

Steps:
  1. find FathomNet concepts (names) that are medusae (true jellyfish + hydromedusae)
  2. download up to --per_concept images per concept, crop each jellyfish box (+10% margin)
  3. drop tiny boxes (< 48 px) and near-duplicate crops (same 256-bit hash rule as our other data)
  4. group crops by DIVE (images from the same dive look alike) and split the dives 60/20/20

Run:   pip install fathomnet
       python fathomnet_download.py                  (default: up to 60 images per concept, 700 crops max)
       python fathomnet_download.py --max_crops 400  (smaller / faster)
"""

import argparse
import io
import re
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import common as C
import extra_common as X

# Medusa genera / groups that appear in MBARI / FathomNet (siphonophores and ctenophores are NOT
# jellyfish in our sense and are excluded on purpose).
MEDUSA_NAMES = [
    "Medusae", "Scyphozoa", "Hydromedusae", "Trachymedusae", "Narcomedusae", "Semaeostomeae",
    "Coronatae", "Rhizostomeae", "Solmissus", "Aegina", "Aeginura", "Atolla", "Periphylla", "Poralia",
    "Mitrocoma", "Benthocodon", "Paraphyllina", "Colobonema", "Crossota", "Pantachogon", "Halitrephes",
    "Voragonema", "Earleria", "Chrysaora", "Aurelia", "Phacellophora", "Aequorea", "Pelagia", "Deepstaria",
    "Stygiomedusa", "Chiarella", "Ptychogastria", "Haliscera", "Botrynema", "Pegantha", "Cunina",
    "Tiaropsidium", "Calycopsis", "Leuckartiara", "Ptychogena", "Stomotoca", "Liriope", "Rhopalonema",
    "Sminthea", "Tetrorchis", "Bathykorus", "Nausithoe", "Tima", "Polyorchis", "Eutonina", "Proboscidactyla",
]
OUT_ROOT = C.DATA_ROOT / "archive_fathomnet" / "crops"
MIN_SIDE = 48


def is_medusa(concept):
    return any(concept == n or concept.startswith(n + " ") for n in MEDUSA_NAMES)


def find_concepts():
    from fathomnet.api import boundingboxes
    try:
        all_concepts = boundingboxes.find_concepts()
    except Exception as e:  # noqa: BLE001
        print("Could not list concepts:", e); all_concepts = []
    found = sorted({c for c in all_concepts if isinstance(c, str) and is_medusa(c)})
    if not found:                                        # fall back to the plain names
        found = MEDUSA_NAMES
    counts = {}
    for c in found:
        try:
            counts[c] = boundingboxes.count_by_concept(c).count
        except Exception:  # noqa: BLE001
            counts[c] = 0
    counts = {k: v for k, v in counts.items() if v > 0}
    print(f"Medusa concepts with boxes: {len(counts)}; total boxes {sum(counts.values())}")
    for k, v in sorted(counts.items(), key=lambda kv: -kv[1])[:25]:
        print(f"   {k:35s} {v}")
    return counts


def dive_of(url, timestamp):
    """MBARI urls look like .../framegrabs/<vehicle>/images/<dive>/<file>; else use the date."""
    m = re.search(r"framegrabs/([^/]+)/images/([^/]+)/", url or "")
    if m:
        return f"{m.group(1)}_{m.group(2)}"
    return f"date_{str(timestamp)[:10]}"


def average_hash(img, size=16):
    g = np.asarray(img.convert("L").resize((size, size), Image.BILINEAR), dtype=np.float32)
    return (g > g.mean()).flatten()


def fetch(url, tries=3):
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "COMP813-student-project"})
            with urllib.request.urlopen(req, timeout=30) as r:
                return Image.open(io.BytesIO(r.read())).convert("RGB")
        except Exception:  # noqa: BLE001
            time.sleep(1 + k)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_concept", type=int, default=60, help="max images downloaded per concept")
    ap.add_argument("--max_crops", type=int, default=700)
    ap.add_argument("--seed", type=int, default=C.SEED)
    args = ap.parse_args()
    from fathomnet.api import images as fn_images

    rng = np.random.default_rng(args.seed)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    X.FATHOM_CSV.parent.mkdir(parents=True, exist_ok=True)
    counts = find_concepts()
    rows, hashes = [], {}
    for concept in sorted(counts, key=lambda c: -counts[c]):
        if len(rows) >= args.max_crops:
            break
        try:
            ims = fn_images.find_by_concept(concept)
        except Exception as e:  # noqa: BLE001
            print(f"   {concept}: failed ({e})"); continue
        rng.shuffle(ims)
        got = 0
        for im in ims[: args.per_concept]:
            url = getattr(im, "url", None)
            boxes = [b for b in (getattr(im, "boundingBoxes", None) or []) if is_medusa(getattr(b, "concept", ""))]
            if not url or not boxes:
                continue
            img = fetch(url)
            if img is None:
                continue
            dive = dive_of(url, getattr(im, "timestamp", ""))
            for b in boxes:
                x, y, w, h = b.x, b.y, b.width, b.height
                if min(w, h) < MIN_SIDE:
                    continue
                mx, my = 0.10 * w, 0.10 * h
                box = (max(0, x - mx), max(0, y - my), min(img.width, x + w + mx), min(img.height, y + h + my))
                crop = img.crop(tuple(map(int, box)))
                fp = average_hash(crop)
                if any((fp != g).sum() <= C.TC_DUP_BITS for g in hashes.get(dive, [])):
                    continue                              # near-copy from the same dive
                hashes.setdefault(dive, []).append(fp)
                name = f"{re.sub('[^A-Za-z0-9]+', '_', concept)}_{getattr(im, 'uuid', len(rows))}_{len(rows)}.jpg"
                path = OUT_ROOT / name
                crop.save(path, quality=95)
                rows.append({"path": str(path), "concept": concept, "group": dive, "url": url,
                             "depth_m": getattr(im, "depthMeters", None), "w": crop.width, "h": crop.height})
                got += 1
        print(f"   {concept:35s} +{got} crops (total {len(rows)})")

    df = pd.DataFrame(rows)
    if df.empty:
        print("No crops downloaded. FathomNet may be unreachable - use the pasted jellyfish instead (still complete).")
        return
    # split whole dives 60/20/20 so the same dive never appears in train and test
    dives = df.group.unique(); rng.shuffle(dives)
    n = len(dives); cut1, cut2 = int(round(0.6 * n)), int(round(0.8 * n))
    split = {d: ("train" if i < cut1 else "val" if i < cut2 else "test") for i, d in enumerate(dives)}
    df["split"] = df.group.map(split)
    df.to_csv(X.FATHOM_CSV, index=False)
    print(f"\nSaved {len(df)} jellyfish crops from {n} dives -> {X.FATHOM_CSV}")
    print(df.split.value_counts().to_string())
    if len(df) < 100:
        print("WARNING: fewer than 100 crops - treat FathomNet as a small extra test only.")


if __name__ == "__main__":
    main()
