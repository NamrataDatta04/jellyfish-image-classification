"""
extra_common.py
===============
Shared helpers for the NEW experiments added on 12 Oct 2026 (the "shortcut study"):

  group_benchmark.py   builds the class x background test sets (groups.csv)
  dfr.py               re-scores every trained model on those sets + last-layer retraining (DFR)
  foundation_probe.py  CLIP / DINOv2 zero-shot and linear probes on the same sets
  insitu_experiment.py (optional) FathomNet in-situ jellyfish: test + small retrain
  stats.py             confidence intervals, worst-group accuracy, Holm-McNemar, cost analysis

Every scoring script writes ONE csv per method into outputs/preds/ with the same columns,
so stats.py can treat CNNs, ViTs, DFR heads and foundation models the same way.

Groups (the idea of Waterbirds / ImageNet-9 "background challenge"):
                         ROV background      clear (aquarium-like) background   grey
  plastic bag            bag_rov  (real)     bag_clear (pasted, = sup_blue)     bag_gray
  jellyfish (Medusozoa)  jelly_rov (pasted)  jelly_clear (real)                 jelly_gray
  background only        bgonly_rov          bgonly_clear                       -
The two REAL groups are the ones every model is trained on; the two PASTED groups are the
"counter" groups. A model that looks at the object should do well on all four.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

import common as C

PRED_DIR = C.OUT_DIR / "preds"            # one csv of per-image predictions per method
GROUP_DIR = C.CROP_DIR / "groups"         # images made by group_benchmark.py
GROUP_CSV = C.OUT_DIR / "groups.csv"      # table of those images
FATHOM_CSV = C.OUT_DIR / "fathomnet" / "fathomnet_jelly.csv"   # made by fathomnet_download.py (optional)
EXTRA_DIR = C.OUT_DIR / "extra"           # tables and figures of the new experiments

CORE_GROUPS = ["bag_rov", "bag_clear", "jelly_clear", "jelly_rov"]   # the 2 x 2 class x background matrix
SAME_BG = ["bag_rov", "jelly_clear"]       # background agrees with the training data
SWAPPED_BG = ["bag_clear", "jelly_rov"]    # background "belongs" to the other class
ALL_GROUPS = ["bag_rov", "bag_gray", "bag_clear", "jelly_clear", "jelly_gray", "jelly_rov",
              "jelly_insitu", "bgonly_rov", "bgonly_clear"]
# Groups used to RE-TRAIN the last layer in DFR. Blue/clear bags are left out on purpose:
# the bright-blue background is then never seen in training, so the test on it is a fair test.
DFR_GROUPS = ["bag_rov", "bag_gray", "jelly_clear", "jelly_gray", "jelly_rov", "jelly_insitu"]


def make_extra_dirs():
    for d in [PRED_DIR, GROUP_DIR, EXTRA_DIR]:
        d.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# The list of image sets every method is scored on
# -----------------------------------------------------------------------------
def eval_table(include_train=False):
    """
    One table with every image a method must score:
      set = train / val / test (normal splits, TrashCan bags) | gval / gtest (group images) | ood
      group = the group name (see top of file) or the split name for normal images
      y_bin = 0 plastic bag, 1 jellyfish, -1 no object / unknown object
      y_fine = fine class id (-1 if none), cluster = video or duplicate group (for the bootstrap)
    """
    df = C.load_splits()
    parts = []
    splits = (["train"] if include_train else []) + ["val", "test"]
    for s in splits:
        rows = C.select_rows(df, s, "trashcan")
        parts.append(pd.DataFrame({"path": rows.path, "set": s, "group": s, "y_bin": rows.binary_id,
                                   "y_fine": rows.fine_id, "cluster": rows.group.astype(str)}))
    ood = df[df.split == "ood_other"]
    parts.append(pd.DataFrame({"path": ood.path, "set": "ood", "group": "ood", "y_bin": -1, "y_fine": -1,
                               "cluster": ood.group.astype(str)}))
    if GROUP_CSV.exists():
        g = pd.read_csv(GROUP_CSV)
        parts.append(g[["path", "set", "group", "y_bin", "y_fine", "cluster"]])
    else:
        print(f"[warning] {GROUP_CSV} not found - run group_benchmark.py first. Scoring normal sets only.")
    out = pd.concat(parts, ignore_index=True)
    out["y_bin"] = out["y_bin"].astype(int)
    out["y_fine"] = out["y_fine"].astype(int)
    return out.reset_index(drop=True)


# -----------------------------------------------------------------------------
# Loading images once (fast: every model reuses the same decoded pixels)
# -----------------------------------------------------------------------------
_CACHE = {}


def load_uint8(paths):
    """Resize every image to 224 x 224 exactly like build_transform('none') and keep it as uint8 (small)."""
    resize = C.transforms.Resize((C.IMG_SIZE, C.IMG_SIZE))
    out = np.zeros((len(paths), C.IMG_SIZE, C.IMG_SIZE, 3), dtype=np.uint8)
    for i, p in enumerate(paths):
        if p not in _CACHE:
            _CACHE[p] = np.asarray(resize(C.open_rgb(p)), dtype=np.uint8)
        out[i] = _CACHE[p]
    return out


def to_model_input(batch_uint8):
    """uint8 (B,H,W,3) -> normalised float tensor on the GPU, identical to ToTensor + Normalize."""
    x = torch.from_numpy(batch_uint8).to(C.DEVICE).permute(0, 3, 1, 2).float() / 255.0
    mean = torch.tensor(C.MEAN, device=C.DEVICE).view(1, 3, 1, 1)
    std = torch.tensor(C.STD, device=C.DEVICE).view(1, 3, 1, 1)
    return (x - mean) / std


# -----------------------------------------------------------------------------
# Saving predictions in one common format
# -----------------------------------------------------------------------------
def save_preds(method, table, probs, task, extra_cols=None, meta=None):
    """
    method: unique name, e.g. 'binary_resnet50_pt_ft_aug-standard_s813__erm'
    probs : (N, K) class probabilities in the order of common.class_names(task)
    Plastic bag is class 0 in both tasks, so p_plastic = probs[:, 0] and the binary decision
    'is it a plastic bag?' = (argmax == 0) works for the binary AND the 7-class models.
    """
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    probs = np.asarray(probs, dtype=np.float64)
    pred = probs.argmax(1)
    out = table[["path", "set", "group", "y_bin", "y_fine", "cluster"]].copy()
    out["method"] = method
    out["task"] = task
    out["p_plastic"] = probs[:, C.PLASTIC_ID]
    out["pred_bin"] = np.where(pred == C.PLASTIC_ID, 0, 1)
    out["pred_fine"] = pred
    out["conf"] = probs.max(1)
    for k, v in (extra_cols or {}).items():
        out[k] = v
    for k, v in (meta or {}).items():
        out[k] = v
    out.to_csv(PRED_DIR / f"{method}.csv", index=False)
    return out


def quick_report(preds):
    """Print a few numbers so the user sees progress in the notebook."""
    r = {}
    for g in CORE_GROUPS:
        sub = preds[preds.group == g]
        if len(sub):
            r[g] = (sub.pred_bin == sub.y_bin).mean()
    if r:
        wg = min(r.values())
        txt = " | ".join(f"{k} {v:.2f}" for k, v in r.items())
        print(f"   group accuracy: {txt} | worst-group {wg:.2f}")
    t = preds[preds.set == "test"]
    if len(t):
        print(f"   normal test accuracy (bag vs jelly): {(t.pred_bin == t.y_bin).mean():.3f}")


# -----------------------------------------------------------------------------
# Linear heads (used by DFR and the foundation-model probes)
# -----------------------------------------------------------------------------
def fit_group_balanced_head(X, y, groups, clusters, Cs=(0.01, 0.03, 0.1, 0.3, 1.0, 3.0), seed=C.SEED):
    """
    Deep Feature Reweighting (Kirichenko, Izmailov & Wilson, ICLR 2023), in its simplest form:
    a logistic-regression head trained on a small set where every GROUP counts the same
    (sample weight = 1 / group size). The regularisation C is chosen by 4-fold cross-validation
    that keeps whole videos / duplicate groups together, maximising WORST-GROUP accuracy.
    Returns (scaler, classifier, chosen C).
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler

    groups = np.asarray(groups)
    gw = pd.Series(groups).map(1.0 / pd.Series(groups).value_counts()).values
    gw = gw / gw.mean()
    best = (-1.0, Cs[0])
    n_folds = min(4, len(np.unique(clusters)))
    for c in Cs:
        accs = []
        for tr, va in GroupKFold(n_splits=n_folds).split(X, y, clusters):
            if len(np.unique(y[tr])) < 2:
                continue
            sc = StandardScaler().fit(X[tr])
            clf = LogisticRegression(C=c, max_iter=3000).fit(sc.transform(X[tr]), y[tr], sample_weight=gw[tr])
            pred = clf.predict(sc.transform(X[va]))
            per_group = [np.mean(pred[groups[va] == g] == y[va][groups[va] == g])
                         for g in np.unique(groups[va])]
            accs.append(min(per_group))
        score = float(np.mean(accs)) if accs else -1
        if score > best[0]:
            best = (score, c)
    sc = StandardScaler().fit(X)
    clf = LogisticRegression(C=best[1], max_iter=5000).fit(sc.transform(X), y, sample_weight=gw)
    return sc, clf, best[1]


def full_proba(clf, X, n_classes):
    """predict_proba with a column for every class (a class missing from the DFR set gets probability 0)."""
    p = clf.predict_proba(X)
    out = np.zeros((len(X), n_classes))
    out[:, clf.classes_.astype(int)] = p
    return out


def knn_ood_score(train_feats, feats, k=10):
    """
    k-nearest-neighbour OOD score (Sun et al., ICML 2022): distance to the k-th closest TRAINING image
    in the normalised feature space. Larger distance = more unusual. We return -distance so that
    a higher score means 'looks like the training data' (the same direction as softmax confidence).
    """
    a = torch.nn.functional.normalize(torch.as_tensor(train_feats, dtype=torch.float32), dim=1)
    b = torch.nn.functional.normalize(torch.as_tensor(feats, dtype=torch.float32), dim=1)
    scores = []
    for i in range(0, len(b), 512):
        d = torch.cdist(b[i:i + 512], a)
        scores.append(-d.topk(min(k, a.shape[0]), largest=False).values[:, -1])
    return torch.cat(scores).numpy()
