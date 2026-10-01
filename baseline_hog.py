"""
baseline_hog.py  (STEP 2)
=========================
Non-deep baselines. Each one comes from a COMP813 lab:

  majority    always predicts the most common training class
              ("naive baseline" idea of the Week 9 sequence lab)
  rule_blue   binary task only: a hand-made rule "bluer than a threshold -> jellyfish",
              with the threshold chosen on validation (like the rule-based detector in the Spam lab).
              It measures directly how much the background colour alone gives away.
  pixels      raw 32x32 pixels -> logistic regression / SVM   (SoftmaxBaseline of the Image Classification lab)
  hog         HOG edge/shape features on a grey image         (HOG lab)
  color       HSV colour histogram only
  hog+color   both together

For every learned feature set we run a VALIDATION grid search (C x classifier x HOG cell size),
like the grid searches in the Linear Regression, HOG and Spam labs; the grid is saved as a
table and a heat-map. The best setting is tested once on the test set and on the robustness sets.

Class imbalance for classic models (extra experiment, saved separately):
  with the best feature set we compare  none / class_weight="balanced" / random undersampling / SMOTE.
  SMOTE (Chawla et al., 2002) makes NEW minority examples by mixing a minority example with one
  of its nearest minority neighbours in feature space.

Run:  python baseline_hog.py --task fine      (and again with --task binary)
"""

import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score

import common as C

HOG_GRID = [dict(orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2)),
            dict(orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2))]
C_GRID = [0.01, 0.1, 1.0]
SIZE = 128        # image size for HOG / colour features
PIXEL_SIZE = 32   # image size for the raw-pixel baseline (as in the lab)


# ---------------------------------------------------------------------------
# Features: turn every image into a vector of numbers
# ---------------------------------------------------------------------------
def hog_features(img, hp):
    grey = np.asarray(img.convert("L").resize((SIZE, SIZE)), dtype=np.float32) / 255.0
    return hog(grey, block_norm="L2-Hys", feature_vector=True, **hp).astype(np.float32)


def color_features(img, bins=8):
    hsv = np.asarray(img.convert("HSV").resize((SIZE, SIZE))).reshape(-1, 3)
    h, _ = np.histogramdd(hsv, bins=(bins, bins, bins), range=((0, 256),) * 3)
    return (h.flatten() / hsv.shape[0]).astype(np.float32)


def pixel_features(img):
    return (np.asarray(img.resize((PIXEL_SIZE, PIXEL_SIZE)), dtype=np.float32) / 255.0).flatten()


def extract(df, feature, hp, pre=None):
    feats = []
    for p in df["path"]:
        img = C.open_rgb(p, pre)
        if feature == "pixels":
            feats.append(pixel_features(img))
            continue
        parts = []
        if "hog" in feature:
            parts.append(hog_features(img, hp))
        if "color" in feature:
            parts.append(color_features(img))
        feats.append(np.concatenate(parts))
    return np.vstack(feats)


def blue_score(df, pre=None):
    """Mean blue minus mean red of the whole picture (blue water gives a high score)."""
    out = []
    for p in df["path"]:
        a = np.asarray(C.open_rgb(p, pre).resize((64, 64)), dtype=np.float32) / 255.0
        out.append(a[..., 2].mean() - a[..., 0].mean())
    return np.array(out)


# ---------------------------------------------------------------------------
# Classifiers
# ---------------------------------------------------------------------------
def make_classifier(kind, c, balanced=True):
    cw = "balanced" if balanced else None
    if kind == "svm":
        return make_pipeline(StandardScaler(), LinearSVC(C=c, max_iter=5000, class_weight=cw))
    return make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=3000, class_weight=cw))


def scores_to_probs(clf, X):
    """Probabilities for AUC/ECE. An SVM gives scores, so we use a softmax of the scores."""
    if hasattr(clf, "predict_proba"):
        return clf.predict_proba(X)
    s = clf.decision_function(X)
    if s.ndim == 1:
        s = np.stack([-s / 2, s / 2], axis=1)
    return C.softmax_np(s)


def smote(X, y, k=5, seed=C.SEED):
    """
    SMOTE: for every class smaller than the biggest one, create new points
    x_new = x + u * (neighbour - x), u random in [0, 1], until all classes are the same size.
    """
    rng = np.random.default_rng(seed)
    counts = np.bincount(y)
    Xs, ys = [X], [y]
    for c in np.where((counts > 0) & (counts < counts.max()))[0]:
        Xc = X[y == c]
        need = counts.max() - len(Xc)
        nn = NearestNeighbors(n_neighbors=min(k + 1, len(Xc))).fit(Xc)
        neigh = nn.kneighbors(Xc, return_distance=False)[:, 1:]
        base = rng.integers(0, len(Xc), need)
        partner = neigh[base, rng.integers(0, neigh.shape[1], need)] if neigh.shape[1] else base
        u = rng.random((need, 1))
        Xs.append(Xc[base] + u * (Xc[partner] - Xc[base])); ys.append(np.full(need, c))
    return np.vstack(Xs), np.concatenate(ys)


def grid_search(feature, tr, va, lab, task):
    """Validation grid search. Returns the best model and the full grid as a table."""
    rows, best = [], {"f1": -1}
    grid = HOG_GRID if "hog" in feature else [None]
    for hp in grid:
        X_tr, X_va = extract(tr, feature, hp), extract(va, feature, hp)
        for kind in ["svm", "logreg"]:
            for c in C_GRID:
                clf = make_classifier(kind, c).fit(X_tr, tr[lab].values)
                f1 = f1_score(va[lab].values, clf.predict(X_va), average="macro")
                cell = f"{hp['pixels_per_cell'][0]}px" if hp else "-"
                rows.append({"feature": feature, "classifier": kind, "hog_cell": cell, "C": c, "val_macro_f1": f1})
                print(f"[{task}] {feature:9s} {kind:6s} C={c:<5} hog cell={cell:5s} val F1={f1:.3f}")
                if f1 > best["f1"]:
                    best = {"f1": f1, "clf": clf, "hp": hp, "kind": kind, "C": c, "feature": feature}
    print(f"==> best {feature}: {best['kind']} C={best['C']} hog={best['hp']} val F1={best['f1']:.3f}")
    return best, pd.DataFrame(rows)


def heatmap(grid_df, path, title):
    """Validation macro-F1 for every setting (like the validation heat-maps in the labs)."""
    piv = grid_df.pivot_table(index="C", columns=["classifier", "hog_cell"], values="val_macro_f1")
    fig, ax = plt.subplots(figsize=(1.6 * piv.shape[1] + 2, 0.8 * piv.shape[0] + 1.8))
    im = ax.imshow(piv.values, cmap="viridis", vmin=0, vmax=1, origin="lower")
    ax.set_xticks(range(piv.shape[1]), [f"{a}\n{b}" for a, b in piv.columns])
    ax.set_yticks(range(piv.shape[0]), piv.index)
    ax.set_ylabel("C")
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            ax.text(j, i, f"{piv.values[i, j]:.2f}", ha="center", va="center", color="w", fontsize=8)
    fig.colorbar(im, label="validation macro-F1"); ax.set_title(title)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["fine", "binary"], default="fine")
    parser.add_argument("--bag_source", choices=C.BAG_SOURCES, default=C.DEFAULT_BAG_SOURCE)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    C.seed_everything()
    C.make_dirs()
    for d in [C.OUT_DIR / "robustness", C.FIG_DIR / "validation"]:
        d.mkdir(parents=True, exist_ok=True)
    names, lab = C.class_names(args.task), C.label_column(args.task)
    tag = "QUICK_" if args.quick else ""
    src = "" if args.bag_source == C.DEFAULT_BAG_SOURCE else f"_bags-{args.bag_source}"

    df = C.load_splits()
    tr, va = C.select_rows(df, "train", args.bag_source), C.select_rows(df, "val", args.bag_source)
    if args.quick:
        tr, va = tr.groupby("fine_label").head(10), va.groupby("fine_label").head(5)
        df = pd.concat([tr, va, df[~df.split.isin(["train", "val"])].groupby(["split", "fine_label", "source"]).head(4)])
    eval_sets = C.evaluation_sets(df, args.bag_source)

    # Each baseline = (name, function giving class probabilities for some rows, hyper-parameters, val F1)
    baselines = []

    # 1. Majority class
    prior = np.bincount(tr[lab].values, minlength=len(names)) / len(tr)
    baselines.append(("majority", lambda rows, pre: np.tile(prior, (len(rows), 1)), {}, None))

    # 2. Rule "bluer = jellyfish" (binary only), threshold chosen on validation
    if args.task == "binary":
        s_va = blue_score(va)
        thresholds = np.quantile(s_va, np.linspace(0.05, 0.95, 19))
        f1s = [f1_score(va[lab].values, (s_va > t).astype(int), average="macro") for t in thresholds]
        thr = float(thresholds[int(np.argmax(f1s))])
        print(f"[binary] rule_blue: best threshold {thr:.3f} val F1={max(f1s):.3f}")

        def rule_probs(rows, pre, thr=thr):
            p_jelly = 1 / (1 + np.exp(-20 * (blue_score(rows, pre) - thr)))
            return np.stack([1 - p_jelly, p_jelly], axis=1)
        baselines.append(("rule_blue", rule_probs, {"threshold": thr}, max(f1s)))

    # 3-6. Learned features with a validation grid search
    all_grids, bests = [], []
    for feature in ["pixels", "hog", "color", "hog+color"]:
        best, grid_df = grid_search(feature, tr, va, lab, args.task)
        all_grids.append(grid_df); bests.append(best)
        heatmap(grid_df, C.FIG_DIR / "validation" / f"{tag}grid_{args.task}{src}_{feature.replace('+', '-')}.png",
                f"Validation search: {feature} ({args.task})")

        def probs_fn(rows, pre, best=best, feature=feature):
            return scores_to_probs(best["clf"], extract(rows, feature, best["hp"], pre))
        baselines.append((f"{feature}+{best['kind']}", probs_fn,
                          {"C": best["C"], "hog": str(best["hp"]), "classifier": best["kind"]}, best["f1"]))
    pd.concat(all_grids).sort_values("val_macro_f1", ascending=False).to_csv(
        C.OUT_DIR / f"{tag}classic_validation_grid_{args.task}{src}.csv", index=False)

    # Test once + robustness sets for every baseline
    rob_rows = []
    for label, probs_fn, hyper, val_f1 in baselines:
        suite = {}
        for set_name, (rows, pre) in eval_sets.items():
            probs = probs_fn(rows, pre)
            suite[set_name] = (rows[lab].values, probs)
            m = C.compute_metrics(rows[lab].values, probs, names)
            rob_rows.append({"run": f"{args.task}_classic_{label}{src}", "task": args.task, "model": label,
                             "aug": "none", "bag_source": args.bag_source, "seed": C.SEED, "eval_set": set_name,
                             "level": int(set_name[-1]) if set_name.startswith("test_uw") else 0,
                             "accuracy": m["accuracy"], "macro_f1": m.get("macro_f1"),
                             "plastic_recall": m["plastic_recall"], "mean_confidence": m["mean_confidence"],
                             "pred_plastic_rate": float((probs.argmax(1) == C.PLASTIC_ID).mean()),
                             "ood_auroc": C.ood_auroc(suite["test"][1], probs) if set_name == "ood_other" else None})
        y_test, p_test = suite["test"]
        test_metrics = C.compute_metrics(y_test, p_test, names)
        short = label.split("+svm")[0].split("+logreg")[0].replace("+", "-")
        run = f"{tag}{args.task}_classic_{short}{src}"
        C.save_json({"run": run, "method": "classic", "task": args.task, "model": label,
                     "pretrained": False, "freeze": False, "aug": "none", "seed": C.SEED, "train_frac": 1.0,
                     "bag_source": args.bag_source, "imbalance": "weighted",
                     "hyperparameters": hyper, "best_val_macro_f1": val_f1, "test": test_metrics,
                     "test_predictions": {"path": eval_sets["test"][0]["path"].tolist(),
                                          "y_true": y_test.tolist(), "y_pred": p_test.argmax(1).tolist()}},
                    C.RESULTS_DIR / f"{run}.json")
        C.plot_confusion(test_metrics["confusion_matrix"], names, C.FIG_DIR / "runs" / f"{run}_confusion.png", run)
        print(f"TEST {label}: acc {test_metrics['accuracy']:.3f} macro-F1 {test_metrics.get('macro_f1', 0):.3f}")
    pd.DataFrame(rob_rows).to_csv(C.OUT_DIR / "robustness" / f"{tag}classic_{args.task}{src}.csv", index=False)

    # Class-imbalance comparison for the best classic feature set (logistic regression, best C)
    if args.bag_source == C.DEFAULT_BAG_SOURCE:
        top = max(bests, key=lambda b: b["f1"])
        X_tr, y_tr = extract(tr, top["feature"], top["hp"]), tr[lab].values
        X_te, y_te = extract(eval_sets["test"][0], top["feature"], top["hp"]), eval_sets["test"][0][lab].values
        rows = []
        for method in ["none", "balanced", "undersample", "smote"]:
            X, y = X_tr, y_tr
            if method == "undersample":
                keep = C.undersample(tr.assign(_i=np.arange(len(tr))), lab, 1.0)["_i"].values
                X, y = X_tr[keep], y_tr[keep]
            if method == "smote":
                X, y = smote(StandardScaler().fit(X_tr).transform(X_tr), y_tr)   # SMOTE in scaled space
                X = StandardScaler().fit(X_tr).inverse_transform(X)
            clf = make_classifier("logreg", top["C"], balanced=(method == "balanced")).fit(X, y)
            m = C.compute_metrics(y_te, clf.predict_proba(X_te), names)
            rows.append({"task": args.task, "feature": top["feature"], "method": method, "train_size": len(y),
                         "accuracy": m["accuracy"], "macro_f1": m["macro_f1"],
                         "balanced_accuracy": m["balanced_accuracy"], "worst_class_recall": m["worst_class_recall"],
                         "plastic_recall": m["plastic_recall"]})
            print(f"[imbalance, classic {top['feature']}] {method:11s} macro-F1 {m['macro_f1']:.3f} "
                  f"worst-class recall {m['worst_class_recall']:.3f}")
        pd.DataFrame(rows).to_csv(C.OUT_DIR / f"{tag}classic_imbalance_{args.task}.csv", index=False)


if __name__ == "__main__":
    main()
