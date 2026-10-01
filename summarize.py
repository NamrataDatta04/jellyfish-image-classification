"""
summarize.py  (STEP 7)
======================
Collects every result file and makes the tables and figures for the report.

Outputs (in outputs/summary/):
  key_numbers.md                all main tables in one file (easiest place to copy numbers from)
  results_all_runs.csv          one row per run (every seed)
  results_mean_std.csv          mean +- std over seeds for every configuration
  table_main_<task>.md/.tex     main comparison (Experiments section)
  table_ablation.md/.tex        ablation: augmentation, frozen vs fine-tuned, scratch vs pretrained
  table_imbalance.md/.tex       class-imbalance methods compared
  table_crossdata.md/.tex       train on one bag source, test on the other (TrashCan vs synthetic)
  mcnemar_<task>.csv            is the best model significantly better than each other model?
  robustness_summary.csv        robustness numbers averaged over seeds
  fig_*.png                     bar charts, learning-rate heat-map, data-efficiency curve,
                                underwater / brightness / noise / FGSM curves, background swap

Run:  python summarize.py            (python summarize.py --quick summarises the smoke test)
"""

import argparse
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import binomtest

import common as C

OUT = C.OUT_DIR / "summary"
METRICS = ["accuracy", "macro_f1", "balanced_accuracy", "worst_class_recall", "auc", "ece",
           "plastic_recall", "plastic_precision"]


# ---------------------------------------------------------------------------
# Reading results
# ---------------------------------------------------------------------------
def config_label(r):
    """A short readable name for one configuration (all seeds of it share this name)."""
    parts = [r["model"]]
    if r["method"] == "deep":
        if not r["pretrained"]:
            parts.append("scratch")
        if r["freeze"]:
            parts.append("frozen")
        parts.append(f"aug={r['aug']}")
        if r.get("imbalance", C.DEFAULT_IMBALANCE) != C.DEFAULT_IMBALANCE:
            parts.append(f"imb={r['imbalance']}")
        if r.get("train_frac", 1.0) < 1.0:
            parts.append(f"{int(r['train_frac'] * 100)}% data")
        if "_lr" in r["run"]:
            parts.append(f"lr={r['lr']:g}")
    if r.get("bag_source", C.DEFAULT_BAG_SOURCE) != C.DEFAULT_BAG_SOURCE:
        parts.append(f"bags={r['bag_source']}")
    return " | ".join(parts)


def load_results(quick):
    rows, preds = [], {}
    for f in sorted(C.RESULTS_DIR.glob("*.json")):
        if f.name.startswith("QUICK_") != quick:
            continue
        r = json.load(open(f))
        row = {k: r.get(k) for k in ["run", "method", "task", "model", "pretrained", "freeze", "aug", "seed",
                                     "train_frac", "params_trainable", "mean_epoch_sec", "epochs_run", "lr",
                                     "best_val_macro_f1", "bag_source", "imbalance"]}
        row["train_frac"] = row["train_frac"] or 1.0
        row["bag_source"] = row["bag_source"] or C.DEFAULT_BAG_SOURCE
        row["imbalance"] = row["imbalance"] or C.DEFAULT_IMBALANCE
        row.update({m: r["test"].get(m) for m in METRICS})
        row["config"] = config_label(row)
        rows.append(row)
        preds[r["run"]] = r["test_predictions"]
    return pd.DataFrame(rows), preds


def mean_std(df):
    g = df.groupby(["task", "method", "config"])
    agg = g[METRICS].agg(["mean", "std"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    agg["n_seeds"] = g.size()
    agg["params_trainable"] = g["params_trainable"].first()
    agg["sec_per_epoch"] = g["mean_epoch_sec"].mean()
    return agg.reset_index()


# ---------------------------------------------------------------------------
# Table helpers
# ---------------------------------------------------------------------------
def fmt(m, s):
    if pd.isna(m):
        return "-"
    return f"{m:.3f}" if pd.isna(s) else f"{m:.3f} ± {s:.3f}"


def pretty_table(agg, cols=("accuracy", "macro_f1", "worst_class_recall", "auc", "plastic_recall", "ece")):
    t = agg[["config", "n_seeds"]].copy()
    for c in cols:
        t[c] = [fmt(m, s) for m, s in zip(agg[f"{c}_mean"], agg[f"{c}_std"])]
    return t


def to_markdown(t):
    lines = ["| " + " | ".join(str(c) for c in t.columns) + " |", "|" + "---|" * len(t.columns)]
    lines += ["| " + " | ".join(str(v) for v in r) + " |" for r in t.values]
    return "\n".join(lines)


def to_latex(t, caption):
    """A ready-to-paste LaTeX table for the CVPR template."""
    cols = [str(c).replace("_", " ") for c in t.columns]
    body = "\n".join(" & ".join(str(v).replace("±", r"$\pm$").replace("_", r"\_").replace("|", "/")
                                for v in r) + r" \\" for r in t.values)
    return (r"\begin{table}[t]\centering\small" + "\n" + r"\caption{" + caption + "}\n"
            + r"\resizebox{\columnwidth}{!}{\begin{tabular}{l" + "c" * (len(cols) - 1) + "}\n\\toprule\n"
            + " & ".join(cols) + r" \\ \midrule" + "\n" + body + "\n" + r"\bottomrule\end{tabular}}\end{table}")


def save_table(t, name, caption, md, title):
    (OUT / f"{name}.md").write_text(to_markdown(t))
    (OUT / f"{name}.tex").write_text(to_latex(t, caption))
    md += [f"\n## {title}\n", to_markdown(t)]


def mcnemar(preds, run_a, run_b):
    """Exact McNemar test on the same test images: counts where only A is right vs only B is right."""
    a = pd.DataFrame(preds[run_a]).set_index("path")
    b = pd.DataFrame(preds[run_b]).set_index("path").reindex(a.index)
    ca, cb = a.y_pred == a.y_true, b.y_pred == b.y_true
    n01, n10 = int((ca & ~cb).sum()), int((~ca & cb).sum())
    p = binomtest(min(n01, n10), n01 + n10, 0.5).pvalue if n01 + n10 > 0 else 1.0
    return n01, n10, p


def bar_chart(agg, task, path):
    a = agg.sort_values("macro_f1_mean")
    fig, ax = plt.subplots(figsize=(8, 0.4 * len(a) + 1.5))
    ax.barh(a["config"], a["macro_f1_mean"], xerr=a["macro_f1_std"].fillna(0), color="tab:blue", alpha=0.8)
    ax.set_xlabel("test macro-F1 (mean ± std over seeds)")
    ax.set_xlim(max(0, a["macro_f1_mean"].min() - 0.1), 1.0)
    ax.set_title(f"Model comparison — {task} task")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def heatmap(piv, path, title, xlabel):
    fig, ax = plt.subplots(figsize=(1.4 * piv.shape[1] + 2.5, 0.7 * piv.shape[0] + 1.8))
    im = ax.imshow(piv.values.astype(float), cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(piv.shape[1]), [str(c) for c in piv.columns]); ax.set_xlabel(xlabel)
    ax.set_yticks(range(piv.shape[0]), piv.index)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            ax.text(j, i, f"{piv.values[i, j]:.3f}", ha="center", va="center", color="w", fontsize=8)
    fig.colorbar(im); ax.set_title(title)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


# ---------------------------------------------------------------------------
# Robustness helpers
# ---------------------------------------------------------------------------
def robustness_tables(quick):
    files = [f for f in (C.OUT_DIR / "robustness").glob("*.csv") if ("QUICK_" in f.name) == quick]
    if not files:
        return None
    rob = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    rob = rob[~rob["run"].str.contains("_frac|_lr")]
    for c, default in [("freeze", np.nan), ("pretrained", np.nan), ("ood_auroc", np.nan),
                       ("bag_source", C.DEFAULT_BAG_SOURCE), ("imbalance", C.DEFAULT_IMBALANCE)]:
        if c not in rob:
            rob[c] = default
        rob[c] = rob[c].fillna(default) if isinstance(default, str) else rob[c]
    labels = []
    for r in rob.itertuples():
        if pd.isna(r.freeze):                                   # classic or CLIP
            lab = r.model
        else:
            lab = f"{r.model} | aug={r.aug}" + (" | frozen" if r.freeze else "") + ("" if r.pretrained else " | scratch")
            if r.imbalance != C.DEFAULT_IMBALANCE:
                lab += f" | imb={r.imbalance}"
        if r.bag_source != C.DEFAULT_BAG_SOURCE:
            lab += f" | bags={r.bag_source}"
        labels.append(lab)
    rob["label"] = labels
    rob["level"] = rob["level"].astype(float)
    uw = rob.eval_set.str.startswith("test_uw")
    rob.loc[uw, "level"] = rob.loc[uw, "eval_set"].str[-1].astype(float)
    return rob.groupby(["task", "label", "eval_set", "level"]).agg(
        accuracy=("accuracy", "mean"), accuracy_std=("accuracy", "std"), macro_f1=("macro_f1", "mean"),
        plastic_recall=("plastic_recall", "mean"), pred_plastic_rate=("pred_plastic_rate", "mean"),
        mean_confidence=("mean_confidence", "mean"), ood_auroc=("ood_auroc", "mean"),
        n_seeds=("seed", "nunique")).reset_index()


def keep_for_plot(label):
    """Keep plots readable: main models only (no frozen / scratch / imbalance / other-bag-source variants)."""
    if any(k in label for k in ["frozen", "imb=", "bags="]):
        return False
    return "scratch" not in label or label.startswith("tinycnn")


def line_plot(rob, task, eval_set, xlabel, path, title, clean_level=0.0):
    sub = rob[(rob.task == task) & rob.label.map(keep_for_plot)]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = plt.cm.tab20(np.linspace(0, 1, 20))
    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]
    for k, (label, g) in enumerate(sub.groupby("label")):
        clean = g[g.eval_set == "test"][["level", "accuracy"]].assign(level=clean_level)
        pts = pd.concat([clean, g[g.eval_set == eval_set][["level", "accuracy"]]]).sort_values("level")
        if len(pts) > 1:
            ax.plot(pts.level, pts.accuracy, marker=markers[k % len(markers)], color=colors[k % 20], label=label)
    ax.set_xlabel(xlabel); ax.set_ylabel("test accuracy"); ax.set_title(title)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.01, 0.5))
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="summarise the QUICK_ smoke-test runs")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    md = ["# Key numbers for the report\n"]

    # ---- 0. data and class balance (from prepare_data.py) ----
    ds = C.OUT_DIR / "data_summary.json"
    if ds.exists():
        s = json.load(open(ds))
        rows = []
        for task in ["binary", "fine"]:
            for split in ["train", "train_unique_photos_only", "val", "test"]:
                b = s["class_balance"][task][split]
                rows.append({"task": task, "split": split, **b["counts"], "imbalance_ratio": round(b["imbalance_ratio"], 2)})
        md += ["\n## Data cleaning\n", "Jellyfish: " + json.dumps(s["jellyfish_duplicates"]),
               "\nTrashCan official split: " + json.dumps(s["trashcan_official_split"]),
               "\nTrashCan bag cleaning: " + json.dumps(s["trashcan_bag_cleaning"]),
               "\n## Class balance (bags = TrashCan)\n"]
        for task in ["binary", "fine"]:
            t = pd.DataFrame([r for r in rows if r["task"] == task]).dropna(axis=1)
            md += [f"\n### {task}\n", to_markdown(t)]

    df, preds = load_results(args.quick)
    if df.empty:
        print("No results found. Run the experiments first."); return
    df.to_csv(OUT / "results_all_runs.csv", index=False)
    agg = mean_std(df)
    agg.to_csv(OUT / "results_mean_std.csv", index=False)
    info = df.drop_duplicates("config").set_index("config")[["bag_source", "imbalance", "train_frac"]]
    agg = agg.join(info, on="config")

    # ---- 1. main comparison: default settings (TrashCan bags, weighted loss, standard aug, full data) ----
    special = "frozen|scratch|% data|aug=none|aug=underwater|lr=|imb=|bags="
    main_mask = ((agg.method != "deep") & (agg.bag_source == C.DEFAULT_BAG_SOURCE)) | \
                ((agg.method == "deep") & ~agg.config.str.contains(special)) | \
                (agg.config.str.startswith("tinycnn") & ~agg.config.str.contains(special.replace("scratch|", "")))
    for task in ["fine", "binary"]:
        a = agg[(agg.task == task) & main_mask].sort_values("macro_f1_mean", ascending=False)
        if a.empty:
            continue
        save_table(pretty_table(a), f"table_main_{task}",
                   f"Test results, {task} task (mean $\\pm$ std over seeds).", md, f"Main comparison — {task} task")
        bar_chart(a, task, OUT / f"fig_main_{task}.png")
        seed_runs = df[(df.task == task) & (df.seed == C.SEED)].drop_duplicates("config").set_index("config")["run"]
        best, rows = a.iloc[0]["config"], []
        for other in a["config"].iloc[1:]:
            if best in seed_runs and other in seed_runs:
                n01, n10, p = mcnemar(preds, seed_runs[best], seed_runs[other])
                rows.append({"best": best, "other": other, "only_best_correct": n01,
                             "only_other_correct": n10, "p_value": round(p, 4), "significant_5pct": p < 0.05})
        if rows:
            mc = pd.DataFrame(rows)
            mc.to_csv(OUT / f"mcnemar_{task}.csv", index=False)
            md += [f"\n### McNemar test — {task}\n", to_markdown(mc)]

    # ---- 2. ablation (fine task; the model with the most variants) ----
    deep = agg[(agg.method == "deep") & (agg.task == "fine") & ~agg.config.str.contains("% data|lr=|imb=|bags=")]
    if not deep.empty:
        base = deep.config.str.split(" | ", regex=False).str[0].value_counts().idxmax()
        abl = deep[deep.config.str.startswith(base + " ")].sort_values("macro_f1_mean", ascending=False)
        save_table(pretty_table(abl), "table_ablation", f"Ablation study with {base} (fine task).", md,
                   f"Ablation — {base}, fine task")

    # ---- 3. class-imbalance methods ----
    imb = agg[(agg.method == "deep") & (agg.task == "fine") & (agg.bag_source == C.DEFAULT_BAG_SOURCE)
              & (agg.train_frac == 1.0) & ~agg.config.str.contains("frozen|scratch|lr=|aug=none|aug=underwater")]
    imb_models = imb[imb.imbalance != C.DEFAULT_IMBALANCE].config.str.split(" | ", regex=False).str[0].unique()
    imb = imb[imb.config.str.split(" | ", regex=False).str[0].isin(imb_models)]
    if not imb.empty:
        t = pretty_table(imb.sort_values("macro_f1_mean", ascending=False),
                         cols=("macro_f1", "balanced_accuracy", "worst_class_recall", "plastic_recall", "ece"))
        t.insert(1, "imbalance_method", imb.sort_values("macro_f1_mean", ascending=False)["imbalance"].values)
        save_table(t, "table_imbalance", "Class-imbalance handling (fine task).", md,
                   "Class-imbalance methods — deep models, fine task")
    for f in sorted(C.OUT_DIR.glob(f"{'QUICK_' if args.quick else ''}classic_imbalance_*.csv")):
        if ("QUICK_" in f.name) == args.quick:
            md += [f"\n### Class-imbalance methods — classic model ({f.stem})\n", to_markdown(pd.read_csv(f).round(3))]

    # ---- 4. learning-rate search (validation macro-F1) ----
    lr_runs = df[df.run.str.contains("_lr")]
    if not lr_runs.empty:
        default = df[(df.method == "deep") & ~df.run.str.contains("_lr|_frac|_imb|_bags") & (df.seed == C.SEED)
                     & (df.aug == "standard") & (~df.freeze.astype(bool)) & (df.pretrained.astype(bool))
                     & df.model.isin(lr_runs.model.unique()) & df.task.isin(lr_runs.task.unique())]
        piv = pd.concat([lr_runs, default]).pivot_table(index="model", columns="lr", values="best_val_macro_f1")
        piv.columns = [f"{c:g}" for c in piv.columns]
        piv.to_csv(OUT / "lr_search_validation.csv")
        md += ["\n## Learning-rate search (validation macro-F1, fine task)\n", to_markdown(piv.round(3).reset_index())]
        heatmap(piv, OUT / "fig_lr_search.png", "Learning-rate search (validation macro-F1)", "learning rate")

    # ---- 5. data-efficiency curve ----
    lc = df[(df.method == "deep") & (df.aug == "standard") & (df.bag_source == C.DEFAULT_BAG_SOURCE)
            & (df.imbalance == C.DEFAULT_IMBALANCE) & ~df.run.str.contains("_lr")]
    lc = lc[lc.model.isin(lc[lc.train_frac < 1].model.unique()) & (~lc.freeze.astype(bool)) & lc.pretrained.astype(bool)]
    if lc.train_frac.nunique() > 1:
        fig, ax = plt.subplots(figsize=(6, 4))
        for (task, model), g in lc.groupby(["task", "model"]):
            s2 = g.groupby("train_frac")["macro_f1"].agg(["mean", "std"]).reset_index()
            ax.errorbar(s2.train_frac * 100, s2["mean"], yerr=s2["std"].fillna(0), marker="o", capsize=3,
                        label=f"{model} ({task})")
        ax.set_xlabel("% of training data used"); ax.set_ylabel("test macro-F1"); ax.grid(alpha=0.3)
        ax.set_title("Data efficiency"); ax.legend()
        fig.tight_layout(); fig.savefig(OUT / "fig_learning_curve.png", dpi=150); plt.close(fig)

    # ---- 6. robustness, cross-dataset, background swap, out-of-distribution ----
    rob = robustness_tables(args.quick)
    if rob is not None:
        rob.to_csv(OUT / "robustness_summary.csv", index=False)
        for task in rob.task.unique():
            r_t = rob[rob.task == task]
            # cross-dataset: train on one bag source, test on both
            cd = r_t[r_t.eval_set.isin(["test_tc", "test_syn"])].pivot_table(
                index="label", columns="eval_set", values="macro_f1").round(3)
            if not cd.empty:
                cd = cd[[c for c in ["test_tc", "test_syn"] if c in cd]].rename(
                    columns={"test_tc": "test on TrashCan bags", "test_syn": "test on synthetic bags"})
                cd = cd.reset_index()
                base = cd.label.str.split(" | bags=", regex=False).str[0]
                has_variant = set(base[cd.label.str.contains("bags=", regex=False)])
                cd = cd[base.isin(has_variant)]          # only models trained on more than one bag source
                cd.insert(1, "trained on", cd.label.str.split("bags=", regex=False).str[1].fillna(C.DEFAULT_BAG_SOURCE))
                cd["label"] = cd.label.str.split(" | bags=", regex=False).str[0]
            if not cd.empty:
                save_table(cd, f"table_crossdata_{task}", f"Cross-dataset macro-F1 ({task}).", md,
                           f"Cross-dataset test (macro-F1) — {task}")
            sub = r_t.copy()
            sub.loc[sub.eval_set.str.startswith("test_uw"), "eval_set"] = "test_uw"
            line_plot(sub, task, "test_uw", "underwater severity", OUT / f"fig_underwater_{task}.png",
                      f"Accuracy under simulated underwater conditions — {task}")
            line_plot(r_t, task, "test_bright", "brightness factor (1 = unchanged)", OUT / f"fig_brightness_{task}.png",
                      f"Accuracy when images are darker / brighter — {task}", clean_level=1.0)
            line_plot(r_t, task, "test_fgsm", "FGSM epsilon (x/255)", OUT / f"fig_fgsm_{task}.png",
                      f"Adversarial robustness (FGSM) — {task}")
            line_plot(r_t, task, "test_noise", "Gaussian noise sigma", OUT / f"fig_noise_{task}.png",
                      f"Robustness to noise — {task}")
            bs = r_t[r_t.eval_set.str.startswith("sup_") & r_t.label.map(keep_for_plot)]
            if not bs.empty:
                piv = bs.pivot_table(index="label", columns="eval_set", values="plastic_recall")
                piv = piv[[c for c in ["sup_orig", "sup_gray", "sup_blue"] if c in piv]].dropna(how="all")
            if not bs.empty and not piv.empty:
                ax = piv.plot(kind="barh", figsize=(8, 0.5 * len(piv) + 2))
                ax.set_xlabel("plastic recall (share of bags still called 'plastic bag')")
                ax.set_title(f"Background-swap test — {task}"); ax.set_xlim(0, 1.05)
                plt.tight_layout(); plt.savefig(OUT / f"fig_background_swap_{task}.png", dpi=150); plt.close()
                md += [f"\n## Background-swap plastic recall — {task}\n", to_markdown(piv.round(3).reset_index())]
            ood = r_t[(r_t.eval_set == "ood_other") & r_t.label.map(keep_for_plot)][
                ["label", "pred_plastic_rate", "mean_confidence", "ood_auroc"]].round(3)
            if not ood.empty:
                md += [f"\n## Unseen objects (out-of-distribution) — {task}\n", to_markdown(ood)]
            bright = r_t[r_t.eval_set.isin(["test", "test_bright"]) & r_t.label.map(keep_for_plot)]
            bright = bright.assign(col=np.where(bright.eval_set == "test", "1.0", bright.level.astype(str)))
            bt = bright.pivot_table(index="label", columns="col", values="accuracy").round(3)
            if not bt.empty:
                md += [f"\n## Accuracy vs brightness — {task}\n", to_markdown(bt.reset_index())]
            uw = r_t[r_t.eval_set.isin(["test", "test_uw3", "test_uw5"]) & r_t.label.map(keep_for_plot)].pivot_table(
                index="label", columns="eval_set", values="accuracy").round(3)
            if not uw.empty:
                md += [f"\n## Accuracy clean vs underwater — {task}\n", to_markdown(uw.reset_index())]

    gc = C.OUT_DIR / "gradcam_background_swap_summary.csv"
    if gc.exists():
        md += ["\n## Grad-CAM: does the model look at the bag?\n", to_markdown(pd.read_csv(gc).round(3))]

    (OUT / "key_numbers.md").write_text("\n".join(md))
    print("\n".join(md))
    print(f"\nAll tables and figures saved in {OUT}")


if __name__ == "__main__":
    main()
