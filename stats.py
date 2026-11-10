"""
stats.py  (NEW STEP D - no GPU, about 2-5 minutes)
==================================================
Turns the per-image predictions in outputs/preds/ into the report's new tables and figures.

 1. Group benchmark (class x background):  accuracy per group, WORST-GROUP accuracy,
    BG-gap = mean accuracy with the "training" background - mean accuracy with the swapped
    background (Xiao et al., ICLR 2021), and the "background only" plastic rate.
 2. Seeds: every number is the mean +- std over the seeds that exist (813/814/815).
 3. 95% confidence intervals by CLUSTER bootstrap: we resample whole videos (bags) and whole
    duplicate groups (jellyfish), not single images, because frames of one video are not
    independent. 2000 resamples, seed-813 model.
 4. Exact McNemar tests with HOLM correction for multiple comparisons:
      (a) ERM vs DFR for the same model on the group benchmark,
      (b) the best model vs every other model on the normal test set.
 5. Out-of-distribution (unseen objects): AUROC of softmax confidence (MSP), energy and kNN scores.
 6. Responsible-AI cost analysis for a clean-up robot: missing a bag costs 1, grabbing a jellyfish
    costs r (r = 0.1 ... 10). The robot grabs when P(bag) >= r / (1 + r) (Bayes rule; nothing is tuned,
    because DFR already uses the validation split), and the cost is measured on the normal test set
    and on the shifted group benchmark.
    Plus risk-coverage curves (the robot may abstain on its least confident cases).

Run:     python stats.py           full version used for the report (2000 bootstrap resamples, slow)
         python stats.py --demo    fast demo for the video: 100 resamples, writes to outputs/extra/demo/
                                   so the report's files are NOT overwritten
Output:  outputs/extra/  (tables .md + .tex, figures .png, extra_key_numbers.md, all_methods.csv)
"""

import argparse
import re

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import f1_score, roc_auc_score

import common as C
import extra_common as X
from summarize import to_latex, to_markdown   # same table style as the existing tables

plt = C.plt
INK, INK2, GRID = "#1f1f1e", "#5f5e5a", "#d9d8d2"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]     # validated categorical slots 1-4
FAMILY_COLOR = {"ERM": SERIES[0], "DFR": SERIES[1], "Foundation": SERIES[2]}
ARCH = {"resnet50": "ResNet-50", "efficientnet_b0": "EfficientNet-B0", "vit_b_16": "ViT-B/16",
        "vgg16": "VGG16", "tinycnn": "TinyCNN", "clip": "CLIP ViT-B/32", "dinov2_base": "DINOv2 ViT-B/14"}
VARIANT = {"erm": "", "dfr": " + DFR", "zeroshot": " zero-shot", "probe": " linear probe",
           "probe_dfr": " probe + group-balanced"}


# -----------------------------------------------------------------------------
def load_preds():
    files = sorted(X.PRED_DIR.glob("*.csv"))
    files = [f for f in files if not f.stem.startswith("QUICK_")]
    if not files:
        raise SystemExit(f"No predictions in {X.PRED_DIR}. Run dfr.py and foundation_probe.py first.")
    return {f.stem: pd.read_csv(f) for f in files}


def describe(method):
    """method name -> (config without seed, seed, pretty label, family, task)."""
    run, variant = method.split("__")
    task = run.split("_")[0]
    m = re.search(r"_s(\d+)$", run)
    seed = int(m.group(1)) if m else C.SEED
    config = re.sub(r"_s\d+$", "", run) + "__" + variant
    arch = next((a for a in sorted(ARCH, key=len, reverse=True) if f"_{a}" in run), run)
    label = ARCH.get(arch, arch)
    if "aug-underwater" in run:
        label += " (underwater aug)"
    extra = re.findall(r"_(bags-\w+|imb-\w+|frac[\d.]+|frozen|scratch)", run.replace("tinycnn_scratch", "tinycnn"))
    if extra:
        label += " [" + ",".join(extra) + "]"
    label += VARIANT.get(variant, " " + variant)
    family = "DFR" if variant in ("dfr", "probe_dfr") else ("ERM" if variant == "erm" else "Foundation")
    return config, seed, label, family, task


def acc(df):
    return float((df.pred_bin == df.y_bin).mean()) if len(df) else np.nan


def method_metrics(p):
    out = {}
    t = p[p.set == "test"]
    out["test_acc"] = acc(t)
    out["test_macro_f1_bin"] = f1_score(t.y_bin, t.pred_bin, average="macro") if len(t) else np.nan
    if p.task.iloc[0] == "fine" and len(t):
        out["test_macro_f1_fine"] = f1_score(t.y_fine, t.pred_fine, average="macro")
    g = p[p.set == "gtest"]
    for grp in X.ALL_GROUPS:
        sub = g[g.group == grp]
        if not len(sub):
            continue
        if grp.startswith("bgonly"):
            out[f"{grp}_plastic_rate"] = float((sub.pred_bin == 0).mean())
        else:
            out[f"acc_{grp}"] = acc(sub)
    core = [out.get(f"acc_{k}") for k in X.CORE_GROUPS]
    if all(v is not None for v in core):
        out["worst_group"] = float(np.nanmin(core))
        out["mean_group"] = float(np.nanmean(core))
        out["bg_gap"] = float(np.mean([out[f"acc_{k}"] for k in X.SAME_BG]) - np.mean([out[f"acc_{k}"] for k in X.SWAPPED_BG]))
    ood, tt = p[p.set == "ood"], t
    if len(ood) and len(tt):
        out["ood_pred_plastic"] = float((ood.pred_bin == 0).mean())
        lab = np.r_[np.ones(len(tt)), np.zeros(len(ood))]
        out["ood_auroc_msp"] = roc_auc_score(lab, np.r_[tt.conf, ood.conf])
        for s in ["energy", "knn"]:
            if s in p and p[s].notna().all():
                out[f"ood_auroc_{s}"] = roc_auc_score(lab, np.r_[tt[s], ood[s]])
    return out


# -----------------------------------------------------------------------------
def cluster_bootstrap(p, stat, B=2000, seed=C.SEED):
    """95% CI: resample clusters (videos / duplicate groups) with replacement inside every group stratum."""
    rng = np.random.default_rng(seed)
    strata = []
    for _, s in p.groupby("group"):
        cl = s.cluster.values
        uniq = np.unique(cl)
        idx = [np.where(cl == u)[0] for u in uniq]
        strata.append((s.reset_index(drop=True), idx))
    vals = []
    for _ in range(B):
        parts = []
        for s, idx in strata:
            pick = rng.integers(0, len(idx), len(idx))
            parts.append(s.iloc[np.concatenate([idx[k] for k in pick])])
        vals.append(stat(pd.concat(parts)))
    return np.nanpercentile(vals, [2.5, 97.5])


def worst_group_stat(d):
    return min(acc(d[d.group == g]) for g in X.CORE_GROUPS)


def macro_f1_stat(d):
    return f1_score(d.y_bin, d.pred_bin, average="macro")


def mcnemar_exact(a_correct, b_correct):
    b = int(np.sum(a_correct & ~b_correct)); c = int(np.sum(~a_correct & b_correct))
    p = 1.0 if b + c == 0 else binomtest(b, b + c, 0.5).pvalue
    return b, c, p


def holm(pvals):
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for k, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - k) * p[i]))
        adj[i] = running
    return adj


# -----------------------------------------------------------------------------
def cost_curve(p, ratios):
    """
    Bayes-optimal decision for a clean-up robot (Elkan, 2001): missing a bag costs 1, grabbing a jellyfish
    costs r, so the robot grabs when P(bag) >= r / (1 + r). No threshold is tuned: the validation split was
    already used to fit the DFR / probe heads, and tuning on it again would be optimistic.
    """
    test = p[p.set == "test"]
    shifted = p[(p.set == "gtest") & p.group.isin(X.CORE_GROUPS)]

    def cost(d, t, r):
        grab = d.p_plastic.values >= t
        bag = d.y_bin.values == 0
        return float((np.sum(bag & ~grab) + r * np.sum(~bag & grab)) / len(d))

    rows = []
    for r in ratios:
        t = r / (1.0 + r)
        rows.append({"ratio": r, "threshold": t, "cost_test": cost(test, t, r), "cost_shifted": cost(shifted, t, r),
                     "bag_recall_shifted": float((shifted[shifted.y_bin == 0].p_plastic >= t).mean()),
                     "jelly_grabbed_shifted": float((shifted[shifted.y_bin == 1].p_plastic >= t).mean())})
    return pd.DataFrame(rows)


def risk_coverage(d):
    conf = np.maximum(d.p_plastic.values, 1 - d.p_plastic.values)
    err = (d.pred_bin.values != d.y_bin.values).astype(float)
    order = np.argsort(-conf)
    cum = np.cumsum(err[order]) / np.arange(1, len(d) + 1)
    cov = np.arange(1, len(d) + 1) / len(d)
    return cov, cum, float(cum.mean())          # last = area under the risk-coverage curve (AURC)


# -----------------------------------------------------------------------------
def fmt(m, s):
    if pd.isna(m):
        return "-"
    return f"{m:.3f}" if pd.isna(s) or s == 0 else f"{m:.3f} ± {s:.3f}"


def write_table(df, name, caption, md_lines, title):
    tex = to_latex(df, caption).replace("→", r"$\rightarrow$")
    (X.EXTRA_DIR / f"{name}.md").write_text(to_markdown(df), encoding="utf-8")
    (X.EXTRA_DIR / f"{name}.tex").write_text(tex, encoding="utf-8")
    md_lines += [f"\n## {title}\n", to_markdown(df)]


def style_axes(ax):
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)


def fig_worst_group(agg, task, path):
    d = agg[agg.task == task].dropna(subset=["worst_group_mean"]).sort_values("worst_group_mean")
    if not len(d):
        return
    fig, ax = plt.subplots(figsize=(7.2, 0.32 * len(d) + 1.4))
    y = np.arange(len(d))
    ax.barh(y, d.worst_group_mean, xerr=d.worst_group_std.fillna(0), height=0.62,
            color=[FAMILY_COLOR[f] for f in d.family], error_kw=dict(ecolor=INK2, lw=0.8, capsize=2))
    ax.scatter(d.test_acc_mean, y, marker="|", s=90, color=INK, zorder=3, label="normal test accuracy")
    ax.set_yticks(y); ax.set_yticklabels(d.label, fontsize=7.5, color=INK)
    ax.set_xlim(0, 1.02); ax.set_xlabel("accuracy (bag vs jellyfish)", color=INK2, fontsize=8)
    style_axes(ax)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in FAMILY_COLOR.values()]
    ax.legend(handles + [ax.collections[0]], [f"worst-group: {k}" for k in FAMILY_COLOR] + ["normal test accuracy"],
              fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.4, -0.12), ncol=2)
    ax.set_title(f"Worst-group accuracy on the class x background benchmark ({task} models)",
                 fontsize=9, color=INK, loc="left")
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def fig_lines(curves, xlabel, ylabel, title, path, logx=False):
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    for (lab, x, y), col in zip(curves, SERIES):
        ax.plot(x, y, lw=2, color=col, label=lab)
    if logx:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel, color=INK2, fontsize=8); ax.set_ylabel(ylabel, color=INK2, fontsize=8)
    style_axes(ax); ax.grid(axis="y", color=GRID, lw=0.6)
    ax.legend(fontsize=7, frameon=False)
    ax.set_title(title, fontsize=9, color=INK, loc="left")
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


# -----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--demo", action="store_true",
                    help="fast run for a demonstration: 100 resamples, results in outputs/extra/demo/")
    args = ap.parse_args()
    X.make_extra_dirs()
    if args.demo:
        import shutil
        args.boot = min(args.boot, 100)
        real = X.EXTRA_DIR
        X.EXTRA_DIR = real / "demo"                       # keep the report's files untouched
        X.EXTRA_DIR.mkdir(parents=True, exist_ok=True)
        for name in ["lowlevel_cue_test.csv", "table_insitu.md"]:   # inputs written by other scripts
            if (real / name).exists():
                shutil.copy(real / name, X.EXTRA_DIR / name)
        print(f"DEMO mode: {args.boot} bootstrap resamples, output in {X.EXTRA_DIR}")
    preds = load_preds()
    md = ["# Key numbers of the new experiments (shortcut study)",
          "Accuracies are for the binary decision 'plastic bag or jellyfish?' (7-class models are mapped to it). "
          "mean ± std over seeds; CIs are 95% cluster-bootstrap intervals of the seed-813 model."]

    # ---- 1-2. metrics per method, aggregated over seeds ----
    rows = []
    for m, p in preds.items():
        config, seed, label, family, task = describe(m)
        rows.append({"method": m, "config": config, "seed": seed, "label": label, "family": family, "task": task,
                     **method_metrics(p)})
    allm = pd.DataFrame(rows)
    allm.to_csv(X.EXTRA_DIR / "all_methods.csv", index=False)
    num = [c for c in allm.columns if c not in ("method", "config", "seed", "label", "family", "task")]
    g = allm.groupby(["task", "config", "label", "family"])
    agg = g[num].mean().add_suffix("_mean").join(g[num].std().add_suffix("_std")).join(g.size().rename("n_seeds")).reset_index()

    # ---- 3. bootstrap CIs (seed-813 or the only run) ----
    ci_rows = []
    for _, r in agg.iterrows():
        cand = allm[(allm.config == r.config)].sort_values("seed")
        m = cand.iloc[0].method
        p = preds[m]
        ci = {"config": r.config}
        t = p[p.set == "test"]
        if len(t):
            ci["test_f1_ci"] = cluster_bootstrap(t, macro_f1_stat, args.boot)
        gt = p[(p.set == "gtest") & p.group.isin(X.CORE_GROUPS)]
        if gt.group.nunique() == 4:
            ci["worst_group_ci"] = cluster_bootstrap(gt, worst_group_stat, args.boot)
        ci_rows.append(ci)
    ci = pd.DataFrame(ci_rows)
    agg = agg.merge(ci, on="config", how="left")

    def ci_txt(v):
        return "-" if not isinstance(v, np.ndarray) else f"[{v[0]:.2f}, {v[1]:.2f}]"

    for task in ["binary", "fine"]:
        d = agg[agg.task == task].sort_values(["family", "worst_group_mean"], ascending=[True, False])
        if not len(d):
            continue
        tab = pd.DataFrame({
            "method": d.label, "seeds": d.n_seeds,
            "test acc": [fmt(a, b) for a, b in zip(d.test_acc_mean, d.test_acc_std)],
            "bag ROV": [fmt(a, b) for a, b in zip(d.get("acc_bag_rov_mean"), d.get("acc_bag_rov_std"))],
            "bag clear": [fmt(a, b) for a, b in zip(d.get("acc_bag_clear_mean"), d.get("acc_bag_clear_std"))],
            "jelly clear": [fmt(a, b) for a, b in zip(d.get("acc_jelly_clear_mean"), d.get("acc_jelly_clear_std"))],
            "jelly ROV": [fmt(a, b) for a, b in zip(d.get("acc_jelly_rov_mean"), d.get("acc_jelly_rov_std"))],
            "worst-group": [fmt(a, b) for a, b in zip(d.get("worst_group_mean"), d.get("worst_group_std"))],
            "worst-group 95% CI": [ci_txt(v) for v in d.get("worst_group_ci", [None] * len(d))],
            "BG-gap": [fmt(a, b) for a, b in zip(d.get("bg_gap_mean"), d.get("bg_gap_std"))],
            "ROV bg only → 'bag'": [fmt(a, b) for a, b in zip(d.get("bgonly_rov_plastic_rate_mean"), d.get("bgonly_rov_plastic_rate_std"))],
        })
        if "acc_jelly_insitu_mean" in d:
            tab["in-situ jelly (FathomNet)"] = [fmt(a, b) for a, b in zip(d.acc_jelly_insitu_mean, d.acc_jelly_insitu_std)]
        write_table(tab, f"table_groups_{task}", f"Class x background benchmark ({task} models).", md,
                    f"Class x background benchmark — {task} models")
        fig_worst_group(agg, task, X.EXTRA_DIR / f"fig_worst_group_{task}.png")

        tab2 = pd.DataFrame({"method": d.label, "seeds": d.n_seeds,
                             "test macro-F1 (bag vs jelly)": [fmt(a, b) for a, b in zip(d.test_macro_f1_bin_mean, d.test_macro_f1_bin_std)],
                             "95% CI": [ci_txt(v) for v in d.get("test_f1_ci", [None] * len(d))]})
        if task == "fine" and "test_macro_f1_fine_mean" in d:
            tab2["test macro-F1 (7 classes)"] = [fmt(a, b) for a, b in zip(d.test_macro_f1_fine_mean, d.test_macro_f1_fine_std)]
        write_table(tab2, f"table_test_ci_{task}", f"Normal test set with 95% cluster-bootstrap CIs ({task}).", md,
                    f"Normal test set with confidence intervals — {task} models")

        ood_cols = [c for c in ["ood_pred_plastic", "ood_auroc_msp", "ood_auroc_energy", "ood_auroc_knn"] if f"{c}_mean" in d]
        tab3 = pd.DataFrame({"method": d.label, **{c.replace("ood_", ""): [fmt(a, b) for a, b in zip(d[f"{c}_mean"], d[f"{c}_std"])]
                                                     for c in ood_cols}})
        write_table(tab3, f"table_ood_{task}", f"Unseen objects: share called 'plastic bag' and OOD AUROC ({task}).", md,
                    f"Unseen objects (out-of-distribution) — {task} models")

    # ---- 4. McNemar + Holm ----
    tests = []
    for m, p in preds.items():
        if not m.endswith("__erm"):
            continue
        m2 = m.replace("__erm", "__dfr")
        if m2 not in preds:
            continue
        a = p[(p.set == "gtest") & p.group.isin(X.CORE_GROUPS)].set_index("path")
        b = preds[m2][(preds[m2].set == "gtest") & preds[m2].group.isin(X.CORE_GROUPS)].set_index("path")
        a, b = a[~a.index.duplicated()], b[~b.index.duplicated()]
        common = a.index.intersection(b.index)
        ac = (a.loc[common].pred_bin == a.loc[common].y_bin).values
        bc = (b.loc[common].pred_bin == b.loc[common].y_bin).values
        n1, n2, pv = mcnemar_exact(bc, ac)
        tests.append({"family": "ERM vs DFR (group benchmark)", "a": f"{describe(m)[4]}: " + describe(m2)[2] + f" (seed {describe(m)[1]})",
                      "b": describe(m)[2] + " (ERM)", "only_a_correct": n1, "only_b_correct": n2, "p": pv})
    for task in ["binary", "fine"]:
        cand = allm[(allm.task == task) & (allm.seed == C.SEED)].dropna(subset=["test_acc"])
        if len(cand) < 2:
            continue
        best = cand.sort_values("test_macro_f1_bin", ascending=False).iloc[0]
        pb = preds[best.method]; pb = pb[pb.set == "test"].set_index("path")
        for _, o in cand.iterrows():
            if o.method == best.method:
                continue
            po = preds[o.method]; po = po[po.set == "test"].set_index("path")
            common = pb.index.intersection(po.index)
            n1, n2, pv = mcnemar_exact((pb.loc[common].pred_bin == pb.loc[common].y_bin).values,
                                       (po.loc[common].pred_bin == po.loc[common].y_bin).values)
            tests.append({"family": f"best vs others, normal test ({task})", "a": best.label, "b": o.label,
                          "only_a_correct": n1, "only_b_correct": n2, "p": pv})
    if tests:
        tt = pd.DataFrame(tests)
        tt["p_holm"] = tt.groupby("family").p.transform(lambda s: holm(s.values))
        tt["significant (Holm, 5%)"] = tt.p_holm < 0.05
        tt.to_csv(X.EXTRA_DIR / "mcnemar_holm.csv", index=False)
        md += ["\n## McNemar tests with Holm correction\n",
               to_markdown(tt.assign(p=tt.p.round(4), p_holm=tt.p_holm.round(4)))]

    # ---- 6. cost analysis + risk-coverage (binary decision; seed-averaged) ----
    ratios = np.round(np.logspace(-1, 1, 21), 3)
    pick = []
    for want in ["binary_resnet50_pt_ft_aug-standard__erm", "binary_resnet50_pt_ft_aug-standard__dfr",
                 "binary_clip__zeroshot", "binary_dinov2_base__probe_dfr",
                 "binary_efficientnet_b0_pt_ft_aug-standard__erm", "binary_efficientnet_b0_pt_ft_aug-standard__dfr",
                 "binary_clip__probe", "binary_dinov2_base__probe"]:
        ms = [m for m in preds if describe(m)[0] == want]
        if ms:
            pick.append((describe(ms[0])[2], ms))
    if pick:
        cost_lines, rc_lines, cost_rows, all_cc = [], [], [], []
        for label, ms in pick:
            cc = pd.concat([cost_curve(preds[m], ratios) for m in ms]).groupby("ratio").mean().reset_index()
            all_cc.append(cc.assign(method=label, config=describe(ms[0])[0]))
            cost_lines.append((label, cc.ratio.values, cc.cost_shifted.values))
            for r in [1.0, 5.012]:
                row = cc.iloc[(cc.ratio - r).abs().argmin()]
                cost_rows.append({"method": label, "cost ratio r (jelly harm / missed bag)": round(float(row.ratio), 1),
                                  "cost / object, normal test": round(row.cost_test, 3),
                                  "cost / object, shifted backgrounds": round(row.cost_shifted, 3),
                                  "bags collected (shifted)": round(row.bag_recall_shifted, 3),
                                  "jellyfish grabbed (shifted)": round(row.jelly_grabbed_shifted, 3)})
            covs, risks, aurcs = [], [], []
            for m in ms:
                d = preds[m]; d = d[(d.set == "gtest") & d.group.isin(X.CORE_GROUPS)]
                cov, risk, aurc = risk_coverage(d)
                covs.append(cov); risks.append(risk); aurcs.append(aurc)
            rc_lines.append((f"{label} (AURC {np.mean(aurcs):.3f})", covs[0], np.mean([np.interp(covs[0], c, r) for c, r in zip(covs, risks)], 0)))
        pd.concat(all_cc).to_csv(X.EXTRA_DIR / "cost_curves.csv", index=False)
        cost_lines, rc_lines = cost_lines[:4], rc_lines[:4]       # the figure shows the first four methods
        fig_lines(cost_lines, "cost ratio r = harm of grabbing a jellyfish / harm of missing a bag", "expected cost per object",
                  "Clean-up robot cost on shifted backgrounds (Bayes threshold r/(1+r))",
                  X.EXTRA_DIR / "fig_cost_curve.png", logx=True)
        fig_lines(rc_lines, "coverage (share of objects the robot decides on)", "error rate on decided objects",
                  "Risk-coverage on the class x background benchmark", X.EXTRA_DIR / "fig_risk_coverage.png")
        write_table(pd.DataFrame(cost_rows), "table_cost", "Cost-sensitive decisions for a clean-up robot.", md,
                    "Cost-sensitive decisions (clean-up robot)")

    # ---- low-level cue test from group_benchmark.py ----
    f = X.EXTRA_DIR / "lowlevel_cue_test.csv"
    if f.exists():
        md += ["\n## Low-level cue test (why the underwater augmentation failed)\n",
               to_markdown(pd.read_csv(f).round(3))]
    f = X.EXTRA_DIR / "table_insitu.md"
    if f.exists():
        md += ["\n## Same-camera test with FathomNet in-situ jellyfish\n", f.read_text(encoding="utf-8")]
    (X.EXTRA_DIR / "extra_key_numbers.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    print(f"\nAll new tables and figures saved in {X.EXTRA_DIR}")


if __name__ == "__main__":
    main()
