"""
report_assets.py  (no GPU, a few seconds)
========================================
Builds the tables, one figure and the number macros used by the LaTeX report, directly from the
result files, so every number in the report can be regenerated:

  outputs/report_assets/numbers.tex     \\newcommand macros with the key numbers quoted in the text
  outputs/report_assets/tab_main.tex    in-distribution comparison (3 seeds)
  outputs/report_assets/tab_groups.tex  class x background benchmark (binary models)
  outputs/report_assets/tab_insitu.tex  FathomNet same-camera experiment
  outputs/report_assets/tab_ablation.tex ablations / imbalance / data size (fine task, ResNet-50)
  outputs/report_assets/fig_dumbbell.pdf worst-group accuracy before -> after the fix

Run after stats.py and run_all.py --stages summary:   python report_assets.py
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(os.environ.get("OUT_DIR", Path(__file__).resolve().parent / "outputs"))
DST = OUT / "report_assets"
DST.mkdir(parents=True, exist_ok=True)

A = pd.read_csv(OUT / "extra" / "all_methods.csv")
R = pd.read_csv(OUT / "summary" / "results_mean_std.csv")
macros = {}


def ms(series, digits=2, pct=False):
    """mean +- std text for LaTeX (std only when >1 value)."""
    v = pd.to_numeric(series, errors="coerce").dropna().values
    if not len(v):
        return "--"
    f = 100 if pct else 1
    m = v.mean() * f
    if len(v) > 1:
        return f"{m:.{digits}f}{{\\scriptsize$\\pm${v.std(ddof=1) * f:.{digits}f}}}"
    return f"{m:.{digits}f}"


def mean(series):
    v = pd.to_numeric(series, errors="coerce").dropna().values
    return float(v.mean()) if len(v) else float("nan")


def rows(config):
    return A[A.config == config]


def macro(name, value, digits=2):
    macros[name] = f"{value:.{digits}f}" if isinstance(value, float) else str(value)


# ---------------------------------------------------------------- main table (3 seeds)
MAIN = [("Majority class", "majority"), ("``Bluer = jellyfish'' rule", "rule_blue"),
        ("Raw pixels (SVM / LogReg)", ("pixels+svm", "pixels+logreg")),
        ("Colour hist.\\ + LogReg", "color+logreg"), ("HOG + SVM", "hog+svm"),
        ("HOG + colour + SVM", "hog+color+svm"),
        ("TinyCNN (scratch)", "tinycnn | scratch | aug=standard"), ("VGG16", "vgg16 | aug=standard"),
        ("ResNet-50", "resnet50 | aug=standard"), ("EfficientNet-B0", "efficientnet_b0 | aug=standard"),
        ("ViT-B/16", "vit_b_16 | aug=standard"), ("CLIP zero-shot", "clip_zeroshot_underwater"),
        ("CLIP linear probe", "clip_linear_probe")]


def rcell(task, config, col, digits=3):
    r = R[(R.task == task) & (R.config == config)]
    if not len(r):
        alt = {"rule_blue": None, "hog+svm": "hog+svm", "color+logreg": "color+logreg"}
        return "--"
    r = r.iloc[0]
    m, s = r.get(f"{col}_mean"), r.get(f"{col}_std")
    if pd.isna(m):
        return "--"
    return f"{m:.{digits}f}" if pd.isna(s) else f"{m:.{digits}f}{{\\scriptsize$\\pm${s:.{digits}f}}}"


lines = []
for name, cfg in MAIN:
    bcfg, fcfg = cfg if isinstance(cfg, tuple) else (cfg, cfg)     # some baselines use a different classifier per task
    seeds = R[(R.task == "fine") & (R.config == fcfg)].n_seeds
    seeds = int(seeds.iloc[0]) if len(seeds) else int(R[(R.task == "binary") & (R.config == bcfg)].n_seeds.iloc[0])
    lines.append(f"{name} & {seeds} & {rcell('binary', bcfg, 'macro_f1')} & {rcell('fine', fcfg, 'macro_f1')} & "
                 f"{rcell('fine', fcfg, 'worst_class_recall')} & {rcell('fine', fcfg, 'ece')} \\\\")


def dino_fine_metrics():
    """Worst-class recall and ECE of the 7-class DINOv2 probe on the normal test set (from its prediction file;
    the result is cached in extra/ so that the report can be rebuilt from results_snapshot/ alone)."""
    cache = OUT / "extra" / "dinov2_fine_test_metrics.csv"
    pred = OUT / "preds" / "fine_dinov2_base__probe.csv"
    if pred.exists():
        p = pd.read_csv(pred)
        p = p[p.set == "test"]
        y, yp, conf = p.y_fine.values, p.pred_fine.values, p.conf.values
        worst = min((yp[y == k] == k).mean() for k in np.unique(y))
        correct, ece = (yp == y).astype(float), 0.0                 # same 15-bin ECE as common.py
        edges = np.linspace(0, 1, 16)
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (conf > lo) & (conf <= hi)
            if m.any():
                ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
        pd.DataFrame([{"worst_class_recall": worst, "ece": ece}]).to_csv(cache, index=False)
    if cache.exists():
        d = pd.read_csv(cache).iloc[0]
        return f"{d.worst_class_recall:.3f}", f"{d.ece:.3f}"
    return "--", "--"


dino_b, dino_f = rows("binary_dinov2_base__probe"), rows("fine_dinov2_base__probe")
dino_worst, dino_ece = dino_fine_metrics()
lines.append(f"DINOv2 linear probe & 1 & {ms(dino_b.test_macro_f1_bin, 3)} & {ms(dino_f.test_macro_f1_fine, 3)} & "
             f"{dino_worst} & {dino_ece} \\\\")
(DST / "tab_main.tex").write_text("\n".join(lines), encoding="utf-8")

# ---------------------------------------------------------------- group table (binary)
GROUP_ROWS = [
    ("\\multicolumn{10}{l}{\\textit{Fine-tuned networks (ERM)}}", None),
    ("TinyCNN", "binary_tinycnn_scratch_ft_aug-standard__erm"),
    ("VGG16", "binary_vgg16_pt_ft_aug-standard__erm"),
    ("ResNet-50", "binary_resnet50_pt_ft_aug-standard__erm"),
    ("EfficientNet-B0", "binary_efficientnet_b0_pt_ft_aug-standard__erm"),
    ("ViT-B/16", "binary_vit_b_16_pt_ft_aug-standard__erm"),
    ("\\multicolumn{10}{l}{\\textit{+ underwater augmentation (ERM)}}", None),
    ("ResNet-50", "binary_resnet50_pt_ft_aug-underwater__erm"),
    ("EfficientNet-B0", "binary_efficientnet_b0_pt_ft_aug-underwater__erm"),
    ("ViT-B/16", "binary_vit_b_16_pt_ft_aug-underwater__erm"),
    ("\\multicolumn{10}{l}{\\textit{+ last-layer retraining (DFR~\\cite{kirichenko2023dfr})}}", None),
    ("TinyCNN", "binary_tinycnn_scratch_ft_aug-standard__dfr"),
    ("VGG16", "binary_vgg16_pt_ft_aug-standard__dfr"),
    ("ResNet-50", "binary_resnet50_pt_ft_aug-standard__dfr"),
    ("EfficientNet-B0", "binary_efficientnet_b0_pt_ft_aug-standard__dfr"),
    ("ViT-B/16", "binary_vit_b_16_pt_ft_aug-standard__dfr"),
    ("\\multicolumn{10}{l}{\\textit{Frozen foundation models}}", None),
    ("CLIP zero-shot", "binary_clip__zeroshot"),
    ("CLIP linear probe", "binary_clip__probe"),
    ("CLIP probe, group-bal.", "binary_clip__probe_dfr"),
    ("DINOv2 linear probe", "binary_dinov2_base__probe"),
    ("DINOv2 probe, group-bal.", "binary_dinov2_base__probe_dfr"),
]
lines = []
for name, cfg in GROUP_ROWS:
    if cfg is None:
        lines.append("\\midrule " + name + " \\\\")
        continue
    d = rows(cfg)
    n = d.seed.nunique()
    lines.append(" & ".join([name, str(n), ms(d.test_acc), ms(d.acc_bag_rov), ms(d.acc_bag_clear), ms(d.acc_jelly_clear),
                             ms(d.acc_jelly_rov), "\\textbf{" + ms(d.worst_group) + "}", ms(d.bgonly_rov_plastic_rate),
                             ms(d.acc_jelly_insitu)]) + " \\\\")
(DST / "tab_groups.tex").write_text("\n".join(lines), encoding="utf-8")

# ---------------------------------------------------------------- macros for the text
for arch, key in [("resnet50", "Res"), ("efficientnet_b0", "Eff"), ("vit_b_16", "Vit"), ("vgg16", "Vgg"),
                  ("tinycnn", "Tiny")]:
    pre = "binary_tinycnn_scratch_ft" if arch == "tinycnn" else f"binary_{arch}_pt_ft"
    e, f = rows(f"{pre}_aug-standard__erm"), rows(f"{pre}_aug-standard__dfr")
    macro(f"{key}ErmWG", mean(e.worst_group)); macro(f"{key}DfrWG", mean(f.worst_group))
    macro(f"{key}ErmBagClear", mean(e.acc_bag_clear)); macro(f"{key}DfrBagClear", mean(f.acc_bag_clear))
    macro(f"{key}ErmTest", mean(e.test_acc)); macro(f"{key}DfrTest", mean(f.test_acc))
    macro(f"{key}ErmBgOnly", mean(e.bgonly_rov_plastic_rate))
    macro(f"{key}ErmInsitu", mean(e.acc_jelly_insitu)); macro(f"{key}DfrInsitu", mean(f.acc_jelly_insitu))
    macro(f"{key}ErmWGsd", float(pd.to_numeric(e.worst_group).std()) if len(e) > 1 else 0.0)
    macro(f"{key}DfrWGsd", float(pd.to_numeric(f.worst_group).std()) if len(f) > 1 else 0.0)
    for _, r_ in e.iterrows():
        macro(f"{key}ErmWGs{dict(zip('345', 'ABC'))[str(int(r_.seed))[-1]]}", float(r_.worst_group))
    for _, r_ in f.iterrows():
        macro(f"{key}DfrWGs{dict(zip('345', 'ABC'))[str(int(r_.seed))[-1]]}", float(r_.worst_group))
    if arch in ("resnet50", "efficientnet_b0", "vit_b_16"):
        u = rows(f"{pre}_aug-underwater__erm")
        macro(f"{key}UwWG", mean(u.worst_group)); macro(f"{key}UwBagClear", mean(u.acc_bag_clear))
erm = A[(A.task == "binary") & A.config.str.endswith("__erm") & A.config.str.contains("_pt_ft_aug-standard")]
macro("ErmBgOnlyMin", float(erm.groupby("config").bgonly_rov_plastic_rate.mean().min()))
macro("ErmBgOnlyMax", float(erm.groupby("config").bgonly_rov_plastic_rate.mean().max()))
macro("ErmTestMin", float(erm.groupby("config").test_acc.mean().min()))
macro("ErmBgOnlyMinPct", float(erm.groupby("config").bgonly_rov_plastic_rate.mean().min()) * 100, 0)
macro("ErmBgOnlyMaxPct", float(erm.groupby("config").bgonly_rov_plastic_rate.mean().max()) * 100, 0)
macro("ErmWGMin", float(erm.groupby("config").worst_group.mean().min()))
macro("ErmWGMax", float(erm.groupby("config").worst_group.mean().max()))
macro("OodPlasticMinPct", float(erm.groupby("config").ood_pred_plastic.mean().min()) * 100, 0)
macro("OodPlasticMaxPct", float(erm.groupby("config").ood_pred_plastic.mean().max()) * 100, 0)
macro("ErmInsituMin", float(erm.groupby("config").acc_jelly_insitu.mean().min()))
macro("ErmInsituMax", float(erm.groupby("config").acc_jelly_insitu.mean().max()))
uw = A[(A.task == "binary") & A.config.str.endswith("__erm") & A.config.str.contains("aug-underwater")]
macro("UwWGMin", float(uw.groupby("config").worst_group.mean().min()))
macro("UwWGMax", float(uw.groupby("config").worst_group.mean().max()))
for cfg, key in [("binary_clip__zeroshot", "ClipZs"), ("binary_clip__probe", "ClipLp"), ("binary_clip__probe_dfr", "ClipGb"),
                 ("binary_dinov2_base__probe", "DinoLp"), ("binary_dinov2_base__probe_dfr", "DinoGb")]:
    d = rows(cfg)
    macro(f"{key}WG", mean(d.worst_group)); macro(f"{key}Test", mean(d.test_acc))
    macro(f"{key}BagClear", mean(d.acc_bag_clear)); macro(f"{key}JellyRov", mean(d.acc_jelly_rov))
    macro(f"{key}BgOnly", mean(d.bgonly_rov_plastic_rate)); macro(f"{key}Insitu", mean(d.acc_jelly_insitu))
    macro(f"{key}OodKnn", mean(d.ood_auroc_knn))
for cfg, key in [("fine_resnet50_pt_ft_aug-standard__erm", "FineResErm"), ("fine_resnet50_pt_ft_aug-standard__dfr", "FineResDfr"),
                 ("fine_vit_b_16_pt_ft_aug-standard__erm", "FineVitErm"), ("fine_vit_b_16_pt_ft_aug-standard__dfr", "FineVitDfr"),
                 ("fine_efficientnet_b0_pt_ft_aug-standard__erm", "FineEffErm"), ("fine_efficientnet_b0_pt_ft_aug-standard__dfr", "FineEffDfr"),
                 ("fine_dinov2_base__probe_dfr", "FineDinoGb")]:
    d = rows(cfg)
    macro(f"{key}WG", mean(d.worst_group)); macro(f"{key}FineFone", mean(d.test_macro_f1_fine))
    macro(f"{key}Insitu", mean(d.acc_jelly_insitu)); macro(f"{key}BagClear", mean(d.acc_bag_clear))
# OOD
for cfg, key in [("binary_resnet50_pt_ft_aug-standard__erm", "Res"), ("binary_vit_b_16_pt_ft_aug-standard__erm", "Vit"),
                 ("binary_efficientnet_b0_pt_ft_aug-standard__erm", "Eff")]:
    d = rows(cfg)
    macro(f"{key}OodMsp", mean(d.ood_auroc_msp)); macro(f"{key}OodEnergy", mean(d.ood_auroc_energy))
    macro(f"{key}OodKnn", mean(d.ood_auroc_knn)); macro(f"{key}OodPlastic", mean(d.ood_pred_plastic))
# McNemar ERM vs DFR
mc = pd.read_csv(OUT / "extra" / "mcnemar_holm.csv")
ed = mc[mc.family.str.startswith("ERM vs DFR")]
sig = ed["significant (Holm, 5%)"].astype(bool)
better = sig & (ed.only_a_correct > ed.only_b_correct)
worse = sig & (ed.only_a_correct < ed.only_b_correct)
macro("McnSig", int(better.sum()), 0); macro("McnWorse", int(worse.sum()), 0); macro("McnTotal", int(len(ed)), 0)
for fam, key in [("binary", "Bin"), ("fine", "Fine")]:
    sub = ed.a.str.startswith(fam + ":")
    macro(f"Mcn{key}Better", int((better & sub).sum()), 0); macro(f"Mcn{key}Total", int(sub.sum()), 0)
# cost analysis (Bayes threshold r/(1+r)), shifted benchmark
cc = pd.read_csv(OUT / "extra" / "cost_curves.csv")
for cfg, key in [("binary_resnet50_pt_ft_aug-standard__erm", "ResErm"), ("binary_resnet50_pt_ft_aug-standard__dfr", "ResDfr"),
                 ("binary_efficientnet_b0_pt_ft_aug-standard__erm", "EffErm"), ("binary_efficientnet_b0_pt_ft_aug-standard__dfr", "EffDfr"),
                 ("binary_clip__zeroshot", "ClipZs"), ("binary_clip__probe", "ClipLp"),
                 ("binary_dinov2_base__probe", "DinoLp"), ("binary_dinov2_base__probe_dfr", "DinoGb")]:
    d = cc[cc.config == cfg].set_index("ratio")
    if not len(d):
        continue
    one = d.iloc[(abs(d.index - 1.0)).argmin()]; five = d.iloc[(abs(d.index - 5.012)).argmin()]
    macro(f"{key}CostOne", float(one.cost_shifted), 3); macro(f"{key}CostFive", float(five.cost_shifted), 3)
    macro(f"{key}RecallFive", float(five.bag_recall_shifted)); macro(f"{key}GrabOne", float(one.jelly_grabbed_shifted) * 100, 0)
# low-level cue test
cue = pd.read_csv(OUT / "extra" / "lowlevel_cue_test.csv")
macro("CueAccNone", float(cue.test_accuracy.iloc[0]) * 100, 1); macro("CueAucNone", float(cue.test_auc.iloc[0]), 3)
macro("CueAccAug", float(cue.test_accuracy.iloc[1]) * 100, 1); macro("CueAucAug", float(cue.test_auc.iloc[1]), 3)
# masks
ml = pd.read_csv(OUT / "extra" / "jelly_mask_log.csv")
macro("MaskOk", float(ml.ok.mean()) * 100, 0)

# ---------------------------------------------------------------- in-situ table
ins = pd.read_csv(OUT / "extra" / "insitu_runs.csv")
lines = []
for model in ["ResNet-50", "CLIP linear probe"]:
    for src, lab in [("kaggle", "Kaggle (clear water)"), ("insitu", "FathomNet (ROV)"), ("both", "both")]:
        d = ins[(ins.model == model) & (ins["jelly source"] == src)]
        if not len(d):
            continue
        lines.append(" & ".join([model if src == "kaggle" else "", lab, str(d.seed.nunique()),
                                 ms(d["test vs Kaggle jellies"]), ms(d["test vs in-situ jellies"]), ms(d.bag_clear),
                                 ms(d.jelly_rov), ms(d.bgonly_rov)]) + " \\\\")
        if model == "ResNet-50":
            k = {"kaggle": "Kag", "insitu": "Ins", "both": "Both"}[src]
            macro(f"Ins{k}FoneKag", mean(d["test vs Kaggle jellies"])); macro(f"Ins{k}FoneIns", mean(d["test vs in-situ jellies"]))
            macro(f"Ins{k}BagClear", mean(d.bag_clear)); macro(f"Ins{k}JellyRov", mean(d.jelly_rov))
            macro(f"Ins{k}BgOnly", mean(d.bgonly_rov))
    lines.append("\\midrule")
(DST / "tab_insitu.tex").write_text("\n".join(lines[:-1]), encoding="utf-8")

# ---------------------------------------------------------------- ablation table (fine, ResNet-50)
ABL = [("Full (ImageNet, fine-tuned, std.\\ aug., class weights)", "resnet50 | aug=standard"),
       ("No augmentation", "resnet50 | aug=none"), ("Underwater augmentation", "resnet50 | aug=underwater"),
       ("Frozen backbone (linear probe)", "resnet50 | frozen | aug=standard"),
       ("No pretraining (from scratch)", "resnet50 | scratch | aug=standard"),
       ("10\\% of training data", "resnet50 | aug=standard | 10% data"), ("25\\% of training data", "resnet50 | aug=standard | 25% data"),
       ("50\\% of training data", "resnet50 | aug=standard | 50% data"),
       ("Imbalance: none (plain CE)", "resnet50 | aug=standard | imb=none"),
       ("Imbalance: oversampling", "resnet50 | aug=standard | imb=oversample"),
       ("Imbalance: undersampling", "resnet50 | aug=standard | imb=undersample"),
       ("Imbalance: focal loss", "resnet50 | aug=standard | imb=focal"),
       ("Imbalance: combination", "resnet50 | aug=standard | imb=combo")]
lines = []
for name, cfg in ABL:
    r = R[(R.task == "fine") & (R.config == cfg)]
    if not len(r):
        continue
    lines.append(f"{name} & {int(r.n_seeds.iloc[0])} & {rcell('fine', cfg, 'macro_f1')} & {rcell('fine', cfg, 'worst_class_recall')} & "
                 f"{rcell('fine', cfg, 'plastic_recall')} & {rcell('fine', cfg, 'ece')} \\\\")
    if name.startswith("No pretraining") or name.startswith("50"):
        lines.append("\\midrule")
(DST / "tab_ablation.tex").write_text("\n".join(lines), encoding="utf-8")

# in-distribution macros (3 seeds)
for task in ["binary", "fine"]:
    for cfg, key in [("resnet50 | aug=standard", "Res"), ("efficientnet_b0 | aug=standard", "Eff"), ("vit_b_16 | aug=standard", "Vit"),
                     ("vgg16 | aug=standard", "Vgg"), ("tinycnn | scratch | aug=standard", "Tiny"),
                     ("resnet50 | aug=underwater", "ResUw"), ("efficientnet_b0 | aug=underwater", "EffUw"), ("vit_b_16 | aug=underwater", "VitUw")]:
        r = R[(R.task == task) & (R.config == cfg)]
        if len(r):
            macro(f"{'Bin' if task == 'binary' else 'Fine'}{key}Fone", float(r.macro_f1_mean.iloc[0]), 3)
            macro(f"{'Bin' if task == 'binary' else 'Fine'}{key}FoneStd", float(r.macro_f1_std.iloc[0]), 3)

(DST / "numbers.tex").write_text("\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(macros.items())) + "\n",
                                 encoding="utf-8")

# ---------------------------------------------------------------- dumbbell figure
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, INK2, GRID = "#1f1f1e", "#5f5e5a", "#d9d8d2"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
items = [("TinyCNN", "binary_tinycnn_scratch_ft_aug-standard__erm", "binary_tinycnn_scratch_ft_aug-standard__dfr", None),
         ("VGG16", "binary_vgg16_pt_ft_aug-standard__erm", "binary_vgg16_pt_ft_aug-standard__dfr", None),
         ("ViT-B/16", "binary_vit_b_16_pt_ft_aug-standard__erm", "binary_vit_b_16_pt_ft_aug-standard__dfr", "binary_vit_b_16_pt_ft_aug-underwater__erm"),
         ("EfficientNet-B0", "binary_efficientnet_b0_pt_ft_aug-standard__erm", "binary_efficientnet_b0_pt_ft_aug-standard__dfr", "binary_efficientnet_b0_pt_ft_aug-underwater__erm"),
         ("ResNet-50", "binary_resnet50_pt_ft_aug-standard__erm", "binary_resnet50_pt_ft_aug-standard__dfr", "binary_resnet50_pt_ft_aug-underwater__erm"),
         ("CLIP ViT-B/32", "binary_clip__probe", "binary_clip__probe_dfr", None),
         ("DINOv2 ViT-B/14", "binary_dinov2_base__probe", "binary_dinov2_base__probe_dfr", None)]
fig, ax = plt.subplots(figsize=(3.4, 2.6))
for i, (lab, a_, b_, u_) in enumerate(items):
    a, b = mean(rows(a_).worst_group), mean(rows(b_).worst_group)
    ax.plot([a, b], [i, i], color=GRID, lw=2.5, zorder=1)
    ax.scatter([a], [i], s=28, color=BLUE, zorder=3, label="standard training / linear probe" if i == 0 else None)
    ax.scatter([b], [i], s=28, color=ORANGE, zorder=3, label="+ group-balanced last layer (DFR)" if i == 0 else None)
    if u_:
        ax.scatter([mean(rows(u_).worst_group)], [i], s=26, marker="x", color=INK2, zorder=3,
                   label="+ underwater augmentation" if lab == "ViT-B/16" else None)
zs = mean(rows("binary_clip__zeroshot").worst_group)
ax.scatter([zs], [5], s=30, marker="D", color=AQUA, zorder=3, label="CLIP zero-shot")
ax.axhline(4.5, color=GRID, lw=0.8, ls="--")
ax.set_yticks(range(len(items))); ax.set_yticklabels([i[0] for i in items], fontsize=7, color=INK)
ax.set_xlim(-0.02, 1.02); ax.set_xlabel("worst-group accuracy (bag vs. jellyfish)", fontsize=7, color=INK2)
ax.tick_params(axis="x", labelsize=7, colors=INK2)
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)
for s in ["left", "bottom"]:
    ax.spines[s].set_color(GRID)
ax.grid(axis="x", color=GRID, lw=0.5); ax.set_axisbelow(True)
ax.legend(fontsize=5.6, frameon=False, loc="upper center", bbox_to_anchor=(0.35, -0.22), ncol=2,
          handletextpad=0.3, columnspacing=0.8)
fig.tight_layout(pad=0.3)
fig.savefig(DST / "fig_dumbbell.pdf"); fig.savefig(DST / "fig_dumbbell.png", dpi=200)
print(f"Report assets written to {DST}")
print((DST / "numbers.tex").read_text())


# =============================================================================
# Extra figures for the report (all generated from the result files)
# =============================================================================
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]    # validated categorical palette, slots 1-4


def style(ax, grid="y"):
    for s_ in ["top", "right"]:
        ax.spines[s_].set_visible(False)
    for s_ in ["left", "bottom"]:
        ax.spines[s_].set_color(GRID)
    ax.tick_params(labelsize=6.5, colors=INK2)
    ax.grid(axis=grid, color=GRID, lw=0.5); ax.set_axisbelow(True)


# ---- (1) heat map: accuracy of every method on every group ----
HM_ROWS = [("ResNet-50", "binary_resnet50_pt_ft_aug-standard__erm"), ("EfficientNet-B0", "binary_efficientnet_b0_pt_ft_aug-standard__erm"),
           ("ViT-B/16", "binary_vit_b_16_pt_ft_aug-standard__erm"), ("VGG16", "binary_vgg16_pt_ft_aug-standard__erm"),
           ("TinyCNN", "binary_tinycnn_scratch_ft_aug-standard__erm"),
           ("ResNet-50 + underwater aug.", "binary_resnet50_pt_ft_aug-underwater__erm"),
           ("EfficientNet-B0 + underwater aug.", "binary_efficientnet_b0_pt_ft_aug-underwater__erm"),
           ("ViT-B/16 + underwater aug.", "binary_vit_b_16_pt_ft_aug-underwater__erm"),
           ("ResNet-50 + DFR", "binary_resnet50_pt_ft_aug-standard__dfr"), ("EfficientNet-B0 + DFR", "binary_efficientnet_b0_pt_ft_aug-standard__dfr"),
           ("ViT-B/16 + DFR", "binary_vit_b_16_pt_ft_aug-standard__dfr"),
           ("CLIP zero-shot", "binary_clip__zeroshot"), ("CLIP linear probe", "binary_clip__probe"),
           ("CLIP probe, group-balanced", "binary_clip__probe_dfr"), ("DINOv2 linear probe", "binary_dinov2_base__probe"),
           ("DINOv2 probe, group-balanced", "binary_dinov2_base__probe_dfr")]
HM_COLS = [("normal\ntest", "test_acc"), ("bag\nROV", "acc_bag_rov"), ("bag\ngrey", "acc_bag_gray"), ("bag\nclear", "acc_bag_clear"),
           ("jelly\nclear", "acc_jelly_clear"), ("jelly\ngrey", "acc_jelly_gray"), ("jelly\nROV", "acc_jelly_rov"),
           ("jelly\nin-situ", "acc_jelly_insitu"), ("empty ROV\n= no bag", "bgonly_rov_plastic_rate")]
M = np.array([[mean(rows(cfg)[col]) if col != "bgonly_rov_plastic_rate" else 1 - mean(rows(cfg)[col]) for _, col in HM_COLS]
              for _, cfg in HM_ROWS])
from matplotlib.colors import LinearSegmentedColormap
cmap = LinearSegmentedColormap.from_list("seq", ["#f4f3ee", "#9ec3ef", "#2a78d6", "#0d3b75"])
fig, ax = plt.subplots(figsize=(7.0, 3.9))
ax.imshow(M, cmap=cmap, vmin=0, vmax=1, aspect="auto")
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=6.3, color="white" if M[i, j] > 0.62 else INK)
ax.set_xticks(range(len(HM_COLS))); ax.set_xticklabels([c for c, _ in HM_COLS], fontsize=6.5, color=INK)
ax.set_yticks(range(len(HM_ROWS))); ax.set_yticklabels([r for r, _ in HM_ROWS], fontsize=6.5, color=INK)
ax.xaxis.tick_top()
for y in [4.5, 7.5, 10.5]:
    ax.axhline(y, color="white", lw=2.5)
ax.axvline(0.5, color="white", lw=2.5); ax.axvline(7.5, color="white", lw=2.5)
for s_ in ax.spines.values():
    s_.set_visible(False)
ax.tick_params(length=0)
fig.tight_layout(pad=0.3)
fig.savefig(DST / "fig_group_heatmap.pdf"); fig.savefig(DST / "fig_group_heatmap.png", dpi=200); plt.close(fig)

# ---- (2) robustness curves for the four pretrained networks (mean +- std over seeds) ----
RB = pd.read_csv(OUT / "summary" / "robustness_summary.csv")
NETS = [("ResNet-50", "resnet50 | aug=standard"), ("EfficientNet-B0", "efficientnet_b0 | aug=standard"),
        ("ViT-B/16", "vit_b_16 | aug=standard"), ("VGG16", "vgg16 | aug=standard")]
PANELS = [("Underwater severity", ["test", "test_uw1", "test_uw3", "test_uw5"], [0, 1, 3, 5]),
          ("Gaussian noise $\\sigma$", ["test", "test_noise", "test_noise", "test_noise"], [0, 0.05, 0.1, 0.2]),
          ("FGSM $\\epsilon$ (/255)", ["test", "test_fgsm", "test_fgsm", "test_fgsm", "test_fgsm"], [0, 1, 2, 4, 8])]
fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.1), sharey=True)
for ax, (title, sets, levels) in zip(axes, PANELS):
    for (lab, cfg), col in zip(NETS, SERIES):
        ys, ss = [], []
        for es, lv in zip(sets, levels):
            q = RB[(RB.task == "binary") & (RB.label == cfg) & (RB.eval_set == es)]
            if es not in ("test", "test_uw1", "test_uw3", "test_uw5"):
                q = q[np.isclose(q.level, lv)]
            ys.append(float(q.accuracy.iloc[0]) if len(q) else np.nan)
            ss.append(float(q.accuracy_std.fillna(0).iloc[0]) if len(q) else 0)
        ys, ss = np.array(ys), np.array(ss)
        ax.plot(levels, ys, color=col, lw=1.6, marker="o", ms=3, label=lab)
        ax.fill_between(levels, ys - ss, ys + ss, color=col, alpha=0.12, lw=0)
    ax.set_title(title, fontsize=7.5, color=INK, loc="left"); style(ax)
axes[0].set_ylabel("binary test accuracy", fontsize=7, color=INK2); axes[0].set_ylim(0.3, 1.02)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, fontsize=6.5, frameon=False, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(pad=0.3, rect=(0, 0.1, 1, 1))
fig.savefig(DST / "fig_robustness.pdf"); fig.savefig(DST / "fig_robustness.png", dpi=200); plt.close(fig)

# ---- (3) cost curves ----
CC = pd.read_csv(OUT / "extra" / "cost_curves.csv")
fig, ax = plt.subplots(figsize=(3.4, 2.1))
for (lab, cfg), col in zip([("ResNet-50", "binary_resnet50_pt_ft_aug-standard__erm"), ("ResNet-50 + DFR", "binary_resnet50_pt_ft_aug-standard__dfr"),
                            ("CLIP zero-shot", "binary_clip__zeroshot"), ("DINOv2 probe, group-bal.", "binary_dinov2_base__probe_dfr")], SERIES):
    d = CC[CC.config == cfg].sort_values("ratio")
    ax.plot(d.ratio, d.cost_shifted, color=col, lw=1.8, label=lab)
ax.set_xscale("log"); style(ax, "both")
ax.set_xlabel("$r$ = harm of grabbing a jellyfish / harm of missing a bag", fontsize=6.5, color=INK2)
ax.set_ylabel("expected cost per object", fontsize=6.5, color=INK2)
ax.legend(fontsize=6, frameon=False)
fig.tight_layout(pad=0.3)
fig.savefig(DST / "fig_cost.pdf"); fig.savefig(DST / "fig_cost.png", dpi=200); plt.close(fig)

# ---- (4) in-situ experiment bars ----
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.2))
srcs = [("kaggle", "Kaggle (clear water)"), ("insitu", "FathomNet (ROV)"), ("both", "both")]
mets = [("test vs Kaggle jellies", "test macro-F1\nKaggle jellies"), ("test vs in-situ jellies", "test macro-F1\nin-situ jellies"),
        ("bag_clear", "bag on clear\nwater (acc.)"), ("jelly_rov", "jelly on ROV\n(acc.)")]
for ax, model in zip(axes, ["ResNet-50", "CLIP linear probe"]):
    x = np.arange(len(mets)); w = 0.26
    for k, ((src, lab), col) in enumerate(zip(srcs, SERIES)):
        d = ins[(ins.model == model) & (ins["jelly source"] == src)]
        ax.bar(x + (k - 1) * w, [d[m].mean() for m, _ in mets], w * 0.92, yerr=[d[m].std() if len(d) > 1 else 0 for m, _ in mets],
               color=col, label=f"trained with {lab} jellyfish", error_kw=dict(lw=0.7, ecolor=INK2, capsize=1.5))
    ax.set_xticks(x); ax.set_xticklabels([l for _, l in mets], fontsize=6.3); ax.set_ylim(0, 1.05)
    ax.set_title(model, fontsize=7.5, color=INK, loc="left"); style(ax)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, fontsize=6.3, frameon=False, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(pad=0.3, rect=(0, 0.1, 1, 1))
fig.savefig(DST / "fig_insitu.pdf"); fig.savefig(DST / "fig_insitu.png", dpi=200); plt.close(fig)

# ---- (5) in-distribution comparison bars (fine task) ----
BARS = [("Majority", "majority"), ("Colour + LogReg", "color+logreg"), ("HOG + SVM", "hog+svm"),
        ("TinyCNN", "tinycnn | scratch | aug=standard"), ("CLIP zero-shot", "clip_zeroshot_underwater"),
        ("VGG16", "vgg16 | aug=standard"), ("CLIP probe", "clip_linear_probe"), ("ResNet-50", "resnet50 | aug=standard"),
        ("EfficientNet-B0", "efficientnet_b0 | aug=standard"), ("ViT-B/16", "vit_b_16 | aug=standard")]
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.0), sharey=True)
for ax, task, ttl in zip(axes, ["binary", "fine"], ["Binary (bag vs. jellyfish)", "Fine (7 classes)"]):
    vals, errs, labs = [], [], []
    for lab, cfg in BARS:
        r = R[(R.task == task) & (R.config == cfg)]
        if len(r):
            labs.append(lab); vals.append(float(r.macro_f1_mean.iloc[0])); errs.append(float(np.nan_to_num(r.macro_f1_std.iloc[0])))
    y = np.arange(len(labs))
    ax.barh(y, vals, xerr=errs, color=SERIES[0], height=0.62, error_kw=dict(lw=0.7, ecolor=INK2, capsize=1.5))
    for yi, v, e in zip(y, vals, errs):
        ax.text(v + e + 0.02, yi, f"{v:.2f}", va="center", fontsize=5.8, color=INK2)
    ax.set_yticks(y); ax.set_yticklabels(labs, fontsize=6.3); ax.set_xlim(0, 1.2)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0]); ax.spines["bottom"].set_bounds(0, 1)   # axis stops at 1.0
    ax.set_title(ttl, fontsize=7.5, color=INK, loc="left"); style(ax, "x")
    ax.set_xlabel("test macro-F1 (mean $\\pm$ std over seeds)", fontsize=6.3, color=INK2)
fig.tight_layout(pad=0.3)
fig.savefig(DST / "fig_main_bars.pdf"); fig.savefig(DST / "fig_main_bars.png", dpi=200); plt.close(fig)
print("figures done")


# ---------------------------------------------------------------- appendix: additional results
def md_table(path):
    """Read a Markdown table written by stats.py / summarize.py into a DataFrame (all cells as text)."""
    if not path.exists():
        return pd.DataFrame()
    rows_ = [l.strip().strip("|").split("|") for l in path.read_text(encoding="utf-8").splitlines()
             if l.strip().startswith("|") and not set(l.strip()) <= set("|-: ")]
    head = [h.strip() for h in rows_[0]]
    body = []
    for r in rows_[1:]:
        r = [c.strip() for c in r]
        extra = len(r) - len(head)                 # labels such as "resnet50 | aug=standard" contain '|'
        body.append([" | ".join(r[:extra + 1])] + r[extra + 1:])
    return pd.DataFrame(body, columns=head)


GB = md_table(OUT / "extra" / "table_groups_binary.md").set_index("method", drop=False)
OODT = md_table(OUT / "extra" / "table_ood_binary.md").set_index("method", drop=False)
XD = md_table(OUT / "summary" / "table_crossdata_binary.md")


def rb(label, es, level):
    q = RB[(RB.task == "binary") & (RB.label == label) & (RB.eval_set == es) & (RB.level == level)]
    return float(q.accuracy.iloc[0]) if len(q) else float("nan")


def first(s):
    """'0.837 ± 0.050' -> '0.84'"""
    try:
        return f"{float(str(s).split('±')[0]):.2f}"
    except ValueError:
        return "--"


EXTRA = [  # (row label, label in stats tables, robustness label, binary config in all_methods, fine config)
    ("TinyCNN", "TinyCNN", "tinycnn | aug=standard | scratch", "binary_tinycnn_scratch_ft_aug-standard",
     "fine_tinycnn_scratch_ft_aug-standard"),
    ("VGG16", "VGG16", "vgg16 | aug=standard", "binary_vgg16_pt_ft_aug-standard", "fine_vgg16_pt_ft_aug-standard"),
    ("ResNet-50", "ResNet-50", "resnet50 | aug=standard", "binary_resnet50_pt_ft_aug-standard",
     "fine_resnet50_pt_ft_aug-standard"),
    ("EfficientNet-B0", "EfficientNet-B0", "efficientnet_b0 | aug=standard", "binary_efficientnet_b0_pt_ft_aug-standard",
     "fine_efficientnet_b0_pt_ft_aug-standard"),
    ("ViT-B/16", "ViT-B/16", "vit_b_16 | aug=standard", "binary_vit_b_16_pt_ft_aug-standard",
     "fine_vit_b_16_pt_ft_aug-standard"),
]
lines = []
for name, lab, rlab, bcfg, fcfg in EXTRA:
    erm, dfr = GB.loc[lab] if lab in GB.index else None, GB.loc[lab + " + DFR"] if lab + " + DFR" in GB.index else None
    def wg(r, cfg):           # value and 95% cluster-bootstrap CI of the seed-813 model (the CI is computed for seed 813)
        if r is None:
            return "--"
        v = rows(cfg)
        v = v[v.seed.astype(str).str.contains("813")].worst_group
        return f"{float(v.iloc[0]):.2f} {r['worst-group 95% CI']}" if len(v) else "--"
    fe, fd = rows(fcfg + "__erm").worst_group, rows(fcfg + "__dfr").worst_group
    ece = R[(R.task == "binary") & (R.config == rlab.replace(" | scratch", "").replace("tinycnn | aug=standard", "tinycnn | scratch | aug=standard"))]
    ece = f"{float(ece.ece_mean.iloc[0]):.2f}" if len(ece) else "--"
    bright = [rb(rlab, "test_bright", lv) for lv in (0.5, 0.75, 1.25, 1.5)]
    o = OODT.loc[lab] if lab in OODT.index else None
    lines.append(f"{name} & {wg(erm, bcfg + '__erm')} & {wg(dfr, bcfg + '__dfr')} & {ms(fe)} & {ms(fd)} & {ece} & "
                 f"{min(bright):.2f}--{max(bright):.2f} & {rb(rlab, 'test_fgsm', 1.0):.2f} & "
                 f"{first(o['pred_plastic']) if o is not None else '--'} & {first(o['auroc_msp']) if o is not None else '--'} & "
                 f"{first(o['auroc_knn']) if o is not None else '--'} \\\\")
for name, lab in [("CLIP zero-shot", "CLIP ViT-B/32 zero-shot"), ("CLIP linear probe", "CLIP ViT-B/32 linear probe"),
                  ("DINOv2 linear probe", "DINOv2 ViT-B/14 linear probe")]:
    g = GB.loc[lab] if lab in GB.index else None
    gb = GB.loc[lab.split(" linear")[0].split(" zero")[0] + " probe + group-balanced"] \
        if "probe" in lab and lab.split(" linear")[0] + " probe + group-balanced" in GB.index else None
    o = OODT.loc[lab] if lab in OODT.index else None
    fcfg = {"CLIP zero-shot": None, "CLIP linear probe": "fine_clip__probe", "DINOv2 linear probe": "fine_dinov2_base__probe"}[name]
    fe = ms(rows(fcfg).worst_group) if fcfg else ms(rows("fine_clip__zeroshot").worst_group)
    fd = ms(rows(fcfg + "_dfr").worst_group) if fcfg else "--"
    lines.append(f"{name} & {'--' if g is None else first(g['worst-group']) + ' ' + g['worst-group 95% CI']} & "
                 f"{'--' if gb is None else first(gb['worst-group']) + ' ' + gb['worst-group 95% CI']} & {fe} & {fd} & "
                 f"-- & -- & -- & {first(o['pred_plastic']) if o is not None else '--'} & "
                 f"{first(o['auroc_msp']) if o is not None else '--'} & {first(o['auroc_knn']) if o is not None else '--'} \\\\")
(DST / "tab_extra.tex").write_text("\n".join(lines), encoding="utf-8")

# cross-dataset study (binary macro-F1)
lines = []
for name, lab in [("ResNet-50", "resnet50 | aug=standard"), ("ViT-B/16", "vit_b_16 | aug=standard"),
                  ("CLIP linear probe", "clip_linear_probe"), ("HOG + SVM", "hog+svm")]:
    cells = []
    for src in ["trashcan", "synthetic", "both"]:
        q = XD[(XD.label == lab) & (XD["trained on"] == src)] if len(XD) else XD
        cells += [f"{float(q.iloc[0]['test on TrashCan bags']):.2f}", f"{float(q.iloc[0]['test on synthetic bags']):.2f}"] \
            if len(q) else ["--", "--"]
    lines.append(f"{name} & " + " & ".join(cells) + " \\\\")
(DST / "tab_cross.tex").write_text("\n".join(lines), encoding="utf-8")
print("appendix tables done")
