"""
run_all.py
==========
Runs the whole project in the right order. Finished runs are skipped, so the same
command can simply be started again after Colab disconnects.

  python run_all.py                    # everything (several GPU hours: split the stages between members)
  python run_all.py --quick            # tiny version (few images, 1 epoch) to check that everything works
  python run_all.py --stages main --seeds 814

Stages, in order:
  data        prepare_data.py   clean, leak-free data table + extra test sets + class-balance check
  classic     baseline_hog.py   majority, "bluer = jellyfish" rule, pixels, HOG, colour (+ SMOTE etc. for imbalance)
  clip        clip_eval.py      CLIP zero-shot + linear probe
  lrsweep     train.py          learning-rate search on the VALIDATION set (ResNet-50, ViT-B/16)
  main        train.py          TinyCNN (scratch), ResNet-50, EfficientNet-B0, ViT-B/16, VGG16 - standard augmentation
  underwater  train.py          ResNet-50, EfficientNet-B0, ViT-B/16 with our underwater-domain augmentation
  imbalance   train.py          ResNet-50 (fine task) with none / oversample / undersample / focal / combo
  crossdata   train.py (+classic, clip)   train on synthetic bags or on both sources, test on both
  ablation    train.py          ResNet-50: no augmentation / frozen backbone / no pretraining
  datasize    train.py          ResNet-50 with 10%, 25%, 50% of the training data
  gradcam     gradcam.py        Grad-CAM analysis of the seed-813 models
  summary     summarize.py      tables and figures for the report

Robustness tests (robustness.py) run straight after each training run. To save disk space only the
seed-813 checkpoints are kept (Grad-CAM needs them); other seeds are deleted after testing.
"""

import argparse
import subprocess
import sys

import common as C

ALL_STAGES = ["data", "classic", "clip", "lrsweep", "main", "underwater", "imbalance", "crossdata",
              "ablation", "datasize", "gradcam", "summary"]
PRETRAINED_MODELS = ["resnet50", "efficientnet_b0", "vit_b_16"]
EXTRA_MAIN_MODELS = ["vgg16"]            # extra comparison model: main stage only
ABLATION_MODEL = "resnet50"
LR_SWEEP_MODELS = ["resnet50", "vit_b_16"]
LR_GRID = ["3e-05", "0.0001", "0.0003"]
CROSSDATA_MODELS = ["resnet50", "vit_b_16"]
IMBALANCE_TO_TEST = ["none", "oversample", "undersample", "focal", "combo"]   # "weighted" = the main run


def sh(cmd):
    """Run one of our scripts with the same Python and stop if it fails."""
    print("\n>>>", " ".join(cmd), flush=True)
    subprocess.run([sys.executable] + cmd, check=True, cwd=C.CODE_DIR)


def opt(opts, key, default):
    return opts[opts.index(key) + 1] if key in opts else default


def train_and_test(opts, seed, quick):
    """Train one model, run the robustness tests, and delete the checkpoint when it is no longer needed."""
    extra = ["--quick"] if quick else []
    sh(["train.py", "--seed", str(seed), "--skip_existing"] + opts + extra)
    lr = float(opt(opts, "--lr", "nan"))
    lr = None if lr != lr else lr                       # nan -> None
    name = C.run_name(opt(opts, "--task", "fine"), opt(opts, "--model", "resnet50"), "--scratch" not in opts,
                      "--freeze" in opts, opt(opts, "--aug", "standard"), seed,
                      float(opt(opts, "--train_frac", 1.0)), lr,
                      opt(opts, "--bag_source", C.DEFAULT_BAG_SOURCE), opt(opts, "--imbalance", C.DEFAULT_IMBALANCE))
    name = ("QUICK_" if quick else "") + name
    ckpt = C.CKPT_DIR / f"{name}.pt"
    if lr is not None:              # learning-rate search: only the validation score is needed
        ckpt.unlink(missing_ok=True)
        return
    if ckpt.exists():
        sh(["robustness.py", "--runs", name, "--skip_existing"] + extra)
        if seed != C.SEED:
            ckpt.unlink()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stages", nargs="*", default=ALL_STAGES, choices=ALL_STAGES)
    p.add_argument("--seeds", nargs="*", type=int, default=[813, 814, 815])
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    seeds = args.seeds[:1] if args.quick else args.seeds
    q = ["--quick"] if args.quick else []
    tasks = ["fine", "binary"]

    if "data" in args.stages and not C.SPLITS_CSV.exists():
        sh(["prepare_data.py"])
    if "classic" in args.stages:
        for t in tasks:
            sh(["baseline_hog.py", "--task", t] + q)
    if "clip" in args.stages:
        for t in tasks:
            sh(["clip_eval.py", "--task", t] + q)
    if "crossdata" in args.stages:            # classic + CLIP trained with the other bag sources
        for t in tasks:
            for src in ["synthetic", "both"]:
                sh(["baseline_hog.py", "--task", t, "--bag_source", src] + q)
                sh(["clip_eval.py", "--task", t, "--bag_source", src] + q)
    if "lrsweep" in args.stages:              # one seed is enough: only the validation score is used
        for m in LR_SWEEP_MODELS:
            for lr in LR_GRID:
                train_and_test(["--task", "fine", "--model", m, "--aug", "standard", "--lr", lr], seeds[0], args.quick)

    for seed in seeds:
        for t in tasks:
            if "main" in args.stages:
                train_and_test(["--task", t, "--model", "tinycnn", "--scratch", "--aug", "standard",
                                "--epochs", "30", "--patience", "8"], seed, args.quick)
                for m in PRETRAINED_MODELS + EXTRA_MAIN_MODELS:
                    train_and_test(["--task", t, "--model", m, "--aug", "standard"], seed, args.quick)
            if "underwater" in args.stages:
                for m in PRETRAINED_MODELS:
                    train_and_test(["--task", t, "--model", m, "--aug", "underwater"], seed, args.quick)
            if "crossdata" in args.stages:
                for m in CROSSDATA_MODELS:
                    for src in ["synthetic", "both"]:
                        train_and_test(["--task", t, "--model", m, "--aug", "standard", "--bag_source", src],
                                       seed, args.quick)
        if "imbalance" in args.stages:
            for method in IMBALANCE_TO_TEST:
                train_and_test(["--task", "fine", "--model", ABLATION_MODEL, "--aug", "standard",
                                "--imbalance", method], seed, args.quick)
        if "ablation" in args.stages:
            base = ["--task", "fine", "--model", ABLATION_MODEL]
            train_and_test(base + ["--aug", "none"], seed, args.quick)
            train_and_test(base + ["--aug", "standard", "--freeze"], seed, args.quick)
            train_and_test(base + ["--aug", "standard", "--scratch", "--epochs", "30", "--patience", "8"],
                           seed, args.quick)
        if "datasize" in args.stages:
            for frac in ["0.1", "0.25", "0.5"]:
                train_and_test(["--task", "fine", "--model", ABLATION_MODEL, "--aug", "standard",
                                "--train_frac", frac], seed, args.quick)
    if "gradcam" in args.stages:
        sh(["gradcam.py"] + q)
    if "summary" in args.stages:
        sh(["summarize.py"] + q)


if __name__ == "__main__":
    main()
