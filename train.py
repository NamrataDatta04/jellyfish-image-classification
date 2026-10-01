"""
train.py  (STEP 3)
==================
Trains ONE deep-learning model and tests it once on the test set.
run_all.py calls this script many times with different options (the experiments).

Main options:
  --task        fine (7 classes) or binary (plastic bag vs jellyfish)
  --model       tinycnn, resnet50, efficientnet_b0, vgg16, vit_b_16, ...
  --aug         none / standard / underwater (our proposed domain augmentation)
  --bag_source  trashcan (real underwater bags, default) / synthetic / both
  --imbalance   none / weighted (default) / oversample / undersample / focal / combo
  --scratch     do not start from ImageNet weights
  --freeze      train only the last layer
  --train_frac  use only part of the training data (data-efficiency study)
  --lr          learning rate (used by the learning-rate search)

Examples:
  python train.py --task fine --model resnet50
  python train.py --task binary --model vit_b_16 --aug underwater --seed 814
  python train.py --task fine --model resnet50 --imbalance combo
  python train.py --task binary --model resnet50 --bag_source synthetic      # cross-dataset study

Outputs:
  outputs/results/<run>.json          settings, learning curves and test metrics
  outputs/checkpoints/<run>.pt        best weights (used by robustness.py and gradcam.py)
  outputs/figures/runs/<run>_*.png    learning curves, confusion matrix, mistakes, ROC
"""

import argparse

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

import common as C


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=["fine", "binary"], default="fine")
    p.add_argument("--model", choices=C.MODEL_CHOICES, default="resnet50")
    p.add_argument("--aug", choices=["none", "standard", "underwater"], default="standard")
    p.add_argument("--bag_source", choices=C.BAG_SOURCES, default=C.DEFAULT_BAG_SOURCE)
    p.add_argument("--imbalance", choices=C.IMBALANCE_METHODS, default=C.DEFAULT_IMBALANCE)
    p.add_argument("--scratch", action="store_true", help="do NOT use ImageNet weights")
    p.add_argument("--freeze", action="store_true", help="train only the last layer")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--patience", type=int, default=5, help="stop if validation macro-F1 does not improve for this many epochs")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=None, help="default: 1e-3 for scratch/frozen models, 1e-4 for fine-tuning")
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--train_frac", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=C.SEED)
    p.add_argument("--quick", action="store_true", help="tiny run (few images, 1 epoch) to check the code works")
    p.add_argument("--skip_existing", action="store_true")
    return p.parse_args()


def show_mistakes(model, loader, names, path, max_show=12):
    """Save a grid of wrongly classified test images (same idea as show_mistakes in the lab)."""
    model.eval()
    imgs, true, pred = [], [], []
    with torch.no_grad():
        for x, y in loader:
            p = model(x.to(C.DEVICE)).argmax(1).cpu()
            for i in (p != y).nonzero().view(-1):
                imgs.append(x[i]); true.append(y[i].item()); pred.append(p[i].item())
            if len(imgs) >= max_show:
                break
    if not imgs:
        return
    cols = 4
    rows = int(np.ceil(min(len(imgs), max_show) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(10, 2.8 * rows))
    axes = np.array(axes).reshape(-1)
    for ax in axes:
        ax.axis("off")
    for ax, im, t, pr in zip(axes, imgs, true, pred):
        ax.imshow(C.denorm(im))
        ax.set_title(f"pred: {names[pr]}\ntrue: {names[t]}", fontsize=8)
    fig.suptitle("Misclassified test images")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def build_training_setup(train_df, lab, n_classes, imbalance, seed):
    """
    Apply the chosen class-imbalance method (see common.py section 7).
    Returns (training rows, sampler or None, loss function).
    """
    sampler = None
    if imbalance == "undersample":
        train_df = C.undersample(train_df, lab, cap_factor=1.0, seed=seed)
    if imbalance == "combo":
        train_df = C.undersample(train_df, lab, cap_factor=2.0, seed=seed)   # mild undersampling first
    if imbalance in ("oversample", "combo"):
        sampler = C.balanced_sampler(train_df[lab].values, n_classes, seed)   # then balanced batches
    weights = C.class_weights(train_df[lab].values, n_classes).to(C.DEVICE)
    if imbalance == "weighted":
        criterion = nn.CrossEntropyLoss(weight=weights)
    elif imbalance == "focal":
        criterion = C.FocalLoss(alpha=weights, gamma=2.0)
    elif imbalance == "combo":
        criterion = C.FocalLoss(alpha=None, gamma=2.0)   # batches are already balanced by the sampler
    else:  # none, oversample, undersample
        criterion = nn.CrossEntropyLoss()
    return train_df, sampler, criterion


def main():
    args = parse_args()
    pretrained = not args.scratch
    name = C.run_name(args.task, args.model, pretrained, args.freeze, args.aug, args.seed,
                      args.train_frac, args.lr, args.bag_source, args.imbalance)
    if args.quick:
        name = "QUICK_" + name
    result_path = C.RESULTS_DIR / f"{name}.json"
    if args.skip_existing and result_path.exists():
        print(f"[skip] {name} already done"); return

    C.seed_everything(args.seed)
    C.make_dirs()
    names = C.class_names(args.task)
    lab = C.label_column(args.task)

    # ---------------- 1. data ----------------
    df = C.load_splits()
    train_df = C.select_rows(df, "train", args.bag_source)
    val_df = C.select_rows(df, "val", args.bag_source)
    test_df = C.select_rows(df, "test", args.bag_source)
    if args.train_frac < 1.0:  # keep the same fraction of every class
        train_df = train_df.groupby("fine_label").sample(frac=args.train_frac, random_state=args.seed)
    if args.quick:
        # keep ~8% of every class, so the class imbalance stays the same as in the full data
        train_df = train_df.groupby("fine_label").sample(frac=0.08, random_state=args.seed)
        val_df, test_df = [d.groupby("fine_label").head(8) for d in (val_df, test_df)]
        args.epochs = 1
    balance_before = C.imbalance_report(train_df, lab, names)
    train_df, sampler, criterion = build_training_setup(train_df, lab, len(names), args.imbalance, args.seed)
    balance_after = C.imbalance_report(train_df, lab, names)
    print(f"Run: {name} | device {C.DEVICE} | train {len(train_df)}, val {len(val_df)}, test {len(test_df)}")
    print(f"Class counts in training data: {balance_after['counts']} "
          f"(imbalance ratio {balance_after['imbalance_ratio']:.2f}, method '{args.imbalance}')")

    train_loader = C.make_loader(train_df, lab, C.build_transform(args.aug), args.batch_size,
                                 shuffle=True, sampler=sampler)
    val_loader = C.make_loader(val_df, lab, C.build_transform("none"), args.batch_size)
    test_loader = C.make_loader(test_df, lab, C.build_transform("none"), args.batch_size)

    # ---------------- 2. model and optimiser ----------------
    model = C.build_model(args.model, len(names), pretrained, args.freeze).to(C.DEVICE)
    n_total, n_train = C.count_params(model)
    print(f"Parameters: total {n_total:,} | trainable {n_train:,}")
    lr = args.lr or (1e-3 if (args.freeze or args.model == "tinycnn" or args.scratch) else 1e-4)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=lr, weight_decay=args.weight_decay)     # AdamW, as in the lab
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)  # lr slowly goes down
    scaler = torch.amp.GradScaler(enabled=C.DEVICE.type == "cuda")

    # ---------------- 3. training loop: keep the model with the best VALIDATION macro-F1 ----------------
    hist = {"tr_loss": [], "tr_acc": [], "va_loss": [], "va_acc": [], "va_f1": [], "epoch_sec": []}
    best_f1, best_state, bad_epochs = -1.0, None, 0
    val_criterion = nn.CrossEntropyLoss()      # plain loss for monitoring, the same for every method
    for ep in range(1, args.epochs + 1):
        with C.Timer() as t:
            tr_loss, tr_acc = C.run_epoch(model, train_loader, criterion, optimizer, scaler)
        scheduler.step()
        va_loss, va_acc = C.run_epoch(model, val_loader, val_criterion)
        y_val, p_val = C.predict(model, val_loader)
        va_f1 = C.compute_metrics(y_val, p_val, names).get("macro_f1", va_acc)
        for k, v in zip(hist, [tr_loss, tr_acc, va_loss, va_acc, va_f1, t.seconds]):
            hist[k].append(v)
        print(f"Epoch {ep:02d} | train loss {tr_loss:.3f} acc {tr_acc*100:.1f}% | "
              f"val loss {va_loss:.3f} acc {va_acc*100:.1f}% F1 {va_f1:.3f} | {t.seconds:.0f}s")
        if va_f1 > best_f1:
            best_f1, bad_epochs = va_f1, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print("Early stopping (no improvement on validation)."); break
    model.load_state_dict(best_state)

    # ---------------- 4. test ONCE ----------------
    y_test, p_test = C.predict(model, test_loader)
    test_metrics = C.compute_metrics(y_test, p_test, names)
    print(f"TEST | acc {test_metrics['accuracy']*100:.2f}% | macro-F1 {test_metrics['macro_f1']:.3f} | "
          f"worst-class recall {test_metrics['worst_class_recall']:.3f} | "
          f"plastic recall {test_metrics['plastic_recall']:.3f} | ECE {test_metrics['ece']:.3f}")

    # ---------------- 5. save everything ----------------
    torch.save({"model": args.model, "task": args.task, "num_classes": len(names),
                "state_dict": best_state, "args": vars(args)}, C.CKPT_DIR / f"{name}.pt")
    C.save_json({
        "run": name, "method": "deep", "task": args.task, "model": args.model,
        "pretrained": pretrained, "freeze": args.freeze, "aug": args.aug, "seed": args.seed,
        "bag_source": args.bag_source, "imbalance": args.imbalance,
        "train_frac": args.train_frac, "lr": lr, "epochs_run": len(hist["tr_loss"]),
        "params_total": n_total, "params_trainable": n_train,
        "mean_epoch_sec": float(np.mean(hist["epoch_sec"])),
        "class_balance_before": balance_before, "class_balance_after": balance_after,
        "history": hist, "best_val_macro_f1": best_f1,
        "test": test_metrics,
        "test_predictions": {"path": test_df["path"].tolist(), "y_true": y_test.tolist(),
                             "y_pred": p_test.argmax(1).tolist()},
    }, result_path)
    C.plot_history(hist, C.FIG_DIR / "runs" / f"{name}_curves.png", name)
    C.plot_confusion(test_metrics["confusion_matrix"], names,
                     C.FIG_DIR / "runs" / f"{name}_confusion.png", f"Test confusion matrix\n{name}")
    show_mistakes(model, test_loader, names, C.FIG_DIR / "runs" / f"{name}_mistakes.png")
    if args.task == "binary":
        C.plot_roc(y_test, p_test[:, C.PLASTIC_ID], C.FIG_DIR / "runs" / f"{name}_roc.png", f"Test ROC\n{name}")
    print(f"Saved results to {result_path}")


if __name__ == "__main__":
    main()
