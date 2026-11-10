"""
common.py
=========
Shared code for the COMP813 project
"Deep Learning Approach to Classify Underwater Plastic Bags and Medusozoa".

Every other script imports this file, so all settings and helper functions
live in ONE place. If you want to change a setting (image size, split sizes,
seed, ...), change it here and every script will use the new value.

What is inside (in order):
   1. Paths and settings
   2. Reproducibility (random seeds)
   3. Class names (what the models must predict)
   4. Underwater image effects (used for augmentation and robustness tests)
   5. Image transforms (resizing, augmentation, normalisation)
   6. Reading the data table and choosing rows (train / val / test, bag source)
   7. Class-imbalance tools (class weights, oversampling, undersampling, focal loss)
   8. Models (TinyCNN from the lab + pretrained torchvision models)
   9. Training and evaluation loops (same idea as run_epoch / evaluate in the labs)
  10. Metrics (accuracy, macro-F1, AUC, calibration, plastic recall, worst-class recall)
  11. Plot and file helpers
"""

import os
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageFilter, ImageEnhance

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models

import matplotlib
matplotlib.use("Agg")  # save figures to files; no screen needed (works on Colab and servers)
import matplotlib.pyplot as plt

from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay,
                             RocCurveDisplay, classification_report)

# =============================================================================
# 1. PATHS AND SETTINGS
# =============================================================================
# The code folder sits inside the "Task" folder, next to the raw data folders:
#   Task/archive_jelly/        jellyfish images (Kaggle)
#   Task/archive_Plastic/      synthetic bag images (Kaggle)
#   Task/archive_trashcan/     TrashCan 1.0 underwater trash dataset (Kaggle copy of UMN data)
#   Task/COMP813_Project_Code/ this code
# On Colab the data is unzipped somewhere else, so the location can be changed
# with the environment variables DATA_ROOT (raw data) and OUT_DIR (results).
CODE_DIR = Path(__file__).resolve().parent
DATA_ROOT = Path(os.environ.get("DATA_ROOT", CODE_DIR.parent))
OUT_DIR = Path(os.environ.get("OUT_DIR", CODE_DIR / "outputs"))

SPLITS_CSV = OUT_DIR / "splits.csv"      # table of every image + its label + its split (made by prepare_data.py)
RESULTS_DIR = OUT_DIR / "results"        # one JSON file per experiment run
CKPT_DIR = OUT_DIR / "checkpoints"       # saved model weights (large; never submitted)
FIG_DIR = OUT_DIR / "figures"            # figures for the report
CROP_DIR = OUT_DIR / "crops"             # cut-out TrashCan objects and background-swap images

SEED = 813                  # the same seed as in the COMP813 labs
IMG_SIZE = 224              # every image is resized to 224 x 224 pixels for the deep models
MEAN = (0.485, 0.456, 0.406)  # ImageNet colour mean / std (pretrained models expect these)
STD = (0.229, 0.224, 0.225)

SPLIT_FRACTIONS = (0.60, 0.20, 0.20)    # train / validation / test (as in the Linear Regression and Spam labs)
DUP_HASH_BITS = 12          # jellyfish: images whose 256-bit hashes differ by <= 12 bits are "the same photo"

# --- TrashCan cleaning settings (see prepare_data.py for the reasons) ---
TC_BAG_CLASS = "trash_bag"
# The video frames have telemetry text at the top and bottom and a "(c) JAMSTEC" watermark.
# We only keep the part of the frame between these relative heights (frame height -> (top, bottom)).
TC_CLEAN_BAND = {270: (0.20, 0.80), 360: (0.28, 0.73)}
TC_MIN_SIDE = 32            # crops smaller than 32 x 32 pixels are too blurry to use
TC_MIN_KEEP = 0.60          # a box must keep >= 60% of its area after removing the text bands
TC_MARGIN = 0.10            # add 10% context around each box when cutting it out
TC_DUP_BITS = 10            # crops in the same video whose hashes differ by <= 10 bits are near-copies
TC_OOD_CLASSES = ["trash_bottle", "trash_can", "trash_container", "animal_fish", "animal_starfish"]
TC_OOD_PER_CLASS = 60       # objects the model never saw in training (out-of-distribution test)

N_SYNTH_PER_FOLDER = 225    # synthetic bags: 225 "plastic" + 225 "garbage" = 450 (similar to TrashCan's 441)

NUM_WORKERS = 0 if os.name == "nt" else 2   # Windows is safer with 0 data-loading workers
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BAG_SOURCES = ["trashcan", "synthetic", "both"]      # which plastic-bag images to TRAIN on
DEFAULT_BAG_SOURCE = "trashcan"                       # main experiments use real underwater bags
IMBALANCE_METHODS = ["none", "weighted", "oversample", "undersample", "focal", "combo"]
DEFAULT_IMBALANCE = "weighted"


def data_paths(root=None):
    """Return the raw data folders inside the Task folder."""
    root = Path(root) if root else DATA_ROOT
    # The TrashCan zip from Kaggle unpacks into "archive (1)/dataset/instance_version",
    # so we search for the "instance_version" folder instead of typing the full path.
    found = sorted((root / "archive_trashcan").glob("**/instance_version"))
    return {
        "jelly": root / "archive_jelly" / "Train_Test_Valid",
        "synthetic_bags": root / "archive_Plastic" / "Bag Classes" / "Bag Classes",
        "trashcan": found[0] if found else root / "archive_trashcan" / "instance_version",
    }


def make_dirs():
    for d in [OUT_DIR, RESULTS_DIR, CKPT_DIR, FIG_DIR, CROP_DIR, FIG_DIR / "runs"]:
        d.mkdir(parents=True, exist_ok=True)


# =============================================================================
# 2. REPRODUCIBILITY
# =============================================================================
def seed_everything(seed=SEED):
    """Fix every random number generator so the same run gives the same result (as in the labs)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =============================================================================
# 3. CLASS NAMES
# =============================================================================
# Folder name in archive_jelly -> clean class name
JELLY_FOLDERS = {
    "barrel_jellyfish": "barrel_jelly",
    "blue_jellyfish": "blue_jelly",
    "compass_jellyfish": "compass_jelly",
    "lions_mane_jellyfish": "lions_mane_jelly",
    "mauve_stinger_jellyfish": "mauve_stinger_jelly",
    "Moon_jellyfish": "moon_jelly",
}
# Two tasks use the same images:
#   "binary": plastic bag vs Medusozoa (any jellyfish)                      -> 2 classes
#   "fine"  : plastic bag + the 6 jellyfish species                          -> 7 classes
FINE_CLASSES = ["plastic_bag"] + sorted(JELLY_FOLDERS.values())
BINARY_CLASSES = ["plastic_bag", "medusozoa"]
PLASTIC_ID = 0  # "plastic_bag" is class number 0 in both tasks


def class_names(task):
    return FINE_CLASSES if task == "fine" else BINARY_CLASSES


def label_column(task):
    """Name of the column in splits.csv that holds the class number for this task."""
    return "fine_id" if task == "fine" else "binary_id"


# =============================================================================
# 4. UNDERWATER IMAGE EFFECTS
# =============================================================================
# A simple physics-based model of how water changes an image:
#     I = J * t + B * (1 - t),    with   t = exp(-beta * d)   for each colour channel
#   J = the clean image, B = the colour of the water ("veiling light"),
#   t = how much light survives after travelling a distance d through water.
# Red light is absorbed fastest, so pictures become blue-green, hazy and blurry.
# (Simplified from Akkaynak & Treibitz, CVPR 2018.)
UW_BETA = np.array([0.55, 0.18, 0.12], dtype=np.float32)   # absorption of Red, Green, Blue
UW_LIGHT = np.array([0.05, 0.40, 0.55], dtype=np.float32)  # blue-green water colour


def underwater_effect(img, severity):
    """Make a PIL image look further underwater. severity 0 = unchanged, 5 = very strong."""
    if severity <= 0:
        return img
    x = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    t = np.exp(-UW_BETA * 0.6 * severity)          # light that survives, per colour channel
    out = x * t + UW_LIGHT * (1.0 - t)             # mix the image with the water colour
    out = Image.fromarray((np.clip(out, 0, 1) * 255).astype(np.uint8))
    return out.filter(ImageFilter.GaussianBlur(radius=0.3 * severity))  # water scatters light -> blur


class FixedUnderwater:
    """Test-time effect: always apply the same underwater severity."""

    def __init__(self, severity):
        self.severity = severity

    def __call__(self, img):
        return underwater_effect(img, self.severity)


class FixedBrightness:
    """Test-time effect: make the whole image darker (<1) or brighter (>1)."""

    def __init__(self, factor):
        self.factor = factor

    def __call__(self, img):
        return ImageEnhance.Brightness(img).enhance(self.factor)


class DomainAugment:
    """
    Our proposed training augmentation ("underwater-domain augmentation").
    The two data sources look different: TrashCan bags are dark, green-grey, low-resolution
    video frames; jellyfish photos are bright, blue and sharp. A model could learn
    "dark and blurry = bag" instead of learning the object. So, with probability p,
    we give ANY training image (bag or jellyfish) random:
      - underwater colour cast and haze,
      - low resolution (shrink, then enlarge again -> like a video frame),
      - brightness change (darker or brighter).
    Then these differences no longer tell the classes apart.
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            img = underwater_effect(img, random.uniform(0.5, 4.0))
        if random.random() < self.p:
            small = random.randint(40, 112)
            img = img.resize((small, small), Image.BILINEAR).resize(img.size, Image.BILINEAR)
        if random.random() < self.p:
            img = ImageEnhance.Brightness(img).enhance(random.uniform(0.45, 1.5))
        return img


def plain_background(h, w, color):
    return np.ones((h, w, 3)) * np.array(color) / 255.0


def bright_blue_background(h, w, rng):
    """A bright blue 'aquarium-like' background, similar to the jellyfish photos (used in the swap test)."""
    top = np.array([0.20, 0.55, 0.95]) + rng.uniform(-0.05, 0.05, 3)
    bottom = np.array([0.02, 0.15, 0.55]) + rng.uniform(-0.03, 0.03, 3)
    y = np.linspace(0, 1, h)[:, None, None]
    bg = np.repeat(top * (1 - y) + bottom * y, w, axis=1)
    small = rng.normal(0, 1, (h // 20 + 1, w // 20 + 1, 3))
    small = (small - small.min()) / (np.ptp(small) + 1e-8)
    noise = np.asarray(Image.fromarray((small * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR)) / 255.0
    return np.clip(bg + 0.05 * (noise - 0.5), 0, 1)


# =============================================================================
# 5. IMAGE TRANSFORMS
# =============================================================================
def build_transform(aug="none"):
    """
    aug = "none"       -> only resize (used for validation and test images)
    aug = "standard"   -> random crop, flip, rotation, colour jitter (common practice)
    aug = "underwater" -> DomainAugment + standard  (our proposed fix, see DomainAugment)
    Every option ends with ToTensor (0-255 -> 0-1) and Normalize (ImageNet mean/std).
    """
    to_tensor = [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
    if aug == "none":
        return transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE))] + to_tensor)
    steps = [
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    ]
    if aug == "underwater":
        steps = [DomainAugment(p=0.5)] + steps
    return transforms.Compose(steps + to_tensor)


def denorm(x):
    """Undo Normalize so a tensor (3,H,W) can be shown as a picture (H,W,3)."""
    m = torch.tensor(MEAN).view(3, 1, 1)
    s = torch.tensor(STD).view(3, 1, 1)
    return (x.detach().cpu() * s + m).clamp(0, 1).permute(1, 2, 0).numpy()


# =============================================================================
# 6. READING THE DATA TABLE AND CHOOSING ROWS
# =============================================================================
def load_splits():
    """Read the table made by prepare_data.py (one row per image)."""
    if not SPLITS_CSV.exists():
        raise FileNotFoundError(f"{SPLITS_CSV} not found. Run prepare_data.py first.")
    return pd.read_csv(SPLITS_CSV)


def select_rows(df, split, bag_source=DEFAULT_BAG_SOURCE):
    """
    All jellyfish rows of one split + the plastic-bag rows of that split from the chosen source.
    bag_source = "trashcan" (real underwater bags), "synthetic" (Kaggle bags on land) or "both".
    """
    sources = ["trashcan", "synthetic"] if bag_source == "both" else [bag_source]
    keep = (df.split == split) & ((df.source == "jelly") | df.source.isin(sources))
    return df[keep]


def open_rgb(path, pre_transform=None):
    img = Image.open(path).convert("RGB")
    return pre_transform(img) if pre_transform else img


class ImageTableDataset(Dataset):
    """A PyTorch dataset that reads the images listed in a table (column 'path') with their labels."""

    def __init__(self, df, label_col, transform, pre_transform=None):
        self.paths = df["path"].tolist()
        self.labels = df[label_col].astype(int).tolist()
        self.transform = transform
        self.pre_transform = pre_transform  # e.g. FixedUnderwater(3) for a robustness test

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = open_rgb(self.paths[i], self.pre_transform)
        return self.transform(img), self.labels[i]


def make_loader(df, label_col, transform, batch_size=32, shuffle=False, pre_transform=None, sampler=None):
    ds = ImageTableDataset(df, label_col, transform, pre_transform)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle and sampler is None, sampler=sampler,
                      num_workers=NUM_WORKERS, pin_memory=DEVICE.type == "cuda")


def evaluation_sets(df, bag_source=DEFAULT_BAG_SOURCE):
    """
    The SAME list of test sets is used for every model, so results can be compared.
    Returns {name: (rows, pre_transform)}.
      test          test split (jellyfish + bags from the training source)
      test_tc       test jellyfish + REAL underwater bags (TrashCan)        -> cross-dataset test
      test_syn      test jellyfish + SYNTHETIC bags on land (Kaggle)        -> cross-dataset test
      test_uw1/3/5  test split made to look further underwater (light / medium / strong)
      sup_orig      TrashCan bag on its own deep-sea background   \
      sup_gray      the same bag on plain grey                     >  background-swap test
      sup_blue      the same bag on bright blue (like jelly photos)/
      ood_other     other TrashCan objects never seen in training (bottles, cans, fish ...)
    """
    test = select_rows(df, "test", bag_source)
    sets = {"test": (test, None),
            "test_tc": (select_rows(df, "test", "trashcan"), None),
            "test_syn": (select_rows(df, "test", "synthetic"), None)}
    for s in [1, 3, 5]:
        sets[f"test_uw{s}"] = (test, FixedUnderwater(s))
    for name in ["sup_orig", "sup_gray", "sup_blue", "ood_other"]:
        rows = df[df.split == name]
        if len(rows):
            sets[name] = (rows, None)
    return sets


# =============================================================================
# 7. CLASS-IMBALANCE TOOLS
# =============================================================================
# A dataset is "imbalanced" when some classes have many more images than others.
# The model then learns to favour the big class. We compare six ways to handle this
# (option --imbalance in train.py):
#   none        plain cross-entropy, no correction (the reference)
#   weighted    cross-entropy where rare classes count more (weight = 1 / class size)
#   oversample  show rare-class images more often (WeightedRandomSampler -> balanced batches)
#   undersample randomly drop images of big classes until every class is as small as the smallest
#   focal       focal loss (Lin et al., 2017): focuses on hard, badly-classified images, + class weights
#   combo       a COMBINATION: mild undersampling (big classes capped at 2x the smallest class)
#               + oversampling (balanced batches) + focal loss
def class_counts(labels, n_classes):
    return np.bincount(np.asarray(labels, dtype=int), minlength=n_classes)


def class_weights(labels, n_classes):
    """Inverse-frequency weights: weight_c = N / (n_classes * count_c). A class half as big counts twice as much."""
    counts = class_counts(labels, n_classes).astype(float)
    return torch.tensor(counts.sum() / (n_classes * np.maximum(counts, 1)), dtype=torch.float32)


def balanced_sampler(labels, n_classes, seed=SEED):
    """Random oversampling: each image is drawn with probability 1 / (size of its class), so batches are balanced."""
    labels = np.asarray(labels, dtype=int)
    per_image = 1.0 / class_counts(labels, n_classes)[labels]
    gen = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(torch.tensor(per_image, dtype=torch.double), num_samples=len(labels),
                                 replacement=True, generator=gen)


def undersample(df, label_col, cap_factor=1.0, seed=SEED):
    """
    Random undersampling: every class keeps at most cap_factor x (size of the smallest class) images.
    Original photos are kept before the dataset's own 'aug-*' copies, so we throw away copies first.
    """
    counts = df[label_col].value_counts()
    cap = int(round(cap_factor * counts.min()))
    rng = np.random.default_rng(seed)
    parts = []
    for c, g in df.groupby(label_col):
        g = g.assign(_r=rng.random(len(g))).sort_values(["is_aug", "_r"])  # originals first, then random order
        parts.append(g.head(cap).drop(columns="_r"))
    return pd.concat(parts)


class FocalLoss(nn.Module):
    """
    Focal loss = cross-entropy * (1 - p_true)^gamma.
    Easy images (p_true close to 1) contribute almost nothing, so training focuses on hard ones.
    alpha = optional class weights (tensor), gamma = focusing strength (2 is the usual value).
    """

    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha, self.gamma = alpha, gamma

    def forward(self, logits, target):
        ce = F.cross_entropy(logits, target, weight=self.alpha, reduction="none")
        p_true = torch.exp(-F.cross_entropy(logits, target, reduction="none"))
        return ((1 - p_true) ** self.gamma * ce).mean()


def imbalance_report(df, label_col, names):
    """Class counts + imbalance ratio (largest class / smallest class). Ratio 1 = perfectly balanced."""
    counts = class_counts(df[label_col], len(names))
    return {"counts": dict(zip(names, counts.tolist())),
            "imbalance_ratio": float(counts.max() / max(counts.min(), 1)),
            "majority_share": float(counts.max() / counts.sum())}


# =============================================================================
# 8. MODELS
# =============================================================================
class TinyCNN(nn.Module):
    """Small CNN trained from scratch: the lab TinyCNN with BatchNorm and one more block for 224 px images."""

    def __init__(self, num_classes):
        super().__init__()

        def block(c_in, c_out):  # two 3x3 convolutions, then halve the image size
            return nn.Sequential(
                nn.Conv2d(c_in, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU(),
                nn.Conv2d(c_out, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU(),
                nn.MaxPool2d(2))

        self.features = nn.Sequential(block(3, 32), block(32, 64), block(64, 128), block(128, 256))
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(self.features(x)).flatten(1)
        return self.fc(self.dropout(x))


MODEL_CHOICES = ["tinycnn", "resnet18", "resnet50", "efficientnet_b0", "vgg16",
                 "mobilenet_v3_large", "convnext_tiny", "vit_b_16"]


def build_model(name, num_classes, pretrained=True, freeze=False):
    """
    Build a model and replace its last layer with a new one for our classes.
      pretrained=True -> start from ImageNet weights (transfer learning)
      freeze=True     -> train only the new last layer (a "linear probe")
    """
    if name == "tinycnn":
        return TinyCNN(num_classes)
    weights = "DEFAULT" if pretrained else None
    m = getattr(models, name)(weights=weights)
    if name.startswith("resnet"):
        m.fc = nn.Linear(m.fc.in_features, num_classes); head = m.fc
    elif name == "efficientnet_b0":
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes); head = m.classifier[1]
    elif name == "vgg16":
        m.classifier[6] = nn.Linear(m.classifier[6].in_features, num_classes); head = m.classifier[6]
    elif name == "mobilenet_v3_large":
        m.classifier[3] = nn.Linear(m.classifier[3].in_features, num_classes); head = m.classifier[3]
    elif name == "convnext_tiny":
        m.classifier[2] = nn.Linear(m.classifier[2].in_features, num_classes); head = m.classifier[2]
    elif name == "vit_b_16":
        m.heads.head = nn.Linear(m.heads.head.in_features, num_classes); head = m.heads.head
    else:
        raise ValueError(f"Unknown model {name}")
    if freeze:
        for p in m.parameters():
            p.requires_grad = False
        for p in head.parameters():
            p.requires_grad = True
    return m


def gradcam_layer(model, name):
    """The layer Grad-CAM looks at: the last convolution block (or the last Transformer block for ViT)."""
    if name == "tinycnn":
        return model.features[-1]
    if name.startswith("resnet"):
        return model.layer4[-1]
    if name == "vgg16":
        return model.features[-2]            # ReLU after the last convolution
    if name in ("efficientnet_b0", "mobilenet_v3_large", "convnext_tiny"):
        return model.features[-1]
    if name == "vit_b_16":
        # ln_1 of the last block: torchvision's ViT passes only the class token of the LAST block's output to the
        # head, so the patch tokens of encoder.layers[-1] get zero gradient and Grad-CAM would be empty.
        # The input of the last self-attention still feeds the class token, so its patch tokens get gradients.
        return model.encoder.layers[-1].ln_1
    raise ValueError(name)


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# =============================================================================
# 9. TRAINING AND EVALUATION LOOPS (same idea as run_epoch / evaluate in the labs)
# =============================================================================
def run_epoch(model, loader, criterion, optimizer=None, scaler=None):
    """One pass over the data. Trains when an optimizer is given, otherwise only measures the loss."""
    training = optimizer is not None
    model.train() if training else model.eval()
    use_amp = DEVICE.type == "cuda"      # mixed precision = faster on a GPU
    total, correct, loss_sum = 0, 0, 0.0
    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            with torch.autocast(device_type=DEVICE.type, enabled=use_amp):
                logits = model(x)
                loss = criterion(logits, y)
            if training:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            loss_sum += loss.item() * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += x.size(0)
    return loss_sum / total, correct / total


@torch.no_grad()
def predict(model, loader):
    """Return the true labels and the predicted probabilities for every image in the loader."""
    model.eval()
    ys, probs = [], []
    for x, y in loader:
        logits = model(x.to(DEVICE)).float()
        probs.append(F.softmax(logits, dim=1).cpu().numpy())
        ys.append(y.numpy())
    return np.concatenate(ys), np.concatenate(probs)


# =============================================================================
# 10. METRICS
# =============================================================================
def expected_calibration_error(y_true, probs, n_bins=15):
    """ECE: the average gap between how confident the model is and how often it is right (0 = perfect)."""
    conf = probs.max(1)
    correct = (probs.argmax(1) == y_true).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf > lo) & (conf <= hi)
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
    return float(ece)


def compute_metrics(y_true, probs, names):
    """Every number we report for one model on one test set."""
    y_true = np.asarray(y_true)
    y_pred = probs.argmax(1)
    labels = list(range(len(names)))
    is_plastic = y_true == PLASTIC_ID
    out = {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "plastic_recall": float((y_pred[is_plastic] == PLASTIC_ID).mean()) if is_plastic.any() else None,
        "mean_confidence": float(probs.max(1).mean()),
    }
    if (y_true >= 0).all():  # real class labels (the out-of-distribution set has label -1)
        out.update({
            "ece": expected_calibration_error(y_true, probs),
            "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
            "report": classification_report(y_true, y_pred, labels=labels, target_names=names,
                                            output_dict=True, zero_division=0),
        })
    if (y_true >= 0).all() and len(np.unique(y_true)) > 1:  # these need more than one true class
        per_class_recall = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
        present = np.isin(labels, np.unique(y_true))
        out["macro_f1"] = float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))
        out["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred))
        out["worst_class_recall"] = float(per_class_recall[present].min())   # key number for imbalance
        out["plastic_precision"] = float(precision_score(is_plastic, y_pred == PLASTIC_ID, zero_division=0))
        try:
            if len(names) == 2:
                out["auc"] = float(roc_auc_score(is_plastic, probs[:, PLASTIC_ID]))
            else:
                out["auc"] = float(roc_auc_score(y_true, probs, multi_class="ovr", average="macro", labels=labels))
        except ValueError:  # e.g. a class missing from a tiny test set
            out["auc"] = None
    return out


def ood_auroc(id_probs, ood_probs):
    """How well the top softmax confidence separates known images (1) from unknown objects (0)."""
    scores = np.concatenate([id_probs.max(1), ood_probs.max(1)])
    is_id = np.concatenate([np.ones(len(id_probs)), np.zeros(len(ood_probs))])
    return float(roc_auc_score(is_id, scores))


def softmax_np(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


# =============================================================================
# 11. PLOT AND FILE HELPERS
# =============================================================================
def plot_confusion(cm, names, path, title):
    fig, ax = plt.subplots(figsize=(1.1 * len(names) + 3, 1.1 * len(names) + 2))
    ConfusionMatrixDisplay(np.array(cm), display_labels=names).plot(
        ax=ax, cmap="Blues", colorbar=False, xticks_rotation=45)
    ax.set_title(title, fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_roc(y_true, p_plastic, path, title):
    """ROC curve for the binary task with plastic bag as the positive class (as in the Logistic Regression lab)."""
    fig, ax = plt.subplots(figsize=(4.5, 4))
    RocCurveDisplay.from_predictions(np.asarray(y_true) == PLASTIC_ID, p_plastic, ax=ax, name="plastic bag")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_title(title, fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_history(hist, path, title):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    axes[0].plot(hist["tr_loss"], label="train"); axes[0].plot(hist["va_loss"], label="validation")
    axes[0].set_title("Loss"); axes[0].set_xlabel("epoch"); axes[0].legend()
    axes[1].plot(hist["tr_acc"], label="train acc"); axes[1].plot(hist["va_acc"], label="val acc")
    axes[1].plot(hist["va_f1"], label="val macro-F1", linestyle="--")
    axes[1].set_title("Accuracy / F1"); axes[1].set_xlabel("epoch"); axes[1].legend()
    fig.suptitle(title, fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def run_name(task, model, pretrained, freeze, aug, seed, train_frac=1.0, lr=None,
             bag_source=DEFAULT_BAG_SOURCE, imbalance=DEFAULT_IMBALANCE):
    """A unique, readable name for one experiment run, e.g. fine_resnet50_pt_ft_aug-standard_s813."""
    name = f"{task}_{model}_{'pt' if pretrained else 'scratch'}_{'frozen' if freeze else 'ft'}_aug-{aug}"
    if bag_source != DEFAULT_BAG_SOURCE:
        name += f"_bags-{bag_source}"
    if imbalance != DEFAULT_IMBALANCE:
        name += f"_imb-{imbalance}"
    if train_frac < 1.0:
        name += f"_frac{train_frac:g}"
    if lr is not None:  # only runs of the learning-rate search carry the lr in their name
        name += f"_lr{lr:g}"
    return f"{name}_s{seed}"


def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1)


class Timer:
    """with Timer() as t: ...   ->  t.seconds is how long the block took."""

    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, *a):
        self.seconds = time.time() - self.t0
