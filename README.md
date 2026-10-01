# Deep Learning Approach to Classify Underwater Plastic Bags and Medusozoa
COMP813 Artificial Intelligence – Final Project (group of 3, Advanced level)

We compare classic features, CNNs, a Vision Transformer and CLIP for telling **plastic bags**
from **jellyfish (Medusozoa)** in underwater images, on two tasks:

* **binary** – plastic bag vs. medusozoa
* **fine** – plastic bag + 6 jellyfish species (7 classes)

The plastic bags come from **TrashCan 1.0**: real deep-sea robot video (cut out of the frames).
The synthetic Kaggle bags (bags pasted on land photos) are kept for a **cross-dataset study**.
We also study **class imbalance** (6 methods), **robustness** (underwater, brightness, noise,
adversarial attacks, background swap, unseen objects) and **explainability** (Grad-CAM).

`PROCESS.md` explains every step in simple English, from downloading the data to the final tables,
so the whole project can be reproduced.

## 1. Folder layout

```
Task/
├── archive_jelly/          Jellyfish Image Dataset (Kaggle)                     – not in the code ZIP
├── archive_Plastic/        Plastic / Paper / Garbage Bag Synthetic Images (Kaggle) – not in the code ZIP
├── archive_trashcan/       TrashCan 1.0 (Kaggle copy of the UMN dataset)        – not in the code ZIP
└── COMP813_Project_Code/
    ├── common.py          shared settings, data selection, imbalance tools, models, metrics
    ├── prepare_data.py    STEP 1  cleaning, TrashCan bag crops, leak-free split, test sets, class balance
    ├── baseline_hog.py    STEP 2  majority, rule, pixels, HOG, colour (+ SMOTE / undersampling)
    ├── train.py           STEP 3  one deep model (options: model, aug, bag source, imbalance method, ...)
    ├── clip_eval.py       STEP 4  CLIP zero-shot + linear probe
    ├── robustness.py      STEP 5  cross-dataset, underwater, brightness, noise, FGSM, background swap, unseen objects
    ├── gradcam.py         STEP 6  Grad-CAM heat-maps + "does the model look at the bag?" numbers
    ├── summarize.py       STEP 7  tables (Markdown + LaTeX), figures, McNemar tests
    ├── run_all.py         runs every stage in order (resumable)
    ├── COMP813_Project_Colab.ipynb   the same steps as Colab cells
    ├── PROCESS.md         the full, reproducible process in simple English
    └── requirements.txt
```

The scripts read the **raw** downloads. If `COMP813_Project_Code` sits inside `Task/`, nothing needs
to be set. Otherwise set the environment variable `DATA_ROOT=/path/to/Task` (and optionally `OUT_DIR`).

## 2. Install

Python 3.10+, then `pip install -r requirements.txt`. A GPU is strongly recommended (a free Colab T4 works).
Pretrained weights (torchvision, Hugging Face CLIP) are downloaded automatically the first time.

## 3. Run

```
python run_all.py --quick      # smoke test: few images, 1 epoch
python run_all.py              # everything, 3 seeds (several GPU hours – split the stages, see below)
```

Stages can be run separately (for example by different group members):

```
python run_all.py --stages data classic clip lrsweep
python run_all.py --stages main underwater --seeds 814
python run_all.py --stages imbalance crossdata ablation datasize
python run_all.py --stages gradcam summary
```

Single experiments:

```
python train.py --task fine --model resnet50 --imbalance combo --seed 813
python train.py --task binary --model vit_b_16 --bag_source synthetic
python robustness.py --runs fine_resnet50_pt_ft_aug-standard_imb-combo_s813
```

Finished runs are skipped, so a command can be restarted after a disconnect.

## 4. Outputs (in `outputs/`)

| Path | Content |
|---|---|
| `splits.csv` | every image: path, source, label, duplicate group / video, split |
| `data_summary.json` | cleaning numbers, leakage found, class balance per split |
| `crops/` | TrashCan bag crops, background-swap images, unseen-object crops |
| `figures/` | data figures, per-run curves / confusion matrices / mistakes / ROC, Grad-CAM |
| `results/*.json` | one file per run: settings, class balance, history, test metrics, predictions |
| `robustness/*.csv` | robustness results per run |
| `summary/` | **report tables and figures** + `key_numbers.md` |
| `checkpoints/` | model weights (seed 813 only). Do **not** submit these |

## 5. Method summary

* **Cleaning.** Jellyfish: `aug-*` copies and near-duplicates (256-bit average hash, Hamming ≤ 12) are grouped;
  the provided split leaks copies into its test set, so we re-split by group. TrashCan: `trash_bag` boxes are
  cut out of the frames after removing the telemetry-text / watermark bands, crops < 32 px are dropped and
  near-identical crops from consecutive frames of the same video are removed; each video is one group
  (the official TrashCan train/val split shares 113 of 115 videos).
* **Split.** 60 / 20 / 20 by group, per class; val/test keep one photo per jellyfish group.
* **Class imbalance.** Measured per split (fine task: ≈ 2.7 : 1 in training, ≈ 8 : 1 counting unique photos).
  Compared: none, class-weighted loss, oversampling, undersampling, focal loss, and a combination
  (mild undersampling + oversampling + focal loss); SMOTE / undersampling / class weights for the classic model.
* **Models.** Majority; "bluer = jellyfish" rule; pixels, HOG, colour + LinearSVC / logistic regression;
  TinyCNN (scratch); ResNet-50, EfficientNet-B0, ViT-B/16, VGG16 (ImageNet, fine-tuned); CLIP zero-shot and linear probe.
* **Training.** AdamW, cosine schedule, early stopping on validation macro-F1, seeds 813/814/815,
  learning rate from a validation search (3e-5 / 1e-4 / 3e-4).
* **Underwater-domain augmentation** (our proposed fix): random colour cast / haze (I = J·t + B·(1−t)),
  low resolution and brightness changes applied to both classes.
* **Evaluation.** Accuracy, macro-F1, balanced accuracy, worst-class recall, AUC, ECE, plastic recall;
  McNemar tests; cross-dataset, underwater, brightness, noise, FGSM, background-swap and unseen-object tests; Grad-CAM.

## 6. Datasets (cite in the report)

* Jellyfish Image Dataset – Kaggle (anshtanwar/jellyfish-types).
* Plastic – Paper – Garbage Bag Synthetic Images – Kaggle (vencerlanz09/plastic-paper-garbage-bag-synthetic-images).
* TrashCan 1.0 – J. Hong, M. Fulton, J. Sattar, "TrashCan: A Semantically-Segmented Dataset towards Visual
  Detection of Marine Debris", arXiv:2007.08097, 2020 (Kaggle copy: mexwell/trashcan-1-0). Images © JAMSTEC J-EDI.
