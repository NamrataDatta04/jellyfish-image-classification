# How this project was built – the full process in simple English

This file explains **everything that was done**, in order, so anyone can repeat it and get the same data,
the same splits and the same kind of results. Every step below is done **by the code**, not by hand, except
downloading the three datasets (Step 1). The numbers in this file come from running the code on
30 September 2026 with seed 813.

---

## Step 0 – The goal

Sea turtles eat plastic bags because the bags look like jellyfish. An underwater camera that can tell
**plastic bags** from **jellyfish (Medusozoa)** could help count and remove marine plastic without harming
jellyfish. We solve two image-classification tasks:

| Task | Classes |
|---|---|
| binary | plastic bag vs. medusozoa (any jellyfish) |
| fine | plastic bag + 6 jellyfish species (barrel, blue, compass, lion's mane, mauve stinger, moon) |

---

## Step 1 – Collecting the datasets (the only manual step)

Download these three Kaggle datasets (a free Kaggle account is needed) and unzip each into the `Task` folder
with exactly these folder names:

| Folder in `Task/` | Dataset | Link | What it contains |
|---|---|---|---|
| `archive_jelly` | Jellyfish Image Dataset | kaggle.com/datasets/anshtanwar/jellyfish-types | 6 species, 224×224 photos, mostly aquarium / clear water |
| `archive_trashcan` | TrashCan 1.0 (copy of the University of Minnesota dataset) | kaggle.com/datasets/mexwell/trashcan-1-0 | 7,212 frames from JAMSTEC deep-sea robot videos, 22 object classes with boxes and outlines (COCO format) |
| `archive_Plastic` | Plastic – Paper – Garbage Bag Synthetic Images | kaggle.com/datasets/vencerlanz09/plastic-paper-garbage-bag-synthetic-images | 5,000 each of plastic, garbage and paper bags pasted on land photos |

After unzipping, the folders look like this (the code finds the rest by itself):

```
Task/
  archive_jelly/Train_Test_Valid/{Train,valid,test}/<species>/*.jpg     (+ 6 species folders at the top)
  archive_trashcan/archive (1)/dataset/instance_version/{train,val}/*.jpg + instances_{train,val}_trashcan.json
  archive_Plastic/Bag Classes/Bag Classes/{Plastic,Garbage,Paper} Bag Images/*.jpg
  COMP813_Project_Code/        (this code)
```

For Google Colab we packed the raw data into two zip files that the notebook unzips:
`COMP813_data.zip` (archive_jelly + archive_Plastic, ≈ 520 MB) and `COMP813_trashcan.zip`
(only the `instance_version` folder of TrashCan, ≈ 200 MB).

---

## Step 2 – Looking at the raw data (what we found)

Before cleaning anything we checked the data. These findings shaped the whole project.

**Jellyfish dataset**

* The 6 species folders at the top of `archive_jelly` are **exact copies** of `Train_Test_Valid/Train`.
* 581 of the 900 training images are `aug-*` files = flipped/rotated **copies** of other photos.
* 163 pairs of near-identical photos (58 exact copies); 7 of these pairs even have **different species labels**.
* 8 of the 79 images in the provided valid/test folders also appear in the provided training folder → the
  provided split **leaks** test images into training.
* Only **369 unique photos** in total (about 60 per species).

**Synthetic bag dataset**

* The bags are cut out and pasted onto **land** photos (forest, walls, floors), never underwater.
* In a first version of the project, all jellyfish were underwater and all bags on land. A model using
  **only the colour histogram** already reached **91 %** accuracy on bag vs. jellyfish, but called only
  **0.3 %** of the same bags "plastic" once they were pasted on an underwater background. The models were
  learning "blue water = jellyfish", not what a bag looks like (a *shortcut*). This is why we moved to
  TrashCan for the main experiments and kept the synthetic bags only for a comparison.

**TrashCan dataset**

* Every image is a **video frame** (480×270 or 480×360) from a deep-sea robot (ROV), with **telemetry text**
  at the top and bottom, a **"© JAMSTEC" watermark**, and black bars in the 480×360 frames.
* There are 908 `trash_bag` objects (727 in the official train file, 181 in val). Many boxes are small
  (median about 68×54 px) and many come from **consecutive frames of the same video** (almost identical).
* The official train and val files share **113 of their 115 validation videos** → the official split leaks too,
  so we do not use it.
* The background is dark, green-grey sea floor with the robot's arm often visible.

---

## Step 3 – Cleaning and preparing the data (`prepare_data.py`)

Run `python prepare_data.py`. It performs these steps and writes `outputs/splits.csv`,
`outputs/data_summary.json`, the crops in `outputs/crops/` and the data figures in `outputs/figures/`.

### 3A. Jellyfish
1. Read only `Train_Test_Valid/{Train,valid,test}` (the top-level species folders are copies).
2. Link every `aug-<k>-<name>` file to its original `<name>`.
3. Compute a 256-bit **average hash** (a fingerprint: shrink to 16×16 grey, 1 where a pixel is brighter than
   the mean) for every photo; photos whose fingerprints differ in ≤ 12 bits are treated as the same photo.
4. Put all copies of a photo in the same **group** (union-find). Result: 979 files → 369 groups.

### 3B. TrashCan plastic bags (the main "plastic bag" class)
1. Read both annotation files and pool them (we re-split later).
2. Keep only objects of class `trash_bag`.
3. **Remove the text / watermark bands**: keep only the clean middle part of each frame
   (rows 20 %–80 % of 480×270 frames, rows 28 %–73 % of 480×360 frames) and cut every box to that band.
4. **Drop crops that are too small or lost too much**: both sides ≥ 32 px and ≥ 60 % of the box area kept.
5. Add 10 % context around the box and **cut the object out** of the frame (a "crop").
6. **Remove near-copies from consecutive frames**: inside each video, a crop whose fingerprint differs in
   ≤ 10 bits from an already kept crop is dropped.
7. Each **video is one group**, so frames of one video never end up in both training and testing.

Result: 908 bag boxes → 385 dropped (text band / too small) → 114 dropped (near-copies) → **409 bag crops from 39 videos**.

### 3C. Synthetic bags (for the cross-dataset study only)
Randomly take 225 "Plastic Bag" + 225 "Garbage Bag" images (450, similar to the 409 TrashCan bags).
Paper bags are not used (paper is not plastic).

### 3D. Splitting 60 / 20 / 20 without leakage
* Jellyfish and synthetic bags: **groups** are shuffled and split 60/20/20 separately for every class
  (as in the lab split ratios). Validation and test keep **one photo per group** (the original, not an aug copy).
* TrashCan bags: **whole videos** are assigned so that about 60/20/20 of the crops land in train/val/test
  (15 / 12 / 12 videos).

### 3E. Extra test sets
* **Background swap** (82 test bags × 3): each TrashCan test bag is cut out with its outline (mask) and pasted
  on its own background, on plain grey, and on bright blue like the jellyfish photos.
* **Unseen objects** (276 crops): bottles, cans, containers, fish and starfish from videos never used for
  training, cleaned the same way. The model never learns these classes, so it should be unsure about them.

### 3F. Class balance check
Counts after cleaning (plastic bags = TrashCan):

| Task | Split | plastic bag | jellyfish | Imbalance ratio (largest / smallest class) |
|---|---|---|---|---|
| binary | train (with the dataset's aug copies) | 245 | 589 | 2.40 : 1 |
| binary | train, unique photos only | 245 | 236 | 1.04 : 1 |
| binary | test | 82 | 74 | 1.11 : 1 |

| Task | Split | plastic bag | barrel | blue | compass | lion's mane | mauve stinger | moon | Ratio |
|---|---|---|---|---|---|---|---|---|---|
| fine | train (with aug copies) | 245 | 98 | 102 | 100 | 96 | 103 | 90 | 2.72 : 1 |
| fine | train, unique photos only | 245 | 31 | 42 | 41 | 45 | 43 | 34 | 7.90 : 1 |
| fine | test | 82 | 10 | 13 | 12 | 15 | 14 | 10 | 8.20 : 1 |

**Conclusion.** The **binary** task is only mildly imbalanced (almost balanced if we count unique photos).
The **fine** task is clearly **imbalanced**: the plastic-bag class is 2.7× larger than each species in training
(7.9× if we count unique photos) and 8× larger in the test set. Accuracy alone would be misleading, so we
report **macro-F1**, **balanced accuracy** and **worst-class recall**, and we compare imbalance-handling methods.

---

## Step 4 – Handling class imbalance

Deep models (`train.py --imbalance ...`, compared in the `imbalance` stage with ResNet-50 on the fine task):

| Method | What it does | Where it comes from |
|---|---|---|
| none | plain cross-entropy (reference) | – |
| weighted (default) | rare classes count more in the loss (weight = 1 / class size) | cost-sensitive learning; `class_weight="balanced"` in the Spam lab |
| oversample | rare-class images are drawn more often, so every batch is balanced (`WeightedRandomSampler`) | random oversampling |
| undersample | big classes are randomly cut down to the size of the smallest class (aug copies removed first) | random undersampling |
| focal | focal loss: easy images count less, hard ones more, plus class weights | Lin et al., ICCV 2017 |
| combo | **combination**: mild undersampling (big classes capped at 2× the smallest) + oversampling (balanced batches) + focal loss | hybrid sampling + loss |

Classic model (`baseline_hog.py`, best feature set with logistic regression): none, class weights,
random undersampling and **SMOTE** (new minority examples made by mixing a minority example with a
nearest neighbour in feature space; Chawla et al., 2002).

Note: the dataset's own `aug-*` copies are already a form of oversampling for the jellyfish classes.

---

## Step 5 – Models and experiments (`run_all.py`)

| Stage | What runs | Lab it builds on |
|---|---|---|
| classic | majority, "bluer = jellyfish" rule, raw pixels, HOG, colour, HOG+colour → SVM / logistic regression (validation grid search + heat-maps), imbalance methods incl. SMOTE | HOG, Logistic Regression, Spam, Image Classification, Week 9 labs |
| clip | CLIP ViT-B/32 zero-shot (3 prompt styles + ensemble) and linear probe | Week 8 CLIP lab |
| lrsweep | learning rate 3e-5 / 1e-4 / 3e-4 for ResNet-50 and ViT-B/16, chosen on validation macro-F1 | Linear Regression lab (validation search) |
| main | TinyCNN (scratch), ResNet-50, EfficientNet-B0, ViT-B/16, VGG16 (ImageNet, fine-tuned), standard augmentation | Image Classification lab |
| underwater | the same pretrained models with our underwater-domain augmentation | – |
| imbalance | ResNet-50 with the 6 imbalance methods | – |
| crossdata | ResNet-50 and ViT-B/16 (+ classic + CLIP) trained on synthetic bags or on both bag sources, tested on both | – |
| ablation | ResNet-50 without augmentation / frozen backbone / without pretraining | – |
| datasize | ResNet-50 with 10 %, 25 %, 50 % of the training data | – |
| gradcam | Grad-CAM heat-maps and the energy-in-mask / pointing-game numbers | – |
| summary | all tables (Markdown + LaTeX), figures and McNemar tests | – |

**Training settings** (all deep models): 224×224 input, ImageNet normalisation, AdamW (weight decay 1e-4),
cosine learning-rate schedule, batch 32, up to 15 epochs (TinyCNN and scratch models 30), early stopping
on **validation macro-F1** (patience 5), the best validation model is tested **once**. Three seeds
(813, 814, 815); results are reported as mean ± standard deviation.

**Underwater-domain augmentation** (our proposed fix): with probability 0.5 each, a random underwater colour
cast and haze (I = J·t + B·(1−t), t = exp(−β·d)), a random low resolution (like a video frame) and a random
brightness change – applied to **both** classes, so that colour, sharpness and brightness no longer tell
bags and jellyfish apart.

---

## Step 6 – Evaluation

* **Metrics:** accuracy, macro-F1 (main metric), balanced accuracy, worst-class recall, ROC-AUC, expected
  calibration error, plastic recall and plastic precision; per-class report and confusion matrix.
* **Statistics:** mean ± std over 3 seeds; exact McNemar test between the best model and every other model.
* **Robustness tests:** cross-dataset (TrashCan bags vs. synthetic bags), underwater severity 1/3/5,
  brightness ×0.5–×1.5, Gaussian noise, FGSM adversarial attack, background swap, unseen objects (OOD AUROC).
* **Explainability:** Grad-CAM; share of the heat on the bag outline (compare with the bag's area) and
  pointing-game hit rate, for the same bag on three backgrounds.

---

## Step 7 – How to reproduce everything

**On Google Colab (recommended, free T4 GPU)**
1. Put `COMP813_Project_Code/`, `COMP813_data.zip` and `COMP813_trashcan.zip` in `MyDrive/COMP813`.
2. Open `COMP813_Project_Colab.ipynb` with Google Colaboratory; Runtime → Change runtime type → T4 GPU.
3. Run section 0 (mount + unzip), 1 (install), 2 (prepare data), 3 (quick test).
4. Run the section 4 cells (split between the three members), then section 5 (Grad-CAM + summary).
5. Read `outputs/summary/key_numbers.md` and the figures in `outputs/summary/` and `outputs/figures/`.

**On your own computer**
```
pip install -r requirements.txt
python prepare_data.py
python run_all.py --quick          # check
python run_all.py                  # everything (needs a GPU)
```

The same seed (813) gives the same splits and crops. GPU results can differ very slightly between machines.

---

## Step 8 – What was done so far (project history)

| Date | What happened |
|---|---|
| 26 Sep | Read the brief and catalog; chose an Advanced 3-member project. Built the first pipeline (PyTorch, lab style) with synthetic bags; found the jellyfish duplicates/leakage and the land-vs-water shortcut (colour-only 91 %). Wrote the project plan. |
| 26 Sep | Checked all 9 lab notebooks; added the lab methods (validation learning-rate search, grid heat-maps, rule and majority baselines, pixel softmax baseline, ROC curves). |
| 27 Sep | Colab quick test passed on a GPU (all stages, pretrained models, CLIP). Fixed two plotting bugs. |
| 28 Sep | Reviewed a group member's TensorFlow script (wrong labels, leakage, imbalance, no ablations); added VGG16 to our pipeline as her contribution. |
| 30 Sep | Added **TrashCan 1.0** as the main plastic-bag source (real underwater bags): cleaning rules, video-level split, new background-swap and unseen-object tests, brightness test, cross-dataset study, class-balance check and **6 imbalance methods** (+ SMOTE for the classic model). Re-tested the whole pipeline on the real data. |

**First look at the TrashCan data** (colour-histogram + logistic regression only, before any deep model):
the colour shortcut is much weaker now that both classes are underwater (binary accuracy 57 % instead of 91 %),
and the imbalance methods already change the result (binary macro-F1: none 0.51, class weights 0.57,
undersampling 0.63, SMOTE 0.53). The full runs will give the final numbers.

---

## Step 9 – Limitations to mention in the report

* TrashCan bags are small, blurry crops from deep-sea video; the jellyfish photos are sharp and bright.
  A model could use sharpness or brightness as a new shortcut – the brightness test, the background swap,
  the colour-only baseline and Grad-CAM check this.
* Only 39 videos contain usable bags, and only about 60 unique photos exist per jellyfish species, so the
  test sets are small (82 bags, 74 jellyfish) – hence 3 seeds, McNemar tests and careful wording.
* The text-band cleaning removes some bags near the frame edges.
* Background swap and underwater effects are simulations, not real photos.
