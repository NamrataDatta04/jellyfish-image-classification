# Dataset preparation – first version (v1)

**Prepared by:** Neesha Naimeen and Banupriya Thamilvanan

This folder keeps the first version of our data-preparation code exactly as it was written, so that
the history of the dataset work is visible. It is a record, **not** part of the current pipeline.

| File | Content |
|---|---|
| `prepare_data_v1.py` | first preparation script (unchanged) |

## Data sources

| Dataset | Source |
|---|---|
| Jellyfish Image Dataset (6 species) | A. Tanwar, Kaggle: https://www.kaggle.com/datasets/anshtanwar/jellyfish-types |
| Plastic – Paper – Garbage Bag Synthetic Images | vencerlanz09, Kaggle: https://www.kaggle.com/datasets/vencerlanz09/plastic-paper-garbage-bag-synthetic-images |
| TrashCan 1.0 (used from the current version on) | Hong, Fulton and Sattar, University of Minnesota: https://conservancy.umn.edu/handle/11299/214865 (we used the Kaggle copy https://www.kaggle.com/datasets/mexwell/trashcan-1-0) |

## What version 1 does

1. **Jellyfish:** reads only `Train_Test_Valid` (the six top-level species folders are exact copies of
   `Train`), links `aug-<k>-<name>` copies to their original, groups near-duplicates with a 256-bit
   average hash (≤ 12 bits apart) and union-find (979 files → 369 unique photos), and reports the leaks in
   the provided split.
2. **Synthetic bags:** 250 "Plastic Bag" + 250 "Garbage Bag" images (seed 813) form the plastic-bag class.
3. **Split:** 60/20/20 by group within each class; validation and test keep one image per group.
4. **Extra tests:** 300 paper bags as unseen objects, and a background-swap set (one bag per image cut out
   with its COCO mask and pasted on its own background, plain grey and a synthetic water background).

## Why the current version is different

Version 1 showed that the task could be solved by background colour alone: a colour-only classifier
reached 91.4 % accuracy, and only 0.3 % of the same bags were still called "plastic" on a water
background. The current `../prepare_data.py` keeps all jellyfish steps of version 1 but uses real
underwater bags from TrashCan as the main bag class; the synthetic bags (now 225 + 225) are used only in the
cross-dataset study. `../PROCESS.md` (Step 3) describes every current step.

## Running it

`prepare_data_v1.py` needs the settings of the first `common.py` (`N_BAG_PER_FOLDER`, `N_OOD_PAPER`,
`N_SUP`, `SUP_DIR`, `synthetic_water_background`), which were replaced in the current version, so it does
not run with the current code. Use `../prepare_data.py` to rebuild the data used in the report.
