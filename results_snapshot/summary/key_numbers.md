# Key numbers for the report


## Data cleaning

Jellyfish: {"images_total": 979, "aug_copies": 581, "near_duplicate_pairs": 163, "exact_duplicate_pairs": 58, "duplicate_pairs_with_different_species_label": 7, "unique_photos(groups)": 369, "provided_valid_test_images": 79, "provided_valid_test_images_with_copy_in_provided_train": 8}

TrashCan official split: {"official_train_videos": 310, "official_val_videos": 115, "videos_in_both_official_train_and_val": 113}

TrashCan bag cleaning: {"boxes": 908, "dropped_text_band_or_too_small": 385, "dropped_near_copy_same_video": 114, "kept": 409, "videos": 39}

## Class balance (bags = TrashCan)


### binary

| task | split | plastic_bag | medusozoa | imbalance_ratio |
|---|---|---|---|---|
| binary | train | 245 | 589 | 2.4 |
| binary | train_unique_photos_only | 245 | 236 | 1.04 |
| binary | val | 82 | 74 | 1.11 |
| binary | test | 82 | 74 | 1.11 |

### fine

| task | split | plastic_bag | barrel_jelly | blue_jelly | compass_jelly | lions_mane_jelly | mauve_stinger_jelly | moon_jelly | imbalance_ratio |
|---|---|---|---|---|---|---|---|---|---|
| fine | train | 245 | 98 | 102 | 100 | 96 | 103 | 90 | 2.72 |
| fine | train_unique_photos_only | 245 | 31 | 42 | 41 | 45 | 43 | 34 | 7.9 |
| fine | val | 82 | 10 | 13 | 13 | 15 | 13 | 10 | 8.2 |
| fine | test | 82 | 10 | 13 | 12 | 15 | 14 | 10 | 8.2 |

## Main comparison — fine task

| config | n_seeds | accuracy | macro_f1 | worst_class_recall | auc | plastic_recall | ece |
|---|---|---|---|---|---|---|---|
| vit_b_16 | aug=standard | 3 | 0.915 ± 0.010 | 0.849 ± 0.007 | 0.687 ± 0.085 | 0.987 ± 0.005 | 0.984 ± 0.028 | 0.067 ± 0.018 |
| efficientnet_b0 | aug=standard | 3 | 0.889 ± 0.043 | 0.837 ± 0.026 | 0.615 ± 0.077 | 0.988 ± 0.004 | 0.923 ± 0.092 | 0.141 ± 0.067 |
| resnet50 | aug=standard | 3 | 0.882 ± 0.043 | 0.824 ± 0.034 | 0.564 ± 0.044 | 0.982 ± 0.009 | 0.931 ± 0.069 | 0.109 ± 0.040 |
| clip_linear_probe | 1 | 0.878 | 0.775 | 0.500 | 0.978 | 0.988 | 0.129 |
| vgg16 | aug=standard | 3 | 0.750 ± 0.040 | 0.731 ± 0.005 | 0.586 ± 0.080 | 0.959 ± 0.005 | 0.695 ± 0.074 | 0.126 ± 0.006 |
| clip_zeroshot_underwater | 1 | 0.654 | 0.539 | 0.100 | 0.929 | 0.732 | 0.078 |
| tinycnn | scratch | aug=standard | 3 | 0.566 ± 0.026 | 0.414 ± 0.032 | 0.151 ± 0.050 | 0.834 ± 0.011 | 0.703 ± 0.067 | 0.128 ± 0.032 |
| hog+svm | 1 | 0.513 | 0.354 | 0.231 | 0.739 | 0.695 | 0.159 |
| hog+color+svm | 1 | 0.346 | 0.301 | 0.100 | 0.737 | 0.329 | 0.223 |
| color+logreg | 1 | 0.276 | 0.283 | 0.100 | 0.722 | 0.220 | 0.409 |
| pixels+logreg | 1 | 0.288 | 0.242 | 0.100 | 0.734 | 0.293 | 0.252 |
| majority | 1 | 0.526 | 0.098 | 0.000 | 0.500 | 1.000 | 0.232 |

### McNemar test — fine

| best | other | only_best_correct | only_other_correct | p_value | significant_5pct |
|---|---|---|---|---|---|
| vit_b_16 | aug=standard | efficientnet_b0 | aug=standard | 6 | 6 | 1.0 | False |
| vit_b_16 | aug=standard | resnet50 | aug=standard | 8 | 6 | 0.7905 | False |
| vit_b_16 | aug=standard | clip_linear_probe | 11 | 5 | 0.2101 | False |
| vit_b_16 | aug=standard | vgg16 | aug=standard | 26 | 7 | 0.0013 | True |
| vit_b_16 | aug=standard | clip_zeroshot_underwater | 47 | 6 | 0.0 | True |
| vit_b_16 | aug=standard | tinycnn | scratch | aug=standard | 57 | 3 | 0.0 | True |
| vit_b_16 | aug=standard | hog+svm | 64 | 1 | 0.0 | True |
| vit_b_16 | aug=standard | hog+color+svm | 94 | 5 | 0.0 | True |
| vit_b_16 | aug=standard | color+logreg | 103 | 3 | 0.0 | True |
| vit_b_16 | aug=standard | pixels+logreg | 99 | 1 | 0.0 | True |
| vit_b_16 | aug=standard | majority | 61 | 0 | 0.0 | True |

## Main comparison — binary task

| config | n_seeds | accuracy | macro_f1 | worst_class_recall | auc | plastic_recall | ece |
|---|---|---|---|---|---|---|---|
| clip_linear_probe | 1 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.237 |
| efficientnet_b0 | aug=standard | 3 | 0.983 ± 0.020 | 0.983 ± 0.020 | 0.967 ± 0.037 | 1.000 ± 0.000 | 0.967 ± 0.037 | 0.044 ± 0.032 |
| clip_zeroshot_underwater | 1 | 0.949 | 0.948 | 0.905 | 0.989 | 0.988 | 0.023 |
| vit_b_16 | aug=standard | 3 | 0.944 ± 0.048 | 0.944 ± 0.048 | 0.894 ± 0.092 | 1.000 ± 0.000 | 0.894 ± 0.092 | 0.055 ± 0.044 |
| resnet50 | aug=standard | 3 | 0.940 ± 0.055 | 0.940 ± 0.055 | 0.886 ± 0.104 | 1.000 ± 0.001 | 0.886 ± 0.104 | 0.075 ± 0.026 |
| tinycnn | scratch | aug=standard | 3 | 0.938 ± 0.024 | 0.938 ± 0.024 | 0.911 ± 0.060 | 0.980 ± 0.005 | 0.911 ± 0.060 | 0.050 ± 0.005 |
| vgg16 | aug=standard | 3 | 0.891 ± 0.106 | 0.889 ± 0.109 | 0.793 ± 0.201 | 1.000 ± 0.000 | 0.793 ± 0.201 | 0.084 ± 0.038 |
| hog+svm | 1 | 0.846 | 0.846 | 0.756 | 0.938 | 0.756 | 0.143 |
| hog+color+svm | 1 | 0.673 | 0.646 | 0.378 | 0.814 | 0.378 | 0.050 |
| pixels+svm | 1 | 0.641 | 0.638 | 0.524 | 0.730 | 0.524 | 0.095 |
| rule_blue | 1 | 0.622 | 0.587 | 0.351 | 0.484 | 0.866 | 0.280 |
| color+logreg | 1 | 0.609 | 0.570 | 0.293 | 0.741 | 0.293 | 0.277 |
| majority | 1 | 0.474 | 0.322 | 0.000 | 0.500 | 0.000 | 0.232 |

### McNemar test — binary

| best | other | only_best_correct | only_other_correct | p_value | significant_5pct |
|---|---|---|---|---|---|
| clip_linear_probe | efficientnet_b0 | aug=standard | 0 | 0 | 1.0 | False |
| clip_linear_probe | clip_zeroshot_underwater | 8 | 0 | 0.0078 | True |
| clip_linear_probe | vit_b_16 | aug=standard | 9 | 0 | 0.0039 | True |
| clip_linear_probe | resnet50 | aug=standard | 3 | 0 | 0.25 | False |
| clip_linear_probe | tinycnn | scratch | aug=standard | 14 | 0 | 0.0001 | True |
| clip_linear_probe | vgg16 | aug=standard | 1 | 0 | 1.0 | False |
| clip_linear_probe | hog+svm | 24 | 0 | 0.0 | True |
| clip_linear_probe | hog+color+svm | 51 | 0 | 0.0 | True |
| clip_linear_probe | pixels+svm | 56 | 0 | 0.0 | True |
| clip_linear_probe | rule_blue | 59 | 0 | 0.0 | True |
| clip_linear_probe | color+logreg | 61 | 0 | 0.0 | True |
| clip_linear_probe | majority | 82 | 0 | 0.0 | True |

## Ablation — resnet50, fine task

| config | n_seeds | accuracy | macro_f1 | worst_class_recall | auc | plastic_recall | ece |
|---|---|---|---|---|---|---|---|
| resnet50 | aug=none | 1 | 0.904 | 0.847 | 0.692 | 0.959 | 0.963 | 0.181 |
| resnet50 | aug=standard | 3 | 0.882 ± 0.043 | 0.824 ± 0.034 | 0.564 ± 0.044 | 0.982 ± 0.009 | 0.931 ± 0.069 | 0.109 ± 0.040 |
| resnet50 | aug=underwater | 3 | 0.827 ± 0.023 | 0.765 ± 0.033 | 0.564 ± 0.149 | 0.968 ± 0.013 | 0.858 ± 0.043 | 0.074 ± 0.028 |
| resnet50 | frozen | aug=standard | 1 | 0.833 | 0.706 | 0.231 | 0.954 | 0.976 | 0.434 |
| resnet50 | scratch | aug=standard | 1 | 0.365 | 0.314 | 0.077 | 0.815 | 0.341 | 0.209 |

## Class-imbalance methods — deep models, fine task

| config | imbalance_method | n_seeds | macro_f1 | balanced_accuracy | worst_class_recall | plastic_recall | ece |
|---|---|---|---|---|---|---|---|
| resnet50 | aug=standard | imb=none | none | 1 | 0.857 | 0.859 | 0.615 | 1.000 | 0.049 |
| resnet50 | aug=standard | weighted | 3 | 0.824 ± 0.034 | 0.841 ± 0.026 | 0.564 ± 0.044 | 0.931 ± 0.069 | 0.109 ± 0.040 |
| resnet50 | aug=standard | imb=focal | focal | 1 | 0.793 | 0.808 | 0.538 | 0.927 | 0.217 |
| resnet50 | aug=standard | imb=undersample | undersample | 1 | 0.789 | 0.858 | 0.538 | 0.768 | 0.047 |
| resnet50 | aug=standard | imb=combo | combo | 1 | 0.744 | 0.786 | 0.385 | 0.890 | 0.206 |
| resnet50 | aug=standard | imb=oversample | oversample | 1 | 0.681 | 0.736 | 0.500 | 0.720 | 0.072 |

### Class-imbalance methods — classic model (classic_imbalance_binary)

| task | feature | method | train_size | accuracy | macro_f1 | balanced_accuracy | worst_class_recall | plastic_recall |
|---|---|---|---|---|---|---|---|---|
| binary | hog+color | none | 834 | 0.699 | 0.679 | 0.713 | 0.427 | 0.427 |
| binary | hog+color | balanced | 834 | 0.731 | 0.717 | 0.744 | 0.488 | 0.488 |
| binary | hog+color | undersample | 490 | 0.763 | 0.754 | 0.774 | 0.549 | 0.549 |
| binary | hog+color | smote | 1178 | 0.744 | 0.732 | 0.756 | 0.512 | 0.512 |

### Class-imbalance methods — classic model (classic_imbalance_fine)

| task | feature | method | train_size | accuracy | macro_f1 | balanced_accuracy | worst_class_recall | plastic_recall |
|---|---|---|---|---|---|---|---|---|
| fine | pixels | none | 834 | 0.404 | 0.29 | 0.309 | 0.1 | 0.5 |
| fine | pixels | balanced | 834 | 0.288 | 0.242 | 0.269 | 0.1 | 0.293 |
| fine | pixels | undersample | 630 | 0.269 | 0.256 | 0.307 | 0.1 | 0.195 |
| fine | pixels | smote | 1715 | 0.327 | 0.262 | 0.288 | 0.1 | 0.354 |

## Learning-rate search (validation macro-F1, fine task)

| model | 3e-05 | 0.0001 | 0.0003 |
|---|---|---|---|
| resnet50 | 0.839 | 0.907 | 0.897 |
| vit_b_16 | 0.9 | 0.856 | 0.756 |

## Cross-dataset test (macro-F1) — binary

| label | trained on | test on TrashCan bags | test on synthetic bags |
|---|---|---|---|
| clip_linear_probe | trashcan | 1.0 | 0.86 |
| clip_linear_probe | both | 0.994 | 1.0 |
| clip_linear_probe | synthetic | 0.871 | 1.0 |
| clip_zeroshot_photo | synthetic | 0.936 | 1.0 |
| clip_zeroshot_underwater | trashcan | 0.948 | 0.956 |
| clip_zeroshot_underwater | both | 0.948 | 0.956 |
| color+logreg | trashcan | 0.57 | 0.326 |
| color+logreg | both | 0.509 | 0.92 |
| color+logreg | synthetic | 0.345 | 0.908 |
| hog+color+logreg | both | 0.623 | 0.902 |
| hog+color+svm | trashcan | 0.646 | 0.435 |
| hog+color+svm | synthetic | 0.349 | 0.895 |
| hog+logreg | synthetic | 0.471 | 0.744 |
| hog+svm | trashcan | 0.846 | 0.476 |
| hog+svm | both | 0.756 | 0.732 |
| majority | trashcan | 0.322 | 0.311 |
| majority | both | 0.322 | 0.311 |
| majority | synthetic | 0.322 | 0.311 |
| pixels+logreg | both | 0.58 | 0.804 |
| pixels+logreg | synthetic | 0.288 | 0.829 |
| resnet50 | aug=standard | trashcan | 0.94 | 0.36 |
| resnet50 | aug=standard | both | 0.994 | 1.0 |
| resnet50 | aug=standard | synthetic | 0.529 | 1.0 |
| rule_blue | trashcan | 0.587 | 0.655 |
| rule_blue | both | 0.596 | 0.659 |
| rule_blue | synthetic | 0.346 | 0.798 |
| vit_b_16 | aug=standard | trashcan | 0.944 | 0.528 |
| vit_b_16 | aug=standard | both | 0.974 | 1.0 |
| vit_b_16 | aug=standard | synthetic | 0.534 | 0.994 |

## Background-swap plastic recall — binary

| label | sup_orig | sup_gray | sup_blue |
|---|---|---|---|
| clip_linear_probe | 1.0 | 1.0 | 0.829 |
| clip_zeroshot_underwater | 1.0 | 1.0 | 0.988 |
| color+logreg | 0.293 | 0.012 | 0.037 |
| efficientnet_b0 | aug=standard | 0.972 | 0.74 | 0.577 |
| efficientnet_b0 | aug=underwater | 0.898 | 0.533 | 0.28 |
| hog+color+svm | 0.378 | 0.122 | 0.22 |
| hog+svm | 0.793 | 0.561 | 0.671 |
| majority | 0.0 | 0.0 | 0.0 |
| pixels+svm | 0.524 | 0.768 | 0.122 |
| resnet50 | aug=standard | 0.882 | 0.419 | 0.301 |
| resnet50 | aug=underwater | 0.805 | 0.321 | 0.134 |
| rule_blue | 0.866 | 0.976 | 0.0 |
| tinycnn | aug=standard | scratch | 0.911 | 0.98 | 0.183 |
| vgg16 | aug=standard | 0.801 | 0.447 | 0.061 |
| vit_b_16 | aug=standard | 0.894 | 0.654 | 0.407 |
| vit_b_16 | aug=underwater | 0.821 | 0.622 | 0.0 |

## Unseen objects (out-of-distribution) — binary

| label | pred_plastic_rate | mean_confidence | ood_auroc |
|---|---|---|---|
| clip_linear_probe | 0.975 | 0.731 | 0.627 |
| clip_zeroshot_underwater | 0.906 | 0.894 | 0.612 |
| color+logreg | 0.471 | 0.846 | 0.568 |
| efficientnet_b0 | aug=standard | 0.843 | 0.883 | 0.679 |
| efficientnet_b0 | aug=underwater | 0.746 | 0.841 | 0.719 |
| hog+color+svm | 0.525 | 0.663 | 0.606 |
| hog+svm | 0.623 | 0.674 | 0.604 |
| majority | 0.0 | 0.706 | 0.5 |
| pixels+svm | 0.54 | 0.653 | 0.597 |
| resnet50 | aug=standard | 0.838 | 0.874 | 0.554 |
| resnet50 | aug=underwater | 0.726 | 0.867 | 0.686 |
| rule_blue | 0.837 | 0.86 | 0.428 |
| tinycnn | aug=standard | scratch | 0.849 | 0.909 | 0.568 |
| vgg16 | aug=standard | 0.775 | 0.877 | 0.687 |
| vit_b_16 | aug=standard | 0.843 | 0.967 | 0.693 |
| vit_b_16 | aug=underwater | 0.78 | 0.905 | 0.68 |

## Accuracy vs brightness — binary

| label | 0.5 | 0.75 | 1.0 | 1.25 | 1.5 |
|---|---|---|---|---|---|
| clip_linear_probe | nan | nan | 1.0 | nan | nan |
| clip_zeroshot_underwater | nan | nan | 0.949 | nan | nan |
| color+logreg | nan | nan | 0.609 | nan | nan |
| efficientnet_b0 | aug=standard | 0.996 | 0.987 | 0.983 | 0.974 | 0.949 |
| efficientnet_b0 | aug=underwater | 0.987 | 0.966 | 0.944 | 0.927 | 0.936 |
| hog+color+svm | nan | nan | 0.673 | nan | nan |
| hog+svm | nan | nan | 0.846 | nan | nan |
| majority | nan | nan | 0.474 | nan | nan |
| pixels+svm | nan | nan | 0.641 | nan | nan |
| resnet50 | aug=standard | 0.964 | 0.959 | 0.94 | 0.938 | 0.94 |
| resnet50 | aug=underwater | 0.934 | 0.904 | 0.895 | 0.912 | 0.919 |
| rule_blue | nan | nan | 0.622 | nan | nan |
| tinycnn | aug=standard | scratch | 0.645 | 0.902 | 0.938 | 0.929 | 0.923 |
| vgg16 | aug=standard | 0.827 | 0.861 | 0.891 | 0.897 | 0.906 |
| vit_b_16 | aug=standard | 0.915 | 0.947 | 0.944 | 0.944 | 0.944 |
| vit_b_16 | aug=underwater | 0.887 | 0.908 | 0.906 | 0.915 | 0.915 |

## Accuracy clean vs underwater — binary

| label | test | test_uw3 | test_uw5 |
|---|---|---|---|
| clip_linear_probe | 1.0 | 1.0 | 0.814 |
| clip_zeroshot_underwater | 0.949 | 0.897 | 0.84 |
| color+logreg | 0.609 | 0.474 | 0.474 |
| efficientnet_b0 | aug=standard | 0.983 | 1.0 | 0.979 |
| efficientnet_b0 | aug=underwater | 0.944 | 0.972 | 0.979 |
| hog+color+svm | 0.673 | 0.538 | 0.609 |
| hog+svm | 0.846 | 0.84 | 0.821 |
| majority | 0.474 | 0.474 | 0.474 |
| pixels+svm | 0.641 | 0.667 | 0.635 |
| resnet50 | aug=standard | 0.94 | 0.942 | 0.947 |
| resnet50 | aug=underwater | 0.895 | 0.942 | 0.964 |
| rule_blue | 0.622 | 0.372 | 0.474 |
| tinycnn | aug=standard | scratch | 0.938 | 0.511 | 0.474 |
| vgg16 | aug=standard | 0.891 | 0.767 | 0.65 |
| vit_b_16 | aug=standard | 0.944 | 0.955 | 0.889 |
| vit_b_16 | aug=underwater | 0.906 | 0.932 | 0.919 |

## Cross-dataset test (macro-F1) — fine

| label | trained on | test on TrashCan bags | test on synthetic bags |
|---|---|---|---|
| clip_linear_probe | trashcan | 0.775 | 0.596 |
| clip_linear_probe | both | 0.758 | 0.77 |
| clip_linear_probe | synthetic | 0.65 | 0.781 |
| clip_zeroshot_photo | synthetic | 0.535 | 0.629 |
| clip_zeroshot_underwater | trashcan | 0.539 | 0.609 |
| clip_zeroshot_underwater | both | 0.539 | 0.609 |
| color+logreg | trashcan | 0.283 | 0.182 |
| color+logreg | both | 0.252 | 0.348 |
| color+logreg | synthetic | 0.208 | 0.353 |
| hog+color+logreg | both | 0.321 | 0.429 |
| hog+color+logreg | synthetic | 0.227 | 0.412 |
| hog+logreg | both | 0.381 | 0.345 |
| hog+logreg | synthetic | 0.224 | 0.331 |
| majority | trashcan | 0.098 | 0.101 |
| majority | both | 0.098 | 0.101 |
| majority | synthetic | 0.098 | 0.101 |
| pixels+logreg | trashcan | 0.242 | 0.23 |
| pixels+logreg | both | 0.286 | 0.31 |
| pixels+logreg | synthetic | 0.21 | 0.342 |
| resnet50 | aug=standard | trashcan | 0.824 | 0.531 |
| resnet50 | aug=standard | both | 0.841 | 0.841 |
| resnet50 | aug=standard | synthetic | 0.554 | 0.832 |
| vit_b_16 | aug=standard | trashcan | 0.849 | 0.618 |
| vit_b_16 | aug=standard | both | 0.823 | 0.855 |
| vit_b_16 | aug=standard | synthetic | 0.623 | 0.841 |

## Background-swap plastic recall — fine

| label | sup_orig | sup_gray | sup_blue |
|---|---|---|---|
| clip_linear_probe | 0.988 | 0.89 | 0.768 |
| clip_zeroshot_underwater | 0.744 | 0.927 | 0.915 |
| color+logreg | 0.22 | 0.012 | 0.012 |
| efficientnet_b0 | aug=standard | 0.915 | 0.675 | 0.602 |
| efficientnet_b0 | aug=underwater | 0.862 | 0.374 | 0.167 |
| hog+color+svm | 0.329 | 0.049 | 0.098 |
| hog+svm | 0.707 | 0.537 | 0.573 |
| majority | 1.0 | 1.0 | 1.0 |
| pixels+logreg | 0.293 | 0.305 | 0.0 |
| resnet50 | aug=none | 0.963 | 0.744 | 0.671 |
| resnet50 | aug=standard | 0.923 | 0.602 | 0.455 |
| resnet50 | aug=underwater | 0.866 | 0.496 | 0.289 |
| tinycnn | aug=standard | scratch | 0.703 | 0.821 | 0.154 |
| vgg16 | aug=standard | 0.703 | 0.455 | 0.26 |
| vit_b_16 | aug=standard | 0.984 | 0.679 | 0.159 |
| vit_b_16 | aug=underwater | 0.907 | 0.724 | 0.179 |

## Unseen objects (out-of-distribution) — fine

| label | pred_plastic_rate | mean_confidence | ood_auroc |
|---|---|---|---|
| clip_linear_probe | 0.978 | 0.876 | 0.333 |
| clip_zeroshot_underwater | 0.641 | 0.665 | 0.456 |
| color+logreg | 0.333 | 0.636 | 0.516 |
| efficientnet_b0 | aug=standard | 0.806 | 0.66 | 0.633 |
| efficientnet_b0 | aug=underwater | 0.582 | 0.535 | 0.684 |
| hog+color+svm | 0.402 | 0.447 | 0.533 |
| hog+svm | 0.587 | 0.389 | 0.491 |
| majority | 1.0 | 0.294 | 0.5 |
| pixels+logreg | 0.37 | 0.476 | 0.594 |
| resnet50 | aug=none | 0.862 | 0.663 | 0.613 |
| resnet50 | aug=standard | 0.831 | 0.772 | 0.535 |
| resnet50 | aug=underwater | 0.671 | 0.716 | 0.6 |
| tinycnn | aug=standard | scratch | 0.547 | 0.686 | 0.428 |
| vgg16 | aug=standard | 0.713 | 0.875 | 0.505 |
| vit_b_16 | aug=standard | 0.857 | 0.897 | 0.588 |
| vit_b_16 | aug=underwater | 0.839 | 0.86 | 0.588 |

## Accuracy vs brightness — fine

| label | 0.5 | 0.75 | 1.0 | 1.25 | 1.5 |
|---|---|---|---|---|---|
| clip_linear_probe | nan | nan | 0.878 | nan | nan |
| clip_zeroshot_underwater | nan | nan | 0.654 | nan | nan |
| color+logreg | nan | nan | 0.276 | nan | nan |
| efficientnet_b0 | aug=standard | 0.88 | 0.878 | 0.889 | 0.87 | 0.855 |
| efficientnet_b0 | aug=underwater | 0.865 | 0.859 | 0.827 | 0.825 | 0.844 |
| hog+color+svm | nan | nan | 0.346 | nan | nan |
| hog+svm | nan | nan | 0.513 | nan | nan |
| majority | nan | nan | 0.526 | nan | nan |
| pixels+logreg | nan | nan | 0.288 | nan | nan |
| resnet50 | aug=none | 0.885 | 0.885 | 0.904 | 0.891 | 0.859 |
| resnet50 | aug=standard | 0.859 | 0.878 | 0.882 | 0.889 | 0.904 |
| resnet50 | aug=underwater | 0.853 | 0.823 | 0.827 | 0.812 | 0.812 |
| tinycnn | aug=standard | scratch | 0.297 | 0.395 | 0.566 | 0.545 | 0.53 |
| vgg16 | aug=standard | 0.754 | 0.741 | 0.75 | 0.774 | 0.784 |
| vit_b_16 | aug=standard | 0.91 | 0.917 | 0.915 | 0.904 | 0.904 |
| vit_b_16 | aug=underwater | 0.838 | 0.84 | 0.85 | 0.868 | 0.872 |

## Accuracy clean vs underwater — fine

| label | test | test_uw3 | test_uw5 |
|---|---|---|---|
| clip_linear_probe | 0.878 | 0.833 | 0.692 |
| clip_zeroshot_underwater | 0.654 | 0.622 | 0.455 |
| color+logreg | 0.276 | 0.077 | 0.058 |
| efficientnet_b0 | aug=standard | 0.889 | 0.878 | 0.818 |
| efficientnet_b0 | aug=underwater | 0.827 | 0.846 | 0.846 |
| hog+color+svm | 0.346 | 0.109 | 0.224 |
| hog+svm | 0.513 | 0.551 | 0.513 |
| majority | 0.526 | 0.526 | 0.526 |
| pixels+logreg | 0.288 | 0.224 | 0.141 |
| resnet50 | aug=none | 0.904 | 0.788 | 0.66 |
| resnet50 | aug=standard | 0.882 | 0.816 | 0.712 |
| resnet50 | aug=underwater | 0.827 | 0.861 | 0.853 |
| tinycnn | aug=standard | scratch | 0.566 | 0.094 | 0.092 |
| vgg16 | aug=standard | 0.75 | 0.776 | 0.677 |
| vit_b_16 | aug=standard | 0.915 | 0.876 | 0.752 |
| vit_b_16 | aug=underwater | 0.85 | 0.831 | 0.776 |

## Grad-CAM: does the model look at the bag?

| run | background | energy_in_mask | pointing_hit_rate | mask_area | plastic_recall | mean_p_plastic | n |
|---|---|---|---|---|---|---|---|
| binary_efficientnet_b0_pt_ft_aug-standard_s813 | blue | 0.548 | 0.817 | 0.486 | 0.488 | 0.515 | 82 |
| binary_efficientnet_b0_pt_ft_aug-standard_s813 | gray | 0.432 | 0.378 | 0.486 | 0.78 | 0.699 | 82 |
| binary_efficientnet_b0_pt_ft_aug-standard_s813 | orig | 0.438 | 0.195 | 0.486 | 1.0 | 0.967 | 82 |
| binary_efficientnet_b0_pt_ft_aug-underwater_s813 | blue | 0.363 | 0.488 | 0.486 | 0.098 | 0.203 | 82 |
| binary_efficientnet_b0_pt_ft_aug-underwater_s813 | gray | 0.35 | 0.232 | 0.486 | 0.524 | 0.559 | 82 |
| binary_efficientnet_b0_pt_ft_aug-underwater_s813 | orig | 0.318 | 0.061 | 0.486 | 0.976 | 0.902 | 82 |
| binary_resnet50_pt_ft_aug-standard_bags-both_s813 | blue | 0.785 | 0.902 | 0.486 | 0.768 | 0.712 | 82 |
| binary_resnet50_pt_ft_aug-standard_bags-both_s813 | gray | 0.77 | 0.902 | 0.486 | 0.951 | 0.848 | 82 |
| binary_resnet50_pt_ft_aug-standard_bags-both_s813 | orig | 0.663 | 0.805 | 0.486 | 1.0 | 0.927 | 82 |
| binary_resnet50_pt_ft_aug-standard_bags-synthetic_s813 | blue | 0.42 | 0.354 | 0.486 | 0.073 | 0.209 | 82 |
| binary_resnet50_pt_ft_aug-standard_bags-synthetic_s813 | gray | 0.555 | 0.671 | 0.486 | 0.207 | 0.322 | 82 |
| binary_resnet50_pt_ft_aug-standard_bags-synthetic_s813 | orig | 0.413 | 0.341 | 0.486 | 0.22 | 0.32 | 82 |
| binary_resnet50_pt_ft_aug-standard_s813 | blue | 0.615 | 0.598 | 0.486 | 0.402 | 0.491 | 82 |
| binary_resnet50_pt_ft_aug-standard_s813 | gray | 0.652 | 0.659 | 0.486 | 0.512 | 0.593 | 82 |
| binary_resnet50_pt_ft_aug-standard_s813 | orig | 0.577 | 0.598 | 0.486 | 0.951 | 0.876 | 82 |
| binary_resnet50_pt_ft_aug-underwater_s813 | blue | 0.621 | 0.744 | 0.486 | 0.134 | 0.192 | 82 |
| binary_resnet50_pt_ft_aug-underwater_s813 | gray | 0.54 | 0.61 | 0.486 | 0.354 | 0.37 | 82 |
| binary_resnet50_pt_ft_aug-underwater_s813 | orig | 0.492 | 0.402 | 0.486 | 0.793 | 0.766 | 82 |
| binary_tinycnn_scratch_ft_aug-standard_s813 | blue | 0.997 | 0.987 | 0.486 | 0.024 | 0.057 | 82 |
| binary_tinycnn_scratch_ft_aug-standard_s813 | gray | 0.273 | 0.0 | 0.486 | 0.951 | 0.946 | 82 |
| binary_tinycnn_scratch_ft_aug-standard_s813 | orig | 0.377 | 0.317 | 0.486 | 0.841 | 0.804 | 82 |
| binary_vgg16_pt_ft_aug-standard_s813 | blue | 0.929 | 0.976 | 0.486 | 0.037 | 0.096 | 82 |
| binary_vgg16_pt_ft_aug-standard_s813 | gray | 0.579 | 0.573 | 0.486 | 0.549 | 0.489 | 82 |
| binary_vgg16_pt_ft_aug-standard_s813 | orig | 0.352 | 0.134 | 0.486 | 1.0 | 0.902 | 82 |
| binary_vit_b_16_pt_ft_aug-standard_bags-both_s813 | blue | nan | nan | 0.486 | 0.72 | 0.676 | 82 |
| binary_vit_b_16_pt_ft_aug-standard_bags-both_s813 | gray | nan | nan | 0.486 | 0.939 | 0.938 | 82 |
| binary_vit_b_16_pt_ft_aug-standard_bags-both_s813 | orig | nan | nan | 0.486 | 0.951 | 0.938 | 82 |
| binary_vit_b_16_pt_ft_aug-standard_bags-synthetic_s813 | blue | nan | nan | 0.486 | 0.0 | 0.023 | 82 |
| binary_vit_b_16_pt_ft_aug-standard_bags-synthetic_s813 | gray | nan | nan | 0.486 | 0.646 | 0.614 | 82 |
| binary_vit_b_16_pt_ft_aug-standard_bags-synthetic_s813 | orig | nan | nan | 0.486 | 0.232 | 0.243 | 82 |
| binary_vit_b_16_pt_ft_aug-standard_s813 | blue | nan | nan | 0.486 | 0.476 | 0.462 | 82 |
| binary_vit_b_16_pt_ft_aug-standard_s813 | gray | nan | nan | 0.486 | 0.744 | 0.734 | 82 |
| binary_vit_b_16_pt_ft_aug-standard_s813 | orig | nan | nan | 0.486 | 0.89 | 0.882 | 82 |
| binary_vit_b_16_pt_ft_aug-underwater_s813 | blue | nan | nan | 0.486 | 0.0 | 0.008 | 82 |
| binary_vit_b_16_pt_ft_aug-underwater_s813 | gray | nan | nan | 0.486 | 0.646 | 0.618 | 82 |
| binary_vit_b_16_pt_ft_aug-underwater_s813 | orig | nan | nan | 0.486 | 0.829 | 0.82 | 82 |
| fine_efficientnet_b0_pt_ft_aug-standard_s813 | blue | 0.722 | 0.939 | 0.486 | 0.5 | 0.369 | 82 |
| fine_efficientnet_b0_pt_ft_aug-standard_s813 | gray | 0.626 | 0.707 | 0.486 | 0.683 | 0.377 | 82 |
| fine_efficientnet_b0_pt_ft_aug-standard_s813 | orig | 0.543 | 0.439 | 0.486 | 0.939 | 0.619 | 82 |
| fine_efficientnet_b0_pt_ft_aug-underwater_s813 | blue | 0.603 | 0.841 | 0.486 | 0.146 | 0.181 | 82 |
| fine_efficientnet_b0_pt_ft_aug-underwater_s813 | gray | 0.597 | 0.634 | 0.486 | 0.451 | 0.298 | 82 |
| fine_efficientnet_b0_pt_ft_aug-underwater_s813 | orig | 0.449 | 0.341 | 0.486 | 0.927 | 0.513 | 82 |
| fine_resnet50_pt_frozen_aug-standard_s813 | blue | 0.496 | 0.354 | 0.486 | 0.402 | 0.212 | 82 |
| fine_resnet50_pt_frozen_aug-standard_s813 | gray | 0.459 | 0.329 | 0.486 | 0.585 | 0.24 | 82 |
| fine_resnet50_pt_frozen_aug-standard_s813 | orig | 0.446 | 0.341 | 0.486 | 0.976 | 0.465 | 82 |
| fine_resnet50_pt_ft_aug-none_s813 | blue | 0.714 | 0.768 | 0.486 | 0.671 | 0.502 | 82 |
| fine_resnet50_pt_ft_aug-none_s813 | gray | 0.717 | 0.744 | 0.486 | 0.744 | 0.533 | 82 |
| fine_resnet50_pt_ft_aug-none_s813 | orig | 0.661 | 0.671 | 0.486 | 0.963 | 0.659 | 82 |
| fine_resnet50_pt_ft_aug-standard_bags-both_s813 | blue | 0.803 | 0.939 | 0.486 | 0.939 | 0.772 | 82 |
| fine_resnet50_pt_ft_aug-standard_bags-both_s813 | gray | 0.811 | 0.927 | 0.486 | 0.988 | 0.877 | 82 |
| fine_resnet50_pt_ft_aug-standard_bags-both_s813 | orig | 0.705 | 0.854 | 0.486 | 1.0 | 0.939 | 82 |
| fine_resnet50_pt_ft_aug-standard_bags-synthetic_s813 | blue | 0.541 | 0.549 | 0.486 | 0.183 | 0.183 | 82 |
| fine_resnet50_pt_ft_aug-standard_bags-synthetic_s813 | gray | 0.708 | 0.841 | 0.486 | 0.378 | 0.283 | 82 |
| fine_resnet50_pt_ft_aug-standard_bags-synthetic_s813 | orig | 0.559 | 0.61 | 0.486 | 0.244 | 0.236 | 82 |
| fine_resnet50_pt_ft_aug-standard_frac0.1_s813 | blue | 0.642 | 0.659 | 0.486 | 0.195 | 0.17 | 82 |
| fine_resnet50_pt_ft_aug-standard_frac0.1_s813 | gray | 0.697 | 0.659 | 0.486 | 0.366 | 0.193 | 82 |
| fine_resnet50_pt_ft_aug-standard_frac0.1_s813 | orig | 0.58 | 0.573 | 0.486 | 0.817 | 0.262 | 82 |
| fine_resnet50_pt_ft_aug-standard_frac0.25_s813 | blue | 0.696 | 0.732 | 0.486 | 0.439 | 0.296 | 82 |
| fine_resnet50_pt_ft_aug-standard_frac0.25_s813 | gray | 0.74 | 0.78 | 0.486 | 0.561 | 0.343 | 82 |
| fine_resnet50_pt_ft_aug-standard_frac0.25_s813 | orig | 0.641 | 0.732 | 0.486 | 0.89 | 0.556 | 82 |
| fine_resnet50_pt_ft_aug-standard_frac0.5_s813 | blue | 0.756 | 0.866 | 0.486 | 0.524 | 0.348 | 82 |
| fine_resnet50_pt_ft_aug-standard_frac0.5_s813 | gray | 0.772 | 0.939 | 0.486 | 0.72 | 0.487 | 82 |
| fine_resnet50_pt_ft_aug-standard_frac0.5_s813 | orig | 0.656 | 0.805 | 0.486 | 0.976 | 0.709 | 82 |
| fine_resnet50_pt_ft_aug-standard_imb-combo_s813 | blue | 0.57 | 0.537 | 0.486 | 0.415 | 0.285 | 82 |
| fine_resnet50_pt_ft_aug-standard_imb-combo_s813 | gray | 0.711 | 0.829 | 0.486 | 0.5 | 0.355 | 82 |
| fine_resnet50_pt_ft_aug-standard_imb-combo_s813 | orig | 0.604 | 0.78 | 0.486 | 0.902 | 0.619 | 82 |
| fine_resnet50_pt_ft_aug-standard_imb-focal_s813 | blue | 0.731 | 0.805 | 0.486 | 0.427 | 0.251 | 82 |
| fine_resnet50_pt_ft_aug-standard_imb-focal_s813 | gray | 0.709 | 0.768 | 0.486 | 0.512 | 0.342 | 82 |
| fine_resnet50_pt_ft_aug-standard_imb-focal_s813 | orig | 0.662 | 0.805 | 0.486 | 0.927 | 0.603 | 82 |
| fine_resnet50_pt_ft_aug-standard_imb-none_s813 | blue | 0.748 | 0.841 | 0.486 | 0.683 | 0.57 | 82 |
| fine_resnet50_pt_ft_aug-standard_imb-none_s813 | gray | 0.793 | 0.939 | 0.486 | 0.805 | 0.698 | 82 |
| fine_resnet50_pt_ft_aug-standard_imb-none_s813 | orig | 0.707 | 0.854 | 0.486 | 1.0 | 0.917 | 82 |
| fine_resnet50_pt_ft_aug-standard_imb-oversample_s813 | blue | 0.554 | 0.463 | 0.486 | 0.354 | 0.271 | 82 |
| fine_resnet50_pt_ft_aug-standard_imb-oversample_s813 | gray | 0.562 | 0.39 | 0.486 | 0.463 | 0.362 | 82 |
| fine_resnet50_pt_ft_aug-standard_imb-oversample_s813 | orig | 0.595 | 0.695 | 0.486 | 0.732 | 0.605 | 82 |
| fine_resnet50_pt_ft_aug-standard_imb-undersample_s813 | blue | 0.642 | 0.744 | 0.486 | 0.341 | 0.28 | 82 |
| fine_resnet50_pt_ft_aug-standard_imb-undersample_s813 | gray | 0.658 | 0.695 | 0.486 | 0.39 | 0.384 | 82 |
| fine_resnet50_pt_ft_aug-standard_imb-undersample_s813 | orig | 0.632 | 0.78 | 0.486 | 0.768 | 0.729 | 82 |
| fine_resnet50_pt_ft_aug-standard_s813 | blue | 0.751 | 0.805 | 0.486 | 0.488 | 0.443 | 82 |
| fine_resnet50_pt_ft_aug-standard_s813 | gray | 0.78 | 0.939 | 0.486 | 0.61 | 0.54 | 82 |
| fine_resnet50_pt_ft_aug-standard_s813 | orig | 0.677 | 0.732 | 0.486 | 0.951 | 0.802 | 82 |
| fine_resnet50_pt_ft_aug-underwater_s813 | blue | 0.753 | 0.866 | 0.486 | 0.22 | 0.2 | 82 |
| fine_resnet50_pt_ft_aug-underwater_s813 | gray | 0.722 | 0.78 | 0.486 | 0.39 | 0.305 | 82 |
| fine_resnet50_pt_ft_aug-underwater_s813 | orig | 0.652 | 0.732 | 0.486 | 0.829 | 0.688 | 82 |
| fine_resnet50_scratch_ft_aug-standard_s813 | blue | 0.88 | 0.958 | 0.486 | 0.28 | 0.194 | 82 |
| fine_resnet50_scratch_ft_aug-standard_s813 | gray | 0.372 | 0.11 | 0.486 | 0.805 | 0.673 | 82 |
| fine_resnet50_scratch_ft_aug-standard_s813 | orig | 0.683 | 0.718 | 0.486 | 0.341 | 0.335 | 82 |
| fine_tinycnn_scratch_ft_aug-standard_s813 | blue | 0.988 | 1.0 | 0.486 | 0.049 | 0.087 | 82 |
| fine_tinycnn_scratch_ft_aug-standard_s813 | gray | 0.296 | 0.037 | 0.486 | 0.878 | 0.771 | 82 |
| fine_tinycnn_scratch_ft_aug-standard_s813 | orig | 0.572 | 0.613 | 0.486 | 0.671 | 0.499 | 82 |
| fine_vgg16_pt_ft_aug-standard_s813 | blue | 0.833 | 0.902 | 0.486 | 0.329 | 0.311 | 82 |
| fine_vgg16_pt_ft_aug-standard_s813 | gray | 0.573 | 0.659 | 0.486 | 0.524 | 0.507 | 82 |
| fine_vgg16_pt_ft_aug-standard_s813 | orig | 0.541 | 0.549 | 0.486 | 0.793 | 0.763 | 82 |
| fine_vit_b_16_pt_ft_aug-standard_bags-both_s813 | blue | nan | nan | 0.486 | 0.341 | 0.343 | 82 |
| fine_vit_b_16_pt_ft_aug-standard_bags-both_s813 | gray | nan | nan | 0.486 | 0.902 | 0.873 | 82 |
| fine_vit_b_16_pt_ft_aug-standard_bags-both_s813 | orig | nan | nan | 0.486 | 0.927 | 0.878 | 82 |
| fine_vit_b_16_pt_ft_aug-standard_bags-synthetic_s813 | blue | nan | nan | 0.486 | 0.098 | 0.147 | 82 |
| fine_vit_b_16_pt_ft_aug-standard_bags-synthetic_s813 | gray | nan | nan | 0.486 | 0.402 | 0.348 | 82 |
| fine_vit_b_16_pt_ft_aug-standard_bags-synthetic_s813 | orig | nan | nan | 0.486 | 0.256 | 0.207 | 82 |
| fine_vit_b_16_pt_ft_aug-standard_s813 | blue | nan | nan | 0.486 | 0.061 | 0.079 | 82 |
| fine_vit_b_16_pt_ft_aug-standard_s813 | gray | nan | nan | 0.486 | 0.805 | 0.69 | 82 |
| fine_vit_b_16_pt_ft_aug-standard_s813 | orig | nan | nan | 0.486 | 1.0 | 0.99 | 82 |
| fine_vit_b_16_pt_ft_aug-underwater_s813 | blue | nan | nan | 0.486 | 0.0 | 0.022 | 82 |
| fine_vit_b_16_pt_ft_aug-underwater_s813 | gray | nan | nan | 0.486 | 0.683 | 0.546 | 82 |
| fine_vit_b_16_pt_ft_aug-underwater_s813 | orig | nan | nan | 0.486 | 0.756 | 0.727 | 82 |