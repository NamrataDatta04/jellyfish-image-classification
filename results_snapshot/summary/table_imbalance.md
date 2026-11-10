| config | imbalance_method | n_seeds | macro_f1 | balanced_accuracy | worst_class_recall | plastic_recall | ece |
|---|---|---|---|---|---|---|---|
| resnet50 | aug=standard | imb=none | none | 1 | 0.857 | 0.859 | 0.615 | 1.000 | 0.049 |
| resnet50 | aug=standard | weighted | 3 | 0.824 ± 0.034 | 0.841 ± 0.026 | 0.564 ± 0.044 | 0.931 ± 0.069 | 0.109 ± 0.040 |
| resnet50 | aug=standard | imb=focal | focal | 1 | 0.793 | 0.808 | 0.538 | 0.927 | 0.217 |
| resnet50 | aug=standard | imb=undersample | undersample | 1 | 0.789 | 0.858 | 0.538 | 0.768 | 0.047 |
| resnet50 | aug=standard | imb=combo | combo | 1 | 0.744 | 0.786 | 0.385 | 0.890 | 0.206 |
| resnet50 | aug=standard | imb=oversample | oversample | 1 | 0.681 | 0.736 | 0.500 | 0.720 | 0.072 |