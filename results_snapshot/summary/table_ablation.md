| config | n_seeds | accuracy | macro_f1 | worst_class_recall | auc | plastic_recall | ece |
|---|---|---|---|---|---|---|---|
| resnet50 | aug=none | 1 | 0.904 | 0.847 | 0.692 | 0.959 | 0.963 | 0.181 |
| resnet50 | aug=standard | 3 | 0.882 ± 0.043 | 0.824 ± 0.034 | 0.564 ± 0.044 | 0.982 ± 0.009 | 0.931 ± 0.069 | 0.109 ± 0.040 |
| resnet50 | aug=underwater | 3 | 0.827 ± 0.023 | 0.765 ± 0.033 | 0.564 ± 0.149 | 0.968 ± 0.013 | 0.858 ± 0.043 | 0.074 ± 0.028 |
| resnet50 | frozen | aug=standard | 1 | 0.833 | 0.706 | 0.231 | 0.954 | 0.976 | 0.434 |
| resnet50 | scratch | aug=standard | 1 | 0.365 | 0.314 | 0.077 | 0.815 | 0.341 | 0.209 |