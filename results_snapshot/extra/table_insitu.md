| model | trained with jellyfish from | seeds | test vs Kaggle jellies (macro-F1) | test vs in-situ jellies (macro-F1) | bag on clear bg (acc) | jelly on ROV bg (acc) | ROV bg only → 'bag' |
|---|---|---|---|---|---|---|---|
| CLIP linear probe | both | 1 | 0.917 | 0.915 | 0.561 | 0.969 | 0.841 |
| CLIP linear probe | insitu | 1 | 0.974 | 0.956 | 0.720 | 0.846 | 0.915 |
| CLIP linear probe | kaggle | 1 | 1.000 | 0.777 | 0.793 | 0.600 | 1.000 |
| ResNet-50 | both | 3 | 0.890 ± 0.069 | 0.901 ± 0.057 | 0.207 ± 0.063 | 0.979 ± 0.018 | 0.890 ± 0.159 |
| ResNet-50 | insitu | 3 | 0.852 ± 0.013 | 0.923 ± 0.018 | 0.671 ± 0.085 | 0.815 ± 0.081 | 0.862 ± 0.218 |
| ResNet-50 | kaggle | 3 | 0.940 ± 0.055 | 0.833 ± 0.035 | 0.305 ± 0.095 | 1.000 ± 0.000 | 0.963 ± 0.042 |