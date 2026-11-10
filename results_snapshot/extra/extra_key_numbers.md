# Key numbers of the new experiments (shortcut study)
Accuracies are for the binary decision 'plastic bag or jellyfish?' (7-class models are mapped to it). mean ± std over seeds; CIs are 95% cluster-bootstrap intervals of the seed-813 model.

## Class x background benchmark — binary models

| method | seeds | test acc | bag ROV | bag clear | jelly clear | jelly ROV | worst-group | worst-group 95% CI | BG-gap | ROV bg only → 'bag' | in-situ jelly (FathomNet) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| DINOv2 ViT-B/14 probe + group-balanced | 1 | 1.000 | 1.000 | 1.000 | 1.000 | 0.969 | 0.969 | [0.92, 1.00] | 0.015 | 0.780 | 0.964 |
| ResNet-50 + DFR | 3 | 0.991 ± 0.010 | 0.984 ± 0.019 | 0.829 ± 0.063 | 1.000 | 0.995 ± 0.009 | 0.829 ± 0.063 | [0.76, 0.96] | 0.080 ± 0.038 | 0.870 ± 0.109 | 0.949 ± 0.026 |
| EfficientNet-B0 + DFR | 3 | 0.983 ± 0.013 | 0.967 ± 0.025 | 0.785 ± 0.071 | 1.000 | 0.979 ± 0.009 | 0.785 ± 0.071 | [0.60, 0.93] | 0.102 ± 0.029 | 0.972 ± 0.025 | 0.940 ± 0.004 |
| ResNet-50 (underwater aug) + DFR | 3 | 0.979 ± 0.004 | 0.959 ± 0.007 | 0.776 ± 0.178 | 1.000 | 1.000 | 0.776 ± 0.178 | [0.69, 0.96] | 0.091 ± 0.085 | 0.715 ± 0.215 | 0.973 ± 0.004 |
| CLIP ViT-B/32 probe + group-balanced | 1 | 0.987 | 0.976 | 0.756 | 1.000 | 1.000 | 0.756 | [0.63, 0.93] | 0.110 | 0.976 | 0.978 |
| EfficientNet-B0 (underwater aug) + DFR | 3 | 0.976 ± 0.010 | 0.955 ± 0.019 | 0.744 ± 0.061 | 1.000 | 0.990 ± 0.009 | 0.744 ± 0.061 | [0.45, 0.88] | 0.111 ± 0.021 | 0.927 ± 0.063 | 0.971 ± 0.013 |
| ViT-B/16 + DFR | 3 | 0.953 ± 0.020 | 0.911 ± 0.037 | 0.309 ± 0.326 | 1.000 | 0.995 ± 0.009 | 0.309 ± 0.326 | [0.38, 0.93] | 0.303 ± 0.172 | 0.679 ± 0.227 | 0.949 ± 0.007 |
| VGG16 + DFR | 1 | 0.974 | 0.951 | 0.280 | 1.000 | 0.985 | 0.280 | [0.08, 0.48] | 0.343 | 0.573 | 0.949 |
| ViT-B/16 (underwater aug) + DFR | 3 | 0.923 ± 0.013 | 0.854 ± 0.024 | 0.069 ± 0.074 | 1.000 | 1.000 | 0.069 ± 0.074 | [0.04, 0.36] | 0.392 ± 0.034 | 0.707 ± 0.129 | 0.971 ± 0.007 |
| TinyCNN + DFR | 1 | 0.904 | 0.829 | 0.000 | 0.986 | 0.877 | 0.000 | [0.00, 0.00] | 0.469 | 0.829 | 0.862 |
| EfficientNet-B0 | 3 | 0.983 ± 0.020 | 0.967 ± 0.037 | 0.593 ± 0.098 | 1.000 | 0.964 ± 0.024 | 0.593 ± 0.098 | [0.35, 0.80] | 0.205 ± 0.075 | 0.992 ± 0.007 | 0.787 ± 0.029 |
| ViT-B/16 | 3 | 0.944 ± 0.048 | 0.894 ± 0.092 | 0.415 ± 0.367 | 1.000 | 0.974 ± 0.018 | 0.415 ± 0.367 | [0.28, 0.81] | 0.253 ± 0.134 | 0.911 ± 0.078 | 0.877 ± 0.026 |
| ResNet-50 | 3 | 0.942 ± 0.051 | 0.890 ± 0.097 | 0.305 ± 0.095 | 1.000 | 1.000 | 0.305 ± 0.095 | [0.25, 0.75] | 0.293 ± 0.043 | 0.967 ± 0.046 | 0.812 ± 0.013 |
| EfficientNet-B0 (underwater aug) | 3 | 0.947 ± 0.035 | 0.898 ± 0.067 | 0.276 ± 0.159 | 1.000 | 0.959 ± 0.009 | 0.276 ± 0.159 | [0.01, 0.26] | 0.331 ± 0.115 | 0.963 ± 0.012 | 0.848 ± 0.014 |
| ResNet-50 (underwater aug) | 3 | 0.895 ± 0.015 | 0.801 ± 0.028 | 0.122 ± 0.021 | 1.000 | 0.995 ± 0.009 | 0.122 ± 0.021 | [0.05, 0.24] | 0.342 ± 0.018 | 0.955 ± 0.049 | 0.872 ± 0.030 |
| VGG16 | 1 | 0.994 | 0.988 | 0.037 | 1.000 | 0.985 | 0.037 | [0.00, 0.08] | 0.483 | 0.902 | 0.826 |
| TinyCNN | 1 | 0.910 | 0.841 | 0.024 | 0.986 | 0.262 | 0.024 | [0.00, 0.07] | 0.771 | 0.841 | 0.841 |
| ViT-B/16 (underwater aug) | 3 | 0.906 ± 0.013 | 0.821 ± 0.025 | 0.000 | 1.000 | 0.949 ± 0.062 | 0.000 | [0.00, 0.00] | 0.436 ± 0.037 | 0.923 ± 0.014 | 0.877 ± 0.074 |
| CLIP ViT-B/32 zero-shot | 1 | 0.949 | 0.988 | 0.988 | 0.905 | 0.785 | 0.785 | [0.68, 0.88] | 0.060 | 0.890 | 0.746 |
| DINOv2 ViT-B/14 linear probe | 1 | 0.994 | 1.000 | 1.000 | 0.986 | 0.708 | 0.708 | [0.60, 0.82] | 0.139 | 1.000 | 0.826 |
| CLIP ViT-B/32 linear probe | 1 | 1.000 | 1.000 | 0.793 | 1.000 | 0.600 | 0.600 | [0.48, 0.71] | 0.304 | 1.000 | 0.645 |

## Normal test set with confidence intervals — binary models

| method | seeds | test macro-F1 (bag vs jelly) | 95% CI |
|---|---|---|---|
| DINOv2 ViT-B/14 probe + group-balanced | 1 | 1.000 | [1.00, 1.00] |
| ResNet-50 + DFR | 3 | 0.991 ± 0.010 | [0.96, 1.00] |
| EfficientNet-B0 + DFR | 3 | 0.983 ± 0.013 | [0.96, 1.00] |
| ResNet-50 (underwater aug) + DFR | 3 | 0.979 ± 0.004 | [0.94, 1.00] |
| CLIP ViT-B/32 probe + group-balanced | 1 | 0.987 | [0.97, 1.00] |
| EfficientNet-B0 (underwater aug) + DFR | 3 | 0.976 ± 0.010 | [0.92, 1.00] |
| ViT-B/16 + DFR | 3 | 0.953 ± 0.020 | [0.85, 1.00] |
| VGG16 + DFR | 1 | 0.974 | [0.92, 1.00] |
| ViT-B/16 (underwater aug) + DFR | 3 | 0.923 ± 0.013 | [0.79, 1.00] |
| TinyCNN + DFR | 1 | 0.904 | [0.71, 1.00] |
| EfficientNet-B0 | 3 | 0.983 ± 0.020 | [1.00, 1.00] |
| ViT-B/16 | 3 | 0.944 ± 0.048 | [0.82, 1.00] |
| ResNet-50 | 3 | 0.942 ± 0.051 | [0.94, 1.00] |
| EfficientNet-B0 (underwater aug) | 3 | 0.947 ± 0.035 | [0.97, 1.00] |
| ResNet-50 (underwater aug) | 3 | 0.895 ± 0.015 | [0.71, 0.99] |
| VGG16 | 1 | 0.994 | [0.98, 1.00] |
| TinyCNN | 1 | 0.910 | [0.75, 1.00] |
| ViT-B/16 (underwater aug) | 3 | 0.906 ± 0.013 | [0.75, 1.00] |
| CLIP ViT-B/32 zero-shot | 1 | 0.948 | [0.90, 0.98] |
| DINOv2 ViT-B/14 linear probe | 1 | 0.994 | [0.98, 1.00] |
| CLIP ViT-B/32 linear probe | 1 | 1.000 | [1.00, 1.00] |

## Unseen objects (out-of-distribution) — binary models

| method | pred_plastic | auroc_msp | auroc_energy | auroc_knn |
|---|---|---|---|---|
| DINOv2 ViT-B/14 probe + group-balanced | 0.837 | 0.696 | - | 0.891 |
| ResNet-50 + DFR | 0.854 ± 0.065 | 0.657 ± 0.124 | - | 0.591 ± 0.021 |
| EfficientNet-B0 + DFR | 0.821 ± 0.014 | 0.693 ± 0.074 | - | 0.577 ± 0.028 |
| ResNet-50 (underwater aug) + DFR | 0.838 ± 0.023 | 0.720 ± 0.028 | - | 0.698 ± 0.044 |
| CLIP ViT-B/32 probe + group-balanced | 0.935 | 0.725 | - | 0.614 |
| EfficientNet-B0 (underwater aug) + DFR | 0.789 ± 0.013 | 0.804 ± 0.023 | - | 0.642 ± 0.024 |
| ViT-B/16 + DFR | 0.826 ± 0.019 | 0.675 ± 0.096 | - | 0.746 ± 0.037 |
| VGG16 + DFR | 0.808 | 0.745 | - | 0.651 |
| ViT-B/16 (underwater aug) + DFR | 0.784 ± 0.015 | 0.723 ± 0.049 | - | 0.729 ± 0.049 |
| TinyCNN + DFR | 0.728 | 0.733 | - | 0.351 |
| EfficientNet-B0 | 0.843 ± 0.032 | 0.679 ± 0.054 | 0.680 ± 0.069 | 0.577 ± 0.028 |
| ViT-B/16 | 0.843 ± 0.031 | 0.693 ± 0.146 | 0.700 ± 0.098 | 0.746 ± 0.037 |
| ResNet-50 | 0.837 ± 0.050 | 0.554 ± 0.029 | 0.557 ± 0.022 | 0.591 ± 0.021 |
| EfficientNet-B0 (underwater aug) | 0.745 ± 0.037 | 0.718 ± 0.042 | 0.723 ± 0.045 | 0.642 ± 0.024 |
| ResNet-50 (underwater aug) | 0.727 ± 0.012 | 0.686 ± 0.052 | 0.673 ± 0.044 | 0.698 ± 0.044 |
| VGG16 | 0.790 | 0.711 | 0.704 | 0.651 |
| TinyCNN | 0.859 | 0.566 | 0.617 | 0.351 |
| ViT-B/16 (underwater aug) | 0.780 ± 0.030 | 0.680 ± 0.094 | 0.675 ± 0.093 | 0.729 ± 0.049 |
| CLIP ViT-B/32 zero-shot | 0.906 | 0.612 | - | 0.614 |
| DINOv2 ViT-B/14 linear probe | 1.000 | 0.793 | - | 0.891 |
| CLIP ViT-B/32 linear probe | 0.975 | 0.627 | - | 0.614 |

## Class x background benchmark — fine models

| method | seeds | test acc | bag ROV | bag clear | jelly clear | jelly ROV | worst-group | worst-group 95% CI | BG-gap | ROV bg only → 'bag' | in-situ jelly (FathomNet) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| EfficientNet-B0 (underwater aug) + DFR | 3 | 0.987 | 1.000 | 0.992 ± 0.014 | 0.973 | 0.949 ± 0.032 | 0.945 ± 0.026 | [0.93, 1.00] | 0.016 ± 0.022 | 1.000 | 0.543 ± 0.038 |
| DINOv2 ViT-B/14 probe + group-balanced | 1 | 1.000 | 1.000 | 1.000 | 1.000 | 0.938 | 0.938 | [0.88, 0.98] | 0.031 | 0.976 | 0.884 |
| ResNet-50 + DFR | 3 | 0.998 ± 0.004 | 1.000 | 0.935 ± 0.028 | 0.995 ± 0.008 | 0.944 ± 0.018 | 0.926 ± 0.024 | [0.88, 0.98] | 0.058 ± 0.010 | 1.000 | 0.645 ± 0.095 |
| ResNet-50 (underwater aug) + DFR | 3 | 0.987 ± 0.011 | 0.988 ± 0.012 | 0.870 ± 0.086 | 0.986 ± 0.014 | 0.954 ± 0.027 | 0.861 ± 0.073 | [0.85, 0.97] | 0.075 ± 0.031 | 0.996 ± 0.007 | 0.652 ± 0.094 |
| EfficientNet-B0 + DFR | 3 | 0.989 ± 0.013 | 0.980 ± 0.025 | 0.841 ± 0.127 | 1.000 | 0.949 ± 0.024 | 0.820 ± 0.090 | [0.70, 0.92] | 0.095 ± 0.044 | 0.996 ± 0.007 | 0.698 ± 0.092 |
| VGG16 + DFR | 1 | 0.981 | 1.000 | 0.817 | 0.959 | 0.954 | 0.817 | [0.64, 0.95] | 0.094 | 0.976 | 0.710 |
| CLIP ViT-B/32 probe + group-balanced | 1 | 0.974 | 0.951 | 0.598 | 1.000 | 0.969 | 0.598 | [0.35, 0.88] | 0.192 | 1.000 | 0.761 |
| ViT-B/16 + DFR | 3 | 1.000 | 1.000 | 0.541 ± 0.201 | 1.000 | 0.990 ± 0.018 | 0.541 ± 0.201 | [0.10, 0.60] | 0.235 ± 0.109 | 0.996 ± 0.007 | 0.780 ± 0.066 |
| ViT-B/16 (underwater aug) + DFR | 3 | 0.991 ± 0.010 | 0.984 ± 0.019 | 0.500 ± 0.328 | 1.000 | 0.954 | 0.500 ± 0.328 | [0.03, 0.23] | 0.265 ± 0.155 | 0.984 ± 0.007 | 0.862 ± 0.059 |
| ResNet-50 [scratch] + DFR | 1 | 0.801 | 0.732 | 0.183 | 0.878 | 0.708 | 0.183 | [0.02, 0.42] | 0.360 | 0.695 | 0.819 |
| TinyCNN + DFR | 1 | 0.897 | 0.841 | 0.171 | 0.959 | 0.723 | 0.171 | [0.02, 0.37] | 0.454 | 0.951 | 0.739 |
| EfficientNet-B0 | 3 | 0.959 ± 0.049 | 0.923 ± 0.092 | 0.602 ± 0.088 | 1.000 | 0.979 ± 0.024 | 0.602 ± 0.088 | [0.33, 0.81] | 0.171 ± 0.084 | 0.963 ± 0.024 | 0.831 ± 0.022 |
| ResNet-50 | 3 | 0.964 ± 0.036 | 0.931 ± 0.069 | 0.480 ± 0.102 | 1.000 | 0.995 ± 0.009 | 0.480 ± 0.102 | [0.33, 0.85] | 0.228 ± 0.015 | 0.919 ± 0.088 | 0.792 ± 0.048 |
| VGG16 | 1 | 0.885 | 0.780 | 0.341 | 1.000 | 1.000 | 0.341 | [0.20, 0.53] | 0.220 | 0.927 | 0.870 |
| ResNet-50 (underwater aug) | 3 | 0.927 ± 0.023 | 0.862 ± 0.043 | 0.280 ± 0.127 | 1.000 | 0.985 | 0.280 ± 0.127 | [0.06, 0.42] | 0.298 ± 0.049 | 0.951 ± 0.032 | 0.872 ± 0.054 |
| ResNet-50 [scratch] | 1 | 0.641 | 0.341 | 0.256 | 0.973 | 0.846 | 0.256 | [0.07, 0.44] | 0.106 | 0.293 | 0.957 |
| EfficientNet-B0 (underwater aug) | 3 | 0.927 ± 0.043 | 0.862 ± 0.081 | 0.167 ± 0.035 | 1.000 | 0.985 ± 0.015 | 0.167 ± 0.035 | [0.03, 0.36] | 0.355 ± 0.051 | 0.951 ± 0.032 | 0.894 ± 0.025 |
| ViT-B/16 | 3 | 0.991 ± 0.015 | 0.984 ± 0.028 | 0.163 ± 0.104 | 1.000 | 1.000 | 0.163 ± 0.104 | [0.01, 0.13] | 0.411 ± 0.053 | 0.980 ± 0.035 | 0.865 ± 0.042 |
| ViT-B/16 (underwater aug) | 3 | 0.942 ± 0.067 | 0.890 ± 0.127 | 0.081 ± 0.071 | 1.000 | 0.954 ± 0.015 | 0.081 ± 0.071 | [0.00, 0.00] | 0.428 ± 0.035 | 0.967 ± 0.031 | 0.911 ± 0.052 |
| TinyCNN | 1 | 0.821 | 0.671 | 0.049 | 0.986 | 0.738 | 0.049 | [0.00, 0.15] | 0.435 | 0.671 | 0.957 |
| DINOv2 ViT-B/14 linear probe | 1 | 1.000 | 1.000 | 1.000 | 1.000 | 0.815 | 0.815 | [0.72, 0.91] | 0.092 | 1.000 | 0.826 |
| CLIP ViT-B/32 linear probe | 1 | 0.994 | 0.988 | 0.768 | 1.000 | 0.877 | 0.768 | [0.51, 0.93] | 0.171 | 1.000 | 0.667 |
| CLIP ViT-B/32 zero-shot | 1 | 0.840 | 0.732 | 0.902 | 0.959 | 0.892 | 0.732 | [0.49, 0.89] | -0.052 | 0.573 | 0.877 |

## Normal test set with confidence intervals — fine models

| method | seeds | test macro-F1 (bag vs jelly) | 95% CI | test macro-F1 (7 classes) |
|---|---|---|---|---|
| EfficientNet-B0 (underwater aug) + DFR | 3 | 0.987 | [0.96, 1.00] | 0.847 ± 0.019 |
| DINOv2 ViT-B/14 probe + group-balanced | 1 | 1.000 | [1.00, 1.00] | 0.825 |
| ResNet-50 + DFR | 3 | 0.998 ± 0.004 | [1.00, 1.00] | 0.891 ± 0.015 |
| ResNet-50 (underwater aug) + DFR | 3 | 0.987 ± 0.011 | [0.95, 1.00] | 0.807 ± 0.017 |
| EfficientNet-B0 + DFR | 3 | 0.989 ± 0.013 | [0.98, 1.00] | 0.850 ± 0.026 |
| VGG16 + DFR | 1 | 0.981 | [0.95, 1.00] | 0.793 |
| CLIP ViT-B/32 probe + group-balanced | 1 | 0.974 | [0.93, 1.00] | 0.718 |
| ViT-B/16 + DFR | 3 | 1.000 | [1.00, 1.00] | 0.871 ± 0.010 |
| ViT-B/16 (underwater aug) + DFR | 3 | 0.991 ± 0.010 | [0.96, 1.00] | 0.857 ± 0.007 |
| ResNet-50 [scratch] + DFR | 1 | 0.801 | [0.65, 0.93] | 0.413 |
| TinyCNN + DFR | 1 | 0.897 | [0.80, 0.98] | 0.498 |
| EfficientNet-B0 | 3 | 0.959 ± 0.049 | [0.94, 1.00] | 0.837 ± 0.026 |
| ResNet-50 | 3 | 0.964 ± 0.036 | [0.94, 1.00] | 0.824 ± 0.034 |
| VGG16 | 1 | 0.884 | [0.76, 1.00] | 0.737 |
| ResNet-50 (underwater aug) | 3 | 0.927 ± 0.023 | [0.79, 0.99] | 0.766 ± 0.036 |
| ResNet-50 [scratch] | 1 | 0.610 | [0.42, 0.89] | 0.314 |
| EfficientNet-B0 (underwater aug) | 3 | 0.927 ± 0.043 | [0.92, 0.99] | 0.750 ± 0.010 |
| ViT-B/16 | 3 | 0.991 ± 0.015 | [1.00, 1.00] | 0.849 ± 0.007 |
| ViT-B/16 (underwater aug) | 3 | 0.942 ± 0.067 | [0.71, 1.00] | 0.785 ± 0.042 |
| TinyCNN | 1 | 0.818 | [0.61, 0.95] | 0.450 |
| DINOv2 ViT-B/14 linear probe | 1 | 1.000 | [1.00, 1.00] | 0.877 |
| CLIP ViT-B/32 linear probe | 1 | 0.994 | [0.98, 1.00] | 0.775 |
| CLIP ViT-B/32 zero-shot | 1 | 0.839 | [0.69, 0.96] | 0.539 |

## Unseen objects (out-of-distribution) — fine models

| method | pred_plastic | auroc_msp | auroc_energy | auroc_knn |
|---|---|---|---|---|
| EfficientNet-B0 (underwater aug) + DFR | 0.942 ± 0.013 | 0.486 ± 0.016 | - | 0.694 ± 0.018 |
| DINOv2 ViT-B/14 probe + group-balanced | 0.946 | 0.534 | - | 0.891 |
| ResNet-50 + DFR | 0.895 ± 0.013 | 0.508 ± 0.052 | - | 0.641 ± 0.021 |
| ResNet-50 (underwater aug) + DFR | 0.880 ± 0.017 | 0.492 ± 0.013 | - | 0.699 ± 0.006 |
| EfficientNet-B0 + DFR | 0.906 ± 0.030 | 0.463 ± 0.018 | - | 0.646 ± 0.035 |
| VGG16 + DFR | 0.917 | 0.409 | - | 0.400 |
| CLIP ViT-B/32 probe + group-balanced | 0.909 | 0.366 | - | 0.614 |
| ViT-B/16 + DFR | 0.952 ± 0.010 | 0.426 ± 0.035 | - | 0.622 ± 0.042 |
| ViT-B/16 (underwater aug) + DFR | 0.953 ± 0.010 | 0.503 ± 0.043 | - | 0.644 ± 0.051 |
| ResNet-50 [scratch] + DFR | 0.583 | 0.431 | - | 0.541 |
| TinyCNN + DFR | 0.670 | 0.505 | - | 0.476 |
| EfficientNet-B0 | 0.806 ± 0.021 | 0.633 ± 0.008 | 0.677 ± 0.017 | 0.646 ± 0.035 |
| ResNet-50 | 0.831 ± 0.048 | 0.535 ± 0.031 | 0.571 ± 0.035 | 0.641 ± 0.021 |
| VGG16 | 0.815 | 0.420 | 0.458 | 0.400 |
| ResNet-50 (underwater aug) | 0.671 ± 0.051 | 0.601 ± 0.016 | 0.615 ± 0.020 | 0.699 ± 0.006 |
| ResNet-50 [scratch] | 0.438 | 0.460 | 0.500 | 0.541 |
| EfficientNet-B0 (underwater aug) | 0.581 ± 0.060 | 0.684 ± 0.020 | 0.718 ± 0.015 | 0.694 ± 0.018 |
| ViT-B/16 | 0.857 ± 0.025 | 0.588 ± 0.024 | 0.632 ± 0.043 | 0.622 ± 0.042 |
| ViT-B/16 (underwater aug) | 0.853 ± 0.073 | 0.595 ± 0.036 | 0.628 ± 0.041 | 0.644 ± 0.051 |
| TinyCNN | 0.460 | 0.421 | 0.423 | 0.476 |
| DINOv2 ViT-B/14 linear probe | 0.996 | 0.487 | - | 0.891 |
| CLIP ViT-B/32 linear probe | 0.978 | 0.333 | - | 0.614 |
| CLIP ViT-B/32 zero-shot | 0.641 | 0.456 | - | 0.614 |

## McNemar tests with Holm correction

| family | a | b | only_a_correct | only_b_correct | p | p_holm | significant (Holm, 5%) |
|---|---|---|---|---|---|---|---|
| ERM vs DFR (group benchmark) | binary: EfficientNet-B0 + DFR (seed 813) | EfficientNet-B0 (ERM) | 24 | 3 | 0.0 | 0.0006 | True |
| ERM vs DFR (group benchmark) | binary: EfficientNet-B0 + DFR (seed 814) | EfficientNet-B0 (ERM) | 17 | 8 | 0.1078 | 0.3233 | False |
| ERM vs DFR (group benchmark) | binary: EfficientNet-B0 + DFR (seed 815) | EfficientNet-B0 (ERM) | 25 | 5 | 0.0003 | 0.0026 | True |
| ERM vs DFR (group benchmark) | binary: EfficientNet-B0 (underwater aug) + DFR (seed 813) | EfficientNet-B0 (underwater aug) (ERM) | 53 | 4 | 0.0 | 0.0 | True |
| ERM vs DFR (group benchmark) | binary: EfficientNet-B0 (underwater aug) + DFR (seed 814) | EfficientNet-B0 (underwater aug) (ERM) | 37 | 2 | 0.0 | 0.0 | True |
| ERM vs DFR (group benchmark) | binary: EfficientNet-B0 (underwater aug) + DFR (seed 815) | EfficientNet-B0 (underwater aug) (ERM) | 52 | 1 | 0.0 | 0.0 | True |
| ERM vs DFR (group benchmark) | binary: ResNet-50 + DFR (seed 813) | ResNet-50 (ERM) | 38 | 1 | 0.0 | 0.0 | True |
| ERM vs DFR (group benchmark) | binary: ResNet-50 + DFR (seed 814) | ResNet-50 (ERM) | 60 | 1 | 0.0 | 0.0 | True |
| ERM vs DFR (group benchmark) | binary: ResNet-50 + DFR (seed 815) | ResNet-50 (ERM) | 57 | 2 | 0.0 | 0.0 | True |
| ERM vs DFR (group benchmark) | binary: ResNet-50 (underwater aug) + DFR (seed 813) | ResNet-50 (underwater aug) (ERM) | 76 | 0 | 0.0 | 0.0 | True |
| ERM vs DFR (group benchmark) | binary: ResNet-50 (underwater aug) + DFR (seed 814) | ResNet-50 (underwater aug) (ERM) | 76 | 1 | 0.0 | 0.0 | True |
| ERM vs DFR (group benchmark) | binary: ResNet-50 (underwater aug) + DFR (seed 815) | ResNet-50 (underwater aug) (ERM) | 54 | 4 | 0.0 | 0.0 | True |
| ERM vs DFR (group benchmark) | binary: TinyCNN + DFR (seed 813) | TinyCNN (ERM) | 41 | 4 | 0.0 | 0.0 | True |
| ERM vs DFR (group benchmark) | binary: VGG16 + DFR (seed 813) | VGG16 (ERM) | 22 | 5 | 0.0015 | 0.0106 | True |
| ERM vs DFR (group benchmark) | binary: ViT-B/16 + DFR (seed 813) | ViT-B/16 (ERM) | 19 | 5 | 0.0066 | 0.0397 | True |
| ERM vs DFR (group benchmark) | binary: ViT-B/16 + DFR (seed 814) | ViT-B/16 (ERM) | 3 | 50 | 0.0 | 0.0 | True |
| ERM vs DFR (group benchmark) | binary: ViT-B/16 + DFR (seed 815) | ViT-B/16 (ERM) | 15 | 0 | 0.0001 | 0.0007 | True |
| ERM vs DFR (group benchmark) | binary: ViT-B/16 (underwater aug) + DFR (seed 813) | ViT-B/16 (underwater aug) (ERM) | 24 | 2 | 0.0 | 0.0001 | True |
| ERM vs DFR (group benchmark) | binary: ViT-B/16 (underwater aug) + DFR (seed 814) | ViT-B/16 (underwater aug) (ERM) | 6 | 2 | 0.2891 | 0.5781 | False |
| ERM vs DFR (group benchmark) | binary: ViT-B/16 (underwater aug) + DFR (seed 815) | ViT-B/16 (underwater aug) (ERM) | 13 | 4 | 0.049 | 0.1962 | False |
| ERM vs DFR (group benchmark) | fine: EfficientNet-B0 + DFR (seed 813) | EfficientNet-B0 (ERM) | 26 | 1 | 0.0 | 0.0 | True |
| ERM vs DFR (group benchmark) | fine: EfficientNet-B0 + DFR (seed 814) | EfficientNet-B0 (ERM) | 29 | 5 | 0.0 | 0.0005 | True |
| ERM vs DFR (group benchmark) | fine: EfficientNet-B0 + DFR (seed 815) | EfficientNet-B0 (ERM) | 20 | 2 | 0.0001 | 0.0011 | True |
| ERM vs DFR (group benchmark) | fine: EfficientNet-B0 (underwater aug) + DFR (seed 813) | EfficientNet-B0 (underwater aug) (ERM) | 77 | 3 | 0.0 | 0.0 | True |
| ERM vs DFR (group benchmark) | fine: EfficientNet-B0 (underwater aug) + DFR (seed 814) | EfficientNet-B0 (underwater aug) (ERM) | 76 | 6 | 0.0 | 0.0 | True |
| ERM vs DFR (group benchmark) | fine: EfficientNet-B0 (underwater aug) + DFR (seed 815) | EfficientNet-B0 (underwater aug) (ERM) | 85 | 5 | 0.0 | 0.0 | True |
| ERM vs DFR (group benchmark) | fine: ResNet-50 + DFR (seed 813) | ResNet-50 (ERM) | 41 | 3 | 0.0 | 0.0 | True |
| ERM vs DFR (group benchmark) | fine: ResNet-50 + DFR (seed 814) | ResNet-50 (ERM) | 56 | 4 | 0.0 | 0.0 | True |
| ERM vs DFR (group benchmark) | fine: ResNet-50 + DFR (seed 815) | ResNet-50 (ERM) | 33 | 5 | 0.0 | 0.0001 | True |
| ERM vs DFR (group benchmark) | fine: ResNet-50 (underwater aug) + DFR (seed 813) | ResNet-50 (underwater aug) (ERM) | 75 | 7 | 0.0 | 0.0 | True |
| ERM vs DFR (group benchmark) | fine: ResNet-50 (underwater aug) + DFR (seed 814) | ResNet-50 (underwater aug) (ERM) | 58 | 3 | 0.0 | 0.0 | True |
| ERM vs DFR (group benchmark) | fine: ResNet-50 (underwater aug) + DFR (seed 815) | ResNet-50 (underwater aug) (ERM) | 46 | 2 | 0.0 | 0.0 | True |
| ERM vs DFR (group benchmark) | fine: ResNet-50 [scratch] + DFR (seed 813) | ResNet-50 [scratch] (ERM) | 49 | 39 | 0.3374 | 0.5781 | False |
| ERM vs DFR (group benchmark) | fine: TinyCNN + DFR (seed 813) | TinyCNN (ERM) | 46 | 25 | 0.017 | 0.085 | False |
| ERM vs DFR (group benchmark) | fine: VGG16 + DFR (seed 813) | VGG16 (ERM) | 57 | 6 | 0.0 | 0.0 | True |
| ERM vs DFR (group benchmark) | fine: ViT-B/16 + DFR (seed 813) | ViT-B/16 (ERM) | 21 | 2 | 0.0001 | 0.0007 | True |
| ERM vs DFR (group benchmark) | fine: ViT-B/16 + DFR (seed 814) | ViT-B/16 (ERM) | 49 | 0 | 0.0 | 0.0 | True |
| ERM vs DFR (group benchmark) | fine: ViT-B/16 + DFR (seed 815) | ViT-B/16 (ERM) | 27 | 0 | 0.0 | 0.0 | True |
| ERM vs DFR (group benchmark) | fine: ViT-B/16 (underwater aug) + DFR (seed 813) | ViT-B/16 (underwater aug) (ERM) | 28 | 1 | 0.0 | 0.0 | True |
| ERM vs DFR (group benchmark) | fine: ViT-B/16 (underwater aug) + DFR (seed 814) | ViT-B/16 (underwater aug) (ERM) | 51 | 1 | 0.0 | 0.0 | True |
| ERM vs DFR (group benchmark) | fine: ViT-B/16 (underwater aug) + DFR (seed 815) | ViT-B/16 (underwater aug) (ERM) | 50 | 1 | 0.0 | 0.0 | True |
| best vs others, normal test (binary) | CLIP ViT-B/32 linear probe | CLIP ViT-B/32 probe + group-balanced | 2 | 0 | 0.5 | 1.0 | False |
| best vs others, normal test (binary) | CLIP ViT-B/32 linear probe | CLIP ViT-B/32 zero-shot | 8 | 0 | 0.0078 | 0.1016 | False |
| best vs others, normal test (binary) | CLIP ViT-B/32 linear probe | DINOv2 ViT-B/14 linear probe | 1 | 0 | 1.0 | 1.0 | False |
| best vs others, normal test (binary) | CLIP ViT-B/32 linear probe | DINOv2 ViT-B/14 probe + group-balanced | 0 | 0 | 1.0 | 1.0 | False |
| best vs others, normal test (binary) | CLIP ViT-B/32 linear probe | EfficientNet-B0 + DFR | 2 | 0 | 0.5 | 1.0 | False |
| best vs others, normal test (binary) | CLIP ViT-B/32 linear probe | EfficientNet-B0 | 0 | 0 | 1.0 | 1.0 | False |
| best vs others, normal test (binary) | CLIP ViT-B/32 linear probe | EfficientNet-B0 (underwater aug) + DFR | 4 | 0 | 0.125 | 1.0 | False |
| best vs others, normal test (binary) | CLIP ViT-B/32 linear probe | EfficientNet-B0 (underwater aug) | 2 | 0 | 0.5 | 1.0 | False |
| best vs others, normal test (binary) | CLIP ViT-B/32 linear probe | ResNet-50 + DFR | 3 | 0 | 0.25 | 1.0 | False |
| best vs others, normal test (binary) | CLIP ViT-B/32 linear probe | ResNet-50 | 3 | 0 | 0.25 | 1.0 | False |
| best vs others, normal test (binary) | CLIP ViT-B/32 linear probe | ResNet-50 (underwater aug) + DFR | 3 | 0 | 0.25 | 1.0 | False |
| best vs others, normal test (binary) | CLIP ViT-B/32 linear probe | ResNet-50 (underwater aug) | 19 | 0 | 0.0 | 0.0001 | True |
| best vs others, normal test (binary) | CLIP ViT-B/32 linear probe | TinyCNN + DFR | 15 | 0 | 0.0001 | 0.0012 | True |
| best vs others, normal test (binary) | CLIP ViT-B/32 linear probe | TinyCNN | 14 | 0 | 0.0001 | 0.0022 | True |
| best vs others, normal test (binary) | CLIP ViT-B/32 linear probe | VGG16 + DFR | 4 | 0 | 0.125 | 1.0 | False |
| best vs others, normal test (binary) | CLIP ViT-B/32 linear probe | VGG16 | 1 | 0 | 1.0 | 1.0 | False |
| best vs others, normal test (binary) | CLIP ViT-B/32 linear probe | ViT-B/16 + DFR | 10 | 0 | 0.002 | 0.0293 | True |
| best vs others, normal test (binary) | CLIP ViT-B/32 linear probe | ViT-B/16 | 9 | 0 | 0.0039 | 0.0547 | False |
| best vs others, normal test (binary) | CLIP ViT-B/32 linear probe | ViT-B/16 (underwater aug) + DFR | 12 | 0 | 0.0005 | 0.0078 | True |
| best vs others, normal test (binary) | CLIP ViT-B/32 linear probe | ViT-B/16 (underwater aug) | 14 | 0 | 0.0001 | 0.0022 | True |
| best vs others, normal test (fine) | ViT-B/16 + DFR | CLIP ViT-B/32 linear probe | 1 | 0 | 1.0 | 1.0 | False |
| best vs others, normal test (fine) | ViT-B/16 + DFR | CLIP ViT-B/32 probe + group-balanced | 4 | 0 | 0.125 | 1.0 | False |
| best vs others, normal test (fine) | ViT-B/16 + DFR | CLIP ViT-B/32 zero-shot | 25 | 0 | 0.0 | 0.0 | True |
| best vs others, normal test (fine) | ViT-B/16 + DFR | DINOv2 ViT-B/14 linear probe | 0 | 0 | 1.0 | 1.0 | False |
| best vs others, normal test (fine) | ViT-B/16 + DFR | DINOv2 ViT-B/14 probe + group-balanced | 0 | 0 | 1.0 | 1.0 | False |
| best vs others, normal test (fine) | ViT-B/16 + DFR | EfficientNet-B0 + DFR | 1 | 0 | 1.0 | 1.0 | False |
| best vs others, normal test (fine) | ViT-B/16 + DFR | EfficientNet-B0 | 3 | 0 | 0.25 | 1.0 | False |
| best vs others, normal test (fine) | ViT-B/16 + DFR | EfficientNet-B0 (underwater aug) + DFR | 2 | 0 | 0.5 | 1.0 | False |
| best vs others, normal test (fine) | ViT-B/16 + DFR | EfficientNet-B0 (underwater aug) | 7 | 0 | 0.0156 | 0.2188 | False |
| best vs others, normal test (fine) | ViT-B/16 + DFR | ResNet-50 + DFR | 0 | 0 | 1.0 | 1.0 | False |
| best vs others, normal test (fine) | ViT-B/16 + DFR | ResNet-50 | 4 | 0 | 0.125 | 1.0 | False |
| best vs others, normal test (fine) | ViT-B/16 + DFR | ResNet-50 (underwater aug) + DFR | 3 | 0 | 0.25 | 1.0 | False |
| best vs others, normal test (fine) | ViT-B/16 + DFR | ResNet-50 (underwater aug) | 15 | 0 | 0.0001 | 0.0009 | True |
| best vs others, normal test (fine) | ViT-B/16 + DFR | ResNet-50 [scratch] + DFR | 31 | 0 | 0.0 | 0.0 | True |
| best vs others, normal test (fine) | ViT-B/16 + DFR | ResNet-50 [scratch] | 56 | 0 | 0.0 | 0.0 | True |
| best vs others, normal test (fine) | ViT-B/16 + DFR | TinyCNN + DFR | 16 | 0 | 0.0 | 0.0005 | True |
| best vs others, normal test (fine) | ViT-B/16 + DFR | TinyCNN | 28 | 0 | 0.0 | 0.0 | True |
| best vs others, normal test (fine) | ViT-B/16 + DFR | VGG16 + DFR | 3 | 0 | 0.25 | 1.0 | False |
| best vs others, normal test (fine) | ViT-B/16 + DFR | VGG16 | 18 | 0 | 0.0 | 0.0001 | True |
| best vs others, normal test (fine) | ViT-B/16 + DFR | ViT-B/16 | 0 | 0 | 1.0 | 1.0 | False |
| best vs others, normal test (fine) | ViT-B/16 + DFR | ViT-B/16 (underwater aug) + DFR | 3 | 0 | 0.25 | 1.0 | False |
| best vs others, normal test (fine) | ViT-B/16 + DFR | ViT-B/16 (underwater aug) | 21 | 0 | 0.0 | 0.0 | True |

## Cost-sensitive decisions (clean-up robot)

| method | cost ratio r (jelly harm / missed bag) | cost / object, normal test | cost / object, shifted backgrounds | bags collected (shifted) | jellyfish grabbed (shifted) |
|---|---|---|---|---|---|
| ResNet-50 | 1.0 | 0.058 | 0.218 | 0.598 | 0.0 |
| ResNet-50 | 5.0 | 0.207 | 0.344 | 0.364 | 0.0 |
| ResNet-50 + DFR | 1.0 | 0.009 | 0.052 | 0.907 | 0.002 |
| ResNet-50 + DFR | 5.0 | 0.036 | 0.11 | 0.797 | 0.0 |
| CLIP ViT-B/32 zero-shot | 1.0 | 0.051 | 0.076 | 0.988 | 0.151 |
| CLIP ViT-B/32 zero-shot | 5.0 | 0.135 | 0.152 | 0.872 | 0.036 |
| DINOv2 ViT-B/14 probe + group-balanced | 1.0 | 0.0 | 0.007 | 1.0 | 0.014 |
| DINOv2 ViT-B/14 probe + group-balanced | 5.0 | 0.0 | 0.02 | 0.994 | 0.007 |
| EfficientNet-B0 | 1.0 | 0.017 | 0.127 | 0.78 | 0.017 |
| EfficientNet-B0 | 5.0 | 0.103 | 0.251 | 0.547 | 0.002 |
| EfficientNet-B0 + DFR | 1.0 | 0.017 | 0.072 | 0.876 | 0.01 |
| EfficientNet-B0 + DFR | 5.0 | 0.038 | 0.142 | 0.758 | 0.005 |
| CLIP ViT-B/32 linear probe | 1.0 | 0.0 | 0.142 | 0.896 | 0.187 |
| CLIP ViT-B/32 linear probe | 5.0 | 0.513 | 0.535 | 0.012 | 0.0 |
| DINOv2 ViT-B/14 linear probe | 1.0 | 0.006 | 0.066 | 1.0 | 0.144 |
| DINOv2 ViT-B/14 linear probe | 5.0 | 0.276 | 0.406 | 0.25 | 0.0 |

## Low-level cue test (why the underwater augmentation failed)

| condition | test_accuracy | test_auc |
|---|---|---|
| no augmentation | 0.936 | 0.989 |
| underwater augmentation on BOTH classes | 0.833 | 0.928 |

## Same-camera test with FathomNet in-situ jellyfish

| model | trained with jellyfish from | seeds | test vs Kaggle jellies (macro-F1) | test vs in-situ jellies (macro-F1) | bag on clear bg (acc) | jelly on ROV bg (acc) | ROV bg only → 'bag' |
|---|---|---|---|---|---|---|---|
| CLIP linear probe | both | 1 | 0.917 | 0.915 | 0.561 | 0.969 | 0.841 |
| CLIP linear probe | insitu | 1 | 0.974 | 0.956 | 0.720 | 0.846 | 0.915 |
| CLIP linear probe | kaggle | 1 | 1.000 | 0.777 | 0.793 | 0.600 | 1.000 |
| ResNet-50 | both | 3 | 0.890 ± 0.069 | 0.901 ± 0.057 | 0.207 ± 0.063 | 0.979 ± 0.018 | 0.890 ± 0.159 |
| ResNet-50 | insitu | 3 | 0.852 ± 0.013 | 0.923 ± 0.018 | 0.671 ± 0.085 | 0.815 ± 0.081 | 0.862 ± 0.218 |
| ResNet-50 | kaggle | 3 | 0.940 ± 0.055 | 0.833 ± 0.035 | 0.305 ± 0.095 | 1.000 ± 0.000 | 0.963 ± 0.042 |