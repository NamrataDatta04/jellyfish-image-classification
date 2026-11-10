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