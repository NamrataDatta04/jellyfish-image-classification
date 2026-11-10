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