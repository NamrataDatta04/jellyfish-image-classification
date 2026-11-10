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