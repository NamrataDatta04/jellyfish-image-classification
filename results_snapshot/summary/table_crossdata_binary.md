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