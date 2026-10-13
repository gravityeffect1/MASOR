# MASOR 2.1
### Multi-task, ordinal-aware segmentation of colorectal histopathology

**Sara-Mihaela Șova** · Medical Student, Carol Davila University of Medicine and Pharmacy · Research, Victor Babeș National Institute of Pathology

MASOR 2.1 is a computational-pathology research pipeline for **tissue localisation, histological segmentation and ordinal-aware grading** of H&E colorectal biopsy images. It combines a **ConvNeXt-Tiny backbone with an FPN-style decoder and task-specific modelling** and was evaluated using **five-fold grouped cross-validation**.

## Results at a glance

| Out-of-fold metric | Result | 95% bootstrap interval |
|:--|--:|:--|
| **Binary tissue localisation Dice** | **0.939** | 0.934–0.943 |
| **Macro foreground Dice** | **0.815** | 0.783–0.843 |
| **Low-grade Dice** | **0.913** | 0.892–0.931 |
| **Adenocarcinoma Dice** | **0.869** | 0.850–0.887 |
| Normal gland Dice | 0.852 | 0.770–0.913 |
| High-grade Dice | 0.625 | 0.544–0.692 |

**1,694 evaluated images · 376 source groups · five outer folds · 5,000 bootstrap replicates.** The table reports bootstrap means computed from the saved OOF analysis, with intervals supplied by the experiment export.

![Fold-level stability](figures/OOF_fold_stability.png)

## Dataset: EBHI-Seg

[EBHI-Seg](https://figshare.com/articles/dataset/EBHI-SEG/21540159) is a public dataset of **H&E-stained colorectal biopsy histopathology images**, introduced by Shi et al. (2023). The original publication describes **2,228 histology images and 2,228 paired ground-truth images**, with six original diagnostic groups: normal, polyp, low-grade intraepithelial neoplasia, high-grade intraepithelial neoplasia, serrated adenoma and adenocarcinoma. The published images are 224 × 224 pixels.

**MASOR 2.1 uses a defined four-category subset / mapping**, rather than all six original classes:

| MASOR label | Images | Ordinal index |
|:--|--:|--:|
| Normal gland | 76 | 0 |
| Low-grade | 637 | 1 |
| High-grade | 186 | 2 |
| Adenocarcinoma invasion | 795 | 3 |
| **Total** | **1,694** | |

These are the **exact counts in `master_manifest.csv`**, not the full original EBHI-Seg distribution. In the model's semantic evaluation, background is an additional segmentation label; it is **not** an extra image-diagnosis category. Original EBHI-Seg ground-truth images and the MASOR experiment's derived/reference masks should not be assumed interchangeable without reviewing preprocessing.

**Dataset paper:** Shi L. et al. *EBHI-Seg: A novel enteroscope biopsy histopathological hematoxylin and eosin image dataset for image segmentation tasks.* Frontiers in Medicine, 2023. DOI: [10.3389/fmed.2023.1114673](https://doi.org/10.3389/fmed.2023.1114673).

## Modelling and evaluation

1. **Backbone and multi-task formulation.** ConvNeXt-Tiny features with FPN-style multiscale decoding; binary localisation, semantic segmentation and ordinal-aware histological representation.
2. **Ordinal progression.** The experiment explicitly encodes the order normal → low-grade → high-grade → adenocarcinoma, rather than treating all grades as unrelated categories.
3. **Group-aware validation.** The 1,694 images were assigned across five outer test folds with 376 unique source-group identifiers. Inspection of `master_manifest.csv` confirms each `group_key` belongs to exactly one outer fold.
4. **Thresholded post-processing.** Fold-specific binary localisation thresholds (0.35, 0.40, 0.55, 0.50, 0.50) with `prior_lambda=0.75` in the supplied fold exports.
5. **Metrics and uncertainty.** Dice/IoU by class and fold, pooled out-of-fold bootstrap estimates with 5,000 replicates, and fold-stability visualisation.

### Five-fold performance

| Fold | Test images | Source groups | Localisation Dice | Macro foreground Dice | Adenocarcinoma Dice |
|--:|--:|--:|--:|--:|--:|
| 0 | 333 | 74 | 0.927 | 0.771 | 0.829 |
| 1 | 324 | 75 | 0.943 | 0.795 | 0.872 |
| 2 | 312 | 75 | 0.945 | 0.791 | 0.869 |
| 3 | 359 | 75 | 0.940 | 0.859 | 0.892 |
| 4 | 366 | 77 | 0.938 | 0.869 | 0.878 |

**What stands out:** strong binary localisation across all five outer folds (Dice **0.927–0.945**), high performance for low-grade and adenocarcinoma categories, and improved macro foreground Dice in folds 3–4. High-grade morphology is the most variable category and an explicit subject for further analysis.

## Experiments

**Primary:** MASOR 2.1 training/evaluation notebook and outputs (`notebooks/`, `results/`, `figures/`).

**Baseline:** a distinct YOLO26m semantic-segmentation experiment (`baselines/`) using **407 paired tiles from 62 source slides, six pixel labels, and a slide-level train/validation/test split**. Its held-out test output is **mIoU 0.402** and **pixel accuracy 0.644**. This is a separate dataset/protocol and therefore **not a like-for-like performance comparison** against MASOR 2.1.

## Repository contents

```text
notebooks/       MASOR 2.1 training/evaluation
baselines/       YOLO26 semantic experiment and configuration
preprocessing/   slide/tissue/tile extraction utilities
results/         OOF statistics, fold comparisons, manifests
figures/         visual predictions and fold-stability analysis
checkpoints/     checkpoint notes and provenance
```

## Reproducing the work

Use the experiment notebooks with their pinned library versions and the appropriate EBHI-Seg-derived data layout. The full trained fold-2 checkpoint is a separate binary artefact; inference validation is in progress. Original images are obtained from the public dataset provider rather than included in Git.

**Scope of claims:** these are strong results within the **reported grouped out-of-fold research evaluation**; they are not clinical deployment metrics or an independent external-hospital validation. Some mask references in this pipeline are derived/weak labels, a relevant consideration for interpreting Dice.

## Acknowledgements

Implementation, experiments, analysis and portfolio documentation: **Sara-Mihaela Șova**. Research conducted in the context of work at the **Victor Babeș National Institute of Pathology**, with collaboration and feedback from **Leonard Savu** and the broader research team.

---

*Research software — not for clinical diagnosis.*
