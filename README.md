# MASOR — Colorectal Histopathology Segmentation and Grading

**Research implementation and experimental documentation by Sara-Mihaela Șova**  
Medical student, Carol Davila University of Medicine and Pharmacy, Bucharest · Research work at the “Victor Babeș” National Institute of Pathology.

MASOR explores multiclass tissue segmentation and ordinal-aware modelling in colorectal histopathology. This repository distinguishes the **MASOR 2.1 experimental pipeline** from an earlier **YOLO26m semantic-segmentation baseline**.

## Study design

- **MASOR 2.1:** ConvNeXt-Tiny/FPN-based multi-task modelling with tissue localisation and semantic/ordinal components, evaluated with **five outer folds**.
- **Evaluation corpus:** 1,694 image records assigned to 376 source groups across the five outer test folds (according to the experiment manifests).
- **Reporting:** out-of-fold (OOF) metrics, fold-wise comparisons and bootstrap uncertainty estimates.
- **YOLO26m-sem baseline:** separate experiment on 407 tiles from 62 source slides with six semantic labels. Its metrics are **not directly comparable** with MASOR 2.1 because the evaluation sets and class definitions differ.

## MASOR 2.1 — selected OOF results

| Metric | Bootstrap mean | Bootstrap interval |
|:--|--:|:--|
| Binary localisation Dice | 0.9386 | 0.9340–0.9428 |
| Macro foreground Dice | 0.8148 | 0.7828–0.8426 |
| Normal gland Dice | 0.8522 | 0.7701–0.9133 |
| Low-grade Dice | 0.9133 | 0.8924–0.9310 |
| High-grade Dice | 0.6246 | 0.5440–0.6916 |
| Adenocarcinoma invasion Dice | 0.8692 | 0.8499–0.8867 |

Intervals are **bootstrap estimates (5,000 replicates)** from the supplied OOF analysis; they are not independent prospective clinical-validation intervals. High-grade tissue showed the weakest segmentation performance and substantial fold variability.

### Fold-wise results

| Fold | Test images | Test groups | Binary Dice | Macro foreground Dice |
|--:|--:|--:|--:|--:|
| 0 | 333 | 74 | 0.9274 | 0.7710 |
| 1 | 324 | 75 | 0.9428 | 0.7948 |
| 2 | 312 | 75 | 0.9452 | 0.7910 |
| 3 | 359 | 75 | 0.9401 | 0.8594 |
| 4 | 366 | 77 | 0.9379 | 0.8691 |

These results use the available experiment exports. They should not be interpreted as clinical diagnostic accuracy.

## Repository organisation

- `notebooks/` — MASOR 2.1 training and evaluation notebook
- `baselines/` — separate YOLO26m semantic-segmentation experiment and data schema
- `preprocessing/` — tissue and tile extraction utilities
- `results/` — fold metrics, bootstrap intervals and manifests
- `figures/` — fold-stability plot and representative prediction panels
- `checkpoints/` — checkpoint metadata; model weights distributed separately when available

## Reproducibility and checkpoint notes

The five fold checkpoints represent **different trained models**, not five layers of one network. Fold 2 can be used as a single-fold inference demonstration once its architecture and loading code are verified.

The original OOF fold comparison exports and the subsequently exported final-checkpoint manifest contain **different SHA-256 values**. These should be treated as distinct checkpoint artefacts until their relationship is established. A complete Fold 2 archive has been recovered; its inference has not yet been independently validated in this repository.

The two experiment families require different dependency/environment configurations; do not assume that one `requirements.txt` exactly reproduces both.

## Research context and credit

This repository documents **Sara-Mihaela Șova's implementation, experiments and analysis** within a broader collaborative computational-pathology effort at the “Victor Babeș” National Institute of Pathology. Acknowledgement: Leonard Savu and the institutional research team for their collaboration on the wider project.

## Status

**Research code / experimental project — not a medical device.** These results are research evaluations and should not be used to diagnose patients.
