# Checkpoints

Trained MASOR 2.2 weights are **not** stored in Git history. They are published as **GitHub Release** assets (`*.pt` is git-ignored). Weights larger than 100 MB cannot be pushed to the repository and belong in a Release.

## Available weights

Experiment `masor22_full_case_s42` (MASOR-2.2, ConvNeXt-Tiny, 224 px, five-fold **case-grouped** cross-validation, seed 42):

| Fold | File | best_epoch | bin_threshold | prior_lambda |
|--:|:--|--:|--:|--:|
| 0 | masor22_full_case_s42_fold_0_final.pt | 36 | 0.35 | 1.0 |
| 1 | masor22_full_case_s42_fold_1_final.pt | 62 | 0.40 | 2.0 |
| 2 | masor22_full_case_s42_fold_2_final.pt | 39 | 0.50 | 0.25 |
| 3 | masor22_full_case_s42_fold_3_final.pt | 35 | 0.50 | 2.0 |
| 4 | masor22_full_case_s42_fold_4_final.pt | 11 | 0.45 | 2.0 |

Per-fold SHA-256 hashes are in `../results/checkpoint_manifest.csv`.

## Provenance caveat

The SHA-256 values in `checkpoint_manifest.csv` differ from the `checkpoint_sha256` column in `fold_comparison.csv`. Until this is reconciled, do not claim that the exported `*_final.pt` files reproduce the out-of-fold scores bit-for-bit. Loading and end-to-end inference from the released checkpoints (see `../infer.py`) have not yet been independently re-validated outside the training notebook.
