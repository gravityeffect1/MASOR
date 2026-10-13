# Evaluation exports

The evaluation contains five outer test folds with 333, 324, 312, 359 and 366 image records respectively (1,694 total). The OOF bootstrap export uses 5,000 replicates.

Files to include in this directory:
- `fold_comparison.csv` — per-fold test metrics and checkpoint hashes
- `OOF_bootstrap_ci.csv` — OOF bootstrap means and confidence intervals
- `master_manifest.csv` — data grouping and fold allocation
- `checkpoint_manifest.csv` and `checkpoint_manifest.json` — model metadata and provenance
- `fold_0_prototype_centers.csv` through `fold_4_prototype_centers.csv` — fold-specific prototype metadata

Interpretation: internal OOF evaluation; not clinical external validation. Note the reported SHA mismatch across checkpoint exports.
