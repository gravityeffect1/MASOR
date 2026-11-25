# Evaluation exports

MASOR 2.2, experiment `masor22_full_case_s42`. Five outer test folds, grouped at the **case level (267 cases)**, with 356, 315, 342, 282 and 399 image records respectively (1,694 total). The OOF bootstrap export uses 5,000 replicates.

Files:
- `OOF_headline_results.json` — headline OOF metrics with 95% intervals (segmentation + image-level ordinal grading)
- `OOF_bootstrap_ci.csv` — OOF bootstrap means and confidence intervals
- `fold_comparison.csv` — per-fold test metrics and checkpoint hashes
- `master_manifest.csv` — data grouping and fold allocation (sha256 ed399191…)
- `checkpoint_manifest.csv` / `checkpoint_manifest.json` — model metadata and provenance
- `fold_0_prototype_centers.csv` … `fold_4_prototype_centers.csv` — fold-specific prototype metadata

Interpretation: internal case-grouped OOF evaluation; not clinical external validation. Note the reported SHA mismatch across checkpoint exports (see `../checkpoints/README.md`).
