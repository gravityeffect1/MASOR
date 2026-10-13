# Checkpoints

Fold 0–4 are separate MASOR 2.1 trained models used in outer-fold evaluation. A full Fold 2 archive has been recovered, but loading/inference still needs to be verified.

The checkpoint SHA-256 hashes from `fold_comparison.csv` differ from those in `checkpoint_manifest.csv/json`; do not claim that exported `*_final.pt` checkpoints exactly reproduce the OOF scores until provenance is reconciled.

Weights larger than 100 MB belong in a GitHub Release (or other binary-hosting service), not the Git repository. No weights are committed here.
