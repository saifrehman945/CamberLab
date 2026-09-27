# legacy/regime_v1 — retired regime-aware OpenFOAM pipeline

This directory archives CamberLab's first design: a four-regime (A/B/C/D)
NACA surrogate trained on the project's own OpenFOAM 12 RANS cases
(123 converged cases across regimes A/C/D; see `results/dataset_clean.csv`).

It was retired because the self-generated dataset was too small and too
unevenly converged (Regime C transitional cases never met the convergence gate)
to support a reliable surrogate. The project now trains on AirfRANS
(Bonnet et al., NeurIPS 2022) — see the root `README.md`.

Files were moved here with `git mv`, so their history is preserved. They are
not maintained and their imports assume the old root layout. The full design
rationale is in `README_complete.md` in this directory.
