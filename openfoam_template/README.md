# OpenFOAM Templates — Regime-Indexed

This directory holds one OpenFOAM 12 master case per flow regime. Each
sample in `cases/case_XXXX/` is rendered from the subdirectory matching
its `regime` field in `params.json` (see `scripts/04_run_cfd.py`).

| Subdir       | Regime | Turbulence model | Wall treatment                                  | Target y+ | endTime |
|--------------|--------|------------------|-------------------------------------------------|-----------|---------|
| `regime_A/`  | A      | `SpalartAllmaras`| Wall functions (`nutUSpaldingWallFunction`)     | ~30       | 2000    |
| `regime_B/`  | B      | `kOmegaSST`      | Low-Re (`kLowReWallFunction`, `omegaWallFunction`, `nutLowReWallFunction`) | <1 (≈0.5) | 5000    |
| `regime_C/`  | C      | *(not yet)*      |                                                 |           |         |
| `regime_D/`  | D      | *(not yet)*      |                                                 |           |         |

Regime classification at DOE time lives in `scripts/01_doe.py`; at inference
time in `scripts/mesh/regime_parameters.py::classify_regime`. Mesh
parameters per regime are in `REGIME_MESH` of the same file.

Each regime subdirectory has its own `README.md` documenting its physics
recipe, patch contract, and Jinja placeholders. See `CLAUDE.md §10` for
the full regime spec and `§4–§5` for OpenFOAM 12 syntax rules.
