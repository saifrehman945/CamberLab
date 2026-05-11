# validation/

Per-regime CFD validation outputs. Each regime gets its own subdirectory
populated by `scripts/08_validate_regimes.py`.

```
validation/
├── README.md                       ← this file
├── validation_summary.csv          ← global summary across regimes (orchestrator output)
├── validation_point.png            ← legacy single-point Ladson plot (scripts/validation.py)
├── regime_A/
│   ├── report.md                   ← pass/fail decision + tables
│   └── figures/
│       ├── cl_cd_comparison.png
│       ├── cp_comparison__<case_id>.png
│       ├── residual_history__<case_id>.png
│       └── mesh_convergence.png
├── regime_B/, regime_C/, regime_D/
```

## How to validate a regime

The single end-to-end command is:

```bash
micromamba activate openfoam
# Stage cases, run CFD, parse, plot, and write report:
python scripts/08_validate_regimes.py --regime A --run --jobs 4

# Validate B and C without (re)running CFD (forceCoeffs.dat already present):
python scripts/08_validate_regimes.py --regime B C

# Re-render plots from existing CFD output without re-staging:
python scripts/08_validate_regimes.py --regime A --report-only
```

Flags:

| Flag | Effect |
|---|---|
| `--regime A B C D` | Which regimes to process |
| `--run` | Invoke `foamRun` after staging (via GNU parallel) |
| `--jobs N` | parallelism for `--run` (default 4) |
| `--no-stage` | Skip staging; assumes cases already on disk |
| `--skip-mesh` | Stage geometry and OpenFOAM files; do not mesh |
| `--force` | Re-mesh and re-render even if outputs exist |
| `--report-only` | Skip stage and run; parse and plot existing outputs only |
| `--window N` | Trailing iterations averaged for Cl/Cd (default 200) |

## Validation data lineage

1. **Reference data** lives in `validation_data/`, populated from NASA TMR
   (`https://tmbwg.github.io/turbmodels/`) plus Coles & Wadcock NACA 4412
   and Abbott & von Doenhoff. Run `scripts/validation/fetch_reference_data.py`
   to re-download and refresh `validation_data/manifest.json`.

2. **Regime C** has no canonical NASA reference. Generate engineering-grade
   reference data with XFOIL eN:

   ```bash
   apt install xfoil          # one-time
   python scripts/validation/generate_xfoil_reference.py --re 5e5 --alpha 0 2 4 6 8
   ```

3. **Cases** are staged under `cases/validation_<regime>_<case_id>/` and
   ignored by git (see `.gitignore`).

4. **Outputs** (`validation/regime_*/report.md`, figures, and
   `validation/validation_summary.csv`) are tracked.

## Acceptance criteria

Each regime defines its own tolerances in `validation_data/regime_<X>/metadata.json`
under `tolerances`. The orchestrator computes |ΔCl|% and |ΔCd|% against the
regime's `experimental_primary` reference and writes a pass/fail line at the
top of `report.md`.

| Regime | ΔCl tol | ΔCd tol | Notes |
|---|---|---|---|
| A | ±5% | ±10% | NACA 0012 Re=6e6, Ladson 80-grit tripped |
| B | ±10% | ±15% | NACA 0012 α=15° + NACA 4412 separation |
| C | ±15% | ±20% | XFOIL eN engineering reference |
| D | ±5% | ±10% | Subset of A; cheaper mesh |

## Phase order

The CLAUDE.md `§13` validation gate must be respected:

1. Lock Regime A — Phase 1
2. Lock Regime B — Phase 2 (blocks B's surrogate samples until passed)
3. Lock Regime C — Phase 3
4. Lock Regime D — Phase 4
5. Generate full 200-case DOE — Phase 5 (blocked on 1–4)
6. Train surrogate — Phase 6
