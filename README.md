# NACASurrogate

Regime-aware parametric surrogate model for NACA 4-digit aerofoils.
Maps `(α, Re, thickness, regime_id) → (Cl, Cd)` using 200 OpenFOAM RANS runs as
training data, with one physics-validated CFD template per flow regime.

---

## Why regime-aware

A single turbulence model and mesh strategy cannot reliably cover the full
aerodynamic design space. Attached turbulent flow, near-stall separated flow,
transitional low-Re flow, and fully turbulent high-Re flow have fundamentally
different governing physics. Using one CFD recipe across all of them blends
distinct error structures into a single dataset and degrades CFD reliability,
validation quality, surrogate smoothness, and extrapolation behavior.

This redesign partitions the design space into four flow regimes, each with its
own validated CFD template (turbulence model, wall treatment, y+ target, mesh
strategy). Samples are then merged into one ML-ready dataset, with `regime_id`
encoded as an explicit input feature on the surrogate.

For the full design rationale see `README_complete.md`. For OpenFOAM and Python
implementation rules see `CLAUDE.md`.

---

## Flow regimes

| ID | Name | α (°) | Re | t/c | Turbulence | Wall treatment | Target y+ | Cells |
|---|---|---|---|---|---|---|---|---|
| A | Attached turbulent | 0–8 | 1.5×10⁶ – 3×10⁶ | 0.10–0.18 | Spalart–Allmaras | Wall functions | 20–50 | 80k–200k |
| B | Near-stall separated | 10–16 | 1×10⁶ – 3×10⁶ | 0.12–0.24 | k-ω SST | Fully resolved | < 1 | 300k–1M |
| C | Transitional low-Re | 0–8 | 3×10⁵ – 1×10⁶ | 0.08–0.15 | k-kL-ω (transition) | Fully resolved | < 1 | 500k–1.2M |
| D | High-Re attached | 0–6 | 2×10⁶ – 5×10⁶ | 0.10–0.18 | Spalart–Allmaras | Wall functions | 30–80 | 50k–150k |

### Global DOE envelope

| Parameter | Symbol | Range | Units |
|---|---|---|---|
| Angle of attack | α | 0 – 16 | degrees |
| Reynolds number | Re | 5×10⁵ – 3×10⁶ | — |
| Max thickness | t | 0.08 – 0.24 | fraction of chord |

Chord = 1.0 m, ν = 1.5×10⁻⁵ m²/s. Regime D extends Re up to 5×10⁶ for the
high-Re subset; the global surrogate is queried over [5×10⁵, 5×10⁶] but the
primary validation envelope remains [5×10⁵, 3×10⁶].

---

## Sample allocation (200 LHS samples)

LHS is performed independently inside each regime's bounding box, then
concatenated. The 80/20 train/test split is stratified by regime so every
regime is represented in both splits.

| Regime | N | Train | Test |
|---|---|---|---|
| A | 80 | 64 | 16 |
| B | 50 | 40 | 10 |
| C | 30 | 24 | 6 |
| D | 40 | 32 | 8 |
| Total | 200 | 160 | 40 |

Indices saved at DOE time and never modified. `random_state = 42`.

---

## Outputs

| Variable | Source | Notes |
|---|---|---|
| Cl | OpenFOAM `forceCoeffs` | Primary |
| Cd | OpenFOAM `forceCoeffs` | Primary |
| L/D | Derived Cl/Cd | Secondary |
| Cp(x/c) | `singleGraph` along wall patch | Phase 2 only |

---

## Surrogates (one global model per output)

Inputs: `X = [α_deg, Re, thickness, regime_onehot_A, regime_onehot_B,
regime_onehot_C, regime_onehot_D]` → outputs Cl, Cd.

- Gaussian Process — `sklearn.GaussianProcessRegressor`, `Matern(ν=2.5) + WhiteKernel`, `n_restarts_optimizer=10`, `normalize_y=True`
- Random Forest — `sklearn.RandomForestRegressor`, `n_estimators=200`
- MLP — `sklearn.MLPRegressor`, layers `(64, 64, 32)`, ReLU, `early_stopping=True`
- Kriging — `smt.surrogate_models.KRG`

Regime is encoded as a 4-column one-hot vector to avoid imposing a false ordinal
relation on the categorical regime variable. Per-regime test-set metrics are
reported in addition to the global metrics.

---

## Toolchain

| Job | Tool |
|---|---|
| DOE | `pyDOE2` (per-regime LHS, then concatenated) |
| Geometry | `numpy` (analytic NACA 4-digit formula) |
| Meshing | `gmsh` Python API |
| CFD | OpenFOAM 12 — `foamRun` with `solver incompressibleFluid` |
| Turbulence | per-regime: `SpalartAllmaras` / `kOmegaSST` / `kkLOmega` |
| Parallelism | GNU `parallel` |
| Surrogates | `scikit-learn`, `smt` |
| Sensitivity | `SALib` (Sobol) |
| Plots | `matplotlib`, `seaborn` |

---

## Directory structure

```
NACASurrogate/
├── README.md
├── README_complete.md
├── CLAUDE.md
├── environment.yml
├── samples.csv               # 200 × [case_id, alpha_deg, Re, thickness, regime]
├── train_idx.npy             # 160 training indices (stratified)
├── test_idx.npy              # 40 test indices (stratified)
├── dataset_clean.csv         # harvested CFD results (regime-tagged)
│
├── scripts/
│   ├── 01_generate_doe.py        # per-regime LHS → merged samples.csv
│   ├── 02_classify_regime.py     # regime classifier utility (inference)
│   ├── 03_generate_geometry.py   # NACA xy → aerofoil.dat per case
│   ├── 04_generate_mesh.py       # gmsh → polyMesh, regime-specific topology
│   ├── 05_prepare_case.py        # render regime template → OpenFOAM case
│   ├── 06_run_cfd.py             # foamRun via GNU parallel
│   ├── 07_harvest_results.py     # parse forceCoeffs + Cp; convergence gate
│   ├── 08_validate_regimes.py    # NACA0012 reference checks per regime
│   ├── 09_train_surrogates.py    # GP / RF / MLP / KRG (global, regime-aware)
│   └── 10_global_validation.py   # parity, Sobol, OOD, learning curves
│
├── templates/                    # regime-specific OpenFOAM templates (Jinja2)
│   ├── regime_A/
│   │   ├── 0/{U.jinja, p, nut, nuTilda}
│   │   ├── constant/momentumTransport
│   │   └── system/{controlDict.jinja, fvSchemes, fvSolution}
│   ├── regime_B/
│   │   ├── 0/{U.jinja, p, k, omega, nut}
│   │   ├── constant/momentumTransport
│   │   └── system/{controlDict.jinja, fvSchemes, fvSolution}
│   ├── regime_C/
│   │   ├── 0/{U.jinja, p, k, kl, omega, nut}
│   │   ├── constant/momentumTransport
│   │   └── system/{controlDict.jinja, fvSchemes, fvSolution}
│   └── regime_D/
│       ├── 0/{U.jinja, p, nut, nuTilda}
│       ├── constant/momentumTransport
│       └── system/{controlDict.jinja, fvSchemes, fvSolution}
│
├── validation/                   # canonical reference cases per regime
│   ├── regime_A/{cases/, references/, report.md}
│   ├── regime_B/{cases/, references/, report.md}
│   ├── regime_C/{cases/, references/, report.md}
│   └── regime_D/{cases/, references/, report.md}
│
├── cases/                        # auto-generated (gitignored)
│   └── case_0000/
│       ├── aerofoil.dat
│       ├── params.json
│       ├── case_metadata.json    # regime, y+ achieved, cells, runtime, residuals
│       └── <OpenFOAM case>
│
├── models/                       # joblib
│   ├── scaler.joblib
│   ├── gp_Cl.joblib  / gp_Cd.joblib
│   ├── rf_Cl.joblib  / rf_Cd.joblib
│   ├── mlp_Cl.joblib / mlp_Cd.joblib
│   └── krg_Cl.joblib / krg_Cd.joblib
│
└── results/
    ├── regime_validation/        # per-regime CFD validation reports
    ├── surrogate_metrics.csv     # global + per-regime metrics
    ├── parity_plots.png
    ├── sobol_sensitivity.png
    ├── ood_test.png
    ├── learning_curve.png
    └── stall_extrapolation.png
```

---

## Pipeline — 10 stages

### Stage 1 — `01_generate_doe.py`
**Inputs:** none. **Outputs:** `samples.csv`, `train_idx.npy`, `test_idx.npy`.

- LHS sampled independently in each regime's bounding box via `pyDOE2.lhs(d=3, samples=N, criterion='maximin')`
- Per-regime allocation: A=80, B=50, C=30, D=40
- Concatenate → 200 rows, columns `[case_id, alpha_deg, Re, thickness, regime]`
- 80/20 split stratified by regime
- `random_state = 42`

### Stage 2 — `02_classify_regime.py`
**Inputs:** `(α, Re, t)`. **Outputs:** regime label ∈ {A, B, C, D, OOD}.

At DOE time the regime is known by construction. This module is the
inference-time classifier (also used during global validation to assign labels
to OOD probe points). See `CLAUDE.md` §11 for the priority rule.

### Stage 3 — `03_generate_geometry.py`
**Inputs:** `samples.csv`. **Outputs:** `cases/case_{i:04d}/aerofoil.dat`, `params.json`.

Same analytic NACA 4-digit formula as the original pipeline (200 cosine-spaced
points, closed polygon: upper TE→LE then lower LE→TE). Symmetric only (m=0)
for Phase 1.

### Stage 4 — `04_generate_mesh.py`
**Inputs:** `aerofoil.dat`, regime label, Re. **Outputs:** `constant/polyMesh/`.

Mesh strategy is regime-specific (table in `CLAUDE.md` §7). First cell height
computed from the regime's target y+ via the flat-plate turbulent BL formula.
`checkMesh` must pass: non-orthogonality < 70°, skewness < 4. Achieved metrics
written to `case_metadata.json`.

### Stage 5 — `05_prepare_case.py`
**Inputs:** regime label, `(α, Re, t)`. **Outputs:** OpenFOAM case from regime template.

Selects `templates/regime_{X}/` and renders Jinja2 placeholders. Template
variables include `UX, UY, UINF, LIFTDIR_X/Y, DRAGDIR_X/Y, NU, K0, OMEGA0,
NUT0, NUTILDA0, KL0`. Each regime's template renders only the placeholders it
needs (SA regimes do not render `K0/OMEGA0`; transition regime additionally
renders `KL0`).

### Stage 6 — `06_run_cfd.py`
**Inputs:** prepared cases. **Outputs:** `log.foamRun`, `postProcessing/`.

```bash
parallel -j 4 \
  "cd {1} && source /opt/openfoam12/etc/bashrc && foamRun > log.foamRun 2>&1" \
  ::: cases/case_*/
```

### Stage 7 — `07_harvest_results.py`
**Inputs:** all cases. **Outputs:** `dataset_all.csv`, `dataset_clean.csv`.

Parses `postProcessing/forceCoeffs/0/coefficient.dat` and `log.foamRun`.
Convergence criteria are regime-aware (see `CLAUDE.md` §8). Final columns:
`[case_id, alpha_deg, Re, thickness, regime, Cl, Cd, L_over_D, Cl_std, Cd_std,
y_plus_achieved, cells, runtime_s, converged]`.

### Stage 8 — `08_validate_regimes.py`
Per-regime CFD validation against canonical NACA0012 references:

| Regime | Probe points | Reference |
|---|---|---|
| A | Re=2e6, α∈{0,4,8}; Re=3e6, α=4 | Abbott & von Doenhoff; NASA TMR SA |
| B | Re=2e6, α∈{12,14,16} | NASA TMR; AGARD experimental data |
| C | Re=5e5, α∈{2,4,6} | XFOIL with eN transition; experimental low-Re data |
| D | Re=4e6, α∈{0,4} | NASA TMR SA reference |

Acceptance: |ΔCl| ≤ 5%, |ΔCd| ≤ 10%, qualitative Cp agreement. A regime is
locked only after passing these checks; only locked regimes contribute samples
to the dataset.

### Stage 9 — `09_train_surrogates.py`
Trains GP / RF / MLP / KRG on the merged dataset with one-hot `regime_id`.
Cross-validation is performed on the training set only (never on the test
indices). All models and the scaler are saved via `joblib.dump`.

### Stage 10 — `10_global_validation.py`
Parity plots (4 models × 2 outputs, colored by regime), Sobol indices (SALib,
N=8192), OOD sweep, learning curves, stall extrapolation, per-regime error
breakdown.

---

## Validation phases (do NOT skip ahead)

| Phase | Goal | Blocking |
|---|---|---|
| 1 | Validate Regime A CFD template against NACA0012 reference | Phases 2–6 |
| 2 | Validate Regime B CFD template (near-stall separated) | Phases 5–6 for B samples |
| 3 | Validate Regime C CFD template (transitional) | Phases 5–6 for C samples |
| 4 | Validate Regime D CFD template (high-Re) | Phases 5–6 for D samples |
| 5 | Generate full 200-case dataset | Phase 6 |
| 6 | Train and validate the unified surrogate | — |

Phase 1 (Regime A) is the active focus and must be locked before any other
phase begins. A regime may be skipped from Phase 5 only if its template fails
to validate; in that case the surrogate is trained on the remaining regimes
and the failed regime is documented as a known gap.

---

## Rules (always apply)

- Environment: `micromamba activate openfoam` — see `CLAUDE.md` §2
- OpenFOAM syntax: use `$FOAM_TUTORIALS`, `foamInfo`, `foamSearch` — see `CLAUDE.md` §4
- Turbulence dict: `constant/momentumTransport` (OpenFOAM 12; not `turbulenceProperties`)
- AoA: rotate inlet velocity, never the mesh
- Templates are per-regime — never use one regime's template for another regime's sample
- Every case writes `case_metadata.json` (regime, achieved y+, cells, runtime, residuals)
- Paths: `pathlib.Path`, relative to `PROJECT_ROOT`
- Models: `joblib.dump` / `joblib.load`
- Logging: `logging` module, not `print()`
- Seed: 42 everywhere
