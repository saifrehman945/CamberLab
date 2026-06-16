# NACASurrogate — Regime-Aware CFD Surrogate Modeling for NACA Aerofoils

## Author
**Saif ur Rehman** — CFD Researcher | Physics-AI & Design Optimization
[Portfolio](https://saifrehman945.github.io/) | [GitHub](https://github.com/saifrehman945)

---

## Table of Contents

1. [Objective](#1-objective)
2. [Why Build a Surrogate?](#2-why-build-a-surrogate)
3. [What Is a Surrogate Model?](#3-what-is-a-surrogate-model)
4. [Why Regime-Aware? — Motivation for the Redesign](#4-why-regime-aware--motivation-for-the-redesign)
5. [Scope and Constraints](#5-scope-and-constraints)
6. [Flow Regime Definitions](#6-flow-regime-definitions)
7. [Design Space and DOE Strategy](#7-design-space-and-doe-strategy)
8. [Output Targets](#8-output-targets)
9. [Toolchain](#9-toolchain)
10. [Pipeline — 10 Stages](#10-pipeline--10-stages)
11. [Per-Regime CFD Strategy](#11-per-regime-cfd-strategy)
12. [Validation Strategy and Phases](#12-validation-strategy-and-phases)
13. [Surrogate Training Strategy](#13-surrogate-training-strategy)
14. [Surrogate Comparison Matrix](#14-surrogate-comparison-matrix)
15. [Metadata Schema](#15-metadata-schema)
16. [Risk Register](#16-risk-register)
17. [Known Limitations](#17-known-limitations)
18. [Directory Structure](#18-directory-structure)
19. [Implementation Sequence](#19-implementation-sequence)
20. [Future Extensions](#20-future-extensions)

---

## 1. Objective

Build a **regime-aware parametric surrogate model** that learns the mapping:

```
(angle_of_attack, Reynolds_number, thickness, flow_regime) → (Cl, Cd)
```

from a curated set of high-fidelity OpenFOAM RANS CFD simulations of NACA 4-digit
aerofoils, where each simulation is run with a turbulence model, wall treatment,
and mesh appropriate to the local flow physics rather than a single global
configuration.

The surrogate replaces the CFD solver during design exploration — reducing
evaluation time from ~5–60 minutes per case to milliseconds — while preserving
the physical fidelity advantages of a regime-specific solver setup.

The primary goal is to **build a physics-aware data pipeline that produces a
clean, defensible aerodynamic dataset** and a corresponding surrogate that is
trustworthy across the design space rather than only in a narrow band.

---

## 2. Why Build a Surrogate?

A single 2D RANS simulation of an aerofoil takes 5–60 minutes depending on
mesh resolution, turbulence model, and convergence behavior. Design
optimization, uncertainty quantification, sensitivity analysis, and
hyperparameter search may require thousands of evaluations. This makes direct
CFD-in-the-loop optimization computationally prohibitive.

A surrogate model (response surface, metamodel) is trained on a small,
structured set of CFD runs and learns to approximate the input–output
relationship. Once trained it evaluates in milliseconds, enabling:

- Gradient-free optimization over the parametric design space
- Sobol-style global sensitivity analysis
- Monte-Carlo uncertainty propagation
- Real-time aerodynamic estimation in conceptual design tools

**Questions this project is set up to answer:**

- How many CFD samples are needed for a reliable surrogate across regimes?
- Does regime-conditioning improve surrogate accuracy versus a single global model?
- Which surrogate type (GP, RF, MLP, Kriging) works best per regime and globally?
- Where does the surrogate fail — at regime boundaries, near stall, outside the training Re range?
- Does the surrogate respect physical constraints (Cl ≈ 0 at α = 0° for symmetric aerofoils, Cd > 0)?
- How does GP uncertainty behave at regime boundaries, and can it be used to drive active learning?

---

## 3. What Is a Surrogate Model?

A surrogate model is a data-driven approximation of an expensive function. In
this context:

- **Inputs (X):** design and operating parameters — angle of attack α, Reynolds
  number Re, thickness t, plus the categorical regime label encoded as one-hot
- **Outputs (y):** aerodynamic performance — Cl and Cd
- **Training data:** N high-fidelity CFD simulations spanning four regimes
- **Inference:** given a new `(α, Re, t)`, classify into a regime, then predict
  Cl and Cd without running CFD

This is a **parametric surrogate** — it generalizes across geometries and
operating conditions, distinct from a time-series surrogate that learns the
temporal evolution of a single fixed case.

---

## 4. Why Regime-Aware? — Motivation for the Redesign

The original NACASurrogate (commit `5150a91`) used a single CFD recipe across
the entire aerodynamic design space:

- one turbulence model (`kOmegaSST`)
- one mesh philosophy (low-Re, fully resolved walls)
- one y+ target (~0.5)
- one solver template
- a single LHS of 100 points over the full `(α, Re, t)` box

This is physically inconsistent. The same LHS sample crosses at least four
fundamentally different flow regimes:

1. **Attached turbulent flow** at moderate Re and moderate α — well-modeled by
   either Spalart–Allmaras or `kOmegaSST` with wall functions on a coarse mesh.
2. **Near-stall separated flow** at α > 10° — requires fully resolved walls,
   careful relaxation, and a finer mesh to capture wake structure. Wall
   functions corrupt the separation prediction.
3. **Transitional low-Re flow** at Re < 1×10⁶ — fully turbulent models predict
   the wrong drag and miss laminar separation bubbles entirely. Requires a
   transition-aware model (`kkLOmega` or `kOmegaSSTLM`).
4. **Fully turbulent high-Re attached flow** at Re > 2×10⁶, low α — accurate
   with Spalart–Allmaras + wall functions; expensive and wasteful with a
   y+ < 1 mesh.

Using a single recipe across all four leads to:

- **Reduced CFD reliability:** transitional cases run with a fully turbulent
  model produce confidently wrong Cl/Cd, polluting the training data.
- **Wasted compute:** running a low-Re mesh at Re = 3×10⁶ requires 5–10×
  more cells than necessary for the physics in that regime.
- **Poor validation quality:** a single mesh and template cannot be validated
  against the canonical reference cases for each regime; what passes for
  Regime A is wrong for Regime C.
- **Discontinuous surrogate behavior:** error structure changes abruptly across
  the design space, producing a non-smooth target function that all of GP, RF,
  MLP, and Kriging struggle to fit well simultaneously.
- **Misleading extrapolation:** OOD predictions inherit the dominant regime's
  error mode and give no indication that the input has crossed a physical
  boundary.

The redesign **partitions the design space into four flow regimes**, each with
a validated CFD template, and merges the results into a single dataset with
`regime_id` as an explicit input feature. The surrogate learns a smoother
target function and is auditable at the regime level.

---

## 5. Scope and Constraints

| Item | Choice | Reason |
|---|---|---|
| Aerofoil family | NACA 4-digit symmetric (Phase 1) | Analytic formula, no CAD, extensively validated |
| CFD solver | OpenFOAM 12 (`foamRun` + `incompressibleFluid`) | Free, scriptable, current Foundation release |
| Meshing | gmsh Python API | Free, fully scriptable, regime-specific topologies |
| DOE | Per-regime LHS, then concatenated | Avoids wasting samples in physically incoherent regions |
| Surrogate library | scikit-learn + smt | Free, well-documented |
| Sensitivity | SALib | Free, supports Sobol indices |
| No proprietary tools | Fluent, ICEM, DAKOTA excluded | Fully reproducible on any Linux machine |

---

## 6. Flow Regime Definitions

Four regimes are defined by `(α, Re, t)` bounding boxes plus a physics
description and a CFD recipe.

### Regime A — Attached turbulent

| Property | Value |
|---|---|
| α | 0°–8° |
| Re | 1.5×10⁶ – 3×10⁶ |
| t/c | 0.10 – 0.18 |
| Physics | Attached turbulent BL, mild adverse pressure gradient, limited separation |
| Turbulence | `SpalartAllmaras` (preferred); `kOmegaSST` with wall functions acceptable |
| Wall treatment | Wall functions |
| Target y+ | 20–50 |
| Prism layers | 15–20 |
| Cell count | 80k – 200k |
| Solver end-time | 2000 iterations |

This is the **first regime to validate**. Its physics is the cleanest, the
canonical reference data is the most abundant (NACA0012 at Re = 2×10⁶ and 3×10⁶
in Abbott & von Doenhoff and on the NASA Turbulence Modelling Resource), and
the CFD setup is robust.

### Regime B — Near-stall separated

| Property | Value |
|---|---|
| α | 10°–16° |
| Re | 1×10⁶ – 3×10⁶ |
| t/c | 0.12 – 0.24 |
| Physics | Strong adverse pressure gradient, partial separation, wake growth, stall onset |
| Turbulence | `kOmegaSST`; Transition SST optional in future work |
| Wall treatment | Fully resolved |
| Target y+ | < 1 |
| Prism layers | 30–40 |
| Cell count | 300k – 1M |
| Solver end-time | 5000 iterations |
| Notes | Pseudo-transient stabilization may be needed; URANS as future option |

Steady RANS becomes unreliable above ~14–16° as the flow transitions to
massive separation. Cases past 16° are excluded.

### Regime C — Transitional low-Re

| Property | Value |
|---|---|
| α | 0°–8° |
| Re | 3×10⁵ – 1×10⁶ |
| t/c | 0.08 – 0.15 |
| Physics | Laminar BL, transition, laminar separation bubbles |
| Turbulence | `kkLOmega` (Walters–Cokljat); fallback `kOmegaSST` with low free-stream Tu |
| Wall treatment | Fully resolved |
| Target y+ | < 1 |
| Prism layers | 35–45 |
| Cell count | 500k – 1.2M |
| Solver end-time | 4000 iterations |

This is the most physically delicate regime. It is implemented only after A
and B are locked.

### Regime D — Fully turbulent high-Re attached

| Property | Value |
|---|---|
| α | 0°–6° |
| Re | 2×10⁶ – 5×10⁶ |
| t/c | 0.10 – 0.18 |
| Physics | Fully turbulent attached flow, minimal transition, minimal separation |
| Turbulence | `SpalartAllmaras` |
| Wall treatment | Wall functions |
| Target y+ | 30–80 |
| Prism layers | 12–18 |
| Cell count | 50k – 150k |
| Solver end-time | 2000 iterations |

Regime D overlaps with Regime A by construction; the classifier (§11) gives
D priority when it applies, because D's mesh is cheaper and the physics
warrants the simpler treatment.

---

## 7. Design Space and DOE Strategy

### Global envelope (used for surrogate inference)

| Parameter | Min | Max | Units |
|---|---|---|---|
| α | 0 | 16 | degrees |
| Re | 5×10⁵ | 3×10⁶ (5×10⁶ for D) | — |
| t/c | 0.08 | 0.24 | — |

### Per-regime sampling

Samples are drawn via Latin Hypercube Sampling independently in each regime's
bounding box, then concatenated:

| Regime | N | Train | Test |
|---|---|---|---|
| A | 80 | 64 | 16 |
| B | 25 | 20 | 5 |
| C | 30 | 24 | 6 |
| D | 40 | 32 | 8 |
| **Total** | **175** | **140** | **35** |

The 80/20 train/test split is stratified by regime, so each regime contributes
its 20% to the test set. This is critical — a random global split would give
an empty C test set under bad luck and prevent per-regime error reporting.

`samples.csv` columns: `[case_id, alpha_deg, Re, thickness, regime]`.
`random_state = 42` for both LHS and the split.

### Rationale for the allocation

- Regime A is largest because it covers the broadest practical operating range
  and gives the surrogate the most leverage in the most-used part of the
  design space.
- Regime B is the smallest CFD-cost-driven allocation (25 samples, reduced
  from an initial 50): each near-stall case runs at 300k–1M cells, by far the
  most expensive in the study, and the surrogate is expected to be less
  accurate near stall regardless of sample count. 25 maximin-LHS points still
  cover its narrow `(α, Re, t)` box adequately.
- Regime C is smallest because the regime is narrowest in Re and the CFD
  setup is the most fragile.
- Regime D is medium because each sample is cheap (smallest mesh, fast
  convergence) but the physical regime is narrow.

---

## 8. Output Targets

### Phase 1 — Scalar outputs

| Output | Source | Notes |
|---|---|---|
| Cl | OpenFOAM `forceCoeffs` | Primary |
| Cd | OpenFOAM `forceCoeffs` | Primary |
| L/D | Derived | Secondary |

### Phase 2 — Field outputs (optional)

| Output | Source | Notes |
|---|---|---|
| Cp(x/c) | `singleGraph` on aerofoil patch | Needs POD/PCA before surrogate training |

---

## 9. Toolchain

| Stage | Tool |
|---|---|
| DOE | `pyDOE2` |
| Geometry | `numpy` |
| Meshing | `gmsh` Python API |
| CFD | OpenFOAM 12 — `foamRun` |
| Turbulence | `SpalartAllmaras`, `kOmegaSST`, `kkLOmega` |
| Templating | `Jinja2` |
| Parallelism | GNU `parallel` |
| Surrogates | `scikit-learn`, `smt` |
| Sensitivity | `SALib` |
| Plots | `matplotlib`, `seaborn` |
| Environment | `micromamba` |

---

## 10. Pipeline — 10 Stages

### Stage 1 — `01_generate_doe.py`

**Inputs:** none. **Outputs:** `samples.csv`, `train_idx.npy`, `test_idx.npy`.

```python
for regime, n, bounds in [
    ("A", 80, ((0,8),  (1.5e6,3e6), (0.10,0.18))),
    ("B", 25, ((10,16),(1e6,3e6),   (0.12,0.24))),
    ("C", 30, ((0,8),  (3e5,1e6),   (0.08,0.15))),
    ("D", 40, ((0,6),  (2e6,5e6),   (0.10,0.18))),
]:
    unit = pyDOE2.lhs(3, samples=n, criterion="maximin", random_state=42)
    samples = scale_to_bounds(unit, bounds)
    df = pd.DataFrame(samples, columns=["alpha_deg","Re","thickness"])
    df["regime"] = regime
    chunks.append(df)
all_samples = pd.concat(chunks).reset_index(drop=True)
all_samples.insert(0, "case_id", [f"case_{i:04d}" for i in range(len(all_samples))])
```

Stratified 80/20 split:

```python
from sklearn.model_selection import train_test_split
train_idx, test_idx = train_test_split(
    all_samples.index.values, test_size=0.20,
    stratify=all_samples["regime"], random_state=42
)
```

### Stage 2 — `02_classify_regime.py`

Inference-time classifier. At DOE time the regime is known by construction,
so this stage exposes a function used by the surrogate for new query points
and by `08_validate_regimes.py` for OOD probes. See `CLAUDE.md` §11 for the
priority rule.

### Stage 3 — `03_generate_geometry.py`

Symmetric NACA 4-digit profile, 200 cosine-spaced points, closed polygon
(upper TE→LE then lower LE→TE). Writes `aerofoil.dat` and `params.json` per
case directory. Identical to the original Stage 2.

### Stage 4 — `04_generate_mesh.py`

Mesh strategy is regime-specific. The script reads `samples.csv`, picks the
regime, computes the first cell height for the regime's y+ target, and uses
a regime-specific gmsh recipe:

```python
Y_PLUS = {"A": 30.0, "B": 0.5, "C": 0.5, "D": 50.0}
LAYERS = {"A": (15, 1.20), "B": (35, 1.10), "C": (40, 1.08), "D": (15, 1.25)}
WAKE_REFINE = {"A": "medium", "B": "strong", "C": "strong", "D": "light"}
```

Mesh accepted only when `checkMesh` passes: non-orthogonality < 70°, skewness
< 4, all cells valid. Achieved mesh metrics are written into the
`case_metadata.json` ahead of CFD execution.

### Stage 5 — `05_prepare_case.py`

Copies `templates/regime_{X}/` to `cases/case_{i:04d}/` and renders all
`.jinja` files with Jinja2. Variables:

```python
context = {
    "UX":        f"{Ux:.6f}",
    "UY":        f"{Uy:.6f}",
    "UINF":      f"{U_inf:.6f}",
    "LIFTDIR_X": f"{lift_x:.6f}",  "LIFTDIR_Y": f"{lift_y:.6f}",
    "DRAGDIR_X": f"{drag_x:.6f}",  "DRAGDIR_Y": f"{drag_y:.6f}",
    "NU":        f"{nu:.6e}",
    "K0":        f"{k0:.6e}",       # SST and transition only
    "OMEGA0":    f"{omega0:.6e}",
    "NUT0":      f"{nut0:.6e}",
    "NUTILDA0":  f"{nutilda0:.6e}", # SA only
    "KL0":       f"{kl0:.6e}",      # kkLOmega only
    "END_TIME":  END_TIME[regime],
}
```

### Stage 6 — `06_run_cfd.py`

GNU parallel over all prepared cases:

```bash
parallel -j 4 \
  "cd {1} && source /opt/openfoam12/etc/bashrc && foamRun > log.foamRun 2>&1" \
  ::: cases/case_*/
```

The Python wrapper logs per-case runtime, captures the parallel job's exit
status, and re-queues failed cases at lower relaxation (a built-in retry hook
for Regime B which is most prone to divergence).

### Stage 7 — `07_harvest_results.py`

Per case:
- Parse `postProcessing/forceCoeffs/0/coefficient.dat` (skip `#`-prefixed lines)
- Compute mean and std of Cl and Cd over the last 200 rows
- Parse `log.foamRun` for residual history and any `FOAM FATAL ERROR`
- Compute achieved y+ from the wall-shear post-processing (function object or
  `wallShearStress` utility) and store min/mean/max
- Compute total cell count from `polyMesh/owner`
- Apply regime-aware convergence criteria (see `CLAUDE.md` §8) and write
  `case_metadata.json`

Build `dataset_clean.csv` containing only converged cases:

```
case_id, alpha_deg, Re, thickness, regime, Cl, Cd, L_over_D,
Cl_std, Cd_std, y_plus_mean, cells, runtime_s, converged
```

### Stage 8 — `08_validate_regimes.py`

Per-regime validation against canonical NACA0012 references. Each regime has
a small dedicated case set under `validation/regime_{X}/cases/` and the
script:

1. Runs each validation case using the regime's template
2. Computes Cl, Cd, and (for Phase 2) Cp(x/c)
3. Compares against reference values stored in
   `validation/regime_{X}/references/*.csv`
4. Writes `validation/regime_{X}/report.md` with a pass/fail decision

Reference sources:

| Regime | Reference |
|---|---|
| A | Abbott & von Doenhoff, *Theory of Wing Sections*; NASA TMR SA NACA0012 case |
| B | NASA TMR k-ω SST NACA0012; AGARD experimental data |
| C | XFOIL with eN transition prediction; experimental low-Re databases |
| D | NASA TMR SA at Re = 4×10⁶ |

Acceptance tolerance: |ΔCl| ≤ 5%, |ΔCd| ≤ 10%, qualitative Cp match.

### Stage 9 — `09_train_surrogates.py`

Trains four model families on the merged dataset with one-hot regime. Save
`preprocessor.joblib`, `gp_Cl.joblib`, `gp_Cd.joblib`, `rf_Cl.joblib`,
`rf_Cd.joblib`, `mlp_Cl.joblib`, `mlp_Cd.joblib`, `krg_Cl.joblib`,
`krg_Cd.joblib`. Hyperparameters tuned via cross-validation on the training
set only.

### Stage 10 — `10_global_validation.py`

Final figures and tables:

| Output | Description |
|---|---|
| `parity_plots.png` | 4×2 grid (model × output), points colored by regime |
| `surrogate_metrics.csv` | global + per-regime R², RMSE, MAE for all 8 model-output pairs |
| `sobol_sensitivity.png` | first-order and total Sobol indices for Cl and Cd |
| `ood_test.png` | Re = 4×10⁶ sweep (above Regime A's bound, inside D's bound) with all models + GP uncertainty band |
| `learning_curve.png` | RMSE vs N_train ∈ {10,20,30,40,60,80,120,140} for GP and RF |
| `stall_extrapolation.png` | Cl vs α ∈ [0°, 18°] with dashed line at α = 16° (training boundary) |

---

## 11. Per-Regime CFD Strategy

A summary of the per-regime CFD recipe (full implementation rules in
`CLAUDE.md` §5, §7, §8):

| Property | Regime A | Regime B | Regime C | Regime D |
|---|---|---|---|---|
| Turbulence | `SpalartAllmaras` | `kOmegaSST` | `kkLOmega` | `SpalartAllmaras` |
| Wall functions | Yes (Spalding) | No (low-Re) | No (low-Re) | Yes (Spalding) |
| Target y+ | 30 | 0.5 | 0.5 | 50 |
| Prism layers | 15–20 | 30–40 | 35–45 | 12–18 |
| Cells | 80k–200k | 300k–1M | 500k–1.2M | 50k–150k |
| End-time | 2000 | 5000 | 4000 | 2000 |
| Relaxation (U / p) | 0.7 / 0.3 | 0.5 / 0.2 | 0.5 / 0.2 | 0.7 / 0.3 |
| `nNonOrth. corr.` | 1 | 2 | 2 | 1 |
| Cl/Cd std tolerance | 0.005 | 0.010 | 0.005 | 0.005 |

For each regime the template lives in `templates/regime_{X}/` and renders
only the placeholders relevant to that regime's turbulence model.

---

## 12. Validation Strategy and Phases

Validation is sequenced. **No regime's samples enter the main dataset until
that regime's template is locked.**

| Phase | Regime | Output | Blocks |
|---|---|---|---|
| 1 | A | `validation/regime_A/report.md` | All other phases |
| 2 | B | `validation/regime_B/report.md` | Phases 5, 6 (for B) |
| 3 | C | `validation/regime_C/report.md` | Phases 5, 6 (for C) |
| 4 | D | `validation/regime_D/report.md` | Phases 5, 6 (for D) |
| 5 | All | `dataset_clean.csv` (175 cases) | Phase 6 |
| 6 | All | `results/` | — |

**Phase 1 deliverables (Regime A):**

- Validated NACA0012 case set at:
  - Re = 2×10⁶, α ∈ {0°, 4°, 8°}
  - Re = 3×10⁶, α = 4°
- Cl, Cd within ±5%/±10% of Abbott & von Doenhoff and NASA TMR references
- Mesh independence study: 3 mesh levels (coarse, medium, fine); Cl change < 1% between medium and fine
- Convergence study: residuals plateau below 1e-5; force coefficients stable to ±0.5%
- Report committed to `validation/regime_A/report.md`

A regime that fails to validate within reasonable effort is documented as a
known gap in `regime_validation/report.md` and excluded from the dataset.

---

## 13. Surrogate Training Strategy

### Input encoding

```python
X = [alpha_deg, Re, thickness, regime_A, regime_B, regime_C, regime_D]  # one-hot regime
```

Continuous columns are standardized; one-hot columns are passed through.
The preprocessor is saved to `models/preprocessor.joblib` and is part of the
model artefact.

### Train/test discipline

The 35 test samples in `test_idx.npy` are sacred:

- Never train on them
- Never use them to select hyperparameters (use CV on the training set only)
- Never re-run CFD based on test set performance
- Report final metrics on the test set exactly once at the end

The split is stratified by regime so every regime is represented in test.

### Models

| Model | Key settings |
|---|---|
| GP | `Matern(ν=2.5) + WhiteKernel`, `n_restarts_optimizer=10`, `normalize_y=True` |
| RF | `n_estimators=200`, `random_state=42` |
| MLP | `hidden_layer_sizes=(64,64,32)`, ReLU, `early_stopping=True`, `max_iter=2000` |
| KRG | `smt.surrogate_models.KRG(theta0=[1e-2])` |

One model per output (Cl, Cd), per family — eight models total.

### Reporting

Report global and per-regime R², RMSE, MAE in `results/surrogate_metrics.csv`.
A model is acceptable when each per-regime R² ≥ 0.90 for Cl and ≥ 0.85 for Cd.
Global R² alone is misleading because regimes with more samples dominate it.

---

## 14. Surrogate Comparison Matrix

| Property | Gaussian Process | Random Forest | MLP | Kriging (smt) |
|---|---|---|---|---|
| Uncertainty quantification | Yes (native) | Approximate (trees) | No | Yes (predict_variances) |
| Works with small N | Very good | Good | Poor below ~120 samples | Very good |
| Interpolates exactly | Yes | No | No | Yes |
| Extrapolation | Reverts to prior, high std | Flat (mean of training) | Unpredictable | Smooth, low confidence |
| Training cost | O(N³) — slow above N=1000 | Fast | Slow (many epochs) | O(N³) |
| Hyperparameter sensitivity | Medium (kernel choice) | Low | High | Low |
| Regime one-hot tolerance | Handles well | Handles well | Sensitive to scaling | Handles well |
| Best for this project | Primary uncertainty-aware model | Robust baseline | Comparison | Kriging baseline |

---

## 15. Metadata Schema

Every case writes `case_metadata.json`:

```json
{
  "case_id": "case_0042",
  "regime": "A",
  "alpha_deg": 4.123,
  "Re": 2.1e6,
  "thickness": 0.12,

  "template": "regime_A",
  "turbulence_model": "SpalartAllmaras",
  "wall_treatment": "nutUSpaldingWallFunction",

  "y_plus_target": 30.0,
  "y_plus_min": 18.4, "y_plus_mean": 31.7, "y_plus_max": 47.2,

  "cells_total": 142308,
  "non_orthogonality_max": 38.4,
  "skewness_max": 1.21,

  "runtime_s": 412.6,
  "iterations": 2000,
  "residuals_final": {"Ux": 3.1e-6, "Uy": 4.8e-6, "p": 7.2e-6, "nuTilda": 9.1e-7},

  "Cl_mean": 0.456, "Cl_std": 0.0012,
  "Cd_mean": 0.0098, "Cd_std": 0.00008,
  "L_over_D": 46.5,

  "converged": true, "ood": false,
  "notes": ""
}
```

The dataset CSV is the joined projection of all `case_metadata.json` files;
the per-case JSON is the source of truth for everything else.

---

## 16. Risk Register

| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| 1 | Regime B (near-stall) does not converge in steady RANS | High | Lose ~14% of dataset (25/175) | Tighter relaxation; longer end-time; URANS as fallback; document failures rather than fake data |
| 2 | `kkLOmega` unavailable in the local OpenFOAM 12 build | Medium | Regime C blocked | Fall back to `kOmegaSST` with low Tu and 0.1% turbulence intensity; log the substitution in metadata |
| 3 | Validation references for low-Re NACA0012 are sparse | Medium | Phase 3 hard to lock | Cross-check against XFOIL with eN transition prediction |
| 4 | Compute budget — Regime B/C cases at up to ~1M cells (B=25, C=30 samples) dominate wall-clock time | Medium | Long wall-clock time | Cap parallel jobs; B reduced 50→25 to bound cost; budget per regime; checkpoint runs |
| 5 | One-hot regime collinearity (sum-to-one) for linear submodels | Low | Numerical instability in some sklearn models | Drop one column for OLS-like models; not a problem for GP/RF/MLP/KRG |
| 6 | Stratified split too small in Regime C (6 test samples) | Medium | Noisy per-regime metric | Report Regime C metrics with confidence intervals; bootstrap |
| 7 | Mesh independence not established for Regime D's coarse mesh | Medium | Systematic bias in D | Run 3-mesh study in Phase 4 validation |
| 8 | Classifier ambiguity for points between regime boxes | Low | Wrong template used for OOD query | `ood` flag in metadata; refuse to write a CFD case for OOD points |
| 9 | Random seed leakage during CV hyperparameter search | Low | Test set contamination | Code review of `09_train_surrogates.py`; CI test that test indices never reach `fit` |
| 10 | Sample exhaustion in active-learning loop | Low | Budget overrun in Phase 8 (future) | Pre-set max iterations; require manual approval to extend |

---

## 17. Known Limitations

These are the failure modes the project is set up to encounter and document:

1. **Stall discontinuity.** Cl drops sharply past stall. Steady RANS itself
   models this poorly; the surrogate cannot recover information that is not
   in the training data.

2. **Extrapolation.** All surrogates degrade outside the training envelope.
   GP signals it via high uncertainty; RF and MLP do not.

3. **Sample efficiency.** Too few samples → high variance surrogate. Too many
   → wasted CFD compute. The learning-curve experiment (Stage 10) quantifies
   this per regime.

4. **Geometry parameterization limit.** NACA 4-digit is a restricted family.
   A surrogate trained on it cannot generalize to NACA 6-series, supercritical,
   or arbitrary aerofoils. Phase 2 adds camber `(m, p)` parameters.

5. **Physics violations.** The surrogate may predict negative Cd or non-zero
   Cl for a symmetric aerofoil at α = 0°. Physics-informed approaches are
   future work.

6. **RANS error inherited by the surrogate.** Training data has its own error
   (RANS underpredicts separation, LES/DNS would give different Cl/Cd). The
   surrogate inherits and potentially amplifies this error. Each regime's
   validation report (`validation/regime_X/report.md`) documents the
   expected CFD error envelope for that regime.

7. **Regime-boundary smoothness.** The one-hot encoding produces a piecewise
   model that may exhibit discontinuities at regime boundaries. This is
   monitored in Stage 10 via cross-regime probe sweeps.

---

## 18. Directory Structure

```
NACASurrogate/
├── README.md                     ← quick design summary
├── README_complete.md            ← this file
├── CLAUDE.md                     ← LLM operating manual
├── environment.yml
├── samples.csv                   ← 175 × [case_id, alpha_deg, Re, thickness, regime]
├── train_idx.npy                 ← 140 stratified training indices
├── test_idx.npy                  ← 35 stratified test indices
├── dataset_clean.csv             ← harvested converged CFD results
│
├── scripts/
│   ├── 01_generate_doe.py
│   ├── 02_classify_regime.py
│   ├── 03_generate_geometry.py
│   ├── 04_generate_mesh.py
│   ├── 05_prepare_case.py
│   ├── 06_run_cfd.py
│   ├── 07_harvest_results.py
│   ├── 08_validate_regimes.py
│   ├── 09_train_surrogates.py
│   └── 10_global_validation.py
│
├── templates/                    ← regime-specific OpenFOAM templates (Jinja2)
│   ├── regime_A/
│   ├── regime_B/
│   ├── regime_C/
│   └── regime_D/
│
├── validation/
│   ├── regime_A/{cases/, references/, report.md}
│   ├── regime_B/{cases/, references/, report.md}
│   ├── regime_C/{cases/, references/, report.md}
│   └── regime_D/{cases/, references/, report.md}
│
├── cases/                        ← auto-generated, gitignored
│   └── case_NNNN/
│       ├── aerofoil.dat
│       ├── params.json
│       ├── case_metadata.json
│       └── <OpenFOAM case>
│
├── models/                       ← joblib artefacts
│   ├── preprocessor.joblib
│   ├── gp_Cl.joblib  / gp_Cd.joblib
│   ├── rf_Cl.joblib  / rf_Cd.joblib
│   ├── mlp_Cl.joblib / mlp_Cd.joblib
│   └── krg_Cl.joblib / krg_Cd.joblib
│
└── results/
    ├── regime_validation/        ← per-regime CFD validation reports
    ├── surrogate_metrics.csv     ← global + per-regime metrics
    ├── parity_plots.png
    ├── sobol_sensitivity.png
    ├── ood_test.png
    ├── learning_curve.png
    └── stall_extrapolation.png
```

---

## 19. Implementation Sequence

The project is built phase by phase. Each phase has a definite deliverable.

### Phase 1 — Regime A lock (active)

1. Build `templates/regime_A/` from `$FOAM_TUTORIALS/incompressibleFluid/airFoil2D/` and `fluid/aerofoilNACA0012Steady/` reference cases
2. Write `04_generate_mesh.py` (Regime A branch only) with y+ = 30 target
3. Write `05_prepare_case.py` (Regime A branch only) — Jinja2 render of SA-specific fields
4. Run NACA0012 validation cases: Re = 2×10⁶ at α ∈ {0°, 4°, 8°} and Re = 3×10⁶ at α = 4°
5. Mesh independence study (coarse, medium, fine)
6. Write `validation/regime_A/report.md`
7. **Gate:** every validation case within ±5%/±10% of reference Cl/Cd

### Phase 2 — Regime B lock

1. Build `templates/regime_B/` from `fluid/aerofoilNACA0012Steady/`
2. Extend `04_generate_mesh.py` with the Regime B branch (y+ = 0.5, dense prism layers)
3. Extend `05_prepare_case.py` to handle kOmegaSST templates
4. Run NACA0012 validation cases: Re = 2×10⁶ at α ∈ {12°, 14°, 16°}
5. Mesh and convergence studies
6. Document Regime B in its report

### Phase 3 — Regime C lock

1. Build `templates/regime_C/` for `kkLOmega` (or fallback)
2. Extend `04_generate_mesh.py` with the Regime C branch
3. Extend `05_prepare_case.py` to handle transition templates
4. Run low-Re NACA0012 validation: Re = 5×10⁵ at α ∈ {2°, 4°, 6°}
5. Compare to XFOIL eN predictions and experimental low-Re databases

### Phase 4 — Regime D lock

1. Build `templates/regime_D/` (SA + wall functions, coarse mesh)
2. Extend `04_generate_mesh.py` with the Regime D branch
3. Run NASA TMR Re = 4×10⁶ validation

### Phase 5 — Full dataset

1. Run `01_generate_doe.py` → `samples.csv`
2. Run `03–06` over all 175 samples in parallel
3. Run `07_harvest_results.py` → `dataset_clean.csv`
4. Audit: every case must have a `case_metadata.json` with `converged=true`
   OR a documented reason in the report

### Phase 6 — Surrogate

1. `09_train_surrogates.py` — train 8 models on 140 training samples
2. `10_global_validation.py` — full figure suite
3. Audit per-regime metrics; document gaps

---

## 20. Future Extensions

In rough order of value:

1. **Cp surrogate.** Sample Cp(x/c) on the aerofoil patch via `singleGraph`,
   compress with POD (5–10 modes), train a separate Cp surrogate. Provides
   the full pressure distribution for downstream design tools.
2. **Active learning.** Use GP uncertainty (or KRG variance) to query the most
   informative next CFD point. Implement Expected Improvement acquisition;
   compare to passive LHS.
3. **Camber parameters.** Phase 2 adds `m` (max camber) and `p` (camber
   position) to the design space, scaling samples to 400–500 LHS points.
4. **Multi-fidelity surrogate.** Combine cheap XFOIL evaluations with
   expensive OpenFOAM evaluations using co-kriging (smt's MFK).
5. **URANS for Regime B.** When steady RANS fails to converge near stall,
   fall back to URANS with time-averaged Cl/Cd. Marks the case with a fidelity
   flag in metadata.
6. **Physics-informed regularization.** Constrain the surrogate to satisfy
   Cl(α=0; symmetric) = 0 and Cd > 0 either via PINN-style penalties or via
   post-hoc projection.
7. **Classification-assisted surrogate.** Train a soft regime classifier
   (logistic regression or small NN) on `(α, Re, t)` and use its probabilities
   to weight per-regime expert models, replacing the hard one-hot.
8. **3D extension.** Move from 2D infinite-wing to 3D finite-wing with sweep,
   taper, and twist parameters — requires complete pipeline rewrite for
   surface meshing.

---

*End of documentation.*
