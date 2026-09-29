# CLAUDE.md — Development Guide for CamberLab (AirfRANS surrogate)

This file tells Claude (or any LLM) how to work on the CamberLab project
correctly. Read this entire file before writing any code or suggesting any
shell commands.

---

## 1. Project Summary

CamberLab predicts the lift and drag coefficients (C_L, C_D) of NACA 4- and
5-digit aerofoils from angle of attack, Reynolds number and section geometry,
plus the wall pressure and skin-friction distributions Cp(x/c) and Cf(x/c),
using fast surrogates (GP, RF, MLP, Kriging) trained on **AirfRANS**
(Bonnet et al., NeurIPS 2022): 1,000 steady 2D incompressible RANS
simulations (OpenFOAM, k-ω SST), Re 2–6×10⁶, α −5° to 15°, chord 1 m.

It deliberately regresses the **coefficients** (and wall curves), not the
flow field: the AirfRANS paper's field models predict drag poorly, and beating
their drag prediction is the headline comparison
(`results/airfrans_benchmark.md`). C_D always comes from the scalar model;
never integrate predicted curves for drag.

The project previously generated its own OpenFOAM data with a four-regime
design (regime classifier, one-hot regime features, per-regime templates).
That pipeline is **retired** and archived in `legacy/regime_v1/`. Do not
revive regimes, regime features or regime templates in the active code.

Final report: `results/airfrans_report.md`. User-facing summary: `README.md`.

---

## 2. Environment — uv + `requirements.txt`

**All Python commands and scripts must be run inside the project virtual
environment at `.venv/`. Never use system Python.**

There is no conda/micromamba environment and no `environment.yml`: every
dependency installs from PyPI. `.python-version` pins Python 3.11, which
`uv venv` picks up automatically (uv downloads the interpreter if needed).

### Setup (first time)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # if uv is not installed
uv venv                                           # creates .venv on Python 3.11
uv pip install -r requirements.txt
```

### Running scripts

```bash
uv run python scripts/01_generate_doe.py
# or
source .venv/bin/activate
python scripts/01_generate_doe.py
```

`uv run` resolves `.venv` without activation — prefer it in automation, docs,
and script docstrings.

### Adding a new dependency

```bash
uv pip install <package>
# Then add the pinned-enough requirement to requirements.txt manually
```

The dependency must be installable from PyPI. If a package only exists on
conda-forge, do not reintroduce conda — find a PyPI equivalent or vendor the
functionality.

### Never do

```bash
pip install ...          # outside .venv (bare/system pip)
python ...               # using system Python
conda activate ...       # no conda/micromamba in this project
```

---

## 3. Data Source — AirfRANS

- Variant: **`PLAID-datasets/AirfRANS_clipped`** (Hugging Face, 18 GB on
  disk / ~36 GB unpacked, 1,000 samples of ~35 MB, PLAID/CGNS samples in 72
  parquet shards). It keeps the original wall-resolved mesh (~180k nodes),
  which the wall shear stress needs. `AirfRANS_remeshed` (611 MB) holds the
  same samples, scalars and splits and still resolves via `--data-dir`, but
  its coarse wall under-resolves shear (~40% of the shear drag). Licence: ODbL
  1.0 (© Safran) — keep the attribution in the app footer and README.
- The user downloads it once; **code never downloads data** and must not
  re-download silently:

  ```bash
  huggingface-cli download PLAID-datasets/AirfRANS_clipped \
    --repo-type dataset --local-dir data/airfrans_clipped
  ```

- Every script resolves the location as `--data-dir` > `$AIRFRANS_DIR` >
  `data/airfrans_clipped` (`scripts/airfrans/io.py::resolve_data_dir`).
  `data/` is gitignored.
- **Memory:** stream samples one at a time via `io.iter_samples` /
  `RawSampleReader`, which keeps one shard open (`pre_buffer=False`). Never
  hold several `pq.ParquetFile`s open: each retains its shard's read buffers,
  which crashed the 8 GB WSL machine. Peak RSS is ~1.8 GB.
- Samples are pickled PLAID sample dicts. They deserialise **only** with
  `pyplaid==0.1.7` (0.1.8+ and 1.x reject the schema); PLAID's PyPI name is
  `pyplaid` (not `plaid` / `plaid-lib`, which are unrelated packages). The
  single `pickle.loads` call lives in `scripts/airfrans/io.py::deserialise`
  — the one sanctioned exception to "no pickle".
- Per-sample scalars: `angle_of_attack` (**radians**), `inlet_velocity`
  (m/s), `C_L`, `C_D`. Nodal fields: `p` (kinematic, relative to the far
  field), `Ux`, `Uy`, `nut`, `implicit_distance`. **No wall shear stress is
  stored**; `scripts/airfrans/surface.py` computes it (P1 gradients
  area-averaged onto wall nodes, τ = 2ν·dev(S)·n with the paper's
  ν = 1.56×10⁻⁵). No original simulation name is stored; `sample_id` is the
  row index in `all_samples`.
- Re = U∞·c/ν with c = 1 m and ν(298.15 K) = 1.5498×10⁻⁵ m²/s from AirfRANS'
  polynomial. AirfRANS' own nominal Re is ~0.6% lower (they effectively use
  ν ≈ 1.56×10⁻⁵); this is a constant factor and does not affect models.
- The aerofoil wall is the mesh boundary loop off the clip box (topological,
  in `scripts/airfrans/geometry.py::extract_surface`), cross-checked against
  `implicit_distance ≈ 0`.

## 4. Pipeline and Feature Schema

| Stage | Script | Output |
|---|---|---|
| Inspect (Gate 2) | `scripts/airfrans/inspect_dataset.py` | `results/airfrans_inspection.md` |
| Ingest (Gate 3) | `scripts/20_ingest_airfrans.py` | `results/airfrans_dataset.csv`, `results/airfrans_surfaces.npz`, `results/airfrans_wall_curves.npz`, `results/airfrans_wall_native.npz` (gitignored) |
| QA (Gate 4) | `scripts/21_qa_airfrans.py` | `results/airfrans_qa_flags.csv`, `results/airfrans_qa.md` |
| Splits | `scripts/22_make_splits.py` | `splits/airfrans_{task}_{train,test}_idx.npy` |
| Wall gate | `scripts/23_surface_gate.py` | `results/airfrans_surface_gate.{md,json}` |
| Train (Gate 6) | `scripts/09_train_surrogates.py --task all [--outputs coeffs\|curves\|all]` | `models/{task}/`, `models/{task}/curves/` |
| Evaluate | `scripts/10_global_validation.py` | `results/airfrans_metrics.csv`, `results/airfrans_curve_metrics.csv`, `results/airfrans_benchmark.md` |
| Query | `scripts/predict.py`, `scripts/11_sweep_curves.py`, `app.py` | — |

Model inputs (`scripts/surrogate/data.py::FEATURES`), standardised with a
`StandardScaler` saved as `models/{task}/preprocessor.joblib`:

```
X = [alpha_deg, log10_Re, t_max, x_tmax, m_max, x_m]      # shape (N, 6)
```

- Geometry features come from `section_features()` applied to the wall
  points at ingestion and to `naca_coordinates(code)` at inference — **the
  same function on both sides**. Never compute query features any other way.
- `m_max` is measured from the geometric chord (nose min-x point to TE), so
  it reads below nominal NACA camber for cambered sections; this is
  consistent between training and inference. `x_m = 0` for symmetric
  sections.
- Targets: `Cl` and `log(Cd)`; Cd predictions are back-transformed with
  `exp`. Metrics are always reported on Cd, not log Cd.
- Curve targets: `Cp` and `Cf` on 101 cosine-spaced x/c stations per surface
  (upper then lower = 202 values), from `surface.curves()`. Cp = p/q∞; Cf =
  τ·t/q∞ with t pointing LE → TE along each surface, so Cf < 0 is reversed
  flow. The ingest also stores `Cl_int_{p,tau}` / `Cd_int_{p,tau}`: C_L and
  C_D re-integrated from the wall, which the wall gate checks against the
  stored values.

## 5. Data Integrity and Split Rules

- **Never edit, drop, impute or fabricate rows** of
  `results/airfrans_dataset.csv`. QA outcomes go to the separate
  `results/airfrans_qa_flags.csv` (`qa_pass`, `qa_reason`); excluded rows
  are filtered at split time, never deleted. Outliers are listed, never
  auto-dropped.
- Splits are the **official AirfRANS memberships** from the dataset card
  (`full` 800/200, `scarce` 200/200 sharing the `full` test set, `reynolds`
  504/496, `aoa` 804/196). They are frozen: `22_make_splits.py` refuses to
  overwrite an index file with different contents.
- Test indices are sacred: never used for fitting, model selection or
  tuning. Hyperparameters (RF `min_samples_leaf`, MLP patience/L2) are
  chosen by 5-fold CV on the training rows only. `10_global_validation.py`
  is the only script that reads test indices.

---

## 6. Python Code Rules

### File structure

```python
#!/usr/bin/env python3
"""
Script: <filename>
Stage:  <stage number and name>
Purpose: <one sentence>

Usage:
    uv run python scripts/<filename>
"""
```

### Paths

All paths must be relative to the project root. Use `pathlib.Path`, never `os.path`:

```python
from pathlib import Path

PROJECT_ROOT  = Path(__file__).resolve().parent.parent
RESULTS_DIR   = PROJECT_ROOT / "results"
MODELS_DIR    = PROJECT_ROOT / "models"
SPLITS_DIR    = PROJECT_ROOT / "splits"
SCRIPTS_DIR   = PROJECT_ROOT / "scripts"
DATA_DIR      = PROJECT_ROOT / "data" / "airfrans_remeshed"   # or --data-dir / $AIRFRANS_DIR
```

### Random seeds

All stochastic operations use `random_state=42` or `np.random.seed(42)`.
Never use unseeded randomness.

### Logging

```python
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(message)s")
log = logging.getLogger(__name__)

log.info("Ingesting 1000 AirfRANS samples...")
log.warning("NACA 2412 at Re=1e7 is outside the training envelope")
log.error("Gate 4 failed: 3.1% of rows fail hard checks")
```

### Data types

- All parameter arrays: `np.float64`
- Sample / split indices: `np.int32`
- Saved models: `joblib.dump` / `joblib.load` (not pickle directly)

---

## 7. Surrogate Modeling Rules

- Families (`scripts/surrogate/models.py`), one model per target per family:
  - **GP**: anisotropic `Matern(nu=2.5) * ConstantKernel + WhiteKernel`,
    `normalize_y=True`, `n_restarts_optimizer=5`.
  - **RF**: 500 trees, `min_samples_leaf` by CV from {1, 2, 4}.
  - **MLP**: `(128, 128, 64)`, `early_stopping=True`, wrapped in
    `TransformedTargetRegressor(StandardScaler)`; patience and L2 by CV.
  - **KRG**: SMT `KRG`, anisotropic θ, `eval_noise=True`.
- **Curves** (`scripts/surrogate/curves.py`): one PCA per quantity, fitted on
  train rows only, with a fixed 20 modes (`N_MODES`; the 5-fold CV
  reconstruction RMSE per k is recorded, not used to choose). k = 20 limits
  only the Cf separation point (~0.015 c truncation floor); don't raise it
  without checking by train CV that the GP realises the gain. GP/KRG fit one
  model per mode (σ per mode → ±2σ band,
  independent modes); RF/MLP fit one multi-output model. Cf curves are
  trained only if `airfrans_surface_gate.json` has `cf_pass`. Saved to
  `models/{task}/curves/{pca,family}_{Cp,Cf}.joblib` + `train_pred.npz`,
  with `curve_info` in `envelope.json`. For `full`, only PCA, GP and MLP
  curve models are committed; KRG (354 MB/quantity) and RF (86 MB) are
  gitignored.
- Persist with `joblib.dump(..., compress=3)` to
  `models/{task}/{family}_{Cl,Cd}.joblib`, plus `preprocessor.joblib`,
  `envelope.json` (train feature ranges, package versions, fit metadata) and
  `train_pred.npz` (Gate 6 reload check). `scikit-learn==1.8.*` and
  `smt==2.15.0` are pinned because the pickles only reload under the
  versions they were saved with — retrain whenever you bump either.
- Inference (`scripts/surrogate/inference.py`):
  `predict(alpha_deg, Re, naca, family="gp", task="full")` →
  `{Cl, Cd, L_over_D, Cl_std, Cd_std, in_envelope, warnings}`.
  Out-of-envelope queries return a prediction **with a warning**, never
  silently and never refused. σ is reported for GP and KRG only.
  `predict_surface(alpha_deg, Re, naca, family, task)` returns the long
  Cp/Cf table (`surface`, `x_c`, `Cp`, `Cf`, `_lo`/`_hi` for GP/KRG);
  `has_curve_models(family, task)` says whether a family's curve models
  exist locally.
- Report R², RMSE, MAE, Spearman ρ and the paper's relative error
  `|(true − pred)/true|` (mean **and** median — the mean blows up for Cl
  near zero). The paper's `mean_score_force` is a raw ratio, not a
  percentage.
- Curve metrics: RMSE, station-mean R², ±2σ coverage, suction-peak MAE,
  separation detection and x/c error, the paper's `mean_rel_p` /
  `mean_rel_wss` on native wall nodes, and C_L/C_D integrated from the
  predicted curves.
- Envelope and limitations to keep in docs: Re 2–6×10⁶, α −5° to 15°, NACA
  4/5-digit only; fully turbulent SST (no transition; Cd biased high vs
  experiment at lower Re); steady RANS near stall (α > ~12°) least reliable;
  accuracy bounded by AirfRANS' own CFD; the computed τ_w runs ~2.5% high
  against AirfRANS' C_D; curve-integrated drag is unreliable.
- The GP marginal likelihood can have several optima. On `reynolds` Cd one
  of them reverses the Re trend (report, failure mode 1). After retraining,
  check that the GP's Cd rises as Re falls before trusting it.

## 8. Tests and Gates

`uv run pytest` must pass. It covers geometry (analytic recovery, mesh-like
resampling), wall physics (exact P1 shear, force integration, Cf sign), the
Gate 3 dataset and wall-curve schema, Gate 6 reloads of Cl/Cd and curve
models, every CLI family (including `--surface`), and a headless run of the
Streamlit app.

The wall gate (`23_surface_gate.py`, decision D4) passes when the median
relative error is ≤ 2% (C_L) / 5% (C_D) and at most 5% of samples exceed 3×
those limits. Cp needs the pressure-only C_L check; Cf needs the full C_D
check. If Cf fails, ship Cp only.

---

## 9. What Not to Do

| Do not | Instead |
|---|---|
| Edit, drop or fabricate dataset rows | Flag in `airfrans_qa_flags.csv`; filter at split time |
| Touch test indices outside `10_global_validation.py` | 5-fold CV on train rows |
| Compute query geometry features ad hoc | `naca_coordinates` + `section_features` |
| Integrate predicted Cp/Cf for drag | Use the scalar C_D model |
| Open every parquet shard at once | `io.iter_samples` (one shard at a time) |
| Commit KRG / RF curve models | They are gitignored; retrain locally |
| Download data from code | Read the local copy; tell the user the download command |
| Upgrade `pyplaid` past 0.1.7 | It cannot read this dataset |
| Bump scikit-learn / smt without retraining | Retrain all tasks, then update the pins |
| Reintroduce regimes / regime one-hot | One unified surrogate over the AirfRANS envelope |
| Use system Python | `uv run python ...` (or activate `.venv`) |
| Use `os.path` | `pathlib.Path` |
| Use `print()` for logging | `logging.info()` / `logging.warning()` |
| Use `fit_transform` on test data | `transform` only on test data |
| Hard-code absolute paths | Use `PROJECT_ROOT` relative paths |
| Use `pickle` directly | `joblib.dump` / `joblib.load` (sole exception: `io.deserialise`) |
| Commit `data/` | It is gitignored |

## 10. Quick Reference — Key Commands

```bash
uv venv && uv pip install -r requirements.txt

uv run python scripts/airfrans/inspect_dataset.py
uv run python scripts/20_ingest_airfrans.py
uv run python scripts/21_qa_airfrans.py
uv run python scripts/22_make_splits.py
uv run python scripts/23_surface_gate.py
uv run python scripts/09_train_surrogates.py --task all            # --outputs coeffs|curves|all
uv run python scripts/10_global_validation.py

uv run python scripts/predict.py --alpha 4 --re 3e6 --naca 2412 --family all
uv run python scripts/predict.py --alpha 4 --re 3e6 --naca 2412 --surface
uv run python scripts/11_sweep_curves.py --naca 23012 --re 4e6
uv run streamlit run app.py
uv run pytest
```

---

## Appendix — Legacy OpenFOAM regime pipeline (retired)

Everything below applies **only** to the archived code in
`legacy/regime_v1/` (OpenFOAM 12 cases, gmsh meshes, per-regime templates).
It is kept for reference if that pipeline is ever revisited; it does not
govern the active AirfRANS code. The regime definitions, classification rule,
case-metadata schema and validation phases are documented in
`legacy/regime_v1/README_complete.md` and in git history.

### L3. OpenFOAM Version

**This project targets OpenFOAM 12 (OpenFOAM Foundation release).**

Do not use OpenFOAM ESI (openfoam.com) syntax — the two forks have diverged.
Foundation release is at [openfoam.org](https://openfoam.org).

OpenFOAM is installed system-level (not a Python package, not in `.venv`).
Source it before any foam commands:

```bash
source /opt/openfoam12/etc/bashrc
```

On this OpenFOAM 12 installation, `simpleFoam` has been superseded by
`foamRun` with `solver incompressibleFluid`. Use `foamRun` in all new
automation and templates.

---

### L4. How to Get Correct OpenFOAM 12 File Syntax

**This is the most important section. Never guess OpenFOAM dictionary syntax.
Always verify using one or more of the three methods below.**

#### Method 1 — Copy from `$FOAM_TUTORIALS`

```bash
echo $FOAM_TUTORIALS                          # typically /opt/openfoam12/tutorials

# Per-regime tutorial anchors:
# Regime A and D (SA + wall functions, attached flow):
ls $FOAM_TUTORIALS/incompressibleFluid/airFoil2D/
ls $FOAM_TUTORIALS/incompressibleFluid/pitzDailySteady/

# Regime B (kOmegaSST, fully resolved walls, separation):
ls $FOAM_TUTORIALS/fluid/aerofoilNACA0012Steady/

# Regime C (transitional kkLOmega):
grep -rl "kkLOmega" $FOAM_TUTORIALS/

# Function objects (forceCoeffs, singleGraph) — used across all regimes:
grep -rl "forceCoeffs" $FOAM_TUTORIALS/
```

**Before writing or editing any OpenFOAM dictionary, open the equivalent
tutorial file for the relevant regime and base your version on it. Do not
write dictionaries from memory.**

#### Method 2 — `foamInfo <keyword>`

```bash
# Schemes / solvers:
foamInfo divSchemes
foamInfo gradSchemes
foamInfo laplacianSchemes
foamInfo SIMPLE
foamInfo GAMG
foamInfo smoothSolver

# Turbulence models used per regime:
foamInfo SpalartAllmaras
foamInfo kOmegaSST
foamInfo kkLOmega

# Boundary condition types:
foamInfo fixedValue
foamInfo inletOutlet
foamInfo freestream
foamInfo nutUSpaldingWallFunction
foamInfo nutkWallFunction
foamInfo nutLowReWallFunction
foamInfo omegaWallFunction
foamInfo kqRWallFunction
foamInfo kLowReWallFunction

# Function objects:
foamInfo forceCoeffs
foamInfo forces
foamInfo singleGraph
```

#### Method 3 — `foamSearch`

```bash
foamSearch $FOAM_TUTORIALS functions liftDir
foamSearch $FOAM_TUTORIALS fvSolution relaxationFactors/equations/U
foamSearch $FOAM_TUTORIALS 0/nut nutUSpaldingWallFunction
foamSearch $FOAM_TUTORIALS 0/nuTilda freestream
foamSearch $FOAM_TUTORIALS constant/momentumTransport SpalartAllmaras
```

#### Verification order for any OpenFOAM keyword

1. Open the relevant regime's tutorial under `$FOAM_TUTORIALS/`
2. Run `foamInfo <keyword>`
3. Run `foamSearch $FOAM_TUTORIALS <file> <keyword>` and read the examples
4. Only then write the dictionary entry

---

### L5. OpenFOAM File Writing Rules

#### Header

Every OpenFOAM file must start with the correct FoamFile header:

```c++
/*--------------------------------*- C++ -*----------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           | Version:  12
     \\/     M anipulation  |
\*---------------------------------------------------------------------------*/
FoamFile
{
    format      ascii;
    class       dictionary;   // or volVectorField, volScalarField, etc.
    location    "system";     // or "constant", "0"
    object      controlDict;  // filename
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
```

The `class` field must match the file type exactly:
- `dictionary` — for `system/` and `constant/` files
- `volVectorField` — for `U`
- `volScalarField` — for `p, k, omega, nut, nuTilda, kl`

#### Indentation and formatting

- 4 spaces for indentation (no tabs)
- Opening brace `{` on the same line as the keyword
- Closing brace `}` on its own line
- Semicolon after every value assignment

#### Turbulence dictionary (`constant/momentumTransport`) — per regime

In OpenFOAM 12 the turbulence dictionary is `constant/momentumTransport`
(not `turbulenceProperties` — that name is from older versions).

**Regimes A and D — Spalart–Allmaras:**

```c++
simulationType  RAS;
RAS
{
    model           SpalartAllmaras;
    turbulence      on;
    printCoeffs     on;
}
```

**Regime B — k-ω SST (fully resolved walls):**

```c++
simulationType  RAS;
RAS
{
    model           kOmegaSST;
    turbulence      on;
    printCoeffs     on;
}
```

**Regime C — k-kL-ω transition model:**

```c++
simulationType  RAS;
RAS
{
    model           kkLOmega;
    turbulence      on;
    printCoeffs     on;
}
```

If `kkLOmega` is not available in the installed build, fall back to
`kOmegaSST` with low-Re wall treatment (no wall functions) and a free-stream
turbulence intensity of ~0.1% to provoke transition. Do NOT silently switch
models — log the substitution in `case_metadata.json`.

#### Field boundary conditions — per regime

**Regimes A and D (SA + wall functions, y+ ≈ 30–80):**

```c++
// 0/nuTilda
internalField   uniform 3e-5;          // ~3*nu free stream
aerofoil { type fixedValue;  value uniform 0; }
inlet    { type freestream;  freestreamValue uniform 3e-5; }
outlet   { type inletOutlet; inletValue uniform 3e-5; value uniform 3e-5; }

// 0/nut
internalField   uniform 0;
aerofoil { type nutUSpaldingWallFunction; value uniform 0; }
inlet    { type calculated;               value uniform 0; }
outlet   { type calculated;               value uniform 0; }
```

`nutUSpaldingWallFunction` is robust across y+ ∈ [1, 300]. For Regime D's
stricter y+ ≈ 30–80 the standard `nutkWallFunction` is equivalent.

**Regime B (k-ω SST, fully resolved, y+ < 1):**

```c++
// 0/k
aerofoil { type kLowReWallFunction; value uniform 1e-10; }

// 0/omega
aerofoil { type omegaWallFunction;  value uniform 1; }

// 0/nut
aerofoil { type nutLowReWallFunction; value uniform 0; }
```

**Regime C (kkLOmega, fully resolved, y+ < 1):**

```c++
// 0/k       — turbulent kinetic energy
aerofoil { type fixedValue; value uniform 1e-10; }

// 0/kl      — laminar kinetic energy
aerofoil { type fixedValue; value uniform 0; }

// 0/omega
aerofoil { type omegaWallFunction; value uniform 1; }

// 0/nut
aerofoil { type nutLowReWallFunction; value uniform 0; }
```

Free-stream turbulence intensity for Regime C should reflect the experimental
reference — typically `Tu = 0.1%` for clean-tunnel low-Re data.

#### Angle of attack implementation

AoA is implemented by rotating the inlet velocity vector — **never by rotating
the mesh**. The mesh always has the chord along the x-axis. For α in radians:

```c++
// 0/U — inlet patch
inlet
{
    type        fixedValue;
    value       uniform (UX UY 0);   // UX = U_inf*cos(α), UY = U_inf*sin(α)
}
```

The `liftDir` and `dragDir` in `forceCoeffs` must also be rotated:

```c++
liftDir     (LIFTDIR_X LIFTDIR_Y 0);   // (-sin(α), cos(α), 0)
dragDir     (DRAGDIR_X DRAGDIR_Y 0);   // ( cos(α), sin(α), 0)
```

These values are computed by `05_prepare_case.py` and substituted into the
regime template via Jinja2.

#### forceCoeffs function object (OpenFOAM 12 syntax)

```c++
functions
{
    forceCoeffs
    {
        type            forceCoeffs;
        libs            ("libforces.so");
        writeControl    timeStep;
        writeInterval   1;

        patches         (aerofoil);
        log             true;

        rho             rhoInf;
        rhoInf          1.225;

        liftDir         (LIFTDIR_X LIFTDIR_Y 0);
        dragDir         (DRAGDIR_X DRAGDIR_Y 0);
        CofR            (0.25 0 0);
        pitchAxis       (0 0 1);

        magUInf         UINF;
        lRef            1.0;
        Aref            1.0;
    }
}
```

Verify against: `foamInfo forceCoeffs` and
`foamSearch $FOAM_TUTORIALS forceCoeffs`.

#### Schemes (`fvSchemes`) — per regime

- Regimes A, D (attached, robust): second-order; `div(phi,U) Gauss linearUpwind grad(U);`
- Regime B (separation): bounded second-order; `div(phi,U) Gauss linearUpwindV grad(U);` with stronger limiters on `div(phi,k)` and `div(phi,omega)` (`Gauss upwind` is acceptable for stability)
- Regime C (transition): same bounded second-order as B; transition equations benefit from upwinded scalar fluxes

#### Solvers and relaxation (`fvSolution`) — per regime

Baseline SIMPLE settings (all regimes):

```c++
SIMPLE
{
    nNonOrthogonalCorrectors 1;
    consistent               yes;
    residualControl
    {
        U        1e-5;
        p        1e-5;
        "(k|omega|nuTilda|kl)" 1e-5;
    }
}

relaxationFactors
{
    equations { U 0.7; "(k|omega|nuTilda|kl)" 0.7; }
    fields    { p 0.3; }
}
```

For Regime B (near-stall), reduce all relaxation factors by 30% and increase
`nNonOrthogonalCorrectors` to 2 if `checkMesh` reports non-orthogonality > 60°.

---


### L7. Meshing Rules (gmsh)

#### gmsh version

Always use gmsh via the Python API (installed from PyPI into `.venv`):

```python
import gmsh
gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 0)
```

#### Mesh quality requirements

Before accepting any mesh, verify with `checkMesh`:

| Metric | Acceptable limit | Ideal |
|---|---|---|
| Max non-orthogonality | < 70° | < 40° |
| Max skewness | < 4 | < 1 |
| Max aspect ratio | < 1000 | < 100 |
| All cells valid | Yes | Yes |

Parse `checkMesh` output in Python and raise an exception if limits are
exceeded — do not silently continue with a bad mesh.

#### Per-regime mesh strategy

| Regime | Target y+ | Prism layers | Growth ratio | Wake refinement | Cell count |
|---|---|---|---|---|---|
| A | 30 | 15–20 | 1.20 | medium (1–2 c) | 80k–200k |
| B | 0.5 | 30–40 | 1.10 | strong (3–5 c) | 300k–1M |
| C | 0.5 | 35–45 | 1.08 | strong (3–5 c) | 500k–1.2M |
| D | 50 | 12–18 | 1.25 | light (0.5–1 c) | 50k–150k |

Domain extent: 20c upstream, 30c downstream, 20c transverse (all regimes).
Wake refinement zone is regime-specific to capture separation/transition.

#### First cell height function (y+ target as input)

```python
def first_cell_height(Re: float, y_plus: float,
                      chord: float = 1.0, nu: float = 1.5e-5,
                      rho: float = 1.225) -> float:
    """Compute wall-normal first cell height for the given target y+.

    Uses the flat-plate turbulent BL correlation Cf = 0.026 / Re^(1/7).
    For low-Re transitional flow (Regime C), this overestimates Cf slightly;
    the resulting mesh is still safe (y+ comes out a little lower than target).
    """
    Cf    = 0.026 / Re ** (1/7)
    U_inf = Re * nu / chord
    tau_w = 0.5 * rho * U_inf**2 * Cf
    u_tau = (tau_w / rho) ** 0.5
    return y_plus * nu / u_tau
```

The y+ target is selected by regime:

```python
Y_PLUS = {"A": 30.0, "B": 0.5, "C": 0.5, "D": 50.0}
```

The achieved y+ is harvested post-run and stored in `case_metadata.json`.

---

### L8. CFD Automation Rules

#### Template substitution

Use Jinja2 for all OpenFOAM template substitution. Never use `.replace()` for
multi-variable substitution.

```python
from jinja2 import Environment, FileSystemLoader

regime_dir = TEMPLATES_DIR / f"regime_{regime}"
env = Environment(loader=FileSystemLoader(regime_dir),
                  keep_trailing_newline=True)

U_template = env.get_template("0/U.jinja")
(case_dir / "0" / "U").write_text(U_template.render(
    UX=f"{Ux:.6f}",
    UY=f"{Uy:.6f}",
    UINF=f"{U_inf:.6f}",
))

cd_template = env.get_template("system/controlDict.jinja")
(case_dir / "system" / "controlDict").write_text(cd_template.render(
    UINF=f"{U_inf:.6f}",
    LIFTDIR_X=f"{lift_x:.6f}", LIFTDIR_Y=f"{lift_y:.6f}",
    DRAGDIR_X=f"{drag_x:.6f}", DRAGDIR_Y=f"{drag_y:.6f}",
))
```

Template files use Jinja2 syntax: `{{ UX }}`, `{{ UY }}`, etc. The template
filename suffix `.jinja` distinguishes template files from already-rendered
files copied verbatim.

#### Parallelism

```bash
parallel -j 4 \
  "cd {1} && source /opt/openfoam12/etc/bashrc && foamRun > log.foamRun 2>&1" \
  ::: cases/case_*/
```

From Python:

```python
case_dirs = sorted(CASES_DIR.glob("case_*"))
case_list = " ".join(str(d) for d in case_dirs)
cmd = (
    f'parallel -j 4 '
    f'"cd {{1}} && source /opt/openfoam12/etc/bashrc && '
    f'foamRun > log.foamRun 2>&1" ::: {case_list}'
)
subprocess.run(cmd, shell=True, check=True)
```

#### Convergence check — regime-aware

A case is converged when ALL of the following are true:

1. `postProcessing/forceCoeffs/0/coefficient.dat` exists
2. File has more than `MIN_ROWS[regime]` time-step rows
3. `std(Cl)` over the last 200 rows < `CL_TOL[regime]`
4. `std(Cd)` over the last 200 rows < `CD_TOL[regime]`
5. `log.foamRun` does not contain `FOAM FATAL ERROR`
6. The achieved y+ on the aerofoil patch is within ±50% of the regime target

```python
MIN_ROWS  = {"A": 800,   "B": 1500,  "C": 1500, "D": 800}
CL_TOL    = {"A": 0.005, "B": 0.010, "C": 0.005, "D": 0.005}
CD_TOL    = {"A": 0.005, "B": 0.010, "C": 0.005, "D": 0.005}
END_TIME  = {"A": 2000,  "B": 5000,  "C": 4000, "D": 2000}
```

Regime B uses looser tolerances and longer end-time because near-stall flow
exhibits persistent low-frequency oscillation even in steady RANS.

---

