# CLAUDE.md — Development Guide for NACASurrogate (Regime-Aware Redesign)

This file tells Claude (or any LLM) how to work on the NACASurrogate project
correctly. Read this entire file before writing any code, editing any OpenFOAM
dictionary, or suggesting any shell commands.

---

## 1. Project Summary

NACASurrogate is a **regime-aware** parametric surrogate modeling pipeline for
NACA 4-digit aerofoils. The aerodynamic design space is partitioned into four
flow regimes, each with its own validated CFD template (turbulence model, wall
treatment, y+ target, mesh strategy). Samples from all regimes are then merged
into a single ML-ready dataset, with `regime_id` encoded as an input feature.

It replaces expensive OpenFOAM RANS simulations with fast data-driven models
(GP, RF, MLP, Kriging) trained on a 200-sample Latin Hypercube dataset spanning
the four regimes.

Full design rationale: `README_complete.md`. Quick design summary: `README.md`.

The four regimes are defined precisely in §10 of this file. The regime
classification rule is in §11. The case metadata schema is in §12.

---

## 2. Environment — Always Use Micromamba

**All Python commands and scripts must be run inside the `openfoam` micromamba
environment. Never use system Python or pip outside this environment.**

### Setup (first time)

```bash
"${SHELL}" <(curl -L micro.mamba.pm/install.sh)
micromamba env create -f environment.yml
micromamba activate openfoam
```

### Daily use

```bash
micromamba activate openfoam
```

### Running scripts

```bash
micromamba run -n openfoam python scripts/01_generate_doe.py
# or
micromamba activate openfoam
python scripts/01_generate_doe.py
```

### Adding a new dependency

```bash
micromamba install -n openfoam <package>            # for conda-forge packages
micromamba run -n openfoam pip install <package>    # for pip-only packages
# Then update environment.yml manually
```

### Never do

```bash
pip install ...          # outside the environment
python ...               # using system Python
conda activate ...       # use micromamba, not conda
```

---

## 3. OpenFOAM Version

**This project targets OpenFOAM 12 (OpenFOAM Foundation release).**

Do not use OpenFOAM ESI (openfoam.com) syntax — the two forks have diverged.
Foundation release is at [openfoam.org](https://openfoam.org).

OpenFOAM is installed system-level (not inside micromamba). Source it before
any foam commands:

```bash
source /opt/openfoam12/etc/bashrc
```

On this OpenFOAM 12 installation, `simpleFoam` has been superseded by
`foamRun` with `solver incompressibleFluid`. Use `foamRun` in all new
automation and templates.

---

## 4. How to Get Correct OpenFOAM 12 File Syntax

**This is the most important section. Never guess OpenFOAM dictionary syntax.
Always verify using one or more of the three methods below.**

### Method 1 — Copy from `$FOAM_TUTORIALS`

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

### Method 2 — `foamInfo <keyword>`

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

### Method 3 — `foamSearch`

```bash
foamSearch $FOAM_TUTORIALS functions liftDir
foamSearch $FOAM_TUTORIALS fvSolution relaxationFactors/equations/U
foamSearch $FOAM_TUTORIALS 0/nut nutUSpaldingWallFunction
foamSearch $FOAM_TUTORIALS 0/nuTilda freestream
foamSearch $FOAM_TUTORIALS constant/momentumTransport SpalartAllmaras
```

### Verification order for any OpenFOAM keyword

1. Open the relevant regime's tutorial under `$FOAM_TUTORIALS/`
2. Run `foamInfo <keyword>`
3. Run `foamSearch $FOAM_TUTORIALS <file> <keyword>` and read the examples
4. Only then write the dictionary entry

---

## 5. OpenFOAM File Writing Rules

### Header

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

### Indentation and formatting

- 4 spaces for indentation (no tabs)
- Opening brace `{` on the same line as the keyword
- Closing brace `}` on its own line
- Semicolon after every value assignment

### Turbulence dictionary (`constant/momentumTransport`) — per regime

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

### Field boundary conditions — per regime

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

### Angle of attack implementation

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

### forceCoeffs function object (OpenFOAM 12 syntax)

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

### Schemes (`fvSchemes`) — per regime

- Regimes A, D (attached, robust): second-order; `div(phi,U) Gauss linearUpwind grad(U);`
- Regime B (separation): bounded second-order; `div(phi,U) Gauss linearUpwindV grad(U);` with stronger limiters on `div(phi,k)` and `div(phi,omega)` (`Gauss upwind` is acceptable for stability)
- Regime C (transition): same bounded second-order as B; transition equations benefit from upwinded scalar fluxes

### Solvers and relaxation (`fvSolution`) — per regime

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

## 6. Python Code Rules

### File structure

```python
#!/usr/bin/env python3
"""
Script: <filename>
Stage:  <stage number and name>
Purpose: <one sentence>

Usage:
    micromamba run -n openfoam python scripts/<filename>
"""
```

### Paths

All paths must be relative to the project root. Use `pathlib.Path`, never `os.path`:

```python
from pathlib import Path

PROJECT_ROOT  = Path(__file__).resolve().parent.parent
CASES_DIR     = PROJECT_ROOT / "cases"
RESULTS_DIR   = PROJECT_ROOT / "results"
MODELS_DIR    = PROJECT_ROOT / "models"
SCRIPTS_DIR   = PROJECT_ROOT / "scripts"
TEMPLATES_DIR = PROJECT_ROOT / "templates"
VALIDATION_DIR = PROJECT_ROOT / "validation"
```

### Random seeds

All stochastic operations use `random_state=42` or `np.random.seed(42)`.
Never use unseeded randomness.

### Logging

```python
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(message)s")
log = logging.getLogger(__name__)

log.info("Generating LHS sample for regime A...")
log.warning("Case 0042 did not converge — skipping")
log.error("checkMesh failed for case 0017")
```

### Subprocess calls (OpenFOAM)

```python
import subprocess

result = subprocess.run(
    ["foamRun"], cwd=case_dir, capture_output=True, text=True
)
if result.returncode != 0:
    log.error(f"foamRun failed in {case_dir}: {result.stderr[-500:]}")
    return False
```

### Data types

- All parameter arrays: `np.float64`
- Case indices: `np.int32`
- Regime labels: stored as `category` dtype in pandas, encoded as one-hot for surrogate input
- Saved models: `joblib.dump` / `joblib.load` (not pickle directly)

---

## 7. Meshing Rules (gmsh)

### gmsh version

Always use gmsh via the Python API (installed in the micromamba environment):

```python
import gmsh
gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 0)
```

### Mesh quality requirements

Before accepting any mesh, verify with `checkMesh`:

| Metric | Acceptable limit | Ideal |
|---|---|---|
| Max non-orthogonality | < 70° | < 40° |
| Max skewness | < 4 | < 1 |
| Max aspect ratio | < 1000 | < 100 |
| All cells valid | Yes | Yes |

Parse `checkMesh` output in Python and raise an exception if limits are
exceeded — do not silently continue with a bad mesh.

### Per-regime mesh strategy

| Regime | Target y+ | Prism layers | Growth ratio | Wake refinement | Cell count |
|---|---|---|---|---|---|
| A | 30 | 15–20 | 1.20 | medium (1–2 c) | 80k–200k |
| B | 0.5 | 30–40 | 1.10 | strong (3–5 c) | 300k–1M |
| C | 0.5 | 35–45 | 1.08 | strong (3–5 c) | 500k–1.2M |
| D | 50 | 12–18 | 1.25 | light (0.5–1 c) | 50k–150k |

Domain extent: 20c upstream, 30c downstream, 20c transverse (all regimes).
Wake refinement zone is regime-specific to capture separation/transition.

### First cell height function (y+ target as input)

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

## 8. CFD Automation Rules

### Template substitution

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

### Parallelism

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

### Convergence check — regime-aware

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

## 9. Surrogate Modeling Rules

### Inputs

The global surrogate consumes:

```
X = [alpha_deg, Re, thickness, regime_onehot_A, regime_onehot_B,
     regime_onehot_C, regime_onehot_D]                  # shape (N, 7)
```

Regime is one-hot encoded (4 columns) to avoid imposing a false ordinal on the
categorical regime variable. Continuous features (`alpha_deg`, `Re`,
`thickness`) are standardized; one-hot columns are passed through unchanged.

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

preproc = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), ["alpha_deg", "Re", "thickness"]),
        ("cat", "passthrough",    ["regime_A", "regime_B", "regime_C", "regime_D"]),
    ]
)
X_train_proc = preproc.fit_transform(X_train)
X_test_proc  = preproc.transform(X_test)
joblib.dump(preproc, MODELS_DIR / "preprocessor.joblib")
```

### Train-test discipline

The 40 test samples saved in `test_idx.npy` at DOE time are sacred:
- Never train on them
- Never use them to select hyperparameters (use cross-validation on the training set only)
- Never re-run CFD based on test set performance
- Report final metrics on the test set exactly once at the end

The split is stratified by regime: each regime contributes its 20% to the
test set so no regime is absent from test-time evaluation.

### Model persistence

```python
joblib.dump(preproc, MODELS_DIR / "preprocessor.joblib")
joblib.dump(gp_Cl,   MODELS_DIR / "gp_Cl.joblib")
joblib.dump(gp_Cd,   MODELS_DIR / "gp_Cd.joblib")
joblib.dump(rf_Cl,   MODELS_DIR / "rf_Cl.joblib")
joblib.dump(rf_Cd,   MODELS_DIR / "rf_Cd.joblib")
joblib.dump(mlp_Cl,  MODELS_DIR / "mlp_Cl.joblib")
joblib.dump(mlp_Cd,  MODELS_DIR / "mlp_Cd.joblib")
joblib.dump(krg_Cl,  MODELS_DIR / "krg_Cl.joblib")
joblib.dump(krg_Cd,  MODELS_DIR / "krg_Cd.joblib")
```

### Metrics — global AND per-regime

Always report R², RMSE, MAE both globally and broken down by regime:

```python
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

def report_metrics(y_true, y_pred, regime_labels):
    rows = [{"slice": "global",
             "R2":   r2_score(y_true, y_pred),
             "RMSE": mean_squared_error(y_true, y_pred, squared=False),
             "MAE":  mean_absolute_error(y_true, y_pred)}]
    for r in ["A", "B", "C", "D"]:
        m = regime_labels == r
        if m.sum() == 0:
            continue
        rows.append({"slice": f"regime_{r}",
                     "R2":   r2_score(y_true[m], y_pred[m]),
                     "RMSE": mean_squared_error(y_true[m], y_pred[m], squared=False),
                     "MAE":  mean_absolute_error(y_true[m], y_pred[m])})
    return rows
```

A model is considered acceptable when each per-regime R² ≥ 0.90 for Cl and
≥ 0.85 for Cd. Global R² alone is misleading because regimes with more samples
dominate.

---

## 10. Flow Regimes Specification

### Regime A — Attached turbulent

| Property | Value |
|---|---|
| α range | 0°–8° |
| Re range | 1.5×10⁶ – 3×10⁶ |
| Thickness | 0.10 – 0.18 |
| Physics | Attached turbulent BL, mild adverse pressure gradients, limited separation |
| Turbulence model | `SpalartAllmaras` (preferred); `kOmegaSST` with wall functions acceptable |
| Wall treatment | Wall functions (`nutUSpaldingWallFunction`) |
| Target y+ | 20–50 (centred at 30) |
| Prism layers | 15–20 |
| Cell count | 80k–200k |
| End time (iterations) | 2000 |
| Notes | First regime to validate. Fast, robust, low cell count. |

### Regime B — Near-stall separated

| Property | Value |
|---|---|
| α range | 10°–16° |
| Re range | 1×10⁶ – 3×10⁶ |
| Thickness | 0.12 – 0.24 |
| Physics | Strong adverse pressure gradients, partial separation, wake growth, stall onset |
| Turbulence model | `kOmegaSST` (fully resolved) |
| Wall treatment | Fully resolved (`nutLowReWallFunction`, `kLowReWallFunction`, `omegaWallFunction`) |
| Target y+ | < 1 (centred at 0.5) |
| Prism layers | 30–40 |
| Cell count | 300k – 1M |
| End time | 5000 |
| Notes | Tighter relaxation; longer run; convergence may stall. URANS as future option. |

### Regime C — Transitional low-Re

| Property | Value |
|---|---|
| α range | 0°–8° |
| Re range | 3×10⁵ – 1×10⁶ |
| Thickness | 0.08 – 0.15 |
| Physics | Laminar BL, transition, laminar separation bubbles |
| Turbulence model | `kkLOmega` (transition); fallback `kOmegaSST` with low Tu inlet |
| Wall treatment | Fully resolved |
| Target y+ | < 1 (centred at 0.5) |
| Prism layers | 35–45 |
| Cell count | 500k – 1.2M |
| End time | 4000 |
| Notes | Most physically delicate regime. Implement only after A and B are locked. |

### Regime D — Fully turbulent high-Re attached

| Property | Value |
|---|---|
| α range | 0°–6° |
| Re range | 2×10⁶ – 5×10⁶ |
| Thickness | 0.10 – 0.18 |
| Physics | Fully turbulent attached flow, minimal transition or separation |
| Turbulence model | `SpalartAllmaras` |
| Wall treatment | Wall functions (`nutkWallFunction` or `nutUSpaldingWallFunction`) |
| Target y+ | 30–80 (centred at 50) |
| Prism layers | 12–18 |
| Cell count | 50k–150k |
| End time | 2000 |
| Notes | Cheapest regime per sample. Overlaps with A; classifier rule disambiguates. |

---

## 11. Regime Classification Rule

At DOE time the regime is known by construction (each sample is drawn from
exactly one regime's bounding box). The classifier is used at **inference**
time — when querying the surrogate at an arbitrary `(α, Re, t)` point, or
when assigning labels to OOD probes during validation.

**Priority rule** (apply in order; first match wins):

```python
def classify_regime(alpha_deg: float, Re: float, thickness: float) -> str:
    if alpha_deg >= 10.0:
        return "B"                          # near-stall takes priority over Re
    if Re < 1.0e6:
        return "C"                          # low-Re takes priority over D
    if Re >= 2.0e6 and alpha_deg <= 6.0 and 0.10 <= thickness <= 0.18:
        return "D"                          # high-Re attached subset
    return "A"                              # default attached turbulent
```

Notes on edge cases:

- The α ∈ (8°, 10°) band falls to regime A under this rule. Phase 1 validation
  is performed at α ≤ 8°, so this is safe.
- The Re ∈ (1×10⁶, 1.5×10⁶) band for low α falls to regime A, which is
  acceptable — A's turbulence model is also valid at this Re.
- A point outside every regime's bounding box (e.g. α=20°, Re=4×10⁶) gets
  the nearest regime label by this rule and is flagged `ood=True` in
  `case_metadata.json`. Such points should not be added to the CFD dataset.

---

## 12. Case Metadata Schema (`case_metadata.json`)

Every case writes a `case_metadata.json` after Stage 7 (harvest):

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
  "y_plus_min": 18.4,
  "y_plus_mean": 31.7,
  "y_plus_max": 47.2,

  "cells_total": 142308,
  "non_orthogonality_max": 38.4,
  "skewness_max": 1.21,

  "runtime_s": 412.6,
  "iterations": 2000,
  "residuals_final": {"Ux": 3.1e-6, "Uy": 4.8e-6, "p": 7.2e-6, "nuTilda": 9.1e-7},

  "Cl_mean": 0.456,
  "Cl_std":  0.0012,
  "Cd_mean": 0.0098,
  "Cd_std":  0.00008,
  "L_over_D": 46.5,

  "converged": true,
  "ood": false,
  "notes": ""
}
```

The dataset CSV is the joined projection of all `case_metadata.json` files;
the per-case JSON is the source of truth for everything else (residual
history, mesh quality, etc.).

---

## 13. Validation Phases — Do Not Skip

The pipeline is implemented and validated regime-by-regime:

| Phase | Goal | Output |
|---|---|---|
| 1 | Lock Regime A template against NACA0012 reference | `validation/regime_A/report.md` |
| 2 | Lock Regime B template (near-stall) | `validation/regime_B/report.md` |
| 3 | Lock Regime C template (transitional) | `validation/regime_C/report.md` |
| 4 | Lock Regime D template (high-Re) | `validation/regime_D/report.md` |
| 5 | Generate full 200-case dataset | `dataset_clean.csv` |
| 6 | Train and validate unified surrogate | `results/` |

A regime is **locked** when its NACA0012 probe cases (table in `README.md`)
agree with the canonical reference within these tolerances:

| Metric | Tolerance |
|---|---|
| ΔCl (vs reference) | ≤ 5% of reference Cl |
| ΔCd (vs reference) | ≤ 10% of reference Cd |
| Cp curve | qualitative match; same suction-peak location |

Phase 1 (Regime A) is the active focus. **Do not begin Phase 2, 3, 4, or 5
until Regime A is locked.** If a regime fails to validate within reasonable
effort, document the failure in its `report.md`, exclude its samples from
the dataset, and proceed to the surrogate stage on the remaining regimes.

---

## 14. What Not to Do

| Do not | Instead |
|---|---|
| Guess OpenFOAM dict syntax | Use `$FOAM_TUTORIALS`, `foamInfo`, `foamSearch` |
| Use `turbulenceProperties` | Use `constant/momentumTransport` (OpenFOAM 12) |
| Use one template for multiple regimes | Each regime has its own template directory |
| Rotate the mesh for AoA | Rotate the inlet velocity vector |
| Use system Python | `micromamba activate openfoam` first |
| Use `os.path` | `pathlib.Path` |
| Use `print()` for logging | `logging.info()` / `logging.warning()` |
| Use `fit_transform` on test data | `transform` only on test data |
| Hard-code absolute paths | Use `PROJECT_ROOT` relative paths |
| Use `pickle` directly | Use `joblib.dump` / `joblib.load` |
| Skip phase validation | Lock each regime against NACA0012 before generating its dataset slice |
| Encode regime as an integer 0–3 | Use one-hot (4 columns); regime is categorical |
| Commit `cases/` to git | Add `cases/` to `.gitignore` |

---

## 15. `.gitignore` Recommendations

```
# Generated case directories (large)
cases/

# OpenFOAM processor directories
processor*/

# Large result files
*.foam
*.vtu
*.vtk

# Python
__pycache__/
*.pyc
.ipynb_checkpoints/

# Environment
.env

# OS
.DS_Store
```

---

## 16. Quick Reference — Key Commands

```bash
# Activate environment
micromamba activate openfoam

# Source OpenFOAM 12
source /opt/openfoam12/etc/bashrc

# Check OpenFOAM version
foamVersion

# Syntax help
foamInfo forceCoeffs
foamInfo SpalartAllmaras
foamInfo kOmegaSST
foamInfo kkLOmega
foamSearch $FOAM_TUTORIALS liftDir

# Per-regime tutorial anchors
ls $FOAM_TUTORIALS/incompressibleFluid/airFoil2D/                # A, D, B baseline
ls $FOAM_TUTORIALS/fluid/aerofoilNACA0012Steady/                 # B reference

# Run pipeline stages
python scripts/01_generate_doe.py
python scripts/02_classify_regime.py        # utility module
python scripts/03_generate_geometry.py
python scripts/04_generate_mesh.py
python scripts/05_prepare_case.py
python scripts/06_run_cfd.py
python scripts/07_harvest_results.py
python scripts/08_validate_regimes.py
python scripts/09_train_surrogates.py
python scripts/10_global_validation.py

# Parallel CFD execution
parallel -j 4 "cd {1} && source /opt/openfoam12/etc/bashrc && foamRun > log.foamRun 2>&1" ::: cases/case_*/

# Mesh quality
cd cases/case_0000 && checkMesh

# Monitor a running case
tail -f cases/case_0000/log.foamRun
```
