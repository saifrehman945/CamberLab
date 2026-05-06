# CLAUDE.md — Development Guide for AeroSurrogate

This file tells Claude (or any LLM) how to work on the AeroSurrogate project correctly.
Read this entire file before writing any code, editing any OpenFOAM dictionary, or
suggesting any shell commands.

---

## 1. Project Summary

AeroSurrogate is a parametric surrogate modeling pipeline for NACA 4-digit aerofoils.
It replaces expensive OpenFOAM RANS simulations with fast data-driven models (GP, RF,
MLP, RBF/Kriging) trained on a Latin Hypercube sample of 100 CFD runs.

Full design rationale and pipeline documentation: see `README.md`.

---

## 2. Environment — Always Use Micromamba

**All Python commands and scripts must be run inside the `openfoam` micromamba
environment. Never use system Python or pip outside this environment.**

### Setup (first time)

```bash
# Install micromamba if not present
"${SHELL}" <(curl -L micro.mamba.pm/install.sh)

# Create the environment from the project file
micromamba env create -f environment.yml

# Activate
micromamba activate openfoam
```

### Daily use

```bash
micromamba activate openfoam
```

### Running scripts

```bash
# Always prefix with micromamba run if not already activated
micromamba run -n openfoam python scripts/01_doe.py

# Or activate first, then run normally
micromamba activate openfoam
python scripts/01_doe.py
```

### Adding a new dependency

```bash
# Add to environment.yml first, then:
micromamba install -n openfoam <package>   # for conda-forge packages
# OR
micromamba run -n openfoam pip install <package>   # for pip-only packages
# Then update environment.yml manually to keep it in sync
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

OpenFOAM is installed system-level (not inside micromamba). Source it before any
foam commands:

```bash
source /opt/openfoam12/etc/bashrc
# or wherever your installation lives — check with: which foamRun
```

On this OpenFOAM 12 installation, `simpleFoam` has been superseded by
`foamRun` with `solver incompressibleFluid`. Use `foamRun` in all new
automation and templates.

Add this to your `.bashrc` so it is always available:

```bash
echo "source /opt/openfoam12/etc/bashrc" >> ~/.bashrc
```

---

## 4. How to Get Correct OpenFOAM 12 File Syntax

**This is the most important section. Never guess OpenFOAM dictionary syntax.
Always verify using one or more of the three methods below.**

### Method 1 — Copy from `$FOAM_TUTORIALS`

The tutorials directory ships with every OpenFOAM installation and contains working,
validated case setups for every solver. It is the ground truth for syntax.

```bash
# Find where tutorials live
echo $FOAM_TUTORIALS
# typically: /opt/openfoam12/tutorials

# List available tutorials by solver type
ls $FOAM_TUTORIALS/incompressibleFluid/

# The most relevant tutorials for this project:
# incompressibleFluid/airFoil2D        — 2D aerofoil far-field BC structure
# fluid/aerofoilNACA0012Steady         — steady aerofoil with kOmegaSST
ls $FOAM_TUTORIALS/incompressibleFluid/airFoil2D/
ls $FOAM_TUTORIALS/fluid/aerofoilNACA0012Steady/

# Copy relevant dictionaries into your working context before editing:
cat $FOAM_TUTORIALS/incompressibleFluid/airFoil2D/system/controlDict
cat $FOAM_TUTORIALS/incompressibleFluid/airFoil2D/system/fvSchemes
cat $FOAM_TUTORIALS/incompressibleFluid/airFoil2D/system/fvSolution
cat $FOAM_TUTORIALS/incompressibleFluid/airFoil2D/0/U
cat $FOAM_TUTORIALS/incompressibleFluid/airFoil2D/0/p
cat $FOAM_TUTORIALS/fluid/aerofoilNACA0012Steady/0/k
cat $FOAM_TUTORIALS/fluid/aerofoilNACA0012Steady/0/omega
cat $FOAM_TUTORIALS/fluid/aerofoilNACA0012Steady/0/nut
cat $FOAM_TUTORIALS/fluid/aerofoilNACA0012Steady/constant/momentumTransport
```

**When writing or editing any OpenFOAM dictionary, open the equivalent tutorial file
first and base your version on it. Do not write dictionaries from memory.**

Other useful tutorials for this project:

```bash
# External aerodynamics with the steady incompressible solver:
ls $FOAM_TUTORIALS/incompressibleFluid/

# Turbulence model reference cases:
ls $FOAM_TUTORIALS/incompressibleFluid/pitzDailySteady/

# For forceCoeffs function object examples:
grep -rl "forceCoeffs" $FOAM_TUTORIALS/
```

### Method 2 — `foamInfo <keyword>`

`foamInfo` is an OpenFOAM command-line tool that shows valid options, source
documentation, and usage examples for any keyword, solver, scheme, or model.

```bash
# Valid discretization schemes for fvSchemes:
foamInfo divSchemes
foamInfo gradSchemes
foamInfo laplacianSchemes
foamInfo interpolationSchemes
foamInfo snGradSchemes

# Valid solver options for fvSolution:
foamInfo SIMPLE
foamInfo PBiCGStab
foamInfo smoothSolver
foamInfo GAMG

# Turbulence models:
foamInfo kOmegaSST
foamInfo kEpsilon

# Boundary condition types:
foamInfo fixedValue
foamInfo inletOutlet
foamInfo zeroGradient
foamInfo nutLowReWallFunction
foamInfo omegaWallFunction
foamInfo kLowReWallFunction

# Function objects:
foamInfo forceCoeffs
foamInfo forces
foamInfo singleGraph
foamInfo fieldAverage
```

Run `foamInfo` before writing any keyword you are unsure about. Its output includes
the C++ source location and the accepted sub-dictionary entries.

### Method 3 — `foamSearch`

`foamSearch` searches across all tutorials for entries in a given file. Use it to
find real-world examples of how a specific keyword is used in context.

```bash
# Find how liftDir is specified across tutorials:
foamSearch $FOAM_TUTORIALS functions liftDir

# Find how SIMPLE relaxation factors are typically set:
foamSearch $FOAM_TUTORIALS fvSolution relaxationFactors/equations/U

# Find kOmegaSST boundary condition usage:
foamSearch $FOAM_TUTORIALS 0/omega omegaWallFunction

# Find wall function usage:
foamSearch $FOAM_TUTORIALS 0/nut nutLowReWallFunction
```

### Summary: verification order for any OpenFOAM keyword

1. Check `$FOAM_TUTORIALS/incompressibleFluid/airFoil2D/` and
   `$FOAM_TUTORIALS/fluid/aerofoilNACA0012Steady/` first
2. If not there, run `foamInfo <keyword>`
3. If still unclear, run `foamSearch $FOAM_TUTORIALS <file> <keyword>` and read the examples
4. Only then write the dictionary entry

---

## 5. OpenFOAM File Writing Rules

When generating or editing any OpenFOAM dictionary file, follow these rules:

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
- `dictionary` — for system/ and constant/ files
- `volVectorField` — for U
- `volScalarField` — for p, k, omega, nut, nuTilda

### Indentation and formatting

- Use 4 spaces for indentation (not tabs)
- Opening brace `{` on the same line as the keyword
- Closing brace `}` on its own line
- Semicolon after every value assignment

### Field boundary conditions for external aerofoil (OpenFOAM 12)

The turbulence model dictionary in OpenFOAM 12 is `constant/momentumTransport`
(not `turbulenceProperties` — that name was used in older versions).

```c++
// constant/momentumTransport
simulationType  RAS;
RAS
{
    model           kOmegaSST;
    turbulence      on;
    printCoeffs     on;
}
```

Wall boundary conditions for low-Re kOmegaSST (y+ < 1):

```c++
// In 0/nut
aerofoil
{
    type    nutLowReWallFunction;
    value   uniform 0;
}

// In 0/k
aerofoil
{
    type    kLowReWallFunction;
    value   uniform 1e-10;
}

// In 0/omega
aerofoil
{
    type    omegaWallFunction;
    value   uniform 1;
}
```

### Angle of attack implementation

AoA is implemented by rotating the inlet velocity vector — **never by rotating the mesh**.
The mesh always has the chord along the x-axis. For angle of attack α (in radians):

```c++
// In 0/U — inlet patch
inlet
{
    type        fixedValue;
    value       uniform (Ux Uy 0);   // Ux = U_inf*cos(α), Uy = U_inf*sin(α)
}
```

The `liftDir` and `dragDir` in `forceCoeffs` must also be rotated:

```c++
liftDir     (-sin(α)  cos(α) 0);
dragDir     ( cos(α)  sin(α) 0);
```

These values are computed by `04_run_cfd.py` and substituted into the template.

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

Verify this syntax against: `foamInfo forceCoeffs` and
`foamSearch $FOAM_TUTORIALS forceCoeffs`

---

## 6. Python Code Rules

### File structure

Every Python script must start with:

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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CASES_DIR    = PROJECT_ROOT / "cases"
RESULTS_DIR  = PROJECT_ROOT / "results"
MODELS_DIR   = PROJECT_ROOT / "models"
SCRIPTS_DIR  = PROJECT_ROOT / "scripts"
TEMPLATE_DIR = PROJECT_ROOT / "openfoam_template"
```

### Random seeds

All stochastic operations use `random_state=42` or `np.random.seed(42)` for
reproducibility. Never use unseeded randomness.

### Logging

Use Python's built-in `logging` module, not `print()` for status messages:

```python
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(message)s")
log = logging.getLogger(__name__)

log.info("Generating LHS sample...")
log.warning("Case 0042 did not converge — skipping")
log.error("checkMesh failed for case 0017")
```

### Error handling for subprocess calls (OpenFOAM)

```python
import subprocess

result = subprocess.run(
    ["foamRun"],
    cwd=case_dir,
    capture_output=True,
    text=True
)
if result.returncode != 0:
    log.error(f"foamRun failed in {case_dir}: {result.stderr[-500:]}")
    return False
```

### Data types

- All parameter arrays: `np.float64`
- Case indices: `np.int32`
- DataFrames: always include `dtype` specification when reading CSV
- Saved models: `joblib.dump` / `joblib.load` (not pickle directly)

```python
import joblib
joblib.dump(model, MODELS_DIR / "gp_Cl.joblib")
model = joblib.load(MODELS_DIR / "gp_Cl.joblib")
```

---

## 7. Meshing Rules (gmsh)

### gmsh version

Always use gmsh via the Python API (installed in the micromamba environment):

```python
import gmsh
gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 0)  # suppress gmsh console output
```

### Mesh quality requirements

Before accepting any mesh, verify with `checkMesh`:

| Metric | Acceptable limit | Ideal |
|---|---|---|
| Max non-orthogonality | < 70° | < 40° |
| Max skewness | < 4 | < 1 |
| Max aspect ratio | < 1000 | < 100 |
| All cells valid | Yes | Yes |

Parse `checkMesh` output in Python and raise an exception if limits are exceeded —
do not silently continue with a bad mesh.

### y+ targeting

Always compute the first cell height analytically before meshing. Never guess:

```python
def first_cell_height(Re: float, chord: float = 1.0, nu: float = 1.5e-5,
                      y_plus: float = 0.5) -> float:
    """Compute wall-normal first cell height for target y+."""
    Cf = 0.026 / Re ** (1/7)                  # turbulent flat plate
    U_inf = Re * nu / chord
    tau_w = 0.5 * 1.225 * U_inf**2 * Cf
    u_tau = (tau_w / 1.225) ** 0.5
    return y_plus * nu / u_tau
```

---

## 8. CFD Automation Rules

### Template substitution

Use Jinja2 for all OpenFOAM template substitution. Never use `.replace()` for
multi-variable substitution — it is fragile and order-dependent:

```python
from jinja2 import Environment, FileSystemLoader

env = Environment(loader=FileSystemLoader(TEMPLATE_DIR))
template = env.get_template("0/U.template")
rendered = template.render(
    UX=f"{Ux:.6f}",
    UY=f"{Uy:.6f}",
    UINF=f"{U_inf:.6f}",
    LIFTDIR_X=f"{lift_x:.6f}",
    LIFTDIR_Y=f"{lift_y:.6f}",
    DRAGDIR_X=f"{drag_x:.6f}",
    DRAGDIR_Y=f"{drag_y:.6f}",
)
(case_dir / "0" / "U").write_text(rendered)
```

Template files use Jinja2 syntax: `{{ UX }}`, `{{ UY }}`, etc.

### Parallelism

Use GNU parallel for running multiple OpenFOAM cases simultaneously:

```bash
# Run 4 cases in parallel — adjust -j to match available CPU cores
parallel -j 4 \
  "cd {1} && source /opt/openfoam12/etc/bashrc && foamRun > log.simpleFoam 2>&1" \
  ::: cases/case_*/
```

From Python, generate the parallel command and execute it:

```python
import subprocess
case_dirs = sorted(CASES_DIR.glob("case_*"))
case_list = " ".join(str(d) for d in case_dirs)
cmd = (
    f'parallel -j 4 '
    f'"cd {{1}} && source /opt/openfoam12/etc/bashrc && '
    f'foamRun > log.simpleFoam 2>&1" ::: {case_list}'
)
subprocess.run(cmd, shell=True, check=True)
```

### Convergence check

A case is converged when ALL of the following are true:
1. `postProcessing/forceCoeffs/0/coefficient.dat` exists
2. File has more than 500 time-step rows
3. `std(Cl)` over the last 200 rows < 0.005
4. `std(Cd)` over the last 200 rows < 0.005
5. `log.simpleFoam` does not contain "FOAM FATAL ERROR"

---

## 9. Surrogate Modeling Rules

### Train-test discipline

The 20 test samples saved in `test_idx.npy` at DOE time are sacred:
- Never train on them
- Never use them to select hyperparameters (use cross-validation on training set only)
- Never re-run CFD based on test set performance
- Report final metrics on the test set exactly once at the end

### Scaling

Always fit the `StandardScaler` on training data only, then apply to test:

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # fit + transform
X_test_scaled  = scaler.transform(X_test)         # transform only — never fit_transform
```

### Model persistence

Save all trained models with joblib. Include the scaler — it is part of the model:

```python
joblib.dump(scaler,    MODELS_DIR / "scaler.joblib")
joblib.dump(gp_Cl,     MODELS_DIR / "gp_Cl.joblib")
joblib.dump(gp_Cd,     MODELS_DIR / "gp_Cd.joblib")
joblib.dump(rf_Cl,     MODELS_DIR / "rf_Cl.joblib")
joblib.dump(rf_Cd,     MODELS_DIR / "rf_Cd.joblib")
joblib.dump(mlp_Cl,    MODELS_DIR / "mlp_Cl.joblib")
joblib.dump(mlp_Cd,    MODELS_DIR / "mlp_Cd.joblib")
```

### Metrics

Always report all three metrics. R² alone is insufficient:

```python
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

metrics = {
    "R2":   r2_score(y_true, y_pred),
    "RMSE": mean_squared_error(y_true, y_pred, squared=False),
    "MAE":  mean_absolute_error(y_true, y_pred),
}
```

---

## 10. What Not to Do

| Do not | Instead |
|---|---|
| Guess OpenFOAM dict syntax | Use `$FOAM_TUTORIALS`, `foamInfo`, `foamSearch` |
| Use `turbulenceProperties` | Use `constant/momentumTransport` (OpenFOAM 12) |
| Rotate the mesh for AoA | Rotate the inlet velocity vector |
| Use system Python | Always `micromamba activate openfoam` first |
| Use `os.path` | Use `pathlib.Path` |
| Use `print()` for logging | Use `logging.info()` / `logging.warning()` |
| Use `fit_transform` on test data | `transform` only on test data |
| Hard-code absolute paths | Use `PROJECT_ROOT` relative paths |
| Use `pickle` directly | Use `joblib.dump` / `joblib.load` |
| Commit `cases/` to git | Add `cases/` to `.gitignore` |

---

## 11. `.gitignore` Recommendations

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

## 12. Quick Reference — Key Commands

```bash
# Activate environment
micromamba activate openfoam

# Source OpenFOAM 12
source /opt/openfoam12/etc/bashrc

# Check OpenFOAM version
foamVersion

# Get syntax help
foamInfo forceCoeffs
foamInfo kOmegaSST
foamSearch $FOAM_TUTORIALS liftDir

# Reference tutorial locations
ls $FOAM_TUTORIALS/incompressibleFluid/airFoil2D/
ls $FOAM_TUTORIALS/fluid/aerofoilNACA0012Steady/

# Run pipeline stages
python scripts/01_doe.py
python scripts/02_geometry.py
python scripts/03_mesh.py
python scripts/04_run_cfd.py
python scripts/05_harvest.py
python scripts/06_train_surrogates.py
python scripts/07_validate.py

# Run CFD cases in parallel (4 cores)
parallel -j 4 "cd {1} && foamRun > log.simpleFoam 2>&1" ::: cases/case_*/

# Check mesh quality for one case
cd cases/case_0000 && checkMesh

# Monitor a running case
tail -f cases/case_0000/log.simpleFoam
```
