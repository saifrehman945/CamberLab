# CFD Surrogate Model for NACA Aerofoil — Project Documentation

## Author
**Saif ur Rehman** — CFD Researcher | Physics-AI & Design Optimization  
[Portfolio](https://saifrehman945.github.io/) | [GitHub](https://github.com/saifrehman945)

---

## Table of Contents

1. [Objective](#1-objective)
2. [Why Build a Surrogate?](#2-why-build-a-surrogate)
3. [What Is a Surrogate Model?](#3-what-is-a-surrogate-model)
4. [Scope and Constraints](#4-scope-and-constraints)
5. [Design Choices](#5-design-choices)
6. [Design Space Definition](#6-design-space-definition)
7. [Output Targets](#7-output-targets)
8. [Toolchain (All Free and Open Source)](#8-toolchain-all-free-and-open-source)
9. [Full Pipeline](#9-full-pipeline)
   - [Stage 1 — Design of Experiments (DOE)](#stage-1--design-of-experiments-doe)
   - [Stage 2 — Parametric Geometry Generation](#stage-2--parametric-geometry-generation)
   - [Stage 3 — Automated Meshing](#stage-3--automated-meshing)
   - [Stage 4 — CFD Simulation (OpenFOAM)](#stage-4--cfd-simulation-openfoam)
   - [Stage 5 — Data Harvesting and Cleaning](#stage-5--data-harvesting-and-cleaning)
   - [Stage 6 — Surrogate Training](#stage-6--surrogate-training)
   - [Stage 7 — Validation, Comparison, and Limitation Study](#stage-7--validation-comparison-and-limitation-study)
   - [Stage 8 (Stretch Goal) — Active Learning Refinement](#stage-8-stretch-goal--active-learning-refinement)
10. [Surrogate Comparison Matrix](#10-surrogate-comparison-matrix)
11. [Known Limitations to Investigate](#11-known-limitations-to-investigate)
12. [Directory Structure](#12-directory-structure)
13. [LLM Coding Prompts Per Stage](#13-llm-coding-prompts-per-stage)

---

## 1. Objective

Build a **parametric surrogate model** that learns the mapping:

```
(angle_of_attack, Reynolds_number, thickness) → (Cl, Cd)
```

from a batch of high-fidelity OpenFOAM RANS CFD simulations of NACA 4-digit aerofoils.

The surrogate replaces the CFD solver during design exploration — reducing evaluation time from ~30 minutes per case (CFD) to milliseconds (surrogate inference).

The **primary goal is educational**: to understand how surrogates work, where they break, and what their practical limitations are — by building the entire pipeline from scratch.

---

## 2. Why Build a Surrogate?

A single CFD simulation of an aerofoil takes 20–60 minutes depending on mesh resolution and convergence. Design optimization, uncertainty quantification, and sensitivity analysis may require thousands of evaluations. This makes direct CFD-in-the-loop optimization computationally prohibitive.

A surrogate model (also called a response surface model or metamodel) is trained on a *small, structured set* of CFD runs and learns to approximate the input–output relationship. Once trained, it evaluates in milliseconds.

**Key questions this project will answer by doing:**
- How many CFD samples are needed for a reliable surrogate?
- Which surrogate type (GP, RF, MLP, RBF) works best for aerodynamic coefficients?
- Where does the surrogate fail — near stall, outside the training range, at high Re?
- Does the surrogate respect physical constraints (e.g. Cl = 0 at α = 0 for symmetric aerofoils)?
- How does uncertainty quantification (from GP) behave near the training boundary?

---

## 3. What Is a Surrogate Model?

A surrogate model is a data-driven approximation of an expensive function. In this context:

- **Input (X):** Design and operating parameters — angle of attack α, Reynolds number Re, aerofoil thickness t
- **Output (y):** Aerodynamic performance — lift coefficient Cl, drag coefficient Cd
- **Training data:** Results from N high-fidelity CFD simulations
- **Inference:** Given new (α, Re, t), predict (Cl, Cd) without running CFD

The surrogate is distinct from the KAN time-series surrogate in previous work (which predicted Cd over time for a *single* fixed geometry). This is a **parametric surrogate** — it generalizes across *different* geometries and operating conditions.

---

## 4. Scope and Constraints

| Item | Choice | Reason |
|---|---|---|
| Aerofoil family | NACA 4-digit | Analytic formula, no CAD needed, extensively validated |
| CFD solver | OpenFOAM 12 (`foamRun` + `solver incompressibleFluid`) | Free, scriptable, industry-standard RANS |
| Meshing | gmsh Python API | Free, fully scriptable, no GUI required |
| DOE | Latin Hypercube Sampling | Better space-filling than full factorial at same N |
| Surrogate library | scikit-learn + smt | Free, well-documented, supports all target model types |
| Sensitivity analysis | SALib | Free, supports Sobol indices |
| No proprietary tools | Fluent, ICEM, DAKOTA excluded | Fully reproducible on any Linux machine |

---

## 5. Design Choices

### Aerofoil: NACA 4-digit series

The NACA 4-digit aerofoil is defined by three shape parameters:
- **m** — maximum camber as a fraction of chord (1st digit / 100)
- **p** — position of maximum camber along chord (2nd digit / 10)
- **t** — maximum thickness as a fraction of chord (3rd+4th digits / 100)

For example, NACA 2412 means: m=0.02, p=0.4, t=0.12.

A symmetric aerofoil (NACA 00XX) has m=0, p=0. NACA 0012 is the standard benchmark.

The upper and lower surface coordinates are given analytically:

```
y_t(x) = 5t * (0.2969*sqrt(x) - 0.1260*x - 0.3516*x^2 + 0.2843*x^3 - 0.1015*x^4)

For symmetric (m=0): y_upper = +y_t, y_lower = -y_t

For cambered: compute camber line yc(x), then rotate y_t perpendicular to it.
```

### Why start with 3 parameters only (α, Re, t)?

More parameters = exponentially more samples needed to fill the space. A 3-parameter space is tractable with 80–120 LHS samples. Camber (m, p) is added in a second phase once the pipeline is validated.

### Why `foamRun` + `incompressibleFluid` (steady RANS)?

- Steady-state is valid for attached flow (α < ~14°)
- Far cheaper than unsteady (pimpleFoam)
- kOmegaSST turbulence model is the industry standard for aerofoil external aerodynamics
- AoA is implemented by rotating inlet velocity direction — mesh stays fixed
- On this OpenFOAM 12 install, `simpleFoam` has been superseded by `foamRun`
  with `solver incompressibleFluid`

---

## 6. Design Space Definition

### Phase 1 parameters (start here)

| Parameter | Symbol | Min | Max | Units | Notes |
|---|---|---|---|---|---|
| Angle of attack | α | 0 | 16 | degrees | Beyond 16° → deep stall, RANS unreliable |
| Reynolds number | Re | 5×10⁵ | 3×10⁶ | — | Subsonic UAV to light aircraft regime |
| Max thickness | t | 0.08 | 0.24 | fraction of chord | NACA 0008 to NACA 0024 |

### Phase 2 parameters (add after Phase 1 works)

| Parameter | Symbol | Min | Max | Units |
|---|---|---|---|---|
| Max camber | m | 0.0 | 0.09 | fraction of chord |
| Camber position | p | 0.2 | 0.6 | fraction of chord |

### Sample size

- Phase 1: **100 LHS samples** over (α, Re, t)
- Split: **80 training / 20 test** — the test set is separated before any surrogate training
- Phase 2 (optional): 200 LHS samples over all 5 parameters

---

## 7. Output Targets

### Phase 1 — Scalar outputs (one value per simulation)

| Output | Symbol | Source | Notes |
|---|---|---|---|
| Lift coefficient | Cl | `forceCoeffs` in OpenFOAM | Primary target |
| Drag coefficient | Cd | `forceCoeffs` in OpenFOAM | Primary target |
| Lift-to-drag ratio | L/D | Derived: Cl/Cd | Useful for optimization |

### Phase 2 — Field output (1D distribution per simulation)

| Output | Symbol | Source | Notes |
|---|---|---|---|
| Pressure coefficient | Cp(x/c) | `singleGraph` sample in OpenFOAM | Needs dimensionality reduction (POD/PCA) before surrogate training |

---

## 8. Toolchain (All Free and Open Source)

| Stage | Tool | Install |
|---|---|---|
| DOE | `pyDOE2` | `pip install pyDOE2` |
| Geometry | `numpy` | `pip install numpy` |
| Meshing | `gmsh` Python API | `pip install gmsh` |
| CFD | OpenFOAM v10+ | [openfoam.org](https://openfoam.org) |
| CFD automation | `bash`, `Python` | built-in |
| Parallel runs | `GNU parallel` | `apt install parallel` |
| Data handling | `pandas`, `numpy` | `pip install pandas` |
| Surrogates | `scikit-learn`, `smt` | `pip install scikit-learn smt` |
| Sensitivity | `SALib` | `pip install SALib` |
| Visualization | `matplotlib`, `seaborn` | `pip install matplotlib seaborn` |

---

## 9. Full Pipeline

### Stage 1 — Design of Experiments (DOE)

**Goal:** Generate a structured set of (α, Re, t) parameter combinations that efficiently covers the 3D design space.

**Method:** Latin Hypercube Sampling (LHS) — divides each parameter axis into N equal intervals and samples one point from each interval, ensuring good space coverage with far fewer points than a full factorial grid.

**Implementation steps:**

1. Install `pyDOE2`: `pip install pyDOE2`
2. Generate a unit LHS array of shape `(N_samples, 3)` where each column is in [0, 1]
3. Scale each column to physical ranges:
   - Column 0 (α): scale from [0, 1] to [0°, 16°]
   - Column 1 (Re): scale from [0, 1] to [5e5, 3e6]
   - Column 2 (t): scale from [0, 1] to [0.08, 0.24]
4. Save the sample matrix as `samples.csv` with columns `[alpha_deg, Re, thickness]`
5. Also save the 80/20 train-test split indices to `train_idx.npy` and `test_idx.npy`

**Expected output:** `samples.csv` — 100 rows × 3 columns, all values within physical bounds.

**Key constraint:** The 20 test samples must be set aside immediately and never used to inform meshing choices, solver settings, or surrogate hyperparameter tuning.

---

### Stage 2 — Parametric Geometry Generation

**Goal:** For each row in `samples.csv`, generate the NACA aerofoil surface coordinates and write them in a format that gmsh can read.

**Method:** Analytic NACA 4-digit formula. No CAD software, no file downloads.

**Implementation steps:**

1. Write a Python function `naca4(t, m=0, p=0, n=200)` that:
   - Generates `n` cosine-spaced x/c points from 0 to 1 (cosine spacing clusters points near LE and TE)
   - Computes thickness distribution `y_t(x)` using the standard NACA formula
   - For symmetric case (m=0): returns upper surface `(x, +y_t)` and lower surface `(x, -y_t)` as numpy arrays
   - For cambered case: computes camber line `y_c(x)`, slope `dy_c/dx`, rotates thickness perpendicular to camber line
   - Returns `(x_upper, y_upper, x_lower, y_lower)` all normalized by chord = 1.0

2. For each sample `i` in `samples.csv`:
   - Call `naca4(t=row['thickness'])` (Phase 1 uses symmetric aerofoils only)
   - Concatenate upper and lower surfaces into a closed polygon (TE → upper → LE → lower → TE)
   - Write to `cases/case_{i:04d}/aerofoil.dat` as space-separated `x y` coordinates
   - Write a metadata file `cases/case_{i:04d}/params.json` with the parameter values

**Expected output:** 100 directories, each containing `aerofoil.dat` and `params.json`.

---

### Stage 3 — Automated Meshing

**Goal:** Generate a 2D C-topology structured mesh around each aerofoil, suitable for RANS simulation with kOmegaSST.

**Method:** gmsh Python API — fully scriptable, no GUI.

**Mesh topology:** C-mesh. The domain extends 20 chord lengths upstream, 30 chord lengths downstream, and 20 chord lengths in the transverse direction. This is sufficient to avoid far-field boundary interference.

**Wall resolution requirement:** y+ < 1 for low-Re kOmegaSST (resolves the viscous sublayer). First cell height `h` is computed from the flat-plate approximation:

```
tau_w = 0.5 * rho * U_inf^2 * Cf
Cf ≈ 0.026 / Re^(1/7)    (turbulent flat plate)
u_tau = sqrt(tau_w / rho)
h = y_plus * nu / u_tau   (target y_plus = 0.5 to be safe)
```

This must be recomputed for each sample because Re varies.

**Implementation steps:**

1. Write a Python function `build_mesh(aerofoil_dat, Re, chord=1.0, output_dir)` using `import gmsh`:
   - Initialize gmsh: `gmsh.initialize()`
   - Read the aerofoil coordinates and create spline curves for upper and lower surfaces
   - Create the C-topology far-field boundary (semicircle upstream + rectangle downstream)
   - Set mesh size fields: fine near the aerofoil surface (target first cell height from y+ formula), coarser in the far field
   - Generate 2D mesh: `gmsh.model.mesh.generate(2)`
   - Write to `cases/case_{i:04d}/constant/polyMesh/` using gmsh's OpenFOAM export, OR write `.msh` file and convert with `gmshToFoam`
   - Run `checkMesh` (OpenFOAM utility) and parse output to confirm max non-orthogonality < 70° and max skewness < 4

2. Loop over all 100 cases and call `build_mesh()` for each

3. Log any cases where `checkMesh` fails — these are excluded from the run queue

**Expected output:** 100 OpenFOAM-format polyMesh directories, all passing `checkMesh`.

**Target mesh size:** ~50,000 cells for 2D. This balances accuracy and per-case runtime.

---

### Stage 4 — CFD Simulation (OpenFOAM)

**Goal:** Run steady RANS (`foamRun` with `solver incompressibleFluid` + `kOmegaSST`) for each case and extract converged Cl and Cd.

**OpenFOAM case structure required:**

```
case_{i}/
├── 0/
│   ├── U           ← inlet velocity computed from Re and alpha
│   ├── p
│   ├── k
│   ├── omega
│   └── nut
├── constant/
│   ├── polyMesh/   ← from Stage 3
│   └── momentumTransport   ← kOmegaSST
├── system/
│   ├── controlDict  ← includes forceCoeffs function object
│   ├── fvSchemes
│   ├── fvSolution
│   └── sampleDict   ← for Cp extraction (Phase 2)
```

**Inlet velocity from Re and alpha:**

```
U_inf = Re * nu / chord      (nu = 1.5e-5 m²/s for air at 20°C, chord = 1.0 m)
Ux = U_inf * cos(alpha_rad)
Uy = U_inf * sin(alpha_rad)
```

**forceCoeffs setup in controlDict:**

```c++
forceCoeffs
{
    type            forceCoeffs;
    libs            ("libforces.so");
    patches         (aerofoil);
    rhoInf          1.225;
    CofR            (0.25 0 0);   // quarter-chord reference point
    liftDir         (0 1 0);      // lift direction (perpendicular to freestream for alpha=0)
    dragDir         (1 0 0);      // drag direction (along freestream for alpha=0)
    // NOTE: for non-zero alpha, liftDir and dragDir must be rotated
    magUInf         <U_inf>;
    lRef            1.0;
    Aref            1.0;
}
```

**Important:** For non-zero AoA, rotate `liftDir` and `dragDir` to align with the flow direction:
```
liftDir = (-sin(alpha), cos(alpha), 0)
dragDir = ( cos(alpha), sin(alpha), 0)
```

**Convergence criteria:** Simulation considered converged when:
- All residuals (Ux, Uy, p, k, omega) < 1e-5 for at least the last 200 iterations
- Cl and Cd values stable (std over last 200 iters < 0.001)

**Automation script:** Write `run_all.py` that:
1. Reads `samples.csv`
2. For each case: substitutes U, liftDir, dragDir into template `0/U` using Python string templating or `sed`
3. Calls `foamRun > log.simpleFoam 2>&1` via `subprocess`
4. Optionally uses `GNU parallel` for concurrent runs: `parallel -j 4 "cd cases/case_{} && foamRun > log.simpleFoam 2>&1" ::: $(seq -w 0 99)`

**Expected output per case:** `postProcessing/forceCoeffs/0/coefficient.dat` containing time-series of Cl and Cd.

---

### Stage 5 — Data Harvesting and Cleaning

**Goal:** Parse all 100 OpenFOAM output directories and build a clean `(X, y)` dataset for surrogate training.

**Implementation steps:**

1. Write `harvest.py` that loops over all case directories:
   - Load `params.json` to get (α, Re, t) → this is the input row X
   - Load `postProcessing/forceCoeffs/0/coefficient.dat`
   - Check that the file exists and has > 200 time steps
   - Parse the last 200 rows, compute mean Cl and mean Cd → output row y
   - Compute std of Cl and Cd over the last 200 rows — flag case if std > 0.005 (not converged)

2. Build a `pandas` DataFrame with columns: `[alpha_deg, Re, thickness, Cl, Cd, LD_ratio, converged]`

3. Filter to keep only `converged == True` rows

4. Save as `dataset_clean.csv`

5. Print a summary: how many cases converged, what % failed, which parameter combinations caused failure (typically high α near stall)

**Expected output:** `dataset_clean.csv` — ideally 90–100 rows. If fewer than 80 training samples survive, re-run failed cases with tighter mesh or more iterations.

---

### Stage 6 — Surrogate Training

**Goal:** Train four surrogate models on the same 80 training samples and compare their performance on the 20 held-out test samples.

**Preprocessing:**

- Load `dataset_clean.csv`
- Use pre-saved `train_idx.npy` and `test_idx.npy` to split (same split every time)
- Input features: `X = [alpha_deg, Re, thickness]` — shape (N, 3)
- Targets: `y_Cl = Cl`, `y_Cd = Cd` — train one surrogate per output, or use multi-output where supported
- Apply `sklearn.preprocessing.StandardScaler` to X — fit on training set only, apply to both train and test

**Model 1 — Gaussian Process (GP / Kriging)**

```python
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel

kernel = ConstantKernel(1.0) * Matern(nu=2.5) + WhiteKernel(noise_level=1e-5)
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, normalize_y=True)
gp.fit(X_train_scaled, y_train)
y_pred, y_std = gp.predict(X_test_scaled, return_std=True)
```

Key property: returns predictive uncertainty (`y_std`). Higher uncertainty = less confident = should query more CFD runs there.

**Model 2 — Random Forest (RF)**

```python
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=200, max_features='sqrt', random_state=42)
rf.fit(X_train_scaled, y_train)
y_pred = rf.predict(X_test_scaled)
```

Key property: robust to outliers, no hyperparameter sensitivity, built-in feature importance.

**Model 3 — MLP Neural Network**

```python
from sklearn.neural_network import MLPRegressor

mlp = MLPRegressor(hidden_layer_sizes=(64, 64, 32), activation='relu',
                   max_iter=2000, early_stopping=True, validation_fraction=0.1,
                   random_state=42)
mlp.fit(X_train_scaled, y_train)
y_pred = mlp.predict(X_test_scaled)
```

Or use `smt` for KAN if desired (bring over from previous work).

**Model 4 — Radial Basis Function (RBF) / Kriging via smt**

```python
from smt.surrogate_models import RBF, KRG

rbf = RBF(d0=5)
rbf.set_training_values(X_train_scaled, y_train)
rbf.train()
y_pred = rbf.predict_values(X_test_scaled)

krg = KRG(theta0=[1e-2], print_global=False)
krg.set_training_values(X_train_scaled, y_train)
krg.train()
y_pred_krg = krg.predict_values(X_test_scaled)
y_std_krg = np.sqrt(krg.predict_variances(X_test_scaled))
```

**Metrics to compute for every model:**

```python
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

r2   = r2_score(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
mae  = mean_absolute_error(y_test, y_pred)
```

Report these for both Cl and Cd for all 4 models.

---

### Stage 7 — Validation, Comparison, and Limitation Study

**Goal:** Understand not just which model performs best, but *where* and *why* each model fails.

**7.1 Parity plots (predicted vs CFD)**

For each model and each output (Cl, Cd): scatter plot of y_pred vs y_true. Perfect prediction = diagonal line. Systematic bias shows as offset; poor precision shows as scatter. Color points by α to see if failure correlates with angle of attack.

**7.2 Model comparison table**

Produce a table of R², RMSE, MAE for each model × each output. This is your headline result.

**7.3 Sobol sensitivity analysis**

After training the GP (which is cheapest to query), use SALib to compute Sobol indices:

```python
from SALib.sample import saltelli
from SALib.analyze import sobol

problem = {
    'num_vars': 3,
    'names': ['alpha', 'Re', 'thickness'],
    'bounds': [[0, 16], [5e5, 3e6], [0.08, 0.24]]
}
param_values = saltelli.sample(problem, 1024)
# scale and predict using the trained GP
Y = gp.predict(scaler.transform(param_values))
Si = sobol.analyze(problem, Y)
# Si['S1'] = first-order indices, Si['ST'] = total-order indices
```

This tells you which input parameter contributes most to variation in Cl/Cd — expected: α dominates Cl, Re matters more for Cd.

**7.4 Deliberate failure experiments**

These are the most important experiments in the project:

| Experiment | What to do | What to observe |
|---|---|---|
| Out-of-distribution (OOD) extrapolation | Query at Re = 4e6 (outside training range) | GP uncertainty spikes; RF/MLP give confident but wrong predictions |
| Sample starvation | Retrain all models on only 20 samples | All accuracy degrades; GP degrades most gracefully |
| Near-stall behaviour | Query at α = 14–16° | All models smooth out the nonlinearity; polynomial RSM worst |
| Physics violation check | Query symmetric aerofoil (t=0.12) at α = 0° | Cl should be ~0; check if surrogate respects this |

**7.5 Learning curve**

Train GP and RF on N = [10, 20, 30, 40, 60, 80] samples each time (random subsets), evaluate on fixed 20-point test set, plot RMSE vs N. This gives the "sample efficiency" answer: how many CFD runs are actually needed?

---

### Stage 8 (Stretch Goal) — Active Learning Refinement

**Goal:** Use GP uncertainty to decide *which new CFD simulation to run next* — rather than pre-specifying all 100 samples upfront.

**Method:** Expected Improvement (EI) acquisition function.

**Concept:** After training a GP on the initial dataset, query it over a dense grid of candidate points. The point with the highest uncertainty (highest `y_std`) is the most informative to simulate next. Run CFD at that point, add to the training set, retrain the GP, repeat.

**Implementation steps:**

1. Train initial GP on 30 samples
2. Generate a candidate grid of 5000 random parameter combinations
3. Predict `y_mean, y_std` for all candidates using the GP
4. Select the candidate with maximum `y_std` (pure uncertainty sampling) or maximum Expected Improvement
5. Add that point to the run queue, execute CFD, harvest Cl/Cd
6. Retrain GP with the new point added
7. Repeat for 20 iterations, then compare final GP accuracy vs a GP trained on 50 random LHS samples

**Expected result:** Active learning should reach the same accuracy as random LHS but with fewer CFD runs.

---

## 10. Surrogate Comparison Matrix

| Property | Gaussian Process | Random Forest | MLP / KAN | RBF |
|---|---|---|---|---|
| Uncertainty quantification | Yes (native) | Approximate only | No | No |
| Works well with small N | Very good | Good | Poor (<100 samples) | Good |
| Interpolates exactly | Yes | No | No | Yes |
| Extrapolation behaviour | Reverts to prior, high uncertainty | Flat (mean of training) | Unpredictable | Diverges |
| Training cost | O(N³) — slow for N>1000 | Fast | Slow (many epochs) | O(N³) |
| Hyperparameter sensitivity | Medium (kernel choice) | Low | High | Low |
| Multi-output support | Separate models | Multi-output native | Multi-output native | Separate models |
| Best for this project | Phase 1 primary model | Robust baseline | Comparison (your expertise) | Classical baseline |

---

## 11. Known Limitations to Investigate

These are the failure modes to document — the point of the project is to encounter and quantify them:

1. **Stall discontinuity:** Cl drops sharply past stall. RANS itself models this poorly (steady-state diverges or gives non-physical results). Surrogate will smooth over this nonlinearity.

2. **Extrapolation:** All surrogates degrade outside the training range. GP at least signals this via high uncertainty. RF and MLP give confidently wrong predictions.

3. **Sample efficiency:** Too few samples → high variance surrogate. Too many → wasted CFD compute. The learning curve experiment (Stage 7.5) quantifies this.

4. **Geometry parameterization limit:** NACA 4-digit is a restricted family. A surrogate trained on it cannot generalize to arbitrary aerofoil shapes (NACA 6-series, supercritical, etc.).

5. **Physics violations:** The surrogate may predict negative Cd (physically impossible) or non-zero Cl for a symmetric aerofoil at α = 0. Physics-informed approaches (Stage 8+) can address this.

6. **RANS model error:** The training data itself has error (RANS underpredicts separation, LES/DNS would give different Cl/Cd). The surrogate inherits and potentially amplifies this error.

---

## 12. Directory Structure

```
aerofoil_surrogate/
├── README.md                     ← this file
├── samples.csv                   ← DOE output (100 × 3)
├── train_idx.npy                 ← indices of 80 training samples
├── test_idx.npy                  ← indices of 20 test samples
├── dataset_clean.csv             ← harvested CFD results
│
├── scripts/
│   ├── 01_doe.py                 ← Stage 1: LHS sampling
│   ├── 02_geometry.py            ← Stage 2: NACA profile generation
│   ├── 03_mesh.py                ← Stage 3: gmsh mesh generation
│   ├── 04_run_cfd.py             ← Stage 4: OpenFOAM case setup + submission
│   ├── 05_harvest.py             ← Stage 5: data harvesting + cleaning
│   ├── 06_train_surrogates.py    ← Stage 6: train all 4 models
│   ├── 07_validate.py            ← Stage 7: metrics, plots, sensitivity
│   └── 08_active_learning.py     ← Stage 8 (stretch)
│
├── openfoam_template/            ← master OpenFOAM case (copy-and-modify)
│   ├── 0/
│   │   ├── U.template
│   │   ├── p
│   │   ├── k
│   │   ├── omega
│   │   └── nut
│   ├── constant/
│   │   └── momentumTransport
│   └── system/
│       ├── controlDict.template
│       ├── fvSchemes
│       └── fvSolution
│
├── cases/                        ← auto-generated, one dir per sample
│   ├── case_0000/
│   ├── case_0001/
│   └── ...
│
├── results/
│   ├── surrogate_metrics.csv     ← R², RMSE, MAE for all models
│   ├── parity_plots/
│   ├── sobol_indices.csv
│   └── learning_curve.png
│
└── models/
    ├── gp_Cl.pkl
    ├── gp_Cd.pkl
    ├── rf_Cl.pkl
    ├── rf_Cd.pkl
    ├── mlp_Cl.pkl
    ├── mlp_Cd.pkl
    └── scaler.pkl
```

---

## 13. LLM Coding Prompts Per Stage

Use these prompts to generate the code for each stage. Feed them one at a time.

---

**Prompt for Stage 1 — DOE:**
```
Write a Python script `01_doe.py` that uses pyDOE2 to generate a Latin Hypercube Sample 
of 100 points over 3 parameters: angle of attack (0 to 16 degrees), Reynolds number 
(5e5 to 3e6), and aerofoil thickness (0.08 to 0.24 as fraction of chord). Scale the 
unit LHS array to physical ranges. Save the full sample as `samples.csv` with columns 
[alpha_deg, Re, thickness]. Also randomly assign 80 samples to training and 20 to test, 
saving the indices as `train_idx.npy` and `test_idx.npy`. Set random seed = 42.
```

---

**Prompt for Stage 2 — Geometry:**
```
Write a Python script `02_geometry.py` that:
1. Defines a function `naca4(t, m=0, p=0, n_points=200)` implementing the NACA 4-digit 
   analytic formula. Use cosine spacing for x/c. Return (x_upper, y_upper, x_lower, 
   y_lower) as numpy arrays, normalized to chord = 1.0.
2. Reads `samples.csv`.
3. For each row, generates the aerofoil coordinates using naca4(t=row['thickness']).
4. Creates directory `cases/case_{i:04d}/`.
5. Writes `cases/case_{i:04d}/aerofoil.dat` as a two-column space-separated file of x y 
   coordinates: upper surface from TE to LE, then lower surface from LE to TE (closed 
   polygon for meshing).
6. Writes `cases/case_{i:04d}/params.json` with the parameter values for that case.
```

---

**Prompt for Stage 3 — Meshing:**
```
Write a Python script `03_mesh.py` using the gmsh Python API that:
1. Defines a function `build_mesh(aerofoil_dat_path, Re, output_dir, chord=1.0, 
   y_plus_target=0.5, nu=1.5e-5)`.
2. Computes the required first cell height for the target y+ using the flat-plate 
   turbulent boundary layer approximation (Cf = 0.026/Re^(1/7)).
3. Reads aerofoil coordinates from the .dat file.
4. Creates a 2D C-mesh in gmsh: domain extends 20c upstream, 30c downstream, 20c 
   transverse. Spline curves for upper and lower aerofoil surfaces.
5. Sets gmsh mesh size fields: fine near the aerofoil (first cell height from step 2), 
   coarse in the far field (~2c).
6. Generates 2D mesh, writes to `{output_dir}/mesh.msh`.
7. Converts to OpenFOAM format using a subprocess call to `gmshToFoam mesh.msh`.
8. Runs `checkMesh` via subprocess and returns True/False based on whether 
   max non-orthogonality < 70 and max skewness < 4.
Reads `samples.csv` and loops over all 100 cases.
```

---

**Prompt for Stage 4 — CFD Setup and Run:**
```
Write a Python script `04_run_cfd.py` that:
1. Reads `samples.csv`.
2. Copies the OpenFOAM template from `openfoam_template/` to `cases/case_{i:04d}/`.
3. For each case, computes:
   - U_inf = Re * nu / chord  (nu=1.5e-5, chord=1.0)
   - Ux = U_inf * cos(alpha_rad), Uy = U_inf * sin(alpha_rad)
   - liftDir = (-sin(alpha_rad), cos(alpha_rad), 0)
   - dragDir = (cos(alpha_rad), sin(alpha_rad), 0)
4. Substitutes these into `0/U.template` and `system/controlDict.template` using 
   Python string .replace() or a Jinja2 template. Saves as `0/U` and `system/controlDict`.
5. Writes a bash script `run_all.sh` that uses GNU parallel to run up to 4 cases 
   simultaneously: `parallel -j 4 "cd cases/case_{} && foamRun > log.simpleFoam 2>&1"`.
Also write the OpenFOAM template files needed:
- `0/U.template` with ALPHA, UX, UY, LIFTDIR, DRAGDIR as placeholders
- `system/controlDict.template` with forceCoeffs function object, UINF placeholder, 
  liftDir and dragDir placeholders. Include writeInterval 50, endTime 2000.
- `constant/momentumTransport` using kOmegaSST
- `system/fvSchemes` and `system/fvSolution` appropriate for steady `incompressibleFluid`
  external 
  aerodynamics (second-order schemes, SIMPLE algorithm, relaxation factors 0.5/0.5/0.7).
```

---

**Prompt for Stage 5 — Data Harvesting:**
```
Write a Python script `05_harvest.py` that:
1. Reads `samples.csv` to get the input parameters for each case.
2. For each case directory `cases/case_{i:04d}/`:
   a. Checks if `postProcessing/forceCoeffs/0/coefficient.dat` exists.
   b. Loads the file (columns: Time, Cm, Cd, Cl, CdPressure, CdViscous, ...) — handle 
      OpenFOAM's comment lines starting with #.
   c. Checks the file has > 200 time steps.
   d. Takes the last 200 rows, computes mean and std of Cl and Cd.
   e. Marks case as converged if std(Cl) < 0.005 AND std(Cd) < 0.005.
3. Builds a pandas DataFrame with columns: 
   [case_id, alpha_deg, Re, thickness, Cl_mean, Cd_mean, Cl_std, Cd_std, 
    LD_ratio, converged].
4. Prints a convergence summary: total cases, converged cases, failed cases with their 
   parameter values.
5. Saves the full DataFrame as `dataset_all.csv` and the converged subset as 
   `dataset_clean.csv`.
```

---

**Prompt for Stage 6 — Surrogate Training:**
```
Write a Python script `06_train_surrogates.py` that:
1. Loads `dataset_clean.csv`, `train_idx.npy`, `test_idx.npy`.
2. Builds X (alpha_deg, Re, thickness) and y_Cl, y_Cd arrays.
3. Splits into train and test using the saved indices.
4. Fits a StandardScaler on X_train, applies to both X_train and X_test. Saves the 
   scaler as `models/scaler.pkl`.
5. Trains four models for Cl prediction and four for Cd prediction:
   - GaussianProcessRegressor with Matern(nu=2.5) + WhiteKernel kernel, 
     n_restarts_optimizer=10, normalize_y=True. Save as `models/gp_Cl.pkl`.
   - RandomForestRegressor with n_estimators=200, random_state=42. Save as 
     `models/rf_Cl.pkl`.
   - MLPRegressor with hidden_layer_sizes=(64, 64, 32), relu, early_stopping=True. 
     Save as `models/mlp_Cl.pkl`.
   - smt KRG (Kriging) model: `from smt.surrogate_models import KRG`. Save as 
     `models/krg_Cl.pkl` using pickle.
6. For each model × each output, compute R², RMSE, MAE on the test set.
7. Print and save a results table as `results/surrogate_metrics.csv`.
8. For the GP models, also report mean predictive uncertainty on the test set.
```

---

**Prompt for Stage 7 — Validation and Limitation Study:**
```
Write a Python script `07_validate.py` that produces the following plots and analyses, 
saving all figures to `results/`:

1. PARITY PLOTS: For each of 4 models × 2 outputs (Cl, Cd): scatter plot of predicted 
   vs CFD truth, with the diagonal line, R² in the title, points colored by alpha_deg. 
   Arrange as a 4×2 grid. Save as `results/parity_plots.png`.

2. SOBOL SENSITIVITY: Using SALib, sample 8192 points with the saltelli sampler over 
   the training ranges. Predict with the trained GP. Compute and plot first-order and 
   total Sobol indices for Cl and Cd. Save as `results/sobol_sensitivity.png`.

3. OUT-OF-DISTRIBUTION TEST: Create a 1D sweep at fixed Re=4e6 (outside training range), 
   alpha=0:16, thickness=0.12. Predict with all 4 models. For GP, also plot the 
   uncertainty band (mean ± 2*std). Show how models diverge outside the training range. 
   Save as `results/ood_test.png`.

4. SAMPLE EFFICIENCY (learning curve): For N_train in [10, 20, 30, 40, 60, 80], 
   randomly subsample N_train points from the full training set (5 different random 
   seeds each), train GP and RF, evaluate on the fixed test set, record RMSE. Plot 
   RMSE vs N_train for both models with error bars (std over seeds). 
   Save as `results/learning_curve.png`.

5. NEAR-STALL ANALYSIS: Create a sweep at alpha = 0:18 degrees (extending slightly 
   beyond training), Re=1e6, thickness=0.12. Plot predicted Cl vs alpha for all 4 
   models. Mark the training boundary at alpha=16 with a dashed vertical line. 
   Save as `results/stall_extrapolation.png`.
```

---

*End of documentation. Each Stage prompt above is self-contained and can be fed directly to an LLM to generate the implementation code.*
