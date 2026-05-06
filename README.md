# NACASurrogate

Parametric surrogate model for NACA 4-digit aerofoils.
Maps `(α, Re, thickness) → (Cl, Cd)` using 100 OpenFOAM RANS runs as training data.

---

## Design space

| Parameter | Symbol | Range | Units |
|---|---|---|---|
| Angle of attack | α | 0 – 16 | degrees |
| Reynolds number | Re | 5×10⁵ – 3×10⁶ | — |
| Max thickness | t | 0.08 – 0.24 | fraction of chord |

- 100 Latin Hypercube samples total
- 80 training / 20 test — split indices saved at DOE time, never changed
- Chord = 1.0 m, air kinematic viscosity ν = 1.5×10⁻⁵ m²/s

## Outputs

| Variable | Source | Notes |
|---|---|---|
| Cl | OpenFOAM `forceCoeffs` | Primary |
| Cd | OpenFOAM `forceCoeffs` | Primary |
| L/D | Derived Cl/Cd | Secondary |

## Surrogates trained (all compared on same test set)

- Gaussian Process — `sklearn.GaussianProcessRegressor`, Matern(ν=2.5) + WhiteKernel
- Random Forest — `sklearn.RandomForestRegressor`, n_estimators=200
- MLP — `sklearn.MLPRegressor`, layers (64, 64, 32), relu, early_stopping
- Kriging — `smt.surrogate_models.KRG`

## Toolchain

| Job | Tool |
|---|---|
| DOE | `pyDOE2` |
| Geometry | `numpy` (analytic NACA formula) |
| Meshing | `gmsh` Python API |
| CFD | OpenFOAM 12 — `foamRun` (`solver incompressibleFluid`) + `kOmegaSST` |
| Parallelism | GNU `parallel` |
| Surrogates | `scikit-learn`, `smt` |
| Sensitivity | `SALib` (Sobol) |
| Plots | `matplotlib`, `seaborn` |

---

## Directory structure

```
NACASurrogate/
├── README.md
├── CLAUDE.md
├── requirements.txt
├── environment.yml
├── samples.csv              # 100×3 DOE output
├── train_idx.npy            # 80 training indices
├── test_idx.npy             # 20 test indices
├── dataset_clean.csv        # harvested CFD results
│
├── scripts/
│   ├── 01_doe.py
│   ├── 02_geometry.py
│   ├── 03_mesh.py
│   ├── 04_run_cfd.py
│   ├── 05_harvest.py
│   ├── 06_train_surrogates.py
│   └── 07_validate.py
│
├── openfoam_template/       # master case — copied and modified per sample
│   ├── 0/
│   │   ├── U.template       # Jinja2 placeholders: UX, UY, UINF, LIFTDIR_X/Y, DRAGDIR_X/Y
│   │   ├── p
│   │   ├── k
│   │   ├── omega
│   │   └── nut
│   ├── constant/
│   │   └── momentumTransport   # kOmegaSST (OpenFOAM 12 — NOT turbulenceProperties)
│   └── system/
│       ├── controlDict.template  # Jinja2 — forceCoeffs with UINF, LIFTDIR, DRAGDIR
│       ├── fvSchemes
│       └── fvSolution
│
├── cases/                   # auto-generated (gitignored)
│   └── case_0000/
│       ├── aerofoil.dat
│       ├── params.json
│       └── <OpenFOAM case>
│
├── models/                  # saved with joblib
│   ├── scaler.joblib
│   ├── gp_Cl.joblib / gp_Cd.joblib
│   ├── rf_Cl.joblib / rf_Cd.joblib
│   ├── mlp_Cl.joblib / mlp_Cd.joblib
│   └── krg_Cl.joblib / krg_Cd.joblib
│
└── results/
    ├── surrogate_metrics.csv
    ├── parity_plots.png
    ├── sobol_sensitivity.png
    ├── ood_test.png
    ├── learning_curve.png
    └── stall_extrapolation.png
```

---

## Pipeline

### Stage 1 — `01_doe.py`

**Inputs:** none
**Outputs:** `samples.csv`, `train_idx.npy`, `test_idx.npy`

- LHS via `pyDOE2.lhs(3, samples=100, criterion='maximin')`
- Scale: α → [0, 16], Re → [5e5, 3e6], t → [0.08, 0.24]
- CSV columns: `[alpha_deg, Re, thickness]`
- Random 80/20 split → `train_idx.npy`, `test_idx.npy`
- `random_state = 42`

---

### Stage 2 — `02_geometry.py`

**Inputs:** `samples.csv`
**Outputs:** `cases/case_{i:04d}/aerofoil.dat`, `cases/case_{i:04d}/params.json`

NACA 4-digit formula (Phase 1: symmetric, m=0):
```
y_t(x) = 5t * (0.2969√x − 0.1260x − 0.3516x² + 0.2843x³ − 0.1015x⁴)
y_upper = +y_t,  y_lower = −y_t
```

- 200 cosine-spaced points: `x = 0.5*(1 − cos(linspace(0, π, 200)))`
- `aerofoil.dat`: two-column `x y` — upper TE→LE then lower LE→TE (closed polygon)
- `params.json`: `{"alpha_deg": ..., "Re": ..., "thickness": ...}`

---

### Stage 3 — `03_mesh.py`

**Inputs:** `aerofoil.dat` per case, Re from `samples.csv`
**Outputs:** OpenFOAM polyMesh in `cases/case_{i:04d}/constant/polyMesh/`

First cell height for y⁺ = 0.5:
```
Cf = 0.026 / Re^(1/7)
U_inf = Re * ν / chord
τ_w = 0.5 * 1.225 * U_inf² * Cf
u_τ = sqrt(τ_w / 1.225)
h = 0.5 * ν / u_τ
```

- C-topology domain: 20c upstream, 30c downstream, 20c transverse
- Target ~50 000 cells (2D)
- Convert: `gmshToFoam mesh.msh` via subprocess
- Accept only if `checkMesh` passes: non-orthogonality < 70°, skewness < 4

---

### Stage 4 — `04_run_cfd.py`

**Inputs:** `samples.csv`, `openfoam_template/`
**Outputs:** `cases/case_{i:04d}/` with all OpenFOAM files, `log.simpleFoam`

Velocity and direction vectors:
```
U_inf   = Re * ν / chord
Ux      = U_inf * cos(α_rad)
Uy      = U_inf * sin(α_rad)
liftDir = (−sin(α_rad),  cos(α_rad), 0)
dragDir = ( cos(α_rad),  sin(α_rad), 0)
```

Jinja2 placeholders in templates: `{{ UX }}`, `{{ UY }}`, `{{ UINF }}`,
`{{ LIFTDIR_X }}`, `{{ LIFTDIR_Y }}`, `{{ DRAGDIR_X }}`, `{{ DRAGDIR_Y }}`

Solver settings:
- `foamRun` with `solver incompressibleFluid`, endTime=2000, writeInterval=100
- `kOmegaSST` in `constant/momentumTransport`
- Schemes: second-order (`linearUpwind`, `Gauss linear`)
- Relaxation: p=0.3, U=0.5, k=0.5, omega=0.5

Run in parallel:
```bash
parallel -j 4 "cd {1} && source /opt/openfoam12/etc/bashrc && foamRun > log.simpleFoam 2>&1" ::: cases/case_*/
```

---

### Stage 5 — `05_harvest.py`

**Inputs:** `cases/case_{i:04d}/postProcessing/forceCoeffs/0/coefficient.dat`
**Outputs:** `dataset_all.csv`, `dataset_clean.csv`

Convergence check — all must pass:
1. `coefficient.dat` exists
2. File has > 500 rows
3. `std(Cl)` over last 200 rows < 0.005
4. `std(Cd)` over last 200 rows < 0.005
5. `log.simpleFoam` has no `FOAM FATAL ERROR`

Output columns: `[case_id, alpha_deg, Re, thickness, Cl, Cd, LD_ratio, Cl_std, Cd_std, converged]`

---

### Stage 6 — `06_train_surrogates.py`

**Inputs:** `dataset_clean.csv`, `train_idx.npy`, `test_idx.npy`
**Outputs:** `models/*.joblib`, `results/surrogate_metrics.csv`

```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)   # never fit_transform on test
joblib.dump(scaler, "models/scaler.joblib")
```

| Model | Key settings |
|---|---|
| GP | `Matern(nu=2.5) + WhiteKernel`, `n_restarts_optimizer=10`, `normalize_y=True` |
| RF | `n_estimators=200`, `random_state=42` |
| MLP | `hidden_layer_sizes=(64,64,32)`, `relu`, `early_stopping=True`, `max_iter=2000` |
| KRG | `smt.KRG(theta0=[1e-2])` |

Train one model per output (Cl and Cd separately). Report R², RMSE, MAE for all.

---

### Stage 7 — `07_validate.py`

**Inputs:** `dataset_clean.csv`, `models/*.joblib`
**Outputs:** five figures in `results/`

| Figure | Description |
|---|---|
| `parity_plots.png` | 4×2 grid: predicted vs CFD for each model × output, colored by α |
| `sobol_sensitivity.png` | S1 and ST bar chart, SALib saltelli N=8192, predict with GP |
| `ood_test.png` | Sweep Re=4×10⁶ (OOD), all models + GP uncertainty band |
| `learning_curve.png` | RMSE vs N_train ∈ [10,20,30,40,60,80], GP and RF, 5 seeds, error bars |
| `stall_extrapolation.png` | Cl vs α=0:18°, all models, dashed line at α=16° (training boundary) |

---

## Rules (always apply)

- Environment: `micromamba activate openfoam` — see `CLAUDE.md`
- OpenFOAM syntax: use `$FOAM_TUTORIALS`, `foamInfo`, `foamSearch` — see `CLAUDE.md`
- Turbulence dict: `constant/momentumTransport` (OpenFOAM 12)
- AoA: rotate inlet velocity, never the mesh
- Paths: `pathlib.Path`, relative to project root
- Models: `joblib.dump` / `joblib.load`
- Logging: `logging` module, not `print()`
- Seed: 42 everywhere
