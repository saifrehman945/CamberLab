<h1 align="center">CamberLab</h1>

<p align="center">
  <strong>Millisecond lift, drag and surface pressure / skin-friction predictions for NACA 4- and 5-digit aerofoils, trained on AirfRANS</strong>
</p>

<p align="center">
  <a href="#license"><img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-blue.svg"></a>
  <img alt="Python 3.11+" src="https://img.shields.io/badge/python-3.11%2B-3776ab.svg">
  <img alt="Managed with uv" src="https://img.shields.io/badge/env-uv-de5fe9.svg">
  <img alt="Data: AirfRANS (ODbL)" src="https://img.shields.io/badge/data-AirfRANS%20(ODbL)-2a78d6.svg">
  <img alt="Streamlit" src="https://img.shields.io/badge/app-Streamlit-ff4b4b.svg">
</p>

---

CamberLab replaces a ~25-minute steady RANS simulation with a surrogate call
that takes milliseconds:

```text
(angle of attack, Reynolds number, NACA section) → (Cl, Cd, L/D, Cp(x/c), Cf(x/c))
```

Live demo: <https://camberlab.streamlit.app/>

The surrogates are trained on **AirfRANS** (Bonnet et al., NeurIPS 2022):
1,000 steady 2D incompressible RANS simulations (OpenFOAM, k-ω SST) of NACA 4-
and 5-digit aerofoils at Re 2–6×10⁶ and α −5° to 15°. Unlike the AirfRANS
paper's baselines, which predict the whole flow field and integrate forces
from it, CamberLab regresses the force coefficients directly. That makes drag
prediction far more accurate: on the paper's own test sets, CamberLab ranks
drag with Spearman ρ_D ≈ 0.98, where the paper's best field model reaches 0.25.
It also predicts the wall pressure and skin-friction distributions, with an
error on the paper's own surface metrics about 4× (pressure) to two orders
of magnitude (wall shear) lower than the field models'.

## Features

- **Any NACA 4- or 5-digit section** (standard and reflex 5-digit mean lines).
  Geometry features are computed by the same code from the CFD mesh wall at
  training time and from the analytic section at query time.
- **Four surrogate families**: Gaussian process, random forest, MLP and
  Kriging (SMT), each fitted for Cl and log Cd.
- **Surface distributions.** Cp(x/c) and signed Cf(x/c) on both surfaces
  (Cf < 0 marks separation), predicted as PCA mode weights by the same
  families. The CFD wall data behind them reproduces AirfRANS' C_L and C_D
  to within 2.7% on every sample.
- **Uncertainty where it is real.** GP and Kriging give posterior ±2σ bands;
  GP bands cover 95–97% of held-out test points on the `full` task.
- **Honest envelope handling.** Queries outside the training range still get a
  prediction, always with a warning; they are never answered silently.
- **Benchmarked like-for-like** on the four official AirfRANS tasks (`full`,
  `scarce`, `reynolds`, `aoa`) with the paper's metrics.
- **Interactive app**: Cl–α, Cd–α, drag polar and L/D sweeps, Cp and Cf
  distributions, section preview and GP ±2σ uncertainty bands.

## Quickstart

### Run the app

```bash
git clone https://github.com/saifrehman945/CamberLab.git
cd CamberLab
curl -LsSf https://astral.sh/uv/install.sh | sh    # if uv is not installed
uv venv && uv pip install -r requirements.txt
uv run streamlit run app.py
```

The app and CLI only need the committed `models/full/` artifacts (~150 MB),
not the dataset. Surface curves ship for the GP and MLP families; KRG and RF
curve models are too large to commit and must be trained locally.

### Predict from the command line

```bash
uv run python scripts/predict.py --alpha 4 --re 3e6 --naca 2412 --family all
uv run python scripts/predict.py --alpha 4 --re 3e6 --naca 2412 --surface --surface-csv cp_cf.csv
uv run python scripts/11_sweep_curves.py --naca 23012 --re 4e6     # α sweep → CSV + PNG
```

### Rebuild everything from the data

Download AirfRANS once (the clipped variant, ≈ 18 GB on disk; code never
downloads it for you):

```bash
huggingface-cli download PLAID-datasets/AirfRANS_clipped \
  --repo-type dataset --local-dir data/airfrans_clipped
```

Then run the pipeline (about 4 hours on a laptop with 8 GB RAM, mostly
curve-model training; reading the data needs ~2 GB):

```bash
uv run python scripts/airfrans/inspect_dataset.py    # checks the data (Gate 2), ~35 min
uv run python scripts/20_ingest_airfrans.py          # → dataset.csv + wall Cp/Cf curves, ~20 min
uv run python scripts/21_qa_airfrans.py              # → results/airfrans_qa.md
uv run python scripts/22_make_splits.py              # official AirfRANS splits
uv run python scripts/23_surface_gate.py             # wall data reproduces C_L/C_D?
uv run python scripts/09_train_surrogates.py --task all                    # Cl/Cd + curves
uv run python scripts/10_global_validation.py        # → results/airfrans_benchmark.md
uv run pytest
```

`09_train_surrogates.py --outputs coeffs` trains only the Cl/Cd models
(~12 min).

Use `--data-dir` or `AIRFRANS_DIR` if the data lives elsewhere.

## Project status

Held-out test results on the official AirfRANS `full` split (800 train / 200
test). Relative error is the paper's `|(true − pred)/true|`; the median is
shown because the mean blows up for Cl near zero.

| Model | Cl R² | Cd R² | Spearman ρ_L | Spearman ρ_D | median rel. err. Cl | median rel. err. Cd |
|---|---|---|---|---|---|---|
| **GP** | 0.9993 | 0.9806 | 0.9995 | 0.980 | 1.07% | 0.59% |
| **KRG** | 0.9993 | 0.9872 | 0.9995 | 0.985 | 0.90% | 0.71% |
| **MLP** | 0.9984 | 0.9785 | 0.9989 | 0.993 | 1.86% | 1.83% |
| **RF** | 0.9921 | 0.9492 | 0.9961 | 0.980 | 4.59% | 3.04% |

Against the paper's field models, using their corrected results (arXiv
2212.07564 v3, Appendix N), with mean relative error expressed as a percentage:

| Task | Our best ρ_D | Paper's best ρ_D | Our best mean rel. err. Cd | Paper's best mean rel. err. Cd |
|---|---|---|---|---|
| `full` | 0.993 (MLP) | 0.250 (MLP) | 1.6% (GP) | 618% (MLP) |
| `scarce` | 0.989 (MLP) | 0.254 (GraphSAGE) | 3.4% (MLP) | 454% (MLP) |
| `reynolds` | 0.965 (MLP) | 0.192 (Graph U-Net) | 2.8% (GP) | 829% (MLP) |
| `aoa` | 0.963 (KRG) | 0.552 (Graph U-Net) | 2.0% (KRG) | 435% (MLP) |

On lift, the paper's field models are competitive in ranking (ρ_L ≈ 0.99) but
not in magnitude: their best mean relative Cl error is 15–38%, against 4–7%
here.

Surface curves on the `full` test set (GP, the app's default):

| Quantity | R² (station mean) | RMSE | ±2σ coverage | Paper metric: ours | Paper metric: their best |
|---|---|---|---|---|---|
| Cp(x/c) | 0.989 | 0.062 | 96% | `mean_rel_p` 2.02 | 8.19 (Graph U-Net) |
| Cf(x/c) | 0.966 | 8.5×10⁻⁴ | 96% | `mean_rel_wss` 0.46 / 0.43 | 105 / 135 (GraphSAGE) |

GP identifies upper-surface separation correctly in 97% of test cases, to
within 0.015 c. For NACA 0012 at Re 6×10⁶ its Cp and Cf lie on NASA TMR's
CFL3D SST solution. Full tables, including the paper's original numbers:
[`results/airfrans_benchmark.md`](results/airfrans_benchmark.md). Full write-up:
[`results/airfrans_report.md`](results/airfrans_report.md).

Known weak spot: **Cd extrapolation to lower Re.** On the `reynolds` task
(train Re 3–5×10⁶), GP and Kriging over-predict Cd by up to 2.6× (GP) and
4.6× (KRG) for a handful of thin, cambered sections at negative α below
Re 3×10⁶. RF and MLP extrapolate drag more gracefully there (≤ 1.27×).

## How it works

1. **Ingest.** Each AirfRANS sample stores α, U∞, C_L and C_D as scalars. The
   aerofoil wall is recovered from the mesh as the boundary loop off the outer
   clip box, and reduced to four section features: max thickness `t_max` and
   its location `x_tmax`, max camber `m_max` and its location `x_m`. Re =
   U∞·c/ν with c = 1 m and ν at 298.15 K. Along the wall, Cp comes from the
   stored pressure, and the wall shear stress (not stored) from the velocity
   gradient in the wall cells. Both are resampled to 101 x/c stations per
   surface. A gate checks that integrating them reproduces the stored C_L
   and C_D (median error 0.00% and 1.7%).
2. **QA.** Hard physical checks (0 of 1,000 rows fail), lift-slope and
   zero-lift-angle sanity checks, leave-one-out GP outlier screening (10 rows
   listed, none dropped), and a near-NACA0012 comparison against Ladson's
   experiment and NASA TMR CFL3D SST. See
   [`results/airfrans_qa.md`](results/airfrans_qa.md).
3. **Split.** The official AirfRANS task memberships, read from the dataset
   card, are frozen in `splits/`. Test rows are used once, for evaluation only.
4. **Train.** Inputs `[α, log10 Re, t_max, x_tmax, m_max, x_m]` are
   standardised. The targets are Cl and log Cd. RF leaf size and MLP
   early-stopping patience are chosen by 5-fold CV on the training rows.
   Cp and Cf curves are compressed to 20 PCA modes each (fitted on training
   rows), and the families predict the mode weights.
5. **Query.** For a NACA code, the section is generated analytically (Abbott &
   von Doenhoff) and passed through the same feature extractor, so query
   features match training features.

## Limitations

- **Envelope:** Re 2–6×10⁶, α −5° to 15°, NACA 4- and 5-digit sections only
  (t/c ≈ 0.05–0.20, camber up to ~7%). Outside it you get a warning and an
  extrapolation.
- **Fully turbulent SST, no transition.** Cd is biased high against tripped or
  free-transition experiments at the lower Re values: +10–25% vs Ladson's
  NACA 0012 data (Re 6×10⁶) for cases at Re 2–3×10⁶, within a few percent
  at 6×10⁶.
- **Near stall** (α > ~12°), steady RANS is the least reliable part of the
  data, and the surrogate inherits that.
- **The surrogate can be no more accurate than AirfRANS' own CFD.**
- **Surface curves:** don't integrate predicted Cp/Cf for drag (errors of
  6–8% even for GP, since drag is a small residual of large pressure forces);
  use the direct Cd model. The computed wall shear stress runs ~2.5% high
  against AirfRANS' own drag. Near stall, PCA slightly smooths
  plateau-shaped features.
- `m_max` is measured from the geometric chord, so it reads slightly below the
  nominal NACA camber for cambered sections. Training and queries are
  consistent, but don't compare it one-to-one with NACA digits.

## Repository layout

```text
CamberLab/
├── app.py                        # Streamlit app (reads models/full/)
├── scripts/
│   ├── airfrans/                 # dataset I/O, geometry, wall Cp/Cf, inspection, plot style
│   ├── 20_ingest_airfrans.py     # samples → results/airfrans_dataset.csv
│   ├── 21_qa_airfrans.py         # hard/soft QA, reference comparison
│   ├── 22_make_splits.py         # frozen official splits → splits/
│   ├── 23_surface_gate.py        # wall Cp/Cf vs stored C_L/C_D
│   ├── 09_train_surrogates.py    # GP / RF / MLP / KRG per task (Cl/Cd + curves)
│   ├── 10_global_validation.py   # test metrics, plots, paper benchmark
│   ├── 11_sweep_curves.py        # α sweep for one section
│   ├── predict.py                # single-point CLI
│   └── surrogate/                # data schema, model families, curves, inference API
├── models/full/                  # committed models (other tasks: regenerate)
├── splits/                       # frozen train/test indices + provenance
├── results/                      # dataset, QA, metrics, benchmark, figures, report
├── validation_data/              # Ladson / NASA TMR reference data
├── tests/                        # geometry, schema, reload, CLI, app
├── legacy/regime_v1/             # retired regime-aware OpenFOAM pipeline
└── data/                         # AirfRANS download (gitignored)
```

## Legacy pipeline

CamberLab originally generated its own OpenFOAM 12 dataset with a four-regime
design (per-regime turbulence models, meshes and validation). That pipeline is
archived, with its history, in [`legacy/regime_v1/`](legacy/regime_v1/).

## Contributing

Issues and pull requests are welcome. Please read [`CLAUDE.md`](CLAUDE.md)
first; it is the development guide for this repo. The essentials:

- Work inside the uv virtual environment; dependencies come from PyPI.
- Never edit, drop or fabricate rows of `results/airfrans_dataset.csv`. QA
  outcomes live in `results/airfrans_qa_flags.csv`.
- Never use test indices for fitting or tuning; use 5-fold CV on train.
- Compute query geometry only via `naca_coordinates` + `section_features`.
- `pathlib`, `logging`, `joblib`, seed 42.

## Citation and data licence

The training data is AirfRANS, © Safran, distributed under the
[Open Database License (ODbL 1.0)](https://opendatacommons.org/licenses/odbl/1-0/)
via [`PLAID-datasets/AirfRANS_clipped`](https://huggingface.co/datasets/PLAID-datasets/AirfRANS_clipped).
If you use CamberLab's models or derived data, please cite:

```bibtex
@inproceedings{bonnet2022airfrans,
  title     = {{AirfRANS}: High Fidelity Computational Fluid Dynamics Dataset for
               Approximating {R}eynolds-Averaged {N}avier--{S}tokes Solutions},
  author    = {Bonnet, Florent and Mazari, Ahmed Jocelyn and Cinnella, Paola and
               Gallinari, Patrick},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS),
               Datasets and Benchmarks Track},
  year      = {2022},
  eprint    = {2212.07564},
  archivePrefix = {arXiv}
}
```

## License

Code is released under the MIT License. See [`LICENSE`](LICENSE). The AirfRANS
data and databases derived from it (including `results/airfrans_dataset.csv`
and `results/airfrans_wall_curves.npz`) remain under the ODbL.
