# CamberLab on AirfRANS — final report

Branch `airfrans-surrogate`, phases 1–9, 2026-09-27: scalar Cl/Cd surrogates.
Branch `airfrans-surface`, phases 1–7, 2026-09-29: moved to the clipped
variant and added wall Cp(x/c) / Cf(x/c) surrogates (§6). Nothing has been
pushed.

## 1. What was built

CamberLab's retired four-regime OpenFOAM dataset (archived in `legacy/regime_v1/`)
was replaced by AirfRANS (Bonnet et al., NeurIPS 2022), read locally from
`PLAID-datasets/AirfRANS_clipped` (originally `AirfRANS_remeshed`; same
samples, scalars and splits, but the clipped variant keeps the original
wall-resolved mesh). Each of the 1,000 samples becomes one row
of `results/airfrans_dataset.csv`. A row holds α, U∞, Re, four section-geometry
features recovered from the CFD mesh wall (`t_max`, `x_tmax`, `m_max`, `x_m`),
and the stored C_L and C_D. The rows are QA-checked without being edited, and
split with the official AirfRANS task memberships. On those splits we fit four
scalar coefficient surrogates (GP, RF, MLP, SMT Kriging) for Cl and log Cd,
per task. The models are then benchmarked against the paper's field models
with the paper's own metrics. A new inference API, `predict(α, Re, naca,
family, task)`, generates analytic NACA 4/5-digit sections and passes them
through the same feature extractor used at ingestion. It powers
`scripts/predict.py`, `scripts/11_sweep_curves.py` and a rewritten Streamlit
app. All gates (1, 2, 3, 4, 6, 8) passed. The surface extension adds a
wall-data gate and curve surrogates for Cp and Cf (§6); `uv run pytest` runs
130 tests green.

## 2. Data findings (Gate 2) and QA (Gate 4)

**Gate 2** (`results/airfrans_inspection.md`):

- Scalars per sample: `angle_of_attack` (**radians**; −0.0862 to 0.2605,
  i.e. −4.94° to 14.93°), `inlet_velocity` (m/s; 31.28 to 93.59), `C_L`, `C_D`.
  Fields: `implicit_distance`, `nut`, `p`, `Ux`, `Uy` (nodal) and `zone_0..5`
  (cell). One unstructured triangular zone per sample, with 155,730–210,414
  nodes in the clipped variant (4,203–9,711 in remeshed). No wall shear
  stress is stored.
- There are 1,000 samples and no original simulation name, so `sample_id` =
  row index. The card's split lists index those same rows.
- Re = U∞·c/ν with ν(298.15 K) = 1.5498×10⁻⁵ m²/s (AirfRANS' polynomial) gives
  Re from 2.02×10⁶ to 6.04×10⁶. AirfRANS' own nominal values are ~0.6% lower:
  both the dataset limits and the `reynolds` split edges imply
  ν ≈ 1.56×10⁻⁵. This is a constant factor with no effect on models.
- The mesh carries no boundary tags. The aerofoil wall is recovered
  topologically, as the boundary loop off the clip box: a single closed loop
  of 843–1,217 nodes per sample in clipped, all with implicit_distance
  exactly 0. (In remeshed: 473–1,670 nodes within |d| ≤ 3.2×10⁻⁴; selecting
  |d| < 10⁻⁶ there would miss about half the wall.)
- Both variants hold the same simulations: re-ingesting from clipped
  reproduces α, U∞, C_L and C_D on all 1,000 rows exactly. The geometry
  features move by at most 2×10⁻⁵ (t_max) to 7×10⁻³ (x_m).
- Streaming the 36 GB clipped variant takes 2.2 s per sample (1.2 s during
  ingestion) at 1.8 GB peak RSS. The reader holds one shard open at a time;
  holding them all retains every visited shard's read buffers and exhausts
  memory.
- **Surprise:** the samples only deserialise with `pyplaid==0.1.7`. Versions
  0.1.8+ and 1.0 reject the serialised schema; `plaid` and `plaid-lib` on PyPI
  are unrelated packages, and `plaid-lib==100.100.100` looks like a squatted
  name.

**Gate 4** (`results/airfrans_qa.md`, flags in `results/airfrans_qa_flags.csv`):

- **Hard checks: 0 of 1,000 rows fail.** No non-finite values, no Cd outside
  (0, 0.3), no |Cl| ≥ 2.5, no duplicates. No rows were excluded from any split.
- Lift slope of near-symmetric sections (|α| ≤ 8°, n = 138): 0.1058 /deg,
  inside the expected 0.09–0.12.
- The zero-lift angle is negative for 100% of the 549 cambered sections, and
  grows more negative with camber down to about −8°.
- Leave-one-out GP outliers (|z| > 5): **rows listed, none dropped** (9 on
  the clipped features; 10 on remeshed: samples 37, 99, 166, 300, 367, 374,
  460, 626, 700, 800). On remeshed, eight are Cd
  outliers, all thin sections (t/c 0.05–0.08) at negative α, mostly cambered.
  Two of them (99, 300; Cd 0.035 and 0.046) show the high drag of lower-surface
  leading-edge separation, and the rest are attached neighbours of that cliff.
  They are physically plausible, so they stay. Samples 374 and 700 are mild Cl
  outliers (|z| ≈ 5.1–5.5) at high camber.

## 3. Reference comparison: Ladson and NASA TMR

21 near-NACA0012 cases (|m_max| < 0.003, 0.11 ≤ t/c ≤ 0.13) span α −4.8° to
13.5° and Re 2.0–6.0×10⁶. Figure: `results/figures/qa/reference_naca0012.png`.

- **Ladson** (NASA TM 4074, Re 6×10⁶, tripped; mean of three grit zones):
  median |ΔCl| = 0.009. Median ΔCd/Cd = +15.5%. The Cd excess falls steadily
  with Re: +14% to +26% for cases at Re 2.0–2.5×10⁶, and +2% to +5% at
  Re 5.7–6.0×10⁶. This is the expected Reynolds effect against a 6×10⁶
  experiment, not a data defect.
- **NASA TMR CFL3D SST** (Re 6×10⁶): three cases lie within ±0.5° of a TMR
  angle. Mean |ΔCl| = 0.009 (TMR Cl interpolated to each case's α); mean
  |ΔCd| = 8.7×10⁻⁴ (10% of the reference Cd). The one case at matching Re
  (sample 997: Re 6.03×10⁶, α 9.66°, t/c 0.1175) agrees to ΔCl = −0.003 and
  ΔCd = −3.6%.

![Near-NACA0012 AirfRANS cases vs Ladson and TMR SST](figures/qa/reference_naca0012.png)

## 4. Benchmark vs the paper

**Comparability.** Our splits are the official AirfRANS memberships from the
dataset card, and QA excluded nothing, so **every test set is identical to
the paper's**. Two differences remain: ours are single deterministic fits
(seed 42), while the paper averages 5 trained copies; and the paper's field
models ingest the full flow mesh, while ours use six scalars.

**Paper numbers.** The primary comparison uses the corrected results: arXiv
2212.07564 v3, Appendix N, Table 27. This follows the authors' 26 May 2023
disclaimer that the main-text models had been trained with plain MSE by
mistake. We verified that the repo's `score_WMSE.json` equals Table 27 value
for value, and that `score_MSE.json` equals main-text Tables 3 and 5. The
original numbers are shown in brackets in `results/airfrans_benchmark.md`.
The metric is `rel_err = |(true − pred)/true|` (`metrics.py:44`), a raw
ratio; below it is expressed as a percentage.

| Task | Our best ρ_D | Paper's best ρ_D | Our best ρ_L | Paper's best ρ_L | Our best mean rel. err. C_D | Paper's best | Our best mean rel. err. C_L | Paper's best |
|---|---|---|---|---|---|---|---|---|
| `full` | 0.993 (MLP) | 0.250 (MLP) | 0.9995 (KRG) | 0.996 (GraphSAGE) | 1.6% (GP) | 618% (MLP) | 4.3% (KRG) | 14.8% (GraphSAGE) |
| `scarce` | 0.989 (MLP) | 0.254 (GraphSAGE) | 0.9992 (GP) | 0.996 (GraphSAGE) | 3.4% (MLP) | 454% (MLP) | 5.3% (GP) | 15.0% (GraphSAGE) |
| `reynolds` | 0.965 (MLP) | 0.192 (Graph U-Net) | 0.9994 (GP) | 0.981 (PointNet) | 2.8% (GP) | 829% (MLP) | 5.0% (GP) | 38.4% (PointNet) |
| `aoa` | 0.963 (KRG) | 0.552 (Graph U-Net) | 0.9984 (GP) | 0.989 (GraphSAGE) | 2.0% (KRG) | 435% (MLP) | 7.3% (GP) | 25.4% (GraphSAGE) |

Numbers are for the models retrained on the clipped variant. Every one of our
16 model/task combinations has ρ_D ≥ 0.91, against a paper-wide best of 0.55.
Our mean relative Cd error is 1.6–9.7%, against 435–1,810% for the field
models. On lift, the field
models rank well (ρ_L 0.96–0.996), but their magnitudes are 3–7× less
accurate than ours.

**Reporting targets for `full`** (Cl R² ≥ 0.99, Cd R² ≥ 0.95, ρ_D ≥ 0.9):
met by GP (0.9993 / 0.9806 / 0.980), KRG (0.9993 / 0.9872 / 0.985) and MLP
(0.9984 / 0.9785 / 0.993). RF misses by a hair on Cd R² (0.9492).

**GP calibration** (share of test points inside ±2σ; the nominal rate is
95.4%): `full` 0.95 (Cl) / 0.95 (Cd); `scarce` 0.95 / 0.94; `reynolds`
0.93 / 0.95; `aoa` 0.94 / 0.89. The bands are well calibrated in
interpolation and slightly overconfident under extrapolation.

## 5. Best family per task, and failure modes

| Task | Recommended | Why |
|---|---|---|
| `full` | **KRG** (GP a close second) | Best Cd R² (0.987) with Cl R² 0.9993; GP has the lowest mean Cd error (1.6%, median 0.59%). KRG and GP also give σ. MLP ranks drag best (ρ_D 0.993) but is 2–3× less accurate in magnitude. |
| `scarce` (200 rows) | **MLP** for Cd, **GP** for Cl | With few rows, GP/KRG Cd degrade (R² 0.93/0.94), while the CV-tuned MLP holds Cd R² 0.946, ρ_D 0.989. GP Cl stays at R² 0.9987. |
| `reynolds` | **MLP** for Cd, **GP/KRG** for Cl | GP now has the lowest mean Cd error (2.8%), but GP and KRG still produce the only large over-predictions (failure mode 1). Cl extrapolates in Re almost perfectly with every smooth model (GP R² 0.9989). |
| `aoa` | **GP** or **KRG** | They extrapolate the stall-side trend: for α > 12.5°, Cl MAE ≈ 0.015–0.018 and Cd median error ≈ 1.3–1.8%. |

Failure modes:

1. **GP/KRG Cd extrapolation to lower Re (`reynolds`).** A handful of thin
   (t/c 0.05–0.06) sections at negative α below Re 3×10⁶ are over-predicted.
   In log-Cd space the GP learns a steep separation cliff from the high-drag
   separated cases at Re 3–5×10⁶ (the QA outliers 99 and 300 are of that
   kind), and it projects the cliff onto attached cases it has never seen at
   lower Re. Retrained on the clipped features, GP Cd R² rose from 0.22 to
   0.91, and its −20% bias at Re < 3×10⁶ disappeared. **This is not a fix.**
   The features moved by only ~10⁻⁴ (standardised), so the jump is the
   marginal-likelihood optimiser landing in a different optimum, which shows
   how fragile this fit is. Five test cases are still over-predicted > 1.5×
   by both GP (worst 2.6×) and KRG (worst 4.6×, sample 166: CFD 0.0083, KRG
   0.0381). MLP and RF have no over-prediction above 1.27×. All four
   families share a few under-predictions to ~0.3× on separated cases.
2. **RF cannot extrapolate (`aoa`).** Its piecewise-constant predictions
   flatten beyond the training α: for α > 12.5° the Cl bias is −0.136 and the
   Cd bias −16%. Its mean relative Cl error, 83%, is inflated further by
   Cl ≈ 0 cases at negative α.
3. **Thin cambered sections at negative α** are the hardest corner for every
   family and task (lower-surface separation). The worst Cd error on `aoa`
   test is ~81–83% for all families, on this kind of case.
4. **Mean relative Cl error is unstable** near Cl = 0 for every model (for
   example GP on `full`: mean 4.5% vs median 1.1%). Read the median.

The tests were evaluated once. No model or hyperparameter was changed after
seeing test results.

## 6. Surface curves: Cp(x/c) and Cf(x/c)

**Wall data.** The datasets store no wall shear stress. `scripts/airfrans/surface.py`
computes it by porting AirfRANS' `metrics.py`: linear (P1) velocity gradients
per triangle are area-averaged onto the wall nodes, τ = 2ν·dev(S)·n with the
paper's ν = 1.56×10⁻⁵, and forces are integrated on the wall polyline. Cp =
p/q∞ (p is kinematic and relative to the far field; stagnation Cp = 1.00).
Cf = τ·t/q∞ is signed with t running from LE to TE on each surface, so
Cf < 0 marks reversed flow. Both are resampled to 101 cosine-spaced x/c
stations per surface: 202 values per case (`results/airfrans_wall_curves.npz`).

**Feasibility gate** (`results/airfrans_surface_gate.md`, D4: median error
≤ 2% C_L / 5% C_D, at most 5% of samples above 3× that). Re-integrated over
all 1,000 samples:

| check | median | 95th pct | worst | pass |
|---|---|---|---|---|
| C_L from pressure only | 0.02% | 0.17% | — | yes |
| C_L, pressure + shear | 0.00% | 0.00% | — | yes |
| C_D, pressure + shear | 1.66% | 2.41% | 2.65% | yes |

Shear carries a median 68% of C_D, so this validates τ_w itself. The C_D
error is a consistent +1.7% bias, not scatter. It is largest at low α and
high Re, where the boundary layer is thinnest, and falls towards 0 at high
α, where pressure drag dominates: τ_w is overestimated by ~2.5%, most likely
by the gradient averaging on the triangulated wall cells (the paper used
VTK's filter on the original mixed-element mesh). On the coarse remeshed
variant the same code recovered only ~40% of the shear drag. That is why the
project moved to clipped.

**Models.** One PCA per quantity, fitted on train rows only. Its mode count
comes from 5-fold CV reconstruction error, capped at 20; every task hits the
cap. At k = 20 the CV reconstruction RMSE is 0.0044 for Cp (0.4% of its
spread) and 1.5×10⁻⁴ for Cf (2.5%, mostly at the leading-edge spikes): the
floor for every family. The existing families predict the 20 mode weights:
GP/KRG with one model per mode (a σ per mode, propagated to a ±2σ curve band
assuming independent modes), RF/MLP with one multi-output model.

**Test results** (`results/airfrans_curve_metrics.csv`; station-mean R²):

| task | GP Cp | KRG Cp | MLP Cp | RF Cp | GP Cf | KRG Cf | MLP Cf | RF Cf |
|---|---|---|---|---|---|---|---|---|
| `full` | 0.989 | 0.989 | 0.983 | 0.895 | 0.966 | 0.963 | 0.961 | 0.829 |
| `scarce` | 0.958 | 0.958 | 0.944 | 0.793 | 0.910 | 0.873 | 0.909 | 0.739 |
| `reynolds` | 0.957 | 0.970 | 0.942 | 0.887 | 0.871 | 0.866 | 0.917 | 0.773 |
| `aoa` | 0.968 | 0.975 | 0.969 | 0.840 | 0.855 | 0.849 | 0.884 | 0.712 |

On `full`, GP has Cp RMSE 0.062 and a suction-peak (min Cp) MAE of 0.10. It
classifies upper-surface separation (Cf < 0 between x/c 0.02 and 0.98;
32 of 200 test cases separate) correctly in 97% of cases, with a
separation-point error of 0.015 c. GP ±2σ bands cover 96% of test stations
(Cp and Cf); KRG bands are overconfident (87%). **Recommended: GP**, the
committed default. KRG is equally accurate but less calibrated; MLP leads
on Cf under extrapolation (`reynolds`, `aoa`). RF is poor at curves.

**Paper comparison.** The paper's `mean_rel_p` and `mean_rel_wss` average
|(true − pred)/true| over wall nodes and then over test cases. We evaluate
them on the same native wall nodes by interpolating our curves (p = Cp·q∞,
τ = Cf·q∞·t). On `full`, ours are 1.4–2.1 (`mean_rel_p`) and 0.43–0.66
(`mean_rel_wss`, x and y) against the paper's 8.2–21.0 and 105–338. On
`aoa`, our `mean_rel_p` (0.53–2.2) is close to the paper's (1.8–2.3). These
ratios explode wherever p or τ crosses zero, so our own metrics above are
the meaningful ones. Full tables: `results/airfrans_benchmark.md`.

**NACA 0012 check** (`results/figures/curves/naca0012_reference.png`). GP
at Re 6×10⁶ and α = 0°, 10° and 15° lies on the NASA TMR CFL3D SST Cp,
including the suction peak. It agrees with Ladson and Gregory's measured Cp
to the usual RANS-vs-experiment level, and follows TMR's SST Cf within its
band.

**Caveats.**
- Do not integrate predicted curves for drag. C_L from the GP curves is
  within 0.9% (median), but C_D is off by 6–8% (GP/KRG) and 50%+ (RF/MLP):
  drag is a small residual of near-cancelling pressure forces. Use the
  scalar C_D models.
- PCA mildly smooths plateau-like features near stall (worst `full` case
  #14, α 14°, Re 2.1×10⁶). The suction peak and separation point are held
  well enough that the pointwise model (option B in the plan) was not
  needed.
- Cf inherits the gate's ~2.5% τ_w overestimate.

**Committed artefacts.** For `full`: PCA, GP (49 MB per quantity) and MLP
curve models. Per-mode KRG (354 MB per quantity) and multi-output RF
(86 MB) are gitignored; retrain them with `09_train_surrogates.py --outputs
curves`. Training all four tasks' curve models takes ~2.5 h.

## 7. Open issues and suggested next steps

- **Deploy weight.** `requirements.txt` now includes `datasets`, `pyarrow` and
  `pyplaid==0.1.7`, which the app never imports. For the live Streamlit demo,
  consider splitting them into a `requirements-ingest.txt`. `models/full/`
  now weighs ~155 MB, of which ~100 MB is the GP curve models; the demo host
  must fit it.
- **Committed models:** only `models/full/` is committed, and there without
  KRG/RF curve models. The `scarce`, `reynolds` and `aoa` model sets are
  gitignored and regenerated with `scripts/09_train_surrogates.py --task all`
  (Cl/Cd ~12 min; curves ~2.5 h).
- **Curve-model size:** a shared-kernel multi-output GP/KRG per quantity
  would cut the curve models ~20× at some accuracy cost. Worth testing if
  size matters more than per-mode length scales.
- **Stale screenshot:** `docs/images/app-overview.png` shows the retired
  regime UI and is no longer referenced from the README. Capture a new one.
- **Low-Re drag extrapolation:** if extrapolation in Re matters, add
  physics-informed structure to the Cd model. For example, fit a residual on
  top of a flat-plate skin-friction baseline ∝ Re^(−1/5), or use a Re-monotone
  kernel, so drag can't fall with decreasing Re. An ensemble of GP and MLP on
  Cd is a cheap mitigation.
- **Features:** `x_tmax` barely varies (0.290–0.303; every NACA 4/5-digit
  section peaks near 30% chord), and the GPs drive its length scale to the
  bound. It is harmless but could be dropped. `m_max` is measured from the
  geometric chord and reads 3–23% below nominal NACA camber for cambered
  sections. This is consistent between training and inference, but should be
  kept in mind when reading it.
- **Seeds:** our numbers are single fits; repeating with 5 seeds would give
  the ± spreads the paper reports.
- **Next models:** a pointwise (x/c-conditioned) model for Cp/Cf near stall,
  if the PCA smoothing there matters. `results/airfrans_surfaces.npz` also
  holds 101-point surfaces for a shape-based (PCA/CST) geometry model.
- **Legacy code** under `legacy/regime_v1/` is archived as-is. Its imports
  assume the old layout, and it is not tested.
