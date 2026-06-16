# OpenFOAM Template — Regime D (Fully Turbulent High-Re Attached)

OpenFOAM 12 master case for **Regime D** of the NACASurrogate pipeline.
Cases are produced by copying this directory into `cases/case_XXXX/` and
rendering the Jinja-suffixed files with each sample's flow conditions
(`scripts/04_run_cfd.py`).

Like Regime A, this template is modelled on
`$FOAM_TUTORIALS/incompressibleFluid/airFoil2D/` (the Foundation's
SpalartAllmaras aerofoil case). See `CLAUDE.md §10` for the regime spec,
`§4–§5` for syntax rules.

## Regime D recipe

| Property         | Value                                       |
|------------------|---------------------------------------------|
| α range          | 0°–6°                                       |
| Re range         | 2×10⁶ – 5×10⁶                               |
| Thickness        | 0.10 – 0.18                                 |
| Turbulence model | `SpalartAllmaras`                           |
| Wall treatment   | Wall functions (`nutUSpaldingWallFunction`) |
| Target y+        | 30–80 (centred at 50)                       |
| Iterations       | 2000 (`endTime` in `controlDict`)           |

## Why the CFD recipe is identical to Regime A

Regimes A and D share the **same turbulence model and wall treatment**
(`SpalartAllmaras` + wall functions). The two regimes are distinguished
**not** by the solver dictionaries but by:

1. **Mesh resolution** — D targets y+ ≈ 50 (vs A's 30) with fewer prism
   layers and a lighter wake (`REGIME_MESH["D"]` in
   `scripts/mesh/regime_parameters.py`: `chord_pts=140`, `normal_pts=80`,
   ~70k cells vs A's ~108k).
2. **Design-space bounds** — D covers higher Re (2–5×10⁶) and a narrower α
   band (0–6°), the fully-turbulent attached corner of the space.

Consequently the `0/`, `constant/`, and `system/` dictionaries here are the
same as Regime A's. This is intentional and correct — duplicating the
template (rather than symlinking) keeps each regime self-contained so a
future divergence (e.g. a D-specific scheme tweak) stays local.

### Wall-function choice: `nutUSpaldingWallFunction`, not `nutkWallFunction`

`validation_data/regime_D/metadata.json` nominally lists `nutkWallFunction`,
but that boundary condition derives ν_t from the **turbulent kinetic energy
`k`**, which the Spalart–Allmaras model does not solve (SA transports
`nuTilda`). Every SA tutorial in `$FOAM_TUTORIALS` (airFoil2D,
drivaerFastback, motorBike) uses `nutUSpaldingWallFunction`, which is
velocity-based and valid for y+ ∈ [1, 300] — comfortably covering D's
y+ ≈ 50. CLAUDE.md §10 lists `nutUSpaldingWallFunction` as an accepted
option for D, so this template uses it.

## Required mesh patches

Same patch contract as all regimes (`scripts/03_mesh.py` enforces naming):

| Patch         | Type    | BC family                             |
|---------------|---------|---------------------------------------|
| `freestream`  | patch   | `freestream*` / `calculated`          |
| `aerofoil`    | wall    | `noSlip` / `nutUSpaldingWallFunction` |
| `frontAndBack`| empty   | `empty` (2D extrusion)                |

## Files

```
0/
├── U.template            ← Jinja (rotated inlet velocity)
├── p                     ← static (freestreamPressure, zeroGradient)
├── nuTilda               ← static (freestream value = 3 × nu_inf)
└── nut                   ← static (calculated freestream + Spalding wall function)

constant/
├── momentumTransport     ← SpalartAllmaras RAS
└── physicalProperties    ← nu = 1.5e-5 m²/s, rho = 1.225 kg/m³

system/
├── controlDict.template  ← Jinja (forceCoeffs + yPlus + aerofoilSamples; endTime 2000)
├── fvSchemes             ← second-order linearUpwind (CLAUDE.md §5)
├── fvSolution            ← SIMPLEC + Phi (potentialFoam preconditioning)
└── decomposeParDict.template ← Jinja (NPROCS)
```

## Jinja placeholders

Same set as Regime A, filled by
`scripts/04_run_cfd.py::build_render_context`:

| Placeholder    | Meaning                                            |
|----------------|----------------------------------------------------|
| `UX`, `UY`     | `U_inf * cos(α)`, `U_inf * sin(α)` — inlet velocity |
| `UINF`         | Free-stream speed magnitude (m/s)                   |
| `LIFTDIR_X/Y`  | `(-sin(α), cos(α))` for `forceCoeffs.liftDir`       |
| `DRAGDIR_X/Y`  | `(cos(α), sin(α))` for `forceCoeffs.dragDir`        |
| `AREF`         | Reference area = `chord × span`                     |
| `NPROCS`       | MPI subdomains                                       |

Angle of attack is realised by **rotating the inlet velocity vector**; the
mesh chord always lies on the x-axis (CLAUDE.md §5, §14).

## Post-processing function objects

Identical to all regimes (`controlDict` consistency): `forceCoeffs`
(Cl/Cd with rotated lift/drag directions), `yPlus` (achieved y+ harvested
into `case_metadata.json` — must land in 30–80 for D), and `aerofoilSamples`
(surface `p` for Cp).

## Solver execution

`scripts/04_run_cfd.py` runs `potentialFoam -initialiseUBCs` to seed `U`
and `p` before `foamRun`. The `Phi` solver entry and `potentialFlow` block
in `fvSolution` exist for this preconditioning step.

## Validation

Phase 4 of `scripts/08_validation.py`. Reference and tolerances live in
`validation_data/regime_D/metadata.json` (Ladson / CFL3D SA NACA 0012 at
Re = 6×10⁶, α ∈ {0°, 4°}, plus a Re = 3×10⁶ A↔D overlap check). Acceptance:

| Metric | Tolerance |
|--------|-----------|
| ΔCl    | ±5%       |
| ΔCd    | ±10%      |
