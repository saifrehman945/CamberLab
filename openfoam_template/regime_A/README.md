# OpenFOAM Template — Regime A (Attached Turbulent)

This directory is the OpenFOAM 12 master case for **Regime A** of the
CamberLab pipeline. Cases are produced by copying this directory into
`cases/case_XXXX/` and rendering the Jinja-suffixed files with each sample's
flow conditions (`scripts/04_run_cfd.py`).

The template is modelled on `$FOAM_TUTORIALS/incompressibleFluid/airFoil2D/`,
adapted to the project's patch contract and Regime A's CFD recipe
(see `CLAUDE.md §10` for the regime spec, `§5` for syntax rules).

## Regime A recipe

| Property         | Value                                  |
|------------------|----------------------------------------|
| α range          | 0°–8°                                  |
| Re range         | 1.5×10⁶ – 3×10⁶                        |
| Thickness        | 0.10 – 0.18                            |
| Turbulence model | `SpalartAllmaras`                      |
| Wall treatment   | Wall functions (`nutUSpaldingWallFunction`) |
| Target y+        | 20–50 (centred at 30)                  |
| Iterations       | 2000 (`endTime` in `controlDict`)      |

## Required mesh patches

The mesh used with this case must expose exactly these boundary patches
(`scripts/03_mesh.py` enforces this naming):

| Patch         | Type    | BC family                       |
|---------------|---------|---------------------------------|
| `freestream`  | patch   | `freestream*` / `calculated`    |
| `aerofoil`    | wall    | `noSlip` / `nutUSpaldingWallFunction` |
| `frontAndBack`| empty   | `empty` (2D extrusion)          |

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
├── controlDict.template  ← Jinja (forceCoeffs liftDir/dragDir + UINF + Aref)
├── fvSchemes             ← second-order linearUpwind (CLAUDE.md §5)
└── fvSolution            ← SIMPLEC + Phi (potentialFoam preconditioning)
```

## Jinja placeholders

Filled by `scripts/04_run_cfd.py:build_render_context`:

| Placeholder    | Meaning                                      |
|----------------|----------------------------------------------|
| `UX`, `UY`     | `U_inf * cos(α)`, `U_inf * sin(α)` — inlet velocity components |
| `UINF`         | Free-stream speed magnitude (m/s)            |
| `LIFTDIR_X/Y`  | `(-sin(α), cos(α))` for `forceCoeffs.liftDir`|
| `DRAGDIR_X/Y`  | `(cos(α), sin(α))` for `forceCoeffs.dragDir` |
| `AREF`         | Reference area = `chord × span` (2D extrusion area) |

Angle of attack is realised by **rotating the inlet velocity vector**; the
mesh chord always lies on the x-axis (CLAUDE.md §5, §14).

## Free-stream Spalart–Allmaras values

Following NASA TMR low-turbulence external-aero guidance:

* `nuTilda_inf = 3 × nu_inf = 4.5e-5 m²/s`  → gives `nut_inf / nu ≈ 0.21`
* `nut_inf` at the freestream patch is `calculated` (derived from `nuTilda`)
* At the wall, `nuTilda = 0` (fixedValue), `nut` uses `nutUSpaldingWallFunction`
  which is robust for y+ ∈ [1, 300] and covers Regime A's target y+ ≈ 30.

These are static (independent of α and Re) because `nu_inf` is fixed in
`constant/physicalProperties`.

## Solver execution

`scripts/04_run_cfd.py` runs `potentialFoam -initialiseUBCs` to seed `U` and
`p` before launching `foamRun`. The `Phi` solver entry and `potentialFlow`
block in `fvSolution` exist for this preconditioning step.

## Validation

This template is the **Phase 1** subject of `scripts/08_validate_regimes.py`.
The Regime A lock criteria, reference data, and tolerances live in
`validation_data/regime_A/metadata.json` (Ladson Re=6×10⁶ NACA 0012,
α ∈ {0°, 4°, 8.3°, 10.12°}). Acceptance:

| Metric | Tolerance |
|--------|-----------|
| ΔCl    | ±5%       |
| ΔCd    | ±10%      |

Do not branch Regimes B/C/D from this directory; each regime gets its own
template once Phase 1 is locked (CLAUDE.md §13).
