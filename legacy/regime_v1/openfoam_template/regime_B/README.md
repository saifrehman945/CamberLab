# OpenFOAM Template — Regime B (Near-Stall Separated)

OpenFOAM 12 master case for **Regime B** of the CamberLab pipeline.
Modelled on `$FOAM_TUTORIALS/fluid/aerofoilNACA0012Steady` (the OpenFOAM
Foundation's reference kOmegaSST aerofoil case) adapted to this project's
patch contract and Regime B's CFD recipe (see `CLAUDE.md §10` for the
regime spec, `§4–§5` for syntax rules).

## Regime B recipe

| Property         | Value                                                            |
|------------------|------------------------------------------------------------------|
| α range          | 10°–16°                                                          |
| Re range         | 1×10⁶ – 3×10⁶                                                    |
| Thickness        | 0.12 – 0.24                                                      |
| Turbulence model | `kOmegaSST` (fully resolved)                                     |
| Wall treatment   | Low-Re (`kLowReWallFunction`, `omegaWallFunction`, `nutLowReWallFunction`) |
| Target y+        | < 1 (centred at 0.5)                                             |
| Iterations       | 5000 (`endTime` in `controlDict`)                                |

## Required mesh patches

Same patch contract as Regime A — `scripts/03_mesh.py` enforces this naming
across all regimes:

| Patch         | Type    | BC family                                |
|---------------|---------|------------------------------------------|
| `freestream`  | patch   | `freestream*` / `inletOutlet` / `calculated` |
| `aerofoil`    | wall    | `noSlip` / `kLowReWallFunction` / `omegaWallFunction` / `nutLowReWallFunction` |
| `frontAndBack`| empty   | `empty` (2D extrusion)                   |

## Files

```
0/
├── U.template       ← Jinja (rotated inlet velocity, noSlip on aerofoil)
├── p                ← static (freestreamPressure / zeroGradient)
├── k.template       ← Jinja (k_inf from build_render_context; kLowReWallFunction on wall)
├── omega.template   ← Jinja (omega_inf from build_render_context; omegaWallFunction on wall)
└── nut              ← static (nutLowReWallFunction on wall)

constant/
├── momentumTransport     ← kOmegaSST RAS
└── physicalProperties    ← nu = 1.5e-5 m²/s, rho = 1.225 kg/m³

system/
├── controlDict.template  ← Jinja (forceCoeffs liftDir/dragDir + UINF + Aref; endTime 5000)
├── fvSchemes             ← bounded linearUpwindV for U; upwind for k/omega
├── fvSolution            ← SIMPLE with 0.7× A's relaxation; nNonOrthogonalCorrectors 2
└── decomposeParDict.template ← Jinja (NPROCS)
```

## Jinja placeholders

Same set as Regime A plus `KINF` and `OMEGAINF` (already computed in
`scripts/04_run_cfd.py::build_render_context`):

| Placeholder    | Meaning                                                |
|----------------|--------------------------------------------------------|
| `UX`, `UY`     | `U_inf * cos(α)`, `U_inf * sin(α)` — inlet velocity     |
| `UINF`         | Free-stream speed magnitude (m/s)                       |
| `LIFTDIR_X/Y`  | `(-sin(α), cos(α))` for `forceCoeffs.liftDir`           |
| `DRAGDIR_X/Y`  | `(cos(α), sin(α))` for `forceCoeffs.dragDir`            |
| `AREF`         | Reference area = `chord × span`                         |
| `KINF`         | `1.5 * (U_inf * Tu)^2` with `Tu = 0.01`                 |
| `OMEGAINF`     | `sqrt(k_inf) / (C_mu^0.25 * L_t)`, `L_t = 0.07 * chord` |
| `NPROCS`       | MPI subdomains                                          |

## Wall BC choice rationale

* `kLowReWallFunction` is the OpenFOAM 12 low-Re k boundary; it sets a
  tiny floor (1e-10) at the wall consistent with the analytical near-wall
  k-profile in the viscous sublayer.
* `omegaWallFunction` is the unified wall function — valid in both the
  low-Re limit (y+ → 0, blending to the analytical ω ∝ ν/y² near-wall
  asymptote) and the high-Re limit. Safe for Regime B's y+ < 1 mesh.
* `nutLowReWallFunction` enforces ν_t = 0 at the wall (no eddy viscosity
  in the laminar sublayer).

The `omegaWallFunction` wall value of 1 is a numerical seed only; the
function object replaces it with the correct ω blending each iteration.

## Solver execution

`scripts/04_run_cfd.py` runs `potentialFoam -initialiseUBCs` before
`foamRun`. The `Phi` solver entry and `potentialFlow` block in
`fvSolution` exist for this preconditioning step. Wall-resolved kOmegaSST
will diverge from a uniform U field on the high-AR BL cells without this
preconditioning.

## Validation

Phase 2 of `scripts/08_validation.py`. Reference: Ladson (NASA TM-4074)
NACA0012 α-sweep at Re = 1.8×10⁶, free transition, α ∈ {10°, 12°, 14°,
16°}. Acceptance per CLAUDE.md §13 (with a graded ΔCl tolerance near stall):

| α    | Reference  | Tolerance       |
|------|------------|-----------------|
| 10°  | Ladson     | ΔCl ≤ 5%        |
| 12°  | Ladson     | ΔCl ≤ 7%        |
| 14°  | Ladson     | ΔCl ≤ 10%       |
| 16°  | Ladson     | ΔCl ≤ 15%, stall sign captured |

Plus ΔCd ≤ 10% and qualitative Cp suction-peak agreement at all four α.
