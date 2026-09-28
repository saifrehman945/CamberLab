# OpenFOAM Template — Regime C (Transitional Low-Re)

OpenFOAM 12 master case for **Regime C** of the CamberLab pipeline.
Cases are produced by copying this directory into `cases/case_XXXX/` and
rendering the Jinja-suffixed files with each sample's flow conditions
(`scripts/04_run_cfd.py`).

Modelled on `$FOAM_TUTORIALS/incompressibleFluid/elipsekkLOmega` (the
Foundation's reference k-kL-ω transition case), adapted to this project's
patch contract and Regime C's CFD recipe. See `CLAUDE.md §10` for the regime
spec, `§4–§5` for syntax rules.

## Regime C recipe

| Property         | Value                                                              |
|------------------|--------------------------------------------------------------------|
| α range          | 0°–8°                                                              |
| Re range         | 3×10⁵ – 1×10⁶                                                      |
| Thickness        | 0.08 – 0.15                                                        |
| Physics          | Laminar BL, transition, laminar separation bubbles                |
| Turbulence model | `kkLOmega` (transition; fallback `kOmegaSST` + low Tu)            |
| Wall treatment   | Fully resolved (`nutLowReWallFunction`)                            |
| Target y+        | < 1 (centred at 0.5)                                               |
| Free-stream Tu   | 0.1% (clean-tunnel; lower than A/B/D to permit transition)        |
| Iterations       | 4000 (`endTime` in `controlDict`)                                  |

## k-kL-ω fields — names matter

OpenFOAM 12's `kkLOmega` solves **three** transport scalars with these exact
field names (verified against the `elipsekkLOmega` tutorial — note the
turbulent-KE field is `kt`, **not** `k`):

| Field   | Meaning                       | Wall BC (y+<1, resolved)        |
|---------|-------------------------------|---------------------------------|
| `kt`    | Turbulent kinetic energy      | `fixedValue 0`                  |
| `kl`    | Laminar kinetic energy        | `fixedValue 0`                  |
| `omega` | Specific dissipation rate     | `zeroGradient`                  |
| `nut`   | Eddy viscosity                | `nutLowReWallFunction`          |

> **Note on CLAUDE.md §5.** That section's Regime-C snippet was written from
> memory against an older model interface — it names the field `k` and uses
> `omegaWallFunction` at the wall. The shipped template follows the canonical
> OpenFOAM 12 `elipsekkLOmega` tutorial instead (`kt`/`kl`, `omega`
> zeroGradient), per the §4 rule "never write dictionaries from memory."

## Required mesh patches

Same patch contract as all regimes (`scripts/03_mesh.py` enforces naming):

| Patch         | Type    | BC family                                                     |
|---------------|---------|---------------------------------------------------------------|
| `freestream`  | patch   | `freestream*` / `inletOutlet` / `calculated`                  |
| `aerofoil`    | wall    | `noSlip` / `fixedValue`(kt,kl) / `zeroGradient`(omega) / `nutLowReWallFunction` |
| `frontAndBack`| empty   | `empty` (2D extrusion)                                        |

## Files

```
0/
├── U.template       ← Jinja (rotated inlet velocity, noSlip on aerofoil)
├── p                ← static (freestreamPressure / zeroGradient)
├── kt.template      ← Jinja (turbulent KE; kt_inf from KINF at Tu=0.1%)
├── kl               ← static (laminar KE; 0 in freestream, fixedValue 0 at wall)
├── omega.template   ← Jinja (omega_inf from OMEGAINF; zeroGradient at wall)
└── nut              ← static (nutLowReWallFunction on wall)

constant/
├── momentumTransport     ← kkLOmega RAS
└── physicalProperties    ← nu = 1.5e-5 m²/s, rho = 1.225 kg/m³

system/
├── controlDict.template  ← Jinja (forceCoeffs + yPlus + aerofoilSamples; endTime 4000)
├── fvSchemes             ← bounded linearUpwindV (U); bounded upwind (kt/kl/omega)
├── fvSolution            ← SIMPLE, B-level relaxation, nNonOrthogonalCorrectors 2
└── decomposeParDict.template ← Jinja (NPROCS)
```

## Jinja placeholders

Same set as Regime B (`KINF`/`OMEGAINF` reused for `kt`/`omega`), filled by
`scripts/04_run_cfd.py::build_render_context`:

| Placeholder    | Meaning                                                        |
|----------------|----------------------------------------------------------------|
| `UX`, `UY`     | `U_inf * cos(α)`, `U_inf * sin(α)` — inlet velocity             |
| `UINF`         | Free-stream speed magnitude (m/s)                               |
| `LIFTDIR_X/Y`  | `(-sin(α), cos(α))` for `forceCoeffs.liftDir`                   |
| `DRAGDIR_X/Y`  | `(cos(α), sin(α))` for `forceCoeffs.dragDir`                    |
| `AREF`         | Reference area = `chord × span`                                 |
| `KINF`         | `1.5 * (U_inf * Tu)^2`, **Tu = 0.001 for regime C** (0.1%)      |
| `OMEGAINF`     | `sqrt(KINF) / (C_mu^0.25 * L_t)`, `L_t = 0.07 * chord`          |
| `NPROCS`       | MPI subdomains                                                   |

### Free-stream turbulence intensity

`build_render_context` selects **Tu = 0.1% for regime C** (vs the 1% used by
A/B/D). This is the defining physics knob for transition: a clean-tunnel
free-stream lets the laminar boundary layer persist and form the separation
bubble before transition, which is exactly what `kkLOmega` is here to
capture. A higher Tu would force premature bypass transition and wash out the
regime. `kl_inf` is zero — laminar KE has no free-stream source.

## Wall BC rationale

* `kt = kl = fixedValue 0` — fully-resolved viscous sublayer: both kinetic
  energies vanish at a no-slip wall (the model regenerates `kl` inside the
  BL).
* `omega = zeroGradient` — on a y+<1 mesh `kkLOmega` sets the near-wall
  `omega` internally; this is the tutorial's wall treatment (cf. regime B,
  which uses the blended `omegaWallFunction` for kOmegaSST — a different
  model).
* `nut = nutLowReWallFunction` — enforces ν_t → 0 in the laminar sublayer.

## Solver execution

`scripts/04_run_cfd.py` runs `potentialFoam -initialiseUBCs` before
`foamRun`. The `Phi` solver entry and `potentialFlow` block in `fvSolution`
exist for this preconditioning step — wall-resolved transition models
diverge from a uniform U field on the high-AR boundary-layer cells without
it.

## Post-processing function objects

Identical to all regimes (`controlDict` consistency): `forceCoeffs`
(Cl/Cd with rotated lift/drag directions), `yPlus` (achieved y+ harvested
into `case_metadata.json` — must land < 1 for C), and `aerofoilSamples`
(surface `p` for Cp / suction-peak and bubble diagnostics).

## Validation

Phase 3 of `scripts/08_validation.py`. No canonical NASA TMR low-Re NACA0012
case exists; references are XFOIL eN (n_crit = 9) polars generated locally
(`validation_data/regime_C/metadata.json`, NACA 0012 at Re = 5×10⁵,
α ∈ {0°, 4°, 6°}) — treat as an engineering reference, not ground truth.
Acceptance per that metadata: ΔCl ≤ 15%, ΔCd ≤ 20%, transition x/c ± 0.10.
