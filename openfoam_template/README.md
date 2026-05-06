## OpenFOAM Base Case

This directory is the reusable OpenFOAM 12 master case for the NACA surrogate
pipeline. Future CFD cases should be created by copying this directory into a
`cases/case_XXXX/` folder and rendering the Jinja templates with the sample's
flow conditions.

The template is aligned with the current project assumptions:

- Steady incompressible external aerodynamics
- OpenFOAM 12 `foamRun` with `solver incompressibleFluid`
- `kOmegaSST` turbulence model in `constant/momentumTransport`
- Angle of attack imposed by rotating the freestream velocity, not the mesh
- `forceCoeffs` output used to harvest `Cl` and `Cd`

## Required mesh patch names

The mesh used with this case must expose exactly these boundary patches:

- `freestream`: outer far-field boundary
- `aerofoil`: the aerofoil wall patch used by `forceCoeffs`
- `frontAndBack`: the 2D extrusion patches, both set to `empty`

This patch contract is what lets every future case remain a variant of the same
base setup.

## Rendered placeholders

These templates are rendered per case:

- `0/U.template`: `UX`, `UY`
- `0/k.template`: `KINF`
- `0/omega.template`: `OMEGAINF`
- `system/controlDict.template`: `UINF`, `LIFTDIR_X`, `LIFTDIR_Y`,
  `DRAGDIR_X`, `DRAGDIR_Y`

## Turbulence inputs

The stage-4 script derives freestream turbulence using:

- turbulence intensity `I = 1%`
- turbulence length scale `L = 0.07c`

with:

- `k = 1.5 * (U_inf * I)^2`
- `omega = sqrt(k) / (Cmu^0.25 * L)`, `Cmu = 0.09`

## Solver note

The original project README refers to `simpleFoam`. On this OpenFOAM 12
installation, `simpleFoam` has been superseded by `foamRun` with
`solver incompressibleFluid`. The case template therefore uses the current
OpenFOAM 12 syntax while keeping the project output conventions, including
`log.simpleFoam` and `postProcessing/forceCoeffs/...`.
