# validation_data/

Authoritative reference datasets for per-regime CFD validation of the
CamberLab project. Every file in this tree is either downloaded from
a public, citable source or computed locally from a published reference
solver (XFOIL) — never synthesized.

## Directory layout

```
validation_data/
├── README.md                 ← this file
├── common/                   ← reference files shared across regimes
│   ├── 0012.abbottdata.cl.dat        Abbott & von Doenhoff NACA0012 Cl
│   ├── 0012.abbottdata.cd.dat        Abbott & von Doenhoff NACA0012 drag polar
│   ├── 0012.mccroskeydata.cl.dat     McCroskey AGARD CL,max data
│   ├── CL_Gregory_expdata.dat        Gregory & O'Reilly Cl
│   ├── CP_Gregory_expdata.dat        Gregory & O'Reilly Cp distributions
│   ├── CLCD_Ladson_expdata.dat       Ladson NASA TM 4074 Cl/Cd (Re=6e6, tripped)
│   ├── CP_Ladson.dat                 Ladson Cp distribution
│   ├── n0012clcd_cfl3d_sa.dat        CFL3D SA Cl/Cd at α∈{0,10,15}
│   ├── n0012cp_cfl3d_sa.dat          CFL3D SA Cp distributions (3 zones)
│   ├── n0012cf_cfl3d_sa.dat          CFL3D SA skin-friction distributions
│   ├── n0012clcd_cfl3d_sst.dat       CFL3D SST Cl/Cd at α∈{0,10,15}
│   ├── n0012cp_cfl3d_sst.dat         CFL3D SST Cp distributions (3 zones)
│   └── n0012cf_cfl3d_sst.dat         CFL3D SST skin-friction distributions
├── regime_A/
│   ├── metadata.json         validation cases + reference mapping
│   ├── raw/                  regime-specific raw files (symlinks to common/ if shared)
│   └── processed/            standardized CSV files (after parsers run)
├── regime_B/
│   ├── metadata.json
│   ├── raw/
│   │   ├── naca4412.cp.expt.dat        Coles & Wadcock Cp (Re=1.52e6, α=13.87°)
│   │   └── exp.profiles.new.dat        Coles & Wadcock 6-station velocity profiles
│   └── processed/
├── regime_C/
│   ├── metadata.json
│   ├── raw/                  populated by scripts/validation/generate_xfoil_reference.py
│   └── processed/
└── regime_D/
    ├── metadata.json
    ├── raw/                  shares common/ NACA0012 data, restricted to low α
    └── processed/
```

## Coordinate and sign conventions (standardized in `processed/`)

- `x/c` ∈ [0, 1] running from leading edge (0) to trailing edge (1).
- Cp surface convention: `Cp = (p − p_inf) / (0.5 ρ U_inf²)`; suction is negative.
- Cl, Cd positive in the lift/drag directions (lift normal to freestream, drag parallel).
- `alpha_deg` in degrees, always.
- All `processed/` CSVs use the same columns:
  - Cl/Cd: `alpha_deg, Cl, Cd, source, Re, M, trip_condition`
  - Cp:    `x_c, Cp, alpha_deg, source, Re, M`

## Sources and citations

### NASA Turbulence Modeling Resource (TMR)

- Project page: https://tmbwg.github.io/turbmodels/
- NACA 0012 main validation: https://tmbwg.github.io/turbmodels/naca0012_val.html
- NACA 0012 SA model results: https://tmbwg.github.io/turbmodels/naca0012_val_sa.html
- NACA 0012 SST model results: https://tmbwg.github.io/turbmodels/naca0012_val_sst.html
- NACA 4412 separation: https://tmbwg.github.io/turbmodels/naca4412sep_val.html

### Experimental data

- Ladson, C. L., "Effects of Independent Variation of Mach and Reynolds
  Numbers on the Low-Speed Aerodynamic Characteristics of the NACA 0012
  Airfoil Section," NASA TM 4074, October 1988.
- Ladson, C. L., Hill, A. S., Johnson, W. G., "Pressure Distributions
  from High Reynolds Number Transonic Tests of a NACA 0012 Airfoil in
  the Langley 0.3-Meter Transonic Cryogenic Tunnel," NASA TM 100526, 1987.
- Gregory, N. and O'Reilly, C. L., "Low-Speed Aerodynamic Characteristics
  of NACA 0012 Aerofoil Sections, including the Effects of Upper-Surface
  Roughness Simulation Hoar Frost," ARC R&M 3726, January 1970.
- Abbott, I. H. and von Doenhoff, A. E., *Theory of Wing Sections*,
  Dover Publications, New York, 1959.
- McCroskey, W. J., "A Critical Assessment of Wind Tunnel Results for the
  NACA 0012 Airfoil," AGARD CP-429, July 1988; also NASA TM 100019, 1987.
- Coles, D. and Wadcock, A. J., "Flying-Hot-Wire Study of Flow Past an
  NACA 4412 Airfoil at Maximum Lift," *AIAA Journal*, Vol. 17, No. 4,
  1979, pp. 321–329.
- Wadcock, A. J., "Flying-Hot-Wire Study of Two-Dimensional Turbulent
  Flow Separation on an NACA 4412 Airfoil at Maximum Lift,"
  NASA CR-152263, February 1979.

### Cross-validation against engineering CFD codes

- Flow360 NACA 0012 validation study (Flexcompute):
  https://docs.flexcompute.com/projects/flow360/en/latest/knowledge_base/validationStudies/NACA0012/NACA0012.html
- CFL3D, FUN3D, NTS, JOE, SUMB, TURNS, GGNS — code-to-code comparison
  hosted on the NASA TMR pages above.

### XFOIL (Regime C local generation)

- Drela, M., "XFOIL: An Analysis and Design System for Low Reynolds
  Number Airfoils," *Low Reynolds Number Aerodynamics*, T. J. Mueller, ed.,
  Lecture Notes in Engineering Vol. 54, Springer-Verlag, Berlin, 1989.

## Provenance

Every file in `validation_data/common/`, `validation_data/regime_B/raw/`, and
the entries listed in each `metadata.json` were downloaded via HTTPS from the
URLs above. The download date and SHA-256 of each file are recorded in the
regime's `metadata.json` under `references[*].file_provenance`. To re-fetch:

```bash
uv run python scripts/validation/fetch_reference_data.py
```

XFOIL-generated reference data for Regime C is reproducible via:

```bash
uv run python scripts/validation/generate_xfoil_reference.py
```

This requires `xfoil` to be present on PATH (`apt install xfoil` on Debian/Ubuntu).
