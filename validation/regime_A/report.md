# Regime A Validation Report

Generated: 2026-05-12T14:05:46 UTC

## Overview

- **Name:** Attached turbulent
- **Physics:** Attached turbulent boundary layer, mild adverse pressure gradient, limited separation
- **Design space:** α ∈ [0, 8], Re ∈ [1500000.0, 3000000.0], t/c ∈ [0.1, 0.18]
- **CFD recipe:** SpalartAllmaras / wall_functions (y+=30.0)

## Tolerance gate

Acceptance: |ΔCl| ≤ 5.0%, |ΔCd| ≤ 10.0% (vs experimental primary).

**Overall result: ❌ FAIL**

## Per-case results

| case_id                 |   α_deg |    Re |   Cl_cfd |   Cd_cfd | ΔCl%   |   ΔCd% | primary_ref                                | pass   |
|:------------------------|--------:|------:|---------:|---------:|:-------|-------:|:-------------------------------------------|:-------|
| naca0012_re6e6_aoa0_sa  |    0    | 6e+06 |  -0.001  |  0.00939 | —      |  16.06 | Ladson tripped (NASA TM 4074, 1988)        | ❌      |
| naca0012_re6e6_aoa4_sa  |    4    | 6e+06 |   0.4337 |  0.01094 | +0.49  |  32.9  | Ladson tripped (Re=6e6, 80-grit)           | ❌      |
| naca0012_re6e6_aoa8_sa  |    8.3  | 6e+06 |   0.8853 |  0.01497 | -0.23  |  42.56 | Ladson tripped (Re=6e6)                    | ❌      |
| naca0012_re6e6_aoa10_sa |   10.12 | 6e+06 |   1.0662 |  0.01771 | -0.42  |  47.42 | Ladson tripped (Re=6e6, 80-grit, α=10.12°) | ❌      |

## Figures

- Cl/Cd comparison: `figures/cl_cd_comparison.png`
- Cp overlay: `figures/cp_comparison__naca0012_re6e6_aoa0_sa.png`
- Cp overlay: `figures/cp_comparison__naca0012_re6e6_aoa10_sa.png`
- Residuals: `figures/residual_history__naca0012_re6e6_aoa0_sa.png`
- Residuals: `figures/residual_history__naca0012_re6e6_aoa4_sa.png`
- Residuals: `figures/residual_history__naca0012_re6e6_aoa8_sa.png`
- Residuals: `figures/residual_history__naca0012_re6e6_aoa10_sa.png`

## Citations

- Ladson, NASA TM 4074, 1988
- Abbott & von Doenhoff, Theory of Wing Sections, 1959
- CFL3D + SA on NASA TMR, https://tmbwg.github.io/turbmodels/naca0012_val_sa.html
- Flow360 NACA0012 validation study
