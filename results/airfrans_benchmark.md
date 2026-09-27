# AirfRANS benchmark — scalar coefficient surrogates vs the paper's field models

Our models regress C_L and C_D directly from (α, Re, section geometry); the paper's models predict the flow field and integrate forces from it. Metrics follow the paper exactly: Spearman's ρ between true and predicted coefficients over the test set, and the mean relative error `|(true − pred)/true|` (Extrality/AirfRANS `metrics.py`, `rel_err`). The relative error is a **raw ratio, not a percentage**: 0.05 = 5%.

**Test sets.** Our splits are the official AirfRANS memberships (read from the dataset card; see `splits/README.md`) and QA excluded no rows, so every test set is identical to the paper's. Our numbers are one deterministic fit per model (seed 42); the paper's are means over 5 trained copies.

**Which paper numbers.** Primary: the corrected results of arXiv:2212.07564 v3, Appendix N Table 27 (`score_WMSE.json`), after the authors' 26 May 2023 disclaimer that the main-text models had been trained with a plain MSE loss. The original main-text numbers (`score_MSE.json`, Tables 3 and 5) are given in brackets. Both files are cached in `results/reference/airfrans_paper/`.

## Task `full`

| model | ρ_D | ρ_L | mean rel. err. C_D | mean rel. err. C_L | median rel. err. C_D | median rel. err. C_L |
|---|---|---|---|---|---|---|
| **ours — GP** | 0.978 | 1.000 | 0.0160 | 0.045 | 0.0051 | 0.0107 |
| **ours — RF** | 0.980 | 0.996 | 0.0432 | 0.114 | 0.0302 | 0.0436 |
| **ours — MLP** | 0.992 | 0.999 | 0.0278 | 0.074 | 0.0193 | 0.0200 |
| **ours — KRG** | 0.987 | 1.000 | 0.0150 | 0.043 | 0.0066 | 0.0092 |
| paper — MLP | 0.250 [-0.117] | 0.993 [0.913] | 6.178 [4.289] | 0.211 [0.769] | — | — |
| paper — GraphSAGE | 0.194 [-0.303] | 0.996 [0.965] | 7.366 [4.050] | 0.148 [0.517] | — | — |
| paper — PointNet | 0.074 [-0.022] | 0.992 [0.938] | 17.392 [14.637] | 0.197 [0.742] | — | — |
| paper — Graph U-Net | 0.092 [-0.138] | 0.995 [0.967] | 13.320 [10.385] | 0.168 [0.489] | — | — |

## Task `scarce`

| model | ρ_D | ρ_L | mean rel. err. C_D | mean rel. err. C_L | median rel. err. C_D | median rel. err. C_L |
|---|---|---|---|---|---|---|
| **ours — GP** | 0.953 | 0.999 | 0.0578 | 0.053 | 0.0264 | 0.0162 |
| **ours — RF** | 0.947 | 0.990 | 0.0690 | 0.217 | 0.0457 | 0.0844 |
| **ours — MLP** | 0.990 | 0.997 | 0.0315 | 0.120 | 0.0183 | 0.0377 |
| **ours — KRG** | 0.972 | 0.999 | 0.0422 | 0.055 | 0.0111 | 0.0188 |
| paper — MLP | 0.248 [-0.242] | 0.993 [0.923] | 4.540 [2.949] | 0.199 [0.662] | — | — |
| paper — GraphSAGE | 0.254 [-0.139] | 0.996 [0.981] | 4.587 [3.504] | 0.150 [0.385] | — | — |
| paper — PointNet | 0.048 [-0.050] | 0.992 [0.949] | 16.048 [8.350] | 0.200 [0.587] | — | — |
| paper — Graph U-Net | 0.074 [-0.095] | 0.994 [0.976] | 10.726 [6.871] | 0.150 [0.418] | — | — |

## Task `reynolds`

| model | ρ_D | ρ_L | mean rel. err. C_D | mean rel. err. C_L | median rel. err. C_D | median rel. err. C_L |
|---|---|---|---|---|---|---|
| **ours — GP** | 0.739 | 0.999 | 0.1577 | 0.051 | 0.0539 | 0.0119 |
| **ours — RF** | 0.959 | 0.996 | 0.0620 | 0.150 | 0.0470 | 0.0468 |
| **ours — MLP** | 0.962 | 0.998 | 0.0646 | 0.105 | 0.0463 | 0.0362 |
| **ours — KRG** | 0.933 | 0.999 | 0.0683 | 0.053 | 0.0234 | 0.0156 |
| paper — MLP | 0.157 [-0.146] | 0.958 [0.642] | 8.293 [13.397] | 0.621 [3.330] | — | — |
| paper — GraphSAGE | 0.039 [0.013] | 0.971 [0.927] | 12.794 [8.972] | 0.433 [0.616] | — | — |
| paper — PointNet | 0.115 [0.006] | 0.981 [0.898] | 17.111 [11.558] | 0.384 [0.897] | — | — |
| paper — Graph U-Net | 0.192 [0.028] | 0.964 [0.904] | 18.103 [13.268] | 0.466 [0.868] | — | — |

## Task `aoa`

| model | ρ_D | ρ_L | mean rel. err. C_D | mean rel. err. C_L | median rel. err. C_D | median rel. err. C_L |
|---|---|---|---|---|---|---|
| **ours — GP** | 0.963 | 0.998 | 0.0207 | 0.069 | 0.0060 | 0.0147 |
| **ours — RF** | 0.911 | 0.978 | 0.0970 | 0.833 | 0.0500 | 0.3536 |
| **ours — MLP** | 0.946 | 0.997 | 0.0459 | 0.144 | 0.0307 | 0.0359 |
| **ours — KRG** | 0.964 | 0.998 | 0.0224 | 0.077 | 0.0071 | 0.0246 |
| paper — MLP | 0.347 [0.038] | 0.957 [0.861] | 4.355 [8.003] | 0.413 [1.061] | — | — |
| paper — GraphSAGE | 0.525 [0.055] | 0.989 [0.908] | 6.047 [5.589] | 0.254 [0.818] | — | — |
| paper — PointNet | 0.089 [0.122] | 0.978 [0.936] | 13.846 [8.991] | 0.442 [0.716] | — | — |
| paper — Graph U-Net | 0.552 [-0.195] | 0.982 [0.934] | 9.814 [10.238] | 0.376 [0.693] | — | — |

## Our models, all test metrics

| task | family | target | R² | RMSE | MAE | Spearman | mean rel. err. | median rel. err. | GP ±2σ coverage |
|---|---|---|---|---|---|---|---|---|---|
| full | GP | Cl | 0.9992 | 0.0162 | 0.0104 | 0.9995 | 0.0447 | 0.0107 | 0.950 |
| full | GP | Cd | 0.9768 | 0.000719 | 0.0002 | 0.9784 | 0.0160 | 0.0051 | 0.965 |
| full | RF | Cl | 0.9923 | 0.0507 | 0.0361 | 0.9962 | 0.1143 | 0.0436 | — |
| full | RF | Cd | 0.9498 | 0.00106 | 0.000628 | 0.9800 | 0.0432 | 0.0302 | — |
| full | MLP | Cl | 0.9984 | 0.023 | 0.0167 | 0.9990 | 0.0740 | 0.0200 | — |
| full | MLP | Cd | 0.9771 | 0.000715 | 0.000397 | 0.9922 | 0.0278 | 0.0193 | — |
| full | KRG | Cl | 0.9992 | 0.0159 | 0.0102 | 0.9995 | 0.0428 | 0.0092 | — |
| full | KRG | Cd | 0.9881 | 0.000515 | 0.000188 | 0.9869 | 0.0150 | 0.0066 | — |
| scarce | GP | Cl | 0.9986 | 0.022 | 0.015 | 0.9991 | 0.0534 | 0.0162 | 0.940 |
| scarce | GP | Cd | 0.8929 | 0.00155 | 0.000694 | 0.9535 | 0.0578 | 0.0264 | 0.925 |
| scarce | RF | Cl | 0.9763 | 0.0891 | 0.0665 | 0.9896 | 0.2172 | 0.0844 | — |
| scarce | RF | Cd | 0.9106 | 0.00141 | 0.000918 | 0.9474 | 0.0690 | 0.0457 | — |
| scarce | MLP | Cl | 0.9938 | 0.0458 | 0.0328 | 0.9966 | 0.1201 | 0.0377 | — |
| scarce | MLP | Cd | 0.9566 | 0.000985 | 0.00048 | 0.9903 | 0.0315 | 0.0183 | — |
| scarce | KRG | Cl | 0.9983 | 0.0241 | 0.0163 | 0.9990 | 0.0550 | 0.0188 | — |
| scarce | KRG | Cd | 0.9395 | 0.00116 | 0.000577 | 0.9723 | 0.0422 | 0.0111 | — |
| reynolds | GP | Cl | 0.9989 | 0.0191 | 0.0129 | 0.9994 | 0.0506 | 0.0119 | 0.933 |
| reynolds | GP | Cd | 0.2190 | 0.00446 | 0.00193 | 0.7387 | 0.1577 | 0.0539 | 0.958 |
| reynolds | RF | Cl | 0.9916 | 0.0525 | 0.0388 | 0.9960 | 0.1498 | 0.0468 | — |
| reynolds | RF | Cd | 0.8620 | 0.00187 | 0.000923 | 0.9595 | 0.0620 | 0.0470 | — |
| reynolds | MLP | Cl | 0.9962 | 0.0355 | 0.0275 | 0.9981 | 0.1052 | 0.0362 | — |
| reynolds | MLP | Cd | 0.8926 | 0.00165 | 0.000864 | 0.9624 | 0.0646 | 0.0463 | — |
| reynolds | KRG | Cl | 0.9988 | 0.0196 | 0.0135 | 0.9994 | 0.0530 | 0.0156 | — |
| reynolds | KRG | Cd | 0.7095 | 0.00272 | 0.000833 | 0.9329 | 0.0683 | 0.0234 | — |
| aoa | GP | Cl | 0.9995 | 0.0194 | 0.0134 | 0.9983 | 0.0694 | 0.0147 | 0.939 |
| aoa | GP | Cd | 0.8291 | 0.00327 | 0.000597 | 0.9630 | 0.0207 | 0.0060 | 0.898 |
| aoa | RF | Cl | 0.9536 | 0.183 | 0.163 | 0.9779 | 0.8331 | 0.3536 | — |
| aoa | RF | Cd | 0.6270 | 0.00483 | 0.0023 | 0.9110 | 0.0970 | 0.0500 | — |
| aoa | MLP | Cl | 0.9986 | 0.0321 | 0.0248 | 0.9969 | 0.1443 | 0.0359 | — |
| aoa | MLP | Cd | 0.8013 | 0.00352 | 0.00102 | 0.9456 | 0.0459 | 0.0307 | — |
| aoa | KRG | Cl | 0.9994 | 0.0204 | 0.0152 | 0.9982 | 0.0774 | 0.0246 | — |
| aoa | KRG | Cd | 0.8288 | 0.00327 | 0.000631 | 0.9636 | 0.0224 | 0.0071 | — |

## Reporting targets (task `full`, not gates)

Cl R² ≥ 0.99, Cd R² ≥ 0.95, Spearman ρ_D ≥ 0.9.

| family | Cl R² | Cd R² | ρ_D | all met |
|---|---|---|---|---|
| GP | 0.9992 | 0.9768 | 0.9784 | yes |
| RF | 0.9923 | 0.9498 | 0.9800 | no |
| MLP | 0.9984 | 0.9771 | 0.9922 | yes |
| KRG | 0.9992 | 0.9881 | 0.9869 | yes |

Mean relative error on C_L is dominated by test cases with C_L near zero (small denominators), which is why the median is also reported.

Figures: `results/figures/eval/parity_{task}_{Cl,Cd}.png`, `results/figures/eval/residuals_{task}.png`.
