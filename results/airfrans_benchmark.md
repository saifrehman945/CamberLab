# AirfRANS benchmark — scalar coefficient surrogates vs the paper's field models

Our models regress C_L and C_D directly from (α, Re, section geometry); the paper's models predict the flow field and integrate forces from it. Metrics follow the paper exactly: Spearman's ρ between true and predicted coefficients over the test set, and the mean relative error `|(true − pred)/true|` (Extrality/AirfRANS `metrics.py`, `rel_err`). The relative error is a **raw ratio, not a percentage**: 0.05 = 5%.

**Test sets.** Our splits are the official AirfRANS memberships (read from the dataset card; see `splits/README.md`) and QA excluded no rows, so every test set is identical to the paper's. Our numbers are one deterministic fit per model (seed 42); the paper's are means over 5 trained copies.

**Which paper numbers.** Primary: the corrected results of arXiv:2212.07564 v3, Appendix N Table 27 (`score_WMSE.json`), after the authors' 26 May 2023 disclaimer that the main-text models had been trained with a plain MSE loss. The original main-text numbers (`score_MSE.json`, Tables 3 and 5) are given in brackets. Both files are cached in `results/reference/airfrans_paper/`.

## Task `full`

| model | ρ_D | ρ_L | mean rel. err. C_D | mean rel. err. C_L | median rel. err. C_D | median rel. err. C_L |
|---|---|---|---|---|---|---|
| **ours — GP** | 0.980 | 0.999 | 0.0158 | 0.045 | 0.0059 | 0.0107 |
| **ours — RF** | 0.980 | 0.996 | 0.0434 | 0.114 | 0.0304 | 0.0459 |
| **ours — MLP** | 0.993 | 0.999 | 0.0279 | 0.077 | 0.0183 | 0.0186 |
| **ours — KRG** | 0.985 | 1.000 | 0.0162 | 0.043 | 0.0071 | 0.0090 |
| paper — MLP | 0.250 [-0.117] | 0.993 [0.913] | 6.178 [4.289] | 0.211 [0.769] | — | — |
| paper — GraphSAGE | 0.194 [-0.303] | 0.996 [0.965] | 7.366 [4.050] | 0.148 [0.517] | — | — |
| paper — PointNet | 0.074 [-0.022] | 0.992 [0.938] | 17.392 [14.637] | 0.197 [0.742] | — | — |
| paper — Graph U-Net | 0.092 [-0.138] | 0.995 [0.967] | 13.320 [10.385] | 0.168 [0.489] | — | — |

## Task `scarce`

| model | ρ_D | ρ_L | mean rel. err. C_D | mean rel. err. C_L | median rel. err. C_D | median rel. err. C_L |
|---|---|---|---|---|---|---|
| **ours — GP** | 0.932 | 0.999 | 0.0433 | 0.053 | 0.0168 | 0.0145 |
| **ours — RF** | 0.948 | 0.990 | 0.0688 | 0.217 | 0.0459 | 0.0799 |
| **ours — MLP** | 0.989 | 0.996 | 0.0339 | 0.127 | 0.0203 | 0.0393 |
| **ours — KRG** | 0.973 | 0.999 | 0.0425 | 0.057 | 0.0112 | 0.0182 |
| paper — MLP | 0.248 [-0.242] | 0.993 [0.923] | 4.540 [2.949] | 0.199 [0.662] | — | — |
| paper — GraphSAGE | 0.254 [-0.139] | 0.996 [0.981] | 4.587 [3.504] | 0.150 [0.385] | — | — |
| paper — PointNet | 0.048 [-0.050] | 0.992 [0.949] | 16.048 [8.350] | 0.200 [0.587] | — | — |
| paper — Graph U-Net | 0.074 [-0.095] | 0.994 [0.976] | 10.726 [6.871] | 0.150 [0.418] | — | — |

## Task `reynolds`

| model | ρ_D | ρ_L | mean rel. err. C_D | mean rel. err. C_L | median rel. err. C_D | median rel. err. C_L |
|---|---|---|---|---|---|---|
| **ours — GP** | 0.954 | 0.999 | 0.0280 | 0.050 | 0.0080 | 0.0125 |
| **ours — RF** | 0.962 | 0.996 | 0.0612 | 0.150 | 0.0460 | 0.0460 |
| **ours — MLP** | 0.965 | 0.998 | 0.0631 | 0.118 | 0.0446 | 0.0332 |
| **ours — KRG** | 0.936 | 0.999 | 0.0665 | 0.052 | 0.0223 | 0.0152 |
| paper — MLP | 0.157 [-0.146] | 0.958 [0.642] | 8.293 [13.397] | 0.621 [3.330] | — | — |
| paper — GraphSAGE | 0.039 [0.013] | 0.971 [0.927] | 12.794 [8.972] | 0.433 [0.616] | — | — |
| paper — PointNet | 0.115 [0.006] | 0.981 [0.898] | 17.111 [11.558] | 0.384 [0.897] | — | — |
| paper — Graph U-Net | 0.192 [0.028] | 0.964 [0.904] | 18.103 [13.268] | 0.466 [0.868] | — | — |

## Task `aoa`

| model | ρ_D | ρ_L | mean rel. err. C_D | mean rel. err. C_L | median rel. err. C_D | median rel. err. C_L |
|---|---|---|---|---|---|---|
| **ours — GP** | 0.962 | 0.998 | 0.0204 | 0.073 | 0.0056 | 0.0141 |
| **ours — RF** | 0.910 | 0.978 | 0.0971 | 0.841 | 0.0491 | 0.3549 |
| **ours — MLP** | 0.947 | 0.997 | 0.0475 | 0.134 | 0.0314 | 0.0375 |
| **ours — KRG** | 0.963 | 0.998 | 0.0204 | 0.081 | 0.0065 | 0.0220 |
| paper — MLP | 0.347 [0.038] | 0.957 [0.861] | 4.355 [8.003] | 0.413 [1.061] | — | — |
| paper — GraphSAGE | 0.525 [0.055] | 0.989 [0.908] | 6.047 [5.589] | 0.254 [0.818] | — | — |
| paper — PointNet | 0.089 [0.122] | 0.978 [0.936] | 13.846 [8.991] | 0.442 [0.716] | — | — |
| paper — Graph U-Net | 0.552 [-0.195] | 0.982 [0.934] | 9.814 [10.238] | 0.376 [0.693] | — | — |

## Our models, all test metrics

| task | family | target | R² | RMSE | MAE | Spearman | mean rel. err. | median rel. err. | GP ±2σ coverage |
|---|---|---|---|---|---|---|---|---|---|
| full | GP | Cl | 0.9993 | 0.0157 | 0.0102 | 0.9995 | 0.0447 | 0.0107 | 0.950 |
| full | GP | Cd | 0.9806 | 0.000659 | 0.000195 | 0.9799 | 0.0158 | 0.0059 | 0.950 |
| full | RF | Cl | 0.9921 | 0.0514 | 0.0365 | 0.9961 | 0.1143 | 0.0459 | — |
| full | RF | Cd | 0.9492 | 0.00107 | 0.000632 | 0.9797 | 0.0434 | 0.0304 | — |
| full | MLP | Cl | 0.9984 | 0.0234 | 0.017 | 0.9989 | 0.0768 | 0.0186 | — |
| full | MLP | Cd | 0.9785 | 0.000692 | 0.000394 | 0.9931 | 0.0279 | 0.0183 | — |
| full | KRG | Cl | 0.9993 | 0.0157 | 0.0103 | 0.9995 | 0.0429 | 0.0090 | — |
| full | KRG | Cd | 0.9872 | 0.000535 | 0.000206 | 0.9852 | 0.0162 | 0.0071 | — |
| scarce | GP | Cl | 0.9987 | 0.0212 | 0.0145 | 0.9992 | 0.0530 | 0.0145 | 0.950 |
| scarce | GP | Cd | 0.9306 | 0.00125 | 0.000511 | 0.9324 | 0.0433 | 0.0168 | 0.940 |
| scarce | RF | Cl | 0.9767 | 0.0885 | 0.066 | 0.9899 | 0.2168 | 0.0799 | — |
| scarce | RF | Cd | 0.9112 | 0.00141 | 0.000916 | 0.9484 | 0.0688 | 0.0459 | — |
| scarce | MLP | Cl | 0.9930 | 0.0484 | 0.034 | 0.9963 | 0.1274 | 0.0393 | — |
| scarce | MLP | Cd | 0.9458 | 0.0011 | 0.000522 | 0.9890 | 0.0339 | 0.0203 | — |
| scarce | KRG | Cl | 0.9985 | 0.0224 | 0.0154 | 0.9991 | 0.0571 | 0.0182 | — |
| scarce | KRG | Cd | 0.9385 | 0.00117 | 0.000583 | 0.9727 | 0.0425 | 0.0112 | — |
| reynolds | GP | Cl | 0.9989 | 0.0191 | 0.0129 | 0.9994 | 0.0498 | 0.0125 | 0.929 |
| reynolds | GP | Cd | 0.9075 | 0.00153 | 0.00035 | 0.9537 | 0.0280 | 0.0080 | 0.946 |
| reynolds | RF | Cl | 0.9916 | 0.0525 | 0.039 | 0.9961 | 0.1497 | 0.0460 | — |
| reynolds | RF | Cd | 0.8580 | 0.0019 | 0.000922 | 0.9621 | 0.0612 | 0.0460 | — |
| reynolds | MLP | Cl | 0.9961 | 0.0358 | 0.0279 | 0.9981 | 0.1183 | 0.0332 | — |
| reynolds | MLP | Cd | 0.8941 | 0.00164 | 0.000844 | 0.9646 | 0.0631 | 0.0446 | — |
| reynolds | KRG | Cl | 0.9988 | 0.0197 | 0.0136 | 0.9994 | 0.0524 | 0.0152 | — |
| reynolds | KRG | Cd | 0.7207 | 0.00267 | 0.000812 | 0.9358 | 0.0665 | 0.0223 | — |
| aoa | GP | Cl | 0.9995 | 0.0199 | 0.0136 | 0.9984 | 0.0726 | 0.0141 | 0.939 |
| aoa | GP | Cd | 0.8291 | 0.00327 | 0.000591 | 0.9624 | 0.0204 | 0.0056 | 0.893 |
| aoa | RF | Cl | 0.9536 | 0.183 | 0.164 | 0.9783 | 0.8406 | 0.3549 | — |
| aoa | RF | Cd | 0.6270 | 0.00483 | 0.00231 | 0.9105 | 0.0971 | 0.0491 | — |
| aoa | MLP | Cl | 0.9985 | 0.0328 | 0.0255 | 0.9970 | 0.1342 | 0.0375 | — |
| aoa | MLP | Cd | 0.8005 | 0.00353 | 0.00104 | 0.9473 | 0.0475 | 0.0314 | — |
| aoa | KRG | Cl | 0.9994 | 0.0205 | 0.0152 | 0.9984 | 0.0806 | 0.0220 | — |
| aoa | KRG | Cd | 0.8297 | 0.00326 | 0.000585 | 0.9625 | 0.0204 | 0.0065 | — |

## Reporting targets (task `full`, not gates)

Cl R² ≥ 0.99, Cd R² ≥ 0.95, Spearman ρ_D ≥ 0.9.

| family | Cl R² | Cd R² | ρ_D | all met |
|---|---|---|---|---|
| GP | 0.9993 | 0.9806 | 0.9799 | yes |
| RF | 0.9921 | 0.9492 | 0.9797 | no |
| MLP | 0.9984 | 0.9785 | 0.9931 | yes |
| KRG | 0.9993 | 0.9872 | 0.9852 | yes |

Mean relative error on C_L is dominated by test cases with C_L near zero (small denominators), which is why the median is also reported.

Figures: `results/figures/eval/parity_{task}_{Cl,Cd}.png`, `results/figures/eval/residuals_{task}.png`.

# Surface curves — Cp(x/c) and Cf(x/c)

Our curve models predict PCA mode weights of the 202-station wall curves from the same six features; the paper's models predict the whole flow field. The paper's surface metrics are `mean_rel_p` and `mean_rel_wss` (x, y components): the mean over wall nodes of `|(true − pred)/true|`, averaged over test samples (`metrics.py`, `Results_test`). We evaluate them on the same native wall nodes by interpolating our curves to each node's x/c (p = Cp·½U∞², τ = Cf·½U∞²·t). The ratio is huge wherever p or τ crosses zero, so our own curve metrics follow.

## Task `full` — paper surface metrics

| model | mean_rel_p | mean_rel_wss x | mean_rel_wss y |
|---|---|---|---|
| **ours — GP** | 2.018 | 0.462 | 0.429 |
| **ours — RF** | 1.712 | 0.645 | 0.638 |
| **ours — MLP** | 1.373 | 0.665 | 0.596 |
| **ours — KRG** | 2.144 | 0.500 | 0.552 |
| paper — MLP | 13.938 [52.614] | 119.341 [104.514] | 188.741 [155.168] |
| paper — GraphSAGE | 18.232 [37.579] | 105.004 [93.322] | 135.021 [110.918] |
| paper — PointNet | 21.009 [31.724] | 219.164 [184.221] | 338.304 [235.412] |
| paper — Graph U-Net | 8.193 [23.928] | 120.034 [108.974] | 225.382 [148.978] |

## Task `scarce` — paper surface metrics

| model | mean_rel_p | mean_rel_wss x | mean_rel_wss y |
|---|---|---|---|
| **ours — GP** | 8.745 | 0.860 | 0.736 |
| **ours — RF** | 10.951 | 0.764 | 0.878 |
| **ours — MLP** | 2.833 | 1.548 | 1.685 |
| **ours — KRG** | 0.725 | 1.221 | 0.912 |
| paper — MLP | 17.001 [34.758] | 108.816 [94.760] | 121.903 [126.195] |
| paper — GraphSAGE | 12.214 [17.612] | 98.646 [86.213] | 125.759 [107.068] |
| paper — PointNet | 15.263 [38.999] | 240.971 [150.997] | 289.589 [190.026] |
| paper — Graph U-Net | 14.508 [28.746] | 99.444 [100.546] | 162.049 [139.444] |

## Task `reynolds` — paper surface metrics

| model | mean_rel_p | mean_rel_wss x | mean_rel_wss y |
|---|---|---|---|
| **ours — GP** | 2.306 | 0.532 | 1.393 |
| **ours — RF** | 4.089 | 0.715 | 0.940 |
| **ours — MLP** | 6.528 | 1.212 | 2.242 |
| **ours — KRG** | 1.820 | 0.574 | 1.240 |
| paper — MLP | 27.633 [50.714] | 160.859 [232.215] | 547.184 [582.922] |
| paper — GraphSAGE | 13.184 [22.018] | 205.847 [146.271] | 375.016 [376.351] |
| paper — PointNet | 18.862 [17.390] | 321.246 [246.873] | 540.569 [623.836] |
| paper — Graph U-Net | 12.073 [22.063] | 175.106 [142.713] | 339.716 [316.586] |

## Task `aoa` — paper surface metrics

| model | mean_rel_p | mean_rel_wss x | mean_rel_wss y |
|---|---|---|---|
| **ours — GP** | 0.741 | 0.608 | 0.844 |
| **ours — RF** | 2.242 | 1.111 | 1.276 |
| **ours — MLP** | 0.953 | 0.573 | 0.656 |
| **ours — KRG** | 0.528 | 0.685 | 0.900 |
| paper — MLP | 2.295 [5.511] | 107.351 [132.478] | 375.807 [289.524] |
| paper — GraphSAGE | 1.762 [4.212] | 68.246 [72.594] | 199.396 [168.911] |
| paper — PointNet | 2.141 [3.606] | 198.778 [130.745] | 552.350 [355.821] |
| paper — Graph U-Net | 2.248 [4.097] | 85.611 [105.932] | 201.092 [255.670] |

## Our curve metrics

| task | family | quantity | modes | RMSE | R² (station mean) | ±2σ coverage | suction-peak MAE | separation detected | separation x/c MAE | C_L from curves (median rel.) | C_D from curves (median rel.) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| full | GP | Cp | 20 | 0.0625 | 0.9894 | 0.960 | 0.103 | — | — | 0.0090 | 0.0784 |
| full | GP | Cf | 20 | 0.000848 | 0.9655 | 0.959 | — | 0.970 | 0.015 | — | — |
| full | RF | Cp | 20 | 0.191 | 0.8948 | — | 0.485 | — | — | 0.0711 | 0.5011 |
| full | RF | Cf | 20 | 0.00153 | 0.8288 | — | — | 0.930 | 0.032 | — | — |
| full | MLP | Cp | 20 | 0.117 | 0.9825 | — | 0.238 | — | — | 0.0482 | 0.5140 |
| full | MLP | Cf | 20 | 0.00108 | 0.9613 | — | — | 0.980 | 0.019 | — | — |
| full | KRG | Cp | 20 | 0.0408 | 0.9885 | 0.877 | 0.074 | — | — | 0.0154 | 0.0635 |
| full | KRG | Cf | 20 | 0.000871 | 0.9628 | 0.868 | — | 0.960 | 0.014 | — | — |
| scarce | GP | Cp | 20 | 0.11 | 0.9584 | 0.934 | 0.151 | — | — | 0.0305 | 0.1497 |
| scarce | GP | Cf | 20 | 0.00171 | 0.9096 | 0.963 | — | 0.965 | 0.032 | — | — |
| scarce | RF | Cp | 20 | 0.211 | 0.7930 | — | 0.526 | — | — | 0.1481 | 0.7352 |
| scarce | RF | Cf | 20 | 0.00211 | 0.7385 | — | — | 0.905 | 0.047 | — | — |
| scarce | MLP | Cp | 20 | 0.251 | 0.9436 | — | 0.598 | — | — | 0.0965 | 1.2427 |
| scarce | MLP | Cf | 20 | 0.00209 | 0.9094 | — | — | 0.915 | 0.031 | — | — |
| scarce | KRG | Cp | 20 | 0.112 | 0.9581 | 0.901 | 0.141 | — | — | 0.0299 | 0.1491 |
| scarce | KRG | Cf | 20 | 0.00192 | 0.8733 | 0.923 | — | 0.945 | 0.044 | — | — |
| reynolds | GP | Cp | 20 | 0.133 | 0.9574 | 0.941 | 0.144 | — | — | 0.0302 | 0.1436 |
| reynolds | GP | Cf | 20 | 0.000946 | 0.8706 | 0.954 | — | 0.950 | 0.023 | — | — |
| reynolds | RF | Cp | 20 | 0.174 | 0.8865 | — | 0.461 | — | — | 0.0823 | 0.6115 |
| reynolds | RF | Cf | 20 | 0.00166 | 0.7734 | — | — | 0.919 | 0.034 | — | — |
| reynolds | MLP | Cp | 20 | 0.204 | 0.9420 | — | 0.444 | — | — | 0.0902 | 1.1079 |
| reynolds | MLP | Cf | 20 | 0.00182 | 0.9169 | — | — | 0.931 | 0.023 | — | — |
| reynolds | KRG | Cp | 20 | 0.117 | 0.9700 | 0.909 | 0.148 | — | — | 0.0237 | 0.1827 |
| reynolds | KRG | Cf | 20 | 0.000971 | 0.8664 | 0.891 | — | 0.940 | 0.022 | — | — |
| aoa | GP | Cp | 20 | 0.126 | 0.9684 | 0.909 | 0.220 | — | — | 0.0314 | 0.1409 |
| aoa | GP | Cf | 20 | 0.00127 | 0.8545 | 0.955 | — | 0.918 | 0.039 | — | — |
| aoa | RF | Cp | 20 | 0.394 | 0.8400 | — | 1.281 | — | — | 0.5182 | 1.2153 |
| aoa | RF | Cf | 20 | 0.00261 | 0.7118 | — | — | 0.842 | 0.085 | — | — |
| aoa | MLP | Cp | 20 | 0.175 | 0.9688 | — | 0.448 | — | — | 0.0735 | 0.8742 |
| aoa | MLP | Cf | 20 | 0.00175 | 0.8841 | — | — | 0.903 | 0.028 | — | — |
| aoa | KRG | Cp | 20 | 0.118 | 0.9745 | 0.866 | 0.186 | — | — | 0.0166 | 0.1209 |
| aoa | KRG | Cf | 20 | 0.00132 | 0.8485 | 0.924 | — | 0.918 | 0.037 | — | — |

Cf values are raw coefficients (≈10⁻³). The separation-detected column is the share of test cases where the prediction agrees with CFD on whether the upper surface has Cf < 0 between x/c = 0.02 and 0.98; the x/c error is over cases where both separate.

Figures: `results/figures/curves/examples_{task}.png`, `results/figures/curves/naca0012_reference.png`.
