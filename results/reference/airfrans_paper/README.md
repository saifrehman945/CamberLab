# AirfRANS paper reference scores (cached verbatim)

Source repository: https://github.com/Extrality/AirfRANS at commit
`6acde648d5a9d81a3a90366abb6c04b7b02fe2a8` (fetched 2026-09-27), files copied
byte-for-byte:

- `metrics.py` — the authors' evaluation code
- `scores/{full,scarce,reynolds,aoa}/score_MSE.json` — models trained with plain MSE
- `scores/{full,scarce,reynolds,aoa}/score_WMSE.json` — models trained with the
  intended surface + volume loss
- `scores/readme`

Paper: Bonnet et al., "AirfRANS: High Fidelity Computational Fluid Dynamics
Dataset for Approximating Reynolds-Averaged Navier-Stokes Solutions",
NeurIPS 2022 Datasets & Benchmarks, arXiv:2212.07564.

## Which numbers are "corrected"

arXiv v3 (1 June 2023) carries a disclaimer dated 26 May 2023: due to an
implementation error, the main-text results were produced by models trained
with a classical MSE loss; results with the loss described in the paper are
in Appendix N. We verified that

- `score_MSE.json` = main-text Table 3 / Table 5 (original numbers), and
- `score_WMSE.json` = Appendix N Table 27 (corrected numbers), value for value.

## Field layout and units (checked in `metrics.py`)

- Model order in every list: MLP, GraphSAGE, PointNet, Graph U-Net.
- Each entry is `[drag, lift]`.
- `mean_score_force`: `rel_err(true, pred) = |(true - pred) / true|`
  (`metrics.py` line 44) averaged over the test simulations, then averaged
  over 5 independently trained copies of each model. It is a **raw ratio,
  not a percentage**: drag 4.29 means a mean relative error of 429%.
- `spearman_coef_mean`: Spearman's ρ between true and predicted coefficients
  over the test set, averaged over the 5 copies.
