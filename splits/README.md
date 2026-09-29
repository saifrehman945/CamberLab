# splits/

Frozen train/test index files for the four AirfRANS tasks. Each `.npy` holds row indices into `results/airfrans_dataset.csv` (identical to `sample_id`).

## Provenance

Membership is the **official AirfRANS split**, as published in the `split` block of the `PLAID-datasets/AirfRANS_clipped` dataset card (`data/airfrans_clipped/README.md`; the `AirfRANS_remeshed` card lists identical splits), whose indices address the rows of the `all_samples` split. The comparison with the AirfRANS paper's test sets is therefore exact, apart from rows excluded by QA (`results/airfrans_qa_flags.csv`, `qa_pass = False`), which are removed before writing.

| task | card train split | card test split |
|---|---|---|
| `full` | `full_train` | `full_test` |
| `scarce` | `scarce_train` | `full_test` |
| `reynolds` | `reynolds_train` | `reynolds_test` |
| `aoa` | `aoa_train` | `aoa_test` |

## Contents

| task | train | test | QA-dropped (train/test) | α train | α test | Re train (×10⁶) | Re test (×10⁶) |
|---|---|---|---|---|---|---|---|
| `full` | 800 | 200 | 0/0 | [-4.93, 14.93] | [-4.94, 14.69] | [2.02, 6.04] | [2.05, 6.01] |
| `scarce` | 200 | 200 | 0/0 | [-4.79, 14.84] | [-4.94, 14.69] | [2.03, 6.00] | [2.05, 6.01] |
| `reynolds` | 504 | 496 | 0/0 | [-4.93, 14.79] | [-4.94, 14.93] | [3.02, 5.03] | [2.02, 6.04] |
| `aoa` | 804 | 196 | 0/0 | [-2.48, 12.45] | [-4.94, 14.93] | [2.02, 6.04] | [2.02, 6.01] |

Checks: no train/test overlap in any task; `scarce` train ⊂ `full` train; `scarce` shares the `full` test set.

Value rules of the paper's extrapolation tasks, verified on this membership:

- [x] reynolds: train/test sharply separated in U∞ (no test U∞ inside the train band)
- [x] reynolds: band edges at nominal Re 3e6 / 5e6 for one ν (implied ν = 1.5614e-05 / 1.5590e-05)
- [x] aoa train α in [-2.5°, 12.5°]
- [x] aoa test α outside (-2.5°, 12.5°)

Re in this project is U∞·c/ν with ν(298.15 K) = 1.5498e-05 m²/s (AirfRANS' polynomial). The `reynolds` train band is U∞ ∈ [46.842, 77.949] m/s = Re [3.022, 5.030]×10⁶ in these units; its edges correspond to the paper's nominal 3×10⁶ / 5×10⁶ with ν ≈ 1.56×10⁻⁵ m²/s, i.e. AirfRANS' nominal Re values are ~0.6% below this project's. The offset is a constant factor, so it does not affect membership or (standardised) model inputs.

## Rules

- Test indices are never used for fitting, model selection or tuning; hyperparameters are chosen by 5-fold CV on the train indices only.
- Splits are frozen: `scripts/22_make_splits.py` refuses to overwrite an index file with different contents.
