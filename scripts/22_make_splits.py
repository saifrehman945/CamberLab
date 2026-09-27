#!/usr/bin/env python3
"""
Script: 22_make_splits.py
Stage:  AirfRANS Phase 5 — frozen train/test splits
Purpose: Write splits/airfrans_{task}_{train,test}_idx.npy for the four
         AirfRANS tasks (full, scarce, reynolds, aoa) from the official
         split membership published in the dataset card, minus QA failures.

Indices are row positions in results/airfrans_dataset.csv (= sample_id).
Once written, splits are frozen: re-running refuses to overwrite files whose
contents would change.

Usage:
    uv run python scripts/22_make_splits.py [--data-dir DIR]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.airfrans import io  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(message)s")
log = logging.getLogger(__name__)

RESULTS_DIR = PROJECT_ROOT / "results"
SPLITS_DIR = PROJECT_ROOT / "splits"
TASKS = {  # task: (card train split, card test split)
    "full": ("full_train", "full_test"),
    "scarce": ("scarce_train", "full_test"),
    "reynolds": ("reynolds_train", "reynolds_test"),
    "aoa": ("aoa_train", "aoa_test"),
}


def write_frozen(path: Path, idx: np.ndarray) -> None:
    if path.exists():
        old = np.load(path)
        if not np.array_equal(old, idx):
            raise RuntimeError(f"{path} exists with different contents; splits are frozen")
        return
    np.save(path, idx)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default=None)
    args = ap.parse_args()

    data_dir = io.resolve_data_dir(args.data_dir)
    card = io.load_splits(data_dir)
    df = pd.read_csv(RESULTS_DIR / "airfrans_dataset.csv", usecols=["sample_id", "alpha_deg", "Re"])
    flags = pd.read_csv(RESULTS_DIR / "airfrans_qa_flags.csv")
    assert (df.sample_id.to_numpy() == np.arange(len(df))).all()
    ok = set(flags.sample_id[flags.qa_pass])

    SPLITS_DIR.mkdir(exist_ok=True)
    summary = []
    for task, (tr_name, te_name) in TASKS.items():
        tr_all, te_all = np.sort(card[tr_name]), np.sort(card[te_name])
        assert not set(tr_all) & set(te_all), f"{task}: train/test overlap in the card"
        tr = np.array([i for i in tr_all if i in ok], dtype=np.int32)
        te = np.array([i for i in te_all if i in ok], dtype=np.int32)
        write_frozen(SPLITS_DIR / f"airfrans_{task}_train_idx.npy", tr)
        write_frozen(SPLITS_DIR / f"airfrans_{task}_test_idx.npy", te)
        a_tr, a_te = df.alpha_deg.to_numpy()[tr], df.alpha_deg.to_numpy()[te]
        r_tr, r_te = df.Re.to_numpy()[tr], df.Re.to_numpy()[te]
        summary.append({"task": task, "train": len(tr), "test": len(te),
                        "dropped_qa_train": len(tr_all) - len(tr), "dropped_qa_test": len(te_all) - len(te),
                        "alpha_train": f"[{a_tr.min():.2f}, {a_tr.max():.2f}]",
                        "alpha_test": f"[{a_te.min():.2f}, {a_te.max():.2f}]",
                        "Re_train_e6": f"[{r_tr.min() / 1e6:.2f}, {r_tr.max() / 1e6:.2f}]",
                        "Re_test_e6": f"[{r_te.min() / 1e6:.2f}, {r_te.max() / 1e6:.2f}]"})
        log.info("%-8s train %d  test %d", task, len(tr), len(te))

    s = {r["task"]: r for r in summary}
    full_tr = set(np.load(SPLITS_DIR / "airfrans_full_train_idx.npy"))
    scarce_tr = set(np.load(SPLITS_DIR / "airfrans_scarce_train_idx.npy"))
    assert scarce_tr <= full_tr, "scarce train must be a subset of full train"
    # Value rules of the paper's tasks, checked against the official membership.
    al = df.alpha_deg.to_numpy()
    rtr = np.load(SPLITS_DIR / "airfrans_reynolds_train_idx.npy")
    rte = np.load(SPLITS_DIR / "airfrans_reynolds_test_idx.npy")
    atr = np.load(SPLITS_DIR / "airfrans_aoa_train_idx.npy")
    ate = np.load(SPLITS_DIR / "airfrans_aoa_test_idx.npy")
    u = pd.read_csv(RESULTS_DIR / "airfrans_dataset.csv", usecols=["U_inf"]).U_inf.to_numpy()
    u_lo, u_hi = u[rtr].min(), u[rtr].max()
    nu_implied = (u_lo / 3e6, u_hi / 5e6)
    rules = {
        # The paper's rule is Re in [3e6, 5e6]; AirfRANS' nominal Re uses ν ≈ 1.56e-5,
        # 0.6% above the ν(298.15 K) this project derives Re with, so the split is
        # checked as a sharp cut in U∞ with edges at nominal Re 3e6 / 5e6 (±1%).
        "reynolds: train/test sharply separated in U∞ (no test U∞ inside the train band)":
            bool(not ((u[rte] >= u_lo) & (u[rte] <= u_hi)).any()),
        f"reynolds: band edges at nominal Re 3e6 / 5e6 for one ν (implied ν = "
        f"{nu_implied[0]:.4e} / {nu_implied[1]:.4e})":
            bool(abs(nu_implied[0] / nu_implied[1] - 1) < 0.01),
        "aoa train α in [-2.5°, 12.5°]": bool(((al[atr] >= -2.5 - 1e-3) & (al[atr] <= 12.5 + 1e-3)).all()),
        "aoa test α outside (-2.5°, 12.5°)": bool(((al[ate] < -2.5 + 1e-3) | (al[ate] > 12.5 - 1e-3)).all()),
    }
    for k, v in rules.items():
        (log.info if v else log.warning)("rule check — %s: %s", k, v)

    md = ["# splits/", "",
          "Frozen train/test index files for the four AirfRANS tasks. Each `.npy` holds row indices "
          "into `results/airfrans_dataset.csv` (identical to `sample_id`).", "",
          "## Provenance", "",
          "Membership is the **official AirfRANS split**, as published in the `split` block of the "
          "`PLAID-datasets/AirfRANS_remeshed` dataset card (`data/airfrans_remeshed/README.md`), "
          "whose indices address the rows of the `all_samples` split. The comparison with the "
          "AirfRANS paper's test sets is therefore exact, apart from rows excluded by QA "
          "(`results/airfrans_qa_flags.csv`, `qa_pass = False`), which are removed before writing.", "",
          "| task | card train split | card test split |", "|---|---|---|"]
    md += [f"| `{t}` | `{a}` | `{b}` |" for t, (a, b) in TASKS.items()]
    md += ["", "## Contents", "",
           "| task | train | test | QA-dropped (train/test) | α train | α test | Re train (×10⁶) | Re test (×10⁶) |",
           "|---|---|---|---|---|---|---|---|"]
    for r in summary:
        md.append(f"| `{r['task']}` | {r['train']} | {r['test']} | {r['dropped_qa_train']}/{r['dropped_qa_test']} "
                  f"| {r['alpha_train']} | {r['alpha_test']} | {r['Re_train_e6']} | {r['Re_test_e6']} |")
    md += ["", "Checks: no train/test overlap in any task; `scarce` train ⊂ `full` train; "
           "`scarce` shares the `full` test set.", "",
           "Value rules of the paper's extrapolation tasks, verified on this membership:", ""]
    md += [f"- [{'x' if v else ' '}] {k}" for k, v in rules.items()]
    md += ["", f"Re in this project is U∞·c/ν with ν(298.15 K) = {io.NU:.4e} m²/s (AirfRANS' polynomial). "
           f"The `reynolds` train band is U∞ ∈ [{u_lo:.3f}, {u_hi:.3f}] m/s = Re "
           f"[{u_lo / io.NU / 1e6:.3f}, {u_hi / io.NU / 1e6:.3f}]×10⁶ in these units; its edges "
           "correspond to the paper's nominal 3×10⁶ / 5×10⁶ with ν ≈ 1.56×10⁻⁵ m²/s, i.e. AirfRANS' "
           "nominal Re values are ~0.6% below this project's. The offset is a constant factor, so it "
           "does not affect membership or (standardised) model inputs."]
    md += ["", "## Rules", "",
           "- Test indices are never used for fitting, model selection or tuning; hyperparameters "
           "are chosen by 5-fold CV on the train indices only.",
           "- Splits are frozen: `scripts/22_make_splits.py` refuses to overwrite an index file with "
           "different contents.", ""]
    (SPLITS_DIR / "README.md").write_text("\n".join(md))
    log.info("Wrote splits/ (%s)", ", ".join(f"{t}: {s[t]['train']}/{s[t]['test']}" for t in s))
    return 0


if __name__ == "__main__":
    sys.exit(main())
