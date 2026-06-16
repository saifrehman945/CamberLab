#!/usr/bin/env python3
"""
Script: 01_doe.py
Stage:  1 — Design of Experiments
Purpose: Generate 175 LHS samples — per-regime allocation (A=80, B=25, C=30,
         D=40), each drawn inside its own bounding box — then concatenate.
         Freeze a stratified 80/20 train/test split.

Output columns:
    case_id, alpha_deg, Re, thickness, regime

Per-regime bounding boxes come from README.md §"Flow regimes" (also CLAUDE.md
§10). Each sample's `regime` label is set by construction (the box it was
drawn from) — not by `classify_regime()`, which is the inference-time
classifier used when querying the surrogate at arbitrary points.

Usage:
    micromamba run -n openfoam python scripts/01_doe.py
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from pyDOE2 import lhs
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SEED = 42
TEST_FRACTION = 0.2

REGIME_SPEC: dict[str, dict] = {
    "A": {"n": 80, "alpha": (0.0,  8.0),  "Re": (1.5e6, 3.0e6), "thickness": (0.10, 0.18)},
    "B": {"n": 25, "alpha": (10.0, 16.0), "Re": (1.0e6, 3.0e6), "thickness": (0.12, 0.24)},
    "C": {"n": 30, "alpha": (0.0,  8.0),  "Re": (3.0e5, 1.0e6), "thickness": (0.08, 0.15)},
    "D": {"n": 40, "alpha": (0.0,  6.0),  "Re": (2.0e6, 5.0e6), "thickness": (0.10, 0.18)},
}


def scale(unit_column: np.ndarray, low: float, high: float) -> np.ndarray:
    return low + unit_column * (high - low)


def lhs_for_regime(regime: str, spec: dict, seed: int) -> pd.DataFrame:
    unit = lhs(3, samples=spec["n"], criterion="maximin", random_state=seed)
    return pd.DataFrame({
        "alpha_deg": scale(unit[:, 0], *spec["alpha"]).astype(np.float64),
        "Re":        scale(unit[:, 1], *spec["Re"]).astype(np.float64),
        "thickness": scale(unit[:, 2], *spec["thickness"]).astype(np.float64),
        "regime":    regime,
    })


def main() -> None:
    log.info("Generating per-regime LHS (seed=%d)...", SEED)

    # Distinct sub-seed per regime so each regime's LHS is independently
    # maximin-optimal inside its own box, yet the whole DOE remains a
    # deterministic function of SEED.
    frames: list[pd.DataFrame] = []
    for offset, (regime, spec) in enumerate(REGIME_SPEC.items()):
        frames.append(lhs_for_regime(regime, spec, seed=SEED + offset))
        log.info(
            "  regime %s: N=%2d  alpha=[%.1f, %.1f]  Re=[%.2e, %.2e]  t=[%.2f, %.2f]",
            regime, spec["n"], *spec["alpha"], *spec["Re"], *spec["thickness"],
        )

    df = pd.concat(frames, ignore_index=True)
    df.insert(0, "case_id", np.arange(len(df), dtype=np.int32))

    for regime, spec in REGIME_SPEC.items():
        sub = df[df["regime"] == regime]
        assert sub["alpha_deg"].between(*spec["alpha"]).all()
        assert sub["Re"].between(*spec["Re"]).all()
        assert sub["thickness"].between(*spec["thickness"]).all()

    samples_path = PROJECT_ROOT / "samples.csv"
    df.to_csv(samples_path, index=False)
    log.info("Wrote %s (%d rows)", samples_path.relative_to(PROJECT_ROOT), len(df))

    train_idx, test_idx = train_test_split(
        df["case_id"].to_numpy(),
        test_size=TEST_FRACTION,
        random_state=SEED,
        stratify=df["regime"].to_numpy(),
    )
    train_idx = np.sort(train_idx.astype(np.int32))
    test_idx  = np.sort(test_idx.astype(np.int32))

    np.save(PROJECT_ROOT / "train_idx.npy", train_idx)
    np.save(PROJECT_ROOT / "test_idx.npy",  test_idx)
    log.info(
        "Wrote train_idx.npy (%d) / test_idx.npy (%d)",
        len(train_idx), len(test_idx),
    )

    split_label = np.where(np.isin(df["case_id"].to_numpy(), test_idx), "test", "train")
    summary = (
        df.assign(split=split_label)
          .groupby(["regime", "split"]).size().unstack(fill_value=0)
    )
    log.info("Per-regime train/test allocation:\n%s", summary.to_string())

    log.info(
        "Global span — alpha=[%.2f, %.2f]  Re=[%.2e, %.2e]  t=[%.3f, %.3f]",
        df["alpha_deg"].min(), df["alpha_deg"].max(),
        df["Re"].min(), df["Re"].max(),
        df["thickness"].min(), df["thickness"].max(),
    )


if __name__ == "__main__":
    main()
