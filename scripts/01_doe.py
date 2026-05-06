#!/usr/bin/env python3
"""
Script: 01_doe.py
Stage:  1 — Design of Experiments
Purpose: Generate a 100-point Latin Hypercube Sample over (alpha, Re, thickness)
         and freeze an 80/20 train-test split.

Usage:
    micromamba run -n openfoam python scripts/01_doe.py
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from pyDOE2 import lhs

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

N_SAMPLES = 100
N_TEST = 20
SEED = 42

ALPHA_RANGE = (0.0, 16.0)
RE_RANGE = (5.0e5, 3.0e6)
THICKNESS_RANGE = (0.08, 0.24)


def scale(unit_column: np.ndarray, low: float, high: float) -> np.ndarray:
    return low + unit_column * (high - low)


def main() -> None:
    log.info("Generating LHS sample (N=%d, seed=%d)...", N_SAMPLES, SEED)
    unit = lhs(3, samples=N_SAMPLES, criterion="maximin", random_state=SEED)

    alpha = scale(unit[:, 0], *ALPHA_RANGE)
    reynolds = scale(unit[:, 1], *RE_RANGE)
    thickness = scale(unit[:, 2], *THICKNESS_RANGE)

    df = pd.DataFrame(
        {
            "alpha_deg": alpha.astype(np.float64),
            "Re": reynolds.astype(np.float64),
            "thickness": thickness.astype(np.float64),
        }
    )

    assert df["alpha_deg"].between(*ALPHA_RANGE).all()
    assert df["Re"].between(*RE_RANGE).all()
    assert df["thickness"].between(*THICKNESS_RANGE).all()

    samples_path = PROJECT_ROOT / "samples.csv"
    df.to_csv(samples_path, index=False)
    log.info("Wrote %s (%d rows)", samples_path.relative_to(PROJECT_ROOT), len(df))

    rng = np.random.default_rng(SEED)
    indices = np.arange(N_SAMPLES, dtype=np.int32)
    rng.shuffle(indices)
    test_idx = np.sort(indices[:N_TEST])
    train_idx = np.sort(indices[N_TEST:])

    train_path = PROJECT_ROOT / "train_idx.npy"
    test_path = PROJECT_ROOT / "test_idx.npy"
    np.save(train_path, train_idx)
    np.save(test_path, test_idx)
    log.info("Wrote %s (%d indices)", train_path.relative_to(PROJECT_ROOT), len(train_idx))
    log.info("Wrote %s (%d indices)", test_path.relative_to(PROJECT_ROOT), len(test_idx))

    log.info(
        "alpha_deg : min=%.3f  max=%.3f  mean=%.3f",
        df["alpha_deg"].min(), df["alpha_deg"].max(), df["alpha_deg"].mean(),
    )
    log.info(
        "Re        : min=%.3e  max=%.3e  mean=%.3e",
        df["Re"].min(), df["Re"].max(), df["Re"].mean(),
    )
    log.info(
        "thickness : min=%.3f  max=%.3f  mean=%.3f",
        df["thickness"].min(), df["thickness"].max(), df["thickness"].mean(),
    )


if __name__ == "__main__":
    main()
