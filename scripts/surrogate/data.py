"""
Module: scripts/surrogate/data.py
Purpose: Dataset loading, one-hot feature encoding, and the sacred train/test
         split — shared by 09_train_surrogates.py and 10_global_validation.py
         so both scripts can never disagree about what "train" and "test" mean.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models"

REGIMES = ["A", "B", "C", "D"]
DATASET_PATH = RESULTS_DIR / "dataset_clean.csv"
SMALL_TEST_SLICE_WARN = 10


def load_dataset(path: Path = DATASET_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run scripts/07_harvest_results.py first "
            "to build the dataset from converged CFD cases."
        )
    return pd.read_csv(path)


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode `regime` against the full REGIMES list.

    Reindexing against REGIMES (rather than only the categories observed in
    `df`) guarantees the feature matrix always has columns
    [alpha_deg, Re, thickness, regime_A, regime_B, regime_C, regime_D],
    regardless of which regimes are actually present. This is what lets
    Regime B/C/D data land later without any column-shape change to the
    ColumnTransformer or persisted preprocessor.
    """
    onehot = (
        pd.get_dummies(df["regime"], prefix="regime")
        .reindex(columns=[f"regime_{r}" for r in REGIMES], fill_value=0)
        .astype(np.int64)
    )
    return pd.concat(
        [
            df[["alpha_deg", "Re", "thickness"]].reset_index(drop=True),
            onehot.reset_index(drop=True),
        ],
        axis=1,
    )


def load_split_indices() -> tuple[np.ndarray, np.ndarray]:
    train_idx = np.load(PROJECT_ROOT / "train_idx.npy")
    test_idx = np.load(PROJECT_ROOT / "test_idx.npy")
    return train_idx, test_idx


def _case_num(case_id: pd.Series) -> pd.Series:
    return case_id.str.removeprefix("case_").astype(int)


def split_dataset(
    df: pd.DataFrame, train_idx: np.ndarray, test_idx: np.ndarray
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Filter dataset_clean.csv rows against train_idx.npy/test_idx.npy.

    The .npy files index into samples.csv's full 175-row case_id space;
    dataset_clean.csv only has rows for cases that have actually converged,
    so coverage is partial today and is logged rather than asserted.
    """
    case_num = _case_num(df["case_id"])
    train_df = df[case_num.isin(train_idx)].reset_index(drop=True)
    test_df = df[case_num.isin(test_idx)].reset_index(drop=True)

    log.info(
        "train_idx coverage: %d/%d case_ids present in dataset_clean.csv (%d not yet converged)",
        len(train_df), len(train_idx), len(train_idx) - len(train_df),
    )
    log.info(
        "test_idx coverage:  %d/%d case_ids present in dataset_clean.csv (%d not yet converged)",
        len(test_df), len(test_idx), len(test_idx) - len(test_df),
    )

    for regime in REGIMES:
        n_train = int((train_df["regime"] == regime).sum())
        if n_train == 0:
            log.warning(
                "regime %s: 0 training examples in current dataset — its one-hot "
                "column will be constant/all-zero until CFD data for that regime lands",
                regime,
            )
        n_test = int((test_df["regime"] == regime).sum())
        if 0 < n_test < SMALL_TEST_SLICE_WARN:
            log.warning(
                "regime %s: only %d test examples — per-regime metrics will be "
                "informational only, not statistically reliable",
                regime, n_test,
            )
    return train_df, test_df


def get_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    return build_feature_frame(df), df["Cl_mean"], df["Cd_mean"]
