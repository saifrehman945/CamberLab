"""
Module: scripts/surrogate/data.py
Purpose: The AirfRANS tabular dataset, its QA flags and the frozen task
         splits — shared by 09_train_surrogates.py, 10_global_validation.py
         and the inference API so they can never disagree about what
         "features", "train" and "test" mean.
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
SPLITS_DIR = PROJECT_ROOT / "splits"
DATASET_PATH = RESULTS_DIR / "airfrans_dataset.csv"
QA_FLAGS_PATH = RESULTS_DIR / "airfrans_qa_flags.csv"

FEATURES = ["alpha_deg", "log10_Re", "t_max", "x_tmax", "m_max", "x_m"]
TARGETS = ["Cl", "Cd"]
TASKS = ["full", "scarce", "reynolds", "aoa"]


def load_dataset(path: Path = DATASET_PATH, flags_path: Path = QA_FLAGS_PATH) -> pd.DataFrame:
    """The ingested dataset joined with its QA flags. Rows are never modified."""
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — run scripts/20_ingest_airfrans.py first.")
    df = pd.read_csv(path, keep_default_na=False, na_values=[""], dtype={"source_name": str})
    if flags_path.exists():
        df = df.merge(pd.read_csv(flags_path, keep_default_na=False), on="sample_id", how="left",
                      validate="one_to_one")
    else:
        log.warning("%s missing — run scripts/21_qa_airfrans.py; treating all rows as qa_pass",
                    flags_path.name)
        df["qa_pass"], df["qa_reason"] = True, ""
    return df


def load_split(task: str) -> tuple[np.ndarray, np.ndarray]:
    if task not in TASKS:
        raise ValueError(f"unknown task {task!r}; expected one of {TASKS}")
    tr = np.load(SPLITS_DIR / f"airfrans_{task}_train_idx.npy")
    te = np.load(SPLITS_DIR / f"airfrans_{task}_test_idx.npy")
    return tr, te


def task_frames(task: str, df: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(train, test) rows of a task. Indices address rows (== sample_id)."""
    df = load_dataset() if df is None else df
    assert (df.sample_id.to_numpy() == np.arange(len(df))).all()
    tr, te = load_split(task)
    train, test = df.iloc[tr].reset_index(drop=True), df.iloc[te].reset_index(drop=True)
    if not (train.qa_pass.all() and test.qa_pass.all()):
        raise RuntimeError(f"task {task}: split contains rows with qa_pass=False")
    return train, test


def get_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Features, Cl, and log(Cd) — Cd is modelled in log space and back-transformed with exp."""
    return (df[FEATURES].astype(np.float64), df["Cl"].to_numpy(np.float64),
            np.log(df["Cd"].to_numpy(np.float64)))


def envelope(train: pd.DataFrame) -> dict:
    """Per-feature min/max over the training rows, plus the raw Re range."""
    env = {f: [float(train[f].min()), float(train[f].max())] for f in FEATURES}
    env["Re"] = [float(train["Re"].min()), float(train["Re"].max())]
    return env
