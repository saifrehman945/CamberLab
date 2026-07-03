"""
Module: scripts/surrogate/regime_bounds.py
Purpose: The ACHIEVED (not nominal) per-regime input envelope, computed from
         the rows a surrogate was actually trained on. This is deliberately
         narrower than the nominal bounding boxes in CLAUDE.md §10, since
         only a subset of any regime's nominal DOE box may have converged
         CFD results at any given time.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from scripts.surrogate.data import MODELS_DIR

log = logging.getLogger(__name__)

BOUNDS_PATH = MODELS_DIR / "regime_bounds.json"
_DIMS = ["alpha_deg", "Re", "thickness"]


def compute_regime_bounds(train_df: pd.DataFrame) -> dict:
    """Per-regime {dim: [lo, hi]} + n_train, for regimes present in train_df.

    A regime absent from train_df is simply absent from the returned dict —
    absence means "not trained," which scripts.surrogate.inference treats as
    an out-of-distribution rejection rather than falling back to some other
    regime's envelope.
    """
    bounds: dict = {}
    for regime, group in train_df.groupby("regime"):
        entry = {dim: [float(group[dim].min()), float(group[dim].max())] for dim in _DIMS}
        entry["n_train"] = int(len(group))
        bounds[regime] = entry
    return bounds


def save_regime_bounds(bounds: dict, path: Path = BOUNDS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(bounds, indent=2) + "\n", encoding="utf-8")
    log.info("Wrote %s (trained regimes: %s)", path, sorted(bounds))


def load_regime_bounds(path: Path = BOUNDS_PATH) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — run scripts/09_train_surrogates.py first.")
    return json.loads(path.read_text(encoding="utf-8"))
