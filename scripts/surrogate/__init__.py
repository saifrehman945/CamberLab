"""
scripts.surrogate — dataset assembly, and OOD-gated inference for the
regime-aware aerofoil surrogate.

Modules
-------
data           : dataset loading, one-hot feature encoding, train/test split
regime_bounds  : achieved per-regime input envelope (from trained data)
inference      : OOD/validity gating + prediction against persisted models
"""

from .data import (
    REGIMES,
    build_feature_frame,
    get_xy,
    load_dataset,
    load_split_indices,
    split_dataset,
)
from .inference import OutOfDistributionError, predict, predict_all, validate_and_classify
from .regime_bounds import compute_regime_bounds, load_regime_bounds, save_regime_bounds

__all__ = [
    "REGIMES",
    "build_feature_frame",
    "get_xy",
    "load_dataset",
    "load_split_indices",
    "split_dataset",
    "compute_regime_bounds",
    "load_regime_bounds",
    "save_regime_bounds",
    "OutOfDistributionError",
    "predict",
    "predict_all",
    "validate_and_classify",
]
