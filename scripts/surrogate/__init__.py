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
from .inference import (
    OutOfDistributionError,
    blend_weights,
    predict,
    predict_all,
    predict_all_blended,
    predict_blended,
    predict_blended_with_uncertainty,
    predict_untrained_regime_extrapolated,
    predict_with_uncertainty,
    resolve_weights,
    validate_and_classify,
)
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
    "predict_blended",
    "predict_all_blended",
    "predict_with_uncertainty",
    "predict_blended_with_uncertainty",
    "predict_untrained_regime_extrapolated",
    "blend_weights",
    "resolve_weights",
    "validate_and_classify",
]
