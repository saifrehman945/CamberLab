"""
Module: scripts/surrogate/inference.py
Purpose: OOD/validity gating and prediction against the persisted surrogate
         models. A query point is rejected — never silently extrapolated —
         if it names or classifies into a regime with no trained model, or
         falls outside that regime's ACHIEVED training envelope.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from scripts.mesh.regime_parameters import classify_regime
from scripts.surrogate.data import MODELS_DIR, REGIMES, build_feature_frame
from scripts.surrogate.regime_bounds import load_regime_bounds

log = logging.getLogger(__name__)

FAMILIES = ["gp", "rf", "mlp", "krg"]
TARGETS = ["Cl", "Cd"]


class OutOfDistributionError(ValueError):
    """Raised when a query point cannot be trusted: untrained regime, or
    outside that regime's actual trained envelope."""


def validate_and_classify(
    alpha_deg: float,
    Re: float,
    thickness: float,
    regime: str | None = None,
    bounds: dict | None = None,
) -> str:
    """Resolve the regime for a query point and enforce the validity gate.

    Returns the resolved regime label on success; raises OutOfDistribution
    Error otherwise. `regime`, if given, is an explicit caller assertion
    (still validated against known labels and against trained bounds) rather
    than a shortcut around classify_regime().
    """
    bounds = bounds if bounds is not None else load_regime_bounds()

    if regime is not None:
        if regime not in REGIMES:
            raise OutOfDistributionError(f"Unknown regime label {regime!r}; must be one of {REGIMES}")
        resolved = regime
    else:
        resolved = classify_regime(alpha_deg, Re, thickness)

    if resolved not in bounds:
        raise OutOfDistributionError(
            f"Point classifies as regime {resolved}, but no trained model exists for "
            f"that regime yet (trained regimes: {sorted(bounds)}). Re-run "
            f"07_harvest_results.py + 09_train_surrogates.py once regime {resolved} "
            f"CFD data is available."
        )

    box = bounds[resolved]
    for name, val in (("alpha_deg", alpha_deg), ("Re", Re), ("thickness", thickness)):
        lo, hi = box[name]
        if not (lo <= val <= hi):
            raise OutOfDistributionError(
                f"{name}={val} outside regime {resolved}'s TRAINED envelope "
                f"[{lo}, {hi}] (n_train={box['n_train']}). Not extrapolating."
            )
    return resolved


@lru_cache(maxsize=None)
def _load_joblib(path_str: str):
    return joblib.load(path_str)


def _predict_one(model, family: str, X_proc: np.ndarray) -> float:
    if family == "krg":
        return float(model.predict_values(np.asarray(X_proc))[0, 0])
    return float(model.predict(X_proc)[0])


def _build_query(alpha_deg: float, Re: float, thickness: float, resolved: str) -> pd.DataFrame:
    return build_feature_frame(
        pd.DataFrame([{"alpha_deg": alpha_deg, "Re": Re, "thickness": thickness, "regime": resolved}])
    )


def predict(
    alpha_deg: float,
    Re: float,
    thickness: float,
    family: str = "gp",
    target: str = "Cl",
    regime: str | None = None,
    models_dir: Path = MODELS_DIR,
) -> float:
    if family not in FAMILIES:
        raise ValueError(f"Unknown family {family!r}; must be one of {FAMILIES}")
    if target not in TARGETS:
        raise ValueError(f"Unknown target {target!r}; must be one of {TARGETS}")

    bounds = load_regime_bounds(models_dir / "regime_bounds.json")
    resolved = validate_and_classify(alpha_deg, Re, thickness, regime, bounds)

    preproc = _load_joblib(str(models_dir / "preprocessor.joblib"))
    model = _load_joblib(str(models_dir / f"{family}_{target}.joblib"))
    X_proc = preproc.transform(_build_query(alpha_deg, Re, thickness, resolved))
    return _predict_one(model, family, X_proc)


def predict_all(
    alpha_deg: float,
    Re: float,
    thickness: float,
    regime: str | None = None,
    models_dir: Path = MODELS_DIR,
) -> dict:
    """Single validation call, all 4 families x 2 targets."""
    bounds = load_regime_bounds(models_dir / "regime_bounds.json")
    resolved = validate_and_classify(alpha_deg, Re, thickness, regime, bounds)

    preproc = _load_joblib(str(models_dir / "preprocessor.joblib"))
    X_proc = preproc.transform(_build_query(alpha_deg, Re, thickness, resolved))

    results: dict[str, dict[str, float]] = {family: {} for family in FAMILIES}
    for family in FAMILIES:
        for target in TARGETS:
            model = _load_joblib(str(models_dir / f"{family}_{target}.joblib"))
            results[family][target] = _predict_one(model, family, X_proc)
    return results
