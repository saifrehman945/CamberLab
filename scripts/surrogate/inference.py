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

# --- Inference-time regime blending (partition of unity) ---------------------
# Adjacent regimes trained on different CFD templates (e.g. A at y+~30 and D at
# y+~50) disagree by a few percent where their achieved envelopes overlap. A
# hard classifier (CLAUDE.md §11) flips between them at a sharp boundary, which
# shows up as a step in a Cl-alpha / Cd-alpha curve that crosses the boundary.
#
# Instead of picking one regime per point, we blend the per-regime expert models
# with weights that vary smoothly across the overlap: each regime's weight
# ramps to zero (with zero slope) as the query approaches that regime's own
# envelope boundary, so the blended prediction is C1-continuous everywhere and
# never uses a model outside the envelope it was trained on.
#
# Blending is confined to VALIDATED regimes. Unvalidated regimes (e.g. C, which
# converges only weakly and is physically distinct — transitional low-Re, with a
# Reynolds-number gap between its envelope and A/D's) are never averaged into a
# validated prediction; they keep the single-model, warn-only path. In practice
# C's envelope does not overlap A/D at all, so it would never blend regardless —
# the validated-only rule just makes that policy explicit rather than incidental.
BLEND_FRAC = 0.5  # ramp completes this fraction of each envelope half-width from the edge


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

    # Unvalidated regime (trained on CFD cases that failed the convergence gate):
    # return a prediction but flag it as low-confidence. Never a rejection.
    if not box.get("validated", True):
        log.warning(
            "Regime %s is UNVALIDATED: its surrogate was trained on CFD cases that "
            "failed the convergence gate (weak/oscillating convergence). The %s "
            "prediction is returned but should be treated as low-confidence.",
            resolved, resolved,
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


def _smoothstep(t: float) -> float:
    """Hermite smoothstep on [0, 1] (zero slope at both ends); clamped outside."""
    if t <= 0.0:
        return 0.0
    if t >= 1.0:
        return 1.0
    return t * t * (3.0 - 2.0 * t)


def blend_weights(
    alpha_deg: float,
    Re: float,
    thickness: float,
    bounds: dict | None = None,
    blend_frac: float = BLEND_FRAC,
) -> dict[str, float]:
    """Partition-of-unity weights over the VALIDATED regimes that contain the point.

    A regime participates only if the query lies inside its achieved envelope on
    every dimension (never extrapolate an expert model). Its raw weight is the
    product over dimensions of smoothstep(margin / (blend_frac * half_width)),
    where `margin` is the distance to the nearest edge on that dimension — so the
    weight is 1 deep in the interior and decays to 0 (smoothly) at any boundary.
    Weights are normalised to sum to 1. Returns {} when no validated regime
    contains the point (the caller then falls back to the single-model gate).
    """
    bounds = bounds if bounds is not None else load_regime_bounds()
    dims = (("alpha_deg", alpha_deg), ("Re", Re), ("thickness", thickness))

    raw: dict[str, float] = {}
    for regime, box in bounds.items():
        if not box.get("validated", True):
            continue  # unvalidated regimes never blend into a validated prediction
        weight = 1.0
        contained = True
        for name, val in dims:
            lo, hi = box[name]
            if not (lo <= val <= hi):
                contained = False
                break
            half = 0.5 * (hi - lo)
            if half <= 0.0:
                factor = 1.0
            else:
                margin = min(val - lo, hi - val)
                factor = _smoothstep(margin / (blend_frac * half))
            weight *= factor
        if contained and weight > 0.0:
            raw[regime] = weight

    total = sum(raw.values())
    if total <= 0.0:
        return {}
    return {regime: w / total for regime, w in raw.items()}


def resolve_weights(
    alpha_deg: float,
    Re: float,
    thickness: float,
    bounds: dict | None = None,
) -> dict[str, float]:
    """Regime weights for a query in auto (unpinned) mode.

    Returns a multi-regime blend when validated envelopes overlap at the point;
    otherwise defers to the strict single-regime gate — which resolves C (with a
    low-confidence warning), and raises OutOfDistributionError for untrained
    regimes or points outside every trained envelope. The single case is returned
    as a degenerate {regime: 1.0} weighting so callers have one code path.
    """
    bounds = bounds if bounds is not None else load_regime_bounds()
    weights = blend_weights(alpha_deg, Re, thickness, bounds)
    if weights:
        return weights
    resolved = validate_and_classify(alpha_deg, Re, thickness, None, bounds)
    return {resolved: 1.0}


def predict_blended(
    alpha_deg: float,
    Re: float,
    thickness: float,
    family: str = "gp",
    target: str = "Cl",
    models_dir: Path = MODELS_DIR,
    weights: dict[str, float] | None = None,
) -> float:
    """Auto-mode prediction: partition-of-unity blend across overlapping regimes.

    Continuous across regime boundaries, unlike the hard-classified `predict`.
    Pass `weights` (from `resolve_weights`) to reuse a precomputed weighting and
    avoid re-running the gate; otherwise it is computed here.
    """
    if family not in FAMILIES:
        raise ValueError(f"Unknown family {family!r}; must be one of {FAMILIES}")
    if target not in TARGETS:
        raise ValueError(f"Unknown target {target!r}; must be one of {TARGETS}")

    if weights is None:
        bounds = load_regime_bounds(models_dir / "regime_bounds.json")
        weights = resolve_weights(alpha_deg, Re, thickness, bounds)

    preproc = _load_joblib(str(models_dir / "preprocessor.joblib"))
    model = _load_joblib(str(models_dir / f"{family}_{target}.joblib"))

    value = 0.0
    for regime, w in weights.items():
        X_proc = preproc.transform(_build_query(alpha_deg, Re, thickness, regime))
        value += w * _predict_one(model, family, X_proc)
    return value


def predict_all_blended(
    alpha_deg: float,
    Re: float,
    thickness: float,
    models_dir: Path = MODELS_DIR,
) -> dict:
    """Auto-mode blend for all 4 families x 2 targets, plus the regime weighting.

    Returns {"weights": {regime: w}, "predictions": {family: {target: value}}}.
    """
    bounds = load_regime_bounds(models_dir / "regime_bounds.json")
    weights = resolve_weights(alpha_deg, Re, thickness, bounds)

    preproc = _load_joblib(str(models_dir / "preprocessor.joblib"))
    x_by_regime = {
        regime: preproc.transform(_build_query(alpha_deg, Re, thickness, regime))
        for regime in weights
    }

    predictions: dict[str, dict[str, float]] = {family: {} for family in FAMILIES}
    for family in FAMILIES:
        for target in TARGETS:
            model = _load_joblib(str(models_dir / f"{family}_{target}.joblib"))
            predictions[family][target] = sum(
                w * _predict_one(model, family, x_by_regime[regime])
                for regime, w in weights.items()
            )
    return {"weights": weights, "predictions": predictions}
