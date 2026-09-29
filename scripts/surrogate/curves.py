"""
Module: scripts/surrogate/curves.py
Purpose: Surface-curve surrogates. Cp(x/c) and Cf(x/c) on the 202-station
         grid (101 cosine-spaced stations per surface, upper then lower) are
         compressed with PCA, and the existing families (models.fit /
         models.predict) predict the PCA mode weights from the same six
         features as the Cl/Cd models.

- PCA is fitted on the task's training rows only, with a fixed N_MODES = 20
  modes per quantity. The 5-fold CV reconstruction error on those rows is
  recorded for k in MODE_RANGE as the truncation floor. At k = 20 on `full`
  that floor is negligible next to the regressors' error for Cp (RMSE 0.0044
  vs >= 0.06) and for overall Cf (1.5e-4 vs ~1e-3). The exception is the Cf
  separation point: truncation alone costs ~0.015 c, as much as GP's
  error there (0.004 c at k = 40). k = 20 is kept as a deliberate
  cost/accuracy trade-off: GP/KRG fit one model per mode.
- GP and KRG fit one model per mode, so each mode has a posterior σ. The
  curve band is ±2·sqrt(Σ σᵢ² φᵢ²), which treats modes as independent (a
  documented approximation). RF and MLP fit one multi-output model; RF's
  tree spread is not propagated, so as for Cl/Cd, bands are GP/KRG only.
- Curve = PCA mean + Σ wᵢ φᵢ.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold

from scripts.surrogate.data import RESULTS_DIR
from scripts.surrogate.models import CV_FOLDS, SEED, fit, predict

log = logging.getLogger(__name__)

CURVES_PATH = RESULTS_DIR / "airfrans_wall_curves.npz"
GATE_PATH = RESULTS_DIR / "airfrans_surface_gate.json"
QUANTITIES = ["Cp", "Cf"]
SIDES = ["upper", "lower"]
PER_MODE_FAMILIES = {"gp", "krg"}          # one model per mode (posterior σ per mode)
N_MODES = 20                                # fixed; see the module docstring
MODE_RANGE = range(4, 31)                   # k values whose CV reconstruction RMSE is recorded


def gated_quantities(gate_path: Path = GATE_PATH) -> list[str]:
    """Quantities that passed the wall-data gate (scripts/23_surface_gate.py): Cp, and Cf if it passed."""
    if not gate_path.exists():
        raise FileNotFoundError(f"{gate_path} not found — run scripts/23_surface_gate.py first.")
    gate = json.loads(gate_path.read_text())
    if not gate["cp_pass"]:
        raise RuntimeError("the wall-data gate failed for Cp; no curve models can be trained")
    return ["Cp", "Cf"] if gate["cf_pass"] else ["Cp"]


def load_curves(path: Path = CURVES_PATH) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """(x_grid, {quantity: (n_samples, 202) array}) with rows == sample_id."""
    with np.load(path) as c:
        assert (c["sample_id"] == np.arange(len(c["sample_id"]))).all()
        return c["x_grid"], {q: np.hstack([c[f"{q}_{s}"] for s in SIDES]) for q in QUANTITIES}


def split_sides(Y: np.ndarray) -> dict[str, np.ndarray]:
    """(n, 202) -> {"upper": (n, 101), "lower": (n, 101)}."""
    half = Y.shape[1] // 2
    return {"upper": Y[:, :half], "lower": Y[:, half:]}


# --------------------------------------------------------------------------
# PCA
# --------------------------------------------------------------------------

def cv_reconstruction_rmse(Y: np.ndarray, modes=MODE_RANGE) -> dict[int, float]:
    """Held-out reconstruction RMSE per mode count, 5-fold CV on the given (training) rows."""
    kf = KFold(CV_FOLDS, shuffle=True, random_state=SEED)
    k_max = max(modes)
    sq = {k: 0.0 for k in modes}
    for fit_idx, held_idx in kf.split(Y):
        pca = PCA(n_components=k_max, svd_solver="full").fit(Y[fit_idx])
        centred = Y[held_idx] - pca.mean_
        w = centred @ pca.components_.T
        for k in modes:
            resid = centred - w[:, :k] @ pca.components_[:k]
            sq[k] += float((resid**2).sum())
    return {k: float(np.sqrt(v / Y.size)) for k, v in sq.items()}


def fit_pca(Y: np.ndarray, n_modes: int = N_MODES) -> tuple[PCA, dict]:
    """PCA with a fixed mode count; the CV reconstruction RMSE per k is recorded, not used to choose."""
    cv = cv_reconstruction_rmse(Y)
    pca = PCA(n_components=n_modes, svd_solver="full").fit(Y)
    return pca, {"n_modes": n_modes, "explained_variance": float(pca.explained_variance_ratio_.sum()),
                 "cv_reconstruction_rmse": {int(m): r for m, r in cv.items()}}


# --------------------------------------------------------------------------
# Mode-weight models
# --------------------------------------------------------------------------

def fit_weights(family: str, X: np.ndarray, W: np.ndarray):
    """Fit a family to the (n, k) mode weights; returns (model or list of per-mode models, info)."""
    if family in PER_MODE_FAMILIES:
        fitted = [fit(family, X, W[:, i]) for i in range(W.shape[1])]
        return [m for m, _ in fitted], {"per_mode": [info for _, info in fitted]}
    return fit(family, X, W)


def predict_weights(model, family: str, X: np.ndarray, return_std: bool = False):
    """(n, k) predicted mode weights; with return_std also (n, k) σ, or None for RF/MLP."""
    if family in PER_MODE_FAMILIES:
        out = [predict(m, family, X, return_std=return_std) for m in model]
        if return_std:
            return np.column_stack([o[0] for o in out]), np.column_stack([o[1] for o in out])
        return np.column_stack(out)
    mean = np.asarray(predict(model, family, X)).reshape(len(X), -1)
    return (mean, None) if return_std else mean


def predict_curves(pca: PCA, model, family: str, X: np.ndarray, return_std: bool = False):
    """(n, 202) curves; with return_std also the (n, 202) propagated σ (None for RF/MLP)."""
    if not return_std:
        return pca.inverse_transform(predict_weights(model, family, X))
    w, s = predict_weights(model, family, X, return_std=True)
    curve = pca.inverse_transform(w)
    if s is None:
        return curve, None
    return curve, np.sqrt((s**2) @ (pca.components_**2))
