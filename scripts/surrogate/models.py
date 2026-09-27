"""
Module: scripts/surrogate/models.py
Purpose: The four surrogate families (GP, RF, MLP, KRG): how each is built
         and how each predicts, in one place for training, evaluation and
         inference.

Targets are modelled as Cl and log(Cd). `predict` returns values in model
space; callers back-transform Cd with exp.
"""

from __future__ import annotations

import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from smt.surrogate_models import KRG

SEED = 42
FAMILIES = ["gp", "rf", "mlp", "krg"]
FAMILY_LABELS = {"gp": "GP", "rf": "RF", "mlp": "MLP", "krg": "KRG"}
RF_LEAF_GRID = [1, 2, 4]
MLP_GRID = {"regressor__n_iter_no_change": [10, 50, 200], "regressor__alpha": [1e-4, 1e-2]}
CV_FOLDS = 5


def build_gp(n_features: int) -> GaussianProcessRegressor:
    kernel = (Matern(length_scale=np.ones(n_features), nu=2.5) * ConstantKernel(1.0)
              + WhiteKernel(1e-3, noise_level_bounds=(1e-10, 1e1)))
    return GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                    n_restarts_optimizer=5, random_state=SEED)


def fit_rf(X: np.ndarray, y: np.ndarray) -> tuple[RandomForestRegressor, dict]:
    """500 trees; min_samples_leaf picked by 5-fold CV on the training rows only."""
    search = _cv_search(RandomForestRegressor(n_estimators=500, random_state=SEED, n_jobs=1),
                        {"min_samples_leaf": RF_LEAF_GRID}, X, y)
    cv = {"min_samples_leaf": search.best_params_["min_samples_leaf"],
          "cv_rmse": {int(p["min_samples_leaf"]): float(-s) for p, s in
                      zip(search.cv_results_["params"], search.cv_results_["mean_test_score"])}}
    return search.best_estimator_, cv


def build_mlp() -> TransformedTargetRegressor:
    # Target scaling matters: the retired pipeline's MLP-Cd failed on unscaled targets.
    return TransformedTargetRegressor(
        regressor=MLPRegressor(hidden_layer_sizes=(128, 128, 64), early_stopping=True,
                               max_iter=5000, random_state=SEED),
        transformer=StandardScaler(),
    )


def _cv_search(estimator, grid: dict, X: np.ndarray, y: np.ndarray):
    return GridSearchCV(estimator, grid, cv=KFold(CV_FOLDS, shuffle=True, random_state=SEED),
                        scoring="neg_root_mean_squared_error", n_jobs=-1).fit(X, y)


def fit_mlp(X: np.ndarray, y: np.ndarray) -> tuple[TransformedTargetRegressor, dict]:
    """Early-stopping patience and L2 penalty picked by 5-fold CV on the training rows only.

    With few rows (scarce: 200) the default patience of 10 epochs stops
    training long before convergence on the small internal validation split.
    """
    search = _cv_search(build_mlp(), MLP_GRID, X, y)
    best = search.best_estimator_
    return best, {"best_params": {k.split("__")[1]: v for k, v in search.best_params_.items()},
                  "cv_rmse": float(-search.best_score_), "n_iter": int(best.regressor_.n_iter_)}


def fit_krg(X: np.ndarray, y: np.ndarray) -> KRG:
    krg = KRG(theta0=[1e-2] * X.shape[1], eval_noise=True, seed=SEED, print_global=False)
    krg.set_training_values(np.asarray(X, dtype=np.float64), np.asarray(y, dtype=np.float64).reshape(-1, 1))
    krg.train()
    return krg


def fit(family: str, X: np.ndarray, y: np.ndarray):
    """Fit one family on (X, y); returns (model, info dict)."""
    if family == "gp":
        model = build_gp(X.shape[1]).fit(X, y)
        return model, {"kernel": str(model.kernel_),
                       "log_marginal_likelihood": float(model.log_marginal_likelihood_value_)}
    if family == "rf":
        return fit_rf(X, y)
    if family == "mlp":
        return fit_mlp(X, y)
    if family == "krg":
        model = fit_krg(X, y)
        return model, {"theta": [float(t) for t in np.ravel(model.optimal_theta)]}
    raise ValueError(f"unknown family {family!r}")


def predict(model, family: str, X: np.ndarray, return_std: bool = False):
    """Model-space prediction; with return_std, also an uncertainty (None if unavailable).

    GP and KRG give their posterior standard deviation; RF gives the spread of
    its trees (a heuristic, not a calibrated interval); MLP gives none.
    """
    X = np.asarray(X, dtype=np.float64)
    if family == "gp":
        if return_std:
            mean, std = model.predict(X, return_std=True)
            return mean, std
        return model.predict(X)
    if family == "krg":
        mean = model.predict_values(X).ravel()
        if return_std:
            return mean, np.sqrt(np.clip(model.predict_variances(X).ravel(), 0, None))
        return mean
    if family == "rf":
        if return_std:
            per_tree = np.stack([t.predict(X) for t in model.estimators_])
            return per_tree.mean(axis=0), per_tree.std(axis=0, ddof=1)
        return model.predict(X)
    if family == "mlp":
        mean = model.predict(X)
        return (mean, None) if return_std else mean
    raise ValueError(f"unknown family {family!r}")
