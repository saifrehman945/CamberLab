#!/usr/bin/env python3
"""
Script: 09_train_surrogates.py
Stage:  9 — Train surrogate models
Purpose: Load results/dataset_clean.csv, fit the shared preprocessor and the
         4 model families (GP, RF, MLP, Kriging) for each of Cl and Cd on the
         training split only, and persist everything under models/.

Only regimes actually present in the training split get a trained model —
absent regimes are skipped with a warning, not synthesized. Re-running this
script after more regimes converge (via 07_harvest_results.py) requires no
code changes.

Usage:
    micromamba run -n openfoam python scripts/09_train_surrogates.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import joblib
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from smt.surrogate_models import KRG

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.surrogate.data import (  # noqa: E402
    REGIMES,
    MODELS_DIR,
    get_xy,
    load_dataset,
    load_split_indices,
    split_dataset,
)
from scripts.surrogate.regime_bounds import compute_regime_bounds, save_regime_bounds  # noqa: E402

SEED = 42
NUMERIC_FEATURES = ["alpha_deg", "Re", "thickness"]
ONEHOT_FEATURES = [f"regime_{r}" for r in REGIMES]
TARGETS = ["Cl", "Cd"]
MLP_MIN_ROWS_FOR_CONFIDENCE = 50

CD_REGIME_A_CAVEAT = (
    "NOTE: validation/regime_A/report.md shows Regime A Cd disagrees with the "
    "Ladson reference by 16-47% at all 4 probe angles despite good Cl agreement. "
    "Cd_A models are trained and persisted as requested, but downstream "
    "consumers should treat Regime A Cd predictions with reduced confidence "
    "pending mesh/turbulence-model refinement."
)


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", "passthrough", ONEHOT_FEATURES),
        ]
    )


def build_models(n_train: int) -> dict:
    if n_train < MLP_MIN_ROWS_FOR_CONFIDENCE:
        log.warning(
            "only %d training rows (<%d) — MLPRegressor's internal early_stopping "
            "validation split will be very small; treat MLP results as provisional",
            n_train, MLP_MIN_ROWS_FOR_CONFIDENCE,
        )
    return {
        "gp": GaussianProcessRegressor(
            kernel=Matern(nu=2.5) + WhiteKernel(),
            n_restarts_optimizer=10,
            normalize_y=True,
            random_state=SEED,
        ),
        "rf": RandomForestRegressor(n_estimators=200, random_state=SEED),
        "mlp": MLPRegressor(
            hidden_layer_sizes=(64, 64, 32),
            activation="relu",
            early_stopping=True,
            random_state=SEED,
        ),
    }


def fit_krg(X_proc, y) -> KRG:
    krg = KRG(print_global=False)
    krg.set_training_values(X_proc, y.to_numpy().reshape(-1, 1))
    krg.train()
    return krg


def main() -> None:
    df = load_dataset()
    train_idx, test_idx = load_split_indices()
    train_df, test_df = split_dataset(df, train_idx, test_idx)
    log.info("train rows: %d, test rows: %d", len(train_df), len(test_df))

    if train_df.empty:
        raise RuntimeError(
            "No converged training rows available — run scripts/07_harvest_results.py "
            "after at least some CFD cases have finished."
        )

    X_train, y_cl_train, y_cd_train = get_xy(train_df)

    preproc = build_preprocessor()
    X_train_proc = preproc.fit_transform(X_train)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(preproc, MODELS_DIR / "preprocessor.joblib")
    log.info("Wrote %s", MODELS_DIR / "preprocessor.joblib")

    targets = {"Cl": y_cl_train, "Cd": y_cd_train}
    for target in TARGETS:
        y = targets[target]
        models = build_models(len(train_df))
        for family, model in models.items():
            model.fit(X_train_proc, y)
            path = MODELS_DIR / f"{family}_{target}.joblib"
            joblib.dump(model, path)
            log.info("Wrote %s", path)

        krg = fit_krg(X_train_proc, y)
        path = MODELS_DIR / f"krg_{target}.joblib"
        joblib.dump(krg, path)
        log.info("Wrote %s", path)

    bounds = compute_regime_bounds(train_df)
    save_regime_bounds(bounds)

    trained_regimes = sorted(bounds)
    absent_regimes = [r for r in REGIMES if r not in bounds]
    log.info("Trained regimes: %s", trained_regimes)
    if absent_regimes:
        log.warning(
            "Regimes with no trained model yet (0 converged training rows): %s",
            absent_regimes,
        )


if __name__ == "__main__":
    main()
