#!/usr/bin/env python3
"""
Script: 10_global_validation.py
Stage:  10 — Global surrogate validation
Purpose: Evaluate the persisted surrogate models on the sacred test split
         exactly once. Writes results/surrogate_metrics.csv (global + per-
         regime R2/RMSE/MAE for every model family x target) and parity
         plots. Never retrains — read-only with respect to models/.

Usage:
    micromamba run -n openfoam python scripts/10_global_validation.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.surrogate.data import (  # noqa: E402
    MODELS_DIR,
    REGIMES,
    RESULTS_DIR,
    get_xy,
    load_dataset,
    load_split_indices,
    split_dataset,
)
from scripts.surrogate.inference import FAMILIES, TARGETS  # noqa: E402

SMALL_TEST_SLICE_WARN = 10
ACCEPTANCE_BAR = {"Cl": 0.90, "Cd": 0.85}

CD_REGIME_A_CAVEAT = (
    "NOTE: Regime A Cd is known to disagree with the Ladson reference by "
    "16-47% at all 4 probe angles (see validation/regime_A/report.md). "
    "Cd_A metrics below reflect surrogate fit to CFD data of uncertain accuracy."
)


def _predict(model, family: str, X_proc):
    if family == "krg":
        return model.predict_values(np.asarray(X_proc))[:, 0]
    return model.predict(X_proc)


def compute_metrics(y_true, y_pred, slice_name: str) -> dict:
    r2 = float(r2_score(y_true, y_pred)) if len(y_true) > 1 else float("nan")
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    return {"slice": slice_name, "R2": r2, "RMSE": rmse, "MAE": mae, "n_samples": int(len(y_true))}


def plot_parity(target: str, y_true: np.ndarray, preds_by_family: dict) -> Path:
    all_vals = [y_true] + list(preds_by_family.values())
    lo = min(v.min() for v in all_vals)
    hi = max(v.max() for v in all_vals)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, label="y = x")
    for family, y_pred in preds_by_family.items():
        ax.scatter(y_true, y_pred, label=family, alpha=0.7)
    ax.set_xlabel(f"CFD {target}")
    ax.set_ylabel(f"Surrogate {target}")
    ax.set_title(f"Parity plot — {target} (test set, n={len(y_true)})")
    ax.legend()
    fig.tight_layout()

    out_path = RESULTS_DIR / f"parity_{target}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main() -> None:
    df = load_dataset()
    train_idx, test_idx = load_split_indices()
    _, test_df = split_dataset(df, train_idx, test_idx)

    if test_df.empty:
        log.warning("Test split has 0 converged rows — nothing to validate yet.")
        return

    X_test, y_cl_test, y_cd_test = get_xy(test_df)
    preproc = joblib.load(MODELS_DIR / "preprocessor.joblib")
    X_test_proc = preproc.transform(X_test)

    targets = {"Cl": y_cl_test, "Cd": y_cd_test}
    rows = []

    for target in TARGETS:
        y_true = targets[target].to_numpy()
        preds_by_family: dict[str, np.ndarray] = {}

        for family in FAMILIES:
            model_path = MODELS_DIR / f"{family}_{target}.joblib"
            if not model_path.exists():
                log.warning("%s missing — skipping", model_path)
                continue
            model = joblib.load(model_path)
            y_pred = np.asarray(_predict(model, family, X_test_proc))
            preds_by_family[family] = y_pred

            global_metrics = compute_metrics(y_true, y_pred, "global")
            global_metrics.update({"target": target, "family": family})
            global_metrics["meets_bar"] = (
                not np.isnan(global_metrics["R2"]) and global_metrics["R2"] >= ACCEPTANCE_BAR[target]
            )
            rows.append(global_metrics)

            for regime in REGIMES:
                mask = (test_df["regime"] == regime).to_numpy()
                n = int(mask.sum())
                if n == 0:
                    continue
                if n < SMALL_TEST_SLICE_WARN:
                    log.warning(
                        "regime %s test slice for %s/%s has only %d samples — "
                        "R2/RMSE/MAE are informational only",
                        regime, family, target, n,
                    )
                regime_metrics = compute_metrics(y_true[mask], y_pred[mask], f"regime_{regime}")
                regime_metrics.update({"target": target, "family": family})
                regime_metrics["meets_bar"] = (
                    not np.isnan(regime_metrics["R2"]) and regime_metrics["R2"] >= ACCEPTANCE_BAR[target]
                )
                rows.append(regime_metrics)

        if preds_by_family:
            out_path = plot_parity(target, y_true, preds_by_family)
            log.info("Wrote %s", out_path)

    metrics_df = pd.DataFrame(
        rows, columns=["target", "family", "slice", "R2", "RMSE", "MAE", "n_samples", "meets_bar"]
    )
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = RESULTS_DIR / "surrogate_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    log.info("Wrote %s (%d rows)", metrics_path, len(metrics_df))

    if "A" in test_df["regime"].unique():
        log.warning(CD_REGIME_A_CAVEAT)


if __name__ == "__main__":
    main()
