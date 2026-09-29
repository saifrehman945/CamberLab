#!/usr/bin/env python3
"""
Script: 09_train_surrogates.py
Stage:  9 — Train surrogate models (AirfRANS)
Purpose: For each AirfRANS task, fit the feature scaler and the four model
         families (GP, RF, MLP, KRG) for Cl and log(Cd), and for the wall
         Cp / Cf curves, on the task's frozen training split only, and
         persist them under models/{task}/.

Writes, per task:
    models/{task}/preprocessor.joblib               StandardScaler over FEATURES
    models/{task}/{family}_{Cl,Cd}.joblib           one model per target per family
    models/{task}/train_pred.npz                    train predictions (Gate 6 reload check)
    models/{task}/curves/pca_{Cp,Cf}.joblib         PCA of the 202-station curves
    models/{task}/curves/{family}_{Cp,Cf}.joblib    mode-weight model(s) per family
    models/{task}/curves/train_pred.npz             train mode-weight predictions (reload check)
    models/{task}/envelope.json                     train feature ranges + fit metadata

Cf curves are trained only if Cf passed scripts/23_surface_gate.py. Test rows
are never touched here. Hyperparameter searches (RF min_samples_leaf, MLP
patience/L2, the PCA mode count) are 5-fold CV on the training rows.

Usage:
    uv run python scripts/09_train_surrogates.py --task full
    uv run python scripts/09_train_surrogates.py --task all [--outputs coeffs|curves|all]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.surrogate.data import (  # noqa: E402
    FEATURES,
    MODELS_DIR,
    TARGETS,
    TASKS,
    envelope,
    get_xy,
    load_dataset,
    task_frames,
)
from scripts.surrogate.curves import fit_pca, fit_weights, gated_quantities, load_curves, predict_weights  # noqa: E402
from scripts.surrogate.models import FAMILIES, fit, predict  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(message)s")
log = logging.getLogger(__name__)

JOBLIB_COMPRESS = 3


def fitted_preprocessor(task: str, train, refit: bool) -> StandardScaler:
    """The task's feature scaler: refitted with the Cl/Cd models, else reloaded (curves only)."""
    path = MODELS_DIR / task / "preprocessor.joblib"
    if refit or not path.exists():
        preproc = StandardScaler().fit(get_xy(train)[0])
        joblib.dump(preproc, path, compress=JOBLIB_COMPRESS)
        return preproc
    preproc = joblib.load(path)
    if not np.allclose(preproc.mean_, get_xy(train)[0].mean().to_numpy()):
        raise RuntimeError(f"{path} was fitted on different training rows; retrain with --outputs all")
    return preproc


def train_curves(task: str, train, X: np.ndarray, families: list[str]) -> dict:
    """PCA + mode-weight models for each gated curve quantity; returns envelope curve_info."""
    out = MODELS_DIR / task / "curves"
    out.mkdir(parents=True, exist_ok=True)
    _, curves = load_curves()
    rows = train.sample_id.to_numpy()
    info: dict = {}
    train_pred: dict[str, np.ndarray] = {"sample_id": rows}
    for q in gated_quantities():
        pca, pca_info = fit_pca(curves[q][rows])
        joblib.dump(pca, out / f"pca_{q}.joblib", compress=JOBLIB_COMPRESS)
        W = pca.transform(curves[q][rows])
        log.info("[%s] %s: %d PCA modes (%.4f%% variance)", task, q, pca_info["n_modes"],
                 100 * pca_info["explained_variance"])
        info[q] = {**pca_info, "families": {}}
        for family in families:
            t0 = time.perf_counter()
            model, meta = fit_weights(family, X, W)
            meta["fit_seconds"] = round(time.perf_counter() - t0, 2)
            path = out / f"{family}_{q}.joblib"
            joblib.dump(model, path, compress=JOBLIB_COMPRESS)
            train_pred[f"{family}_{q}"] = predict_weights(model, family, X)
            info[q]["families"][family] = meta
            log.info("[%s] %s_%s curves fitted in %.1fs → %s (%.1f MB)", task, family, q,
                     meta["fit_seconds"], path.relative_to(PROJECT_ROOT), path.stat().st_size / 1e6)
    pred_path = out / "train_pred.npz"
    if pred_path.exists():
        with np.load(pred_path) as old:
            if np.array_equal(old["sample_id"], rows):
                train_pred = {**{k: old[k] for k in old.files}, **train_pred}
    np.savez(pred_path, **train_pred)
    return info


def train_task(task: str, df, families: list[str], outputs: str) -> None:
    train, test = task_frames(task, df)
    log.info("[%s] train rows %d (test rows %d untouched)", task, len(train), len(test))
    X_df, y_cl, y_logcd = get_xy(train)
    out = MODELS_DIR / task
    out.mkdir(parents=True, exist_ok=True)

    do_coeffs = outputs in ("coeffs", "all")
    preproc = fitted_preprocessor(task, train, refit=do_coeffs)
    X = preproc.transform(X_df)
    env_path = out / "envelope.json"
    previous = json.loads(env_path.read_text()) if env_path.exists() else {}
    curve_info = previous.get("curve_info", {})
    if outputs in ("curves", "all"):
        for q, new in train_curves(task, train, X, families).items():
            # keep other families' fit info when only some were retrained on the same PCA
            old = curve_info.get(q, {})
            kept = old.get("families", {}) if old.get("n_modes") == new["n_modes"] else {}
            curve_info[q] = {**new, "families": {**kept, **new["families"]}}
    if not do_coeffs:
        env_path.write_text(json.dumps({**previous, "curve_info": curve_info}, indent=2))
        return

    targets = {"Cl": y_cl, "Cd": y_logcd}
    info: dict = {}
    train_pred: dict[str, np.ndarray] = {"sample_id": train.sample_id.to_numpy()}
    for target in TARGETS:
        for family in families:
            t0 = time.perf_counter()
            model, meta = fit(family, X, targets[target])
            meta["fit_seconds"] = round(time.perf_counter() - t0, 2)
            path = out / f"{family}_{target}.joblib"
            joblib.dump(model, path, compress=JOBLIB_COMPRESS)
            train_pred[f"{family}_{target}"] = predict(model, family, X)
            info[f"{family}_{target}"] = meta
            log.info("[%s] %s_%s fitted in %.1fs → %s (%.1f MB)", task, family, target,
                     meta["fit_seconds"], path.relative_to(PROJECT_ROOT), path.stat().st_size / 1e6)

    pred_path = out / "train_pred.npz"
    if pred_path.exists():
        with np.load(pred_path) as old:
            if np.array_equal(old["sample_id"], train_pred["sample_id"]):
                train_pred = {**{k: old[k] for k in old.files}, **train_pred}
    np.savez(pred_path, **train_pred)
    fit_info = {**previous.get("fit_info", {}), **info}
    env_path.write_text(json.dumps({
        "task": task,
        "features": FEATURES,
        "targets": {"Cl": "Cl", "Cd": "log(Cd); predictions back-transformed with exp"},
        "n_train": len(train),
        "envelope": envelope(train),
        "versions": {p: version(p) for p in ("scikit-learn", "smt", "numpy", "scipy", "joblib")},
        "fit_info": fit_info,
        "curve_info": curve_info,
    }, indent=2))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", choices=[*TASKS, "all"], default="full")
    ap.add_argument("--family", choices=[*FAMILIES, "all"], default="all")
    ap.add_argument("--outputs", choices=["coeffs", "curves", "all"], default="all",
                    help="Cl/Cd models, wall-curve models, or both (curves need 23_surface_gate.py)")
    args = ap.parse_args()

    df = load_dataset()
    tasks = TASKS if args.task == "all" else [args.task]
    families = FAMILIES if args.family == "all" else [args.family]
    for task in tasks:
        train_task(task, df, families, args.outputs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
