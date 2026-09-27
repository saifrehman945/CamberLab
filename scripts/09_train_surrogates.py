#!/usr/bin/env python3
"""
Script: 09_train_surrogates.py
Stage:  9 — Train surrogate models (AirfRANS)
Purpose: For each AirfRANS task, fit the feature scaler and the four model
         families (GP, RF, MLP, KRG) for Cl and log(Cd) on the task's frozen
         training split only, and persist them under models/{task}/.

Writes, per task:
    models/{task}/preprocessor.joblib        StandardScaler over FEATURES
    models/{task}/{family}_{Cl,Cd}.joblib    one model per target per family
    models/{task}/envelope.json              train feature ranges + fit metadata
    models/{task}/train_pred.npz             train predictions (Gate 6 reload check)

Test rows are never touched here. The only hyperparameter search (RF
min_samples_leaf) is 5-fold CV on the training rows.

Usage:
    uv run python scripts/09_train_surrogates.py --task full
    uv run python scripts/09_train_surrogates.py --task all
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
from scripts.surrogate.models import FAMILIES, fit, predict  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(message)s")
log = logging.getLogger(__name__)

JOBLIB_COMPRESS = 3


def train_task(task: str, df, families: list[str]) -> None:
    train, test = task_frames(task, df)
    log.info("[%s] train rows %d (test rows %d untouched)", task, len(train), len(test))
    X_df, y_cl, y_logcd = get_xy(train)
    out = MODELS_DIR / task
    out.mkdir(parents=True, exist_ok=True)

    preproc = StandardScaler().fit(X_df)
    X = preproc.transform(X_df)
    joblib.dump(preproc, out / "preprocessor.joblib", compress=JOBLIB_COMPRESS)

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
    env_path = out / "envelope.json"
    previous = json.loads(env_path.read_text()) if env_path.exists() else {}
    fit_info = {**previous.get("fit_info", {}), **info}
    env_path.write_text(json.dumps({
        "task": task,
        "features": FEATURES,
        "targets": {"Cl": "Cl", "Cd": "log(Cd); predictions back-transformed with exp"},
        "n_train": len(train),
        "envelope": envelope(train),
        "versions": {p: version(p) for p in ("scikit-learn", "smt", "numpy", "scipy", "joblib")},
        "fit_info": fit_info,
    }, indent=2))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", choices=[*TASKS, "all"], default="full")
    ap.add_argument("--family", choices=[*FAMILIES, "all"], default="all")
    args = ap.parse_args()

    df = load_dataset()
    tasks = TASKS if args.task == "all" else [args.task]
    families = FAMILIES if args.family == "all" else [args.family]
    for task in tasks:
        train_task(task, df, families)
    return 0


if __name__ == "__main__":
    sys.exit(main())
