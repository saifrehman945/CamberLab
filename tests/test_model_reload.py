"""Gate 6: every persisted model reloads under the pinned versions and reproduces
its own training-set predictions (models/{task}/train_pred.npz) exactly.

Runs in the pytest process, which is separate from the training process.
"""

from __future__ import annotations

import json
import sys
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.surrogate.data import TARGETS, TASKS, get_xy, task_frames  # noqa: E402
from scripts.surrogate.models import FAMILIES, predict  # noqa: E402

MODELS_DIR = PROJECT_ROOT / "models"
# models/full/ is committed; the other tasks exist only after a local `--task all` run.
AVAILABLE = [t for t in TASKS if (MODELS_DIR / t / "envelope.json").exists()]
CASES = [(t, f, y) for t in AVAILABLE for f in FAMILIES for y in TARGETS]

pytestmark = pytest.mark.skipif(not (MODELS_DIR / "full" / "envelope.json").exists(),
                                reason="run scripts/09_train_surrogates.py --task all first")


def test_full_task_models_present():
    assert "full" in AVAILABLE


@pytest.mark.parametrize("task", AVAILABLE)
def test_pinned_versions_match_training(task):
    trained = json.loads((MODELS_DIR / task / "envelope.json").read_text())["versions"]
    for pkg in ("scikit-learn", "smt"):
        assert version(pkg) == trained[pkg], f"{pkg}: installed {version(pkg)}, trained {trained[pkg]}"


@pytest.mark.parametrize("task,family,target", CASES)
def test_reload_reproduces_train_predictions(task, family, target):
    d = MODELS_DIR / task
    train, _ = task_frames(task)
    X = joblib.load(d / "preprocessor.joblib").transform(get_xy(train)[0])
    model = joblib.load(d / f"{family}_{target}.joblib")
    with np.load(d / "train_pred.npz") as saved:
        assert np.array_equal(saved["sample_id"], train.sample_id.to_numpy())
        expected = saved[f"{family}_{target}"]
    got = predict(model, family, X)
    assert np.array_equal(got, expected), float(np.max(np.abs(got - expected)))
