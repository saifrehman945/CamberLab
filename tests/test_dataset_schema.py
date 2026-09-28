"""Gate 3 schema checks on results/airfrans_dataset.csv (skipped until ingestion has run)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
DATASET = PROJECT_ROOT / "results" / "airfrans_dataset.csv"
SURFACES = PROJECT_ROOT / "results" / "airfrans_surfaces.npz"

pytestmark = pytest.mark.skipif(not DATASET.exists(), reason="run scripts/20_ingest_airfrans.py first")


def _ingest_module():
    spec = importlib.util.spec_from_file_location("ingest", PROJECT_ROOT / "scripts" / "20_ingest_airfrans.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    return pd.read_csv(DATASET, keep_default_na=False, na_values=[""],
                       dtype={"source_name": str})


def test_columns(df):
    assert list(df.columns) == _ingest_module().COLUMNS


def test_gate3(df):
    for label, ok in _ingest_module().gate3(df, 1000):
        assert ok, label


def test_sample_ids_are_row_indices(df):
    assert (df.sample_id.to_numpy() == np.arange(len(df))).all()


def test_re_consistent_with_velocity(df):
    from scripts.airfrans.io import NU
    assert np.allclose(df.Re, df.U_inf / NU, rtol=1e-9)
    assert np.allclose(df.log10_Re, np.log10(df.Re), rtol=1e-12)


def test_surfaces_match_dataset(df):
    s = np.load(SURFACES)
    assert (s["sample_id"] == df.sample_id.to_numpy()).all()
    assert s["y_upper"].shape == (len(df), 101) == s["y_lower"].shape
    t = s["y_upper"] - s["y_lower"]
    assert np.allclose(t.max(axis=1), df.t_max, atol=2e-3)
