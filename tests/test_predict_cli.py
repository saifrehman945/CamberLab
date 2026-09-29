"""Smoke tests for scripts/predict.py and the inference API (every family)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.surrogate.models import FAMILIES  # noqa: E402

pytestmark = pytest.mark.skipif(not (PROJECT_ROOT / "models" / "full" / "envelope.json").exists(),
                                reason="run scripts/09_train_surrogates.py first")


def _cli():
    spec = importlib.util.spec_from_file_location("predict_cli", PROJECT_ROOT / "scripts" / "predict.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("family", [*FAMILIES, "all"])
def test_cli_every_family(family, capsys):
    rc = _cli().main(["--alpha", "4", "--re", "3e6", "--naca", "2412", "--family", family, "--json"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    expected = FAMILIES if family == "all" else [family]
    assert list(out["predictions"]) == expected
    for r in out["predictions"].values():
        assert 0.3 < r["Cl"] < 0.9 and 0.005 < r["Cd"] < 0.02
        assert r["in_envelope"] is True


def test_cli_human_output_and_five_digit():
    assert _cli().main(["--alpha", "2", "--re", "4e6", "--naca", "23012", "--family", "all"]) == 0


def test_cli_rejects_bad_code():
    assert _cli().main(["--alpha", "2", "--re", "4e6", "--naca", "12345"]) == 2


def test_out_of_envelope_warns_but_predicts():
    from scripts.surrogate.inference import predict
    r = predict(4.0, 1.0e7, "2412", "gp")
    assert not r["in_envelope"]
    assert any("Re" in w for w in r["warnings"])
    assert np.isfinite(r["Cl"]) and r["Cd"] > 0


def test_uncertainty_only_for_gp_and_krg():
    from scripts.surrogate.inference import predict
    for family in FAMILIES:
        r = predict(4.0, 3e6, "0012", family)
        has_std = r["Cl_std"] is not None and r["Cd_std"] is not None
        assert has_std == (family in {"gp", "krg"})


def test_symmetric_section_near_zero_lift_at_zero_alpha():
    from scripts.surrogate.inference import predict
    assert abs(predict(0.0, 4e6, "0012", "gp")["Cl"]) < 0.02


HAS_CURVES = (PROJECT_ROOT / "models" / "full" / "curves").exists()
CURVE_FAMILIES = [f for f in FAMILIES
                  if all((PROJECT_ROOT / "models" / "full" / "curves" / f"{f}_{q}.joblib").exists()
                         for q in ("Cp",))]


@pytest.mark.skipif(not HAS_CURVES, reason="run scripts/09_train_surrogates.py --outputs curves first")
@pytest.mark.parametrize("family", CURVE_FAMILIES)
def test_surface_every_family(family, tmp_path, capsys):
    csv = tmp_path / "surface.csv"
    rc = _cli().main(["--alpha", "4", "--re", "3e6", "--naca", "2412", "--family", family,
                      "--surface-csv", str(csv), "--json"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    s = out["surface"][family]
    # attached, cambered section at 4°: suction peak on the upper surface near the nose
    assert s["summary"]["surface_Cp_min"] == "upper"
    assert s["summary"]["x_c_Cp_min"] < 0.1
    assert -2.5 < s["summary"]["Cp_min"] < -0.5
    assert len(s["curves"]["x_c"]) == 202
    assert csv.exists()


@pytest.mark.skipif(not HAS_CURVES, reason="run scripts/09_train_surrogates.py --outputs curves first")
def test_surface_bands_only_for_gp_and_krg():
    from scripts.surrogate.inference import predict_surface
    for family in CURVE_FAMILIES:
        c = predict_surface(4.0, 3e6, "0012", family)
        assert ("Cp_lo" in c) == (family in {"gp", "krg"})
        assert np.isfinite(c.Cp).all()


@pytest.mark.skipif(not HAS_CURVES, reason="run scripts/09_train_surrogates.py --outputs curves first")
def test_surface_missing_family_is_a_clean_error(monkeypatch):
    from scripts.surrogate import inference
    monkeypatch.setattr(inference, "has_curve_models", lambda family, task="full": False)
    assert _cli().main(["--alpha", "4", "--re", "3e6", "--naca", "2412", "--surface"]) == 2
