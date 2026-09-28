"""Geometry tests: analytic NACA sections and mesh-like resampling (Phase 3 gate)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.airfrans.geometry import (  # noqa: E402
    _standard_5digit_constants,
    naca_coordinates,
    parse_naca,
    section_features,
)

CODES = ["0012", "2412", "4415", "23012", "23112"]
FEATURES = ["t_max", "x_tmax", "m_max", "x_m"]


def resample_like_mesh(xy: np.ndarray, n_nodes: int = 150) -> np.ndarray:
    """Resample a dense TE->upper->LE->lower loop to n_nodes, clustered at LE and TE.

    Each surface is parametrised by arc length and sampled at cosine-spaced
    arc-length stations, mimicking the wall-node distribution of a CFD mesh.
    """
    le = int(np.argmin(xy[:, 0]))
    upper = xy[:le + 1][::-1]                          # LE -> TE
    lower = np.vstack([xy[le:], xy[:1]])               # LE -> TE
    n_side = n_nodes // 2 + 1
    stations = 0.5 * (1 - np.cos(np.linspace(0, np.pi, n_side)))

    def resample(curve):
        s = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(curve, axis=0).T))])
        s /= s[-1]
        return np.column_stack([np.interp(stations, s, curve[:, 0]),
                                np.interp(stations, s, curve[:, 1])])

    up, lo = resample(upper), resample(lower)
    return np.vstack([up[::-1], lo[1:-1]])             # TE -> upper -> LE -> lower


@pytest.mark.parametrize("code", CODES)
def test_analytic_recovery(code):
    f = section_features(naca_coordinates(code))
    t_nominal = int(code[-2:]) / 100
    assert abs(f["t_max"] - t_nominal) < 1e-3
    assert abs(f["x_tmax"] - 0.30) < 0.01
    assert f["thickness"].shape == (101,) and f["camber"].shape == (101,)


@pytest.mark.parametrize("code", CODES)
def test_features_survive_sparse_sampling(code):
    dense = section_features(naca_coordinates(code, n=4001))
    sparse = section_features(resample_like_mesh(naca_coordinates(code, n=4001), 150))
    for key in FEATURES:
        assert abs(dense[key] - sparse[key]) < 2e-3, (key, dense[key], sparse[key])


@pytest.mark.parametrize("code", CODES)
def test_features_invariant_to_rotation_scale_translation(code):
    xy = naca_coordinates(code, n=401)
    ang = np.radians(7.0)
    rot = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    moved = (xy @ rot.T) * 2.5 + np.array([3.0, -1.0])
    a, b = section_features(xy), section_features(moved)
    for key in FEATURES:
        assert abs(a[key] - b[key]) < 1e-6, key


def test_loop_orientation_does_not_matter():
    xy = naca_coordinates("4415", n=401)
    a, b = section_features(xy), section_features(xy[::-1])
    for key in FEATURES:
        assert abs(a[key] - b[key]) < 1e-9, key


def test_symmetric_section_has_no_camber():
    f = section_features(naca_coordinates("0012"))
    assert abs(f["m_max"]) < 1e-4 and f["x_m"] == 0.0


@pytest.mark.parametrize("code,x_m", [("23012", 0.15), ("24012", 0.20), ("25012", 0.25),
                                      ("2412", 0.40), ("23112", 0.15)])
def test_camber_location(code, x_m):
    # Max-camber location is recovered; the magnitude is measured from the
    # geometric chord (see section_features), so it is only checked for sign.
    f = section_features(naca_coordinates(code))
    assert abs(f["x_m"] - x_m) < 0.03
    assert f["m_max"] > 0


def test_standard_5digit_constants_match_table():
    # Abbott & von Doenhoff, standard 5-digit mean line at C_l = 0.3.
    table = {0.10: (0.1260, 51.64), 0.15: (0.2025, 15.957),
             0.20: (0.2900, 6.643), 0.25: (0.3910, 3.230)}
    for p, (r_tab, k1_tab) in table.items():
        r, k1 = _standard_5digit_constants(p, 0.3)
        assert abs(r - r_tab) < 1e-3
        assert abs(k1 - k1_tab) / k1_tab < 0.01


@pytest.mark.parametrize("bad", ["12", "123456", "abcd", "23212", "2012", "0212"])
def test_parse_naca_rejects_bad_codes(bad):
    with pytest.raises(ValueError):
        parse_naca(bad)


def test_parse_naca_accepts_prefix():
    assert parse_naca("NACA 2412")["code"] == "2412"
