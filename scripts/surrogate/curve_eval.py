"""
Module: scripts/surrogate/curve_eval.py
Purpose: Metrics for predicted wall Cp / Cf curves, as pure functions of
         arrays, so 10_global_validation.py (the only stage that reads test
         indices) stays an orchestrator.

- curve_metrics:   RMSE, per-station R², suction-peak and separation errors,
                   ±2σ coverage
- NativeWall:      the native wall nodes written by 20_ingest_airfrans.py
- paper_surface_errors: AirfRANS' mean_rel_p / mean_rel_wss on native nodes
- integrated_coefficients: C_L / C_D integrated from predicted curves
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.metrics import r2_score

from scripts.airfrans.surface import integrate_forces
from scripts.surrogate.curves import split_sides
from scripts.surrogate.data import RESULTS_DIR

NATIVE_PATH = RESULTS_DIR / "airfrans_wall_native.npz"
SEPARATION_WINDOW = (0.02, 0.98)   # x/c: ignore the stagnation region and the TE node


def separation_xc(cf_upper: np.ndarray, x_grid: np.ndarray) -> np.ndarray:
    """First x/c on the upper surface with Cf < 0 (inside SEPARATION_WINDOW), NaN if attached."""
    inside = (x_grid >= SEPARATION_WINDOW[0]) & (x_grid <= SEPARATION_WINDOW[1])
    neg = (cf_upper < 0) & inside
    first = np.argmax(neg, axis=1)
    return np.where(neg.any(axis=1), x_grid[first], np.nan)


def curve_metrics(quantity: str, true: np.ndarray, pred: np.ndarray, std: np.ndarray | None,
                  x_grid: np.ndarray) -> dict:
    """Metrics for (n, 202) true / predicted curves of one quantity."""
    out = {"RMSE": float(np.sqrt(np.mean((pred - true) ** 2))),
           "MAE": float(np.mean(np.abs(pred - true))),
           "R2_station_mean": float(r2_score(true, pred, multioutput="uniform_average")),
           "R2_flat": float(r2_score(true.ravel(), pred.ravel()))}
    if std is not None:
        out["coverage_2sigma"] = float(np.mean(np.abs(true - pred) <= 2 * std))
    if quantity == "Cp":
        out["suction_peak_MAE"] = float(np.mean(np.abs(pred.min(axis=1) - true.min(axis=1))))
    if quantity == "Cf":
        s_true = separation_xc(split_sides(true)["upper"], x_grid)
        s_pred = separation_xc(split_sides(pred)["upper"], x_grid)
        sep_true, sep_pred = ~np.isnan(s_true), ~np.isnan(s_pred)
        both = sep_true & sep_pred
        out["n_separated_true"] = int(sep_true.sum())
        out["separation_detect_accuracy"] = float(np.mean(sep_true == sep_pred))
        out["separation_xc_MAE"] = float(np.mean(np.abs(s_pred[both] - s_true[both]))) if both.any() else np.nan
    return out


class NativeWall:
    """Ragged native wall data (one block of nodes per sample), from 20_ingest_airfrans.py."""

    def __init__(self, path: Path = NATIVE_PATH):
        if not path.exists():
            raise FileNotFoundError(f"{path} not found — run scripts/20_ingest_airfrans.py first.")
        with np.load(path) as z:
            self._d = {k: z[k] for k in z.files}
        assert (self._d["sample_id"] == np.arange(len(self._d["sample_id"]))).all()

    def __getitem__(self, sample_id: int) -> dict[str, np.ndarray]:
        a, b = self._d["offsets"][sample_id], self._d["offsets"][sample_id + 1]
        return {k: self._d[k][a:b] for k in ("xy", "x_c", "upper", "p", "tau", "tangent")}


def _on_native(curve: np.ndarray, x_grid: np.ndarray, node: dict) -> np.ndarray:
    """A (202,) upper+lower curve interpolated to the native wall nodes' x/c on their own surface."""
    sides = split_sides(curve[None])
    out = np.empty(len(node["x_c"]))
    for name, mask in (("upper", node["upper"]), ("lower", ~node["upper"])):
        out[mask] = np.interp(node["x_c"][mask], x_grid, sides[name][0])
    return out


def paper_surface_errors(native: NativeWall, sample_ids, q_inf: np.ndarray, x_grid: np.ndarray,
                         cp: np.ndarray, cf: np.ndarray | None) -> dict:
    """AirfRANS' mean_rel_p and mean_rel_wss (per component) over the given test samples.

    Per sample: mean over native wall nodes of |(true − pred)/true| (metrics.py
    rel_err), then averaged over samples, as in the paper's Results_test.
    Predicted p = Cp·q∞ and τ = Cf·q∞·t, with t the true wall tangent. The
    ratio blows up where p or τ is near zero, so it is reported for
    comparability only.
    """
    rel_p, rel_wss = [], []
    for k, sid in enumerate(sample_ids):
        node = native[int(sid)]
        p_pred = _on_native(cp[k], x_grid, node) * q_inf[k]
        rel_p.append(np.mean(np.abs((node["p"] - p_pred) / node["p"])))
        if cf is not None:
            tau_pred = (_on_native(cf[k], x_grid, node) * q_inf[k])[:, None] * node["tangent"]
            rel_wss.append(np.mean(np.abs((node["tau"] - tau_pred) / node["tau"]), axis=0))
    out = {"mean_rel_p": float(np.mean(rel_p))}
    if cf is not None:
        out["mean_rel_wss_x"], out["mean_rel_wss_y"] = (float(v) for v in np.mean(rel_wss, axis=0))
    return out


def integrated_coefficients(native: NativeWall, sample_ids, alpha_rad: np.ndarray, u_inf: np.ndarray,
                            x_grid: np.ndarray, cp: np.ndarray, cf: np.ndarray | None) -> np.ndarray:
    """(n, 2) [C_D, C_L] integrated from predicted curves on each sample's native wall (no shear if cf is None)."""
    out = np.empty((len(sample_ids), 2))
    for k, sid in enumerate(sample_ids):
        node = native[int(sid)]
        q = 0.5 * u_inf[k] ** 2
        p = _on_native(cp[k], x_grid, node) * q
        tau = (np.zeros_like(node["tangent"]) if cf is None
               else (_on_native(cf[k], x_grid, node) * q)[:, None] * node["tangent"])
        f = integrate_forces(node["xy"], p, tau, alpha_rad[k], u_inf[k])
        out[k] = [f["Cd_int_p"] + f["Cd_int_tau"], f["Cl_int_p"] + f["Cl_int_tau"]]
    return out
