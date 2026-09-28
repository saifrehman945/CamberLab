"""
Module: scripts/surrogate/inference.py
Purpose: Query the persisted AirfRANS surrogates at (α, Re, NACA code).

Geometry features come from the same functions used at ingestion
(scripts.airfrans.geometry.naca_coordinates + section_features), so query
features are computed exactly like training features. Queries outside the
training envelope still get a prediction, always with a warning — never
silently, never refused.
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

from scripts.airfrans.geometry import naca_coordinates, parse_naca, section_features
from scripts.surrogate.data import FEATURES, MODELS_DIR, TASKS
from scripts.surrogate.models import FAMILIES, predict as model_predict

log = logging.getLogger(__name__)

UNCERTAINTY_FAMILIES = {"gp", "krg"}   # families with a posterior standard deviation
NEAR_STALL_ALPHA = 12.0                # steady RANS is least reliable above this
# Slack on the train min/max before a query counts as out of envelope, so the nominal
# AirfRANS edges (α −5°/15°, Re 2e6/6e6) do not trip warnings a hair outside the samples.
ALPHA_TOL_DEG = 0.1
RE_TOL_REL = 0.02


@lru_cache(maxsize=256)
def naca_features(naca: str) -> dict:
    """Scalar geometry features of an analytic NACA section (cached per code)."""
    f = section_features(naca_coordinates(naca))
    return {k: float(f[k]) for k in ("t_max", "x_tmax", "m_max", "x_m")}


@lru_cache(maxsize=None)
def _load(task: str, name: str):
    return joblib.load(MODELS_DIR / task / f"{name}.joblib")


@lru_cache(maxsize=None)
def load_envelope(task: str = "full") -> dict:
    return json.loads((MODELS_DIR / task / "envelope.json").read_text())


def _check(task: str, family: str) -> None:
    if task not in TASKS:
        raise ValueError(f"unknown task {task!r}; expected one of {TASKS}")
    if family not in FAMILIES:
        raise ValueError(f"unknown family {family!r}; expected one of {FAMILIES}")


def feature_frame(alpha_deg, Re: float, naca: str) -> pd.DataFrame:
    alpha = np.atleast_1d(np.asarray(alpha_deg, dtype=np.float64))
    geo = naca_features(parse_naca(naca)["code"])
    return pd.DataFrame({"alpha_deg": alpha, "log10_Re": np.log10(Re), **geo})[FEATURES]


def envelope_warnings(X: pd.DataFrame, Re: float, naca: str, task: str) -> list[str]:
    env = load_envelope(task)["envelope"]
    warnings = []
    lo, hi = env["Re"]
    if not lo * (1 - RE_TOL_REL) <= Re <= hi * (1 + RE_TOL_REL):
        warnings.append(f"Re = {Re:.3g} is outside the training range [{lo:.3g}, {hi:.3g}]")
    a_lo, a_hi = env["alpha_deg"]
    a = X["alpha_deg"]
    if (a < a_lo - ALPHA_TOL_DEG).any() or (a > a_hi + ALPHA_TOL_DEG).any():
        warnings.append(f"α outside the training range [{a_lo:.2f}°, {a_hi:.2f}°]")
    for f in ("t_max", "x_tmax", "m_max", "x_m"):
        v = float(X[f].iloc[0])
        f_lo, f_hi = env[f]
        if not f_lo <= v <= f_hi:
            warnings.append(f"NACA {naca}: {f} = {v:.4f} is outside the training range "
                            f"[{f_lo:.4f}, {f_hi:.4f}]")
    if (a > NEAR_STALL_ALPHA).any():
        warnings.append(f"α > {NEAR_STALL_ALPHA:g}°: steady RANS near stall is the least reliable "
                        "part of the training data")
    return warnings


def predict_curve(alpha_deg, Re: float, naca: str, family: str = "gp", task: str = "full") -> pd.DataFrame:
    """Vectorised prediction over an array of α at fixed Re and section.

    Columns: alpha_deg, Cl, Cd, L_over_D, Cl_std, Cd_std, and for families with
    a posterior σ also Cl_lo/Cl_hi and Cd_lo/Cd_hi (±2σ; the Cd band is
    exp(log Cd ± 2σ), so it is asymmetric). Cd_std is the delta-method
    Cd·σ(log Cd). Attribute `warnings` lists envelope issues.
    """
    _check(task, family)
    X_df = feature_frame(alpha_deg, Re, naca)
    X = _load(task, "preprocessor").transform(X_df)
    out = pd.DataFrame({"alpha_deg": X_df["alpha_deg"].to_numpy()})
    for target in ("Cl", "Cd"):
        mean, std = model_predict(_load(task, f"{family}_{target}"), family, X, return_std=True)
        if family not in UNCERTAINTY_FAMILIES:
            std = None
        if target == "Cl":
            out["Cl"] = mean
            out["Cl_std"] = std if std is not None else np.nan
            if std is not None:
                out["Cl_lo"], out["Cl_hi"] = mean - 2 * std, mean + 2 * std
        else:
            cd = np.exp(mean)
            out["Cd"] = cd
            out["Cd_std"] = cd * std if std is not None else np.nan
            if std is not None:
                out["Cd_lo"], out["Cd_hi"] = np.exp(mean - 2 * std), np.exp(mean + 2 * std)
    out["L_over_D"] = out["Cl"] / out["Cd"]
    out.attrs["warnings"] = envelope_warnings(X_df, Re, parse_naca(naca)["code"], task)
    return out


def predict(alpha_deg: float, Re: float, naca: str, family: str = "gp", task: str = "full") -> dict:
    """Single-point prediction.

    Returns {Cl, Cd, L_over_D, Cl_std, Cd_std, in_envelope, warnings}. Cl_std
    and Cd_std are None for families without a posterior σ (RF, MLP).
    """
    row = predict_curve([alpha_deg], Re, naca, family, task)
    r = row.iloc[0]
    warnings = row.attrs["warnings"]
    in_envelope = not any("outside" in w for w in warnings)
    for w in warnings:
        log.warning("NACA %s, α=%.2f°, Re=%.3g: %s", naca, alpha_deg, Re, w)
    return {
        "Cl": float(r.Cl), "Cd": float(r.Cd), "L_over_D": float(r.L_over_D),
        "Cl_std": None if np.isnan(r.Cl_std) else float(r.Cl_std),
        "Cd_std": None if np.isnan(r.Cd_std) else float(r.Cd_std),
        "in_envelope": in_envelope, "warnings": warnings,
    }


def predict_all(alpha_deg: float, Re: float, naca: str, task: str = "full") -> dict[str, dict]:
    return {family: predict(alpha_deg, Re, naca, family, task) for family in FAMILIES}
