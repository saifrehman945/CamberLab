#!/usr/bin/env python3
"""
Module: scripts/validation/compare.py
Purpose: Reusable functions for computing Cl/Cd error metrics and Cp deltas
         between a CFD case and one or more reference datasets.

All functions return plain Python dicts or pandas DataFrames so the orchestrator
can write them to JSON and CSV without further conversion.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.validation.parsers import (
    load_force_coeffs,
    load_ladson_clcd,
    load_naca4412_cp,
    load_tmr_clcd,
    load_tmr_cp,
    load_xfoil_cp,
    load_xfoil_polar,
)

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
VALIDATION_DATA_DIR = PROJECT_ROOT / "validation_data"


# ---------------------------------------------------------------------------
# Error metric helpers
# ---------------------------------------------------------------------------

def percent_error(value: float, reference: float) -> float:
    if reference == 0.0 or np.isnan(reference):
        return float("nan")
    return 100.0 * (value - reference) / reference


def abs_pct_error(value: float, reference: float) -> float:
    return abs(percent_error(value, reference))


# ---------------------------------------------------------------------------
# Cl / Cd comparison
# ---------------------------------------------------------------------------

def compare_clcd(
    case_dir: Path | str,
    reference_case: dict,
    window: int = 200,
) -> dict:
    """Compare a single CFD case's force coefficients against the metadata-declared references.

    Parameters
    ----------
    case_dir : Path
        OpenFOAM case directory containing postProcessing/forceCoeffs/0/.
    reference_case : dict
        One element of `metadata.json["validation_cases"]`.
    window : int
        Number of trailing iterations to average for Cl/Cd.

    Returns
    -------
    dict with keys: case_id, alpha_deg, Re, Cl_cfd, Cl_std, Cd_cfd, Cd_std,
                    references=[{label, Cl_ref, Cd_ref, Cl_err_pct, Cd_err_pct}, ...],
                    converged (bool), tolerances_met (bool).
    """
    coeffs = load_force_coeffs(case_dir, window=window)

    out = {
        "case_id":    reference_case["case_id"],
        "airfoil":    reference_case["airfoil"],
        "alpha_deg":  float(reference_case["alpha_deg"]),
        "Re":         float(reference_case["Re"]),
        "M":          float(reference_case.get("Mach", float("nan"))),
        "Cl_cfd":     coeffs["Cl_mean"],
        "Cl_std":     coeffs["Cl_std"],
        "Cd_cfd":     coeffs["Cd_mean"],
        "Cd_std":     coeffs["Cd_std"],
        "n_iters":    coeffs["n_iters"],
        "references": [],
    }

    refs = reference_case.get("references", {})
    for role, entry in refs.items():
        cl_ref = entry.get("Cl")
        cd_ref = entry.get("Cd")
        if cl_ref is None and cd_ref is None and entry.get("tool") == "xfoil":
            # XFOIL engineering reference: Cl/Cd live in a polar summary CSV,
            # matched by angle of attack rather than declared inline.
            xf = _xfoil_clcd_from_polar(entry, out["alpha_deg"])
            if xf is not None:
                cl_ref, cd_ref = xf
        if cl_ref is None and cd_ref is None:
            continue
        out["references"].append({
            "role":        role,
            "label":       entry.get("label", role),
            "Cl_ref":      cl_ref,
            "Cd_ref":      cd_ref,
            "Cl_err_pct":  percent_error(coeffs["Cl_mean"], cl_ref) if cl_ref is not None else None,
            "Cd_err_pct":  percent_error(coeffs["Cd_mean"], cd_ref) if cd_ref is not None else None,
        })

    return out


def evaluate_tolerances(comparison: dict, tolerances: dict) -> dict:
    """Apply per-regime tolerances to a Cl/Cd comparison and return a pass/fail dict."""
    cl_tol = tolerances.get("delta_Cl_pct", 5.0)
    cd_tol = tolerances.get("delta_Cd_pct", 10.0)

    # The 'experimental_primary' reference governs the pass/fail decision; others are informational.
    primary = next(
        (r for r in comparison["references"] if r["role"] == "experimental_primary"),
        None,
    ) or next(iter(comparison["references"]), None)

    if primary is None:
        return {"pass": False, "reason": "no reference values available"}

    cl_err = primary.get("Cl_err_pct")
    cd_err = primary.get("Cd_err_pct")
    cl_ok = cl_err is None or abs(cl_err) <= cl_tol
    cd_ok = cd_err is None or abs(cd_err) <= cd_tol
    return {
        "pass":     bool(cl_ok and cd_ok),
        "primary_label":  primary["label"],
        "Cl_err_pct":     cl_err,
        "Cd_err_pct":     cd_err,
        "Cl_tol_pct":     cl_tol,
        "Cd_tol_pct":     cd_tol,
        "reason":         "" if (cl_ok and cd_ok)
                          else f"Cl_err={cl_err}, tol={cl_tol}; Cd_err={cd_err}, tol={cd_tol}",
    }


# ---------------------------------------------------------------------------
# Cp comparison
# ---------------------------------------------------------------------------

def _latest_time_dir(parent: Path) -> Path | None:
    """Return the numerically-largest time-step subdirectory of `parent`."""
    def time_key(d: Path) -> float:
        try:
            return float(d.name)
        except ValueError:
            return -1.0

    time_dirs = sorted((d for d in parent.glob("*") if d.is_dir()), key=time_key)
    return time_dirs[-1] if time_dirs else None


def load_openfoam_cp(case_dir: Path | str) -> pd.DataFrame:
    """Load the airfoil-surface Cp distribution sampled by OpenFOAM.

    Primary source is the `aerofoilSamples` surfaces function object
    (raw format), which writes
        case_dir/postProcessing/aerofoilSamples/<lastTime>/aerofoil.xy
    with columns: face_x, face_y, face_z, p  (p kinematic, m²/s²). A legacy
    `singleGraph` line sample (`*_p.xy`, 2 columns x, p) is used as a fallback.

    Cp = (p - p_inf) / (0.5 * U_inf²), with the freestream speed taken from
    case_dir/params.json (U_inf = Re·ν, chord = 1). Density factors out of the
    incompressible kinematic form. Returns columns: x_c, Cp, U_inf, source.
    """
    case = Path(case_dir)
    import json

    params = json.loads((case / "params.json").read_text(encoding="utf-8"))
    Re = float(params["Re"])
    nu = float(params.get("nu", 1.5e-5))
    U_inf = Re * nu / 1.0  # chord = 1
    p_inf = 0.0            # kinematic pressure at far field is typically 0
    q_inf = 0.5 * U_inf ** 2  # density factors out in incompressible kinematic form

    # --- Primary: surfaces (aerofoilSamples) raw output -------------------
    samp_dir = case / "postProcessing" / "aerofoilSamples"
    if samp_dir.exists():
        last = _latest_time_dir(samp_dir)
        xy = next(iter(last.glob("*.xy")), None) if last else None
        if xy is not None:
            raw = pd.read_csv(xy, sep=r"\s+", comment="#", header=None,
                              engine="python")
            # raw columns are face_x, face_y, face_z, p — x is first, p is last.
            df = pd.DataFrame({"x_c": raw.iloc[:, 0].astype(np.float64)})
            df["Cp"] = (raw.iloc[:, -1].astype(np.float64) - p_inf) / q_inf
            df["U_inf"] = U_inf
            df["source"] = f"OpenFOAM ({case.name})"
            return df[["x_c", "Cp", "U_inf", "source"]]

    # --- Fallback: legacy singleGraph line sample -------------------------
    sg_dir = case / "postProcessing" / "singleGraph"
    if not sg_dir.exists():
        raise FileNotFoundError(
            f"no aerofoilSamples or singleGraph Cp output found in {case}"
        )
    last = _latest_time_dir(sg_dir)
    if last is None:
        raise FileNotFoundError(f"no time-step output under {sg_dir}")
    candidates = list(last.glob("*_p.xy")) + list(last.glob("*p_*.xy"))
    if not candidates:
        raise FileNotFoundError(f"no *_p.xy or *p_*.xy file under {last}")

    df = pd.read_csv(candidates[0], sep=r"\s+", comment="#", header=None,
                     names=["x_c", "p"], engine="python")
    df["Cp"] = (df["p"] - p_inf) / q_inf
    df["U_inf"] = U_inf
    df["source"] = f"OpenFOAM ({case.name})"
    return df[["x_c", "Cp", "U_inf", "source"]]


def compare_cp(
    case_dir: Path | str,
    reference_case: dict,
    sample_points: int = 50,
) -> dict | None:
    """Resample CFD and reference Cp onto a common x/c grid and report errors.

    Returns None if neither side has Cp data.
    """
    refs = reference_case.get("references", {})
    cp_ref_df = _load_reference_cp(refs)
    if cp_ref_df is None or cp_ref_df.empty:
        return None

    try:
        cp_cfd_df = load_openfoam_cp(case_dir)
    except FileNotFoundError as exc:
        log.warning("Cp not available for %s (%s)", reference_case["case_id"], exc)
        return None

    grid = np.linspace(0.02, 0.98, sample_points)
    cp_ref = _interp_cp(cp_ref_df, grid)
    cp_cfd = _interp_cp(cp_cfd_df, grid)
    if cp_ref is None or cp_cfd is None:
        return None

    delta = cp_cfd - cp_ref
    return {
        "case_id":    reference_case["case_id"],
        "x_c":        grid.tolist(),
        "Cp_cfd":     cp_cfd.tolist(),
        "Cp_ref":     cp_ref.tolist(),
        "delta_Cp":   delta.tolist(),
        "rms_delta_Cp":  float(np.sqrt(np.mean(delta ** 2))),
        "max_abs_delta_Cp": float(np.max(np.abs(delta))),
    }


def _resolve_xfoil_polar(entry: dict) -> Path | None:
    """Find the XFOIL polar CSV for an engineering_reference entry.

    Prefers the explicit `expected_summary_file`; otherwise looks for a
    `*_polar.csv` sibling of the entry's `expected_file` (the Cp dump). All
    AoAs for one Reynolds number share a single polar, so a sibling glob is a
    safe fallback for entries that only declare a Cp file.
    """
    summary = entry.get("expected_summary_file")
    if summary:
        return VALIDATION_DATA_DIR / summary
    cp_file = entry.get("expected_file")
    if cp_file:
        cp_dir = (VALIDATION_DATA_DIR / cp_file).parent
        polars = sorted(cp_dir.glob("*_polar.csv"))
        if polars:
            return polars[0]
    return None


def _xfoil_clcd_from_polar(entry: dict, alpha_deg: float) -> tuple[float, float] | None:
    """Look up (Cl, Cd) for `alpha_deg` in an XFOIL polar CSV. None if unavailable."""
    polar = _resolve_xfoil_polar(entry)
    if polar is None or not polar.exists():
        return None
    df = load_xfoil_polar(polar)
    match = df[np.isclose(df["alpha_deg"], alpha_deg, atol=0.05)]
    if match.empty:
        log.warning("XFOIL polar %s has no row for α=%.3f°", polar.name, alpha_deg)
        return None
    row = match.iloc[0]
    return float(row["Cl"]), float(row["Cd"])


def _load_reference_cp(refs: dict) -> pd.DataFrame | None:
    """Locate and load the Cp reference declared in a metadata.json entry."""
    eng_ref = refs.get("engineering_reference") or {}
    if eng_ref.get("tool") == "xfoil" and "expected_file" in eng_ref:
        path = VALIDATION_DATA_DIR / eng_ref["expected_file"]
        if path.exists():
            return load_xfoil_cp(path, eng_ref.get("label", "XFOIL"))

    cfd_ref = refs.get("cfd_reference") or {}
    if "file_cp" in cfd_ref:
        path = VALIDATION_DATA_DIR / cfd_ref["file_cp"]
        if path.exists():
            df = load_tmr_cp(path, cfd_ref.get("label", "TMR CFD"))
            zone = cfd_ref.get("cp_zone")
            if zone:
                m = _zone_filter(df, zone)
                if not m.empty:
                    return m
            return df

    exp = refs.get("experimental_primary") or {}
    if "file_cp" in exp:
        path = VALIDATION_DATA_DIR / exp["file_cp"]
        if path.exists():
            if "naca4412" in str(path).lower():
                return load_naca4412_cp(path)
            return load_tmr_cp(path, exp.get("label", "experimental"))
    return None


def _zone_filter(df: pd.DataFrame, zone_descriptor: str) -> pd.DataFrame:
    """Pick rows in a Cp DataFrame whose `alpha_deg` matches the zone label."""
    import re
    m = re.search(r"alpha\s*=\s*(-?\d+(?:\.\d+)?)", zone_descriptor)
    if not m:
        return df
    target = float(m.group(1))
    return df[np.isclose(df["alpha_deg"], target, atol=0.5)]


def _interp_cp(df: pd.DataFrame, grid: np.ndarray) -> np.ndarray | None:
    if df.empty:
        return None
    x = df["x_c"].to_numpy(dtype=np.float64)
    y = df["Cp"].to_numpy(dtype=np.float64)
    order = np.argsort(x)
    x, y = x[order], y[order]
    mask = (x >= 0.0) & (x <= 1.0)
    if mask.sum() < 5:
        return None
    return np.interp(grid, x[mask], y[mask])


# ---------------------------------------------------------------------------
# Aggregate per-regime
# ---------------------------------------------------------------------------

def build_summary_dataframe(comparisons: list[dict]) -> pd.DataFrame:
    """Flatten the per-case comparison dicts into a tidy DataFrame."""
    rows = []
    for c in comparisons:
        base = {
            "case_id":   c["case_id"],
            "airfoil":   c["airfoil"],
            "alpha_deg": c["alpha_deg"],
            "Re":        c["Re"],
            "Cl_cfd":    c["Cl_cfd"],
            "Cd_cfd":    c["Cd_cfd"],
            "Cl_std":    c.get("Cl_std"),
            "Cd_std":    c.get("Cd_std"),
            "n_iters":   c.get("n_iters"),
        }
        for ref in c.get("references", []):
            row = dict(base)
            row.update({
                "ref_role":     ref["role"],
                "ref_label":    ref["label"],
                "Cl_ref":       ref.get("Cl_ref"),
                "Cd_ref":       ref.get("Cd_ref"),
                "Cl_err_pct":   ref.get("Cl_err_pct"),
                "Cd_err_pct":   ref.get("Cd_err_pct"),
            })
            rows.append(row)
        if not c.get("references"):
            rows.append(base)
    return pd.DataFrame(rows)
