#!/usr/bin/env python3
"""
Script: 20_ingest_airfrans.py
Stage:  AirfRANS Phase 3 — ingest to a tabular dataset
Purpose: Turn every AirfRANS sample into one row of
         results/airfrans_dataset.csv (inputs, geometry features, C_L, C_D,
         and C_L / C_D re-integrated from the wall), and save the 101-point
         surfaces and the wall Cp / Cf curves in one streaming pass.

Writes:
    results/airfrans_dataset.csv        one row per sample
    results/airfrans_surfaces.npz       y_upper / y_lower on the 101-point grid
    results/airfrans_wall_curves.npz    Cp / Cf, upper and lower, 101 stations each
    results/airfrans_wall_native.npz    native wall nodes (ragged; gitignored)

If a previous airfrans_dataset.csv exists (e.g. from the remeshed variant),
every row's α, U∞, C_L and C_D must match it, since both variants hold the
same simulations. Nothing is dropped or edited here; data quality is
scripts/21_qa_airfrans.py and the wall check is scripts/23_surface_gate.py.
Exits non-zero if the Gate 3 checks fail.

Usage:
    uv run python scripts/20_ingest_airfrans.py [--data-dir DIR]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.airfrans import io  # noqa: E402
from scripts.airfrans.geometry import section_features  # noqa: E402
from scripts.airfrans.surface import curves, forces, wall_data  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(message)s")
log = logging.getLogger(__name__)

RESULTS_DIR = PROJECT_ROOT / "results"
DATASET_PATH = RESULTS_DIR / "airfrans_dataset.csv"
SURFACES_PATH = RESULTS_DIR / "airfrans_surfaces.npz"
CURVES_PATH = RESULTS_DIR / "airfrans_wall_curves.npz"
NATIVE_PATH = RESULTS_DIR / "airfrans_wall_native.npz"

INT_COLUMNS = ["Cl_int_p", "Cl_int_tau", "Cd_int_p", "Cd_int_tau"]
COLUMNS = ["sample_id", "source_name", "alpha_deg", "U_inf", "Re", "log10_Re",
           "t_max", "x_tmax", "m_max", "x_m", "Cl", "Cd", "n_wall_nodes", *INT_COLUMNS]
CURVE_KEYS = ["Cp_upper", "Cp_lower", "Cf_upper", "Cf_lower"]
SAME_SIMULATION = ["alpha_deg", "U_inf", "Cl", "Cd"]   # must match any previous ingest row for row


def gate3(df: pd.DataFrame, n_expected: int) -> list[tuple[str, bool]]:
    num = df.drop(columns=["source_name"])
    return [
        (f"{n_expected} rows", len(df) == n_expected),
        ("no NaNs in numeric columns", not num.isna().any().any()),
        ("t_max in [0.04, 0.22]", df.t_max.between(0.04, 0.22).all()),
        ("|m_max| <= 0.08", (df.m_max.abs() <= 0.08).all()),
        ("alpha_deg in [-5, 15]", df.alpha_deg.between(-5.0, 15.0).all()),
        ("Re in [1.9e6, 6.1e6]", df.Re.between(1.9e6, 6.1e6).all()),
    ]


def compare_with_previous(df: pd.DataFrame, previous: pd.DataFrame) -> bool:
    """Same simulations, row for row: scalars equal (to CSV precision); log geometry drift."""
    if len(previous) != len(df) or not (previous.sample_id.to_numpy() == df.sample_id.to_numpy()).all():
        log.error("previous dataset has different rows (%d vs %d)", len(previous), len(df))
        return False
    ok = True
    for col in SAME_SIMULATION:
        diff = np.abs(df[col].to_numpy() - previous[col].to_numpy())
        tol = 1e-9 * np.abs(previous[col].to_numpy()) + 1e-12
        n_bad = int((diff > tol).sum())
        (log.info if n_bad == 0 else log.error)("previous-ingest check — %s: %d rows differ (max |Δ| %.3g)",
                                                 col, n_bad, diff.max())
        ok &= n_bad == 0
    for col in ["t_max", "x_tmax", "m_max", "x_m"]:
        diff = np.abs(df[col].to_numpy() - previous[col].to_numpy())
        log.info("geometry drift vs previous ingest — %-6s median |Δ| %.2e, max |Δ| %.2e",
                 col, np.median(diff), diff.max())
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default=None)
    args = ap.parse_args()

    data_dir = io.resolve_data_dir(args.data_dir)
    n = len(io.RawSampleReader(data_dir))
    log.info("Ingesting %d AirfRANS samples from %s", n, data_dir)

    previous = pd.read_csv(DATASET_PATH) if DATASET_PATH.exists() else None

    rows, uppers, lowers, grid = [], [], [], None
    curve_rows: dict[str, list[np.ndarray]] = {k: [] for k in CURVE_KEYS}
    native: dict[str, list[np.ndarray]] = {k: [] for k in ("xy", "x_c", "upper", "p", "tau", "tangent")}
    for d in tqdm(io.iter_samples(data_dir, flow=True), total=n, desc="ingesting"):
        w = wall_data(d)
        feats = section_features(w.xy)
        u_inf = d.scalars[io.SCALAR_UINF]
        re = float(io.reynolds(u_inf))
        rows.append({
            "sample_id": d.index,
            "source_name": "",   # PLAID AirfRANS samples carry no simulation name (Phase 2)
            "alpha_deg": float(np.degrees(d.scalars[io.SCALAR_AOA])),   # stored in radians
            "U_inf": u_inf,
            "Re": re,
            "log10_Re": float(np.log10(re)),
            "t_max": feats["t_max"], "x_tmax": feats["x_tmax"],
            "m_max": feats["m_max"], "x_m": feats["x_m"],
            "Cl": d.scalars[io.SCALAR_CL],
            "Cd": d.scalars[io.SCALAR_CD],
            "n_wall_nodes": len(w.xy),
            **forces(w, d),
        })
        uppers.append(feats["y_upper"])
        lowers.append(feats["y_lower"])
        grid = feats["x_grid"]
        for k, v in curves(w).items():
            curve_rows[k].append(v)
        for k in native:
            native[k].append(getattr(w, k))

    df = pd.DataFrame(rows, columns=COLUMNS)
    df["sample_id"] = df["sample_id"].astype(np.int32)
    same_as_previous = True if previous is None else compare_with_previous(df, previous)
    RESULTS_DIR.mkdir(exist_ok=True)
    df.to_csv(DATASET_PATH, index=False, float_format="%.10g")
    sample_id = df["sample_id"].to_numpy()
    np.savez_compressed(SURFACES_PATH, sample_id=sample_id,
                        x_grid=grid, y_upper=np.vstack(uppers), y_lower=np.vstack(lowers))
    np.savez_compressed(CURVES_PATH, sample_id=sample_id, x_grid=grid,
                        **{k: np.vstack(v) for k, v in curve_rows.items()})
    counts = np.array([len(v) for v in native["x_c"]])
    np.savez_compressed(NATIVE_PATH, sample_id=sample_id,
                        offsets=np.concatenate([[0], np.cumsum(counts)]).astype(np.int64),
                        **{k: np.concatenate(v) for k, v in native.items()})
    for path in (DATASET_PATH, SURFACES_PATH, CURVES_PATH, NATIVE_PATH):
        log.info("Wrote %s (%.1f MB)", path.relative_to(PROJECT_ROOT), path.stat().st_size / 1e6)

    for col in ["alpha_deg", "Re", "t_max", "x_tmax", "m_max", "x_m", "Cl", "Cd", "n_wall_nodes"]:
        log.info("  %-12s min %.5g  max %.5g", col, df[col].min(), df[col].max())

    passed = True
    checks = gate3(df, n)
    checks.append(("wall curves finite", all(np.isfinite(np.vstack(v)).all() for v in curve_rows.values())))
    if previous is not None:
        checks.append(("α, U∞, C_L, C_D identical to the previous ingest", same_as_previous))
    for label, ok in checks:
        (log.info if ok else log.error)("Gate 3 — %s: %s", label, "PASS" if ok else "FAIL")
        passed &= bool(ok)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
