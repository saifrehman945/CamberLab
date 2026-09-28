#!/usr/bin/env python3
"""
Script: 20_ingest_airfrans.py
Stage:  AirfRANS Phase 3 — ingest to a tabular dataset
Purpose: Turn every AirfRANS_remeshed sample into one row of
         results/airfrans_dataset.csv (inputs, geometry features, C_L, C_D)
         and save the 101-point surfaces to results/airfrans_surfaces.npz.

Nothing is dropped or edited here; data quality is Phase 4
(scripts/21_qa_airfrans.py). Exits non-zero if the Gate 3 checks fail.

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
from scripts.airfrans.geometry import extract_surface, section_features  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(message)s")
log = logging.getLogger(__name__)

RESULTS_DIR = PROJECT_ROOT / "results"
DATASET_PATH = RESULTS_DIR / "airfrans_dataset.csv"
SURFACES_PATH = RESULTS_DIR / "airfrans_surfaces.npz"

COLUMNS = ["sample_id", "source_name", "alpha_deg", "U_inf", "Re", "log10_Re",
           "t_max", "x_tmax", "m_max", "x_m", "Cl", "Cd", "n_wall_nodes"]


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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default=None)
    args = ap.parse_args()

    data_dir = io.resolve_data_dir(args.data_dir)
    n = len(io.RawSampleReader(data_dir))
    log.info("Ingesting %d AirfRANS samples from %s", n, data_dir)

    rows, uppers, lowers, grid = [], [], [], None
    for d in tqdm(io.iter_samples(data_dir), total=n, desc="ingesting"):
        wall = extract_surface(d)
        feats = section_features(wall)
        u_inf = d.scalars[io.SCALAR_UINF]
        re = float(io.reynolds(u_inf))
        rows.append({
            "sample_id": d.index,
            "source_name": "",   # AirfRANS_remeshed samples carry no simulation name (Phase 2)
            "alpha_deg": float(np.degrees(d.scalars[io.SCALAR_AOA])),   # stored in radians
            "U_inf": u_inf,
            "Re": re,
            "log10_Re": float(np.log10(re)),
            "t_max": feats["t_max"], "x_tmax": feats["x_tmax"],
            "m_max": feats["m_max"], "x_m": feats["x_m"],
            "Cl": d.scalars[io.SCALAR_CL],
            "Cd": d.scalars[io.SCALAR_CD],
            "n_wall_nodes": len(wall),
        })
        uppers.append(feats["y_upper"])
        lowers.append(feats["y_lower"])
        grid = feats["x_grid"]

    df = pd.DataFrame(rows, columns=COLUMNS)
    df["sample_id"] = df["sample_id"].astype(np.int32)
    RESULTS_DIR.mkdir(exist_ok=True)
    df.to_csv(DATASET_PATH, index=False, float_format="%.10g")
    np.savez_compressed(SURFACES_PATH, sample_id=df["sample_id"].to_numpy(),
                        x_grid=grid, y_upper=np.vstack(uppers), y_lower=np.vstack(lowers))
    log.info("Wrote %s (%d rows) and %s", DATASET_PATH.relative_to(PROJECT_ROOT), len(df),
             SURFACES_PATH.relative_to(PROJECT_ROOT))

    for col in ["alpha_deg", "Re", "t_max", "x_tmax", "m_max", "x_m", "Cl", "Cd", "n_wall_nodes"]:
        log.info("  %-12s min %.5g  max %.5g", col, df[col].min(), df[col].max())

    passed = True
    for label, ok in gate3(df, n):
        (log.info if ok else log.error)("Gate 3 — %s: %s", label, "PASS" if ok else "FAIL")
        passed &= bool(ok)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
