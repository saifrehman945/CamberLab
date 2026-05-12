#!/usr/bin/env python3
"""
Script: 02_geometry.py
Stage:  2 — Geometry generation (NACA 4-digit, symmetric)
Purpose: Read samples.csv and produce, for each design point, a closed-polygon
         aerofoil coordinate file and a params.json (stamped with regime) under
         cases/case_{i:04d}/.

The surface-point count is taken from REGIME_MESH[regime]["surface_points"] so
that the geometry resolution is consistent with the regime's downstream mesh
strategy.

Usage:
    micromamba run -n openfoam python scripts/02_geometry.py
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from mesh.geometry import naca_symmetric, closed_polygon            # noqa: E402
from mesh.regime_parameters import REGIME_MESH, classify_regime     # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(message)s")
log = logging.getLogger(__name__)

SAMPLES_PATH = PROJECT_ROOT / "samples.csv"
CASES_DIR = PROJECT_ROOT / "cases"


def write_aerofoil_dat(path: Path, coords: np.ndarray) -> None:
    lines = [f"{x:.8f} {y:.8f}" for x, y in coords]
    path.write_text("\n".join(lines) + "\n")


def write_params_json(
    path: Path,
    alpha_deg: float,
    Re: float,
    thickness: float,
    regime: str,
) -> None:
    payload = {
        "alpha_deg": float(alpha_deg),
        "Re":        float(Re),
        "thickness": float(thickness),
        "regime":    regime,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")


def write_surfaces_npz(path: Path, upper: np.ndarray, lower: np.ndarray) -> None:
    """Side-channel storage of upper/lower surfaces (LE -> TE each), consumed
    directly by the structured-C-grid builder without re-parsing aerofoil.dat.
    """
    np.savez(path, upper=upper, lower=lower)


def main() -> None:
    if not SAMPLES_PATH.exists():
        raise FileNotFoundError(f"{SAMPLES_PATH} not found — run 01_doe.py first")

    df = pd.read_csv(
        SAMPLES_PATH,
        dtype={"alpha_deg": np.float64, "Re": np.float64, "thickness": np.float64},
    )
    log.info("Loaded %d design points from %s", len(df), SAMPLES_PATH.relative_to(PROJECT_ROOT))

    CASES_DIR.mkdir(exist_ok=True)

    regime_counts: dict[str, int] = {}
    for i, row in df.iterrows():
        case_dir = CASES_DIR / f"case_{i:04d}"
        case_dir.mkdir(exist_ok=True)

        regime = classify_regime(row["alpha_deg"], row["Re"], row["thickness"])
        regime_counts[regime] = regime_counts.get(regime, 0) + 1
        cfg = REGIME_MESH.get(regime)
        # If a regime is not yet populated, fall back to A's surface count so
        # the geometry stage doesn't block the rest of the pipeline.
        n_points = (cfg or REGIME_MESH["A"])["surface_points"]

        upper, lower = naca_symmetric(row["thickness"], n=n_points)
        coords = closed_polygon(upper, lower)

        write_aerofoil_dat(case_dir / "aerofoil.dat", coords)
        write_surfaces_npz(case_dir / "aerofoil_surfaces.npz", upper, lower)
        write_params_json(
            case_dir / "params.json",
            alpha_deg=row["alpha_deg"],
            Re=row["Re"],
            thickness=row["thickness"],
            regime=regime,
        )

        if i % 20 == 0 or i == len(df) - 1:
            log.info(
                "case_%04d  α=%6.2f°  Re=%.2e  t=%.4f  regime=%s  → %d points",
                i, row["alpha_deg"], row["Re"], row["thickness"], regime, len(coords),
            )

    log.info("Wrote %d cases under %s/", len(df), CASES_DIR.relative_to(PROJECT_ROOT))
    log.info("Regime distribution: %s",
             ", ".join(f"{r}={n}" for r, n in sorted(regime_counts.items())))


if __name__ == "__main__":
    main()
