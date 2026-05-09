#!/usr/bin/env python3
"""
Script: 02_geometry.py
Stage:  2 — Geometry generation (NACA 4-digit, symmetric)
Purpose: Read samples.csv and produce, for each design point, a closed-polygon
         aerofoil coordinate file and a params.json under cases/case_{i:04d}/.

Usage:
    micromamba run -n openfoam python scripts/02_geometry.py
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLES_PATH = PROJECT_ROOT / "samples.csv"
CASES_DIR = PROJECT_ROOT / "cases"

N_POINTS = 321


def naca_thickness(x: np.ndarray, t: float) -> np.ndarray:
    """NACA 4-digit half-thickness distribution (closed trailing edge)."""
    return 5.0 * t * (
        0.2969 * np.sqrt(x)
        - 0.1260 * x
        - 0.3516 * x ** 2
        + 0.2843 * x ** 3
        - 0.1036 * x ** 4
    )


def aerofoil_polygon(thickness: float, n: int = N_POINTS) -> np.ndarray:
    """Return a closed-polygon (x, y) array for a symmetric NACA 4-digit section.

    Ordering: upper surface TE→LE, then lower surface LE→TE. The leading-edge
    point (0, 0) is shared once between the two halves.
    """
    x = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, n)))
    y_t = naca_thickness(x, thickness)

    upper = np.column_stack((x[::-1], y_t[::-1]))           # TE → LE  (n points)
    lower = np.column_stack((x[1:], -y_t[1:]))              # LE+1 → TE (n-1 points)
    return np.vstack((upper, lower)).astype(np.float64)


def write_aerofoil_dat(path: Path, coords: np.ndarray) -> None:
    lines = [f"{x:.8f} {y:.8f}" for x, y in coords]
    path.write_text("\n".join(lines) + "\n")


def write_params_json(path: Path, alpha_deg: float, Re: float, thickness: float) -> None:
    payload = {
        "alpha_deg": float(alpha_deg),
        "Re": float(Re),
        "thickness": float(thickness),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")


def main() -> None:
    if not SAMPLES_PATH.exists():
        raise FileNotFoundError(f"{SAMPLES_PATH} not found — run 01_doe.py first")

    df = pd.read_csv(
        SAMPLES_PATH,
        dtype={"alpha_deg": np.float64, "Re": np.float64, "thickness": np.float64},
    )
    log.info("Loaded %d design points from %s", len(df), SAMPLES_PATH.relative_to(PROJECT_ROOT))

    CASES_DIR.mkdir(exist_ok=True)

    for i, row in df.iterrows():
        case_dir = CASES_DIR / f"case_{i:04d}"
        case_dir.mkdir(exist_ok=True)

        coords = aerofoil_polygon(row["thickness"], N_POINTS)
        write_aerofoil_dat(case_dir / "aerofoil.dat", coords)
        write_params_json(
            case_dir / "params.json",
            alpha_deg=row["alpha_deg"],
            Re=row["Re"],
            thickness=row["thickness"],
        )

        if i % 20 == 0 or i == len(df) - 1:
            log.info(
                "case_%04d  α=%6.2f°  Re=%.2e  t=%.4f  → %d points",
                i, row["alpha_deg"], row["Re"], row["thickness"], len(coords),
            )

    log.info("Wrote %d cases under %s/", len(df), CASES_DIR.relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    main()
