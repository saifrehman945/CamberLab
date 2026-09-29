"""
Module: scripts/airfrans/reference.py
Purpose: Readers for the NACA 0012 reference data in validation_data/common
         (Ladson / Gregory experiments, NASA TMR CFL3D SST and SA solutions),
         used by the QA stage and the surface-curve evaluation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
REF_DIR = PROJECT_ROOT / "validation_data" / "common"


def read_tecplot_zones(path: Path) -> dict[str, np.ndarray]:
    """Numeric rows of a simple Tecplot ASCII file, keyed by zone title ("default" before any zone)."""
    zones: dict[str, list] = {}
    current = "default"
    for line in path.read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("#") or s.lower().startswith("variables"):
            continue
        if s.lower().startswith("zone"):
            current = s.split("=", 1)[1].strip().strip('"') if "=" in s else f"zone{len(zones)}"
            continue
        zones.setdefault(current, []).append([float(v) for v in s.split()])
    return {k: np.asarray(v) for k, v in zones.items()}
