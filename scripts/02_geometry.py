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

N_POINTS = 200

STL_Z_LOWER = -0.5
STL_Z_UPPER = +0.5


def naca_thickness(x: np.ndarray, t: float) -> np.ndarray:
    """NACA 4-digit half-thickness distribution (open trailing edge)."""
    return 5.0 * t * (
        0.2969 * np.sqrt(x)
        - 0.1260 * x
        - 0.3516 * x ** 2
        + 0.2843 * x ** 3
        - 0.1015 * x ** 4
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


def _format_facet(normal: np.ndarray, vertices: np.ndarray) -> str:
    nx, ny, nz = normal
    chunks = [f"  facet normal {nx:.8e} {ny:.8e} {nz:.8e}", "    outer loop"]
    for vx, vy, vz in vertices:
        chunks.append(f"      vertex {vx:.8e} {vy:.8e} {vz:.8e}")
    chunks.append("    endloop")
    chunks.append("  endfacet")
    return "\n".join(chunks)


def _triangle_normal(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    n = np.cross(p1 - p0, p2 - p0)
    norm = np.linalg.norm(n)
    if norm == 0.0:
        return n
    return n / norm


def write_aerofoil_stl(
    path: Path,
    coords: np.ndarray,
    z_lo: float = STL_Z_LOWER,
    z_hi: float = STL_Z_UPPER,
    name: str = "aerofoil",
) -> None:
    """Write an ASCII STL of the aerofoil polygon extruded between z_lo and z_hi.

    The 2D polygon is CCW when viewed from +z (upper surface TE→LE, lower LE→TE,
    closed implicitly from coords[-1] back to coords[0]). Three triangulated
    surfaces are emitted: back cap at z_hi (normal +z), front cap at z_lo
    (normal -z), and a side ribbon (two triangles per polygon edge).
    """
    n = len(coords)
    if n < 3:
        raise ValueError("aerofoil polygon needs at least 3 points to triangulate")

    back = np.column_stack((coords[:, 0], coords[:, 1], np.full(n, z_hi)))
    front = np.column_stack((coords[:, 0], coords[:, 1], np.full(n, z_lo)))

    facets: list[str] = []

    # Back cap at z_hi: fan from index 0; CCW order produces outward normal +z.
    z_plus = np.array([0.0, 0.0, +1.0])
    for i in range(1, n - 1):
        facets.append(_format_facet(z_plus, np.array([back[0], back[i], back[i + 1]])))

    # Front cap at z_lo: same fan with reversed winding so normal is -z.
    z_minus = np.array([0.0, 0.0, -1.0])
    for i in range(1, n - 1):
        facets.append(_format_facet(z_minus, np.array([front[0], front[i + 1], front[i]])))

    # Side ribbon: two triangles per edge (i, (i+1) % n), including the closing
    # trailing-edge segment. Outward normal points away from the airfoil
    # interior, which for a CCW polygon is (dy, -dx, 0) where (dx, dy) = P_{i+1}-P_i.
    for i in range(n):
        j = (i + 1) % n
        p_i_front = front[i]
        p_i_back = back[i]
        p_j_front = front[j]
        p_j_back = back[j]

        normal = _triangle_normal(p_i_front, p_j_front, p_j_back)
        facets.append(_format_facet(normal, np.array([p_i_front, p_j_front, p_j_back])))
        facets.append(_format_facet(normal, np.array([p_i_front, p_j_back, p_i_back])))

    body = "\n".join(facets)
    path.write_text(f"solid {name}\n{body}\nendsolid {name}\n")


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

        geometry_dir = case_dir / "constant" / "geometry"
        geometry_dir.mkdir(parents=True, exist_ok=True)
        write_aerofoil_stl(geometry_dir / "aerofoil.stl", coords)

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
