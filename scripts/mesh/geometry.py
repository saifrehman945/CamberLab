"""
Geometry helpers — pure Python, no gmsh.

Produces:
  - cosine-clustered NACA 4-digit symmetric surface points (upper, lower),
  - the named farfield-point set required by the C+H topology,
  - a back-compat closed-polygon loader for the legacy aerofoil.dat.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np


def cosine_x(n: int) -> np.ndarray:
    """n cosine-spaced x in [0, 1] with denser sampling at both endpoints."""
    return 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, n)))


def naca_thickness(x: np.ndarray, t: float) -> np.ndarray:
    """NACA 4-digit half-thickness distribution (open-TE form)."""
    return 5.0 * t * (
        0.2969 * np.sqrt(x)
        - 0.1260 * x
        - 0.3516 * x ** 2
        + 0.2843 * x ** 3
        - 0.1015 * x ** 4
    )


def naca_symmetric(thickness: float, n: int = 300) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric NACA-4 surface, sharply closed TE.

    Returns
    -------
    upper, lower : (n, 2) float64 arrays, each running LE (0, 0) -> TE (1, 0).
    The TE y-value is forced to 0 so the two surfaces meet at a single point.
    """
    x = cosine_x(n)
    y_t = naca_thickness(x, thickness)
    y_t[-1] = 0.0                  # sharp closed TE
    upper = np.column_stack((x,  y_t)).astype(np.float64)
    lower = np.column_stack((x, -y_t)).astype(np.float64)
    return upper, lower


def closed_polygon(upper: np.ndarray, lower: np.ndarray) -> np.ndarray:
    """Stitch (upper LE->TE) and (lower LE->TE) into a closed polygon
    ordered TE -> LE (upper) -> TE (lower), matching the legacy aerofoil.dat
    convention written by 02_geometry.py.
    """
    up_te_to_le = upper[::-1]                              # TE -> LE
    lo_le_to_te = lower[1:]                                # LE+1 -> TE
    return np.vstack((up_te_to_le, lo_le_to_te)).astype(np.float64)


def load_polygon(path: Path) -> np.ndarray:
    """Load a 2-column aerofoil.dat polygon (back-compat with 02_geometry.py)."""
    coords = np.loadtxt(path, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"{path} must contain two columns of x y coordinates")
    if len(coords) < 10:
        raise ValueError(f"{path} has too few points for a usable spline")
    if np.allclose(coords[0], coords[-1]):
        coords = coords[:-1]
    return coords


@dataclass(frozen=True)
class FarfieldPoints:
    """Named 2-D point set used by the C+H 4-block topology.

    Coordinate conventions:
      chord lies along +x with LE at (0, 0) and TE at (1, 0).
      Upstream radius Rc and transverse half-height Ht are equal so the upstream
      arc is a true semicircle centered on the LE.
    """
    LE_FAR:   tuple[float, float]   # (-Rc, 0)
    TOP_MID:  tuple[float, float]   # (0,  +Ht)
    BOT_MID:  tuple[float, float]   # (0,  -Ht)
    TOP_TE:   tuple[float, float]   # (1,  +Ht)
    BOT_TE:   tuple[float, float]   # (1,  -Ht)
    TOP_OUT:  tuple[float, float]   # (1+Lw, +Ht)
    BOT_OUT:  tuple[float, float]   # (1+Lw, -Ht)
    OUT_MID:  tuple[float, float]   # (1+Lw, 0)

    def as_dict(self) -> dict[str, tuple[float, float]]:
        return asdict(self)


def farfield_points(
    upstream_radius: float,
    downstream_length: float,
    transverse_extent: float,
    chord: float = 1.0,
) -> FarfieldPoints:
    """Build the FarfieldPoints set from chord-multiple extents.

    Note
    ----
    `upstream_radius` and `transverse_extent` are both expressed as multiples
    of chord; for the C+H topology to use a clean semicircular upstream cap
    they must be EQUAL. We do not silently force them equal; we raise if they
    differ so the caller sees the constraint.
    """
    if abs(upstream_radius - transverse_extent) > 1e-9:
        raise ValueError(
            f"Upstream radius ({upstream_radius}) must equal transverse extent "
            f"({transverse_extent}) for the C+H topology's semicircular cap."
        )
    Rc = upstream_radius * chord
    Lw = downstream_length * chord
    Ht = transverse_extent * chord
    return FarfieldPoints(
        LE_FAR  = (-Rc,        0.0),
        TOP_MID = ( 0.0,       +Ht),
        BOT_MID = ( 0.0,       -Ht),
        TOP_TE  = ( chord,     +Ht),
        BOT_TE  = ( chord,     -Ht),
        TOP_OUT = ( chord + Lw, +Ht),
        BOT_OUT = ( chord + Lw, -Ht),
        OUT_MID = ( chord + Lw,  0.0),
    )
