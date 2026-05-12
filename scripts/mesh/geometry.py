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


def naca_symmetric(
    thickness: float,
    n: int = 300,
    te_chord_fraction: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric NACA-4 surface with an optional blunt trailing edge.

    Parameters
    ----------
    thickness         : NACA 4-digit max half-thickness ratio (e.g. 0.12).
    n                 : Total surface points per side (LE -> TE inclusive).
    te_chord_fraction : Where to truncate the airfoil, in [chord] units.
                        1.0  -> sharp closed TE (legacy behaviour: y forced to 0
                                at the last point).
                        <1.0 -> blunt TE. The airfoil is sampled on
                                x in [0, te_chord_fraction] and the natural
                                NACA-4 half-thickness at te_chord_fraction is
                                kept as the TE half-thickness (no closure
                                fudge). Use this to avoid the quasi-sharp TE
                                that produces sliver cells in a structured
                                mesh.

    Returns
    -------
    upper, lower : (n, 2) float64 arrays, each running LE (0, 0) -> TE.
                   For a blunt TE, upper[-1] == (te_chord_fraction, +h_te) and
                   lower[-1] == (te_chord_fraction, -h_te) with h_te > 0.
    """
    if not (0.5 < te_chord_fraction <= 1.0):
        raise ValueError(
            f"te_chord_fraction must lie in (0.5, 1.0], got {te_chord_fraction}"
        )
    x = cosine_x(n) * te_chord_fraction
    y_t = naca_thickness(x, thickness)
    if te_chord_fraction >= 1.0 - 1e-12:
        y_t[-1] = 0.0              # sharp closed TE (legacy)
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
    """Named 2-D point set used by the C+H 6-block topology.

    Coordinate conventions:
      chord lies along +x with LE at (0, 0) and TE at (1, 0).
      Upstream radius Rc and transverse half-height Ht are equal so the upstream
      arc is a true semicircle centered on the LE.

    The T_* points sit on the vertical interface at x = chord + L_trans, which
    separates the transition wake block (immediately downstream of TE) from
    the main wake block (continuing to the outlet).
    """
    LE_FAR:   tuple[float, float]   # (-Rc, 0)
    TOP_MID:  tuple[float, float]   # (0,  +Ht)
    BOT_MID:  tuple[float, float]   # (0,  -Ht)
    TOP_TE:   tuple[float, float]   # (1,  +Ht)
    BOT_TE:   tuple[float, float]   # (1,  -Ht)
    T_TOP:    tuple[float, float]   # (1+Lt, +Ht)  transition->main interface, top
    T_BOT:    tuple[float, float]   # (1+Lt, -Ht)  transition->main interface, bottom
    T_MID:    tuple[float, float]   # (1+Lt, 0)    transition->main interface, wake centreline
    TOP_OUT:  tuple[float, float]   # (1+Lw, +Ht)
    BOT_OUT:  tuple[float, float]   # (1+Lw, -Ht)
    OUT_MID:  tuple[float, float]   # (1+Lw, 0)

    def as_dict(self) -> dict[str, tuple[float, float]]:
        return asdict(self)


def farfield_points(
    upstream_radius: float,
    downstream_length: float,
    transverse_extent: float,
    transition_wake_length: float,
    chord: float = 1.0,
    te_chord_fraction: float = 1.0,
) -> FarfieldPoints:
    """Build the FarfieldPoints set from chord-multiple extents.

    The TE column (TOP_TE, BOT_TE) anchors at x = te_chord_fraction * chord so
    the wall-normal seams above and below the airfoil remain vertical when the
    airfoil is truncated for a blunt TE. All downstream extents are measured
    from this column.

    Notes
    -----
    - `upstream_radius` and `transverse_extent` must be EQUAL (semicircular cap).
    - `transition_wake_length` must be strictly between 0 and `downstream_length`.
      The transition wake block occupies x in [te_x, te_x + Lt]; the main wake
      block occupies x in [te_x + Lt, te_x + Lw], where te_x = te_chord_fraction*chord.
    """
    if abs(upstream_radius - transverse_extent) > 1e-9:
        raise ValueError(
            f"Upstream radius ({upstream_radius}) must equal transverse extent "
            f"({transverse_extent}) for the C+H topology's semicircular cap."
        )
    if not (0.0 < transition_wake_length < downstream_length):
        raise ValueError(
            f"transition_wake_length ({transition_wake_length}) must lie strictly "
            f"between 0 and downstream_length ({downstream_length})."
        )
    if not (0.5 < te_chord_fraction <= 1.0):
        raise ValueError(
            f"te_chord_fraction must lie in (0.5, 1.0], got {te_chord_fraction}"
        )
    Rc = upstream_radius * chord
    Lw = downstream_length * chord
    Lt = transition_wake_length * chord
    Ht = transverse_extent * chord
    te_x = te_chord_fraction * chord
    return FarfieldPoints(
        LE_FAR  = (-Rc,        0.0),
        TOP_MID = ( 0.0,       +Ht),
        BOT_MID = ( 0.0,       -Ht),
        TOP_TE  = ( te_x,      +Ht),
        BOT_TE  = ( te_x,      -Ht),
        T_TOP   = ( te_x + Lt, +Ht),
        T_BOT   = ( te_x + Lt, -Ht),
        T_MID   = ( te_x + Lt,  0.0),
        TOP_OUT = ( te_x + Lw, +Ht),
        BOT_OUT = ( te_x + Lw, -Ht),
        OUT_MID = ( te_x + Lw,  0.0),
    )
