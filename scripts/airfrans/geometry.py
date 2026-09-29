"""
Module: scripts/airfrans/geometry.py
Purpose: Aerofoil section geometry shared by ingestion (mesh-extracted walls)
         and inference (analytic NACA sections), so training and query
         features are computed by exactly the same code.

- extract_surface   ordered (x, y) wall loop from an AirfRANS sample
- section_features  t_max, x_tmax, m_max, x_m (+ 101-point t(x), c(x)) from
                    any closed, ordered aerofoil point loop
- naca_coordinates  analytic NACA 4-digit and 5-digit (standard and reflex)
                    sections, closed trailing edge

NACA equations follow Abbott & von Doenhoff, "Theory of Wing Sections"
(1959), §6.4 and Appendix I, and NACA Report 537 / 610 for the 5-digit and
reflex mean lines.
"""

from __future__ import annotations

import re

import numpy as np

N_GRID = 101
WALL_TOL = 1e-3          # |implicit_distance| sanity bound on wall nodes (topology defines the wall)
SYMMETRIC_CAMBER_TOL = 5e-4   # |m_max| below this => section treated as symmetric
NOSE_WINDOW = 0.005           # chord fraction of nodes used to fit the leading edge
NOSE_MIN_NEIGHBOURS = 3       # at least this many loop neighbours each side in the fit

# Mesh clip box of the PLAID AirfRANS variants (dataset cards: [-2, 4] x [-1.5, 1.5]).
_BOX_ATOL = 1e-9


# --------------------------------------------------------------------------
# Wall extraction
# --------------------------------------------------------------------------

def _boundary_edges(triangles: np.ndarray) -> np.ndarray:
    edges = np.sort(np.vstack([triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]]), axis=1)
    uniq, counts = np.unique(edges, axis=0, return_counts=True)
    return uniq[counts == 1]


def _wall_edges(sample) -> np.ndarray:
    """Mesh boundary edges that are not on the outer clip box: the aerofoil wall."""
    x, y = sample.x, sample.y
    on_box = (np.isclose(x, x.min(), atol=_BOX_ATOL) | np.isclose(x, x.max(), atol=_BOX_ATOL)
              | np.isclose(y, y.min(), atol=_BOX_ATOL) | np.isclose(y, y.max(), atol=_BOX_ATOL))
    edges = _boundary_edges(sample.triangles)
    return edges[~on_box[edges].any(axis=1)]


def wall_node_mask(sample) -> np.ndarray:
    """Boolean per-node mask of aerofoil wall nodes (topological definition)."""
    mask = np.zeros(len(sample.x), dtype=bool)
    mask[np.unique(_wall_edges(sample))] = True
    return mask


def _chain_loop(edges: np.ndarray) -> np.ndarray:
    """Order a set of edges forming one closed loop into a node sequence."""
    adjacency: dict[int, list[int]] = {}
    for a, b in edges:
        adjacency.setdefault(int(a), []).append(int(b))
        adjacency.setdefault(int(b), []).append(int(a))
    if any(len(v) != 2 for v in adjacency.values()):
        raise ValueError("wall edges do not form a simple closed loop")
    start = int(edges[0, 0])
    order = [start]
    prev, cur = start, adjacency[start][0]
    while cur != start:
        order.append(cur)
        nxt = adjacency[cur][0] if adjacency[cur][0] != prev else adjacency[cur][1]
        prev, cur = cur, nxt
    if len(order) != len(adjacency):
        raise ValueError(f"wall has {len(adjacency)} nodes but the loop visits {len(order)} "
                         "(more than one closed boundary)")
    return np.asarray(order)


def wall_loop(sample) -> np.ndarray:
    """Node indices of the aerofoil wall of one AirfRANS sample, in loop order (not repeated).

    `sample` is a scripts.airfrans.io.SampleData. Wall nodes are the mesh
    boundary nodes off the clip box; they are chained along boundary edges
    (never sorted by polar angle, which misorders thin or reflexed trailing
    edges). All wall nodes are checked to have |implicit_distance| < WALL_TOL.
    """
    order = _chain_loop(_wall_edges(sample))
    d_max = float(np.abs(sample.implicit_distance[order]).max())
    if d_max > WALL_TOL:
        raise ValueError(f"wall node has |implicit_distance| = {d_max:.3g} > {WALL_TOL:g}")
    return order


def extract_surface(sample) -> np.ndarray:
    """Ordered (x, y) wall points of one AirfRANS sample, shape (n, 2), not repeated at the end.

    `sample` is a scripts.airfrans.io.SampleData (or a plaid Sample, which is
    converted); see wall_loop for how the wall is found.
    """
    if not hasattr(sample, "triangles"):
        from scripts.airfrans.io import extract
        sample = extract(sample, index=-1)
    order = wall_loop(sample)
    return np.column_stack([sample.x[order], sample.y[order]])


# --------------------------------------------------------------------------
# Section features
# --------------------------------------------------------------------------

def cosine_grid(n: int = N_GRID) -> np.ndarray:
    return 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, n)))


def _leading_edge(pts: np.ndarray) -> tuple[np.ndarray, int]:
    """Sub-node leading edge: the minimum-x point of a parabola x(y) fitted to the nose.

    The raw min-x node is ill-conditioned in y (the surface is tangent to the
    vertical there), so its position — and therefore the chord angle —
    depends on where the mesh happens to put nodes. Fitting x = a·y² + b·y + c
    through the nodes around it gives a node-placement-independent LE.
    Returns (LE point, index of the min-x node).
    """
    n = len(pts)
    i = int(np.argmin(pts[:, 0]))
    span = float(pts[:, 0].max() - pts[:, 0].min())
    window = NOSE_WINDOW * span
    idx = [i]
    for step in (+1, -1):
        j, k = i, 0
        while True:
            j = (j + step) % n
            k += 1
            if k > NOSE_MIN_NEIGHBOURS and pts[j, 0] - pts[i, 0] > window:
                break
            if k > n // 4:
                break
            idx.append(j)
    nose = pts[idx]
    a, b, c = np.polyfit(nose[:, 1], nose[:, 0], 2)
    if a <= 0:
        return pts[i].copy(), i
    y_v = -b / (2 * a)
    return np.array([c - b * b / (4 * a), y_v]), i


def _normalise(xy: np.ndarray) -> tuple[np.ndarray, int, int]:
    """Translate LE to origin, de-rotate chord onto +x, scale chord to 1.

    LE = min-x point of the nose (see _leading_edge); TE = max-x point
    (midpoint of the TE base if blunt). Iterated because de-rotation moves the
    min-x point. The fitted LE is inserted into the loop, so the returned loop
    has one more point than the input. Returns (loop, LE index, TE index).
    """
    pts = np.asarray(xy, dtype=np.float64)
    for _ in range(20):
        le, _ = _leading_edge(pts)
        x = pts[:, 0]
        te_cand = np.where(x >= x.max() - 1e-4 * (x.max() - x.min()))[0]
        te = pts[te_cand].mean(axis=0)
        chord_vec = te - le
        chord = float(np.hypot(*chord_vec))
        ang = np.arctan2(chord_vec[1], chord_vec[0])
        c, s = np.cos(-ang), np.sin(-ang)
        pts = ((pts - le) @ np.array([[c, -s], [s, c]]).T) / chord
        if np.hypot(*le) < 1e-12 * chord and abs(ang) < 1e-12 and abs(chord - 1) < 1e-12:
            break
    le, i = _leading_edge(pts)
    n = len(pts)
    nxt, prv = pts[(i + 1) % n], pts[(i - 1) % n]
    # insert the LE between node i and the neighbour on the far side of it in y
    after = (nxt[1] - pts[i, 1]) * (le[1] - pts[i, 1]) > 0 if prv[1] != nxt[1] else True
    pos = i + 1 if after else i
    pts = np.insert(pts, pos, le, axis=0)
    le_idx = pos
    te_idx = int(np.argmax(pts[:, 0]))
    return pts, le_idx, te_idx


def _arc(n: int, start: int, stop: int, step: int) -> np.ndarray:
    idx = [start]
    i = start
    while i != stop:
        i = (i + step) % n
        idx.append(i)
    return np.asarray(idx)


def _interp_on_grid(arc_pts: np.ndarray, grid: np.ndarray) -> np.ndarray:
    order = np.argsort(arc_pts[:, 0], kind="stable")
    x, y = arc_pts[order, 0], arc_pts[order, 1]
    x, keep = np.unique(x, return_index=True)
    return np.interp(grid, x, y[keep])


def _refined_extremum(x: np.ndarray, f: np.ndarray, i: int) -> tuple[float, float]:
    """Parabolic refinement of an extremum at grid index i (non-uniform grid)."""
    if i == 0 or i == len(x) - 1:
        return float(x[i]), float(f[i])
    xs, fs = x[i - 1:i + 2], f[i - 1:i + 2]
    a, b, c = np.polyfit(xs, fs, 2)
    if a == 0:
        return float(x[i]), float(f[i])
    xv = -b / (2 * a)
    if not xs[0] <= xv <= xs[2]:
        return float(x[i]), float(f[i])
    return float(xv), float(np.polyval([a, b, c], xv))


def section_features(xy: np.ndarray, n_grid: int = N_GRID) -> dict:
    """Thickness/camber features of a closed, ordered aerofoil point loop.

    Returns t_max, x_tmax, m_max (signed), x_m, and the n_grid-point arrays
    x_grid, thickness, camber, y_upper, y_lower on a cosine-spaced grid.
    x_m is set to 0 for symmetric sections (|m_max| < SYMMETRIC_CAMBER_TOL),
    where the camber location is undefined.

    Camber is measured from the geometric chord (nose min-x point to TE). For
    cambered NACA sections the true nose lies slightly above NACA's parametric
    chord, so m_max reads below the nominal NACA camber (e.g. 0.0369 for 4415,
    0.0141 for 23012). The offset is a property of the shape, identical for
    mesh-extracted and analytic points, so training and query features agree.
    """
    pts = np.asarray(xy, dtype=np.float64)
    if np.allclose(pts[0], pts[-1]):
        pts = pts[:-1]
    pts, le, te = _normalise(pts)
    n = len(pts)
    arc_a = pts[_arc(n, le, te, +1)]
    arc_b = pts[_arc(n, le, te, -1)]
    grid = cosine_grid(n_grid)
    ya, yb = _interp_on_grid(arc_a, grid), _interp_on_grid(arc_b, grid)
    y_u, y_l = (ya, yb) if ya.mean() >= yb.mean() else (yb, ya)
    thickness = y_u - y_l
    camber = 0.5 * (y_u + y_l)

    x_tmax, t_max = _refined_extremum(grid, thickness, int(np.argmax(thickness)))
    i_m = int(np.argmax(np.abs(camber)))
    x_m, m_max = _refined_extremum(grid, camber, i_m)
    if abs(m_max) < SYMMETRIC_CAMBER_TOL:
        x_m = 0.0
    return {"t_max": t_max, "x_tmax": x_tmax, "m_max": m_max, "x_m": x_m,
            "x_grid": grid, "thickness": thickness, "camber": camber,
            "y_upper": y_u, "y_lower": y_l}


# --------------------------------------------------------------------------
# Analytic NACA sections
# --------------------------------------------------------------------------

def naca_half_thickness(x: np.ndarray, t: float) -> np.ndarray:
    """NACA 4/5-digit half-thickness, closed-TE coefficient (-0.1036)."""
    return 5.0 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2
                      + 0.2843 * x**3 - 0.1036 * x**4)


def _camber_4digit(x: np.ndarray, m: float, p: float) -> tuple[np.ndarray, np.ndarray]:
    if m == 0.0 or p == 0.0:
        return np.zeros_like(x), np.zeros_like(x)
    fwd = x < p
    yc = np.where(fwd, m / p**2 * (2 * p * x - x**2),
                  m / (1 - p)**2 * ((1 - 2 * p) + 2 * p * x - x**2))
    dyc = np.where(fwd, 2 * m / p**2 * (p - x), 2 * m / (1 - p)**2 * (p - x))
    return yc, dyc


def _standard_5digit_constants(p: float, cl_design: float) -> tuple[float, float]:
    """(r, k1) of the standard 5-digit mean line for max-camber position p.

    r solves p = r (1 - sqrt(r / 3)) (location of zero curvature), and k1
    follows from the thin-aerofoil design lift coefficient. Reproduces the
    Abbott & von Doenhoff table (e.g. p = 0.15 -> r = 0.2025, k1 = 15.957 at
    C_l = 0.3).
    """
    lo, hi = p, 1.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if mid * (1 - np.sqrt(mid / 3)) < p:
            lo = mid
        else:
            hi = mid
    r = 0.5 * (lo + hi)
    q = ((3 * r - 7 * r**2 + 8 * r**3 - 4 * r**4) / np.sqrt(r * (1 - r))
         - 1.5 * (1 - 2 * r) * (np.pi / 2 - np.arcsin(1 - 2 * r)))
    return r, 6.0 * cl_design / q


# Reflex 5-digit mean line (Q = 1), tabulated for C_l = 0.3 (Abbott & von Doenhoff).
_REFLEX_TABLE = {  # P digit: (r, k1, k2/k1)
    2: (0.1300, 51.990, 0.000764),
    3: (0.2170, 15.793, 0.00677),
    4: (0.3180, 6.520, 0.0303),
    5: (0.4410, 3.191, 0.1355),
}


def _camber_5digit(x: np.ndarray, L: int, P: int, Q: int) -> tuple[np.ndarray, np.ndarray]:
    cl_design = 0.15 * L
    if cl_design == 0.0:
        return np.zeros_like(x), np.zeros_like(x)
    p = P / 20.0
    if Q == 0:
        r, k1 = _standard_5digit_constants(p, cl_design)
        front = x < r
        yc = np.where(front, k1 / 6 * (x**3 - 3 * r * x**2 + r**2 * (3 - r) * x),
                      k1 * r**3 / 6 * (1 - x))
        dyc = np.where(front, k1 / 6 * (3 * x**2 - 6 * r * x + r**2 * (3 - r)),
                       -k1 * r**3 / 6 * np.ones_like(x))
        return yc, dyc
    if Q == 1:
        if P not in _REFLEX_TABLE:
            raise ValueError(f"reflex 5-digit mean line tabulated only for P in "
                             f"{sorted(_REFLEX_TABLE)}; got P={P}")
        r, k1, k21 = _REFLEX_TABLE[P]
        scale = cl_design / 0.3
        front = x < r
        yc = np.where(front,
                      (x - r)**3 - k21 * (1 - r)**3 * x - r**3 * x + r**3,
                      k21 * (x - r)**3 - k21 * (1 - r)**3 * x - r**3 * x + r**3)
        dyc = np.where(front,
                       3 * (x - r)**2 - k21 * (1 - r)**3 - r**3,
                       3 * k21 * (x - r)**2 - k21 * (1 - r)**3 - r**3)
        return scale * k1 / 6 * yc, scale * k1 / 6 * dyc
    raise ValueError(f"5-digit third digit must be 0 (standard) or 1 (reflex); got {Q}")


def parse_naca(code: str) -> dict:
    """Validate a NACA 4- or 5-digit code and return its design parameters."""
    code = str(code).strip().upper().removeprefix("NACA").strip()
    if not re.fullmatch(r"\d{4}|\d{5}", code):
        raise ValueError(f"not a NACA 4- or 5-digit code: {code!r}")
    if len(code) == 4:
        m, p, t = int(code[0]) / 100, int(code[1]) / 10, int(code[2:]) / 100
        if (m == 0) != (p == 0):
            raise ValueError(f"NACA {code}: camber and camber position must both be zero or non-zero")
        return {"family": 4, "code": code, "m": m, "p": p, "t": t}
    L, P, Q, t = int(code[0]), int(code[1]), int(code[2]), int(code[3:]) / 100
    if Q not in (0, 1):
        raise ValueError(f"NACA {code}: third digit must be 0 or 1")
    if L > 0 and P == 0:
        raise ValueError(f"NACA {code}: camber position digit must be non-zero")
    return {"family": 5, "code": code, "L": L, "P": P, "Q": Q, "t": t}


def naca_coordinates(code: str, n: int = 201) -> np.ndarray:
    """Closed-TE NACA section as an ordered loop: TE -> upper -> LE -> lower -> TE.

    2·(n // 2) points in total (the TE point is not repeated); cosine-spaced
    in x on each surface. Chord 1, LE at origin.
    """
    spec = parse_naca(code)
    if spec["t"] <= 0:
        raise ValueError(f"NACA {spec['code']}: thickness must be positive")
    n_side = n // 2 + 1
    x = cosine_grid(n_side)
    if spec["family"] == 4:
        yc, dyc = _camber_4digit(x, spec["m"], spec["p"])
    else:
        yc, dyc = _camber_5digit(x, spec["L"], spec["P"], spec["Q"])
    yt = naca_half_thickness(x, spec["t"])
    theta = np.arctan(dyc)
    xu, yu = x - yt * np.sin(theta), yc + yt * np.cos(theta)
    xl, yl = x + yt * np.sin(theta), yc - yt * np.cos(theta)
    upper = np.column_stack([xu, yu])[::-1]        # TE -> LE
    lower = np.column_stack([xl, yl])[1:-1]        # after LE, before TE
    return np.vstack([upper, lower])
