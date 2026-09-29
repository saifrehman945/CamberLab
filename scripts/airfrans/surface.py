"""
Module: scripts/airfrans/surface.py
Purpose: Wall quantities of one AirfRANS sample (pressure coefficient Cp and
         signed skin-friction coefficient Cf), their resampling onto the
         202-station x/c grid, and the force integration used to check them
         against the stored C_L / C_D (the Phase 3 feasibility gate).

The datasets store no wall shear stress, so it is computed from the velocity
gradient at the wall, porting AirfRANS' own metrics.py
(results/reference/airfrans_paper/metrics.py):

- WallShearStress:      τ = 2ν·dev(S)·n with n the wall normal into the fluid
- Compute_coefficients: F = ∮ τ ds − ∮ p·n ds on the wall polyline (values
                        averaged onto segments × segment length), rotated
                        into (drag, lift) by α, coefficient = 2F/U∞²

p is OpenFOAM's kinematic pressure (p/ρ, m²/s²) relative to the far field,
so Cp = p / (½U∞²). Cf = (τ·t) / (½U∞²) with t the unit tangent pointing from
the leading edge to the trailing edge along each surface: Cf < 0 marks
reversed flow (separation).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from scripts.airfrans.geometry import N_GRID, _arc, _interp_on_grid, _normalise, cosine_grid, wall_loop

# ν used by AirfRANS' metrics.py for the wall shear stress (and so for their C_D).
# io.NU (from the generator's polynomial) is kept for Re; the two differ by ~0.6%.
WALL_NU = 1.56e-5


# --------------------------------------------------------------------------
# Wall gradients and shear
# --------------------------------------------------------------------------

def wall_gradients(sample, order: np.ndarray) -> np.ndarray:
    """∇U at the wall nodes `order`, shape (n, 2, 2) with [i, j] = ∂U_i/∂x_j.

    P1 (linear) gradient per triangle, averaged onto each wall node over its
    incident triangles, weighted by triangle area. Only triangles touching the
    wall are evaluated.
    """
    tri = sample.triangles
    is_wall = np.zeros(len(sample.x), dtype=bool)
    is_wall[order] = True
    tri = tri[is_wall[tri].any(axis=1)]

    x, y = sample.x[tri], sample.y[tri]                      # (m, 3)
    ux, uy = sample.fields["Ux"][tri], sample.fields["Uy"][tri]
    # For vertices 0,1,2: grad f = (1/2A) Σ f_k (y_{k+1} − y_{k+2}, x_{k+2} − x_{k+1})
    by = np.roll(y, -1, axis=1) - np.roll(y, -2, axis=1)
    bx = np.roll(x, -2, axis=1) - np.roll(x, -1, axis=1)
    two_a = (x[:, 1] - x[:, 0]) * (y[:, 2] - y[:, 0]) - (x[:, 2] - x[:, 0]) * (y[:, 1] - y[:, 0])
    grad = np.empty((len(tri), 2, 2))
    for i, f in enumerate((ux, uy)):
        grad[:, i, 0] = (f * by).sum(axis=1) / two_a
        grad[:, i, 1] = (f * bx).sum(axis=1) / two_a
    area = 0.5 * np.abs(two_a)

    local = -np.ones(len(sample.x), dtype=np.int64)
    local[order] = np.arange(len(order))
    acc = np.zeros((len(order), 2, 2))
    wsum = np.zeros(len(order))
    for k in range(3):
        node = local[tri[:, k]]
        on = node >= 0
        np.add.at(acc, node[on], grad[on] * area[on, None, None])
        np.add.at(wsum, node[on], area[on])
    return acc / wsum[:, None, None]


def loop_normals(xy: np.ndarray) -> np.ndarray:
    """Unit normals at the nodes of a closed loop, pointing out of the enclosed body (into the fluid).

    Node normal = normalised sum of the length-weighted normals of its two
    adjacent segments; the orientation is fixed by the loop's signed area.
    """
    seg = np.roll(xy, -1, axis=0) - xy                       # segment k: node k -> k+1
    n_seg = np.column_stack([seg[:, 1], -seg[:, 0]])          # right-hand normal, ∝ length
    if _signed_area(xy) < 0:                                  # clockwise loop: right normal points inward
        n_seg = -n_seg
    n_node = n_seg + np.roll(n_seg, 1, axis=0)
    return n_node / np.linalg.norm(n_node, axis=1, keepdims=True)


def wall_shear(grad: np.ndarray, normals: np.ndarray, nu: float = WALL_NU) -> np.ndarray:
    """Kinematic wall shear stress vector (m²/s²), AirfRANS' WallShearStress."""
    s = 0.5 * (grad + grad.transpose(0, 2, 1))
    s = s - np.trace(s, axis1=1, axis2=2)[:, None, None] * np.eye(2)[None] / 3
    return 2 * nu * np.einsum("nij,nj->ni", s, normals)


def _signed_area(xy: np.ndarray) -> float:
    x, y = xy[:, 0], xy[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


# --------------------------------------------------------------------------
# Force integration
# --------------------------------------------------------------------------

def integrate_forces(xy: np.ndarray, p: np.ndarray, tau: np.ndarray, alpha_rad: float,
                     u_inf: float) -> dict[str, float]:
    """(C_D, C_L) from wall pressure and shear on a closed wall loop, split into both parts.

    Mirrors AirfRANS' Compute_coefficients: nodal values are averaged onto
    each segment and multiplied by its length; the force on the body is
    ∮ τ ds − ∮ p n ds (n into the fluid), rotated by α into (drag, lift).
    """
    seg = np.roll(xy, -1, axis=0) - xy
    length = np.linalg.norm(seg, axis=1)
    n_seg = np.column_stack([seg[:, 1], -seg[:, 0]]) / length[:, None]
    if _signed_area(xy) < 0:
        n_seg = -n_seg
    p_seg = 0.5 * (p + np.roll(p, -1))
    tau_seg = 0.5 * (tau + np.roll(tau, -1, axis=0))
    f_p = -(p_seg[:, None] * n_seg * length[:, None]).sum(axis=0)
    f_tau = (tau_seg * length[:, None]).sum(axis=0)
    c, s = np.cos(alpha_rad), np.sin(alpha_rad)
    basis = np.array([[c, s], [-s, c]])
    cp_ = 2 * basis @ f_p / u_inf**2
    ct_ = 2 * basis @ f_tau / u_inf**2
    return {"Cd_int_p": float(cp_[0]), "Cd_int_tau": float(ct_[0]),
            "Cl_int_p": float(cp_[1]), "Cl_int_tau": float(ct_[1])}


# --------------------------------------------------------------------------
# One sample -> native wall data and 202-station curves
# --------------------------------------------------------------------------

@dataclass
class WallData:
    """Native wall nodes of one sample, in loop order."""

    xy: np.ndarray        # (n, 2) physical coordinates
    p: np.ndarray         # (n,) kinematic pressure
    tau: np.ndarray       # (n, 2) kinematic wall shear stress
    tangent: np.ndarray   # (n, 2) unit tangent, LE -> TE on each surface
    x_c: np.ndarray       # (n,) chordwise position in the normalised section
    upper: np.ndarray     # (n,) bool: node on the upper surface (the TE node counts as upper)
    q_inf: float          # ½ U∞²

    @property
    def cp(self) -> np.ndarray:
        return self.p / self.q_inf

    @property
    def cf(self) -> np.ndarray:
        return np.einsum("ni,ni->n", self.tau, self.tangent) / self.q_inf


def _arcs(xy: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalised section loop (LE inserted) and its upper / lower arcs, each LE -> TE.

    Arc indices address the normalised loop. The inserted LE point sits at
    index `le`; loop node k of the input maps to k (k < le) or k + 1 (k ≥ le).
    """
    pts, le, te = _normalise(xy)
    n = len(pts)
    arc_a, arc_b = _arc(n, le, te, +1), _arc(n, le, te, -1)
    if pts[arc_a, 1].mean() >= pts[arc_b, 1].mean():
        return pts, arc_a, arc_b
    return pts, arc_b, arc_a


def wall_frame(xy: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-node (x/c, upper-surface mask, LE -> TE unit tangent) of an ordered wall loop.

    x/c comes from the normalised section (LE at 0, TE at 1, chord on +x; see
    geometry._normalise). The TE node closes both arcs and is counted on the
    upper surface.
    """
    pts, upper_arc, lower_arc = _arcs(xy)
    le = int(upper_arc[0])                   # the LE point _normalise inserted; not a mesh node
    up, lo = (_to_native(arc, le) for arc in (upper_arc, lower_arc))

    tangent = np.empty_like(xy)
    for arc in (lo, up):   # upper last: the TE node, shared by both arcs, takes the upper tangent
        t = np.gradient(xy[arc], axis=0)   # along the arc, LE -> TE; one-sided at the ends
        tangent[arc] = t / np.linalg.norm(t, axis=1, keepdims=True)
    upper = np.zeros(len(xy), dtype=bool)
    upper[up] = True
    native = np.arange(len(xy))
    return pts[native + (native >= le), 0], upper, tangent


def wall_data(sample) -> WallData:
    """Wall pressure, shear, tangents and x/c of one sample (needs sample.fields)."""
    order = wall_loop(sample)
    xy = np.column_stack([sample.x[order], sample.y[order]])
    tau = wall_shear(wall_gradients(sample, order), loop_normals(xy))
    x_c, upper, tangent = wall_frame(xy)
    u_inf = sample.scalars[_scalar("SCALAR_UINF")]
    return WallData(xy=xy, p=sample.fields["p"][order], tau=tau, tangent=tangent, x_c=x_c,
                    upper=upper, q_inf=0.5 * u_inf**2)


def _scalar(name: str) -> str:
    from scripts.airfrans import io   # deferred: io pulls in pyarrow/yaml, not needed for pure geometry
    return getattr(io, name)


def _to_native(arc: np.ndarray, le: int) -> np.ndarray:
    """Normalised-loop arc indices -> wall-loop node indices, dropping the inserted LE point."""
    arc = arc[arc != le]
    return arc - (arc > le)


def curves(wall: WallData, n_grid: int = N_GRID) -> dict[str, np.ndarray]:
    """Cp and Cf on the cosine x/c grid, per surface: keys Cp_upper, Cp_lower, Cf_upper, Cf_lower.

    The two surfaces share their end nodes (LE and TE), so both curves start
    and end at the same values.
    """
    grid = cosine_grid(n_grid)
    le_node = int(np.argmin(wall.x_c))
    te_node = int(np.argmax(wall.x_c))
    out = {}
    cp, cf = wall.cp, wall.cf
    for name, mask in (("upper", wall.upper), ("lower", ~wall.upper)):
        idx = np.union1d(np.flatnonzero(mask), [le_node, te_node])
        xc = wall.x_c[idx]
        out[f"Cp_{name}"] = _interp_on_grid(np.column_stack([xc, cp[idx]]), grid)
        out[f"Cf_{name}"] = _interp_on_grid(np.column_stack([xc, cf[idx]]), grid)
    return out


def forces(wall: WallData, sample) -> dict[str, float]:
    """integrate_forces for one sample's wall data at its own α and U∞."""
    return integrate_forces(wall.xy, wall.p, wall.tau, sample.scalars[_scalar("SCALAR_AOA")],
                            sample.scalars[_scalar("SCALAR_UINF")])
