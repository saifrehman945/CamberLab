"""
Boundary-layer math: first-cell height from y+, δ_99 correlation, and a Newton
solver for the geometric progression ratio needed to cover a given total
distance with a prescribed first-cell height and cell count.

These helpers are pure numerics — no gmsh, no I/O, fully unit-testable.
"""

from __future__ import annotations

import math


NU_AIR = 1.5e-5   # m^2/s  — standard sea-level kinematic viscosity
RHO_AIR = 1.225   # kg/m^3 — standard sea-level density


def first_cell_height(
    reynolds_number: float,
    y_plus: float,
    chord: float = 1.0,
    nu: float = NU_AIR,
    rho: float = RHO_AIR,
) -> float:
    """Wall-normal first-cell height for a target y+ on a turbulent flat plate.

    Uses the standard correlation
        C_f   = 0.026 / Re^(1/7)
        tau_w = 0.5 * rho * U_inf^2 * C_f
        u_tau = sqrt(tau_w / rho)
        h     = y+ * nu / u_tau
    which is the same form used in the legacy 03_mesh.py and quoted in
    CLAUDE.md §7.
    """
    cf = 0.026 / reynolds_number ** (1.0 / 7.0)
    u_inf = reynolds_number * nu / chord
    tau_w = 0.5 * rho * u_inf ** 2 * cf
    u_tau = (tau_w / rho) ** 0.5
    return y_plus * nu / u_tau


def bl_thickness(reynolds_number: float, chord: float = 1.0) -> float:
    """Advisory δ_99 from the turbulent flat-plate correlation
        delta_99 = 0.37 * c / Re^(1/5).
    """
    return 0.37 * chord / reynolds_number ** 0.2


def progression_sum(h1: float, r: float, n: int) -> float:
    """Total length of a geometric progression: h1 + h1*r + ... + h1*r^(n-1).

    For large n with r meaningfully above 1, r**n can overflow float64. In that
    regime the sum is unambiguously huge — much larger than any physical
    total_length — so we return math.inf, which lets the bisection upper-bound
    check in solve_progression succeed and steer r back down.
    """
    if abs(r - 1.0) < 1e-12:
        return h1 * n
    try:
        return h1 * (r ** n - 1.0) / (r - 1.0)
    except OverflowError:
        return math.inf


def solve_progression(
    h1: float,
    total_length: float,
    n_cells: int,
    r_lo: float = 1.0001,
    r_hi: float = 10.0,
    tol: float = 1e-8,
    max_iter: int = 200,
) -> float:
    """Solve for r such that h1 * (r^n - 1) / (r - 1) = total_length.

    Uses bisection because the function is monotonic in r for r > 1 and the
    domain is well-bounded. Returns the smallest r in [r_lo, r_hi] satisfying
    the constraint. Raises ValueError if total_length is unreachable.
    """
    if h1 * n_cells > total_length:
        raise ValueError(
            f"Uniform spacing alone (h1 * n = {h1*n_cells:.4g}) exceeds "
            f"requested total length {total_length:.4g}; reduce h1 or n."
        )
    if progression_sum(h1, r_hi, n_cells) < total_length:
        raise ValueError(
            f"Even r={r_hi} cannot reach total length {total_length:.4g} "
            f"with h1={h1:.4g}, n={n_cells}; raise n_cells or r_hi."
        )

    lo, hi = r_lo, r_hi
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        if progression_sum(h1, mid, n_cells) < total_length:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


def cells_needed(h1: float, r: float, total_length: float) -> int:
    """Minimum n such that h1 * (r^n - 1) / (r - 1) >= total_length, for r > 1."""
    if r <= 1.0:
        return math.ceil(total_length / h1)
    return math.ceil(math.log(1.0 + total_length * (r - 1.0) / h1) / math.log(r))
