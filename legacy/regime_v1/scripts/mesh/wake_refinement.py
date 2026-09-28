"""
Wake-refinement diagnostics.

The wake block grading itself lives inside topology.py because it is part of
the transfinite block construction. This module exposes pure helpers that the
caller (or future Regime B work) can use to size wake parameters or to inspect
the resulting cell size discontinuity at the trailing edge.
"""

from __future__ import annotations

from .boundary_layer import progression_sum, solve_progression


def first_wake_cell_size(total_length: float, n_cells: int, progression: float) -> float:
    """Streamwise size of the first wake cell (the cell immediately downstream
    of TE) for a Progression-graded wake of total length and n_cells cells."""
    if abs(progression - 1.0) < 1e-12:
        return total_length / n_cells
    return total_length * (progression - 1.0) / (progression ** n_cells - 1.0)


def solve_wake_progression(first_cell: float, total_length: float, n_cells: int) -> float:
    """Inverse of first_wake_cell_size: pick the progression that gives the
    desired first cell. Useful for tying the wake's TE spacing to the airfoil's
    TE chordwise spacing (avoiding an aspect-ratio cliff at the trailing edge).
    """
    return solve_progression(h1=first_cell, total_length=total_length, n_cells=n_cells)


def te_aspect_ratio(first_wake_h: float, normal_h: float) -> float:
    """Cell aspect ratio at TE: streamwise / wall-normal. Logged as a diagnostic;
    high values flag the typical 'wake coarser than airfoil TE' problem.
    """
    if normal_h <= 0.0:
        return float("inf")
    return first_wake_h / normal_h


__all__ = [
    "first_wake_cell_size",
    "solve_wake_progression",
    "te_aspect_ratio",
    "progression_sum",
]
