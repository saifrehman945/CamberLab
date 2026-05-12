"""
Per-regime mesh parameter dictionary and inference-time classifier.

REGIME_MESH is the single source of truth for everything the meshing pipeline
needs to know about a regime. Only Regime A is populated; B / C / D are kept
as explicit None placeholders so a future caller that asks for an unimplemented
regime fails loudly instead of silently meshing with A's settings.

classify_regime follows the priority rule from CLAUDE.md §11.
"""

from __future__ import annotations


REGIME_MESH: dict[str, dict | None] = {
    "A": {
        # --- wall / boundary layer ----------------------------------------
        "y_plus_target":            30.0,
        "bl_growth_ratio":          1.20,
        "bl_layers":                22,
        "wall_treatment":           "wall_function",   # consumed by CFD stage, not topology

        # --- surface discretisation (raw airfoil sampling) ----------------
        "surface_points":           300,                # cosine-spaced per side

        # --- transfinite point counts -------------------------------------
        "chord_pts_upper":          160,                # along upper airfoil (LE -> TE)
        "chord_pts_lower":          160,                # along lower airfoil (LE -> TE)
        "normal_pts":               80,                 # wall-normal direction
        "wake_pts":                 140,                # along wake (TE -> outlet)

        # --- transfinite grading laws -------------------------------------
        # le_te_cluster: gmsh "Bump" beta on airfoil curves. Smaller -> tighter
        # clustering at BOTH endpoints (LE and TE simultaneously).
        "le_te_cluster":            0.05,
        # wake_progression: geometric growth ratio along the wake (cells
        # coarsen downstream of TE).
        "wake_progression":         1.04,
        # north_arc_to_horiz_ratio: split of the north edge of the over/under
        # airfoil blocks between the upstream semicircle arc and the
        # horizontal top/bottom run from x=0 to x=1.
        "north_arc_to_horiz_ratio": 1.0,

        # --- farfield extents (chord multiples) ---------------------------
        "upstream_radius":          20.0,
        "downstream_length":        30.0,
        "transverse_extent":        20.0,

        # --- 2D quasi-3D extrusion ----------------------------------------
        "spanwise_thickness":       0.05,
        "spanwise_layers":          1,

        # --- topology dispatch --------------------------------------------
        "topology":                 "c_grid_4block",

        # --- quality acceptance gates -------------------------------------
        "non_orthogonality_max":    70.0,
        "skewness_max":             4.0,
        "aspect_ratio_max":         5000.0,
        "min_hex_fraction":         0.999,

        # --- advisory cell-count band (warning only) ----------------------
        "target_cells_min":         40_000,
        "target_cells_max":         200_000,
    },
    "B": None,
    "C": None,
    "D": None,
}


def get_regime_config(regime: str) -> dict:
    """Fetch REGIME_MESH[regime] with a loud error if missing or unimplemented."""
    if regime not in REGIME_MESH:
        raise KeyError(
            f"Unknown regime '{regime}'. Known: {sorted(REGIME_MESH)}"
        )
    cfg = REGIME_MESH[regime]
    if cfg is None:
        raise NotImplementedError(
            f"Regime '{regime}' is not yet implemented. "
            f"Only Regime A is supported in the current meshing overhaul."
        )
    return cfg


def classify_regime(alpha_deg: float, Re: float, thickness: float) -> str:
    """Priority classifier from CLAUDE.md §11. First match wins."""
    if alpha_deg >= 10.0:
        return "B"
    if Re < 1.0e6:
        return "C"
    if Re >= 2.0e6 and alpha_deg <= 6.0 and 0.10 <= thickness <= 0.18:
        return "D"
    return "A"
