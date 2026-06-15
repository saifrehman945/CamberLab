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

        # --- blunt trailing edge ------------------------------------------
        # The airfoil is truncated at x = te_chord_fraction * chord and the
        # natural NACA-4 half-thickness at that x is kept as the TE half-
        # thickness. This eliminates the quasi-sharp tip that produces sliver
        # cells when a closed-TE NACA-4 is meshed structurally.
        # te_blunt_pts is the number of nodes ACROSS the blunt back wall
        # (TE_UP -> TE_MID -> TE_LO together), shared by the upper and lower
        # halves equally. Accepts either:
        #   "auto"  - use the maximum value the blunt-back geometry can fit
        #             given h_nu_first (wall-normal first cell at TE_UP).
        #             Best default: the count then varies per case, matching
        #             whatever the (Re, t/c) combination physically supports.
        #   int N>=3 - use N as a ceiling; the per-case adaptive solver may
        #             still reduce N if h_nu_first * (N - 1) > h_te.
        "te_chord_fraction":        0.99,
        "te_blunt_pts":             "auto",

        # --- transfinite point counts -------------------------------------
        "chord_pts_upper":          160,                # along upper airfoil (LE -> TE)
        "chord_pts_lower":          160,                # along lower airfoil (LE -> TE)
        "normal_pts":               100,                 # wall-normal direction
        "wake_pts":                 220,                # along MAIN wake (T_MID -> outlet)

        # --- wake transition block ----------------------------------------
        # A short structured block immediately downstream of TE that inherits
        # airfoil-TE chordwise spacing and grades outward to the main-wake
        # spacing. Removes the cell-size cliff that produces skewed cells at
        # the TE junction in a plain C-grid.
        "transition_wake_length":     0.65,    # chord multiples (0.3 - 0.5 typical)
        "transition_wake_pts":        120,     # nodes along the transition block
        # None -> derive progression from airfoil TE chord cell (recommended).
        # Set a float (e.g. 1.045) to force a specific value.
        "transition_wake_progression": None,

        # --- transfinite grading laws -------------------------------------
        # le_te_cluster: gmsh "Bump" beta on airfoil curves. Smaller -> tighter
        # clustering at BOTH endpoints (LE and TE simultaneously).
        "le_te_cluster":            0.09,
        # wake_progression: geometric growth ratio along the MAIN wake block
        # (cells coarsen from transition->main interface out to the outlet).
        "wake_progression":         1.015,
        # north_arc_to_horiz_ratio: split of the north edge of the over/under
        # airfoil blocks between the upstream semicircle arc and the
        # horizontal top/bottom run from x=0 to x=1.
        "north_arc_to_horiz_ratio": 0.65,

        # --- farfield extents (chord multiples) ---------------------------
        "upstream_radius":          20.0,
        "downstream_length":        30.0,
        "transverse_extent":        20.0,

        # --- 2D quasi-3D extrusion ----------------------------------------
        "spanwise_thickness":       0.05,
        "spanwise_layers":          1,

        # --- topology dispatch --------------------------------------------
        "topology":                 "c_grid_6block",

        # --- quality acceptance gates -------------------------------------
        "non_orthogonality_max":    70.0,
        "skewness_max":             4.0,
        "aspect_ratio_max":         5000.0,
        "min_hex_fraction":         0.999,

        # --- advisory cell-count band (warning only) ----------------------
        "target_cells_min":         40_000,
        "target_cells_max":         200_000,
    },
    "B": {
        # --- wall / boundary layer ----------------------------------------
        # Fully resolved walls: y+ < 1, no wall functions. Required for
        # kOmegaSST to capture separation onset at the near-stall regime.
        "y_plus_target":            0.5,
        "bl_growth_ratio":          1.30,
        "bl_layers":                35,
        "wall_treatment":           "low_re",          # consumed by CFD stage

        # --- surface discretisation (raw airfoil sampling) ----------------
        "surface_points":           300,

        # --- blunt trailing edge ------------------------------------------
        "te_chord_fraction":        0.99,
        "te_blunt_pts":             60,

        # --- transfinite point counts -------------------------------------
        # Suction-side resolution increased to capture adverse-pressure-
        # gradient separation. Wall-normal stack is taller to fit a y+~0.5
        # first cell + 35 BL layers + smooth transition to farfield.
        "chord_pts_upper":          220,
        "chord_pts_lower":          220,
        "normal_pts":               150,
        "wake_pts":                 180,

        # --- wake transition block ----------------------------------------
        "transition_wake_length":     1.0,
        "transition_wake_pts":        220,
        "transition_wake_progression": None,

        # --- transfinite grading laws -------------------------------------
        "le_te_cluster":            0.09,
        "wake_progression":         1.012,
        "north_arc_to_horiz_ratio": 0.65,

        # --- farfield extents (chord multiples) ---------------------------
        "upstream_radius":          20.0,
        "downstream_length":        30.0,
        "transverse_extent":        20.0,

        # --- 2D quasi-3D extrusion ----------------------------------------
        "spanwise_thickness":       0.05,
        "spanwise_layers":          1,

        # --- topology dispatch --------------------------------------------
        "topology":                 "c_grid_6block",

        # --- quality acceptance gates -------------------------------------
        # Tighter than A. Near-stall flow with high-AR resolved BL cells
        # is sensitive to non-orthogonality; >60deg risks divergence even
        # with nNonOrthogonalCorrectors=2 baked into the B fvSolution.
        "non_orthogonality_max":    70.0,
        "skewness_max":             3.0,
        "aspect_ratio_max":         50000.0,
        "min_hex_fraction":         0.999,

        # --- advisory cell-count band (warning only) ----------------------
        "target_cells_min":         300_000,
        "target_cells_max":         1_000_000,
    },
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
