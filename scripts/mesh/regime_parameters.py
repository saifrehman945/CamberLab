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
        "y_plus_target":            0.5,
        "bl_growth_ratio":          1.10,
        "bl_layers":                35,     # verify spans delta (~0.02c @ TE); see note
        "wall_treatment":           "low_re",

        # --- surface discretisation ---------------------------------------
        "surface_points":           300,

        # --- blunt trailing edge ------------------------------------------
        # FIX: was 200, which is 4x the documented intent and reproduces the
        # "mesh explosion" the comment warns against. ~50 is the geometric
        # ceiling that covers even t/c=0.24 at first cell ~5e-6, ratio ~1.11.
        # Treated as a ceiling; per-case solver may reduce for thin/low-Re B.
        "te_chord_fraction":        0.99,
        "te_blunt_pts":             50,

        # --- transfinite point counts -------------------------------------
        "chord_pts_upper":          220,
        "chord_pts_lower":          220,
        "normal_pts":               150,
        "wake_pts":                 280,

        # --- wake transition block ----------------------------------------
        "transition_wake_length":     1.0,
        "transition_wake_pts":        160,
        "transition_wake_progression": None,

        # --- transfinite grading laws -------------------------------------
        "le_te_cluster":            0.09,
        "wake_progression":         1.012,
        "north_arc_to_horiz_ratio": 0.65,

        # --- farfield extents (chord multiples) ---------------------------
        # FIX: raised from 20/30/20. Near-stall lift is farfield-sensitive;
        # 20c transverse biases Cl high via blockage at alpha 14-16 deg.
        # Alternative: keep ~25c and add a point-vortex far-field BC.
        "upstream_radius":          50.0,
        "downstream_length":        50.0,
        "transverse_extent":        50.0,

        # --- 2D quasi-3D extrusion ----------------------------------------
        "spanwise_thickness":       0.05,
        "spanwise_layers":          1,

        # --- BoundaryLayer-field meshing ----------------------------------
        # Regime B's ~2 micron first cell makes transfinite interpolation fold
        # the near-wall cells (degenerate cells all along the airfoil + wake;
        # gmshToFoam then aborts). So B meshes with gmsh's BoundaryLayer field
        # (true wall-normal layers) + a frontal-quad fill rather than the
        # c_grid_6block transfinite topology. See mesh.bl_field for the why.
        #
        # NOTE: the chord_pts_*, normal_pts, wake_pts, transition_wake_*,
        # le_te_cluster, north_arc_to_horiz_ratio and te_blunt_pts keys above
        # are INERT under this topology (kept for reference / metadata). Under
        # bl_field, near-wall spacing is set by y_plus_target + bl_growth_ratio
        # and the outer fill by the size knobs below.
        "topology":                 "bl_field",
        # Raised 2.0 -> 4.0: the outermost BL cells were too small relative to the
        # frontal-fill mesh, leaving a distorted annular region around the BL. A
        # thicker BL field (more layers, larger BL-edge cell) hands off to the
        # quad fill more gently. n_bl and h_bl_edge are derived from this.
        "bl_thickness_factor":      4.0,    # BL field thickness = factor * delta_99
        # Tightened 1.20 -> 1.15: this caps every cell-to-cell coarsening
        # transition (airfoil->far, TE->surf, wake->far). Smaller -> smoother
        # transition at the BL / fill interface (lower skewness & non-ortho).
        "max_growth_ratio":         1.15,   # cell-to-cell coarsening cap; ALL
                                            # transitions (airfoil->far, TE->surf,
                                            # wake->far) are sized from this.
        # Kept at 1.0. The BL-field -> fill interface smoothness is governed by
        # max_growth_ratio (g) ALONE: bl_field grades every transition so per-cell
        # growth never exceeds g, regardless of the far cell size (see af_grade in
        # build_bl_mesh). Shrinking far_cell to 0.25 did NOT smooth the interface;
        # it only flooded the ~8900c^2 far field with ~140k uniform 0.25c cells,
        # pushing the mesh to ~400k cells and exhausting RAM. With g=1.15 doing the
        # smoothing, 1.0 is the right asymptotic far size (~224k cells total).
        "far_cell_size":            1.0,    # chord multiples — max cell at far field
        "le_cluster_factor":        0.50,   # blunt-TE corner cell = factor * surf cell
        "le_refine_radius":         0.05,   # chord multiples around the TE corners
        # Wake corridor retuned for near-stall separated flow: longer (12 -> 20c so
        # it stays refined 15-20c downstream), narrower (0.6 -> 0.5c) and much
        # finer (8 -> 2x surf cell) to preserve wake vortical structures and cut
        # interpolation error in the separated wake. Realised by the `box` field
        # in bl_field.build_bl_mesh (Surface -> BL -> Wake -> Far size hierarchy).
        "wake_box_length":          20.0,   # chord multiples of fine wake corridor
        "wake_box_halfwidth":       0.5,    # chord multiples above/below wake centreline
        "wake_cell_size":           2.0,    # multiples of the surface cell in the corridor
                                            # NOTE: now the dominant cell-count cost
                                            # (~135k over the 20c corridor). Raise to
                                            # 3-4 to roughly halve total cells if RAM
                                            # is tight; the wake stays well-resolved.
        "mesh_smoothing":           5,      # Laplacian smoothing passes on the fill

        # --- meshing resource controls ------------------------------------
        # recombine_algorithm: 1 = Blossom (best quality, heaviest RAM/CPU; the
        # validated NACA0012 quality used it). Set to 0 (simple) to cut peak gmsh
        # RAM dramatically on large meshes at a small quality cost.
        "recombine_algorithm":      1,
        # Hard ceiling on the estimated 3-D cell count. build_bl_mesh estimates
        # the mesh size from the size field BEFORE meshing and aborts with an
        # actionable error if it exceeds this, so a runaway never reaches gmsh and
        # exhausts RAM. Overridable per run via env NACA_MESH_MAX_CELLS.
        "max_mesh_cells":           1_200_000,

        # --- quality acceptance gates -------------------------------------
        # Realised on the verified bl_field mesh (NACA0012/Re=6e6): non-ortho
        # 67.8, skewness 1.00, aspect 946, hex 0.98.
        #   - non-ortho peaks at the BL / frontal-fill interface; 70 is the
        #     CLAUDE.md ceiling. Pair with nNonOrthogonalCorrectors>=2 and the
        #     reduced relaxation the Regime B fvSolution already uses (CLAUDE §5).
        #   - frontal-quad fill leaves ~2% triangles (-> prisms); realised hex
        #     fraction ~0.98, so 0.85 is a comfortable floor.
        "non_orthogonality_max":    70.0,
        "skewness_max":             3.0,
        "aspect_ratio_max":         5000.0,
        "min_hex_fraction":         0.85,

        # --- advisory cell-count band (warning only) ----------------------
        # Lowered from 300k: that floor was calibrated for the structured
        # transfinite grid, whose far field carries many cells. The bl_field
        # mesh fills the far field with coarse unstructured quads, so an
        # equivalent near-wall resolution lands ~80-150k total (NACA0012/Re=6e6
        # ~ 87k). Raise far-field/wake refinement (far_cell_size, grade_distance,
        # wake_* knobs) if a case needs more wake resolution.
        "target_cells_min":         60_000,
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
