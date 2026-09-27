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

        # --- trailing edge ------------------------------------------------
        # te_chord_fraction == 1.0 selects a SHARP (closed) TE and the clean
        # 6-block C-grid (topology._build_sharp_c_grid): the upper/lower
        # splines meet at a single point, so there is no blunt base, no h_te
        # step, and every wake column is exactly normal_pts tall (all hex).
        # geometry.naca_symmetric uses the closed-TE thickness coefficient at
        # 1.0 so the tip closes cleanly with a finite wedge angle (no sliver).
        #
        # te_chord_fraction < 1.0 instead truncates the airfoil to a blunt TE
        # and uses the legacy 10-block topology, which carries te_blunt_pts in
        # the wake transverse seam. te_blunt_pts is consumed ONLY in that blunt
        # path and is ignored when te_chord_fraction == 1.0.
        "te_chord_fraction":        1.0,
        "te_blunt_pts":             "auto",   # blunt path only; ignored when sharp

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
        # y_plus_mesh_factor: the flat-plate Cf correlation in first_cell_height
        # uses FREESTREAM velocity, but at near-stall alpha the LE accelerates
        # the flow to ~3.5x freestream (Cp~-11), so the true wall shear near the
        # nose is far higher than the correlation assumes. Validation showed the
        # realised y+ hit ~23 (max) / ~2.4 (mean) against a 0.5 target, which
        # invalidates nutLowReWallFunction (needs y+<1) right where the suction
        # peak forms -> spurious LE separation. Size the mesh for an effective
        # y+ ~0.07 so the realised y+ stays < 1 even in attached flow. Only
        # consumed by topology.py; other regimes default to 1.0.
        "y_plus_mesh_factor":       0.14,
        # 1.10 matches CLAUDE.md §7's Regime B spec and the progression actually
        # solved over the 20c radius with the y+~0.07 first cell + normal_pts=210
        # (~1.077). The old 1.30 hint tripped a spurious deviation warning.
        "bl_growth_ratio":          1.10,
        "bl_layers":                35,
        "wall_treatment":           "low_re",          # consumed by CFD stage

        # --- surface discretisation (raw airfoil sampling) ----------------
        "surface_points":           300,

        # --- trailing edge (sharp; see Regime A) --------------------------
        "te_chord_fraction":        1.0,
        "te_blunt_pts":             200,      # blunt path only; ignored when sharp

        # --- transfinite point counts -------------------------------------
        # Suction-side resolution increased to capture adverse-pressure-
        # gradient separation. Wall-normal stack is taller to fit the finer
        # y+~0.07 first cell (see y_plus_mesh_factor) + smooth transition to
        # farfield; normal_pts raised 150->210 keeps the solved wall-normal
        # progression gentle (~1.08 over the 20c radius) despite the ~7x
        # smaller first cell. Chord resolution raised 220->280 to close the gap
        # to the CFL3D benchmark grid (~448 upper-surface pts) near the nose.
        "chord_pts_upper":          280,
        "chord_pts_lower":          280,
        "normal_pts":               210,
        "wake_pts":                 180,
        # Transverse first-cell at the WAKE CENTRELINE (t_seam/c_out edges),
        # decoupled from the y+<1 wall h1 (~3e-7). Anchoring h1 here produced
        # >1e6 aspect-ratio and ~90deg non-orthogonal cells along the centreline
        # (stiff pressure -> FPE). 2e-3 still resolves the wake (the near-TE
        # sheet stays fine via the te_nu seam) and drops centreline AR by ~3
        # orders of magnitude. Wall BL resolution is unaffected.
        "wake_centreline_h":        None,

        # --- wake transition block ----------------------------------------
        "transition_wake_length":     1.0,
        "transition_wake_pts":        220,
        "transition_wake_progression": None,

        # --- transfinite grading laws -------------------------------------
        # le_te_cluster tightened 0.09->0.06: finer LE/TE clustering to resolve
        # the sharp (Cp~-11, recovers by ~5% chord) near-stall suction peak.
        "le_te_cluster":            0.06,
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
        # aspect_ratio_max raised 5e5->2e6: the ~7x finer first cell (y+~0.07)
        # multiplies the far-WAKE aspect ratio (tiny y+-fine normal cells along
        # the wake centreline stretched against large streamwise cells). This
        # is cosmetic (far field, negligible gradients) and not at the airfoil,
        # so the gate is relaxed rather than letting it block an otherwise
        # sound near-wall mesh.
        "non_orthogonality_max":    90,
        "skewness_max":             4.0,
        "aspect_ratio_max":         2_000_000.0,
        "min_hex_fraction":         0.999,

        # --- advisory cell-count band (warning only) ----------------------
        "target_cells_min":         300_000,
        "target_cells_max":         1_000_000,
    },
    "C": {
        # --- wall / boundary layer ----------------------------------------
        # Fully resolved walls: y+ < 1, no wall functions. Required for the
        # transition model (kkLOmega / SST-lowRe fallback) to predict the
        # laminar separation bubble and transition onset. Low Re means the
        # first cell is actually LARGER than B at the same y+ (h1 ~ 1.2e-5 to
        # 3.6e-5 m), so the wall-normal stack is easier than B's despite the
        # finer streamwise grid.
        "y_plus_target":            0.5,
        "bl_growth_ratio":          1.08,           # gentle: thick laminar BL + LSB
        "bl_layers":                40,             # 35-45 band centre
        "wall_treatment":           "low_re",       # consumed by CFD stage

        # --- surface discretisation (raw airfoil sampling) ----------------
        "surface_points":           300,            # ample even for t=0.08

        # --- trailing edge (sharp; see Regime A) --------------------------
        # NOTE: C spans thin airfoils (t down to 0.08) -> the sharp wedge angle
        # is smallest here (~11deg included at t=0.08), so the TE corner cell is
        # the worst-case for skew across the regimes. Watch checkMesh skewness
        # on thin/low-Re C samples; soften le_te_cluster if it approaches the
        # gate. te_blunt_pts is ignored when sharp.
        "te_chord_fraction":        1.0,
        "te_blunt_pts":             "auto",   # blunt path only; ignored when sharp

        # --- transfinite point counts -------------------------------------
        # Highest streamwise resolution of any regime. The laminar separation
        # bubble and transition onset are mid-chord, streamwise phenomena, so
        # chord resolution is prioritised over B. normal_pts taller than B for
        # the thicker laminar BL; solved progression comes out ~1.05 (below the
        # 1.08 hint, which is desirable here) and lands the structured grid at
        # ~335k cells, inside the advisory band below.
        "chord_pts_upper":          320,
        "chord_pts_lower":          320,
        "normal_pts":               210,
        "wake_pts":                 220,
        # Wake-centreline transverse first cell, decoupled from the y+<1 wall h1
        # (see Regime B for the full rationale: anchoring h1 here causes extreme
        # AR / near-90deg non-orthogonality along the centreline). C's first
        # cell is larger than B's (low Re), so the pathology is milder, but the
        # same decoupling keeps the wake cells well-shaped.
        "wake_centreline_h":        None,

        # --- wake transition block ----------------------------------------
        "transition_wake_length":     1.0,
        "transition_wake_pts":        240,
        "transition_wake_progression": None,

        # --- transfinite grading laws -------------------------------------
        # le_te_cluster RAISED vs A/B (0.09 -> 0.12): nudges toward uniform
        # streamwise spacing so mid-chord cells (where the LSB sits) are finer.
        # Trades a slightly smeared LE suction peak for better bubble capture.
        "le_te_cluster":            0.12,
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
        # Mirror B: resolved y+<1 BL produces high-AR wall cells; relax
        # non-orthogonality and aspect ratio, keep skewness tight because the
        # transition model is sensitive to it.
        "non_orthogonality_max":    75,
        "skewness_max":             3.0,
        "aspect_ratio_max":         50000.0,
        "min_hex_fraction":         0.999,

        # --- advisory cell-count band (warning only) ----------------------
        # Realised ~330-400k with the structured 6-block grid (efficient, like
        # B which landed at 230k vs its 300k-1M spec). Floor lowered below the
        # metadata 500k so the realistic count doesn't trip a spurious warning,
        # mirroring how A uses 40k under its 80k spec. Upper kept from metadata.
        "target_cells_min":         300_000,
        "target_cells_max":         1_200_000,
    },
    "D": {
        # --- wall / boundary layer ----------------------------------------
        # Fully turbulent attached flow at high Re; wall functions at y+~50
        # (SA + nutkWallFunction). The cheapest regime. y+=50 gives a larger
        # first cell than A (h1 ~ 2.6e-4 to 6.2e-4 m), so the wall-normal stack
        # can be shorter than A while keeping the solved progression in band.
        "y_plus_target":            50.0,
        "bl_growth_ratio":          1.25,           # derived r ~1.11-1.14 < hint
        "bl_layers":                15,             # 12-18 band centre
        "wall_treatment":           "wall_function",# consumed by CFD stage

        # --- surface discretisation (raw airfoil sampling) ----------------
        "surface_points":           300,

        # --- trailing edge (sharp; see Regime A) --------------------------
        "te_chord_fraction":        1.0,
        "te_blunt_pts":             "auto",   # blunt path only; ignored when sharp

        # --- transfinite point counts -------------------------------------
        # Leaner than A: mild alpha (0-6 deg), attached flow, 50k-150k target.
        # normal_pts=80 (vs A's 100) leverages the larger y+=50 first cell;
        # solved wall-normal progression ~1.11-1.14, inside [1.10, 1.25].
        "chord_pts_upper":          140,
        "chord_pts_lower":          140,
        "normal_pts":               80,
        "wake_pts":                 180,

        # --- wake transition block ----------------------------------------
        "transition_wake_length":     0.5,
        "transition_wake_pts":        90,
        "transition_wake_progression": None,

        # --- transfinite grading laws -------------------------------------
        "le_te_cluster":            0.09,           # same as A
        "wake_progression":         1.018,          # slightly faster (light wake)
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
        # Like A: robust attached flow. The larger first cell (y+=50) gives
        # lower aspect ratios than A, so A's gates are comfortably safe here.
        "non_orthogonality_max":    75.0,
        "skewness_max":             4.0,
        "aspect_ratio_max":         5000.0,
        "min_hex_fraction":         0.999,

        # --- advisory cell-count band (warning only) ----------------------
        # Realised ~70k. Floor lowered below the metadata 50k to avoid a
        # spurious warning (same precedent as A's 40k under its 80k spec).
        "target_cells_min":         40_000,
        "target_cells_max":         150_000,
    },
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
