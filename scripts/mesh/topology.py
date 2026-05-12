"""
Structured C+H 4-block transfinite mesh for a NACA airfoil (Regime A).

Topology
========

The mesh is composed of four transfinite quad blocks that share corners at
the airfoil LE, the airfoil TE, the outlet midline point OUT_MID, and the
upstream farfield point LE_FAR::

                                                F_TOP_OUT (1+Lw, +Ht)
                F_TOP_MID (0,+Ht) - F_TOP_TE (1,+Ht) -------+
                  /                       |                 |
                 /          BLOCK U        |    BLOCK UW    |
                /     (over airfoil)       | (upper wake)   |
               /                           |                |
   F_LE_FAR - + ----- airfoil_upper ---- TE_AF -- wake_ax --+ OUT_MID (1+Lw,0)
  (-Rc,0)     \\                          |                 |
               \\          BLOCK L        |    BLOCK LW    |
                \\    (under airfoil)     | (lower wake)   |
                 \\                       |                 |
                F_BOT_MID (0,-Ht) - F_BOT_TE (1,-Ht) -------+
                                                F_BOT_OUT (1+Lw,-Ht)

Each block is a transfinite quad surface; the four 2-D surfaces are extruded
one cell in the +z direction (with `recombine=True`) to produce a single layer
of hexes, giving OpenFOAM the quasi-2-D mesh it expects with empty
front/back patches.

Curve direction conventions (start -> end)
------------------------------------------
  C_af_up   : LE_AF -> TE_AF       (upper airfoil spline)
  C_af_lo   : LE_AF -> TE_AF       (lower airfoil spline)
  C_arc_up  : LE_FAR -> TOP_MID    (upper-left semicircle quadrant)
  C_arc_lo  : LE_FAR -> BOT_MID    (lower-left semicircle quadrant)
  C_top_h   : TOP_MID -> TOP_TE
  C_bot_h   : BOT_MID -> BOT_TE
  C_top_tail: TOP_TE -> TOP_OUT
  C_bot_tail: BOT_TE -> BOT_OUT
  C_outU    : TOP_OUT -> OUT_MID
  C_outL    : OUT_MID -> BOT_OUT
  C_le_rad  : LE_FAR -> LE_AF
  C_te_nu   : TE_AF -> TOP_TE
  C_te_nl   : TE_AF -> BOT_TE
  C_wake_ax : TE_AF -> OUT_MID

Block loops (signed curve tags, CCW with interior on the left)
--------------------------------------------------------------
  Block U  : [+af_up, +te_nu, -top_h, -arc_up, +le_rad]
  Block L  : [+arc_lo, +bot_h, -te_nl, -af_lo, -le_rad]
  Block UW : [+wake_ax, -outU, -top_tail, -te_nu]
  Block LW : [+bot_tail, -outL, -wake_ax, +te_nl]

Transfinite point-count constraints
-----------------------------------
  N_af_up   = N_af_lo   = chord_pts (regime cfg)
  N_arc + N_h - 1       = chord_pts          (shared TOP_MID/BOT_MID node)
  N_le_rad  = N_te_nu  = N_te_nl  = N_outU  = N_outL  = normal_pts
  N_wake_ax = N_top_tail = N_bot_tail        = wake_pts
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np

from .boundary_layer import first_cell_height, solve_progression, NU_AIR, RHO_AIR
from .geometry import naca_symmetric, farfield_points

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _split_north_counts(chord_pts: int, ratio: float) -> tuple[int, int]:
    """Split chord_pts (total nodes on a north edge) into (n_arc, n_horiz)
    such that n_arc + n_horiz - 1 == chord_pts (the shared TOP_MID node is
    counted once across the two sub-curves).
    """
    total_cells = chord_pts - 1
    arc_cells = max(1, round(total_cells * ratio / (ratio + 1.0)))
    horiz_cells = max(1, total_cells - arc_cells)
    return arc_cells + 1, horiz_cells + 1


def _block_offsets(loop_sizes: list[int]) -> list[int]:
    """Starting index of each block's entities inside a multi-surface
    `geo.extrude` return list. Each surface contributes 2 + loop_size entries.
    """
    offsets, cursor = [], 0
    for n in loop_sizes:
        offsets.append(cursor)
        cursor += 2 + n
    return offsets


def _classify_lateral(gmsh_module, surface_tag: int,
                      airfoil_curves: set[int],
                      farfield_curves: set[int]) -> str:
    """Return 'aerofoil', 'freestream', or 'internal' for an extruded lateral surface
    by inspecting which source curve it inherits.
    """
    bounds = {
        abs(t) for d, t in gmsh_module.model.getBoundary(
            [(2, surface_tag)], oriented=False, recursive=False
        ) if d == 1
    }
    if bounds & airfoil_curves:
        return "aerofoil"
    if bounds & farfield_curves:
        return "freestream"
    return "internal"


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build_c_grid(
    case_dir: Path,
    params: dict,
    cfg: dict,
    chord: float = 1.0,
) -> dict[str, float]:
    """Build the 4-block transfinite C+H mesh for one case.

    Writes <case_dir>/mesh.msh. Returns a dict of mesh metrics suitable for
    merging into `case_metadata.json` (cell count, first-cell height used,
    derived progression ratio, etc.).
    """
    import gmsh  # local import — module-level import would force a gmsh dep at import time

    start_time = time.perf_counter()

    # ---- physics-derived spacings ----------------------------------------
    h1 = first_cell_height(params["Re"], cfg["y_plus_target"], chord=chord)

    # ---- topology dimensions --------------------------------------------
    ff = farfield_points(
        upstream_radius=cfg["upstream_radius"],
        downstream_length=cfg["downstream_length"],
        transverse_extent=cfg["transverse_extent"],
        chord=chord,
    )
    Rc = cfg["upstream_radius"] * chord
    Lw = cfg["downstream_length"] * chord
    Ht = cfg["transverse_extent"] * chord

    # Wall-normal progression: derived to match h1 exactly across n cells
    # covering the whole upstream radius. The dict's bl_growth_ratio is a soft
    # hint; we warn if the derived ratio differs by more than ~15%.
    n_normal_cells = cfg["normal_pts"] - 1
    r_normal = solve_progression(h1=h1, total_length=Rc, n_cells=n_normal_cells)
    hint_r = cfg["bl_growth_ratio"]
    if abs(r_normal - hint_r) / hint_r > 0.15:
        log.warning(
            "%s: derived normal-direction progression %.4f deviates from regime hint %.4f "
            "(h1=%.3e m, n=%d, L=%.2f m). Adjust normal_pts or y_plus_target.",
            case_dir.name, r_normal, hint_r, h1, n_normal_cells, Rc,
        )

    r_wake = float(cfg["wake_progression"])

    # ---- surface geometry ------------------------------------------------
    upper, lower = naca_symmetric(params["thickness"], n=cfg["surface_points"])

    # ---- gmsh setup ------------------------------------------------------
    gmsh.clear()
    gmsh.model.add(case_dir.name)
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    gmsh.option.setNumber("Mesh.SaveAll", 0)
    gmsh.option.setNumber("Mesh.Algorithm", 8)               # Frontal-Delaunay-for-Quads (fallback)
    gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 1)  # Blossom
    gmsh.option.setNumber("Mesh.RecombineAll", 1)
    gmsh.option.setNumber("Mesh.ElementOrder", 1)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)

    geo = gmsh.model.geo

    # ---- corner points ---------------------------------------------------
    p_LE_AF  = geo.addPoint(0.0,    0.0,    0.0)
    p_TE_AF  = geo.addPoint(chord,  0.0,    0.0)
    p_LE_FAR = geo.addPoint(*ff.LE_FAR,  0.0)
    p_TOP_MID = geo.addPoint(*ff.TOP_MID, 0.0)
    p_BOT_MID = geo.addPoint(*ff.BOT_MID, 0.0)
    p_TOP_TE  = geo.addPoint(*ff.TOP_TE,  0.0)
    p_BOT_TE  = geo.addPoint(*ff.BOT_TE,  0.0)
    p_TOP_OUT = geo.addPoint(*ff.TOP_OUT, 0.0)
    p_BOT_OUT = geo.addPoint(*ff.BOT_OUT, 0.0)
    p_OUT_MID = geo.addPoint(*ff.OUT_MID, 0.0)

    # ---- airfoil splines (LE -> TE for both upper and lower) -------------
    # Skip the first and last sample (LE and TE are explicit corner points
    # so the spline endpoints are shared with the topology corners).
    up_inner = [geo.addPoint(float(x), float(y), 0.0) for x, y in upper[1:-1]]
    lo_inner = [geo.addPoint(float(x), float(y), 0.0) for x, y in lower[1:-1]]
    C_af_up = geo.addSpline([p_LE_AF, *up_inner, p_TE_AF])
    C_af_lo = geo.addSpline([p_LE_AF, *lo_inner, p_TE_AF])

    # ---- farfield curves ------------------------------------------------
    # 90 deg arcs around LE_AF (center). gmsh chooses the short arc.
    C_arc_up = geo.addCircleArc(p_LE_FAR, p_LE_AF, p_TOP_MID)
    C_arc_lo = geo.addCircleArc(p_LE_FAR, p_LE_AF, p_BOT_MID)
    C_top_h    = geo.addLine(p_TOP_MID, p_TOP_TE)
    C_bot_h    = geo.addLine(p_BOT_MID, p_BOT_TE)
    C_top_tail = geo.addLine(p_TOP_TE, p_TOP_OUT)
    C_bot_tail = geo.addLine(p_BOT_TE, p_BOT_OUT)
    C_outU     = geo.addLine(p_TOP_OUT, p_OUT_MID)
    C_outL     = geo.addLine(p_OUT_MID, p_BOT_OUT)

    # ---- internal seam curves (wall-normal + wake centerline) ------------
    C_le_rad  = geo.addLine(p_LE_FAR, p_LE_AF)
    C_te_nu   = geo.addLine(p_TE_AF,  p_TOP_TE)
    C_te_nl   = geo.addLine(p_TE_AF,  p_BOT_TE)
    C_wake_ax = geo.addLine(p_TE_AF,  p_OUT_MID)

    # ---- transfinite line counts ----------------------------------------
    chord_pts = cfg["chord_pts_upper"]
    if cfg["chord_pts_lower"] != chord_pts:
        # symmetric airfoils only — fail loud if a future regime asks for asymmetric.
        raise ValueError(
            f"chord_pts_upper ({chord_pts}) must equal chord_pts_lower "
            f"({cfg['chord_pts_lower']}) for symmetric NACA mesh."
        )
    n_arc, n_h = _split_north_counts(chord_pts, cfg["north_arc_to_horiz_ratio"])
    normal_pts = cfg["normal_pts"]
    wake_pts   = cfg["wake_pts"]
    bump = float(cfg["le_te_cluster"])

    # Airfoil — Bump law clusters at BOTH endpoints (LE and TE simultaneously).
    geo.mesh.setTransfiniteCurve(C_af_up, chord_pts, "Bump", bump)
    geo.mesh.setTransfiniteCurve(C_af_lo, chord_pts, "Bump", bump)

    # North-edge segments — uniform along the arc and horizontal partitions.
    geo.mesh.setTransfiniteCurve(C_arc_up, n_arc, "Progression", 1.0)
    geo.mesh.setTransfiniteCurve(C_arc_lo, n_arc, "Progression", 1.0)
    geo.mesh.setTransfiniteCurve(C_top_h,  n_h,   "Progression", 1.0)
    geo.mesh.setTransfiniteCurve(C_bot_h,  n_h,   "Progression", 1.0)

    # Wall-normal seams — Progression with sign matching the curve direction.
    # We always want the SMALL cell at the airfoil (or wake centerline) end.
    #   C_le_rad : LE_FAR (start) -> LE_AF  (end)   small at end   -> coef < 0
    #   C_te_nu  : TE_AF  (start) -> TOP_TE (end)   small at start -> coef > 0
    #   C_te_nl  : TE_AF  (start) -> BOT_TE (end)   small at start -> coef > 0
    #   C_outU   : TOP_OUT(start) -> OUT_MID(end)   small at end   -> coef < 0
    #   C_outL   : OUT_MID(start) -> BOT_OUT(end)   small at start -> coef > 0
    geo.mesh.setTransfiniteCurve(C_le_rad, normal_pts, "Progression", -r_normal)
    geo.mesh.setTransfiniteCurve(C_te_nu,  normal_pts, "Progression", +r_normal)
    geo.mesh.setTransfiniteCurve(C_te_nl,  normal_pts, "Progression", +r_normal)
    geo.mesh.setTransfiniteCurve(C_outU,   normal_pts, "Progression", -r_normal)
    geo.mesh.setTransfiniteCurve(C_outL,   normal_pts, "Progression", +r_normal)

    # Wake direction — small cells at TE end of each wake curve.
    geo.mesh.setTransfiniteCurve(C_wake_ax, wake_pts, "Progression", +r_wake)
    geo.mesh.setTransfiniteCurve(C_top_tail, wake_pts, "Progression", +r_wake)
    geo.mesh.setTransfiniteCurve(C_bot_tail, wake_pts, "Progression", +r_wake)

    # ---- block loops & surfaces -----------------------------------------
    loop_U  = geo.addCurveLoop([+C_af_up,  +C_te_nu,  -C_top_h,  -C_arc_up,  +C_le_rad])
    loop_L  = geo.addCurveLoop([+C_arc_lo, +C_bot_h,  -C_te_nl,  -C_af_lo,   -C_le_rad])
    loop_UW = geo.addCurveLoop([+C_wake_ax, -C_outU,  -C_top_tail, -C_te_nu])
    loop_LW = geo.addCurveLoop([+C_bot_tail, -C_outL, -C_wake_ax, +C_te_nl])

    S_U  = geo.addPlaneSurface([loop_U])
    S_L  = geo.addPlaneSurface([loop_L])
    S_UW = geo.addPlaneSurface([loop_UW])
    S_LW = geo.addPlaneSurface([loop_LW])

    source_surfaces = [S_U, S_L, S_UW, S_LW]
    loop_sizes = [5, 5, 4, 4]

    # Transfinite-surface corners (the order picks the parametric (i,j) axes;
    # gmsh maps them to (0,0)(1,0)(1,1)(0,1) in the listed order).
    geo.mesh.setTransfiniteSurface(S_U,  "Left",
                                    [p_LE_AF,  p_TE_AF, p_TOP_TE, p_LE_FAR])
    geo.mesh.setTransfiniteSurface(S_L,  "Left",
                                    [p_LE_FAR, p_BOT_TE, p_TE_AF, p_LE_AF])
    geo.mesh.setTransfiniteSurface(S_UW, "Left",
                                    [p_TE_AF,  p_OUT_MID, p_TOP_OUT, p_TOP_TE])
    geo.mesh.setTransfiniteSurface(S_LW, "Left",
                                    [p_BOT_TE, p_BOT_OUT, p_OUT_MID, p_TE_AF])

    for s in source_surfaces:
        geo.mesh.setRecombine(2, s)

    # ---- spanwise extrusion ---------------------------------------------
    extr = geo.extrude(
        [(2, s) for s in source_surfaces],
        0.0, 0.0, cfg["spanwise_thickness"] * chord,
        numElements=[cfg["spanwise_layers"]],
        heights=[1.0],
        recombine=True,
    )

    geo.synchronize()

    # ---- slice extrude result by per-block loop size --------------------
    offsets = _block_offsets(loop_sizes)
    per_block = {}
    for name, off, n in zip(["U", "L", "UW", "LW"], offsets, loop_sizes):
        block_entities = extr[off : off + 2 + n]
        per_block[name] = {
            "top":      block_entities[0][1],     # (2, tag) — back face at +z
            "volume":   block_entities[1][1],     # (3, tag)
            "laterals": [t for d, t in block_entities[2 : 2 + n] if d == 2],
        }

    # ---- classify laterals into aerofoil / freestream / internal seam ----
    airfoil_curves  = {C_af_up, C_af_lo}
    farfield_curves = {C_arc_up, C_arc_lo, C_top_h, C_bot_h,
                       C_top_tail, C_bot_tail, C_outU, C_outL}

    aerofoil_tags: set[int] = set()
    freestream_tags: set[int] = set()
    all_laterals: set[int] = set()
    for b in per_block.values():
        all_laterals.update(b["laterals"])
    for s_tag in all_laterals:
        kind = _classify_lateral(gmsh, s_tag, airfoil_curves, farfield_curves)
        if kind == "aerofoil":
            aerofoil_tags.add(s_tag)
        elif kind == "freestream":
            freestream_tags.add(s_tag)
        # 'internal' seams get no physical group

    if not aerofoil_tags:
        raise RuntimeError("Failed to identify any aerofoil surfaces after extrusion")
    if not freestream_tags:
        raise RuntimeError("Failed to identify any freestream surfaces after extrusion")

    front_back_tags = list(source_surfaces) + [b["top"] for b in per_block.values()]
    volume_tags     = [b["volume"] for b in per_block.values()]

    # ---- physical groups (named for gmshToFoam) -------------------------
    g_front_back = gmsh.model.addPhysicalGroup(2, front_back_tags)
    gmsh.model.setPhysicalName(2, g_front_back, "frontAndBack")
    g_aerofoil   = gmsh.model.addPhysicalGroup(2, sorted(aerofoil_tags))
    gmsh.model.setPhysicalName(2, g_aerofoil, "aerofoil")
    g_freestream = gmsh.model.addPhysicalGroup(2, sorted(freestream_tags))
    gmsh.model.setPhysicalName(2, g_freestream, "freestream")
    g_fluid = gmsh.model.addPhysicalGroup(3, volume_tags)
    gmsh.model.setPhysicalName(3, g_fluid, "fluid")

    # ---- mesh --------------------------------------------------------------
    gmsh.model.mesh.generate(3)
    _, elem_tags, _ = gmsh.model.mesh.getElements(3)
    cell_count = int(sum(len(t) for t in elem_tags))

    mesh_path = case_dir / "mesh.msh"
    gmsh.write(str(mesh_path))

    runtime = time.perf_counter() - start_time

    return {
        "first_cell_height": float(h1),
        "normal_progression": float(r_normal),
        "wake_progression":   float(r_wake),
        "n_arc_pts":          int(n_arc),
        "n_horiz_pts":        int(n_h),
        "cell_count_gmsh":    cell_count,
        "mesh_runtime_s":     float(runtime),
    }
