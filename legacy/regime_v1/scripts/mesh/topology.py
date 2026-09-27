"""
Structured C+H transfinite mesh for a NACA airfoil.

Two trailing-edge topologies, dispatched on ``te_chord_fraction``:

  * ``te_chord_fraction == 1.0`` -> SHARP TE (``_build_sharp_c_grid``). The
    upper/lower splines meet at a single TE point, so there is no blunt base,
    no ``h_te`` step, and no inner/outer wake band. Every wake column is
    exactly ``normal_pts`` tall, the mesh stays all-hex, and the wake cut is a
    plain branch cut from the TE. This is the textbook airfoil C-grid and the
    preferred path (see ``regime_parameters.py``).

  * ``te_chord_fraction < 1.0`` -> BLUNT TE (``build_c_grid`` body, the
    10-block topology documented below). Retained for cases that deliberately
    want a finite-thickness base; it carries the blunt count in the wake
    transverse seam, which the sharp path avoids entirely.

The blunt 10-block topology is documented below.

Blunt-TE topology
=================

Ten transfinite quad blocks. The airfoil is truncated at x = te_chord_fraction
so the trailing edge has a finite half-thickness h_te (NACA-4 thickness
evaluated at the truncation x). The blunt back is a wall, split into
upper (TE_UP -> TE_MID) and lower (TE_MID -> TE_LO) halves. The wake
centreline starts at TE_MID.

The wake (transition + main, upper + lower) is split at y = +/- h_te into an
INNER band (centreline -> h_te) and an OUTER band (h_te -> farfield):
  - the inner band carries the blunt-TE cell count (blunt_pts), kept thin and
    confined near the centreline so it no longer fans out into the far field;
  - the outer band carries normal_pts, so the transition->main interface ties
    to normal_pts (matching the airfoil wall-normal seam te_nu/te_nl).
This avoids the old COMPOUND west side (te_nu + blunt back) whose summed count
(normal_pts + n_blunt_cells) on the opposite east edge fanned out into extreme
aspect-ratio / near-90deg cells. The inner blocks use the blunt-back grading on
both transverse edges and the outer TRANSITION block uses te_nu grading on both
(skew-free); only the outer MAIN block coarsens transversely toward the outlet
(wake_centreline_h), a gentle far-field fan-out.

                F_TOP_MID -- F_TOP_TE --- T_TOP ---- F_TOP_OUT
                  /            | UTW_out  | UMW_out   |
                / Block U      TE_UP ---- TI_UP ----- OI_UP     (y = +h_te)
   F_LE_FAR --                 | UTW_in   | UMW_in    |
                / airfoil      TE_MID --- T_MID ----- OUT_MID   (y = 0, wake CL)
                \\             | LTW_in   | LMW_in    |
                 \\ Block L    TE_LO ---- TI_LO ----- OI_LO     (y = -h_te)
                  \\           | LTW_out  | LMW_out   |
                F_BOT_MID -- F_BOT_TE --- T_BOT ---- F_BOT_OUT

Curve direction conventions (start -> end)
------------------------------------------
  Airfoil:
    C_af_up      : LE_AF  -> TE_UP    (upper airfoil spline; ends at blunt TE corner)
    C_af_lo      : LE_AF  -> TE_LO    (lower airfoil spline; ends at blunt TE corner)
    C_te_blunt_up: TE_UP  -> TE_MID   (upper half of blunt back; AIRFOIL wall)
    C_te_blunt_lo: TE_MID -> TE_LO    (lower half of blunt back; AIRFOIL wall)

  Farfield:
    C_arc_up   : LE_FAR -> TOP_MID
    C_arc_lo   : LE_FAR -> BOT_MID
    C_top_h    : TOP_MID -> TOP_TE
    C_bot_h    : BOT_MID -> BOT_TE
    C_top_trans  : TOP_TE -> T_TOP
    C_top_main   : T_TOP  -> TOP_OUT
    C_bot_trans  : BOT_TE -> T_BOT
    C_bot_main   : T_BOT  -> BOT_OUT
    C_out_up_in  : OUT_MID -> OI_UP    (outlet, inner band, upper)
    C_out_up_out : OI_UP   -> TOP_OUT  (outlet, outer band, upper)
    C_out_lo_in  : OUT_MID -> OI_LO    (outlet, inner band, lower)
    C_out_lo_out : OI_LO   -> BOT_OUT  (outlet, outer band, lower)

  Internal seams (interior to the fluid domain):
    C_le_rad     : LE_FAR -> LE_AF
    C_te_nu      : TE_UP  -> TOP_TE   (wall-normal seam above blunt corner)
    C_te_nl      : TE_LO  -> BOT_TE   (wall-normal seam below blunt corner)
    C_wake_trans : TE_MID -> T_MID    (wake centreline, transition block)
    C_wake_main  : T_MID  -> OUT_MID  (wake centreline, main block)
    C_hseam_t_up : TE_UP  -> TI_UP    (inner/outer seam @ +h_te, transition)
    C_hseam_t_lo : TE_LO  -> TI_LO    (inner/outer seam @ -h_te, transition)
    C_hseam_m_up : TI_UP  -> OI_UP    (inner/outer seam @ +h_te, main)
    C_hseam_m_lo : TI_LO  -> OI_LO    (inner/outer seam @ -h_te, main)
    C_vseam_t_up_in  : T_MID -> TI_UP (transition->main interface, inner, upper)
    C_vseam_t_up_out : TI_UP -> T_TOP (transition->main interface, outer, upper)
    C_vseam_t_lo_in  : T_MID -> TI_LO (transition->main interface, inner, lower)
    C_vseam_t_lo_out : TI_LO -> T_BOT (transition->main interface, outer, lower)

Block loops (signed curve tags, CCW with interior on the left)
--------------------------------------------------------------
  Block U    : [+af_up,      +te_nu,           -top_h,       -arc_up, +le_rad]
  Block L    : [+arc_lo,     +bot_h,           -te_nl,       -af_lo,  -le_rad]
  Block UTW_in : [+wake_trans, +vseam_t_up_in,  -hseam_t_up, +te_blunt_up]
  Block UTW_out: [+hseam_t_up,  +vseam_t_up_out, -top_trans,  -te_nu]
  Block UMW_in : [+wake_main,   +out_up_in,      -hseam_m_up, -vseam_t_up_in]
  Block UMW_out: [+hseam_m_up,  +out_up_out,     -top_main,   -vseam_t_up_out]
  Block LTW_in : [+hseam_t_lo,  -vseam_t_lo_in,  -wake_trans, +te_blunt_lo]
  Block LTW_out: [+bot_trans,   -vseam_t_lo_out, -hseam_t_lo, +te_nl]
  Block LMW_in : [+hseam_m_lo,  -out_lo_in,      -wake_main,  +vseam_t_lo_in]
  Block LMW_out: [+bot_main,    -out_lo_out,     -hseam_m_lo, +vseam_t_lo_out]

Every wake block is a clean 4-corner quad; TE_UP/TE_LO are now corners of the
inner transition blocks, so no compound sides remain.

Transfinite consistency
-----------------------
  N_af_up      = N_af_lo                                    = chord_pts
  N_arc + N_h - 1                                           = chord_pts
  N_le_rad     = N_te_nu = N_te_nl                          = normal_pts
  N_te_blunt_up = N_te_blunt_lo                             = te_blunt_pts (blunt_pts)
  OUTER transverse (h_te -> farfield), normal_pts:
    N_vseam_t_up_out = N_vseam_t_lo_out = N_out_up_out = N_out_lo_out = normal_pts
  INNER transverse (centreline -> h_te), blunt_pts:
    N_vseam_t_up_in  = N_vseam_t_lo_in  = N_out_up_in  = N_out_lo_in  = blunt_pts
  N_wake_trans = N_top_trans = N_bot_trans = N_hseam_t_* = transition_wake_pts
  N_wake_main  = N_top_main  = N_bot_main  = N_hseam_m_* = wake_pts
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np

from .boundary_layer import first_cell_height, solve_progression
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


def _bump_endpoint_spacing(length: float, n_cells: int, beta: float) -> float:
    """Approximate spacing of the end cell under gmsh's Bump law.

    gmsh's Bump law places cell-size as a beta-shape function of parameter t,
    with the cell at the endpoint approximately `beta * L / N` for beta < 1.
    Good enough as a target for the transition block's first cell; the
    realised transition first-cell is logged as a diagnostic.
    """
    return beta * length / n_cells


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
# Shared gmsh setup / finalisation (used by both TE topologies)
# ---------------------------------------------------------------------------

def _gmsh_setup(gmsh_module, model_name: str) -> None:
    """Reset gmsh and apply the structured-quad meshing options shared by both
    the sharp and blunt builders."""
    gmsh_module.clear()
    gmsh_module.model.add(model_name)
    gmsh_module.option.setNumber("General.Terminal", 0)
    # Geometry.Tolerance is the node-coincidence threshold used when the geo
    # kernel removes duplicate points on synchronize. Its default (1e-8) is
    # RELATIVE to the domain bounding box (~50c here), so a first cell below
    # ~5e-7 m makes wall-adjacent nodes look coincident and gmsh collapses them
    # into degenerate (triangular) quads -> gmshToFoam fails. Regime B/C resolve
    # y+<1 with first cells ~3e-7, so drop the tolerance well below that.
    gmsh_module.option.setNumber("Geometry.Tolerance", 1e-12)
    gmsh_module.option.setNumber("Mesh.MshFileVersion", 2.2)
    gmsh_module.option.setNumber("Mesh.SaveAll", 0)
    gmsh_module.option.setNumber("Mesh.Algorithm", 8)               # Frontal-Delaunay-for-Quads (fallback)
    gmsh_module.option.setNumber("Mesh.RecombinationAlgorithm", 1)  # Blossom
    gmsh_module.option.setNumber("Mesh.RecombineAll", 1)
    gmsh_module.option.setNumber("Mesh.ElementOrder", 1)
    gmsh_module.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh_module.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    gmsh_module.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)


def _finalize_and_write(
    gmsh_module, geo, source_surfaces: list[int], loop_sizes: list[int],
    block_names: list[str], airfoil_curves: set[int], farfield_curves: set[int],
    case_dir: Path, cfg: dict, chord: float, start_time: float,
) -> tuple[int, float]:
    """Spanwise-extrude the 2-D block surfaces, classify the lateral patches
    into aerofoil / freestream / internal, tag physical groups, mesh, and write
    ``mesh.msh``. Shared by the sharp and blunt builders.

    Returns ``(cell_count, runtime_s)``.
    """
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
    per_block: dict[str, dict] = {}
    for name, off, n in zip(block_names, offsets, loop_sizes):
        block_entities = extr[off : off + 2 + n]
        per_block[name] = {
            "top":      block_entities[0][1],     # (2, tag) — back face at +z
            "volume":   block_entities[1][1],     # (3, tag)
            "laterals": [t for d, t in block_entities[2 : 2 + n] if d == 2],
        }

    # ---- classify laterals into aerofoil / freestream / internal seam ----
    aerofoil_tags: set[int] = set()
    freestream_tags: set[int] = set()
    all_laterals: set[int] = set()
    for b in per_block.values():
        all_laterals.update(b["laterals"])
    for s_tag in all_laterals:
        kind = _classify_lateral(gmsh_module, s_tag, airfoil_curves, farfield_curves)
        if kind == "aerofoil":
            aerofoil_tags.add(s_tag)
        elif kind == "freestream":
            freestream_tags.add(s_tag)
        # 'internal' seams (incl. the wake branch cut) get no physical group

    if not aerofoil_tags:
        raise RuntimeError("Failed to identify any aerofoil surfaces after extrusion")
    if not freestream_tags:
        raise RuntimeError("Failed to identify any freestream surfaces after extrusion")

    front_back_tags = list(source_surfaces) + [b["top"] for b in per_block.values()]
    volume_tags     = [b["volume"] for b in per_block.values()]

    # ---- physical groups (named for gmshToFoam) -------------------------
    g_front_back = gmsh_module.model.addPhysicalGroup(2, front_back_tags)
    gmsh_module.model.setPhysicalName(2, g_front_back, "frontAndBack")
    g_aerofoil   = gmsh_module.model.addPhysicalGroup(2, sorted(aerofoil_tags))
    gmsh_module.model.setPhysicalName(2, g_aerofoil, "aerofoil")
    g_freestream = gmsh_module.model.addPhysicalGroup(2, sorted(freestream_tags))
    gmsh_module.model.setPhysicalName(2, g_freestream, "freestream")
    g_fluid = gmsh_module.model.addPhysicalGroup(3, volume_tags)
    gmsh_module.model.setPhysicalName(3, g_fluid, "fluid")

    # ---- mesh -----------------------------------------------------------
    gmsh_module.model.mesh.generate(3)
    _, elem_tags, _ = gmsh_module.model.mesh.getElements(3)
    cell_count = int(sum(len(t) for t in elem_tags))

    gmsh_module.write(str(case_dir / "mesh.msh"))
    runtime = time.perf_counter() - start_time
    return cell_count, runtime


# ---------------------------------------------------------------------------
# Sharp-TE builder (preferred; te_chord_fraction == 1.0)
# ---------------------------------------------------------------------------

def _build_sharp_c_grid(
    gmsh_module, case_dir: Path, params: dict, cfg: dict, chord: float,
) -> dict[str, float]:
    """Clean 6-block transfinite C-grid for a SHARP (closed) trailing edge.

    The upper/lower airfoil splines meet at a single TE point, so there is no
    blunt base and no h_te step: every wake column is exactly ``normal_pts``
    tall and the mesh is all-hex. Blocks:

        U, L          : around the airfoil (5-edge, north split arc+horizontal)
        UTW, UMW      : upper wake — transition then main
        LTW, LMW      : lower wake — transition then main

    The wake centreline (TE -> T_MID -> OUT_MID) is an internal branch cut
    shared by the upper and lower wake blocks. The wall-normal seam grading
    (te_nu/te_nl) is reused on the wake transverse seams so the U/UTW and
    UTW/UMW interfaces are conformal in count AND distribution (skew-free).
    """
    start_time = time.perf_counter()

    # ---- physics-derived spacings ----------------------------------------
    y_plus_mesh = cfg["y_plus_target"] * cfg.get("y_plus_mesh_factor", 1.0)
    h1 = first_cell_height(params["Re"], y_plus_mesh, chord=chord)

    # ---- farfield geometry (TE column anchored at x = chord) -------------
    ff = farfield_points(
        upstream_radius=cfg["upstream_radius"],
        downstream_length=cfg["downstream_length"],
        transverse_extent=cfg["transverse_extent"],
        transition_wake_length=cfg["transition_wake_length"],
        chord=chord,
        te_chord_fraction=1.0,
    )
    Rc = cfg["upstream_radius"] * chord
    Lw = cfg["downstream_length"] * chord
    Lt = cfg["transition_wake_length"] * chord
    Lm = Lw - Lt
    Ht = cfg["transverse_extent"] * chord
    te_x = chord

    # ---- wall-normal progression (match h1 across the upstream radius) ---
    n_normal_cells = cfg["normal_pts"] - 1
    r_normal = solve_progression(h1=h1, total_length=Rc, n_cells=n_normal_cells)
    hint_r = cfg["bl_growth_ratio"]
    if abs(r_normal - hint_r) / hint_r > 0.15:
        log.warning(
            "%s: derived normal-direction progression %.4f deviates from regime hint %.4f "
            "(h1=%.3e m, n=%d, L=%.2f m). Adjust normal_pts or y_plus_target.",
            case_dir.name, r_normal, hint_r, h1, n_normal_cells, Rc,
        )

    r_wake_main = float(cfg["wake_progression"])

    # ---- transition-wake progression: match the airfoil TE cell ----------
    n_trans_cells = cfg["transition_wake_pts"] - 1
    h_te_target = _bump_endpoint_spacing(
        length=te_x, n_cells=cfg["chord_pts_upper"] - 1, beta=cfg["le_te_cluster"],
    )
    explicit_r_trans = cfg.get("transition_wake_progression")
    if explicit_r_trans is None:
        try:
            r_wake_trans = solve_progression(
                h1=h_te_target, total_length=Lt, n_cells=n_trans_cells,
            )
        except ValueError as exc:
            raise RuntimeError(
                f"{case_dir.name}: cannot solve transition wake progression "
                f"(h_TE_target={h_te_target:.3e}, Lt={Lt:.3f}, n={n_trans_cells}). "
                f"Adjust transition_wake_length or transition_wake_pts. "
                f"Underlying error: {exc}"
            ) from exc
    else:
        r_wake_trans = float(explicit_r_trans)

    # ---- outer-band outlet progression (wake_centreline_h) ---------------
    # The main wake grades its transverse first cell from h1 (transition
    # interface, west) to wake_centreline_h (outlet, east); None -> r_normal
    # (no far-wake coarsening). The transition block keeps r_normal on both
    # transverse edges (skew-free). With no blunt base, the seam spans the full
    # transverse height Ht.
    h_seam_first = cfg.get("wake_centreline_h") or h1
    if h_seam_first < h1:
        log.warning(
            "%s: wake_centreline_h (%.3e) is finer than the wall h1 (%.3e); "
            "this defeats the AR/non-orthogonality decoupling.",
            case_dir.name, h_seam_first, h1,
        )
    r_seam_outer = solve_progression(
        h1=h_seam_first, total_length=Ht, n_cells=n_normal_cells
    )

    # ---- surface geometry (sharp closed TE) ------------------------------
    upper, lower = naca_symmetric(
        params["thickness"], n=cfg["surface_points"], te_chord_fraction=1.0
    )

    # ---- gmsh setup ------------------------------------------------------
    _gmsh_setup(gmsh_module, case_dir.name)
    geo = gmsh_module.model.geo

    # ---- corner points (TE is a single shared point) --------------------
    p_LE_AF   = geo.addPoint(0.0,  0.0, 0.0)
    p_TE      = geo.addPoint(te_x, 0.0, 0.0)
    p_LE_FAR  = geo.addPoint(*ff.LE_FAR,  0.0)
    p_TOP_MID = geo.addPoint(*ff.TOP_MID, 0.0)
    p_BOT_MID = geo.addPoint(*ff.BOT_MID, 0.0)
    p_TOP_TE  = geo.addPoint(*ff.TOP_TE,  0.0)
    p_BOT_TE  = geo.addPoint(*ff.BOT_TE,  0.0)
    p_T_TOP   = geo.addPoint(*ff.T_TOP,   0.0)
    p_T_BOT   = geo.addPoint(*ff.T_BOT,   0.0)
    p_T_MID   = geo.addPoint(*ff.T_MID,   0.0)
    p_TOP_OUT = geo.addPoint(*ff.TOP_OUT, 0.0)
    p_BOT_OUT = geo.addPoint(*ff.BOT_OUT, 0.0)
    p_OUT_MID = geo.addPoint(*ff.OUT_MID, 0.0)

    # ---- airfoil splines (LE -> single sharp TE point) ------------------
    up_inner = [geo.addPoint(float(x), float(y), 0.0) for x, y in upper[1:-1]]
    lo_inner = [geo.addPoint(float(x), float(y), 0.0) for x, y in lower[1:-1]]
    C_af_up = geo.addSpline([p_LE_AF, *up_inner, p_TE])
    C_af_lo = geo.addSpline([p_LE_AF, *lo_inner, p_TE])

    # ---- farfield curves -------------------------------------------------
    C_arc_up    = geo.addCircleArc(p_LE_FAR, p_LE_AF, p_TOP_MID)
    C_arc_lo    = geo.addCircleArc(p_LE_FAR, p_LE_AF, p_BOT_MID)
    C_top_h     = geo.addLine(p_TOP_MID, p_TOP_TE)
    C_bot_h     = geo.addLine(p_BOT_MID, p_BOT_TE)
    C_top_trans = geo.addLine(p_TOP_TE, p_T_TOP)
    C_top_main  = geo.addLine(p_T_TOP,  p_TOP_OUT)
    C_bot_trans = geo.addLine(p_BOT_TE, p_T_BOT)
    C_bot_main  = geo.addLine(p_T_BOT,  p_BOT_OUT)
    C_out_up    = geo.addLine(p_OUT_MID, p_TOP_OUT)   # outlet, upper half
    C_out_lo    = geo.addLine(p_OUT_MID, p_BOT_OUT)   # outlet, lower half

    # ---- internal seams --------------------------------------------------
    C_le_rad     = geo.addLine(p_LE_FAR, p_LE_AF)
    C_te_nu      = geo.addLine(p_TE, p_TOP_TE)        # wall-normal seam above TE
    C_te_nl      = geo.addLine(p_TE, p_BOT_TE)        # wall-normal seam below TE
    C_wake_trans = geo.addLine(p_TE,    p_T_MID)      # wake cut, transition
    C_wake_main  = geo.addLine(p_T_MID, p_OUT_MID)    # wake cut, main
    C_vseam_t_up = geo.addLine(p_T_MID, p_T_TOP)      # transition->main, upper
    C_vseam_t_lo = geo.addLine(p_T_MID, p_T_BOT)      # transition->main, lower

    # ---- transfinite line counts ----------------------------------------
    chord_pts = cfg["chord_pts_upper"]
    if cfg["chord_pts_lower"] != chord_pts:
        raise ValueError(
            f"chord_pts_upper ({chord_pts}) must equal chord_pts_lower "
            f"({cfg['chord_pts_lower']}) for symmetric NACA mesh."
        )
    n_arc, n_h = _split_north_counts(chord_pts, cfg["north_arc_to_horiz_ratio"])
    normal_pts = cfg["normal_pts"]
    wake_pts   = cfg["wake_pts"]
    trans_pts  = cfg["transition_wake_pts"]
    bump = float(cfg["le_te_cluster"])

    # Airfoil — Bump clusters at both LE and TE.
    geo.mesh.setTransfiniteCurve(C_af_up, chord_pts, "Bump", bump)
    geo.mesh.setTransfiniteCurve(C_af_lo, chord_pts, "Bump", bump)

    # North-edge segments — uniform.
    geo.mesh.setTransfiniteCurve(C_arc_up, n_arc, "Progression", 1.0)
    geo.mesh.setTransfiniteCurve(C_arc_lo, n_arc, "Progression", 1.0)
    geo.mesh.setTransfiniteCurve(C_top_h,  n_h,   "Progression", 1.0)
    geo.mesh.setTransfiniteCurve(C_bot_h,  n_h,   "Progression", 1.0)

    # Wall-normal seams — small cell at the wall/TE (y=0) end.
    #   C_le_rad : LE_FAR -> LE_AF   small at end   | -r_normal
    #   C_te_nu  : TE     -> TOP_TE  small at start | +r_normal
    #   C_te_nl  : TE     -> BOT_TE  small at start | +r_normal
    geo.mesh.setTransfiniteCurve(C_le_rad, normal_pts, "Progression", -r_normal)
    geo.mesh.setTransfiniteCurve(C_te_nu,  normal_pts, "Progression", +r_normal)
    geo.mesh.setTransfiniteCurve(C_te_nl,  normal_pts, "Progression", +r_normal)

    # Wake transverse seams — normal_pts, small cell at the wake centreline
    # (y=0) end so the count AND distribution match te_nu/te_nl across the
    # U/UTW and UTW/UMW interfaces. The main-wake outlet uses r_seam_outer
    # (wake_centreline_h) to coarsen the far wake transversely.
    #   C_vseam_t_up : T_MID   -> T_TOP    small at start (T_MID)   | +r_normal
    #   C_vseam_t_lo : T_MID   -> T_BOT    small at start (T_MID)   | +r_normal
    #   C_out_up     : OUT_MID -> TOP_OUT  small at start (OUT_MID) | +r_seam_outer
    #   C_out_lo     : OUT_MID -> BOT_OUT  small at start (OUT_MID) | +r_seam_outer
    geo.mesh.setTransfiniteCurve(C_vseam_t_up, normal_pts, "Progression", +r_normal)
    geo.mesh.setTransfiniteCurve(C_vseam_t_lo, normal_pts, "Progression", +r_normal)
    geo.mesh.setTransfiniteCurve(C_out_up,     normal_pts, "Progression", +r_seam_outer)
    geo.mesh.setTransfiniteCurve(C_out_lo,     normal_pts, "Progression", +r_seam_outer)

    # Transition wake (streamwise) — small cells at the TE end.
    geo.mesh.setTransfiniteCurve(C_wake_trans, trans_pts, "Progression", +r_wake_trans)
    geo.mesh.setTransfiniteCurve(C_top_trans,  trans_pts, "Progression", +r_wake_trans)
    geo.mesh.setTransfiniteCurve(C_bot_trans,  trans_pts, "Progression", +r_wake_trans)

    # Main wake (streamwise) — small cells at the transition->main interface.
    geo.mesh.setTransfiniteCurve(C_wake_main, wake_pts, "Progression", +r_wake_main)
    geo.mesh.setTransfiniteCurve(C_top_main,  wake_pts, "Progression", +r_wake_main)
    geo.mesh.setTransfiniteCurve(C_bot_main,  wake_pts, "Progression", +r_wake_main)

    # ---- block loops & surfaces (6 clean quads + 2 five-edge airfoil) ----
    loop_U   = geo.addCurveLoop([+C_af_up,  +C_te_nu, -C_top_h, -C_arc_up, +C_le_rad])
    loop_L   = geo.addCurveLoop([+C_arc_lo, +C_bot_h, -C_te_nl, -C_af_lo,  -C_le_rad])
    loop_UTW = geo.addCurveLoop([+C_wake_trans, +C_vseam_t_up, -C_top_trans, -C_te_nu])
    loop_UMW = geo.addCurveLoop([+C_wake_main,  +C_out_up,     -C_top_main,  -C_vseam_t_up])
    loop_LTW = geo.addCurveLoop([+C_bot_trans,  -C_vseam_t_lo, -C_wake_trans, +C_te_nl])
    loop_LMW = geo.addCurveLoop([+C_bot_main,   -C_out_lo,     -C_wake_main,  +C_vseam_t_lo])

    S_U   = geo.addPlaneSurface([loop_U])
    S_L   = geo.addPlaneSurface([loop_L])
    S_UTW = geo.addPlaneSurface([loop_UTW])
    S_UMW = geo.addPlaneSurface([loop_UMW])
    S_LTW = geo.addPlaneSurface([loop_LTW])
    S_LMW = geo.addPlaneSurface([loop_LMW])

    source_surfaces = [S_U, S_L, S_UTW, S_UMW, S_LTW, S_LMW]
    loop_sizes      = [5, 5, 4, 4, 4, 4]
    block_names     = ["U", "L", "UTW", "UMW", "LTW", "LMW"]

    # Transfinite-surface corners (CCW; matches each loop's corner order).
    geo.mesh.setTransfiniteSurface(S_U,   "Left", [p_LE_AF,  p_TE,      p_TOP_TE,  p_LE_FAR])
    geo.mesh.setTransfiniteSurface(S_L,   "Left", [p_LE_FAR, p_BOT_TE,  p_TE,      p_LE_AF])
    geo.mesh.setTransfiniteSurface(S_UTW, "Left", [p_TE,     p_T_MID,   p_T_TOP,   p_TOP_TE])
    geo.mesh.setTransfiniteSurface(S_UMW, "Left", [p_T_MID,  p_OUT_MID, p_TOP_OUT, p_T_TOP])
    geo.mesh.setTransfiniteSurface(S_LTW, "Left", [p_BOT_TE, p_T_BOT,   p_T_MID,   p_TE])
    geo.mesh.setTransfiniteSurface(S_LMW, "Left", [p_T_BOT,  p_BOT_OUT, p_OUT_MID, p_T_MID])

    for s in source_surfaces:
        geo.mesh.setRecombine(2, s)

    # ---- spanwise extrusion + finalise ----------------------------------
    airfoil_curves  = {C_af_up, C_af_lo}
    farfield_curves = {
        C_arc_up, C_arc_lo, C_top_h, C_bot_h,
        C_top_trans, C_top_main, C_bot_trans, C_bot_main,
        C_out_up, C_out_lo,
    }
    cell_count, runtime = _finalize_and_write(
        gmsh_module, geo, source_surfaces, loop_sizes, block_names,
        airfoil_curves, farfield_curves, case_dir, cfg, chord, start_time,
    )

    # ---- diagnostics -----------------------------------------------------
    if abs(r_wake_trans - 1.0) < 1e-9:
        h_trans_first = Lt / n_trans_cells
    else:
        h_trans_first = Lt * (r_wake_trans - 1.0) / (r_wake_trans ** n_trans_cells - 1.0)
    if abs(r_wake_main - 1.0) < 1e-9:
        h_main_first = Lm / (wake_pts - 1)
    else:
        h_main_first = Lm * (r_wake_main - 1.0) / (r_wake_main ** (wake_pts - 1) - 1.0)
    h_trans_last = h_trans_first * (r_wake_trans ** (n_trans_cells - 1))

    log.info(
        "%s [sharp TE] cells=%d | wake column = normal_pts (%d) tall | "
        "h_TE_target=%.3e trans_first=%.3e (ratio=%.2f) trans_last=%.3e "
        "main_first=%.3e",
        case_dir.name, cell_count, normal_pts,
        h_te_target, h_trans_first, h_trans_first / max(h_te_target, 1e-30),
        h_trans_last, h_main_first,
    )

    return {
        "first_cell_height":           float(h1),
        "normal_progression":          float(r_normal),
        "seam_progression":            float(r_seam_outer),
        "wake_progression":            float(r_wake_main),
        "transition_wake_progression": float(r_wake_trans),
        "h_te_airfoil_target":         float(h_te_target),
        "h_trans_first":               float(h_trans_first),
        "h_trans_last":                float(h_trans_last),
        "h_main_first":                float(h_main_first),
        "te_chord_fraction":           1.0,
        "te_half_thickness":           0.0,
        "te_blunt_progression":        None,
        "n_arc_pts":                   int(n_arc),
        "n_horiz_pts":                 int(n_h),
        "n_seam_pts":                  int(normal_pts),
        "n_blunt_inner_pts":           0,
        "cell_count_gmsh":             int(cell_count),
        "mesh_runtime_s":              float(runtime),
    }


# ---------------------------------------------------------------------------
# Blunt-TE builder (legacy; te_chord_fraction < 1.0)
# ---------------------------------------------------------------------------

def build_c_grid(
    case_dir: Path,
    params: dict,
    cfg: dict,
    chord: float = 1.0,
) -> dict[str, float]:
    """Build the 6-block transfinite C+H mesh (with transition wake) for one case.

    Writes <case_dir>/mesh.msh. Returns a dict of mesh metrics suitable for
    merging into `case_metadata.json` (cell count, first-cell height used,
    derived progression ratios, etc.).
    """
    import gmsh  # local import — keeps gmsh dep optional at module-import time

    # Dispatch: a sharp (closed) TE uses the clean 6-block C-grid, where every
    # wake column is exactly normal_pts tall and no blunt count enters the wake
    # transverse seam. Only te_chord_fraction < 1.0 falls through to the blunt
    # 10-block topology below.
    if float(cfg["te_chord_fraction"]) >= 1.0 - 1e-12:
        return _build_sharp_c_grid(gmsh, case_dir, params, cfg, chord)

    start_time = time.perf_counter()

    # ---- physics-derived spacings ----------------------------------------
    # y_plus_mesh_factor (default 1.0) shrinks the meshing y+ below the
    # physical target where the flat-plate Cf correlation under-predicts the
    # true wall shear (e.g. Regime B's near-stall LE acceleration). The
    # regime's recorded y_plus_target is unchanged; only the cell size is.
    y_plus_mesh = cfg["y_plus_target"] * cfg.get("y_plus_mesh_factor", 1.0)
    h1 = first_cell_height(params["Re"], y_plus_mesh, chord=chord)

    # ---- blunt trailing edge --------------------------------------------
    te_frac = float(cfg["te_chord_fraction"])
    if not (0.5 < te_frac < 1.0):
        raise ValueError(
            f"{case_dir.name}: te_chord_fraction must lie in (0.5, 1.0), "
            f"got {te_frac}. The blunt-TE topology requires a truncated airfoil."
        )
    # te_blunt_pts accepts:
    #   "auto" — defer to the geometry-derived maximum (set below once
    #            h_nu_first / h_te are known).
    #   int N>=3 — explicit ceiling; the adaptive solver may still reduce it.
    raw_blunt = cfg["te_blunt_pts"]
    auto_blunt = isinstance(raw_blunt, str) and raw_blunt.lower() == "auto"
    if auto_blunt:
        blunt_pts = None
        n_blunt_cells = None
    else:
        blunt_pts = int(raw_blunt)
        if blunt_pts < 3:
            raise ValueError(
                f"{case_dir.name}: te_blunt_pts must be >= 3 or 'auto' "
                f"(got {raw_blunt}); the blunt-back wall needs at least "
                f"two cells per half."
            )
        n_blunt_cells = blunt_pts - 1
    te_x = te_frac * chord

    # ---- topology dimensions --------------------------------------------
    ff = farfield_points(
        upstream_radius=cfg["upstream_radius"],
        downstream_length=cfg["downstream_length"],
        transverse_extent=cfg["transverse_extent"],
        transition_wake_length=cfg["transition_wake_length"],
        chord=chord,
        te_chord_fraction=te_frac,
    )
    Rc = cfg["upstream_radius"] * chord
    Lw = cfg["downstream_length"] * chord
    Lt = cfg["transition_wake_length"] * chord
    Lm = Lw - Lt                                # length of MAIN wake block
    Ht = cfg["transverse_extent"] * chord

    # Wall-normal progression: derived to match h1 exactly across n cells
    # covering the whole upstream radius.
    n_normal_cells = cfg["normal_pts"] - 1
    r_normal = solve_progression(h1=h1, total_length=Rc, n_cells=n_normal_cells)
    hint_r = cfg["bl_growth_ratio"]
    if abs(r_normal - hint_r) / hint_r > 0.15:
        log.warning(
            "%s: derived normal-direction progression %.4f deviates from regime hint %.4f "
            "(h1=%.3e m, n=%d, L=%.2f m). Adjust normal_pts or y_plus_target.",
            case_dir.name, r_normal, hint_r, h1, n_normal_cells, Rc,
        )

    r_wake_main = float(cfg["wake_progression"])

    # ---- transition-wake progression: auto-derive to match airfoil TE cell
    n_trans_cells = cfg["transition_wake_pts"] - 1
    h_te_target = _bump_endpoint_spacing(
        length=te_x,
        n_cells=cfg["chord_pts_upper"] - 1,
        beta=cfg["le_te_cluster"],
    )
    explicit_r_trans = cfg.get("transition_wake_progression")
    if explicit_r_trans is None:
        try:
            r_wake_trans = solve_progression(
                h1=h_te_target,
                total_length=Lt,
                n_cells=n_trans_cells,
            )
        except ValueError as exc:
            raise RuntimeError(
                f"{case_dir.name}: cannot solve transition wake progression "
                f"(h_TE_target={h_te_target:.3e}, Lt={Lt:.3f}, n={n_trans_cells}). "
                f"Adjust transition_wake_length or transition_wake_pts. "
                f"Underlying error: {exc}"
            ) from exc
    else:
        r_wake_trans = float(explicit_r_trans)

    # ---- surface geometry ------------------------------------------------
    upper, lower = naca_symmetric(
        params["thickness"], n=cfg["surface_points"], te_chord_fraction=te_frac
    )
    h_te = float(upper[-1, 1])
    if h_te <= 0.0:
        raise RuntimeError(
            f"{case_dir.name}: blunt-TE half-thickness came out non-positive "
            f"(h_te={h_te:.3e}). Check te_chord_fraction and airfoil thickness."
        )

    # ---- blunt-back progression: match the first cell of te_nu at TE_UP --
    # te_nu spans (Ht - h_te) with normal_pts cells using r_normal grading
    # (small cells at TE_UP end). The blunt-back's first cell at TE_UP should
    # match te_nu's first cell so the west compound side of UTW has a smooth
    # cell-size transition through the intermediate point TE_UP.
    L_te_nu = Ht - h_te
    if abs(r_normal - 1.0) < 1e-12:
        h_nu_first = L_te_nu / n_normal_cells
    else:
        h_nu_first = L_te_nu * (r_normal - 1.0) / (r_normal ** n_normal_cells - 1.0)

    # Adaptive blunt-TE cell count. The constraint h_nu_first * n <= h_te is
    # needed for solve_progression to find a growing geometric progression
    # across the blunt back. We compute the largest n that satisfies it with
    # 10% headroom (so the bisection has room to find r > 1).
    #   - "auto" mode: n is set to this geometry-derived maximum.
    #   - explicit int: the configured value is used as a ceiling; reduced
    #     only if it doesn't fit. NACA0012/Re=6e6 (validated) keeps its
    #     configured value because the geometry comfortably accommodates it.
    max_blunt_cells = int(h_te / (1.10 * h_nu_first))
    if max_blunt_cells < 2:
        raise RuntimeError(
            f"{case_dir.name}: blunt TE too thin for adaptive grading "
            f"(h_te={h_te:.3e}, h_nu_first={h_nu_first:.3e}). "
            f"Consider lowering te_chord_fraction or raising normal_pts."
        )
    if auto_blunt:
        n_blunt_cells = max_blunt_cells
        blunt_pts = n_blunt_cells + 1
        log.info(
            "%s: te_blunt_pts auto -> %d (h_te=%.3e, h_nu_first=%.3e)",
            case_dir.name, blunt_pts, h_te, h_nu_first,
        )
    elif max_blunt_cells < n_blunt_cells:
        log.info(
            "%s: te_blunt_pts downgraded %d -> %d "
            "(h_te=%.3e, h_nu_first=%.3e)",
            case_dir.name, n_blunt_cells + 1, max_blunt_cells + 1,
            h_te, h_nu_first,
        )
        n_blunt_cells = max_blunt_cells
        blunt_pts = n_blunt_cells + 1

    try:
        r_blunt = solve_progression(
            h1=h_nu_first, total_length=h_te, n_cells=n_blunt_cells,
        )
    except ValueError as exc:
        raise RuntimeError(
            f"{case_dir.name}: cannot solve blunt-TE progression "
            f"(h_nu_first={h_nu_first:.3e}, h_te={h_te:.3e}, n={n_blunt_cells}). "
            f"Adjust te_chord_fraction or te_blunt_pts. Underlying error: {exc}"
        ) from exc

    # ---- outer-band outlet progression (wake_centreline_h) ---------------
    # The wake is split at y = +/- h_te into an INNER band (near the centreline,
    # carrying the blunt-TE cells) and an OUTER band (h_te -> farfield) carrying
    # normal_pts cells, so the transition->main interface ties to normal_pts and
    # the old compound-side count inflation (normal_pts + n_blunt_cells) is gone.
    # That inflation was the skew trigger: the blunt cells, packed into [0, h_te]
    # on the west edge, fanned out across the full 20c height on the east edge
    # (coarse wake-centreline grading) over the short transition block -> extreme
    # aspect-ratio / near-90deg non-orthogonal cells. (See the block section.)
    #
    # wake_centreline_h survives as the OUTER-MAIN outlet grading: the outer-main
    # block grades its transverse first cell from h1 (transition interface, west)
    # to wake_centreline_h (outlet, east), coarsening the far wake to control its
    # aspect ratio so the y+<1 wall h1 (~3e-7 for B/C) does not run to the outlet.
    # The transition block stays r_normal on BOTH transverse edges (skew-free),
    # so the residual (gentle, far-field) fan-out is confined to the long main
    # wake. None -> r_normal (legacy A/D behaviour, no coarsening).
    L_outer = Ht - h_te
    h_seam_first = cfg.get("wake_centreline_h") or h1
    if h_seam_first < h1:
        log.warning(
            "%s: wake_centreline_h (%.3e) is finer than the wall h1 (%.3e); "
            "this defeats the AR/non-orthogonality decoupling.",
            case_dir.name, h_seam_first, h1,
        )
    r_seam_outer = solve_progression(
        h1=h_seam_first, total_length=L_outer, n_cells=n_normal_cells
    )

    # ---- gmsh setup ------------------------------------------------------
    _gmsh_setup(gmsh, case_dir.name)
    geo = gmsh.model.geo

    # ---- corner points ---------------------------------------------------
    p_LE_AF   = geo.addPoint(0.0,   0.0,    0.0)
    p_TE_UP   = geo.addPoint(te_x, +h_te,   0.0)
    p_TE_MID  = geo.addPoint(te_x,  0.0,    0.0)
    p_TE_LO   = geo.addPoint(te_x, -h_te,   0.0)
    p_LE_FAR  = geo.addPoint(*ff.LE_FAR,  0.0)
    p_TOP_MID = geo.addPoint(*ff.TOP_MID, 0.0)
    p_BOT_MID = geo.addPoint(*ff.BOT_MID, 0.0)
    p_TOP_TE  = geo.addPoint(*ff.TOP_TE,  0.0)
    p_BOT_TE  = geo.addPoint(*ff.BOT_TE,  0.0)
    p_T_TOP   = geo.addPoint(*ff.T_TOP,   0.0)
    p_T_BOT   = geo.addPoint(*ff.T_BOT,   0.0)
    p_T_MID   = geo.addPoint(*ff.T_MID,   0.0)
    p_TOP_OUT = geo.addPoint(*ff.TOP_OUT, 0.0)
    p_BOT_OUT = geo.addPoint(*ff.BOT_OUT, 0.0)
    p_OUT_MID = geo.addPoint(*ff.OUT_MID, 0.0)

    # ---- inner/outer split points (wake bands at y = +/- h_te) -----------
    # The transition->main interface (x_T) and the outlet (x_O) are each split
    # at +/- h_te so the blunt-TE cells stay in a thin inner band near the
    # centreline instead of fanning out across the full transverse height.
    x_T = ff.T_MID[0]      # transition->main interface x
    x_O = ff.OUT_MID[0]    # outlet x
    p_TI_UP = geo.addPoint(x_T, +h_te, 0.0)
    p_TI_LO = geo.addPoint(x_T, -h_te, 0.0)
    p_OI_UP = geo.addPoint(x_O, +h_te, 0.0)
    p_OI_LO = geo.addPoint(x_O, -h_te, 0.0)

    # ---- airfoil splines (LE -> truncated blunt TE) ----------------------
    # upper[-1] = TE_UP, lower[-1] = TE_LO; the spline endpoints are wired to
    # the corner points above so the airfoil mesh shares vertices with the
    # blunt-back wall edges.
    up_inner = [geo.addPoint(float(x), float(y), 0.0) for x, y in upper[1:-1]]
    lo_inner = [geo.addPoint(float(x), float(y), 0.0) for x, y in lower[1:-1]]
    C_af_up = geo.addSpline([p_LE_AF, *up_inner, p_TE_UP])
    C_af_lo = geo.addSpline([p_LE_AF, *lo_inner, p_TE_LO])

    # ---- blunt-back wall edges (part of the AIRFOIL boundary) -----------
    C_te_blunt_up = geo.addLine(p_TE_UP,  p_TE_MID)
    C_te_blunt_lo = geo.addLine(p_TE_MID, p_TE_LO)

    # ---- farfield curves (upstream cap, outlet, top/bottom split) -------
    C_arc_up   = geo.addCircleArc(p_LE_FAR, p_LE_AF, p_TOP_MID)
    C_arc_lo   = geo.addCircleArc(p_LE_FAR, p_LE_AF, p_BOT_MID)
    C_top_h    = geo.addLine(p_TOP_MID, p_TOP_TE)
    C_bot_h    = geo.addLine(p_BOT_MID, p_BOT_TE)
    C_top_trans = geo.addLine(p_TOP_TE, p_T_TOP)
    C_top_main  = geo.addLine(p_T_TOP,  p_TOP_OUT)
    C_bot_trans = geo.addLine(p_BOT_TE, p_T_BOT)
    C_bot_main  = geo.addLine(p_T_BOT,  p_BOT_OUT)
    # Outlet split at y = +/- h_te into inner (near centreline) + outer bands.
    C_out_up_in  = geo.addLine(p_OUT_MID, p_OI_UP)
    C_out_up_out = geo.addLine(p_OI_UP,   p_TOP_OUT)
    C_out_lo_in  = geo.addLine(p_OUT_MID, p_OI_LO)
    C_out_lo_out = geo.addLine(p_OI_LO,   p_BOT_OUT)

    # ---- internal seam curves (wall-normal + wake centreline) ------------
    C_le_rad    = geo.addLine(p_LE_FAR, p_LE_AF)
    C_te_nu     = geo.addLine(p_TE_UP,  p_TOP_TE)
    C_te_nl     = geo.addLine(p_TE_LO,  p_BOT_TE)
    C_wake_trans = geo.addLine(p_TE_MID, p_T_MID)
    C_wake_main  = geo.addLine(p_T_MID,  p_OUT_MID)
    # Horizontal seams along y = +/- h_te separating the inner/outer wake bands.
    C_hseam_t_up = geo.addLine(p_TE_UP, p_TI_UP)   # transition, upper
    C_hseam_t_lo = geo.addLine(p_TE_LO, p_TI_LO)   # transition, lower
    C_hseam_m_up = geo.addLine(p_TI_UP, p_OI_UP)   # main, upper
    C_hseam_m_lo = geo.addLine(p_TI_LO, p_OI_LO)   # main, lower
    # Vertical seams at the transition->main interface, split into inner/outer.
    C_vseam_t_up_in  = geo.addLine(p_T_MID, p_TI_UP)
    C_vseam_t_up_out = geo.addLine(p_TI_UP, p_T_TOP)
    C_vseam_t_lo_in  = geo.addLine(p_T_MID, p_TI_LO)
    C_vseam_t_lo_out = geo.addLine(p_TI_LO, p_T_BOT)

    # ---- transfinite line counts ----------------------------------------
    chord_pts = cfg["chord_pts_upper"]
    if cfg["chord_pts_lower"] != chord_pts:
        raise ValueError(
            f"chord_pts_upper ({chord_pts}) must equal chord_pts_lower "
            f"({cfg['chord_pts_lower']}) for symmetric NACA mesh."
        )
    n_arc, n_h = _split_north_counts(chord_pts, cfg["north_arc_to_horiz_ratio"])
    normal_pts = cfg["normal_pts"]
    wake_pts   = cfg["wake_pts"]
    trans_pts  = cfg["transition_wake_pts"]
    bump = float(cfg["le_te_cluster"])
    # Inner wake bands (y in [0, h_te]) carry the blunt-TE cell count (blunt_pts);
    # outer bands (y in [h_te, Ht]) carry normal_pts so the transition->main
    # interface ties to normal_pts. Both counts are already set above.

    # Airfoil — Bump law clusters at BOTH endpoints (LE and TE simultaneously).
    geo.mesh.setTransfiniteCurve(C_af_up, chord_pts, "Bump", bump)
    geo.mesh.setTransfiniteCurve(C_af_lo, chord_pts, "Bump", bump)

    # Blunt-back wall — Progression with small cells at the airfoil corner
    # (TE_UP for the upper half, TE_LO for the lower half) so the cell next
    # to TE_UP/TE_LO matches te_nu/te_nl's first cell at that corner.
    #   C_te_blunt_up : TE_UP  -> TE_MID  (small at start)     | +r_blunt
    #   C_te_blunt_lo : TE_MID -> TE_LO   (small at end)       | -r_blunt
    geo.mesh.setTransfiniteCurve(C_te_blunt_up, blunt_pts, "Progression", +r_blunt)
    geo.mesh.setTransfiniteCurve(C_te_blunt_lo, blunt_pts, "Progression", -r_blunt)

    # North-edge segments — uniform along each partition.
    geo.mesh.setTransfiniteCurve(C_arc_up, n_arc, "Progression", 1.0)
    geo.mesh.setTransfiniteCurve(C_arc_lo, n_arc, "Progression", 1.0)
    geo.mesh.setTransfiniteCurve(C_top_h,  n_h,   "Progression", 1.0)
    geo.mesh.setTransfiniteCurve(C_bot_h,  n_h,   "Progression", 1.0)

    # Wall-normal seams — coef sign chosen so the small cell sits at the
    # airfoil (wall-corner) end of each curve.
    #   C_le_rad : LE_FAR -> LE_AF   small at end   | -r_normal
    #   C_te_nu  : TE_UP  -> TOP_TE  small at start | +r_normal
    #   C_te_nl  : TE_LO  -> BOT_TE  small at start | +r_normal
    geo.mesh.setTransfiniteCurve(C_le_rad, normal_pts, "Progression", -r_normal)
    geo.mesh.setTransfiniteCurve(C_te_nu,  normal_pts, "Progression", +r_normal)
    geo.mesh.setTransfiniteCurve(C_te_nl,  normal_pts, "Progression", +r_normal)

    # OUTER transverse band (h_te -> farfield): normal_pts cells, r_normal,
    # small cell at the y=h_te (wall-corner) end — identical grading to
    # te_nu/te_nl, so the outer TRANSITION block has matching distributions on
    # both transverse edges (skew-free). The outer-MAIN outlet uses r_seam_outer
    # (wake_centreline_h) so the far wake coarsens — gentle, far-field fan-out.
    #   C_vseam_t_up_out : TI_UP   -> T_TOP    small at start (TI_UP)  | +r_normal
    #   C_vseam_t_lo_out : TI_LO   -> T_BOT    small at start (TI_LO)  | +r_normal
    #   C_out_up_out     : OI_UP   -> TOP_OUT  small at start (OI_UP)  | +r_seam_outer
    #   C_out_lo_out     : OI_LO   -> BOT_OUT  small at start (OI_LO)  | +r_seam_outer
    geo.mesh.setTransfiniteCurve(C_vseam_t_up_out, normal_pts, "Progression", +r_normal)
    geo.mesh.setTransfiniteCurve(C_vseam_t_lo_out, normal_pts, "Progression", +r_normal)
    geo.mesh.setTransfiniteCurve(C_out_up_out,     normal_pts, "Progression", +r_seam_outer)
    geo.mesh.setTransfiniteCurve(C_out_lo_out,     normal_pts, "Progression", +r_seam_outer)

    # INNER transverse band (centreline -> h_te): blunt_pts cells, r_blunt,
    # small cell at the y=h_te end (continuity with te_nu at TE_UP) — identical
    # grading to the blunt back, so the inner blocks have matching distributions
    # on both transverse edges (skew-free) and the blunt-TE refinement stays
    # confined to the thin near-centreline band.
    #   C_vseam_t_up_in : T_MID   -> TI_UP   small at end (TI_UP)  | -r_blunt
    #   C_vseam_t_lo_in : T_MID   -> TI_LO   small at end (TI_LO)  | -r_blunt
    #   C_out_up_in     : OUT_MID -> OI_UP   small at end (OI_UP)  | -r_blunt
    #   C_out_lo_in     : OUT_MID -> OI_LO   small at end (OI_LO)  | -r_blunt
    geo.mesh.setTransfiniteCurve(C_vseam_t_up_in, blunt_pts, "Progression", -r_blunt)
    geo.mesh.setTransfiniteCurve(C_vseam_t_lo_in, blunt_pts, "Progression", -r_blunt)
    geo.mesh.setTransfiniteCurve(C_out_up_in,     blunt_pts, "Progression", -r_blunt)
    geo.mesh.setTransfiniteCurve(C_out_lo_in,     blunt_pts, "Progression", -r_blunt)

    # Horizontal seams (y = +/- h_te) inherit the streamwise grading of the band
    # they border: transition seams match the transition block (small at the TE
    # end), main seams match the main wake (small at the interface end).
    geo.mesh.setTransfiniteCurve(C_hseam_t_up, trans_pts, "Progression", +r_wake_trans)
    geo.mesh.setTransfiniteCurve(C_hseam_t_lo, trans_pts, "Progression", +r_wake_trans)
    geo.mesh.setTransfiniteCurve(C_hseam_m_up, wake_pts,  "Progression", +r_wake_main)
    geo.mesh.setTransfiniteCurve(C_hseam_m_lo, wake_pts,  "Progression", +r_wake_main)

    # Transition wake — small cells at TE end of each curve.
    geo.mesh.setTransfiniteCurve(C_wake_trans, trans_pts, "Progression", +r_wake_trans)
    geo.mesh.setTransfiniteCurve(C_top_trans,  trans_pts, "Progression", +r_wake_trans)
    geo.mesh.setTransfiniteCurve(C_bot_trans,  trans_pts, "Progression", +r_wake_trans)

    # Main wake — small cells at T_*  (transition->main interface) end.
    geo.mesh.setTransfiniteCurve(C_wake_main, wake_pts, "Progression", +r_wake_main)
    geo.mesh.setTransfiniteCurve(C_top_main,  wake_pts, "Progression", +r_wake_main)
    geo.mesh.setTransfiniteCurve(C_bot_main,  wake_pts, "Progression", +r_wake_main)

    # ---- block loops & surfaces -----------------------------------------
    # U/L keep their 5-edge loops (north split into arc + horizontal run). The
    # wake is now 8 clean 4-corner quads: each half (transition, main) is split
    # at y = +/- h_te into an inner block (near centreline, blunt-TE band) and an
    # outer block (normal_pts band). TE_UP/TE_LO are now true corners (of the
    # inner transition blocks), so no compound 5-edge sides remain.
    loop_U   = geo.addCurveLoop([+C_af_up,  +C_te_nu, -C_top_h, -C_arc_up, +C_le_rad])
    loop_L   = geo.addCurveLoop([+C_arc_lo, +C_bot_h, -C_te_nl, -C_af_lo,  -C_le_rad])

    loop_UTW_in  = geo.addCurveLoop([+C_wake_trans, +C_vseam_t_up_in,  -C_hseam_t_up, +C_te_blunt_up])
    loop_UTW_out = geo.addCurveLoop([+C_hseam_t_up,  +C_vseam_t_up_out, -C_top_trans,  -C_te_nu])
    loop_UMW_in  = geo.addCurveLoop([+C_wake_main,  +C_out_up_in,      -C_hseam_m_up, -C_vseam_t_up_in])
    loop_UMW_out = geo.addCurveLoop([+C_hseam_m_up, +C_out_up_out,     -C_top_main,   -C_vseam_t_up_out])

    loop_LTW_in  = geo.addCurveLoop([+C_hseam_t_lo, -C_vseam_t_lo_in,  -C_wake_trans, +C_te_blunt_lo])
    loop_LTW_out = geo.addCurveLoop([+C_bot_trans,  -C_vseam_t_lo_out, -C_hseam_t_lo, +C_te_nl])
    loop_LMW_in  = geo.addCurveLoop([+C_hseam_m_lo, -C_out_lo_in,      -C_wake_main,  +C_vseam_t_lo_in])
    loop_LMW_out = geo.addCurveLoop([+C_bot_main,   -C_out_lo_out,     -C_hseam_m_lo, +C_vseam_t_lo_out])

    S_U       = geo.addPlaneSurface([loop_U])
    S_L       = geo.addPlaneSurface([loop_L])
    S_UTW_in  = geo.addPlaneSurface([loop_UTW_in])
    S_UTW_out = geo.addPlaneSurface([loop_UTW_out])
    S_UMW_in  = geo.addPlaneSurface([loop_UMW_in])
    S_UMW_out = geo.addPlaneSurface([loop_UMW_out])
    S_LTW_in  = geo.addPlaneSurface([loop_LTW_in])
    S_LTW_out = geo.addPlaneSurface([loop_LTW_out])
    S_LMW_in  = geo.addPlaneSurface([loop_LMW_in])
    S_LMW_out = geo.addPlaneSurface([loop_LMW_out])

    source_surfaces = [S_U, S_L,
                       S_UTW_in, S_UTW_out, S_UMW_in, S_UMW_out,
                       S_LTW_in, S_LTW_out, S_LMW_in, S_LMW_out]
    loop_sizes      = [5, 5, 4, 4, 4, 4, 4, 4, 4, 4]

    # Transfinite-surface corners (CCW; matches each loop's corner order).
    geo.mesh.setTransfiniteSurface(S_U, "Left", [p_LE_AF,  p_TE_UP,  p_TOP_TE, p_LE_FAR])
    geo.mesh.setTransfiniteSurface(S_L, "Left", [p_LE_FAR, p_BOT_TE, p_TE_LO,  p_LE_AF])

    geo.mesh.setTransfiniteSurface(S_UTW_in,  "Left", [p_TE_MID, p_T_MID,   p_TI_UP,   p_TE_UP])
    geo.mesh.setTransfiniteSurface(S_UTW_out, "Left", [p_TE_UP,  p_TI_UP,   p_T_TOP,   p_TOP_TE])
    geo.mesh.setTransfiniteSurface(S_UMW_in,  "Left", [p_T_MID,  p_OUT_MID, p_OI_UP,   p_TI_UP])
    geo.mesh.setTransfiniteSurface(S_UMW_out, "Left", [p_TI_UP,  p_OI_UP,   p_TOP_OUT, p_T_TOP])

    geo.mesh.setTransfiniteSurface(S_LTW_in,  "Left", [p_TE_LO,  p_TI_LO,   p_T_MID,   p_TE_MID])
    geo.mesh.setTransfiniteSurface(S_LTW_out, "Left", [p_BOT_TE, p_T_BOT,   p_TI_LO,   p_TE_LO])
    geo.mesh.setTransfiniteSurface(S_LMW_in,  "Left", [p_TI_LO,  p_OI_LO,   p_OUT_MID, p_T_MID])
    geo.mesh.setTransfiniteSurface(S_LMW_out, "Left", [p_T_BOT,  p_BOT_OUT, p_OI_LO,   p_TI_LO])

    for s in source_surfaces:
        geo.mesh.setRecombine(2, s)

    # ---- spanwise extrusion + finalise ----------------------------------
    # The blunt-back edges (te_blunt_up/lo) are part of the AIRFOIL boundary,
    # so their extruded laterals must end up on the no-slip wall patch.
    airfoil_curves  = {C_af_up, C_af_lo, C_te_blunt_up, C_te_blunt_lo}
    farfield_curves = {
        C_arc_up, C_arc_lo, C_top_h, C_bot_h,
        C_top_trans, C_top_main, C_bot_trans, C_bot_main,
        C_out_up_in, C_out_up_out, C_out_lo_in, C_out_lo_out,
    }
    block_names = ["U", "L",
                   "UTW_in", "UTW_out", "UMW_in", "UMW_out",
                   "LTW_in", "LTW_out", "LMW_in", "LMW_out"]
    cell_count, runtime = _finalize_and_write(
        gmsh, geo, source_surfaces, loop_sizes, block_names,
        airfoil_curves, farfield_curves, case_dir, cfg, chord, start_time,
    )

    # ---- diagnostics on cell-size handoff -------------------------------
    # Realised first-cell at TE in the transition block (from the analytic
    # progression). Compare against the airfoil's Bump endpoint estimate.
    if abs(r_wake_trans - 1.0) < 1e-9:
        h_trans_first = Lt / n_trans_cells
    else:
        h_trans_first = Lt * (r_wake_trans - 1.0) / (r_wake_trans ** n_trans_cells - 1.0)
    # First main-wake cell.
    if abs(r_wake_main - 1.0) < 1e-9:
        h_main_first = Lm / (wake_pts - 1)
    else:
        h_main_first = Lm * (r_wake_main - 1.0) / (r_wake_main ** (wake_pts - 1) - 1.0)
    # Cell at east end of transition (== last cell of transition block).
    h_trans_last = h_trans_first * (r_wake_trans ** (n_trans_cells - 1))

    log.info(
        "%s wake handoff: h_TE_target=%.3e | trans_first=%.3e (ratio=%.2f) | "
        "trans_last=%.3e | main_first=%.3e (ratio=%.2f) | blunt h_te=%.3e "
        "(at x=%.3f) r_blunt=%.3f",
        case_dir.name,
        h_te_target,
        h_trans_first, h_trans_first / max(h_te_target, 1e-30),
        h_trans_last,
        h_main_first, h_main_first / max(h_trans_last, 1e-30),
        h_te, te_x, r_blunt,
    )

    return {
        "first_cell_height":          float(h1),
        "normal_progression":         float(r_normal),
        "seam_progression":           float(r_seam_outer),
        "wake_progression":           float(r_wake_main),
        "transition_wake_progression": float(r_wake_trans),
        "h_te_airfoil_target":        float(h_te_target),
        "h_trans_first":              float(h_trans_first),
        "h_trans_last":               float(h_trans_last),
        "h_main_first":               float(h_main_first),
        "te_chord_fraction":          float(te_frac),
        "te_half_thickness":          float(h_te),
        "te_blunt_progression":       float(r_blunt),
        "n_arc_pts":                  int(n_arc),
        "n_horiz_pts":                int(n_h),
        "n_seam_pts":                 int(normal_pts),
        "n_blunt_inner_pts":          int(blunt_pts),
        "cell_count_gmsh":            cell_count,
        "mesh_runtime_s":             float(runtime),
    }
