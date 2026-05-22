"""
Structured C+H 6-block transfinite mesh for a NACA airfoil (Regime A) with
a blunt trailing edge.

Topology
========

Six transfinite quad blocks. The airfoil is truncated at x = te_chord_fraction
so the trailing edge has a finite half-thickness h_te (NACA-4 thickness
evaluated at the truncation x). The blunt back is a wall, split into
upper (TE_UP -> TE_MID) and lower (TE_MID -> TE_LO) halves. The wake
centreline starts at TE_MID. A short "transition wake" block sits between
the airfoil-side blocks and the main wake blocks; it inherits the airfoil's
TE chordwise spacing on its south face and grades outward to the main wake's
spacing on its east face. Its west face is COMPOUND: the (tall) wall-normal
seam te_nu/te_nl above/below the blunt corner, plus the (short) blunt-back
half. The compound side has two segments but only 4 transfinite corners, so
gmsh still meshes it as a structured quad with consistent (i,j) parametrics.

                F_TOP_MID -- F_TOP_TE  -- T_TOP -- F_TOP_OUT
                  /            |          |          |
                 /             |          |          |
                / Block U    UTW         UMW         |
   F_LE_FAR --                |          |          |
                / airfoil --  TE_UP +                |
                              |    \                 |
                              | (BB up)              |
                              TE_MID -- T_MID --- OUT_MID
                              | (BB lo)              |
                              |    /                 |
                / airfoil --  TE_LO +                |
                \\             |          |          |
                 \\ Block L   LTW         LMW         |
                  \\          |          |          |
                F_BOT_MID -- F_BOT_TE  -- T_BOT -- F_BOT_OUT

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
    C_outU     : TOP_OUT -> OUT_MID
    C_outL     : OUT_MID -> BOT_OUT
    C_top_trans  : TOP_TE -> T_TOP
    C_top_main   : T_TOP  -> TOP_OUT
    C_bot_trans  : BOT_TE -> T_BOT
    C_bot_main   : T_BOT  -> BOT_OUT

  Internal seams (interior to the fluid domain):
    C_le_rad     : LE_FAR -> LE_AF
    C_te_nu      : TE_UP  -> TOP_TE   (vertical seam above blunt corner)
    C_te_nl      : TE_LO  -> BOT_TE   (vertical seam below blunt corner)
    C_wake_trans : TE_MID -> T_MID    (wake centreline, transition block)
    C_wake_main  : T_MID  -> OUT_MID  (wake centreline, main block)
    C_t_seam_up  : T_MID  -> T_TOP    (seam at transition->main interface)
    C_t_seam_lo  : T_MID  -> T_BOT

Block loops (signed curve tags, CCW with interior on the left)
--------------------------------------------------------------
  Block U   : [+af_up,       +te_nu,     -top_h,      -arc_up,      +le_rad]
  Block L   : [+arc_lo,      +bot_h,     -te_nl,      -af_lo,       -le_rad]
  Block UTW : [+wake_trans,  +t_seam_up, -top_trans,  -te_nu,       +te_blunt_up]
  Block UMW : [+wake_main,   -outU,      -top_main,   -t_seam_up]
  Block LTW : [+bot_trans,   -t_seam_lo, -wake_trans, +te_blunt_lo, +te_nl]
  Block LMW : [+bot_main,    -outL,      -wake_main,  +t_seam_lo]

The transfinite corners of UTW are (TE_MID, T_MID, T_TOP, TOP_TE); TE_UP is
an intermediate point on UTW's west compound side. Likewise the corners of
LTW are (BOT_TE, T_BOT, T_MID, TE_MID) with TE_LO intermediate.

Transfinite consistency
-----------------------
  N_af_up      = N_af_lo                                    = chord_pts
  N_arc + N_h - 1                                           = chord_pts
  N_le_rad                                                  = normal_pts
  N_te_nu      = N_te_nl                                    = normal_pts
  N_te_blunt_up = N_te_blunt_lo                             = te_blunt_pts
  N_t_seam_up  = N_t_seam_lo  = N_outU = N_outL
                                = normal_pts + te_blunt_pts - 1
  N_wake_trans = N_top_trans  = N_bot_trans                 = transition_wake_pts
  N_wake_main  = N_top_main   = N_bot_main                  = wake_pts
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
# Main builder
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

    start_time = time.perf_counter()

    # ---- physics-derived spacings ----------------------------------------
    h1 = first_cell_height(params["Re"], cfg["y_plus_target"], chord=chord)

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

    # ---- seam (t_seam_up/lo) and outlet (c_outU/L) progression -----------
    # These curves now span Ht with (normal_pts - 1 + n_blunt_cells) cells so
    # they match the cell count of UTW/LTW's compound west side. Re-solve the
    # progression to keep the first cell at the wake-centreline end matched
    # to the airfoil-side h1.
    n_seam_cells = n_normal_cells + n_blunt_cells
    r_seam = solve_progression(h1=h1, total_length=Ht, n_cells=n_seam_cells)

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
    C_outU      = geo.addLine(p_TOP_OUT, p_OUT_MID)
    C_outL      = geo.addLine(p_OUT_MID, p_BOT_OUT)

    # ---- internal seam curves (wall-normal + wake centreline) ------------
    C_le_rad    = geo.addLine(p_LE_FAR, p_LE_AF)
    C_te_nu     = geo.addLine(p_TE_UP,  p_TOP_TE)
    C_te_nl     = geo.addLine(p_TE_LO,  p_BOT_TE)
    C_wake_trans = geo.addLine(p_TE_MID, p_T_MID)
    C_wake_main  = geo.addLine(p_T_MID,  p_OUT_MID)
    C_t_seam_up  = geo.addLine(p_T_MID,  p_T_TOP)
    C_t_seam_lo  = geo.addLine(p_T_MID,  p_T_BOT)

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
    # Combined node count along the UTW/LTW compound west sides and the
    # corresponding east-side (t_seam) and outlet (c_out) edges.
    seam_pts = normal_pts + n_blunt_cells

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
    # airfoil (or wake centreline) end of each curve.
    #   curve            : start -> end                 small at  | coef sign
    #   C_le_rad         : LE_FAR -> LE_AF              end       | -r_normal
    #   C_te_nu          : TE_UP  -> TOP_TE             start     | +r_normal
    #   C_te_nl          : TE_LO  -> BOT_TE             start     | +r_normal
    # The t_seam/c_out edges now use seam_pts and r_seam so their cell count
    # matches the compound (te_nu + te_blunt) west side of UTW/LTW.
    #   C_t_seam_up      : T_MID  -> T_TOP              start     | +r_seam
    #   C_t_seam_lo      : T_MID  -> T_BOT              start     | +r_seam
    #   C_outU           : TOP_OUT -> OUT_MID           end       | -r_seam
    #   C_outL           : OUT_MID -> BOT_OUT           start     | +r_seam
    geo.mesh.setTransfiniteCurve(C_le_rad,    normal_pts, "Progression", -r_normal)
    geo.mesh.setTransfiniteCurve(C_te_nu,     normal_pts, "Progression", +r_normal)
    geo.mesh.setTransfiniteCurve(C_te_nl,     normal_pts, "Progression", +r_normal)
    geo.mesh.setTransfiniteCurve(C_t_seam_up, seam_pts,   "Progression", +r_seam)
    geo.mesh.setTransfiniteCurve(C_t_seam_lo, seam_pts,   "Progression", +r_seam)
    geo.mesh.setTransfiniteCurve(C_outU,      seam_pts,   "Progression", -r_seam)
    geo.mesh.setTransfiniteCurve(C_outL,      seam_pts,   "Progression", +r_seam)

    # Transition wake — small cells at TE end of each curve.
    geo.mesh.setTransfiniteCurve(C_wake_trans, trans_pts, "Progression", +r_wake_trans)
    geo.mesh.setTransfiniteCurve(C_top_trans,  trans_pts, "Progression", +r_wake_trans)
    geo.mesh.setTransfiniteCurve(C_bot_trans,  trans_pts, "Progression", +r_wake_trans)

    # Main wake — small cells at T_*  (transition->main interface) end.
    geo.mesh.setTransfiniteCurve(C_wake_main, wake_pts, "Progression", +r_wake_main)
    geo.mesh.setTransfiniteCurve(C_top_main,  wake_pts, "Progression", +r_wake_main)
    geo.mesh.setTransfiniteCurve(C_bot_main,  wake_pts, "Progression", +r_wake_main)

    # ---- block loops & surfaces -----------------------------------------
    # UTW and LTW have 5-edge loops but only 4 transfinite corners; TE_UP
    # (resp. TE_LO) sits as an intermediate point on the compound west side.
    loop_U   = geo.addCurveLoop([+C_af_up,      +C_te_nu,     -C_top_h,    -C_arc_up,    +C_le_rad])
    loop_L   = geo.addCurveLoop([+C_arc_lo,     +C_bot_h,     -C_te_nl,    -C_af_lo,     -C_le_rad])
    loop_UTW = geo.addCurveLoop([+C_wake_trans, +C_t_seam_up, -C_top_trans, -C_te_nu,    +C_te_blunt_up])
    loop_UMW = geo.addCurveLoop([+C_wake_main,  -C_outU,      -C_top_main,  -C_t_seam_up])
    loop_LTW = geo.addCurveLoop([+C_bot_trans,  -C_t_seam_lo, -C_wake_trans, +C_te_blunt_lo, +C_te_nl])
    loop_LMW = geo.addCurveLoop([+C_bot_main,   -C_outL,      -C_wake_main,  +C_t_seam_lo])

    S_U   = geo.addPlaneSurface([loop_U])
    S_L   = geo.addPlaneSurface([loop_L])
    S_UTW = geo.addPlaneSurface([loop_UTW])
    S_UMW = geo.addPlaneSurface([loop_UMW])
    S_LTW = geo.addPlaneSurface([loop_LTW])
    S_LMW = geo.addPlaneSurface([loop_LMW])

    source_surfaces = [S_U, S_L, S_UTW, S_UMW, S_LTW, S_LMW]
    loop_sizes      = [5,    5,    5,      4,      5,      4]

    # Transfinite-surface corners (CCW order picks the parametric (i,j) axes).
    geo.mesh.setTransfiniteSurface(S_U,   "Left",
                                    [p_LE_AF,  p_TE_UP,  p_TOP_TE,  p_LE_FAR])
    geo.mesh.setTransfiniteSurface(S_L,   "Left",
                                    [p_LE_FAR, p_BOT_TE, p_TE_LO,   p_LE_AF])
    geo.mesh.setTransfiniteSurface(S_UTW, "Left",
                                    [p_TE_MID, p_T_MID,  p_T_TOP,   p_TOP_TE])
    geo.mesh.setTransfiniteSurface(S_UMW, "Left",
                                    [p_T_MID,  p_OUT_MID, p_TOP_OUT, p_T_TOP])
    geo.mesh.setTransfiniteSurface(S_LTW, "Left",
                                    [p_BOT_TE, p_T_BOT,  p_T_MID,   p_TE_MID])
    geo.mesh.setTransfiniteSurface(S_LMW, "Left",
                                    [p_T_BOT,  p_BOT_OUT, p_OUT_MID, p_T_MID])

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
    block_names = ["U", "L", "UTW", "UMW", "LTW", "LMW"]
    per_block: dict[str, dict] = {}
    for name, off, n in zip(block_names, offsets, loop_sizes):
        block_entities = extr[off : off + 2 + n]
        per_block[name] = {
            "top":      block_entities[0][1],     # (2, tag) — back face at +z
            "volume":   block_entities[1][1],     # (3, tag)
            "laterals": [t for d, t in block_entities[2 : 2 + n] if d == 2],
        }

    # ---- classify laterals into aerofoil / freestream / internal seam ----
    # The blunt-back edges (te_blunt_up/lo) are part of the AIRFOIL boundary,
    # so their extruded laterals must end up on the no-slip wall patch.
    airfoil_curves  = {C_af_up, C_af_lo, C_te_blunt_up, C_te_blunt_lo}
    farfield_curves = {
        C_arc_up, C_arc_lo, C_top_h, C_bot_h,
        C_top_trans, C_top_main, C_bot_trans, C_bot_main,
        C_outU, C_outL,
    }

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

    # ---- mesh -----------------------------------------------------------
    gmsh.model.mesh.generate(3)
    _, elem_tags, _ = gmsh.model.mesh.getElements(3)
    cell_count = int(sum(len(t) for t in elem_tags))

    mesh_path = case_dir / "mesh.msh"
    gmsh.write(str(mesh_path))

    runtime = time.perf_counter() - start_time

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
        "seam_progression":           float(r_seam),
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
        "n_seam_pts":                 int(seam_pts),
        "cell_count_gmsh":            cell_count,
        "mesh_runtime_s":             float(runtime),
    }
