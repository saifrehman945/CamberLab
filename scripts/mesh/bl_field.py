"""
Boundary-layer-field mesh for a NACA airfoil (Regime B and any regime whose
``cfg["topology"] == "bl_field"``).

Why this exists
===============
The structured 6-block transfinite C+H grid in :mod:`mesh.topology` meshes the
entire wall-to-far-field span of each block with a single transfinite
interpolation (TFI). For Regime A (y+~30, first cell ~1e-4 m, 20c domain) that
is fine. For Regime B (y+=0.5, first cell ~2e-6 m, 50c domain) it is not: TFI
linearly blends the ~2 micron wall cell against the ~metre far-field cell over a
curved wall, so the near-wall node offset follows the wall->far-field direction
instead of the wall normal and *folds* the near-wall cells. The result is ~800
degenerate / negative-volume cells smeared along the whole airfoil surface and
the wake centreline; gmshToFoam then cannot match the collapsed wall faces and
aborts ("Are patches in incremental order?").

The fix is to stop using TFI for the boundary layer. gmsh's ``BoundaryLayer``
mesh *field* grows structured quad layers along the *true* wall normal (it never
folds), wraps the rounded leading edge, and fans the convex blunt-trailing-edge
corners. The region outside the boundary layer is filled with
Frontal-Delaunay-for-Quads + Blossom recombination, graded by a size field.

Why we extrude the 2-D mesh by hand
===================================
gmsh's ``BoundaryLayer`` field is a *2-D* mesher and is INCOMPATIBLE with
``geo.extrude``: a surface that is the base of a geometric extrusion is meshed
under extrusion constraints that conflict with the field, collapsing ~2000 BL
quads to zero-area (verified on gmsh 4.15). So we mesh the airfoil plane in 2-D
only (``generate(2)`` — clean), then perform the one-cell spanwise extrusion
ourselves: every 2-D quad becomes a hex, every fill triangle a prism, and every
wall / far-field boundary edge a lateral quad face. The resulting 3-D ``.msh``
converts through gmshToFoam with zero inverted or unmatched faces.

Interface
=========
``build_bl_mesh(case_dir, params, cfg, chord=1.0) -> dict`` matches
:func:`mesh.topology.build_c_grid`: it writes ``<case_dir>/mesh.msh`` and returns
a metrics dict merged into ``case_metadata.json`` by ``write_metadata``.

Patch (physical-group) names are identical to the transfinite path so the rest
of the pipeline (``rewrite_boundary_types``, the OpenFOAM templates) is
unchanged: ``frontAndBack`` (empty), ``aerofoil`` (wall), ``freestream``
(patch), ``fluid`` (cellZone).

Trade-off: the frontal-quad fill leaves a small fraction of triangles (-> prisms
after extrusion), so the mesh is *not* 100 % hex. ``min_hex_fraction`` is relaxed
for this topology (see ``regime_parameters``).
"""

from __future__ import annotations

import logging
import math
import os
import time
from pathlib import Path

import numpy as np

try:
    import resource                       # POSIX-only; absent on Windows
except ImportError:                       # pragma: no cover
    resource = None

from .boundary_layer import bl_thickness, cells_needed, first_cell_height
from .geometry import farfield_points, naca_symmetric

log = logging.getLogger(__name__)

# gmsh MSH-2.2 element type ids
_TRI, _QUAD, _HEX, _PRISM = 2, 3, 5, 6


# ---------------------------------------------------------------------------
# RAM guards
# ---------------------------------------------------------------------------
# A fine bl_field mesh (low first cell + long refined wake + large domain) can
# generate hundreds of thousands of cells; gmsh's Frontal-Delaunay-for-Quads +
# Blossom recombination then needs gigabytes, enough to swap-thrash or OOM-kill
# the whole machine. Two layers of protection: (1) cap gmsh's thread count (peak
# RAM scales with parallel mesher threads) and optionally the process address
# space, so a runaway fails instead of taking the box down; (2) estimate the
# cell count from the size field and refuse to start a mesh over budget.

def _apply_gmsh_resource_limits(gmsh_module) -> tuple[int, float | None]:
    """Cap gmsh threads and, if ``NACA_MESH_MEM_GB`` is set, the process address
    space (RLIMIT_AS). Returns (threads, mem_cap_bytes_or_None)."""
    env_threads = os.environ.get("NACA_MESH_THREADS")
    threads = (max(1, int(env_threads)) if env_threads
               else max(1, min(4, os.cpu_count() or 4)))
    gmsh_module.option.setNumber("General.NumThreads", threads)

    mem_cap = None
    mem_gb = os.environ.get("NACA_MESH_MEM_GB")
    if mem_gb and resource is not None:
        mem_cap = float(mem_gb) * (1024 ** 3)
        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        new_hard = (int(mem_cap) if hard == resource.RLIM_INFINITY
                    else min(int(mem_cap), hard))
        resource.setrlimit(resource.RLIMIT_AS, (int(mem_cap), new_hard))
    return threads, mem_cap


def _estimate_2d_cells(cfg: dict, surf_cell: float, far_cell: float,
                       wake_len: float, wake_hw: float, wake_cell: float,
                       n_bl: int, perimeter: float, chord: float) -> int:
    """Rough upper-bound estimate of the 2-D fill cell count from the size field.

    Far field at ``far_cell`` over the (semicircle + downstream rectangle)
    domain, the wake corridor at ``wake_cell``, and ``n_bl`` structured layers
    around the airfoil perimeter. A 1.3x margin covers the graded ramp regions.
    Conservative by design — meant to catch RAM-exhausting meshes, not to be
    exact.
    """
    R = float(cfg["upstream_radius"]) * chord
    Ld = float(cfg["downstream_length"]) * chord
    H = float(cfg["transverse_extent"]) * chord
    domain_area = 0.5 * math.pi * R * R + (Ld + chord) * (2.0 * H)
    far_cells = domain_area / (far_cell ** 2)
    wake_cells = (wake_len * 2.0 * wake_hw) / (wake_cell ** 2)
    bl_cells = (perimeter / surf_cell) * n_bl
    return int(1.3 * (far_cells + wake_cells + bl_cells))


# ---------------------------------------------------------------------------
# 2-D mesh extraction
# ---------------------------------------------------------------------------

def _extract_2d(gmsh_module, surface_tag: int,
                airfoil_curves: list[int], outer_curves: list[int]):
    """Pull the generated 2-D mesh out of gmsh into plain numpy arrays.

    Returns (xy, quads, tris, af_edges, out_edges) with node ids renumbered to a
    contiguous 1..N. ``quads``/``tris`` index the fluid surface cells;
    ``*_edges`` are wall / far-field boundary edges (node pairs).
    """
    tags, coords, _ = gmsh_module.model.mesh.getNodes()
    coords = np.asarray(coords, dtype=np.float64).reshape(-1, 3)
    remap = {int(t): i + 1 for i, t in enumerate(tags)}     # gmsh tag -> 1..N
    xy = coords[:, :2]

    def edges_of(curves: list[int]) -> list[tuple[int, int]]:
        out: list[tuple[int, int]] = []
        for c in curves:
            etypes, _, enodes = gmsh_module.model.mesh.getElements(1, c)
            for etype, conn in zip(etypes, enodes):
                if etype == 1:                              # 2-node line
                    out += [(remap[conn[i]], remap[conn[i + 1]])
                            for i in range(0, len(conn), 2)]
        return out

    quads = np.zeros((0, 4), dtype=np.int64)
    tris = np.zeros((0, 3), dtype=np.int64)
    etypes, _, enodes = gmsh_module.model.mesh.getElements(2, surface_tag)
    for etype, conn in zip(etypes, enodes):
        conn = np.asarray(conn, dtype=np.int64)
        if etype == _QUAD:
            quads = np.vectorize(remap.get)(conn.reshape(-1, 4))
        elif etype == _TRI:
            tris = np.vectorize(remap.get)(conn.reshape(-1, 3))

    return xy, quads, tris, edges_of(airfoil_curves), edges_of(outer_curves)


def _write_extruded_msh(
    path: Path,
    xy: np.ndarray,
    quads: np.ndarray,
    tris: np.ndarray,
    af_edges: list[tuple[int, int]],
    out_edges: list[tuple[int, int]],
    dz: float,
    n_layers: int,
) -> int:
    """Spanwise-extrude the 2-D mesh and write an MSH-2.2 volume mesh.

    Quads -> hexes, triangles -> prisms; wall / far-field edges -> lateral quad
    faces; the z=0 and z=dz copies of every cell -> the frontAndBack patch.
    Node ordering (bottom face first, CCW) gives positive cell volumes, so
    gmshToFoam performs no inversions. Returns the 3-D cell count.

    Physical ids: frontAndBack=1, aerofoil=2, freestream=3, fluid=4.
    """
    L = max(1, int(n_layers))
    quads = np.asarray(quads, dtype=np.int64).reshape(-1, 4)
    tris = np.asarray(tris, dtype=np.int64).reshape(-1, 3)

    # (1) Force every base cell counter-clockwise so the extruded hex/prism has
    # positive volume. gmsh's recombination occasionally emits a clockwise cell;
    # extruding it with the bottom-face-first ordering would invert it.
    def _signed_area(cells: np.ndarray) -> np.ndarray:
        p = xy[cells - 1]                         # (m, k, 2); ids are 1-based
        x, y = p[..., 0], p[..., 1]
        k = p.shape[1]
        return 0.5 * sum(x[:, i] * y[:, (i + 1) % k] - x[:, (i + 1) % k] * y[:, i]
                         for i in range(k))
    if len(quads):
        quads[_signed_area(quads) < 0.0] = quads[_signed_area(quads) < 0.0][:, ::-1]
    if len(tris):
        tris[_signed_area(tris) < 0.0] = tris[_signed_area(tris) < 0.0][:, ::-1]

    # (2) Keep only nodes actually referenced (Blossom recombination orphans a
    # few hundred interior nodes; unused points trip a checkMesh warning).
    edge_arr = (np.asarray(af_edges + out_edges, dtype=np.int64).ravel()
                if (af_edges or out_edges) else np.zeros(0, dtype=np.int64))
    used = np.unique(np.concatenate([quads.ravel(), tris.ravel(), edge_arr]))
    new = np.zeros(len(xy) + 1, dtype=np.int64)   # old 1-based id -> compact id
    new[used] = np.arange(1, len(used) + 1)
    n = len(used)
    xy = xy[used - 1]
    quads = new[quads]
    tris = new[tris]
    af_edges = [(int(new[a]), int(new[b])) for a, b in af_edges]
    out_edges = [(int(new[a]), int(new[b])) for a, b in out_edges]

    def nid(node_1based: int, level: int) -> int:
        return node_1based + level * n            # node id at z-level `level`

    # Element/node counts are known analytically, so we can write the MSH header
    # counts up front and STREAM every node and element line straight to disk in
    # bounded batches. The previous version built one Python list holding every
    # line and then "\n".join'd it into a single multi-hundred-MB string (peak
    # ~2x that), which on a large mesh was a second RAM spike on top of gmsh.
    n_cells = L * (len(quads) + len(tris))
    n_elems = (n_cells                                   # volume hexes/prisms
               + 2 * (len(quads) + len(tris))            # frontAndBack faces
               + L * (len(af_edges) + len(out_edges)))   # lateral wall/far faces

    BATCH = 65536
    buf: list[str] = []

    with path.open("w") as fh:
        def flush() -> None:
            if buf:
                fh.write("\n".join(buf))
                fh.write("\n")
                buf.clear()

        fh.write("\n".join([
            "$MeshFormat", "2.2 0 8", "$EndMeshFormat",
            "$PhysicalNames", "4",
            '2 1 "frontAndBack"', '2 2 "aerofoil"',
            '2 3 "freestream"', '3 4 "fluid"',
            "$EndPhysicalNames",
            "$Nodes", str((L + 1) * n),
        ]) + "\n")

        # --- nodes ---
        for k in range(L + 1):
            z = f"{dz * k / L:.12g}"
            base = k * n
            for i in range(n):
                buf.append(f"{base + i + 1} {xy[i, 0]:.12g} {xy[i, 1]:.12g} {z}")
                if len(buf) >= BATCH:
                    flush()
        flush()
        fh.write("$EndNodes\n$Elements\n")
        fh.write(f"{n_elems}\n")

        eid = 0

        def emit(etype: int, phys: int, nodes) -> None:
            nonlocal eid
            eid += 1
            buf.append(f"{eid} {etype} 2 {phys} {phys} " + " ".join(map(str, nodes)))
            if len(buf) >= BATCH:
                flush()

        # --- volume cells (fluid, phys 4) ---
        for k in range(L):
            for q in quads:
                a, b, c, d = (int(v) for v in q)
                emit(_HEX, 4, [nid(a, k), nid(b, k), nid(c, k), nid(d, k),
                               nid(a, k + 1), nid(b, k + 1), nid(c, k + 1), nid(d, k + 1)])
            for t in tris:
                a, b, c = (int(v) for v in t)
                emit(_PRISM, 4, [nid(a, k), nid(b, k), nid(c, k),
                                 nid(a, k + 1), nid(b, k + 1), nid(c, k + 1)])

        # --- frontAndBack patch (phys 1): z=0 bottom + z=dz top faces ---
        for lvl in (0, L):
            for q in quads:
                a, b, c, d = (int(v) for v in q)
                emit(_QUAD, 1, [nid(a, lvl), nid(b, lvl), nid(c, lvl), nid(d, lvl)])
            for t in tris:
                a, b, c = (int(v) for v in t)
                emit(_TRI, 1, [nid(a, lvl), nid(b, lvl), nid(c, lvl)])

        # --- lateral wall / far-field faces ---
        for phys, edges in ((2, af_edges), (3, out_edges)):
            for k in range(L):
                for a, b in edges:
                    emit(_QUAD, phys, [nid(a, k), nid(b, k), nid(b, k + 1), nid(a, k + 1)])

        flush()
        fh.write("$EndElements\n")

    return n_cells


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build_bl_mesh(
    case_dir: Path,
    params: dict,
    cfg: dict,
    chord: float = 1.0,
) -> dict[str, float]:
    """Build a boundary-layer-field + frontal-quad mesh for one case.

    Geometry (z=0 plane, chord along +x, LE at the origin):

      * Airfoil — upper spline (LE -> TE_UP), lower spline (LE -> TE_LO) and a
        straight blunt-back wall (TE_UP -> TE_LO); a hole in the fluid domain.
      * Far field — upstream semicircle of radius ``upstream_radius`` centred on
        the LE, closed downstream by top/bottom rails to the outlet at
        x = te_x + downstream_length (same extent the transfinite path uses).

    The ``BoundaryLayer`` field is attached to the three airfoil curves; the
    outer fill is driven by growth-rate-limited Distance/Threshold fields
    (airfoil proximity + blunt-TE corners) plus a long, narrow, smoothly-blended
    wake corridor. The 2-D mesh is generated, then extruded one cell in span by
    :func:`_write_extruded_msh`.
    """
    import gmsh  # local import keeps gmsh optional at module-import time

    start_time = time.perf_counter()

    Re = float(params["Re"])
    thickness = float(params["thickness"])

    # ---- physics-derived near-wall spacing -------------------------------
    h1 = first_cell_height(Re, cfg["y_plus_target"], chord=chord)
    r_bl = float(cfg["bl_growth_ratio"])

    # Total BL field thickness: a multiple of the turbulent delta_99 so the
    # structured layers fully enclose the boundary layer before the fill takes
    # over. The realised layer count and BL-edge cell size derive from it.
    delta99 = bl_thickness(Re, chord)
    bl_total = float(cfg.get("bl_thickness_factor", 2.0)) * delta99
    n_bl = max(2, cells_needed(h1, r_bl, bl_total))
    h_bl_edge = h1 * r_bl ** (n_bl - 1)          # cell size at the BL outer edge

    # ---- blunt trailing edge --------------------------------------------
    te_frac = float(cfg["te_chord_fraction"])
    upper, lower = naca_symmetric(
        thickness, n=int(cfg["surface_points"]), te_chord_fraction=te_frac
    )
    h_te = float(upper[-1, 1])
    if h_te <= 0.0:
        raise RuntimeError(
            f"{case_dir.name}: blunt-TE half-thickness non-positive "
            f"(h_te={h_te:.3e}); check te_chord_fraction / thickness."
        )
    te_x = te_frac * chord

    # ---- fill size targets ----------------------------------------------
    # Surface (chordwise) cell size: match the BL-edge cell so the fill joins
    # the boundary layer smoothly, clamped so the airfoil keeps ~150-600 cells
    # around it regardless of Re.
    surf_cell = float(np.clip(h_bl_edge, chord / 600.0, chord / 150.0))
    far_cell = float(cfg.get("far_cell_size", 1.0)) * chord
    # Every coarsening transition is sized from a single target cell-to-cell
    # growth ratio so no interface is harsher than it. A linear Threshold ramp
    # has local growth ~ 1 + (SizeMax-SizeMin)/(DistMax-DistMin), so the grading
    # distance to span a size change at growth g is (size change)/(g-1).
    g = max(1.05, float(cfg.get("max_growth_ratio", 1.20)))
    af_grade = (far_cell - surf_cell) / (g - 1.0)          # BL-edge band -> far
    # Trailing-edge corner refinement (the blunt back is too short for surf_cell
    # to resolve), graded back to surf_cell so it rejoins the airfoil band
    # smoothly. The nose stays at surf_cell (~30 cells round it is enough).
    le_cell = surf_cell * float(cfg.get("le_cluster_factor", 0.50))
    refine_radius = float(cfg.get("le_refine_radius", 0.05)) * chord
    te_grade = (surf_cell - le_cell) / (g - 1.0)
    # Wake corridor: long, narrow, smoothly blended into the far field.
    wake_cell = float(cfg.get("wake_cell_size", 8.0)) * surf_cell
    wake_len = float(cfg.get("wake_box_length", 12.0)) * chord
    wake_hw = float(cfg.get("wake_box_halfwidth", 0.6)) * chord
    wake_grade = (far_cell - wake_cell) / (g - 1.0)

    # ---- far-field reference points -------------------------------------
    ff = farfield_points(
        upstream_radius=cfg["upstream_radius"],
        downstream_length=cfg["downstream_length"],
        transverse_extent=cfg["transverse_extent"],
        transition_wake_length=cfg["transition_wake_length"],
        chord=chord,
        te_chord_fraction=te_frac,
    )

    # ---- gmsh setup (2-D only) ------------------------------------------
    gmsh.clear()
    gmsh.model.add(case_dir.name)
    gmsh.option.setNumber("General.Terminal", 0)
    threads, mem_cap = _apply_gmsh_resource_limits(gmsh)
    gmsh.option.setNumber("Mesh.Algorithm", 8)               # Frontal-Delaunay for Quads
    gmsh.option.setNumber("Mesh.RecombinationAlgorithm",
                          int(cfg.get("recombine_algorithm", 1)))  # 1=Blossom, 0=simple
    gmsh.option.setNumber("Mesh.RecombineAll", 1)
    gmsh.option.setNumber("Mesh.ElementOrder", 1)
    # Size comes only from the background field, not points/curvature/boundary.
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    # Laplacian smoothing of the unstructured fill improves far-field triangle
    # angles; it does not touch the structured BL-field layers.
    gmsh.option.setNumber("Mesh.Smoothing", int(cfg.get("mesh_smoothing", 5)))

    geo = gmsh.model.geo

    # ---- airfoil points & curves ----------------------------------------
    p_LE = geo.addPoint(0.0, 0.0, 0.0)               # leading edge / arc centre
    p_TE_UP = geo.addPoint(te_x, +h_te, 0.0)
    p_TE_LO = geo.addPoint(te_x, -h_te, 0.0)
    up_inner = [geo.addPoint(float(x), float(y), 0.0) for x, y in upper[1:-1]]
    lo_inner = [geo.addPoint(float(x), float(y), 0.0) for x, y in lower[1:-1]]
    C_af_up = geo.addSpline([p_LE, *up_inner, p_TE_UP])   # LE -> TE_UP
    C_af_lo = geo.addSpline([p_LE, *lo_inner, p_TE_LO])   # LE -> TE_LO
    C_te = geo.addLine(p_TE_UP, p_TE_LO)                  # blunt back wall

    airfoil_curves = [C_af_up, C_te, C_af_lo]   # wall patch + proximity refinement
    # The BoundaryLayer field grows on the upper/lower surfaces ONLY; the blunt
    # back wall (C_te) is deliberately excluded. Including it made the upper and
    # lower wall-normal layers collide behind the TE into a fan singularity
    # (high aspect-ratio, distorted cells, poor BL->wake handoff). With C_te out
    # of the BL, the two surface BL strips terminate at the TE corners and fan
    # cleanly into the wake fill, and the short blunt back is meshed by the
    # frontal-quad algorithm (still a wall patch via airfoil_curves above).
    bl_curves = [C_af_up, C_af_lo]
    loop_af = geo.addCurveLoop([C_af_up, C_te, -C_af_lo])

    # ---- far-field outer boundary (CCW: fluid on the left) --------------
    p_LE_FAR = geo.addPoint(*ff.LE_FAR, 0.0)
    p_TOP_MID = geo.addPoint(*ff.TOP_MID, 0.0)
    p_BOT_MID = geo.addPoint(*ff.BOT_MID, 0.0)
    p_TOP_OUT = geo.addPoint(*ff.TOP_OUT, 0.0)
    p_BOT_OUT = geo.addPoint(*ff.BOT_OUT, 0.0)

    C_arc_up = geo.addCircleArc(p_TOP_MID, p_LE, p_LE_FAR)   # quarter, upper-front
    C_arc_lo = geo.addCircleArc(p_LE_FAR, p_LE, p_BOT_MID)   # quarter, lower-front
    C_bot = geo.addLine(p_BOT_MID, p_BOT_OUT)               # bottom rail -> outlet
    C_outlet = geo.addLine(p_BOT_OUT, p_TOP_OUT)            # outlet
    C_top = geo.addLine(p_TOP_OUT, p_TOP_MID)               # top rail -> front

    outer_curves = [C_arc_up, C_arc_lo, C_bot, C_outlet, C_top]
    loop_outer = geo.addCurveLoop(outer_curves)

    surf = geo.addPlaneSurface([loop_outer, loop_af])    # airfoil = hole
    geo.synchronize()

    # ---- size field (controls the fill outside the BL) -------------------
    # Three growth-rate-limited sources combined by Min:
    #   th_af — coarsen from the BL edge (surf_cell) out to far_cell
    #   th_te — refine the blunt-TE corners, grade back to surf_cell
    #   box   — a long, narrow, smoothly-blended wake corridor
    F = gmsh.model.mesh.field

    d_af = F.add("Distance")
    F.setNumbers(d_af, "CurvesList", airfoil_curves)
    F.setNumber(d_af, "Sampling", 800)
    th_af = F.add("Threshold")
    F.setNumber(th_af, "InField", d_af)
    F.setNumber(th_af, "SizeMin", surf_cell)
    F.setNumber(th_af, "SizeMax", far_cell)
    F.setNumber(th_af, "DistMin", bl_total)        # fine through the BL band
    F.setNumber(th_af, "DistMax", bl_total + af_grade)

    d_te = F.add("Distance")
    F.setNumbers(d_te, "PointsList", [p_TE_UP, p_TE_LO])
    th_te = F.add("Threshold")
    F.setNumber(th_te, "InField", d_te)
    F.setNumber(th_te, "SizeMin", le_cell)
    F.setNumber(th_te, "SizeMax", surf_cell)       # rejoin the airfoil band
    F.setNumber(th_te, "DistMin", refine_radius)
    F.setNumber(th_te, "DistMax", refine_radius + te_grade)

    box = F.add("Box")
    F.setNumber(box, "VIn", wake_cell)
    F.setNumber(box, "VOut", far_cell)
    F.setNumber(box, "XMin", te_x)
    F.setNumber(box, "XMax", te_x + wake_len)
    F.setNumber(box, "YMin", -wake_hw)
    F.setNumber(box, "YMax", +wake_hw)
    F.setNumber(box, "ZMin", -1.0)
    F.setNumber(box, "ZMax", +1.0)
    F.setNumber(box, "Thickness", wake_grade)      # growth-limited blend to far

    bg = F.add("Min")
    F.setNumbers(bg, "FieldsList", [th_af, th_te, box])
    F.setAsBackgroundMesh(bg)

    # ---- boundary-layer field -------------------------------------------
    bl = F.add("BoundaryLayer")
    F.setNumbers(bl, "CurvesList", bl_curves)    # upper/lower only; C_te excluded
    F.setNumber(bl, "Size", h1)
    F.setNumber(bl, "Ratio", r_bl)
    F.setNumber(bl, "Thickness", bl_total)
    F.setNumber(bl, "Quads", 1)
    # Fan the two TE corners: with the blunt back out of the BL these are the
    # free ends of the upper/lower BL strips, so the fan splays the terminating
    # layers into the wake fill instead of collapsing them onto the back wall.
    F.setNumbers(bl, "FanPointsList", [p_TE_UP, p_TE_LO])
    F.setNumbers(bl, "PointsList", [p_LE, p_TE_UP, p_TE_LO])
    F.setAsBoundaryLayer(bl)

    # ---- cell-count budget guard (fail before gmsh exhausts RAM) --------
    def _arclen(pts: np.ndarray) -> float:
        d = np.diff(pts, axis=0)
        return float(np.sqrt((d ** 2).sum(axis=1)).sum())
    perimeter = _arclen(upper) + _arclen(lower) + 2.0 * h_te
    est_2d = _estimate_2d_cells(
        cfg, surf_cell, far_cell, wake_len, wake_hw, wake_cell,
        n_bl, perimeter, chord,
    )
    est_3d = est_2d * max(1, int(cfg["spanwise_layers"]))
    budget = int(os.environ.get("NACA_MESH_MAX_CELLS",
                                cfg.get("max_mesh_cells", 1_200_000)))
    log.info(
        "%s mesh-size estimate ~%d 3-D cells (budget %d; %d gmsh threads, "
        "mem cap %s)",
        case_dir.name, est_3d, budget, threads,
        f"{mem_cap / 1024**3:.1f} GB" if mem_cap else "off",
    )
    if est_3d > budget:
        raise RuntimeError(
            f"{case_dir.name}: estimated ~{est_3d:,} cells exceeds the budget of "
            f"{budget:,}; this mesh would likely exhaust RAM. Coarsen "
            f"'wake_cell_size' (dominant cost) or 'far_cell_size', or raise the "
            f"budget via 'max_mesh_cells' / env NACA_MESH_MAX_CELLS if the machine "
            f"can handle it."
        )

    # ---- 2-D mesh, then manual spanwise extrusion -----------------------
    gmsh.model.mesh.generate(2)
    xy, quads, tris, af_edges, out_edges = _extract_2d(
        gmsh, surf, airfoil_curves, outer_curves
    )
    if len(quads) == 0 and len(tris) == 0:
        raise RuntimeError(f"{case_dir.name}: 2-D mesh produced no surface cells")
    if not af_edges:
        raise RuntimeError(f"{case_dir.name}: no aerofoil boundary edges extracted")
    if not out_edges:
        raise RuntimeError(f"{case_dir.name}: no freestream boundary edges extracted")

    mesh_path = case_dir / "mesh.msh"
    cell_count = _write_extruded_msh(
        mesh_path, xy, quads, tris, af_edges, out_edges,
        dz=cfg["spanwise_thickness"] * chord,
        n_layers=int(cfg["spanwise_layers"]),
    )

    runtime = time.perf_counter() - start_time
    n_quad, n_tri = int(len(quads)), int(len(tris))
    hex_frac_2d = n_quad / (n_quad + n_tri) if (n_quad + n_tri) else 0.0

    log.info(
        "%s bl_field mesh: h1=%.3e r=%.3f n_bl=%d (BL thick=%.3e, edge cell=%.3e) "
        "surf_cell=%.3e far_cell=%.2f | 2D quads=%d tris=%d (quad frac=%.4f) "
        "-> 3D cells=%d",
        case_dir.name, h1, r_bl, n_bl, bl_total, h_bl_edge,
        surf_cell, far_cell, n_quad, n_tri, hex_frac_2d, cell_count,
    )

    return {
        "first_cell_height":    float(h1),
        "normal_progression":   float(r_bl),       # alias for metadata schema
        "bl_growth_ratio":      float(r_bl),
        "bl_layers":            int(n_bl),
        "bl_total_thickness":   float(bl_total),
        "bl_edge_cell":         float(h_bl_edge),
        "delta99":              float(delta99),
        "surface_cell_size":    float(surf_cell),
        "le_cell_size":         float(le_cell),
        "far_cell_size":        float(far_cell),
        "te_chord_fraction":    float(te_frac),
        "te_half_thickness":    float(h_te),
        "n_quads_2d":           n_quad,
        "n_tris_2d":            n_tri,
        "quad_fraction_2d":     float(hex_frac_2d),
        "cell_count_gmsh":      int(cell_count),
        "mesh_runtime_s":       float(runtime),
    }
