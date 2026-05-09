#!/usr/bin/env python3
"""
Script: 03_mesh.py
Stage:  3 — Automated meshing
Purpose: Build a gmsh-based C-domain mesh for each generated aerofoil case,
         convert it to OpenFOAM format, and enforce basic checkMesh quality
         limits before the CFD stage.

Usage:
    micromamba run -n openfoam python scripts/03_mesh.py
    micromamba run -n openfoam python scripts/03_mesh.py --case-id 0 1 2
    micromamba run -n openfoam python scripts/03_mesh.py --force
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import logging
import re
import shlex
import shutil
import subprocess
from pathlib import Path

import numpy as np

try:
    import gmsh  # type: ignore[import-not-found]
except ModuleNotFoundError:
    gmsh = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CASES_DIR = PROJECT_ROOT / "cases"
OPENFOAM_BASHRC = Path("/opt/openfoam12/etc/bashrc")

CHORD = 1.0
NU = 1.5e-5
RHO = 1.225
TARGET_Y_PLUS = 0.5

UPSTREAM_RADIUS = 20.0 * CHORD
DOWNSTREAM_LENGTH = 30.0 * CHORD
TRANSVERSE_EXTENT = 20.0 * CHORD
SPANWISE_THICKNESS = 0.05 * CHORD

BOUNDARY_LAYER_RATIO = 1.18
FARFIELD_SIZE = 1.2

AIRFOIL_TE_SPLIT_X = 0.92
AIRFOIL_LE_SPLIT_X = 0.08
AIRFOIL_TE_NODES = 90
AIRFOIL_MID_NODES = 150
AIRFOIL_LE_NODES = 120
TRAILING_EDGE_CAP_SCALE = 0.4
TRAILING_EDGE_CAP_MIN = 1.0e-4
TRAILING_EDGE_CAP_MAX = 5.0e-4

AIRFOIL_REFINEMENT_SIZE = 0.0035
AIRFOIL_REFINEMENT_DIST_MIN = 0.08
AIRFOIL_REFINEMENT_DIST_MAX = 3.0

SEPARATION_REFINEMENT_SIZE = 0.003
SEPARATION_ZONE_XMIN = 0.7
SEPARATION_ZONE_XMAX = 1.2
SEPARATION_ZONE_YMIN = -0.15
SEPARATION_ZONE_YMAX = 0.15

WAKE_STATION_X = (1.01, 1.5, 4.0, 10.0, DOWNSTREAM_LENGTH)
WAKE_HALF_HEIGHTS = (0.001, 0.02, 0.09, 0.3, 0.8)
WAKE_CONNECTOR_NODES = 18
WAKE_TRANSVERSE_NODES = 28
WAKE_STREAMWISE_NODES = (110, 180, 220, 180)

WAKE_NEAR_SIZE = 0.005
WAKE_MID_SIZE = 0.012
WAKE_FAR_SIZE = 0.03

NON_ORTHOGONALITY_LIMIT = 70.0
SKEWNESS_LIMIT = 4.0
ASPECT_RATIO_LIMIT = 10000.0

MINIMAL_CONTROL_DICT = """\
/*--------------------------------*- C++ -*----------------------------------*\\
  =========                 |
  \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\\\    /   O peration     | Website:  https://openfoam.org
    \\\\  /    A nd           | Version:  12
     \\\\/     M anipulation  |
\\*---------------------------------------------------------------------------*/
FoamFile
{
    format      ascii;
    class       dictionary;
    location    "system";
    object      controlDict;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

application     gmshToFoam;

startFrom       startTime;

startTime       0;

stopAt          endTime;

endTime         1;

deltaT          1;

writeControl    timeStep;

writeInterval   1;

purgeWrite      0;

writeFormat     ascii;

writePrecision  10;

writeCompression off;

timeFormat      general;

timePrecision   6;

runTimeModifiable true;

// ************************************************************************* //
"""


@dataclass(frozen=True)
class AirfoilTopology:
    loop_tag: int
    curve_tags: list[int]
    boundary_layer_curve_tags: list[int]
    upper_curve_tags: list[int]
    lower_curve_tags: list[int]
    te_upper_point_tag: int
    te_lower_point_tag: int
    le_point_tag: int


@dataclass(frozen=True)
class SurfaceTopology:
    fluid_surface_tags: list[int]
    outer_curve_tags: list[int]
    airfoil_curve_tags: list[int]
    wake_surface_tag: int
    embedded_curve_tags: list[int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate gmsh meshes for one or more cases, convert them with "
            "gmshToFoam, and validate quality with checkMesh."
        )
    )
    parser.add_argument(
        "--case-id",
        type=int,
        nargs="*",
        help="Specific case IDs to process. Defaults to every case with params.json.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild meshes even if constant/polyMesh already exists.",
    )
    return parser.parse_args()


def require_gmsh():
    if gmsh is None:
        raise ModuleNotFoundError(
            "The gmsh Python API is not available. Create or activate the "
            "'openfoam' micromamba environment from environment.yml first."
        )
    return gmsh


def collect_case_dirs(case_ids: list[int] | None) -> list[Path]:
    if case_ids:
        case_dirs = [CASES_DIR / f"case_{case_id:04d}" for case_id in case_ids]
    else:
        case_dirs = sorted(
            case_dir
            for case_dir in CASES_DIR.glob("case_*")
            if (case_dir / "params.json").exists()
        )

    missing = [case_dir for case_dir in case_dirs if not (case_dir / "params.json").exists()]
    if missing:
        names = ", ".join(case_dir.name for case_dir in missing)
        raise FileNotFoundError(f"Missing params.json for: {names}")

    return case_dirs


def load_params(case_dir: Path) -> dict[str, float]:
    payload = json.loads((case_dir / "params.json").read_text())
    return {
        "alpha_deg": float(payload["alpha_deg"]),
        "Re": float(payload["Re"]),
        "thickness": float(payload["thickness"]),
    }


def first_cell_height(
    reynolds_number: float,
    chord: float = CHORD,
    nu: float = NU,
    y_plus: float = TARGET_Y_PLUS,
) -> float:
    """Compute wall-normal first cell height from the project y+ rule."""
    cf = 0.026 / reynolds_number ** (1.0 / 7.0)
    u_inf = reynolds_number * nu / chord
    tau_w = 0.5 * RHO * u_inf ** 2 * cf
    u_tau = (tau_w / RHO) ** 0.5
    return y_plus * nu / u_tau


def boundary_layer_thickness(reynolds_number: float, chord: float = CHORD) -> float:
    """Estimate turbulent boundary-layer thickness near the trailing edge."""
    delta_99 = 0.37 * chord / reynolds_number ** 0.2
    return float(np.clip(1.25 * delta_99, 0.02 * chord, 0.05 * chord))


def load_aerofoil_coordinates(aerofoil_dat_path: Path) -> np.ndarray:
    coords = np.loadtxt(aerofoil_dat_path, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"{aerofoil_dat_path} must contain two columns of x y coordinates")
    if len(coords) < 10:
        raise ValueError(f"{aerofoil_dat_path} has too few points for a usable spline")
    if np.allclose(coords[0], coords[-1]):
        coords = coords[:-1]
    return coords


def reset_case_mesh(case_dir: Path) -> None:
    poly_mesh_dir = case_dir / "constant" / "polyMesh"
    if poly_mesh_dir.exists():
        shutil.rmtree(poly_mesh_dir)

    for filename in ("mesh.msh", "log.gmshToFoam", "log.checkMesh"):
        path = case_dir / filename
        if path.exists():
            path.unlink()


def ensure_case_scaffold(case_dir: Path) -> None:
    (case_dir / "system").mkdir(parents=True, exist_ok=True)
    (case_dir / "constant").mkdir(parents=True, exist_ok=True)
    control_dict_path = case_dir / "system" / "controlDict"
    if not control_dict_path.exists():
        control_dict_path.write_text(MINIMAL_CONTROL_DICT)


def sharpen_airfoil_trailing_edge(coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    leading_edge_index = int(np.argmin(coords[:, 0]))
    upper_surface = np.array(coords[: leading_edge_index + 1], copy=True)
    lower_surface = np.array(coords[leading_edge_index:], copy=True)

    trailing_edge_x = 0.5 * float(upper_surface[0, 0] + lower_surface[-1, 0])
    original_half_gap = 0.5 * abs(float(upper_surface[0, 1] - lower_surface[-1, 1]))
    cap_half_height = np.clip(
        TRAILING_EDGE_CAP_SCALE * original_half_gap,
        TRAILING_EDGE_CAP_MIN,
        TRAILING_EDGE_CAP_MAX,
    )

    upper_surface[0] = np.array([trailing_edge_x, cap_half_height], dtype=np.float64)
    lower_surface[-1] = np.array([trailing_edge_x, -cap_half_height], dtype=np.float64)

    return upper_surface, lower_surface


def segment_split_indices(surface_coords: np.ndarray, target_xs: tuple[float, ...]) -> list[int]:
    interior_x = surface_coords[1:-1, 0]
    if len(interior_x) < 4:
        raise ValueError("Airfoil surface has too few points to segment for mesh control")

    indices: list[int] = []
    previous = 0

    for offset, target_x in enumerate(target_xs):
        candidate = 1 + int(np.argmin(np.abs(interior_x - target_x)))
        candidate = max(candidate, previous + 2)

        remaining = len(target_xs) - offset - 1
        max_candidate = len(surface_coords) - 3 - 2 * remaining
        candidate = min(candidate, max_candidate)

        indices.append(candidate)
        previous = candidate

    return indices


def add_curve_segments(geo, point_tags: list[int], split_indices: list[int]) -> list[int]:
    bounds = [0, *split_indices, len(point_tags) - 1]
    curve_tags: list[int] = []

    for start, end in zip(bounds[:-1], bounds[1:]):
        segment_points = point_tags[start : end + 1]
        if len(segment_points) == 2:
            curve_tags.append(geo.addLine(segment_points[0], segment_points[1]))
        else:
            curve_tags.append(geo.addSpline(segment_points))

    return curve_tags


def add_airfoil_loop(gmsh_module, coords: np.ndarray) -> AirfoilTopology:
    geo = gmsh_module.model.geo
    upper_surface, lower_surface = sharpen_airfoil_trailing_edge(coords)

    te_upper_x, te_upper_y = upper_surface[0]
    te_lower_x, te_lower_y = lower_surface[-1]
    le_x, le_y = upper_surface[-1]
    te_upper_point_tag = geo.addPoint(float(te_upper_x), float(te_upper_y), 0.0, FARFIELD_SIZE)
    te_lower_point_tag = geo.addPoint(float(te_lower_x), float(te_lower_y), 0.0, FARFIELD_SIZE)
    le_point_tag = geo.addPoint(float(le_x), float(le_y), 0.0, FARFIELD_SIZE)

    upper_internal_tags = [
        geo.addPoint(float(x), float(y), 0.0, FARFIELD_SIZE) for x, y in upper_surface[1:-1]
    ]
    lower_internal_tags = [
        geo.addPoint(float(x), float(y), 0.0, FARFIELD_SIZE) for x, y in lower_surface[1:-1]
    ]

    upper_point_tags = [te_upper_point_tag, *upper_internal_tags, le_point_tag]
    lower_point_tags = [le_point_tag, *lower_internal_tags, te_lower_point_tag]

    upper_curve_tags = add_curve_segments(
        geo,
        upper_point_tags,
        segment_split_indices(upper_surface, (AIRFOIL_TE_SPLIT_X, AIRFOIL_LE_SPLIT_X)),
    )
    lower_curve_tags = add_curve_segments(
        geo,
        lower_point_tags,
        segment_split_indices(lower_surface, (AIRFOIL_LE_SPLIT_X, AIRFOIL_TE_SPLIT_X)),
    )

    for curve_tag, n_nodes in zip(
        upper_curve_tags,
        (AIRFOIL_TE_NODES, AIRFOIL_MID_NODES, AIRFOIL_LE_NODES),
    ):
        geo.mesh.setTransfiniteCurve(curve_tag, n_nodes)

    for curve_tag, n_nodes in zip(
        lower_curve_tags,
        (AIRFOIL_LE_NODES, AIRFOIL_MID_NODES, AIRFOIL_TE_NODES),
    ):
        geo.mesh.setTransfiniteCurve(curve_tag, n_nodes)

    trailing_edge_cap = geo.addLine(te_lower_point_tag, te_upper_point_tag)
    geo.mesh.setTransfiniteCurve(trailing_edge_cap, WAKE_TRANSVERSE_NODES)

    airfoil_loop = geo.addCurveLoop([*upper_curve_tags, *lower_curve_tags, trailing_edge_cap])
    return AirfoilTopology(
        loop_tag=airfoil_loop,
        curve_tags=[*upper_curve_tags, *lower_curve_tags, trailing_edge_cap],
        boundary_layer_curve_tags=[*upper_curve_tags, *lower_curve_tags],
        upper_curve_tags=upper_curve_tags,
        lower_curve_tags=lower_curve_tags,
        te_upper_point_tag=te_upper_point_tag,
        te_lower_point_tag=te_lower_point_tag,
        le_point_tag=le_point_tag,
    )


def add_fluid_surfaces(gmsh_module, airfoil: AirfoilTopology) -> SurfaceTopology:
    geo = gmsh_module.model.geo

    center = geo.addPoint(0.0, 0.0, 0.0, FARFIELD_SIZE)
    top = geo.addPoint(0.0, TRANSVERSE_EXTENT, 0.0, FARFIELD_SIZE)
    left = geo.addPoint(-UPSTREAM_RADIUS, 0.0, 0.0, FARFIELD_SIZE)
    bottom = geo.addPoint(0.0, -TRANSVERSE_EXTENT, 0.0, FARFIELD_SIZE)
    outlet_top = geo.addPoint(DOWNSTREAM_LENGTH, TRANSVERSE_EXTENT, 0.0, FARFIELD_SIZE)
    outlet_bottom = geo.addPoint(DOWNSTREAM_LENGTH, -TRANSVERSE_EXTENT, 0.0, FARFIELD_SIZE)

    wake_upper_points: list[int] = []
    wake_lower_points: list[int] = []
    for x_station, half_height in zip(WAKE_STATION_X, WAKE_HALF_HEIGHTS):
        wake_upper_points.append(geo.addPoint(float(x_station), float(half_height), 0.0, FARFIELD_SIZE))
        wake_lower_points.append(geo.addPoint(float(x_station), float(-half_height), 0.0, FARFIELD_SIZE))

    outlet_wake_upper = wake_upper_points[-1]
    outlet_wake_lower = wake_lower_points[-1]

    wake_upper_segments = [
        geo.addLine(wake_upper_points[i], wake_upper_points[i + 1])
        for i in range(len(wake_upper_points) - 1)
    ]
    wake_lower_segments = [
        geo.addLine(wake_lower_points[i], wake_lower_points[i + 1])
        for i in range(len(wake_lower_points) - 1)
    ]
    wake_cross_sections = [
        geo.addLine(wake_upper_points[i], wake_lower_points[i]) for i in range(len(wake_upper_points))
    ]
    wake_connector_upper = geo.addLine(airfoil.te_upper_point_tag, wake_upper_points[0])
    wake_connector_lower = geo.addLine(airfoil.te_lower_point_tag, wake_lower_points[0])

    for curve_tag in wake_cross_sections:
        geo.mesh.setTransfiniteCurve(curve_tag, WAKE_TRANSVERSE_NODES)

    geo.mesh.setTransfiniteCurve(wake_connector_upper, WAKE_CONNECTOR_NODES)
    geo.mesh.setTransfiniteCurve(wake_connector_lower, WAKE_CONNECTOR_NODES)

    for curve_tag, n_nodes in zip(wake_upper_segments, WAKE_STREAMWISE_NODES):
        geo.mesh.setTransfiniteCurve(curve_tag, n_nodes)
    for curve_tag, n_nodes in zip(wake_lower_segments, WAKE_STREAMWISE_NODES):
        geo.mesh.setTransfiniteCurve(curve_tag, n_nodes)

    top_curve = geo.addLine(top, outlet_top)
    outlet_upper = geo.addLine(outlet_top, outlet_wake_upper)
    outlet_lower = geo.addLine(outlet_wake_lower, outlet_bottom)
    bottom_curve = geo.addLine(outlet_bottom, bottom)
    lower_arc = geo.addCircleArc(bottom, center, left)
    upper_arc = geo.addCircleArc(left, center, top)
    leading_edge_connector = geo.addLine(airfoil.le_point_tag, left)

    wake_loop = geo.addCurveLoop(
        [
            wake_connector_upper,
            *wake_upper_segments,
            wake_cross_sections[-1],
            *[-curve_tag for curve_tag in reversed(wake_lower_segments)],
            -wake_connector_lower,
            airfoil.curve_tags[-1],
        ]
    )
    wake_surface = geo.addPlaneSurface([wake_loop])

    upper_outer_loop = geo.addCurveLoop(
        [
            top_curve,
            outlet_upper,
            *[-curve_tag for curve_tag in reversed(wake_upper_segments)],
            -wake_connector_upper,
            *airfoil.upper_curve_tags,
            leading_edge_connector,
            upper_arc,
        ]
    )
    lower_outer_loop = geo.addCurveLoop(
        [
            -leading_edge_connector,
            *airfoil.lower_curve_tags,
            wake_connector_lower,
            *wake_lower_segments,
            outlet_lower,
            bottom_curve,
            lower_arc,
        ]
    )

    upper_outer_surface = geo.addPlaneSurface([upper_outer_loop])
    lower_outer_surface = geo.addPlaneSurface([lower_outer_loop])

    return SurfaceTopology(
        fluid_surface_tags=[upper_outer_surface, lower_outer_surface, wake_surface],
        outer_curve_tags=[
            top_curve,
            outlet_upper,
            wake_cross_sections[-1],
            outlet_lower,
            bottom_curve,
            lower_arc,
            upper_arc,
        ],
        airfoil_curve_tags=airfoil.curve_tags,
        wake_surface_tag=wake_surface,
        embedded_curve_tags=wake_cross_sections[:-1],
    )


def classify_lateral_surfaces(
    gmsh_module,
    lateral_surface_tags: list[int],
    outer_curve_tags: list[int],
    airfoil_curve_tags: list[int],
) -> tuple[list[int], list[int]]:
    outer_curve_set = set(outer_curve_tags)
    airfoil_curve_set = set(airfoil_curve_tags)

    freestream_surfaces: list[int] = []
    aerofoil_surfaces: list[int] = []

    for surface_tag in lateral_surface_tags:
        boundary_curves = {
            abs(curve_tag)
            for dim, curve_tag in gmsh_module.model.getBoundary(
                [(2, surface_tag)],
                combined=False,
                oriented=False,
                recursive=False,
            )
            if dim == 1
        }

        if boundary_curves & outer_curve_set:
            freestream_surfaces.append(surface_tag)
        elif boundary_curves & airfoil_curve_set:
            aerofoil_surfaces.append(surface_tag)

    if not freestream_surfaces:
        raise RuntimeError("Failed to identify any freestream surfaces after extrusion")
    if not aerofoil_surfaces:
        raise RuntimeError("Failed to identify any aerofoil wall surfaces after extrusion")

    return freestream_surfaces, aerofoil_surfaces


def configure_mesh_fields(
    gmsh_module,
    airfoil_curve_tags: list[int],
    first_layer_height: float,
    bl_thickness: float,
) -> None:
    field = gmsh_module.model.mesh.field

    distance_field = field.add("Distance")
    field.setNumbers(distance_field, "CurvesList", airfoil_curve_tags)
    field.setNumber(distance_field, "Sampling", 1000)

    airfoil_threshold = field.add("Threshold")
    field.setNumber(airfoil_threshold, "InField", distance_field)
    field.setNumber(airfoil_threshold, "SizeMin", AIRFOIL_REFINEMENT_SIZE)
    field.setNumber(airfoil_threshold, "SizeMax", FARFIELD_SIZE)
    field.setNumber(airfoil_threshold, "DistMin", AIRFOIL_REFINEMENT_DIST_MIN)
    field.setNumber(airfoil_threshold, "DistMax", AIRFOIL_REFINEMENT_DIST_MAX)

    separation_zone = field.add("Box")
    field.setNumber(separation_zone, "VIn", SEPARATION_REFINEMENT_SIZE)
    field.setNumber(separation_zone, "VOut", FARFIELD_SIZE)
    field.setNumber(separation_zone, "XMin", SEPARATION_ZONE_XMIN)
    field.setNumber(separation_zone, "XMax", SEPARATION_ZONE_XMAX)
    field.setNumber(separation_zone, "YMin", SEPARATION_ZONE_YMIN)
    field.setNumber(separation_zone, "YMax", SEPARATION_ZONE_YMAX)
    field.setNumber(separation_zone, "ZMin", -1.0)
    field.setNumber(separation_zone, "ZMax", 1.0)

    wake_near = field.add("Box")
    field.setNumber(wake_near, "VIn", WAKE_NEAR_SIZE)
    field.setNumber(wake_near, "VOut", FARFIELD_SIZE)
    field.setNumber(wake_near, "XMin", 1.0)
    field.setNumber(wake_near, "XMax", 5.0)
    field.setNumber(wake_near, "YMin", -0.15)
    field.setNumber(wake_near, "YMax", 0.15)
    field.setNumber(wake_near, "ZMin", -1.0)
    field.setNumber(wake_near, "ZMax", 1.0)

    wake_mid = field.add("Box")
    field.setNumber(wake_mid, "VIn", WAKE_MID_SIZE)
    field.setNumber(wake_mid, "VOut", FARFIELD_SIZE)
    field.setNumber(wake_mid, "XMin", 5.0)
    field.setNumber(wake_mid, "XMax", 12.0)
    field.setNumber(wake_mid, "YMin", -0.4)
    field.setNumber(wake_mid, "YMax", 0.4)
    field.setNumber(wake_mid, "ZMin", -1.0)
    field.setNumber(wake_mid, "ZMax", 1.0)

    wake_far = field.add("Box")
    field.setNumber(wake_far, "VIn", WAKE_FAR_SIZE)
    field.setNumber(wake_far, "VOut", FARFIELD_SIZE)
    field.setNumber(wake_far, "XMin", 12.0)
    field.setNumber(wake_far, "XMax", DOWNSTREAM_LENGTH)
    field.setNumber(wake_far, "YMin", -1.0)
    field.setNumber(wake_far, "YMax", 1.0)
    field.setNumber(wake_far, "ZMin", -1.0)
    field.setNumber(wake_far, "ZMax", 1.0)

    minimum = field.add("Min")
    field.setNumbers(
        minimum,
        "FieldsList",
        [airfoil_threshold, separation_zone, wake_near, wake_mid, wake_far],
    )
    field.setAsBackgroundMesh(minimum)

    # BoundaryLayer is NOT a size field — it must be registered via
    # setAsBoundaryLayer so gmsh extrudes prismatic layers along the wall
    # curves. Folding it into the Min above only consumes its Size as a
    # generic sizing hint and produces no inflation layer.
    boundary_layer = field.add("BoundaryLayer")
    field.setNumbers(boundary_layer, "CurvesList", airfoil_curve_tags)
    field.setNumber(boundary_layer, "Size", first_layer_height)
    field.setNumber(boundary_layer, "Ratio", BOUNDARY_LAYER_RATIO)
    field.setNumber(boundary_layer, "Thickness", bl_thickness)
    field.setNumber(boundary_layer, "Quads", 1)
    field.setAsBoundaryLayer(boundary_layer)


def add_named_physical_groups(
    gmsh_module,
    base_surface_tags: list[int],
    outer_curve_tags: list[int],
    airfoil_curve_tags: list[int],
) -> None:
    tolerance = 1e-9
    top_surface_tags: list[int] = []
    lateral_surfaces: list[int] = []

    for dim, tag in gmsh_module.model.getEntities(2):
        if tag in base_surface_tags:
            continue

        _, _, z_min, _, _, z_max = gmsh_module.model.getBoundingBox(dim, tag)
        if abs(z_min - SPANWISE_THICKNESS) < tolerance and abs(z_max - SPANWISE_THICKNESS) < tolerance:
            top_surface_tags.append(tag)
        elif z_min < -tolerance or z_max > SPANWISE_THICKNESS + tolerance:
            continue
        elif z_min < tolerance and z_max > SPANWISE_THICKNESS - tolerance:
            lateral_surfaces.append(tag)

    volume_tags = [tag for dim, tag in gmsh_module.model.getEntities(3)]

    freestream_surfaces, aerofoil_surfaces = classify_lateral_surfaces(
        gmsh_module,
        lateral_surfaces,
        outer_curve_tags,
        airfoil_curve_tags,
    )

    front_back_tag = gmsh_module.model.addPhysicalGroup(2, [*base_surface_tags, *top_surface_tags])
    gmsh_module.model.setPhysicalName(2, front_back_tag, "frontAndBack")

    freestream_tag = gmsh_module.model.addPhysicalGroup(2, freestream_surfaces)
    gmsh_module.model.setPhysicalName(2, freestream_tag, "freestream")

    aerofoil_tag = gmsh_module.model.addPhysicalGroup(2, aerofoil_surfaces)
    gmsh_module.model.setPhysicalName(2, aerofoil_tag, "aerofoil")

    volume_group = gmsh_module.model.addPhysicalGroup(3, volume_tags)
    gmsh_module.model.setPhysicalName(3, volume_group, "fluid")


def count_volume_cells(gmsh_module) -> int:
    _, element_tags, _ = gmsh_module.model.mesh.getElements(3)
    return int(sum(len(tags) for tags in element_tags))


def run_openfoam_command(case_dir: Path, command: str, log_name: str) -> None:
    log_path = case_dir / log_name
    full_command = (
        f"source {shlex.quote(str(OPENFOAM_BASHRC))} && "
        f"{command} > {shlex.quote(log_name)} 2>&1"
    )
    result = subprocess.run(
        ["bash", "-lc", full_command],
        cwd=case_dir,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        log_tail = ""
        if log_path.exists():
            log_tail = log_path.read_text(errors="ignore")[-1200:]
        raise RuntimeError(
            f"{command} failed in {case_dir.name}; inspect {log_name}\n{log_tail}"
        )


def rewrite_boundary_types(boundary_path: Path) -> None:
    patch_types = {
        "freestream": "patch",
        "aerofoil": "wall",
        "frontAndBack": "empty",
    }

    lines = boundary_path.read_text().splitlines()
    current_patch: str | None = None

    for index, line in enumerate(lines):
        stripped = line.strip()

        if stripped in patch_types:
            current_patch = stripped
            continue

        if current_patch is None:
            continue

        if stripped.startswith("type"):
            indent = line[: len(line) - len(line.lstrip())]
            lines[index] = f"{indent}type            {patch_types[current_patch]};"
            continue

        if stripped.startswith("physicalType"):
            indent = line[: len(line) - len(line.lstrip())]
            lines[index] = f"{indent}physicalType    {patch_types[current_patch]};"
            continue

        if stripped == "}":
            current_patch = None

    boundary_path.write_text("\n".join(lines) + "\n")


def extract_metric(pattern: str, text: str) -> float | None:
    match = re.search(pattern, text, re.MULTILINE)
    if not match:
        return None
    return float(match.group(1))


def parse_check_mesh_log(log_path: Path) -> dict[str, float | None]:
    text = log_path.read_text()
    return {
        "max_non_orthogonality": extract_metric(
            r"Mesh non-orthogonality Max:\s*([0-9.eE+-]+)",
            text,
        ),
        "max_skewness": extract_metric(
            r"Max skewness =\s*([0-9.eE+-]+)",
            text,
        ),
        "max_aspect_ratio": extract_metric(
            r"Max aspect ratio[:=]\s*([0-9.eE+-]+)",
            text,
        ),
    }


def validate_mesh_quality(metrics: dict[str, float | None], case_dir: Path) -> None:
    non_orthogonality = metrics["max_non_orthogonality"]
    skewness = metrics["max_skewness"]
    aspect_ratio = metrics["max_aspect_ratio"]

    if non_orthogonality is None:
        raise RuntimeError(f"Could not parse non-orthogonality from {case_dir / 'log.checkMesh'}")
    if skewness is None:
        raise RuntimeError(f"Could not parse skewness from {case_dir / 'log.checkMesh'}")

    if non_orthogonality >= NON_ORTHOGONALITY_LIMIT:
        raise RuntimeError(
            f"{case_dir.name} failed checkMesh: max non-orthogonality "
            f"{non_orthogonality:.3f} >= {NON_ORTHOGONALITY_LIMIT:.1f}"
        )
    if skewness >= SKEWNESS_LIMIT:
        raise RuntimeError(
            f"{case_dir.name} failed checkMesh: max skewness "
            f"{skewness:.3f} >= {SKEWNESS_LIMIT:.1f}"
        )
    if aspect_ratio is not None and aspect_ratio >= ASPECT_RATIO_LIMIT:
        raise RuntimeError(
            f"{case_dir.name} failed checkMesh: max aspect ratio "
            f"{aspect_ratio:.3f} >= {ASPECT_RATIO_LIMIT:.1f}"
        )


def build_mesh(
    aerofoil_dat_path: Path,
    reynolds_number: float,
    output_dir: Path,
    chord: float = CHORD,
    target_y_plus: float = TARGET_Y_PLUS,
) -> dict[str, float]:
    gmsh_module = require_gmsh()

    coords = load_aerofoil_coordinates(aerofoil_dat_path)
    first_layer_height = first_cell_height(
        reynolds_number,
        chord=chord,
        nu=NU,
        y_plus=target_y_plus,
    )
    bl_thickness = boundary_layer_thickness(reynolds_number, chord=chord)

    gmsh_module.clear()
    gmsh_module.model.add(output_dir.name)

    gmsh_module.option.setNumber("General.Terminal", 0)
    gmsh_module.option.setNumber("Mesh.MshFileVersion", 2.2)
    gmsh_module.option.setNumber("Mesh.SaveAll", 0)
    gmsh_module.option.setNumber("Mesh.Algorithm", 6)            # Frontal-Delaunay
    gmsh_module.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh_module.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    gmsh_module.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)

    airfoil = add_airfoil_loop(gmsh_module, coords)
    surface_topology = add_fluid_surfaces(gmsh_module, airfoil)

    extruded_entities = gmsh_module.model.geo.extrude(
        [(2, surface_tag) for surface_tag in surface_topology.fluid_surface_tags],
        0.0,
        0.0,
        SPANWISE_THICKNESS,
        [1],
        [1.0],
        recombine=True,
    )

    gmsh_module.model.geo.synchronize()
    gmsh_module.model.mesh.embed(
        1,
        surface_topology.embedded_curve_tags,
        2,
        surface_topology.wake_surface_tag,
    )

    add_named_physical_groups(
        gmsh_module,
        surface_topology.fluid_surface_tags,
        surface_topology.outer_curve_tags,
        surface_topology.airfoil_curve_tags,
    )
    configure_mesh_fields(
        gmsh_module,
        airfoil.boundary_layer_curve_tags,
        first_layer_height,
        bl_thickness,
    )

    gmsh_module.model.mesh.generate(3)
    cell_count = count_volume_cells(gmsh_module)

    mesh_path = output_dir / "mesh.msh"
    gmsh_module.write(str(mesh_path))

    ensure_case_scaffold(output_dir)
    run_openfoam_command(output_dir, "gmshToFoam mesh.msh", "log.gmshToFoam")
    rewrite_boundary_types(output_dir / "constant" / "polyMesh" / "boundary")
    run_openfoam_command(output_dir, "checkMesh", "log.checkMesh")

    metrics = parse_check_mesh_log(output_dir / "log.checkMesh")
    validate_mesh_quality(metrics, output_dir)

    return {
        "first_cell_height": first_layer_height,
        "boundary_layer_thickness": bl_thickness,
        "cell_count": float(cell_count),
        "max_non_orthogonality": float(metrics["max_non_orthogonality"]),
        "max_skewness": float(metrics["max_skewness"]),
        "max_aspect_ratio": float(metrics["max_aspect_ratio"] or -1.0),
    }


def main() -> None:
    args = parse_args()

    if not CASES_DIR.exists():
        raise FileNotFoundError(f"{CASES_DIR} not found — run 02_geometry.py first")
    if not OPENFOAM_BASHRC.exists():
        raise FileNotFoundError(f"{OPENFOAM_BASHRC} not found")

    gmsh_module = require_gmsh()
    case_dirs = collect_case_dirs(args.case_id)
    if not case_dirs:
        log.warning("No case directories with params.json were found under %s", CASES_DIR)
        return

    gmsh_module.initialize()
    failures: list[str] = []
    success_count = 0

    try:
        for case_dir in case_dirs:
            mesh_boundary_path = case_dir / "constant" / "polyMesh" / "boundary"
            if mesh_boundary_path.exists() and not args.force:
                log.info("Skipping %s — mesh already exists (use --force to rebuild)", case_dir.name)
                continue

            try:
                reset_case_mesh(case_dir)
                params = load_params(case_dir)
                metrics = build_mesh(
                    case_dir / "aerofoil.dat",
                    params["Re"],
                    case_dir,
                    chord=CHORD,
                    target_y_plus=TARGET_Y_PLUS,
                )

                success_count += 1
                log.info(
                    "%s  Re=%.3e  h1=%.3e m  cells=%d  nonOrtho=%.2f  skew=%.3f",
                    case_dir.name,
                    params["Re"],
                    metrics["first_cell_height"],
                    int(metrics["cell_count"]),
                    metrics["max_non_orthogonality"],
                    metrics["max_skewness"],
                )
            except Exception as exc:  # noqa: BLE001
                failures.append(f"{case_dir.name}: {exc}")
                log.error("Meshing failed for %s: %s", case_dir.name, exc)

        if failures:
            for failure in failures:
                log.error("%s", failure)
            raise RuntimeError(
                f"Meshing completed with {len(failures)} failure(s); "
                f"{success_count} case(s) passed."
            )

        log.info("Successfully meshed %d case(s)", success_count)
    finally:
        gmsh_module.finalize()


if __name__ == "__main__":
    main()
