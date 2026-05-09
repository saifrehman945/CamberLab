#!/usr/bin/env python3
"""
Script: 03_mesh.py
Stage:  3 — Automated meshing
Purpose: Build an OpenFOAM-native airfoil mesh using a snappyHexMesh slab
         workflow:

         1. blockMesh background slab
         2. surfaceFeatures
         3. snappyHexMesh with wall layers
         4. extrudeMesh from the snapped slab face
         5. createPatch to recover a single frontAndBack patch
         6. checkMesh quality validation

Usage:
    micromamba run -n openfoam python scripts/03_mesh.py
    micromamba run -n openfoam python scripts/03_mesh.py --case-id 0 1 2
    micromamba run -n openfoam python scripts/03_mesh.py --force
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import shlex
import shutil
import subprocess
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CASES_DIR = PROJECT_ROOT / "cases"
OPENFOAM_BASHRC = Path("/opt/openfoam12/etc/bashrc")

CHORD = 1.0
NU = 1.5e-5
RHO = 1.225
TARGET_Y_PLUS = 1.0

UPSTREAM_RADIUS = 20.0 * CHORD
DOWNSTREAM_LENGTH = 30.0 * CHORD
TRANSVERSE_EXTENT = 20.0 * CHORD
SPANWISE_THICKNESS = 0.05 * CHORD

BACKGROUND_X_CELLS = 128
BACKGROUND_Y_CELLS = 96
BACKGROUND_Z_CELLS = 1

SNAPPY_LAYER_COUNT = 30
SNAPPY_LAYER_RATIO = 1.18
NEAR_BODY_X_MIN = -0.5 * CHORD
NEAR_BODY_X_MAX = 1.5 * CHORD
NEAR_BODY_Y_HALF = 1.5 * CHORD
WAKE_CORE_X_MIN = 0.75 * CHORD
WAKE_CORE_Y_HALF = 1.0 * CHORD

MAX_LOCAL_CELLS = 250000
MAX_GLOBAL_CELLS = 2500000
N_CELLS_BETWEEN_LEVELS = 4

NON_ORTHOGONALITY_LIMIT = 70.0
SKEWNESS_LIMIT = 4.0
ASPECT_RATIO_LIMIT = 10000.0

SLAB_CASE_NAME = "mesh_slab"

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

application     blockMesh;

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

MESH_QUALITY_DICT = """\
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
    object      meshQualityDict;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

maxNonOrtho 65;

maxBoundarySkewness 20;
maxInternalSkewness 4;

maxConcave 80;

minVol -1e30;
minTetQuality 1e-15;
minTwist 0.02;
minDeterminant 0.001;
minFaceWeight 0.05;
minVolRatio 0.01;

nSmoothScale   4;
errorReduction 0.75;

relaxed
{
    maxNonOrtho 70;
}

// ************************************************************************* //
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate OpenFOAM-native airfoil meshes using "
            "blockMesh + snappyHexMesh + extrudeMesh, then validate them with "
            "checkMesh."
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


def fmt(value: float) -> str:
    return f"{value:.10g}"


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
    return float(np.clip(1.25 * delta_99, 0.01 * chord, 0.05 * chord))


def layer_stack_thickness(
    first_layer_height: float,
    expansion_ratio: float = SNAPPY_LAYER_RATIO,
    n_layers: int = SNAPPY_LAYER_COUNT,
) -> float:
    if abs(expansion_ratio - 1.0) < 1e-12:
        return first_layer_height * n_layers
    return first_layer_height * (expansion_ratio ** n_layers - 1.0) / (expansion_ratio - 1.0)


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

    slab_case_dir = case_dir / SLAB_CASE_NAME
    if slab_case_dir.exists():
        shutil.rmtree(slab_case_dir)

    for filename in (
        "log.extrudeMesh",
        "log.createPatch",
        "log.checkMesh",
    ):
        path = case_dir / filename
        if path.exists():
            path.unlink()


def ensure_case_scaffold(case_dir: Path) -> None:
    (case_dir / "system").mkdir(parents=True, exist_ok=True)
    (case_dir / "constant").mkdir(parents=True, exist_ok=True)
    control_dict_path = case_dir / "system" / "controlDict"
    if not control_dict_path.exists():
        control_dict_path.write_text(MINIMAL_CONTROL_DICT)


def base_background_cell_size() -> float:
    dx = (UPSTREAM_RADIUS + DOWNSTREAM_LENGTH) / BACKGROUND_X_CELLS
    dy = (2.0 * TRANSVERSE_EXTENT) / BACKGROUND_Y_CELLS
    return min(dx, dy)


def refinement_level(target_size: float, base_size: float) -> int:
    if target_size <= 0.0:
        raise ValueError("target_size must be positive")
    if target_size >= base_size:
        return 0
    return int(math.ceil(math.log(base_size / target_size, 2.0)))


def choose_refinement_levels(layer_thickness: float) -> dict[str, int]:
    return {
        "surface": 5,
        "near_body": 4,
        "wake_core": 3,
    }


def write_airfoil_obj(path: Path, coords: np.ndarray, z_min: float, z_max: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    n_points = len(coords)
    lines = ["o aerofoil"]

    for x, y in coords:
        lines.append(f"v {fmt(float(x))} {fmt(float(y))} {fmt(z_min)}")
    for x, y in coords:
        lines.append(f"v {fmt(float(x))} {fmt(float(y))} {fmt(z_max)}")

    for i in range(n_points):
        j = (i + 1) % n_points
        front_i = i + 1
        front_j = j + 1
        back_i = n_points + i + 1
        back_j = n_points + j + 1
        lines.append(f"f {front_i} {front_j} {back_j}")
        lines.append(f"f {front_i} {back_j} {back_i}")

    path.write_text("\n".join(lines) + "\n")


def render_block_mesh_dict() -> str:
    z_half = 0.5 * SPANWISE_THICKNESS
    x_min = -UPSTREAM_RADIUS
    x_max = DOWNSTREAM_LENGTH
    y_min = -TRANSVERSE_EXTENT
    y_max = TRANSVERSE_EXTENT

    return f"""\
/*--------------------------------*- C++ -*----------------------------------*\\
  =========                 |
  \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\\\    /   O peration     | Website:  https://openfoam.org
    \\\\  /    A nd           | Version:  12
     \\\\/     M anipulation  |
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    format      ascii;
    class       dictionary;
    object      blockMeshDict;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

convertToMeters 1;

vertices
(
    ({fmt(x_min)} {fmt(y_min)} {fmt(-z_half)})
    ({fmt(x_max)} {fmt(y_min)} {fmt(-z_half)})
    ({fmt(x_max)} {fmt(y_max)} {fmt(-z_half)})
    ({fmt(x_min)} {fmt(y_max)} {fmt(-z_half)})
    ({fmt(x_min)} {fmt(y_min)} {fmt(z_half)})
    ({fmt(x_max)} {fmt(y_min)} {fmt(z_half)})
    ({fmt(x_max)} {fmt(y_max)} {fmt(z_half)})
    ({fmt(x_min)} {fmt(y_max)} {fmt(z_half)})
);

blocks
(
    hex (0 1 2 3 4 5 6 7)
    ({BACKGROUND_X_CELLS} {BACKGROUND_Y_CELLS} {BACKGROUND_Z_CELLS})
    simpleGrading (1 1 1)
);

boundary
(
    freestream
    {{
        type patch;
        faces
        (
            (0 4 5 1)
            (1 5 6 2)
            (3 2 6 7)
            (0 3 7 4)
        );
    }}

    symFront
    {{
        type patch;
        faces
        (
            (4 7 6 5)
        );
    }}

    symBack
    {{
        type patch;
        faces
        (
            (0 1 2 3)
        );
    }}
);

// ************************************************************************* //
"""


def render_snappy_hex_mesh_dict(
    levels: dict[str, int],
    first_layer_height: float,
    layer_thickness: float,
) -> str:
    z_half = 0.5 * SPANWISE_THICKNESS
    return f"""\
/*--------------------------------*- C++ -*----------------------------------*\\
  =========                 |
  \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\\\    /   O peration     | Website:  https://openfoam.org
    \\\\  /    A nd           | Version:  12
     \\\\/     M anipulation  |
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    format      ascii;
    class       dictionary;
    object      snappyHexMeshDict;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

castellatedMesh on;
snap            on;
addLayers       on;

geometry
{{
    aerofoil
    {{
        type triSurfaceMesh;
        file "aerofoil.obj";
    }}

    nearBodyBox
    {{
        type searchableBox;
        min ({fmt(NEAR_BODY_X_MIN)} {fmt(-NEAR_BODY_Y_HALF)} {fmt(-z_half)});
        max ({fmt(NEAR_BODY_X_MAX)} {fmt(NEAR_BODY_Y_HALF)} {fmt(z_half)});
    }}

    wakeCore
    {{
        type searchableBox;
        min ({fmt(WAKE_CORE_X_MIN)} {fmt(-WAKE_CORE_Y_HALF)} {fmt(-z_half)});
        max ({fmt(DOWNSTREAM_LENGTH)} {fmt(WAKE_CORE_Y_HALF)} {fmt(z_half)});
    }}
}};

castellatedMeshControls
{{
    maxLocalCells {MAX_LOCAL_CELLS};
    maxGlobalCells {MAX_GLOBAL_CELLS};
    minRefinementCells 0;
    nCellsBetweenLevels {N_CELLS_BETWEEN_LEVELS};

    features ();

    refinementSurfaces
    {{
        aerofoil
        {{
            level ({levels["surface"]} {levels["surface"]});
            patchInfo
            {{
                type wall;
            }}
        }}
    }}

    resolveFeatureAngle 30;

    refinementRegions
    {{
        nearBodyBox
        {{
            mode inside;
            levels ((1e15 {levels["near_body"]}));
        }}

        wakeCore
        {{
            mode inside;
            levels ((1e15 {levels["wake_core"]}));
        }}
    }}

    insidePoint (-10 0 0);
    allowFreeStandingZoneFaces true;
}}

snapControls
{{
    nSmoothPatch 5;
    tolerance 2.5;
    nSolveIter 100;
    nRelaxIter 8;

    nFeatureSnapIter 0;
    implicitFeatureSnap true;
    explicitFeatureSnap false;
    multiRegionFeatureSnap false;
}}

addLayersControls
{{
    relativeSizes false;

    layers
    {{
        aerofoil
        {{
            nSurfaceLayers {SNAPPY_LAYER_COUNT};
        }}
    }}

    expansionRatio {fmt(SNAPPY_LAYER_RATIO)};
    firstLayerThickness {fmt(first_layer_height)};
    minThickness {fmt(0.1 * layer_thickness)};

    nGrow 0;
    featureAngle 130;
    slipFeatureAngle 30;
    nRelaxIter 8;
    nSmoothSurfaceNormals 1;
    nSmoothNormals 5;
    nSmoothThickness 10;
    maxFaceThicknessRatio 0.5;
    maxThicknessToMedialRatio 0.3;
    minMedianAxisAngle 90;
    nBufferCellsNoExtrude 0;
    nMedialAxisIter 10;
    nLayerIter 50;
}}

meshQualityControls
{{
    #include "meshQualityDict"
}}

mergeTolerance 1e-6;

// ************************************************************************* //
"""


def render_extrude_mesh_dict() -> str:
    return f"""\
/*--------------------------------*- C++ -*----------------------------------*\\
  =========                 |
  \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\\\    /   O peration     | Website:  https://openfoam.org
    \\\\  /    A nd           | Version:  12
     \\\\/     M anipulation  |
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    format      ascii;
    class       dictionary;
    object      extrudeMeshDict;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

constructFrom patch;
sourceCase "{SLAB_CASE_NAME}";
sourcePatches (symFront);
exposedPatchName symBack;

flipNormals false;

extrudeModel        linearNormal;

nLayers             1;
expansionRatio      1.0;

linearNormalCoeffs
{{
    thickness       {fmt(SPANWISE_THICKNESS)};
}}

mergeFaces false;
mergeTol 0;

// ************************************************************************* //
"""


def render_create_patch_dict() -> str:
    return """\
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
    object      createPatchDict;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

pointSync false;

patches
(
    {
        name frontAndBack;

        patchInfo
        {
            type empty;
        }

        constructFrom patches;
        patches (symFront symBack);
    }
);

// ************************************************************************* //
"""


def write_meshing_inputs(
    case_dir: Path,
    coords: np.ndarray,
    first_layer_height: float,
    layer_thickness: float,
) -> dict[str, int]:
    slab_case_dir = case_dir / SLAB_CASE_NAME
    ensure_case_scaffold(case_dir)
    ensure_case_scaffold(slab_case_dir)

    levels = choose_refinement_levels(layer_thickness)

    airfoil_obj_path = slab_case_dir / "constant" / "triSurface" / "aerofoil.obj"
    write_airfoil_obj(airfoil_obj_path, coords, -0.5 * SPANWISE_THICKNESS, 0.5 * SPANWISE_THICKNESS)

    (slab_case_dir / "system" / "blockMeshDict").write_text(render_block_mesh_dict())
    (slab_case_dir / "system" / "meshQualityDict").write_text(MESH_QUALITY_DICT)
    (slab_case_dir / "system" / "snappyHexMeshDict").write_text(
        render_snappy_hex_mesh_dict(levels, first_layer_height, layer_thickness)
    )

    (case_dir / "system" / "meshQualityDict").write_text(MESH_QUALITY_DICT)
    (case_dir / "system" / "extrudeMeshDict").write_text(render_extrude_mesh_dict())
    (case_dir / "system" / "createPatchDict").write_text(render_create_patch_dict())

    return levels


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
            log_tail = log_path.read_text(errors="ignore")[-2000:]
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


def extract_int(pattern: str, text: str) -> int | None:
    match = re.search(pattern, text, re.MULTILINE)
    if not match:
        return None
    return int(match.group(1))


def parse_check_mesh_log(log_path: Path) -> dict[str, float | int | None]:
    text = log_path.read_text()
    return {
        "cell_count": extract_int(r"cells:\s+([0-9]+)", text),
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


def parse_snappy_layer_log(log_path: Path) -> dict[str, float | int]:
    text = log_path.read_text()
    matches = re.findall(
        r"^aerofoil\s+\d+\s+(\d+)\s+([0-9.eE+-]+)\s+([0-9.eE+-]+)\s*$",
        text,
        re.MULTILINE,
    )
    if not matches:
        return {
            "wall_layers": -1,
            "wall_layer_thickness": -1.0,
            "wall_first_layer_height": -1.0,
        }

    layers, first_or_total, total = matches[-1]
    first_layer_height = -1.0
    if len(matches) >= 2:
        first_layer_height = float(matches[-2][1])

    return {
        "wall_layers": int(layers),
        "wall_layer_thickness": float(total),
        "wall_first_layer_height": first_layer_height,
    }


def validate_mesh_quality(metrics: dict[str, float | int | None], case_dir: Path) -> None:
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
    coords = load_aerofoil_coordinates(aerofoil_dat_path)
    first_layer_height = first_cell_height(
        reynolds_number,
        chord=chord,
        nu=NU,
        y_plus=target_y_plus,
    )
    estimated_delta_99 = boundary_layer_thickness(reynolds_number, chord=chord)
    layered_thickness = layer_stack_thickness(first_layer_height)
    layer_thickness = float(min(layered_thickness, 0.75 * estimated_delta_99))

    levels = write_meshing_inputs(output_dir, coords, first_layer_height, layer_thickness)
    slab_case_dir = output_dir / SLAB_CASE_NAME

    run_openfoam_command(slab_case_dir, "blockMesh", "log.blockMesh")
    run_openfoam_command(slab_case_dir, "snappyHexMesh -overwrite", "log.snappyHexMesh")
    layer_metrics = parse_snappy_layer_log(slab_case_dir / "log.snappyHexMesh")
    if int(layer_metrics["wall_layers"]) == 0:
        log.warning(
            "%s retained zero wall layers after snappyHexMesh; geometry/wake mesh is usable "
            "but y+ compliance is not achieved yet.",
            output_dir.name,
        )

    run_openfoam_command(output_dir, "extrudeMesh", "log.extrudeMesh")
    run_openfoam_command(output_dir, "createPatch -overwrite", "log.createPatch")
    rewrite_boundary_types(output_dir / "constant" / "polyMesh" / "boundary")
    run_openfoam_command(output_dir, "checkMesh -meshQuality", "log.checkMesh")

    metrics = parse_check_mesh_log(output_dir / "log.checkMesh")
    validate_mesh_quality(metrics, output_dir)

    return {
        "first_cell_height": first_layer_height,
        "boundary_layer_thickness": layer_thickness,
        "cell_count": float(metrics["cell_count"] or -1),
        "max_non_orthogonality": float(metrics["max_non_orthogonality"]),
        "max_skewness": float(metrics["max_skewness"]),
        "max_aspect_ratio": float(metrics["max_aspect_ratio"] or -1.0),
        "surface_refinement_level": float(levels["surface"]),
        "wake_refinement_level": float(levels["wake_core"]),
        "wall_layers": float(layer_metrics["wall_layers"]),
        "wall_layer_thickness": float(layer_metrics["wall_layer_thickness"]),
    }


def main() -> None:
    args = parse_args()

    if not CASES_DIR.exists():
        raise FileNotFoundError(f"{CASES_DIR} not found — run 02_geometry.py first")
    if not OPENFOAM_BASHRC.exists():
        raise FileNotFoundError(f"{OPENFOAM_BASHRC} not found")

    case_dirs = collect_case_dirs(args.case_id)
    if not case_dirs:
        log.warning("No case directories with params.json were found under %s", CASES_DIR)
        return

    failures: list[str] = []
    success_count = 0

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
                "%s  Re=%.3e  y+=%.1f  h1=%.3e m  cells=%d  ref=%d  nonOrtho=%.2f  skew=%.3f",
                case_dir.name,
                params["Re"],
                TARGET_Y_PLUS,
                metrics["first_cell_height"],
                int(metrics["cell_count"]),
                int(metrics["surface_refinement_level"]),
                metrics["max_non_orthogonality"],
                metrics["max_skewness"],
            )
            if int(metrics["wall_layers"]) >= 0:
                log.info(
                    "%s  retained wall layers=%d  layerThickness=%.3e m",
                    case_dir.name,
                    int(metrics["wall_layers"]),
                    metrics["wall_layer_thickness"],
                )
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{case_dir.name}: {exc}")
            log.error("Meshing failed for %s: %s", case_dir.name, exc)

    if failures:
        for failure in failures:
            log.error("%s", failure)
        raise RuntimeError(
            f"Meshing completed with {len(failures)} failure(s); "
            f"{success_count} case(s) succeeded."
        )

    log.info("Meshing completed successfully for %d case(s)", success_count)


if __name__ == "__main__":
    main()
