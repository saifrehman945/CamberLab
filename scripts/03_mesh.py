#!/usr/bin/env python3
"""
Script: 03_mesh.py
Stage:  3 — Automated meshing (OpenFOAM-native: snappyHexMesh + extrudeMesh)

Pipeline per case:
    blockMesh         → 1-cell-thick 3D background slab
    surfaceFeatures   → extract feature edges from aerofoil.stl into .eMesh
    snappyHexMesh     → castellate + snap + add boundary layers (3D slab still)
    extrudeMesh       → rebuild as a clean 1-layer 2D-equivalent mesh
    createPatch       → merge patches into freestream + frontAndBack (empty)
    checkMesh         → validate quality

Geometry STL is produced by 02_geometry.py at constant/geometry/aerofoil.stl.

Usage:
    micromamba run -n openfoam python scripts/03_mesh.py
    micromamba run -n openfoam python scripts/03_mesh.py --case-id 0 1 2
    micromamba run -n openfoam python scripts/03_mesh.py --force
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import shlex
import shutil
import subprocess
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CASES_DIR = PROJECT_ROOT / "cases"
TEMPLATE_DIR = PROJECT_ROOT / "openfoam_template"
OPENFOAM_BASHRC = Path("/opt/openfoam12/etc/bashrc")

CHORD = 1.0
NU = 1.5e-5
RHO = 1.225

# Wall-resolved low-Re kOmegaSST. snappyHexMesh's layer-addition algorithm
# silently rejects extrusion when the absolute first-layer thickness drops
# below ~3e-5 m on a coarsely-refined outer band, so y+=1 (h_1 ≈ 1e-5 at our
# Re range) is unreachable here without surface refinement levels that blow
# the memory budget. y+ ≈ 5 sits at the edge of the viscous sublayer and is
# handled correctly by kLowReWallFunction / omegaWallFunction.
TARGET_Y_PLUS = 5.0
MESH_SPAN = 0.05
# Empirical lower bound for snappy layer extrusion: the internal threshold is
# ~3e-5 m on the bg mesh density used here; 5e-5 gives a safe margin.
H_MIN_SNAPPY = 5e-5

N_SURFACE_LAYERS = 10
EXPANSION_RATIO = 1.20
MIN_THICKNESS_FACTOR = 0.1

RELAXED_N_SURFACE_LAYERS = 6
RELAXED_EXPANSION_RATIO = 1.30
RELAXED_MIN_THICKNESS_FACTOR = 0.05

LAYER_COVERAGE_MIN_FRACTION = 0.5  # average layers must reach 50% of nSurfaceLayers

NON_ORTHOGONALITY_LIMIT = 70.0
SKEWNESS_LIMIT = 4.0
ASPECT_RATIO_LIMIT = 10000.0

EXPECTED_PATCHES = {"freestream", "aerofoil", "frontAndBack"}

MESH_TEMPLATES = (
    "system/blockMeshDict.template",
    "system/snappyHexMeshDict.template",
    "system/extrudeMeshDict.template",
)
MESH_STATIC_DICTS = (
    "system/surfaceFeaturesDict",
    "system/meshQualityDict",
    "system/createPatchDict",
)

JINJA_ENV = Environment(loader=FileSystemLoader(str(TEMPLATE_DIR)))

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate snappyHexMesh-based meshes per case using the 6-step "
            "OpenFOAM-native pipeline (blockMesh → surfaceFeatures → "
            "snappyHexMesh → extrudeMesh → createPatch → checkMesh)."
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

    missing_stl = [case_dir for case_dir in case_dirs if not (case_dir / "constant" / "geometry" / "aerofoil.stl").exists()]
    if missing_stl:
        names = ", ".join(case_dir.name for case_dir in missing_stl)
        raise FileNotFoundError(
            f"Missing constant/geometry/aerofoil.stl for: {names} — run 02_geometry.py first"
        )

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
    """Wall-normal first-cell height from a turbulent flat-plate y+ rule.

    Clamped to H_MIN_SNAPPY: at high Re the formula yields h_1 < 3e-5 m and
    snappy silently skips layer extrusion below that threshold.
    """
    cf = 0.026 / reynolds_number ** (1.0 / 7.0)
    u_inf = reynolds_number * nu / chord
    tau_w = 0.5 * RHO * u_inf ** 2 * cf
    u_tau = (tau_w / RHO) ** 0.5
    return max(y_plus * nu / u_tau, H_MIN_SNAPPY)


def reset_case_mesh(case_dir: Path) -> None:
    """Remove any pre-existing meshing artefacts so the run starts clean."""
    poly_mesh_dir = case_dir / "constant" / "polyMesh"
    if poly_mesh_dir.exists():
        shutil.rmtree(poly_mesh_dir)

    eMesh = case_dir / "constant" / "geometry" / "aerofoil.eMesh"
    if eMesh.exists():
        eMesh.unlink()

    cleanup_files = [
        "log.blockMesh",
        "log.surfaceFeatures",
        "log.snappyHexMesh",
        "log.extrudeMesh",
        "log.createPatch",
        "log.checkMesh",
        "system/blockMeshDict",
        "system/snappyHexMeshDict",
        "system/surfaceFeaturesDict",
        "system/meshQualityDict",
        "system/extrudeMeshDict",
        "system/createPatchDict",
    ]
    for relative in cleanup_files:
        path = case_dir / relative
        if path.exists():
            path.unlink()


def ensure_case_scaffold(case_dir: Path) -> None:
    (case_dir / "system").mkdir(parents=True, exist_ok=True)
    (case_dir / "constant" / "geometry").mkdir(parents=True, exist_ok=True)
    control_dict_path = case_dir / "system" / "controlDict"
    if not control_dict_path.exists():
        control_dict_path.write_text(MINIMAL_CONTROL_DICT)


def render_mesh_templates(case_dir: Path, context: dict[str, str]) -> None:
    """Render mesh-time .template files and copy static mesh-time dicts."""
    for relative in MESH_TEMPLATES:
        target_relative = relative.removesuffix(".template")
        target_path = case_dir / target_relative
        target_path.parent.mkdir(parents=True, exist_ok=True)
        rendered = JINJA_ENV.get_template(relative).render(**context)
        target_path.write_text(rendered.rstrip() + "\n")

    for relative in MESH_STATIC_DICTS:
        source_path = TEMPLATE_DIR / relative
        target_path = case_dir / relative
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target_path)


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
            log_tail = log_path.read_text(errors="ignore")[-1500:]
        raise RuntimeError(
            f"{command} failed in {case_dir.name}; inspect {log_name}\n{log_tail}"
        )


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
        "n_cells": extract_metric(
            r"\bcells:\s*([0-9]+)",
            text,
        ),
        "n_faces_frontAndBack": extract_metric(
            r"frontAndBack\s+(\d+)\s+\d+",
            text,
        ),
    }


def validate_mesh_quality(metrics: dict[str, float | None], case_dir: Path) -> None:
    non_orthogonality = metrics["max_non_orthogonality"]
    skewness = metrics["max_skewness"]
    aspect_ratio = metrics["max_aspect_ratio"]
    n_cells = metrics["n_cells"]
    n_faces_fab = metrics["n_faces_frontAndBack"]

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

    # 2D sanity: frontAndBack should carry exactly one face per cell on each side.
    if n_cells is not None and n_faces_fab is not None and n_faces_fab != 2 * n_cells:
        raise RuntimeError(
            f"{case_dir.name} mesh is not 2D: frontAndBack has {int(n_faces_fab)} "
            f"faces, expected {int(2 * n_cells)} (= 2 × cells)"
        )


def parse_layer_coverage(snappy_log: Path) -> float | None:
    """Pull average layer coverage on the aerofoil patch from the snappy log.

    The patch summary line is "aerofoil <faces> <avgLayers> ...". We take the
    last occurrence (the final summary printed after layer addition completes).
    Returns None when no summary is found.
    """
    if not snappy_log.exists():
        return None
    text = snappy_log.read_text(errors="ignore")
    matches = re.findall(r"^\s*aerofoil\s+\d+\s+([0-9.]+)", text, re.MULTILINE)
    if matches:
        return float(matches[-1])
    return None


def boundary_patch_names(case_dir: Path) -> set[str]:
    boundary_path = case_dir / "constant" / "polyMesh" / "boundary"
    if not boundary_path.exists():
        return set()
    names: set[str] = set()
    in_block = False
    depth = 0
    for raw_line in boundary_path.read_text().splitlines():
        line = raw_line.strip()
        if not in_block:
            if line.startswith("(") and not line.endswith(")"):
                in_block = True
                depth = 0
            continue
        if line == "{":
            depth += 1
            continue
        if line == "}":
            depth -= 1
            if depth < 0:
                break
            continue
        if depth == 0 and line and not line.startswith("//") and not line.startswith(")"):
            token = line.split()[0]
            if token.isidentifier():
                names.add(token)
    return names


def render_context(
    first_layer_thickness: float,
    n_surface_layers: int,
    expansion_ratio: float,
    min_thickness_factor: float,
) -> dict[str, str]:
    return {
        "FIRST_LAYER_THICKNESS": f"{first_layer_thickness:.10g}",
        "MIN_THICKNESS": f"{first_layer_thickness * min_thickness_factor:.10g}",
        "N_SURFACE_LAYERS": str(int(n_surface_layers)),
        "EXPANSION_RATIO": f"{expansion_ratio:.6g}",
        "MESH_SPAN": f"{MESH_SPAN:.6g}",
    }


def run_mesh_pipeline(case_dir: Path) -> None:
    run_openfoam_command(case_dir, "blockMesh", "log.blockMesh")
    run_openfoam_command(case_dir, "surfaceFeatures", "log.surfaceFeatures")
    run_openfoam_command(case_dir, "snappyHexMesh -overwrite", "log.snappyHexMesh")
    run_openfoam_command(case_dir, "extrudeMesh", "log.extrudeMesh")
    run_openfoam_command(case_dir, "createPatch -overwrite", "log.createPatch")
    run_openfoam_command(case_dir, "checkMesh", "log.checkMesh")


def attempt_build(
    case_dir: Path,
    first_layer_thickness: float,
    n_surface_layers: int,
    expansion_ratio: float,
    min_thickness_factor: float,
) -> dict[str, float]:
    reset_case_mesh(case_dir)
    ensure_case_scaffold(case_dir)
    context = render_context(
        first_layer_thickness=first_layer_thickness,
        n_surface_layers=n_surface_layers,
        expansion_ratio=expansion_ratio,
        min_thickness_factor=min_thickness_factor,
    )
    render_mesh_templates(case_dir, context)
    run_mesh_pipeline(case_dir)

    metrics = parse_check_mesh_log(case_dir / "log.checkMesh")
    validate_mesh_quality(metrics, case_dir)

    patches = boundary_patch_names(case_dir)
    missing = EXPECTED_PATCHES - patches
    extra = patches - EXPECTED_PATCHES
    if missing or extra:
        raise RuntimeError(
            f"{case_dir.name} polyMesh/boundary patch set is wrong; "
            f"missing={sorted(missing)} extra={sorted(extra)}"
        )

    layers_avg = parse_layer_coverage(case_dir / "log.snappyHexMesh")
    if layers_avg is None or layers_avg < LAYER_COVERAGE_MIN_FRACTION * n_surface_layers:
        raise RuntimeError(
            f"{case_dir.name} layer coverage too low: avg={layers_avg} "
            f"(need >= {LAYER_COVERAGE_MIN_FRACTION:.0%} of {n_surface_layers})"
        )

    return {
        "first_cell_height": first_layer_thickness,
        "n_surface_layers": float(n_surface_layers),
        "expansion_ratio": float(expansion_ratio),
        "cell_count": float(metrics["n_cells"] or 0.0),
        "max_non_orthogonality": float(metrics["max_non_orthogonality"] or 0.0),
        "max_skewness": float(metrics["max_skewness"] or 0.0),
        "max_aspect_ratio": float(metrics["max_aspect_ratio"] or -1.0),
        "layers_avg": float(layers_avg) if layers_avg is not None else -1.0,
    }


def build_mesh(case_dir: Path, params: dict[str, float]) -> dict[str, float]:
    first_layer = first_cell_height(params["Re"])

    try:
        return attempt_build(
            case_dir,
            first_layer_thickness=first_layer,
            n_surface_layers=N_SURFACE_LAYERS,
            expansion_ratio=EXPANSION_RATIO,
            min_thickness_factor=MIN_THICKNESS_FACTOR,
        )
    except RuntimeError as primary_exc:
        log.warning(
            "%s primary attempt failed (%s); retrying with relaxed BL params",
            case_dir.name,
            primary_exc,
        )
        return attempt_build(
            case_dir,
            first_layer_thickness=first_layer,
            n_surface_layers=RELAXED_N_SURFACE_LAYERS,
            expansion_ratio=RELAXED_EXPANSION_RATIO,
            min_thickness_factor=RELAXED_MIN_THICKNESS_FACTOR,
        )


def main() -> None:
    args = parse_args()

    if not CASES_DIR.exists():
        raise FileNotFoundError(f"{CASES_DIR} not found — run 02_geometry.py first")
    if not OPENFOAM_BASHRC.exists():
        raise FileNotFoundError(f"{OPENFOAM_BASHRC} not found")
    if not TEMPLATE_DIR.exists():
        raise FileNotFoundError(f"{TEMPLATE_DIR} not found")

    case_dirs = collect_case_dirs(args.case_id)
    if not case_dirs:
        log.warning("No case directories with params.json were found under %s", CASES_DIR)
        return

    failures: list[str] = []
    success_count = 0

    for case_dir in case_dirs:
        mesh_boundary_path = case_dir / "constant" / "polyMesh" / "boundary"
        if mesh_boundary_path.exists() and not args.force:
            log.info(
                "Skipping %s — mesh already exists (use --force to rebuild)",
                case_dir.name,
            )
            continue

        try:
            params = load_params(case_dir)
            metrics = build_mesh(case_dir, params)

            success_count += 1
            log.info(
                "%s  Re=%.3e  h1=%.3e m  cells=%d  layers≈%.2f  nonOrtho=%.2f  skew=%.3f",
                case_dir.name,
                params["Re"],
                metrics["first_cell_height"],
                int(metrics["cell_count"]),
                metrics["layers_avg"],
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


if __name__ == "__main__":
    main()
