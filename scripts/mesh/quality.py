"""
Quality validation, OpenFOAM glue, and per-case metadata writing.

Most of the helpers below are ports of the original 03_mesh.py functions
(parse_check_mesh_log, validate_mesh_quality, rewrite_boundary_types,
run_openfoam_command). They have been generalised to accept a regime cfg
dict instead of hard-coded module constants.

write_metadata writes case_metadata.json in the schema expected by CLAUDE.md
§12; downstream stages (07_harvest_results) patch in the convergence /
force-coefficient fields after the CFD run.
"""

from __future__ import annotations

import json
import logging
import re
import shlex
import subprocess
from pathlib import Path

log = logging.getLogger(__name__)


OPENFOAM_BASHRC = Path("/opt/openfoam12/etc/bashrc")


# ---------------------------------------------------------------------------
# checkMesh log parsing
# ---------------------------------------------------------------------------

_METRIC_PATTERNS = {
    "max_non_orthogonality": r"Mesh non-orthogonality Max:\s*([0-9.eE+-]+)",
    "max_skewness":          r"Max skewness =\s*([0-9.eE+-]+)",
    "max_aspect_ratio":      r"Max aspect ratio[:=]\s*([0-9.eE+-]+)",
    "cells_total":           r"cells:\s*([0-9]+)",
    "points_total":          r"points:\s*([0-9]+)",
    "faces_total":           r"faces:\s*([0-9]+)",
    "hexahedra":             r"hexahedra:\s*([0-9]+)",
    "prisms":                r"prisms:\s*([0-9]+)",
    "wedges":                r"wedges:\s*([0-9]+)",
    "pyramids":              r"pyramids:\s*([0-9]+)",
    "tet_wedges":            r"tet wedges:\s*([0-9]+)",
    "tetrahedra":            r"tetrahedra:\s*([0-9]+)",
    "polyhedra":             r"polyhedra:\s*([0-9]+)",
}


def _extract(pattern: str, text: str) -> float | None:
    m = re.search(pattern, text, re.MULTILINE)
    if not m:
        return None
    return float(m.group(1))


def parse_check_mesh(log_path: Path) -> dict[str, float | None]:
    """Parse the relevant metrics out of a checkMesh log."""
    text = log_path.read_text(errors="ignore")
    metrics: dict[str, float | None] = {
        name: _extract(pattern, text) for name, pattern in _METRIC_PATTERNS.items()
    }
    # Derived: hex fraction (1.0 means an all-hex mesh).
    cells = metrics.get("cells_total") or 0.0
    hexes = metrics.get("hexahedra") or 0.0
    metrics["hex_fraction"] = float(hexes / cells) if cells > 0 else None
    return metrics


# ---------------------------------------------------------------------------
# Quality gates
# ---------------------------------------------------------------------------

def validate_quality(
    metrics: dict[str, float | None],
    cfg: dict,
    case_dir: Path,
) -> None:
    """Raise if any blocking quality gate is violated; warn for advisories."""
    non_ortho = metrics.get("max_non_orthogonality")
    skew      = metrics.get("max_skewness")
    aspect    = metrics.get("max_aspect_ratio")
    cells     = metrics.get("cells_total")
    hex_frac  = metrics.get("hex_fraction")

    if non_ortho is None:
        raise RuntimeError(f"Could not parse non-orthogonality from {case_dir/'log.checkMesh'}")
    if skew is None:
        raise RuntimeError(f"Could not parse skewness from {case_dir/'log.checkMesh'}")

    if non_ortho >= cfg["non_orthogonality_max"]:
        raise RuntimeError(
            f"{case_dir.name}: max non-orthogonality {non_ortho:.3f} "
            f">= limit {cfg['non_orthogonality_max']:.1f}"
        )
    if skew >= cfg["skewness_max"]:
        raise RuntimeError(
            f"{case_dir.name}: max skewness {skew:.3f} "
            f">= limit {cfg['skewness_max']:.1f}"
        )
    if aspect is not None and aspect >= cfg["aspect_ratio_max"]:
        raise RuntimeError(
            f"{case_dir.name}: max aspect ratio {aspect:.3f} "
            f">= limit {cfg['aspect_ratio_max']:.1f}"
        )
    if hex_frac is not None and hex_frac < cfg.get("min_hex_fraction", 0.999):
        raise RuntimeError(
            f"{case_dir.name}: hex fraction {hex_frac:.4f} below required "
            f"{cfg.get('min_hex_fraction', 0.999):.4f} — transfinite blocks did "
            f"not all recombine to hexes."
        )

    if cells is not None:
        lo, hi = cfg.get("target_cells_min"), cfg.get("target_cells_max")
        if lo is not None and cells < lo:
            log.warning(
                "%s: cell count %d below advisory minimum %d",
                case_dir.name, int(cells), int(lo),
            )
        if hi is not None and cells > hi:
            log.warning(
                "%s: cell count %d above advisory maximum %d",
                case_dir.name, int(cells), int(hi),
            )


# ---------------------------------------------------------------------------
# OpenFOAM glue
# ---------------------------------------------------------------------------

def run_openfoam_command(case_dir: Path, command: str, log_name: str) -> None:
    """Run an OpenFOAM command in case_dir with the OpenFOAM 12 environment sourced."""
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
        tail = ""
        if log_path.exists():
            tail = log_path.read_text(errors="ignore")[-1200:]
        raise RuntimeError(
            f"{command} failed in {case_dir.name}; inspect {log_name}\n{tail}"
        )


_PATCH_TYPES = {
    "freestream":   "patch",
    "aerofoil":     "wall",
    "frontAndBack": "empty",
}


def rewrite_boundary_types(boundary_path: Path) -> None:
    """Rewrite the patch types in constant/polyMesh/boundary after gmshToFoam.

    gmshToFoam emits every physical-surface group as type `patch`; OpenFOAM
    needs `wall` on the aerofoil and `empty` on the front/back side faces for
    the quasi-2D solve to behave correctly.
    """
    lines = boundary_path.read_text().splitlines()
    current_patch: str | None = None

    for index, line in enumerate(lines):
        stripped = line.strip()

        if stripped in _PATCH_TYPES:
            current_patch = stripped
            continue
        if current_patch is None:
            continue

        if stripped.startswith("type"):
            indent = line[: len(line) - len(line.lstrip())]
            lines[index] = f"{indent}type            {_PATCH_TYPES[current_patch]};"
            continue
        if stripped.startswith("physicalType"):
            indent = line[: len(line) - len(line.lstrip())]
            lines[index] = f"{indent}physicalType    {_PATCH_TYPES[current_patch]};"
            continue
        if stripped == "}":
            current_patch = None

    boundary_path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Metadata writer
# ---------------------------------------------------------------------------

def write_metadata(
    case_dir: Path,
    regime: str,
    cfg: dict,
    params: dict,
    metrics: dict,
) -> None:
    """Write case_metadata.json in the schema from CLAUDE.md §12.

    Fields filled later by downstream stages (CFD + harvest) are set to None.
    """
    cells_total = metrics.get("cells_total")
    payload = {
        "case_id":   case_dir.name,
        "regime":    regime,

        "alpha_deg": float(params["alpha_deg"]),
        "Re":        float(params["Re"]),
        "thickness": float(params["thickness"]),

        "template":         f"regime_{regime}",
        "turbulence_model": metrics.get("turbulence_model", "SpalartAllmaras"),
        "wall_treatment":   cfg.get("wall_treatment"),

        "y_plus_target":   cfg.get("y_plus_target"),
        "y_plus_min":      None,
        "y_plus_mean":     None,
        "y_plus_max":      None,

        "mesh": {
            "topology":              cfg.get("topology"),
            "cells_total":           int(cells_total) if cells_total else None,
            "hex_fraction":          metrics.get("hex_fraction"),
            "non_orthogonality_max": metrics.get("max_non_orthogonality"),
            "skewness_max":          metrics.get("max_skewness"),
            "aspect_ratio_max":      metrics.get("max_aspect_ratio"),
            "first_cell_height":     metrics.get("first_cell_height"),
            "bl_layers":             cfg.get("bl_layers"),
            "bl_growth_ratio":       cfg.get("bl_growth_ratio"),
            "normal_pts":            cfg.get("normal_pts"),
            "chord_pts_upper":       cfg.get("chord_pts_upper"),
            "chord_pts_lower":       cfg.get("chord_pts_lower"),
            "wake_pts":              cfg.get("wake_pts"),
            "wake_progression":      cfg.get("wake_progression"),
            "le_te_cluster":         cfg.get("le_te_cluster"),
            "upstream_radius":       cfg.get("upstream_radius"),
            "downstream_length":     cfg.get("downstream_length"),
            "transverse_extent":     cfg.get("transverse_extent"),
            "spanwise_thickness":    cfg.get("spanwise_thickness"),
            "spanwise_layers":       cfg.get("spanwise_layers"),
        },

        "mesh_runtime_s":  metrics.get("mesh_runtime_s"),
        "iterations":      None,
        "runtime_s":       None,
        "residuals_final": None,

        "Cl_mean": None,
        "Cl_std":  None,
        "Cd_mean": None,
        "Cd_std":  None,
        "L_over_D": None,

        "converged": None,
        "ood":       False,
        "notes":     "",
    }
    (case_dir / "case_metadata.json").write_text(json.dumps(payload, indent=2) + "\n")
