#!/usr/bin/env python3
"""
Script: 03_mesh.py
Stage:  3 — Automated meshing (regime-aware structured C+H mesh)
Purpose: Dispatch on each case's regime, build the corresponding structured
         transfinite mesh via the `mesh` package, convert it to OpenFOAM
         format with gmshToFoam, validate quality with checkMesh, and write
         case_metadata.json.

Only Regime A is implemented in this overhaul; cases tagged with a different
regime are skipped with a warning.

Usage:
    uv run python scripts/03_mesh.py
    uv run python scripts/03_mesh.py --case-id 0 1 2
    uv run python scripts/03_mesh.py --force
    uv run python scripts/03_mesh.py --regime A
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from mesh import (                                               # noqa: E402
    REGIME_MESH,
    classify_regime,
    build_c_grid,
    parse_check_mesh,
    validate_quality,
    rewrite_boundary_types,
    run_openfoam_command,
    write_metadata,
)

try:
    import gmsh                                                  # noqa: F401, E402
except ModuleNotFoundError:
    gmsh = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(message)s")
log = logging.getLogger(__name__)

CASES_DIR = PROJECT_ROOT / "cases"
OPENFOAM_BASHRC = Path("/opt/openfoam12/etc/bashrc")

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build regime-aware structured meshes for one or more cases."
    )
    parser.add_argument(
        "--case-id",
        type=int,
        nargs="*",
        help="Specific case IDs to process. Defaults to every case with params.json.",
    )
    parser.add_argument(
        "--case-dir",
        type=Path,
        nargs="+",
        metavar="PATH",
        help="Arbitrary case directory paths (e.g. validation_* dirs). "
             "Bypasses the case_XXXX naming convention.",
    )
    parser.add_argument(
        "--regime",
        type=str,
        default=None,
        help="Only mesh cases whose params.json regime matches this letter.",
    )
    parser.add_argument(
        "--regime-override",
        type=str,
        default=None,
        metavar="LETTER",
        help="Force a specific regime for all targeted cases, ignoring params.json "
             "classification. Useful when params.json has no regime key.",
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
            "The gmsh Python API is not available. Install the project "
            "environment first: uv venv && uv pip install -r requirements.txt"
        )
    return gmsh


def collect_case_dirs(
    case_ids: list[int] | None,
    case_dirs_explicit: list[Path] | None = None,
) -> list[Path]:
    if case_dirs_explicit:
        case_dirs = [Path(p).resolve() for p in case_dirs_explicit]
    elif case_ids:
        case_dirs = [CASES_DIR / f"case_{cid:04d}" for cid in case_ids]
    else:
        case_dirs = sorted(
            d for d in CASES_DIR.glob("case_*") if (d / "params.json").exists()
        )
    missing = [d for d in case_dirs if not (d / "params.json").exists()]
    if missing:
        names = ", ".join(d.name for d in missing)
        raise FileNotFoundError(f"Missing params.json for: {names}")
    return case_dirs


def load_params(case_dir: Path) -> dict:
    payload = json.loads((case_dir / "params.json").read_text())
    return {
        "alpha_deg": float(payload["alpha_deg"]),
        "Re":        float(payload["Re"]),
        "thickness": float(payload["thickness"]),
        "regime":    payload.get("regime"),
    }


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


def mesh_one_case(case_dir: Path, regime_override: str | None = None) -> dict:
    params = load_params(case_dir)
    regime = regime_override or params["regime"] or classify_regime(
        params["alpha_deg"], params["Re"], params["thickness"]
    )
    cfg = REGIME_MESH.get(regime)
    if cfg is None:
        raise NotImplementedError(
            f"{case_dir.name}: regime '{regime}' is not implemented yet. "
            f"Only Regime A is supported."
        )

    reset_case_mesh(case_dir)
    ensure_case_scaffold(case_dir)

    mesh_metrics = build_c_grid(case_dir, params=params, cfg=cfg)

    run_openfoam_command(case_dir, "gmshToFoam mesh.msh", "log.gmshToFoam")
    rewrite_boundary_types(case_dir / "constant" / "polyMesh" / "boundary")
    run_openfoam_command(case_dir, "checkMesh", "log.checkMesh")

    check_metrics = parse_check_mesh(case_dir / "log.checkMesh")
    metrics = {**mesh_metrics, **check_metrics}

    validate_quality(metrics, cfg, case_dir)

    write_metadata(case_dir, regime=regime, cfg=cfg, params=params, metrics=metrics)
    return metrics


def main() -> None:
    args = parse_args()

    if not getattr(args, "case_dir", None) and not CASES_DIR.exists():
        raise FileNotFoundError(f"{CASES_DIR} not found — run 02_geometry.py first")
    if not OPENFOAM_BASHRC.exists():
        raise FileNotFoundError(f"{OPENFOAM_BASHRC} not found")

    gmsh_module = require_gmsh()
    case_dirs = collect_case_dirs(args.case_id, getattr(args, "case_dir", None))
    if not case_dirs:
        log.warning("No case directories with params.json were found under %s", CASES_DIR)
        return

    regime_override: str | None = getattr(args, "regime_override", None)

    gmsh_module.initialize()
    failures: list[str] = []
    skipped: list[str] = []
    success_count = 0

    try:
        for case_dir in case_dirs:
            mesh_boundary = case_dir / "constant" / "polyMesh" / "boundary"
            if mesh_boundary.exists() and not args.force:
                log.info("Skipping %s — mesh already exists (use --force to rebuild)",
                         case_dir.name)
                continue

            params = load_params(case_dir)
            regime = regime_override or params["regime"] or classify_regime(
                params["alpha_deg"], params["Re"], params["thickness"]
            )
            if args.regime and regime != args.regime:
                skipped.append(f"{case_dir.name} (regime {regime} != requested {args.regime})")
                continue
            if REGIME_MESH.get(regime) is None:
                skipped.append(f"{case_dir.name} (regime {regime} not implemented)")
                continue

            try:
                metrics = mesh_one_case(case_dir, regime_override=regime_override)
                success_count += 1
                log.info(
                    "%s  regime=%s  Re=%.3e  h1=%.3e m  cells=%d  nonOrtho=%.2f  "
                    "skew=%.3f  hex=%.4f",
                    case_dir.name,
                    regime,
                    params["Re"],
                    metrics["first_cell_height"],
                    int(metrics.get("cells_total") or 0),
                    metrics.get("max_non_orthogonality") or float("nan"),
                    metrics.get("max_skewness") or float("nan"),
                    metrics.get("hex_fraction") or float("nan"),
                )
            except Exception as exc:                                # noqa: BLE001
                failures.append(f"{case_dir.name}: {exc}")
                log.error("Meshing failed for %s: %s", case_dir.name, exc)

        if skipped:
            log.info("Skipped %d case(s) due to regime filter / unimplemented regime:",
                     len(skipped))
            for note in skipped:
                log.info("  - %s", note)
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
