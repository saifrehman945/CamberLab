#!/usr/bin/env python3
"""
Script: scripts/validation/generate_validation_cases.py
Purpose: Materialize the OpenFOAM cases declared in validation_data/regime_<X>/metadata.json
         under cases/validation_<regime>_<case_id>/ using the existing pipeline
         scripts (02_geometry, 03_mesh, 04_run_cfd).

This script does NOT run foamRun — it only stages the cases. Run them with:

    parallel -j 4 \
      "cd {1} && source /opt/openfoam12/etc/bashrc && foamRun > log.foamRun 2>&1" \
      ::: cases/validation_*/

or via 08_validate_regimes.py which orchestrates both stages.

Usage:
    micromamba run -n openfoam python scripts/validation/generate_validation_cases.py --regime A
    micromamba run -n openfoam python scripts/validation/generate_validation_cases.py --regime A B
    micromamba run -n openfoam python scripts/validation/generate_validation_cases.py --regime A --force
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import sys
from pathlib import Path
from types import ModuleType

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
CASES_DIR = PROJECT_ROOT / "cases"
VALIDATION_DATA_DIR = PROJECT_ROOT / "validation_data"
VALIDATION_DIR = PROJECT_ROOT / "validation"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage OpenFOAM validation cases per regime.")
    p.add_argument("--regime", nargs="+", default=["A"], choices=["A", "B", "C", "D"],
                   help="Which regime(s) to stage. Default: A.")
    p.add_argument("--force", action="store_true",
                   help="Rebuild mesh and re-render even if outputs already exist.")
    p.add_argument("--skip-mesh", action="store_true",
                   help="Stage only geometry and OpenFOAM files; do not call gmsh.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Module loaders (mirrors validation.py's pattern of importing 0X_*.py scripts)
# ---------------------------------------------------------------------------

def load_pipeline(name: str, relpath: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, SCRIPTS_DIR / relpath)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load {relpath}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Case directory naming
# ---------------------------------------------------------------------------

def validation_case_dir(regime: str, case_id: str) -> Path:
    return CASES_DIR / f"validation_{regime}_{case_id}"


# ---------------------------------------------------------------------------
# Case staging
# ---------------------------------------------------------------------------

def stage_case(
    case_dir: Path,
    case_meta: dict,
    geometry: ModuleType,
    mesh: ModuleType | None,
    run_cfd: ModuleType,
    force: bool,
    skip_mesh: bool,
) -> dict:
    """Stage a single validation case using the existing pipeline modules.

    The current single-template pipeline is used. Once per-regime templates
    exist under templates/regime_<X>/, swap the render call below.
    """
    case_dir.mkdir(parents=True, exist_ok=True)
    alpha_deg = float(case_meta["alpha_deg"])
    Re = float(case_meta["Re"])
    thickness = float(case_meta["thickness"])

    # 1) Geometry
    coords = geometry.aerofoil_polygon(thickness, geometry.N_POINTS)
    geometry.write_aerofoil_dat(case_dir / "aerofoil.dat", coords)
    geometry.write_params_json(
        case_dir / "params.json",
        alpha_deg=alpha_deg,
        Re=Re,
        thickness=thickness,
    )
    (case_dir / "validation_meta.json").write_text(
        json.dumps({
            "regime":         case_meta.get("_regime"),
            "case_id":        case_meta["case_id"],
            "airfoil":        case_meta["airfoil"],
            "Mach":           case_meta.get("Mach"),
            "primary_role":   case_meta.get("primary_role"),
            "metadata_path":  str(case_meta.get("_metadata_path", "")),
        }, indent=2) + "\n",
        encoding="utf-8",
    )

    # 2) Mesh
    if not skip_mesh:
        boundary_path = case_dir / "constant" / "polyMesh" / "boundary"
        if boundary_path.exists() and not force:
            log.info("[%s] mesh present — skipping", case_dir.name)
        else:
            if force:
                mesh.reset_case_mesh(case_dir)
            gmsh_module = mesh.require_gmsh()
            gmsh_module.initialize()
            try:
                params = mesh.load_params(case_dir)
                metrics = mesh.build_mesh(
                    case_dir / "aerofoil.dat",
                    params["Re"],
                    case_dir,
                    chord=mesh.CHORD,
                    target_y_plus=mesh.TARGET_Y_PLUS,
                )
                log.info(
                    "[%s] meshed cells=%d nonOrtho=%.2f skew=%.3f",
                    case_dir.name,
                    int(metrics["cell_count"]),
                    metrics["max_non_orthogonality"],
                    metrics["max_skewness"],
                )
            finally:
                gmsh_module.finalize()

    # 3) Render OpenFOAM templates (current single-template path)
    run_cfd.render_case(case_dir)

    return {"case_id": case_meta["case_id"], "case_dir": str(case_dir)}


def main() -> int:
    args = parse_args()

    geometry = load_pipeline("naca_geometry", "02_geometry.py")
    mesh = None if args.skip_mesh else load_pipeline("naca_mesh", "03_mesh.py")
    run_cfd = load_pipeline("naca_run_cfd", "04_run_cfd.py")

    staged: list[dict] = []
    for regime in args.regime:
        meta_path = VALIDATION_DATA_DIR / f"regime_{regime}" / "metadata.json"
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        for case_meta in metadata["validation_cases"]:
            case_meta["_regime"] = regime
            case_meta["_metadata_path"] = str(meta_path.relative_to(PROJECT_ROOT))
            case_dir = validation_case_dir(regime, case_meta["case_id"])
            log.info(
                "staging regime=%s case=%s α=%.2f° Re=%.2e t=%.3f → %s",
                regime, case_meta["case_id"], case_meta["alpha_deg"],
                case_meta["Re"], case_meta["thickness"],
                case_dir.relative_to(PROJECT_ROOT),
            )
            staged.append(stage_case(case_dir, case_meta, geometry, mesh, run_cfd,
                                     force=args.force, skip_mesh=args.skip_mesh))

    log.info("staged %d validation cases across regimes: %s",
             len(staged), ", ".join(args.regime))
    return 0


if __name__ == "__main__":
    sys.exit(main())
