#!/usr/bin/env python3
"""
Script: 08_validation.py
Stage:  8 — CFD model-setup validation against published reference data

Purpose
-------
Validate the OpenFOAM **model setup** (per-regime templates, mesh, boundary
conditions, turbulence model) against published reference Cl/Cd (and Cp where
available) drawn from `validation_data/regime_<X>/metadata.json`.

Scope
-----
This is *model-setup* validation, not per-CFD-run quality control:
    - No pass/fail tolerance gating
    - No markdown report
    - No interpolation: cases are run at the exact AoAs declared in metadata.json
    - Outputs are plots + a summary CSV (or a mesh-quality CSV in --mesh-only)

Usage
-----
    # Stage + compare (Regime A). Solver must already have run, or pass --run.
    uv run python scripts/08_validation.py --regime A

    # Stage + run + compare (single case at a time, decomposed across 4 cores)
    uv run python scripts/08_validation.py --regime A --run

    # Regenerate geometry + mesh only — no CFD, no reference comparison.
    # Emits validation/mesh_quality.csv with cells / non-ortho / skewness / y+ target.
    uv run python scripts/08_validation.py --regime A --mesh-only --force

    # Several regimes at once, each case solved on 8 cores via MPI decomposition
    uv run python scripts/08_validation.py --regime A B --run --jobs 8

    # Re-plot / re-summarize from existing cases (no staging, no run)
    uv run python scripts/08_validation.py --regime A --no-stage
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.validation.compare import (  # noqa: E402  (path insert above)
    build_summary_dataframe,
    compare_clcd,
)
from scripts.validation.generate_validation_cases import (  # noqa: E402
    load_pipeline,
    stage_case,
    validation_case_dir,
)
from scripts.validation.plots import (  # noqa: E402
    plot_cl_cd_comparison,
    plot_cp_comparison,
)

VALIDATION_DATA_DIR = PROJECT_ROOT / "validation_data"
VALIDATION_DIR = PROJECT_ROOT / "validation"
OPENFOAM_BASHRC = Path("/opt/openfoam12/etc/bashrc")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Single-script CFD model-setup validation. Runs at the exact AoAs "
            "declared in validation_data/regime_<X>/metadata.json."
        )
    )
    p.add_argument("--regime", nargs="+", default=["A"], choices=["A", "B", "C", "D"],
                   help="Regime(s) to validate. Default: A.")
    p.add_argument("--mesh-only", action="store_true",
                   help="Regenerate geometry + mesh and exit. Writes mesh_quality.csv; "
                        "skips CFD execution and reference comparison.")
    p.add_argument("--run", action="store_true",
                   help="Invoke foamRun on each staged case after staging.")
    p.add_argument("--jobs", type=int, default=4,
                   help="MPI subdomains per case (written to system/decomposeParDict). "
                        "Cases run one at a time on this many cores. Default 4.")
    p.add_argument("--no-stage", action="store_true",
                   help="Skip staging (assumes cases already exist on disk).")
    p.add_argument("--force", action="store_true",
                   help="Re-mesh and re-render even if outputs exist.")
    p.add_argument("--window", type=int, default=200,
                   help="Trailing iterations averaged for Cl/Cd. Default 200.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Metadata + case dir discovery
# ---------------------------------------------------------------------------

def load_metadata(regime: str) -> dict:
    path = VALIDATION_DATA_DIR / f"regime_{regime}" / "metadata.json"
    return json.loads(path.read_text(encoding="utf-8"))


def discover_case_dirs(regime: str) -> dict[str, Path]:
    metadata = load_metadata(regime)
    out: dict[str, Path] = {}
    for case_meta in metadata["validation_cases"]:
        d = validation_case_dir(regime, case_meta["case_id"])
        if d.exists():
            out[case_meta["case_id"]] = d
    return out


# ---------------------------------------------------------------------------
# Staging
# ---------------------------------------------------------------------------

def stage_regime(regime: str, force: bool, nprocs: int) -> tuple[dict[str, Path], list[dict]]:
    """Stage every case declared in metadata.json (exact AoA — no fuzzy match).

    Returns (case_dirs, mesh_records) where mesh_records summarises each case's
    mesh quality for the --mesh-only CSV output.
    """
    geometry = load_pipeline("naca_geometry", "02_geometry.py")
    mesh = load_pipeline("naca_mesh", "03_mesh.py")
    run_cfd = load_pipeline("naca_run_cfd", "04_run_cfd.py")

    metadata = load_metadata(regime)
    metadata_rel = (VALIDATION_DATA_DIR / f"regime_{regime}" / "metadata.json").relative_to(PROJECT_ROOT)

    case_dirs: dict[str, Path] = {}
    mesh_records: list[dict] = []

    for case_meta in metadata["validation_cases"]:
        case_meta["_regime"] = regime
        case_meta["_metadata_path"] = str(metadata_rel)

        case_dir = validation_case_dir(regime, case_meta["case_id"])
        log.info(
            "[%s] staging %s α=%.4f° Re=%.2e t=%.3f",
            regime, case_meta["case_id"], case_meta["alpha_deg"],
            case_meta["Re"], case_meta["thickness"],
        )
        stage_case(case_dir, case_meta, geometry, mesh, run_cfd,
                   force=force, skip_mesh=False, nprocs=nprocs)
        case_dirs[case_meta["case_id"]] = case_dir
        mesh_records.append(_collect_mesh_record(regime, case_meta, case_dir))

    return case_dirs, mesh_records


def _collect_mesh_record(regime: str, case_meta: dict, case_dir: Path) -> dict:
    """Pull mesh-quality fields out of case_metadata.json into a flat row."""
    meta_path = case_dir / "case_metadata.json"
    cmeta: dict = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    m = cmeta.get("mesh", {}) or {}
    return {
        "regime":                regime,
        "case_id":               case_meta["case_id"],
        "airfoil":               case_meta.get("airfoil", ""),
        "alpha_deg":             float(case_meta["alpha_deg"]),
        "Re":                    float(case_meta["Re"]),
        "thickness":             float(case_meta["thickness"]),
        "cells_total":           m.get("cells_total"),
        "non_orthogonality_max": m.get("non_orthogonality_max"),
        "skewness_max":          m.get("skewness_max"),
        "aspect_ratio_max":      m.get("aspect_ratio_max"),
        "first_cell_height":     m.get("first_cell_height"),
        "y_plus_target":         cmeta.get("y_plus_target"),
        "bl_layers":             m.get("bl_layers"),
        "bl_growth_ratio":       m.get("bl_growth_ratio"),
        "mesh_runtime_s":        cmeta.get("mesh_runtime_s"),
    }


# ---------------------------------------------------------------------------
# Solver execution
# ---------------------------------------------------------------------------

def run_solver(case_dirs: dict[str, Path], nprocs: int) -> None:
    if not case_dirs:
        log.warning("no cases to run")
        return

    nprocs = max(nprocs, 1)
    log.info("running foamRun on %d cases sequentially (np=%d per case)",
             len(case_dirs), nprocs)

    if nprocs > 1:
        inner = (
            f"decomposePar -force > log.decomposePar 2>&1 && "
            f"mpirun --oversubscribe --allow-run-as-root -np {nprocs} potentialFoam -initialiseUBCs -parallel "
            f"> log.potentialFoam 2>&1 && "
            f"mpirun --oversubscribe --allow-run-as-root -np {nprocs} foamRun -parallel "
            f"> log.foamRun 2>&1 && "
            f"reconstructPar -latestTime > log.reconstructPar 2>&1"
        )
    else:
        inner = (
            f"potentialFoam -initialiseUBCs > log.potentialFoam 2>&1 && "
            f"foamRun > log.foamRun 2>&1"
        )

    for cid, case_dir in case_dirs.items():
        log.info("  [%s] %s", cid, case_dir.relative_to(PROJECT_ROOT))
        cmd = f"source {OPENFOAM_BASHRC} && {inner}"
        subprocess.run(["bash", "-lc", cmd], cwd=case_dir, check=True)


# ---------------------------------------------------------------------------
# Comparison + plotting
# ---------------------------------------------------------------------------

def collect_comparisons(
    case_dirs: dict[str, Path], metadata: dict, window: int,
) -> list[dict]:
    """Compare each case's Cl/Cd against every reference declared in metadata.json."""
    out: list[dict] = []
    for case_meta in metadata["validation_cases"]:
        cid = case_meta["case_id"]
        case_dir = case_dirs.get(cid)
        if case_dir is None:
            log.warning("[%s] case dir missing — skipping", cid)
            continue
        try:
            out.append(compare_clcd(case_dir, case_meta, window=window))
        except FileNotFoundError as exc:
            log.warning("[%s] %s — solver may not have produced forceCoeffs yet", cid, exc)
    return out


def plot_regime(regime: str, case_dirs: dict[str, Path],
                comparisons: list[dict], metadata: dict) -> None:
    figures_dir = VALIDATION_DIR / f"regime_{regime}" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    plot_cl_cd_comparison(comparisons, figures_dir / "cl_cd_comparison.png")

    for case_meta in metadata["validation_cases"]:
        cid = case_meta["case_id"]
        case_dir = case_dirs.get(cid)
        if case_dir is None:
            continue
        plot_cp_comparison(case_dir, case_meta,
                           figures_dir / f"cp_comparison__{cid}.png")


# ---------------------------------------------------------------------------
# CSV writers
# ---------------------------------------------------------------------------

def write_mesh_quality_csv(records: list[dict]) -> None:
    if not records:
        log.warning("no mesh records to write")
        return
    out_csv = VALIDATION_DIR / "mesh_quality.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(out_csv, index=False)
    log.info("wrote %s (%d rows)", out_csv.relative_to(PROJECT_ROOT), len(records))


def write_validation_summary(frames: list[pd.DataFrame]) -> None:
    if not frames:
        return
    combined = pd.concat(frames, ignore_index=True)
    out_csv = VALIDATION_DIR / "validation_summary.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out_csv, index=False)
    log.info("wrote %s (%d rows)", out_csv.relative_to(PROJECT_ROOT), len(combined))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()
    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)

    if args.mesh_only and args.no_stage:
        log.warning("--mesh-only with --no-stage does nothing: staging is what produces meshes")

    overall_summary: list[pd.DataFrame] = []
    overall_mesh: list[dict] = []

    for regime in args.regime:
        log.info("==== Regime %s ====", regime)
        metadata = load_metadata(regime)

        if args.no_stage:
            case_dirs = discover_case_dirs(regime)
        else:
            case_dirs, mesh_records = stage_regime(regime, force=args.force,
                                                   nprocs=args.jobs)
            overall_mesh.extend(mesh_records)

        if args.mesh_only:
            continue   # skip CFD execution + reference comparison

        if args.run:
            run_solver(case_dirs, nprocs=args.jobs)

        comparisons = collect_comparisons(case_dirs, metadata, args.window)
        if not comparisons:
            log.warning("[Regime %s] no comparisons available — skipping plots", regime)
            continue

        plot_regime(regime, case_dirs, comparisons, metadata)

        df = build_summary_dataframe(comparisons)
        df.insert(0, "regime", regime)
        overall_summary.append(df)

    if args.mesh_only:
        write_mesh_quality_csv(overall_mesh)
    else:
        write_validation_summary(overall_summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
