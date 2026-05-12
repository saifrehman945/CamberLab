#!/usr/bin/env python3
"""
Script: 08_validate_regimes.py
Stage:  8 — Per-regime CFD validation against authoritative reference data
Purpose: End-to-end runner for the validation workflow. For each requested
         regime:
           1. Stage the OpenFOAM cases declared in validation_data/regime_<X>/metadata.json
           2. (Optionally) launch foamRun via GNU parallel
           3. Parse forceCoeffs, compute Cl/Cd/Cp errors against the reference data
           4. Apply per-regime pass/fail tolerances
           5. Generate the per-regime figure suite (Cl/Cd bars, Cp overlay,
              residual histories, optional mesh convergence)
           6. Write validation/regime_<X>/report.md and
              validation/validation_summary.csv

This script is the single command Phase-1 contributors run to lock a regime.

Usage:
    micromamba run -n openfoam python scripts/08_validate_regimes.py --regime A
    micromamba run -n openfoam python scripts/08_validate_regimes.py --regime A --run --jobs 4
    micromamba run -n openfoam python scripts/08_validate_regimes.py --regime A B --no-stage
    micromamba run -n openfoam python scripts/08_validate_regimes.py --regime A --report-only
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shlex
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.validation.compare import (  # noqa: E402  (path insert above)
    build_summary_dataframe,
    compare_clcd,
    evaluate_tolerances,
)
from scripts.validation.generate_validation_cases import (  # noqa: E402
    validation_case_dir,
    load_pipeline,
    stage_case,
)
from scripts.validation.plots import generate_regime_plots  # noqa: E402
from scripts.validation.report import render_regime_report   # noqa: E402

VALIDATION_DATA_DIR = PROJECT_ROOT / "validation_data"
VALIDATION_DIR = PROJECT_ROOT / "validation"
OPENFOAM_BASHRC = Path("/opt/openfoam12/etc/bashrc")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="End-to-end per-regime CFD validation orchestrator."
    )
    p.add_argument("--regime", nargs="+", default=["A"], choices=["A", "B", "C", "D"],
                   help="Regime(s) to validate. Default: A.")
    p.add_argument("--run", action="store_true",
                   help="Invoke foamRun on each staged case after staging.")
    p.add_argument("--jobs", type=int, default=4,
                   help="GNU parallel concurrency for --run. Default 4.")
    p.add_argument("--no-stage", action="store_true",
                   help="Skip staging (assumes cases already exist).")
    p.add_argument("--skip-mesh", action="store_true",
                   help="Stage geometry and OpenFOAM files only; do not mesh.")
    p.add_argument("--force", action="store_true",
                   help="Re-mesh and re-render even if outputs exist.")
    p.add_argument("--report-only", action="store_true",
                   help="Skip staging and execution; only parse and plot.")
    p.add_argument("--window", type=int, default=200,
                   help="Trailing iterations averaged for Cl/Cd. Default 200.")
    return p.parse_args()


def load_metadata(regime: str) -> dict:
    path = VALIDATION_DATA_DIR / f"regime_{regime.upper()}" / "metadata.json"
    return json.loads(path.read_text(encoding="utf-8"))


def stage_regime(regime: str, force: bool, skip_mesh: bool) -> dict[str, Path]:
    """Stage all validation cases for one regime; return mapping case_id → case_dir."""
    geometry = load_pipeline("naca_geometry", "02_geometry.py")
    mesh = None if skip_mesh else load_pipeline("naca_mesh", "03_mesh.py")
    run_cfd = load_pipeline("naca_run_cfd", "04_run_cfd.py")

    metadata = load_metadata(regime)
    case_dirs: dict[str, Path] = {}
    for case_meta in metadata["validation_cases"]:
        # XFOIL-only Regime C reference cases are not run in OpenFOAM unless
        # the user provides an XFOIL reference and explicitly asks for CFD;
        # for now, skip cases whose airfoil is not symmetric NACA (e.g. NACA4412)
        # only at the user's request.
        case_meta["_regime"] = regime
        case_dir = validation_case_dir(regime, case_meta["case_id"])
        log.info(
            "[%s] staging %s α=%.2f° Re=%.2e t=%.3f",
            regime, case_meta["case_id"], case_meta["alpha_deg"],
            case_meta["Re"], case_meta["thickness"],
        )
        stage_case(case_dir, case_meta, geometry, mesh, run_cfd,
                   force=force, skip_mesh=skip_mesh)
        case_dirs[case_meta["case_id"]] = case_dir
    return case_dirs


def discover_case_dirs(regime: str) -> dict[str, Path]:
    metadata = load_metadata(regime)
    out: dict[str, Path] = {}
    for case_meta in metadata["validation_cases"]:
        d = validation_case_dir(regime, case_meta["case_id"])
        if d.exists():
            out[case_meta["case_id"]] = d
    return out


def run_solver(case_dirs: dict[str, Path], jobs: int) -> None:
    if not case_dirs:
        log.warning("no cases to run")
        return
    case_list = " ".join(shlex.quote(str(d)) for d in case_dirs.values())
    cmd = (
        f"parallel -j {jobs} "
        f'"cd {{1}} && source {OPENFOAM_BASHRC} && foamRun > log.foamRun 2>&1" '
        f"::: {case_list}"
    )
    # GNU parallel defaults to /bin/sh (dash on Ubuntu) for each job, which
    # has no `source` builtin and chokes on OpenFOAM's bash-only bashrc.
    # Force bash for the per-job shell.
    env = {**os.environ, "PARALLEL_SHELL": "/bin/bash"}
    log.info("running foamRun on %d cases (parallel -j %d)", len(case_dirs), jobs)
    subprocess.run(cmd, shell=True, check=True, env=env)


def collect_comparisons(
    case_dirs: dict[str, Path],
    metadata: dict,
    window: int,
) -> tuple[list[dict], list[dict]]:
    comparisons: list[dict] = []
    tolerance_results: list[dict] = []
    for case_meta in metadata["validation_cases"]:
        cid = case_meta["case_id"]
        case_dir = case_dirs.get(cid)
        if case_dir is None:
            log.warning("[%s] case dir missing — skipping", cid)
            continue
        try:
            comp = compare_clcd(case_dir, case_meta, window=window)
        except FileNotFoundError as exc:
            log.warning("[%s] missing forceCoeffs (%s) — solver may not have run", cid, exc)
            continue
        comparisons.append(comp)
        tolerance_results.append(evaluate_tolerances(comp, metadata.get("tolerances", {})))
    return comparisons, tolerance_results


def main() -> int:
    args = parse_args()
    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)

    overall_summary: list[dict] = []

    for regime in args.regime:
        log.info("==== Regime %s ====", regime)
        metadata = load_metadata(regime)

        if not args.no_stage and not args.report_only:
            case_dirs = stage_regime(regime, force=args.force, skip_mesh=args.skip_mesh)
        else:
            case_dirs = discover_case_dirs(regime)

        if args.run and not args.report_only:
            run_solver(case_dirs, jobs=args.jobs)

        comparisons, tolerance_results = collect_comparisons(case_dirs, metadata, args.window)
        if not comparisons:
            log.warning("[Regime %s] no comparisons available — skipping plots/report", regime)
            continue

        figures = generate_regime_plots(
            regime=regime,
            case_dirs=case_dirs,
            comparisons=comparisons,
            metadata=metadata,
            mesh_records=None,
        )

        report_path = VALIDATION_DIR / f"regime_{regime}" / "report.md"
        render_regime_report(
            regime=regime,
            metadata=metadata,
            comparisons=comparisons,
            tolerance_results=tolerance_results,
            figures=figures,
            output_path=report_path,
        )

        # accumulate into the global summary CSV
        df = build_summary_dataframe(comparisons)
        df["regime"] = regime
        for comp, tres in zip(comparisons, tolerance_results):
            cid = comp["case_id"]
            df.loc[df["case_id"] == cid, "tolerance_pass"] = tres["pass"]
        overall_summary.append(df)

    if overall_summary:
        import pandas as pd
        combined = pd.concat(overall_summary, ignore_index=True)
        out_csv = VALIDATION_DIR / "validation_summary.csv"
        combined.to_csv(out_csv, index=False)
        log.info("wrote %s (%d rows)", out_csv, len(combined))
    return 0


if __name__ == "__main__":
    sys.exit(main())
