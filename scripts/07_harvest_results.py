#!/usr/bin/env python3
"""
Script: 07_harvest_results.py
Stage:  7 — Harvest CFD results
Purpose: Parse postProcessing/ output for every meshed case, apply the
         regime-aware convergence gate (CLAUDE.md §8), fill the CFD-result
         fields of each case's case_metadata.json in place, and assemble
         results/dataset_clean.csv from every case that converges.

This script is read-only with respect to CFD case directories except for
rewriting each case's own case_metadata.json in place (its documented job
per CLAUDE.md §12). No mesh, field, log, or postProcessing file is modified.

Usage:
    micromamba run -n openfoam python scripts/07_harvest_results.py
"""

from __future__ import annotations

import json
import logging
import re
import sys
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.validation.parsers import load_force_coeffs, load_residual_history  # noqa: E402
from scripts.surrogate.data import UNVALIDATED_REGIMES  # noqa: E402

CASES_DIR = PROJECT_ROOT / "cases"
RESULTS_DIR = PROJECT_ROOT / "results"

# CLAUDE.md §8 — regime-aware convergence gate.
MIN_ROWS = {"A": 800, "B": 1500, "C": 1500, "D": 800}
CL_TOL = {"A": 0.005, "B": 0.010, "C": 0.005, "D": 0.005}
CD_TOL = {"A": 0.005, "B": 0.010, "C": 0.005, "D": 0.005}
Y_PLUS = {"A": 30.0, "B": 0.5, "C": 0.5, "D": 50.0}

DATASET_COLUMNS = [
    "case_id", "regime", "alpha_deg", "Re", "thickness",
    "Cl_mean", "Cl_std", "Cd_mean", "Cd_std", "L_over_D",
    "y_plus_mean", "iterations", "runtime_s", "validated",
]

_EXEC_TIME_RE = re.compile(r"ExecutionTime\s*=\s*([\d.eE+-]+)\s*s")


# ---------------------------------------------------------------------------
# Per-case parsing
# ---------------------------------------------------------------------------

def discover_case_dirs() -> list[Path]:
    return sorted(d for d in CASES_DIR.glob("case_*") if (d / "case_metadata.json").exists())


def find_solver_log(case_dir: Path) -> Path | None:
    for name in ("log.simpleFoam", "log.foamRun"):
        p = case_dir / name
        if p.exists():
            return p
    return None


def has_cfd_run(case_dir: Path) -> bool:
    return find_solver_log(case_dir) is not None or (
        case_dir / "postProcessing" / "forceCoeffs" / "0"
    ).exists()


def has_fatal_error(log_path: Path) -> bool:
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    return "FOAM FATAL ERROR" in text


def parse_y_plus(case_dir: Path) -> dict | None:
    """Parse the last row for the aerofoil patch from postProcessing/yPlus/0/yPlus.dat."""
    path = case_dir / "postProcessing" / "yPlus" / "0" / "yPlus.dat"
    if not path.exists():
        return None
    last = None
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        tokens = s.split()
        if len(tokens) < 5 or tokens[1] != "aerofoil":
            continue
        try:
            last = {
                "y_plus_min": float(tokens[2]),
                "y_plus_max": float(tokens[3]),
                "y_plus_mean": float(tokens[4]),
            }
        except ValueError:
            continue
    return last


def parse_runtime_and_iterations(
    coeff_result: dict | None, log_path: Path | None
) -> tuple[float | None, int | None]:
    iterations = coeff_result["n_iters"] if coeff_result else None
    runtime_s = None
    if log_path is not None:
        matches = _EXEC_TIME_RE.findall(log_path.read_text(encoding="utf-8", errors="ignore"))
        if matches:
            runtime_s = float(matches[-1])
    return runtime_s, iterations


def parse_final_residuals(log_path: Path | None) -> dict | None:
    if log_path is None:
        return None
    df = load_residual_history(log_path)
    if df.empty or "step" not in df.columns:
        return None
    last = df.iloc[-1].drop(labels=["step"])
    return {k: (float(v) if pd.notna(v) else None) for k, v in last.items()}


def evaluate_convergence(
    regime: str,
    coeff_result: dict | None,
    y_plus_result: dict | None,
    fatal: bool,
    y_plus_target: float | None,
) -> tuple[bool, str]:
    """CLAUDE.md §8's 6-point regime-aware convergence rule."""
    if fatal:
        return False, "FOAM FATAL ERROR found in solver log"
    if coeff_result is None:
        return False, "forceCoeffs unreadable or missing"
    n_iters = coeff_result["n_iters"]
    if n_iters < MIN_ROWS[regime]:
        return False, f"only {n_iters} forceCoeffs rows < MIN_ROWS[{regime}]={MIN_ROWS[regime]}"
    if coeff_result["Cl_std"] >= CL_TOL[regime]:
        return False, f"Cl_std={coeff_result['Cl_std']:.5f} >= CL_TOL[{regime}]={CL_TOL[regime]}"
    if coeff_result["Cd_std"] >= CD_TOL[regime]:
        return False, f"Cd_std={coeff_result['Cd_std']:.5f} >= CD_TOL[{regime}]={CD_TOL[regime]}"
    if y_plus_result is None:
        return False, "y+ data missing (postProcessing/yPlus/0/yPlus.dat not found)"
    if y_plus_target is None:
        return False, f"no y+ target known for regime {regime!r}"
    y_plus_mean = y_plus_result["y_plus_mean"]
    if abs(y_plus_mean - y_plus_target) > 0.5 * y_plus_target:
        return False, (
            f"achieved y+ mean={y_plus_mean:.2f} outside +-50% of target {y_plus_target}"
        )
    return True, "converged"


def harvest_case(case_dir: Path) -> dict | None:
    """Fill in case_metadata.json's result fields; return a dataset row if converged."""
    case_id = case_dir.name
    metadata_path = case_dir / "case_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    regime = metadata.get("regime")

    if not has_cfd_run(case_dir):
        log.info("%s: no CFD run yet — skipping", case_id)
        return None

    log_path = find_solver_log(case_dir)
    try:
        coeff_result = load_force_coeffs(case_dir)
    except (FileNotFoundError, RuntimeError) as exc:
        log.warning("%s: forceCoeffs unreadable (%s)", case_id, exc)
        coeff_result = None

    fatal = has_fatal_error(log_path) if log_path is not None else False
    y_plus_result = parse_y_plus(case_dir)
    runtime_s, iterations = parse_runtime_and_iterations(coeff_result, log_path)
    residuals_final = parse_final_residuals(log_path)
    y_plus_target = metadata.get("y_plus_target") or Y_PLUS.get(regime)

    converged, reason = evaluate_convergence(regime, coeff_result, y_plus_result, fatal, y_plus_target)

    metadata["y_plus_min"] = y_plus_result["y_plus_min"] if y_plus_result else None
    metadata["y_plus_mean"] = y_plus_result["y_plus_mean"] if y_plus_result else None
    metadata["y_plus_max"] = y_plus_result["y_plus_max"] if y_plus_result else None
    metadata["iterations"] = iterations
    metadata["runtime_s"] = runtime_s
    metadata["residuals_final"] = residuals_final
    if coeff_result is not None:
        metadata["Cl_mean"] = coeff_result["Cl_mean"]
        metadata["Cl_std"] = coeff_result["Cl_std"]
        metadata["Cd_mean"] = coeff_result["Cd_mean"]
        metadata["Cd_std"] = coeff_result["Cd_std"]
        metadata["L_over_D"] = (
            coeff_result["Cl_mean"] / coeff_result["Cd_mean"]
            if coeff_result["Cd_mean"] not in (None, 0)
            else None
        )
    else:
        metadata["Cl_mean"] = metadata["Cl_std"] = None
        metadata["Cd_mean"] = metadata["Cd_std"] = None
        metadata["L_over_D"] = None
    metadata["converged"] = converged
    metadata["notes"] = reason

    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    # Normal path: only converged cases enter the dataset. Exception: a regime
    # in UNVALIDATED_REGIMES is included despite failing the gate, provided we
    # still parsed usable force coefficients, and is tagged validated=False so
    # downstream code can warn on it (CLAUDE.md §8 gate stays strict elsewhere).
    include_unvalidated = (
        not converged and regime in UNVALIDATED_REGIMES and coeff_result is not None
    )
    if not converged and not include_unvalidated:
        log.info("%s: not converged (%s)", case_id, reason)
        return None
    if include_unvalidated:
        log.warning(
            "%s: INCLUDED as UNVALIDATED regime-%s row despite failing gate (%s)",
            case_id, regime, reason,
        )

    return {
        "case_id": case_id,
        "regime": regime,
        "alpha_deg": metadata.get("alpha_deg"),
        "Re": metadata.get("Re"),
        "thickness": metadata.get("thickness"),
        "Cl_mean": metadata["Cl_mean"],
        "Cl_std": metadata["Cl_std"],
        "Cd_mean": metadata["Cd_mean"],
        "Cd_std": metadata["Cd_std"],
        "L_over_D": metadata["L_over_D"],
        "y_plus_mean": metadata["y_plus_mean"],
        "iterations": metadata["iterations"],
        "runtime_s": metadata["runtime_s"],
        "validated": bool(converged),
    }


# ---------------------------------------------------------------------------
# Dataset assembly
# ---------------------------------------------------------------------------

def build_dataset_clean(records: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(records, columns=DATASET_COLUMNS)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "dataset_clean.csv"
    df.to_csv(out_path, index=False)
    log.info("Wrote %s (%d rows)", out_path.relative_to(PROJECT_ROOT), len(df))
    return df


def main() -> None:
    case_dirs = discover_case_dirs()
    log.info("Found %d meshed cases with case_metadata.json", len(case_dirs))

    records: list[dict] = []
    counts = {"not_run": 0, "converged": 0, "failed": 0}
    for case_dir in case_dirs:
        had_run = has_cfd_run(case_dir)
        try:
            record = harvest_case(case_dir)
        except Exception as exc:  # one bad case must never abort the run
            log.error("%s: harvest failed with an unexpected error (%s) — skipping", case_dir.name, exc)
            counts["failed"] += 1
            continue

        if record is not None:
            counts["converged"] += 1
            records.append(record)
        elif had_run:
            counts["failed"] += 1
        else:
            counts["not_run"] += 1

    log.info(
        "Harvest summary: converged=%d  failed=%d  not_run=%d",
        counts["converged"], counts["failed"], counts["not_run"],
    )
    if records:
        by_regime = pd.DataFrame(records).groupby("regime").size()
        log.info("Converged cases per regime:\n%s", by_regime.to_string())
    else:
        log.warning("No converged cases found — results/dataset_clean.csv will be empty")

    build_dataset_clean(records)


if __name__ == "__main__":
    main()
