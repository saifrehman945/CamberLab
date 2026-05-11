#!/usr/bin/env python3
"""
Module: scripts/validation/parsers.py
Purpose: Load NASA TMR and other public reference datasets into standardized
         pandas DataFrames. All loaders normalize column names, coordinate
         conventions (x/c ∈ [0,1]) and the Cp sign convention so downstream
         comparison logic is dataset-agnostic.

Inputs are the raw `.dat` files under `validation_data/`. Outputs are
DataFrames with the schemas declared in `validation_data/README.md`.

Usage (as a library):
    from scripts.validation.parsers import (
        load_ladson_clcd, load_abbott_cl, load_tmr_cp, load_tmr_clcd,
        load_naca4412_cp, load_metadata,
    )
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
VALIDATION_DATA_DIR = PROJECT_ROOT / "validation_data"


# ---------------------------------------------------------------------------
# Low-level: Tecplot-style ASCII reader
# ---------------------------------------------------------------------------

_ZONE_RE = re.compile(r'^\s*zone\s*,\s*t\s*=\s*"([^"]*)"', re.IGNORECASE)
_VARIABLES_RE = re.compile(r'^\s*variables\s*=\s*(.+)$', re.IGNORECASE)


def _read_tecplot_zones(path: Path) -> tuple[list[str], list[tuple[str, np.ndarray]]]:
    """Parse a TMR-style Tecplot ASCII file into (variables, [(zone_title, data)]).

    Lines starting with ``#`` are comments. The first non-comment ``variables=``
    line declares column names. Each ``zone, t="..."`` line starts a new data
    block; if the file has no zone line, all numeric rows form a single
    unnamed zone (empty title).
    """
    if not path.exists():
        raise FileNotFoundError(f"reference data file not found: {path}")

    variables: list[str] = []
    zones: list[tuple[str, list[list[float]]]] = []
    current_zone_title: str | None = None
    current_rows: list[list[float]] = []

    def flush_zone() -> None:
        nonlocal current_rows, current_zone_title
        if current_rows or current_zone_title is not None:
            title = current_zone_title if current_zone_title is not None else ""
            zones.append((title, current_rows))
        current_rows = []
        current_zone_title = None

    with path.open("r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue

            zm = _ZONE_RE.match(line)
            if zm:
                if current_rows or current_zone_title is not None:
                    flush_zone()
                current_zone_title = zm.group(1)
                continue

            vm = _VARIABLES_RE.match(line)
            if vm:
                variables = _split_variables(vm.group(1))
                continue

            tokens = line.split()
            try:
                row = [float(t) for t in tokens]
            except ValueError:
                # Stray non-numeric line — skip but log at debug level.
                log.debug("skipping non-numeric line in %s: %r", path.name, line)
                continue
            current_rows.append(row)

    flush_zone()
    materialized: list[tuple[str, np.ndarray]] = [
        (title, np.asarray(rows, dtype=np.float64))
        for title, rows in zones
        if rows
    ]
    return variables, materialized


def _split_variables(raw: str) -> list[str]:
    """Split a tecplot variables= line into clean column names."""
    parts = re.findall(r'"([^"]*)"', raw)
    if not parts:
        parts = [p.strip() for p in raw.split(",")]
    cleaned = []
    for p in parts:
        name = p.strip().lower()
        name = re.sub(r"[\s,]*deg[\s,]*", "_deg", name)
        name = name.replace("/", "_")
        name = re.sub(r"[^a-z0-9_]+", "_", name).strip("_")
        cleaned.append(name)
    return cleaned


# ---------------------------------------------------------------------------
# Cl / Cd loaders
# ---------------------------------------------------------------------------

def load_ladson_clcd(
    path: Path | str | None = None,
    zone_filter: str | None = "80 grit",
) -> pd.DataFrame:
    """Ladson NASA TM 4074 Cl/Cd table (NACA 0012, Re=6e6, tripped).

    The original file contains three zones (80-grit, 120-grit, 180-grit
    roughness) representing different trip conditions. Pass `zone_filter`
    to keep only zones whose title contains that substring; pass `None`
    to return all rows tagged in the `source` column. The canonical
    "fully turbulent" reference is the 80-grit zone (the default).

    Returns columns: alpha_deg, Cl, Cd, source, Re, M, trip_condition, zone.
    """
    p = Path(path) if path else VALIDATION_DATA_DIR / "common" / "CLCD_Ladson_expdata.dat"
    _, zones = _read_tecplot_zones(p)
    if not zones:
        raise ValueError(f"no numeric rows found in {p}")
    rows = []
    for title, arr in zones:
        if arr.shape[1] < 3:
            continue
        if zone_filter is not None and title != zone_filter:
            continue
        df = pd.DataFrame(arr[:, :3], columns=["alpha_deg", "Cl", "Cd"])
        df["source"] = f"Ladson NASA TM 4074 ({title})" if title else "Ladson NASA TM 4074"
        df["zone"] = title
        rows.append(df)
    if not rows:
        raise ValueError(
            f"no zones matched filter {zone_filter!r} in {p} "
            f"(available: {[t for t, _ in zones]})"
        )
    out = pd.concat(rows, ignore_index=True)
    out["Re"] = 6.0e6
    out["M"] = 0.15
    out["trip_condition"] = "transition fixed at x/c=0.05"
    return out


def load_abbott_cl(path: Path | str | None = None) -> pd.DataFrame:
    """Abbott & von Doenhoff NACA 0012 Cl(α) (digitized)."""
    p = Path(path) if path else VALIDATION_DATA_DIR / "common" / "0012.abbottdata.cl.dat"
    _, zones = _read_tecplot_zones(p)
    arr = np.vstack([z for _, z in zones])
    df = pd.DataFrame(arr[:, :2], columns=["alpha_deg", "Cl"])
    df["Cd"] = np.nan
    df["source"] = "Abbott & von Doenhoff (digitized)"
    df["Re"] = 6.0e6
    df["M"] = 0.0
    df["trip_condition"] = "natural"
    return df


def load_abbott_drag_polar(path: Path | str | None = None) -> pd.DataFrame:
    """Abbott & von Doenhoff NACA 0012 drag polar Cl→Cd (digitized).

    Note: this file is a polar (Cl vs Cd), not a function of α. Cross-reference
    with load_abbott_cl to recover α.
    """
    p = Path(path) if path else VALIDATION_DATA_DIR / "common" / "0012.abbottdata.cd.dat"
    _, zones = _read_tecplot_zones(p)
    arr = np.vstack([z for _, z in zones])
    df = pd.DataFrame(arr[:, :2], columns=["Cl", "Cd"])
    df["source"] = "Abbott & von Doenhoff drag polar (digitized)"
    df["Re"] = 6.0e6
    df["M"] = 0.0
    df["trip_condition"] = "natural"
    return df


def load_gregory_cl(path: Path | str | None = None) -> pd.DataFrame:
    """Gregory & O'Reilly NACA 0012 Cl(α) (Re=3e6, digitized)."""
    p = Path(path) if path else VALIDATION_DATA_DIR / "common" / "CL_Gregory_expdata.dat"
    _, zones = _read_tecplot_zones(p)
    arr = np.vstack([z for _, z in zones])
    df = pd.DataFrame(arr[:, :2], columns=["alpha_deg", "Cl"])
    df["Cd"] = np.nan
    df["source"] = "Gregory & O'Reilly (ARC R&M 3726, 1970, digitized)"
    df["Re"] = 3.0e6
    df["M"] = 0.0
    df["trip_condition"] = "tripped"
    return df


def load_tmr_clcd(path: Path | str, label: str) -> pd.DataFrame:
    """NASA TMR CFD code reference Cl/Cd table (e.g. CFL3D SA, CFL3D SST)."""
    p = Path(path)
    _, zones = _read_tecplot_zones(p)
    arr = np.vstack([z for _, z in zones])
    df = pd.DataFrame(arr[:, :3], columns=["alpha_deg", "Cl", "Cd"])
    df["source"] = label
    df["Re"] = 6.0e6
    df["M"] = 0.15
    df["trip_condition"] = "fully turbulent"
    return df


# ---------------------------------------------------------------------------
# Cp loaders
# ---------------------------------------------------------------------------

def load_tmr_cp(path: Path | str, label: str | None = None) -> pd.DataFrame:
    """NASA TMR Cp distribution file with multiple α zones.

    Returns columns: x_c, Cp, alpha_deg, source, Re, M.
    Each zone title of the form 'alpha=10' or 'Re=6 million, alpha=.0169, ...'
    is parsed to extract the angle of attack.
    """
    p = Path(path)
    _, zones = _read_tecplot_zones(p)
    rows = []
    for title, arr in zones:
        if arr.shape[1] < 2:
            continue
        alpha = _extract_alpha_from_zone_title(title)
        df = pd.DataFrame(arr[:, :2], columns=["x_c", "Cp"])
        df["alpha_deg"] = alpha
        df["source"] = label or p.stem
        rows.append(df)
    out = pd.concat(rows, ignore_index=True)
    out["Re"] = 6.0e6
    out["M"] = 0.15
    return out


def load_naca4412_cp(path: Path | str | None = None) -> pd.DataFrame:
    """Coles & Wadcock NACA 4412 Cp distribution at α=13.87°, Re=1.52e6.

    The raw file gives (x, y, Cp) walking around the airfoil starting from
    the upper-surface trailing edge, through the leading-edge suction peak,
    and back to the lower-surface trailing edge. We keep all points and
    expose `y_c` so callers can split upper/lower if needed.
    """
    p = Path(path) if path else VALIDATION_DATA_DIR / "regime_B" / "raw" / "naca4412.cp.expt.dat"
    _, zones = _read_tecplot_zones(p)
    arr = np.vstack([z for _, z in zones])
    df = pd.DataFrame(arr[:, :3], columns=["x_c", "y_c", "Cp"])
    df["surface"] = np.where(df["y_c"] >= 0.0, "upper", "lower")
    df["alpha_deg"] = 13.87
    df["source"] = "Coles & Wadcock (AIAA J. 17, 1979)"
    df["Re"] = 1.52e6
    df["M"] = 0.0
    return df


# ---------------------------------------------------------------------------
# OpenFOAM forceCoeffs loader (matches forms used by 04_run_cfd.py)
# ---------------------------------------------------------------------------

FORCE_COEFFS_COLUMNS = ["t", "Cm", "Cd", "Cl", "Cl_f", "Cl_r"]


def load_force_coeffs(case_dir: Path | str, window: int = 200) -> dict:
    """Average Cl, Cd, Cm over the last `window` rows of a forceCoeffs.dat.

    Looks under either `postProcessing/forceCoeffs/0/forceCoeffs.dat` or
    `postProcessing/forceCoeffs/0/coefficient.dat`, whichever exists.
    """
    case = Path(case_dir)
    candidates = [
        case / "postProcessing" / "forceCoeffs" / "0" / "forceCoeffs.dat",
        case / "postProcessing" / "forceCoeffs" / "0" / "coefficient.dat",
    ]
    coeff_path = next((c for c in candidates if c.exists()), None)
    if coeff_path is None:
        raise FileNotFoundError(
            f"forceCoeffs file not found under {case}/postProcessing/forceCoeffs/0/"
        )

    df = pd.read_csv(
        coeff_path,
        sep=r"\s+",
        comment="#",
        header=None,
        names=FORCE_COEFFS_COLUMNS[: _detect_num_cols(coeff_path)],
        dtype=np.float64,
        engine="python",
    )
    if len(df) < 50:
        raise RuntimeError(
            f"{coeff_path} has only {len(df)} rows; the solver likely diverged"
        )

    tail = df.tail(min(window, len(df)))
    return {
        "Cl_mean": float(tail["Cl"].mean()),
        "Cl_std":  float(tail["Cl"].std()),
        "Cd_mean": float(tail["Cd"].mean()),
        "Cd_std":  float(tail["Cd"].std()),
        "Cm_mean": float(tail["Cm"].mean()) if "Cm" in tail else float("nan"),
        "n_iters": int(len(df)),
        "n_window": int(len(tail)),
        "coeff_path": str(coeff_path),
    }


def _detect_num_cols(path: Path) -> int:
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            return min(len(stripped.split()), len(FORCE_COEFFS_COLUMNS))
    return len(FORCE_COEFFS_COLUMNS)


# ---------------------------------------------------------------------------
# Residual log loader
# ---------------------------------------------------------------------------

_RESIDUAL_RE = re.compile(
    r"Solving for (\w+), Initial residual = ([-+0-9.eE]+), Final residual = ([-+0-9.eE]+)"
)


def load_residual_history(log_path: Path | str) -> pd.DataFrame:
    """Parse Initial-residual values from an OpenFOAM solver log.

    Returns a wide DataFrame indexed by an integer step with one column per
    solved field (Ux, Uy, p, k, omega, nuTilda, kl, ...). Missing-field cells
    are NaN so matplotlib can plot whatever fields exist for each regime.
    """
    p = Path(log_path)
    if not p.exists():
        raise FileNotFoundError(f"solver log not found: {p}")

    field_series: dict[str, list[float]] = {}
    seen_count: dict[str, int] = {}
    n_steps = 0
    step_records: list[dict[str, float]] = []
    current_step: dict[str, float] = {}

    for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith("Time = "):
            if current_step:
                step_records.append(current_step)
                n_steps += 1
            current_step = {}
            continue
        m = _RESIDUAL_RE.search(line)
        if not m:
            continue
        field, initial, _final = m.group(1), float(m.group(2)), float(m.group(3))
        # Use first residual per field per step (avoids duplicates from non-orth corrections)
        current_step.setdefault(field, initial)
        seen_count[field] = seen_count.get(field, 0) + 1

    if current_step:
        step_records.append(current_step)

    if not step_records:
        return pd.DataFrame(columns=["step"])

    df = pd.DataFrame(step_records)
    df.insert(0, "step", np.arange(1, len(df) + 1, dtype=np.int32))
    return df


# ---------------------------------------------------------------------------
# Metadata loader
# ---------------------------------------------------------------------------

def load_metadata(regime: str, root: Path | str | None = None) -> dict:
    """Load validation_data/regime_<X>/metadata.json into a dict.

    `regime` may be 'A', 'a', 'regime_A', etc.
    """
    base = Path(root) if root else VALIDATION_DATA_DIR
    norm = regime.strip().upper().replace("REGIME_", "")
    if norm not in {"A", "B", "C", "D"}:
        raise ValueError(f"unknown regime: {regime!r} (expected A, B, C or D)")
    p = base / f"regime_{norm}" / "metadata.json"
    return json.loads(p.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

_ALPHA_IN_TITLE_RE = re.compile(r"alpha\s*=\s*(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)")


def _extract_alpha_from_zone_title(title: str) -> float:
    m = _ALPHA_IN_TITLE_RE.search(title)
    if not m:
        return float("nan")
    return float(m.group(1))


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------

def _smoke() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(message)s")
    log.info("Ladson Cl/Cd: %d rows", len(load_ladson_clcd()))
    log.info("Abbott Cl:    %d rows", len(load_abbott_cl()))
    log.info("Gregory Cl:   %d rows", len(load_gregory_cl()))
    log.info(
        "CFL3D SA Cp:  %d rows in %d α-zones",
        *_count_zones(VALIDATION_DATA_DIR / "common" / "n0012cp_cfl3d_sa.dat"),
    )
    log.info(
        "NACA4412 Cp:  %d rows",
        len(load_naca4412_cp()),
    )


def _count_zones(path: Path) -> tuple[int, int]:
    _, zones = _read_tecplot_zones(path)
    total = sum(z.shape[0] for _, z in zones)
    return total, len(zones)


if __name__ == "__main__":
    _smoke()
