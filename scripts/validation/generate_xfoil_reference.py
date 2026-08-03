#!/usr/bin/env python3
"""
Script: scripts/validation/generate_xfoil_reference.py
Purpose: Produce engineering-grade reference Cl/Cd/Cp/transition-location data
         for Regime C (transitional low-Re NACA0012) using XFOIL with the eN
         transition prediction (n_crit=9 by default for clean tunnel).

XFOIL is not on PyPI; install it from your distro (`apt install xfoil` on
Debian/Ubuntu) or build from source. The script drives XFOIL via stdin and
parses the resulting `.pwrt`/polar text files.

Outputs (written under validation_data/regime_C/raw/):
    xfoil_naca0012_re5e5_polar.csv                   ← α, Cl, Cd, Cm, x_trans_upper, x_trans_lower
    xfoil_naca0012_re5e5_aoa{0,4,6}.dat              ← x/c, Cp per alpha

Usage:
    uv run python scripts/validation/generate_xfoil_reference.py
    uv run python scripts/validation/generate_xfoil_reference.py --re 8e5 --alpha 0 2 4 6 8
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
REGIME_C_RAW = PROJECT_ROOT / "validation_data" / "regime_C" / "raw"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate Regime C reference data with XFOIL.")
    p.add_argument("--airfoil", default="NACA 0012", help="XFOIL airfoil identifier")
    p.add_argument("--re", type=float, default=5.0e5, help="Reynolds number")
    p.add_argument("--mach", type=float, default=0.05, help="Mach number")
    p.add_argument("--ncrit", type=float, default=9.0, help="eN amplification factor (clean tunnel ≈ 9)")
    p.add_argument("--alpha", type=float, nargs="+", default=[0.0, 2.0, 4.0, 6.0, 8.0],
                   help="Angles of attack to evaluate (degrees)")
    p.add_argument("--iter", type=int, default=200, help="XFOIL max iterations per α")
    p.add_argument("--panels", type=int, default=240, help="Panel count (PANE NPAN)")
    return p.parse_args()


def require_xfoil() -> str:
    path = shutil.which("xfoil")
    if path is None:
        raise RuntimeError(
            "xfoil not found on PATH. Install with `apt install xfoil` on Debian/Ubuntu "
            "or `brew install xfoil` on macOS, then re-run."
        )
    return path


def airfoil_tag(airfoil: str) -> str:
    return airfoil.replace(" ", "").lower()


def re_tag(Re: float) -> str:
    """Compact Reynolds tag: 5e5 → 're5e5', 1e6 → 're1e6'."""
    return "re" + f"{Re:.0e}".replace("e+0", "e").replace("e-0", "e-").replace("e+", "e")


def alpha_tag(alpha: float) -> str:
    """Filesystem-safe AoA tag matching metadata: 4.0 → 'aoa4', -2.5 → 'aoam2p5'."""
    return "aoa" + f"{alpha:g}".replace("-", "m").replace(".", "p")


def write_xfoil_script(
    workdir: Path,
    airfoil: str,
    Re: float,
    Mach: float,
    ncrit: float,
    alphas: list[float],
    max_iter: int,
    n_panels: int,
    polar_file: Path,
    cp_files: dict[float, Path],
) -> str:
    """Build the XFOIL command stream for a polar sweep with Cp dumps per α."""
    lines: list[str] = []
    lines.append(airfoil)
    lines.append("PANE")
    lines.append(f"PPAR")
    lines.append(f"N {n_panels}")
    lines.append("")
    lines.append("")
    lines.append("OPER")
    lines.append(f"VISC {Re:.6g}")
    lines.append(f"MACH {Mach:.4f}")
    lines.append("VPAR")
    lines.append(f"N {ncrit:.2f}")
    lines.append("")
    lines.append(f"ITER {max_iter}")
    lines.append(f"PACC")
    lines.append(str(polar_file))
    lines.append("")
    for a in alphas:
        lines.append(f"ALFA {a:.3f}")
        cp_path = cp_files[a]
        lines.append("CPWR")
        lines.append(str(cp_path))
    lines.append("PACC")
    lines.append("")
    lines.append("QUIT")
    return "\n".join(lines) + "\n"


def run_xfoil(xfoil: str, script: str, workdir: Path) -> subprocess.CompletedProcess:
    log.info("running xfoil (%d-line script) in %s", script.count("\n"), workdir)
    proc = subprocess.run(
        [xfoil],
        input=script,
        capture_output=True,
        text=True,
        cwd=workdir,
        timeout=600,
    )
    (workdir / "xfoil.stdout.log").write_text(proc.stdout, encoding="utf-8")
    (workdir / "xfoil.stderr.log").write_text(proc.stderr, encoding="utf-8")
    if proc.returncode != 0:
        log.error("xfoil exited %d", proc.returncode)
        log.error("stderr (last 500 chars): %s", proc.stderr[-500:])
    return proc


def parse_xfoil_polar(path: Path) -> pd.DataFrame:
    """Parse an XFOIL polar accumulation file (PACC output).

    Columns vary by XFOIL build but typically include:
        alpha  CL  CD  CDp  CM  Top_Xtr  Bot_Xtr
    We auto-detect by finding the header line that contains 'alpha'.
    """
    text = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    header_idx = None
    for i, line in enumerate(text):
        low = line.lower()
        if "alpha" in low and ("cl" in low and "cd" in low):
            header_idx = i
            break
    if header_idx is None:
        raise ValueError(f"could not locate header in {path}")
    header = [h.strip().lower() for h in text[header_idx].split()]
    rows: list[list[float]] = []
    for line in text[header_idx + 2:]:  # skip the dashed separator
        if not line.strip():
            continue
        try:
            rows.append([float(t) for t in line.split()])
        except ValueError:
            continue
    df = pd.DataFrame(rows, columns=header[: len(rows[0])])
    rename = {
        "alpha": "alpha_deg",
        "cl":    "Cl",
        "cd":    "Cd",
        "cdp":   "Cdp",
        "cm":    "Cm",
        "top_xtr": "x_trans_upper",
        "bot_xtr": "x_trans_lower",
    }
    return df.rename(columns={k: v for k, v in rename.items() if k in df.columns})


def parse_xfoil_cp(path: Path) -> pd.DataFrame:
    """Parse an XFOIL CPWR output file into (x_c, Cp)."""
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    rows: list[list[float]] = []
    for line in lines:
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        try:
            tokens = [float(t) for t in s.split()]
        except ValueError:
            continue
        if len(tokens) >= 2:
            rows.append(tokens[:2])
    return pd.DataFrame(rows, columns=["x_c", "Cp"])


def main() -> int:
    args = parse_args()
    xfoil = require_xfoil()
    REGIME_C_RAW.mkdir(parents=True, exist_ok=True)

    tag = airfoil_tag(args.airfoil)
    polar_csv_out = REGIME_C_RAW / f"xfoil_{tag}_{re_tag(args.re)}_polar.csv"

    with tempfile.TemporaryDirectory(prefix="xfoil_") as tmp:
        workdir = Path(tmp)
        polar_raw = workdir / "polar.txt"
        cp_files = {a: workdir / f"cp_alpha{a:+06.2f}.dat".replace(".", "p") for a in args.alpha}
        script = write_xfoil_script(
            workdir=workdir,
            airfoil=args.airfoil,
            Re=args.re,
            Mach=args.mach,
            ncrit=args.ncrit,
            alphas=list(args.alpha),
            max_iter=args.iter,
            n_panels=args.panels,
            polar_file=polar_raw,
            cp_files=cp_files,
        )
        (workdir / "xfoil.in").write_text(script, encoding="utf-8")
        proc = run_xfoil(xfoil, script, workdir)
        if not polar_raw.exists():
            log.error("XFOIL did not produce %s; check %s/xfoil.stdout.log",
                      polar_raw, workdir)
            return proc.returncode or 2

        polar = parse_xfoil_polar(polar_raw)
        polar.to_csv(polar_csv_out, index=False)
        log.info("wrote polar → %s (%d rows)", polar_csv_out, len(polar))

        for alpha, src in cp_files.items():
            if not src.exists():
                log.warning("missing Cp file for α=%.2f (XFOIL likely failed to converge)", alpha)
                continue
            dest = REGIME_C_RAW / f"xfoil_{tag}_{re_tag(args.re)}_{alpha_tag(alpha)}.dat"
            cp = parse_xfoil_cp(src)
            with dest.open("w", encoding="utf-8") as fh:
                fh.write(f"# XFOIL Cp distribution: {args.airfoil}\n")
                fh.write(f"# Re={args.re:.3e}  Mach={args.mach}  ncrit={args.ncrit}\n")
                fh.write(f"# alpha={alpha:.3f} deg\n")
                fh.write('variables="x/c","Cp"\n')
                for _, row in cp.iterrows():
                    fh.write(f"{row['x_c']:.6f}  {row['Cp']:.6f}\n")
            log.info("wrote Cp → %s (%d rows)", dest.name, len(cp))

    return 0


if __name__ == "__main__":
    sys.exit(main())
