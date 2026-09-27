#!/usr/bin/env python3
"""
Script: 11_sweep_curves.py
Stage:  11 — Sweep an α-polar for one section (AirfRANS surrogate)
Purpose: Predict Cl(α), Cd(α) and the drag polar for a NACA 4/5-digit section
         at fixed Re, from −5° to 15° by default, and save a CSV plus a figure.

Usage:
    uv run python scripts/11_sweep_curves.py --naca 2412 --re 3e6
    uv run python scripts/11_sweep_curves.py --naca 23012 --re 4e6 --family krg --task aoa
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.airfrans.plotting import INK_MUTED, SERIES, plt, save  # noqa: E402
from scripts.surrogate.data import TASKS  # noqa: E402
from scripts.surrogate.inference import FAMILIES, predict_curve  # noqa: E402
from scripts.surrogate.models import FAMILY_LABELS  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(message)s")
log = logging.getLogger(__name__)

OUT_DIR = PROJECT_ROOT / "results" / "sweeps"


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--naca", required=True, help="NACA 4- or 5-digit code")
    p.add_argument("--re", type=float, required=True, help="Reynolds number (fixed for the sweep)")
    p.add_argument("--family", choices=FAMILIES, default="gp")
    p.add_argument("--task", choices=TASKS, default="full")
    p.add_argument("--alpha-min", type=float, default=-5.0)
    p.add_argument("--alpha-max", type=float, default=15.0)
    p.add_argument("--n", type=int, default=81, help="number of α points")
    p.add_argument("--out", type=Path, default=None, help="output stem (default results/sweeps/...)")
    return p.parse_args(argv)


def plot(df, naca: str, re: float, family: str, task: str):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    band = "Cl_lo" in df
    a = df.alpha_deg
    axes[0].plot(a, df.Cl, color=SERIES[0])
    axes[1].plot(a, df.Cd, color=SERIES[0])
    if band:
        axes[0].fill_between(a, df.Cl_lo, df.Cl_hi, color=SERIES[0], alpha=0.15, lw=0, label="±2σ")
        axes[1].fill_between(a, df.Cd_lo, df.Cd_hi, color=SERIES[0], alpha=0.15, lw=0, label="±2σ")
        axes[0].legend(loc="upper left")
    axes[2].plot(df.Cd, df.Cl, color=SERIES[0])
    i = int(np.argmax(df.L_over_D))
    axes[2].plot(df.Cd.iloc[i], df.Cl.iloc[i], "o", ms=7, color=SERIES[1])
    axes[2].annotate(f"max L/D {df.L_over_D.iloc[i]:.0f} at α {a.iloc[i]:.1f}°",
                     (df.Cd.iloc[i], df.Cl.iloc[i]), xytext=(8, -12), textcoords="offset points",
                     color=INK_MUTED, fontsize=8.5)
    axes[0].set(xlabel="α (deg)", ylabel="Cl", title="Lift")
    axes[1].set(xlabel="α (deg)", ylabel="Cd", title="Drag")
    axes[2].set(xlabel="Cd", ylabel="Cl", title="Drag polar")
    fig.suptitle(f"NACA {naca}, Re {re:.3g} — {FAMILY_LABELS[family]} surrogate (task '{task}')", y=1.02)
    fig.tight_layout()
    return fig


def main(argv=None) -> int:
    args = parse_args(argv)
    alphas = np.linspace(args.alpha_min, args.alpha_max, args.n)
    try:
        df = predict_curve(alphas, args.re, args.naca, args.family, args.task)
    except ValueError as exc:
        log.error("%s", exc)
        return 2
    for w in df.attrs["warnings"]:
        log.warning("%s", w)
    stem = args.out or OUT_DIR / f"sweep_NACA{args.naca}_Re{args.re:.0f}_{args.family}_{args.task}"
    stem.parent.mkdir(parents=True, exist_ok=True)
    # append suffixes explicitly: with_suffix() would eat any '.' already in the stem
    df.to_csv(stem.parent / f"{stem.name}.csv", index=False, float_format="%.6g")
    save(plot(df, args.naca, args.re, args.family, args.task), stem.parent / f"{stem.name}.png")
    i = int(np.argmax(df.L_over_D))
    log.info("NACA %s, Re %.3g: Cl %.3f→%.3f, max L/D %.1f at α %.1f°; wrote %s.{csv,png}",
             args.naca, args.re, df.Cl.iloc[0], df.Cl.iloc[-1], df.L_over_D.iloc[i],
             df.alpha_deg.iloc[i], stem.relative_to(PROJECT_ROOT) if stem.is_relative_to(PROJECT_ROOT) else stem)
    return 0


if __name__ == "__main__":
    sys.exit(main())
