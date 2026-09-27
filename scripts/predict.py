#!/usr/bin/env python3
"""
Script: predict.py
Purpose: Thin CLI over scripts.surrogate.inference — query the AirfRANS
         surrogate at one (α, Re, NACA section) point. Out-of-envelope
         queries are answered with a warning, never silently.

Usage:
    uv run python scripts/predict.py --alpha 4 --re 3e6 --naca 2412
    uv run python scripts/predict.py --alpha 4 --re 3e6 --naca 23012 --family all
    uv run python scripts/predict.py --alpha 4 --re 3e6 --naca 2412 --task scarce --json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.surrogate.data import TASKS  # noqa: E402
from scripts.surrogate.inference import FAMILIES, naca_features, predict  # noqa: E402
from scripts.surrogate.models import FAMILY_LABELS  # noqa: E402


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--alpha", type=float, required=True, help="angle of attack, degrees")
    p.add_argument("--re", type=float, required=True, help="Reynolds number (chord-based)")
    p.add_argument("--naca", required=True, help="NACA 4- or 5-digit code, e.g. 2412 or 23012")
    p.add_argument("--family", choices=[*FAMILIES, "all"], default="gp")
    p.add_argument("--task", choices=TASKS, default="full", help="which trained model set to use")
    p.add_argument("--json", action="store_true", help="emit machine-readable JSON on stdout")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    families = FAMILIES if args.family == "all" else [args.family]
    try:
        results = {f: predict(args.alpha, args.re, args.naca, f, args.task) for f in families}
    except ValueError as exc:
        log.error("%s", exc)
        return 2

    if args.json:
        sys.stdout.write(json.dumps({"alpha_deg": args.alpha, "Re": args.re, "naca": args.naca,
                                     "task": args.task, "geometry": naca_features(args.naca),
                                     "predictions": results}, indent=2) + "\n")
        return 0

    geo = naca_features(args.naca)
    log.info("NACA %s at α = %.2f°, Re = %.3g (models: task '%s')", args.naca, args.alpha, args.re, args.task)
    log.info("  geometry: t_max %.4f @ %.3f c, m_max %.4f @ %.3f c",
             geo["t_max"], geo["x_tmax"], geo["m_max"], geo["x_m"])
    for family, r in results.items():
        cl = f"{r['Cl']:.4f}" + (f" ± {2 * r['Cl_std']:.4f}" if r["Cl_std"] is not None else "")
        cd = f"{r['Cd']:.5f}" + (f" ± {2 * r['Cd_std']:.5f}" if r["Cd_std"] is not None else "")
        log.info("  %-3s  Cl %s   Cd %s   L/D %.1f", FAMILY_LABELS[family], cl, cd, r["L_over_D"])
    if any(r["Cl_std"] is not None for r in results.values()):
        log.info("  (± values are 2σ posterior bands)")
    if not next(iter(results.values()))["in_envelope"]:
        log.warning("Query is outside the training envelope — treat as extrapolation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
