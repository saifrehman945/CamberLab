#!/usr/bin/env python3
"""
Script: predict.py
Purpose: Thin CLI over scripts.surrogate.inference — query the AirfRANS
         surrogate at one (α, Re, NACA section) point: Cl, Cd and, with
         --surface, the wall Cp(x/c) and Cf(x/c) curves. Out-of-envelope
         queries are answered with a warning, never silently.

Usage:
    uv run python scripts/predict.py --alpha 4 --re 3e6 --naca 2412
    uv run python scripts/predict.py --alpha 4 --re 3e6 --naca 23012 --family all
    uv run python scripts/predict.py --alpha 4 --re 3e6 --naca 2412 --task scarce --json
    uv run python scripts/predict.py --alpha 4 --re 3e6 --naca 2412 --surface --surface-csv cp.csv
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.surrogate.data import TASKS  # noqa: E402
from scripts.surrogate.curve_eval import SEPARATION_WINDOW  # noqa: E402
from scripts.surrogate.inference import FAMILIES, naca_features, predict, predict_surface  # noqa: E402
from scripts.surrogate.models import FAMILY_LABELS  # noqa: E402


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--alpha", type=float, required=True, help="angle of attack, degrees")
    p.add_argument("--re", type=float, required=True, help="Reynolds number (chord-based)")
    p.add_argument("--naca", required=True, help="NACA 4- or 5-digit code, e.g. 2412 or 23012")
    p.add_argument("--family", choices=[*FAMILIES, "all"], default="gp")
    p.add_argument("--task", choices=TASKS, default="full", help="which trained model set to use")
    p.add_argument("--json", action="store_true", help="emit machine-readable JSON on stdout")
    p.add_argument("--surface", action="store_true", help="also predict the wall Cp and Cf curves")
    p.add_argument("--surface-csv", type=Path, default=None,
                   help="write the wall curves (all requested families) to this CSV; implies --surface")
    return p.parse_args(argv)


def surface_summary(curve: pd.DataFrame) -> dict:
    """Suction peak and upper-surface separation point of one predicted surface table."""
    i = int(curve.Cp.idxmin())
    out = {"Cp_min": float(curve.Cp[i]), "x_c_Cp_min": float(curve.x_c[i]),
           "surface_Cp_min": str(curve.surface[i])}
    if "Cf" in curve:
        up = curve[(curve.surface == "upper") & curve.x_c.between(*SEPARATION_WINDOW)]
        neg = up[up.Cf < 0]
        out["x_c_separation_upper"] = float(neg.x_c.iloc[0]) if len(neg) else None
    return out


def main(argv=None) -> int:
    args = parse_args(argv)
    families = FAMILIES if args.family == "all" else [args.family]
    try:
        results = {f: predict(args.alpha, args.re, args.naca, f, args.task) for f in families}
        want_surface = args.surface or args.surface_csv is not None
        surfaces = ({f: predict_surface(args.alpha, args.re, args.naca, f, args.task) for f in families}
                    if want_surface else {})
    except (ValueError, FileNotFoundError) as exc:
        log.error("%s", exc)
        return 2
    if args.surface_csv is not None:
        pd.concat([c.assign(family=f) for f, c in surfaces.items()], ignore_index=True).to_csv(
            args.surface_csv, index=False, float_format="%.6g")
        log.info("Wrote wall curves to %s", args.surface_csv)

    if args.json:
        payload = {"alpha_deg": args.alpha, "Re": args.re, "naca": args.naca, "task": args.task,
                   "geometry": naca_features(args.naca), "predictions": results}
        if surfaces:
            payload["surface"] = {f: {"summary": surface_summary(c),
                                      "curves": c.replace({np.nan: None}).to_dict(orient="list")}
                                  for f, c in surfaces.items()}
        sys.stdout.write(json.dumps(payload, indent=2) + "\n")
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
    for family, curve in surfaces.items():
        s = surface_summary(curve)
        sep = s.get("x_c_separation_upper", "n/a (Cf not modelled)")
        sep = "attached" if sep is None else (f"x/c {sep:.3f}" if isinstance(sep, float) else sep)
        log.info("  %-3s  suction peak Cp %.3f at x/c %.3f (%s)   upper-surface separation: %s",
                 FAMILY_LABELS[family], s["Cp_min"], s["x_c_Cp_min"], s["surface_Cp_min"], sep)
    if not next(iter(results.values()))["in_envelope"]:
        log.warning("Query is outside the training envelope — treat as extrapolation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
