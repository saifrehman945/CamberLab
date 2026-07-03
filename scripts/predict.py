#!/usr/bin/env python3
"""
Script: predict.py
Purpose: Thin CLI over scripts.surrogate.inference — query the trained
         surrogate at a single (alpha_deg, Re, thickness) point. Points that
         name or classify into an untrained regime, or fall outside that
         regime's achieved training envelope, are rejected rather than
         silently extrapolated.

Usage:
    micromamba run -n openfoam python scripts/predict.py --alpha 4.0 --re 2.1e6 --thickness 0.12
    micromamba run -n openfoam python scripts/predict.py --alpha 20 --re 4e6 --thickness 0.12  # expect rejection
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.surrogate.inference import (  # noqa: E402
    FAMILIES,
    OutOfDistributionError,
    predict,
    predict_all,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--alpha", type=float, required=True, help="angle of attack, degrees")
    p.add_argument("--re", type=float, required=True, help="Reynolds number")
    p.add_argument("--thickness", type=float, required=True, help="max thickness / chord")
    p.add_argument("--regime", choices=["A", "B", "C", "D"], default=None,
                    help="explicit regime assertion (skips classify_regime)")
    p.add_argument("--family", choices=[*FAMILIES, "all"], default="all")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.family == "all":
            result = predict_all(args.alpha, args.re, args.thickness, regime=args.regime)
            print(f"{'family':<8}{'Cl':>12}{'Cd':>12}")
            for family, targets in result.items():
                print(f"{family:<8}{targets['Cl']:>12.5f}{targets['Cd']:>12.5f}")
        else:
            cl = predict(args.alpha, args.re, args.thickness, family=args.family, target="Cl", regime=args.regime)
            cd = predict(args.alpha, args.re, args.thickness, family=args.family, target="Cd", regime=args.regime)
            print(f"{args.family}: Cl={cl:.5f}  Cd={cd:.5f}")
    except OutOfDistributionError as exc:
        print(f"REJECTED (out of distribution): {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
