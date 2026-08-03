#!/usr/bin/env python3
"""
Script: 11_sweep_curves.py
Stage:  11 — Surrogate aerodynamic curves
Purpose: Sweep angle of attack at a fixed (Re, thickness) through the trained
         surrogate and plot the four standard aerodynamic curves:
           1) Cl vs alpha
           2) Cd vs alpha
           3) L/D vs alpha
           4) drag polar (Cl vs Cd)

Every alpha point is passed through scripts.surrogate.inference, so the same
OOD / trained-envelope gate used at prediction time applies here: points that
fall outside a trained regime's achieved envelope (or classify into an
untrained regime like B) are skipped, not extrapolated. Whatever survives is
plotted, and a console summary reports how many points were dropped and why.

Two modes:
  * --regime <R>  pins a single regime (one model over its one-hot column).
  * omit --regime for AUTO mode, which blends the overlapping validated regime
    experts with a partition of unity (scripts.surrogate.inference.resolve_
    weights). Unlike a hard per-point classifier, this is continuous across
    regime boundaries, so a curve spanning e.g. the A/D overlap has no step.
    The alpha span where more than one regime contributes is reported on the
    console. Unvalidated regimes (C) never blend — they keep the single-model path.

Usage:
    uv run python scripts/11_sweep_curves.py \
        --re 2.1e6 --thickness 0.12 --regime A
    uv run python scripts/11_sweep_curves.py \
        --re 5e5 --thickness 0.12 --regime C --family gp
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.surrogate.inference import (  # noqa: E402
    OutOfDistributionError,
    predict,
    predict_blended,
    resolve_weights,
)
from scripts.surrogate.regime_bounds import load_regime_bounds  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--re", type=float, required=True, help="Reynolds number (fixed for the sweep)")
    p.add_argument("--thickness", type=float, required=True, help="max thickness / chord (fixed)")
    p.add_argument("--regime", choices=["A", "B", "C", "D"], default=None,
                   help="pin a single regime (recommended); omit to auto-classify per alpha")
    p.add_argument("--family", choices=["gp", "rf", "mlp", "krg"], default="gp",
                   help="model family (default gp — the recommended one)")
    p.add_argument("--alpha-min", type=float, default=None, help="sweep start (deg); default = regime envelope")
    p.add_argument("--alpha-max", type=float, default=None, help="sweep end (deg); default = regime envelope")
    p.add_argument("--n", type=int, default=61, help="number of alpha points (default 61)")
    p.add_argument("--out", type=Path, default=None,
                   help="optional PNG path to save to; omit to just display the figure")
    return p.parse_args()


def default_alpha_range(regime: str | None, cli_min, cli_max) -> tuple[float, float]:
    """Fall back to the pinned regime's achieved alpha envelope when the user
    doesn't specify a range, so the sweep lands inside trained territory."""
    lo, hi = 0.0, 8.0
    if regime is not None:
        bounds = load_regime_bounds()
        if regime in bounds:
            lo, hi = bounds[regime]["alpha_deg"]
    if cli_min is not None:
        lo = cli_min
    if cli_max is not None:
        hi = cli_max
    return lo, hi


def _fmt_weights(weights: dict) -> str:
    """'A:0.58 D:0.42' — compact regime-composition label for the skip/summary log."""
    return " ".join(f"{r}:{w:.2f}" for r, w in sorted(weights.items()))


def sweep(re: float, thickness: float, regime: str | None, family: str,
          alphas: np.ndarray) -> dict:
    """Return arrays of kept alpha/Cl/Cd plus per-point regime composition.

    Pinned mode (`regime` given) uses the single-model `predict`. Auto mode
    (`regime is None`) blends overlapping validated regimes via `predict_blended`
    so the curve stays continuous across regime boundaries; the per-point weight
    dict is recorded so the caller can shade the multi-regime blend span.
    """
    kept_alpha, kept_cl, kept_cd = [], [], []
    kept_weights: list[dict] = []
    skipped: list[tuple[float, str]] = []
    for a in alphas:
        try:
            if regime is None:
                weights = resolve_weights(a, re, thickness)
                cl = predict_blended(a, re, thickness, family=family, target="Cl", weights=weights)
                cd = predict_blended(a, re, thickness, family=family, target="Cd", weights=weights)
            else:
                weights = {regime: 1.0}
                cl = predict(a, re, thickness, family=family, target="Cl", regime=regime)
                cd = predict(a, re, thickness, family=family, target="Cd", regime=regime)
        except OutOfDistributionError as exc:
            skipped.append((float(a), str(exc)))
            continue
        kept_alpha.append(float(a))
        kept_cl.append(cl)
        kept_cd.append(cd)
        kept_weights.append(weights)
    return {
        "alpha": np.array(kept_alpha),
        "Cl": np.array(kept_cl),
        "Cd": np.array(kept_cd),
        "weights": kept_weights,
        "skipped": skipped,
    }


def _blend_spans(alpha: np.ndarray, weights: list[dict]) -> list[tuple[float, float]]:
    """Contiguous alpha intervals where more than one regime contributes."""
    spans: list[tuple[float, float]] = []
    start = None
    for i, w in enumerate(weights):
        multi = len(w) > 1
        if multi and start is None:
            start = alpha[i]
        elif not multi and start is not None:
            spans.append((start, alpha[i - 1]))
            start = None
    if start is not None:
        spans.append((start, alpha[-1]))
    return spans


def plot_curves(data: dict, meta: dict):
    """Build the 2x2 curve figure and return it (caller decides show vs save)."""
    alpha, cl, cd = data["alpha"], data["Cl"], data["Cd"]
    l_over_d = np.divide(cl, cd, out=np.full_like(cl, np.nan), where=cd != 0)

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    title = (
        f"Surrogate curves — regime {meta['regime']}, family {meta['family']}, "
        f"Re={meta['re']:.3g}, t={meta['thickness']:.3f}"
    )
    fig.suptitle(title, fontsize=13)

    ax = axes[0, 0]
    ax.plot(alpha, cl, "o-", color="tab:blue")
    ax.set_xlabel("alpha (deg)"); ax.set_ylabel("Cl"); ax.set_title("Cl vs alpha")
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(alpha, cd, "o-", color="tab:red")
    ax.set_xlabel("alpha (deg)"); ax.set_ylabel("Cd"); ax.set_title("Cd vs alpha")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(alpha, l_over_d, "o-", color="tab:green")
    ax.set_xlabel("alpha (deg)"); ax.set_ylabel("L/D = Cl/Cd"); ax.set_title("L/D vs alpha")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(cd, cl, "o-", color="tab:purple")
    ax.set_xlabel("Cd"); ax.set_ylabel("Cl"); ax.set_title("Drag polar (Cl vs Cd)")
    ax.grid(True, alpha=0.3)

    if meta["caveat"]:
        fig.text(0.5, 0.005, meta["caveat"], ha="center", fontsize=8, color="dimgray", wrap=True)

    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    return fig


def main() -> int:
    args = parse_args()
    lo, hi = default_alpha_range(args.regime, args.alpha_min, args.alpha_max)
    alphas = np.linspace(lo, hi, args.n)
    log.info("Sweeping alpha %.3f..%.3f (%d pts) at Re=%.3g t=%.3f regime=%s family=%s",
             lo, hi, args.n, args.re, args.thickness, args.regime or "auto", args.family)

    data = sweep(args.re, args.thickness, args.regime, args.family, alphas)
    n_kept = len(data["alpha"])
    if n_kept == 0:
        log.error(
            "No alpha points fell inside a trained envelope — nothing to plot. "
            "Check that Re/thickness sit inside a trained regime (see models/regime_bounds.json)."
        )
        for a, why in data["skipped"][:3]:
            log.error("  e.g. alpha=%.2f rejected: %s", a, why)
        return 2

    log.info("Kept %d/%d points; %d skipped (out of trained envelope).",
             n_kept, len(alphas), len(data["skipped"]))

    if args.regime is None:
        spans = _blend_spans(data["alpha"], data["weights"])
        if spans:
            log.info("Auto mode: %d regime-blend span(s) (continuous across boundaries): %s",
                     len(spans), ", ".join(f"[{s:.2f}, {e:.2f}] deg" for s, e in spans))
            mid = len(data["weights"]) // 2
            log.info("  e.g. at alpha=%.2f the blend is %s",
                     data["alpha"][mid], _fmt_weights(data["weights"][mid]))
        else:
            single = sorted({r for w in data["weights"] for r in w})
            log.info("Auto mode: no overlap along this sweep — single regime %s throughout.", single)

    # Trust caveats surfaced on the plot itself.
    caveats = []
    if args.regime == "C" or any("C" in w for w in data["weights"]):
        caveats.append("Regime C is UNVALIDATED (weak CFD convergence) — treat as low-confidence.")
    if args.regime is None:
        caveats.append("Auto mode: overlapping regimes blended (partition of unity) — C1-continuous across regime boundaries, not a single CFD template.")
    if args.family == "mlp":
        caveats.append("mlp Cd is unreliable — prefer gp.")
    caveats.append("Absolute Cd carries a known offset (see validation/regime_A/report.md); shapes are more reliable than magnitudes.")
    meta = {
        "regime": args.regime or "auto", "family": args.family,
        "re": args.re, "thickness": args.thickness, "caveat": "  ".join(caveats),
    }

    fig = plot_curves(data, meta)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.out, dpi=150)
        log.info("Wrote %s", args.out)
    else:
        plt.show()
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
