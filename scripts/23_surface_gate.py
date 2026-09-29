#!/usr/bin/env python3
"""
Script: 23_surface_gate.py
Stage:  Surface Phase 3 — wall-data feasibility gate
Purpose: Decide whether the extracted wall Cp and Cf are good enough to model,
         by checking that C_L and C_D re-integrated from them reproduce the
         stored AirfRANS coefficients over all 1,000 samples.

Reads the Cl_int_* / Cd_int_* columns of results/airfrans_dataset.csv and
results/airfrans_wall_curves.npz, both written by 20_ingest_airfrans.py. No
dataset I/O. Samples are never dropped; the worst ones are listed.

Pass criteria (decision D4), per quantity, over all samples:
    median relative error ≤ LIMIT and at most 5% of samples above 3 × LIMIT
    with LIMIT = 2% for C_L and 5% for C_D.

- Cp passes if the pressure-only C_L (Cl_int_p) meets the C_L criterion.
- Cf passes if the full C_D (pressure + shear) meets the C_D criterion and the
  full C_L meets the C_L criterion. Pressure drag is small and mesh-sensitive,
  so the C_D check tests the wall shear stress above all.

Writes results/airfrans_surface_gate.{md,json} and results/figures/surface/.
Exits non-zero only if Cp fails (a Cf failure means shipping Cp only, per D2).

Usage:
    uv run python scripts/23_surface_gate.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.airfrans.plotting import INK_MUTED, REFERENCE, SERIES, plt, save  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(message)s")
log = logging.getLogger(__name__)

RESULTS_DIR = PROJECT_ROOT / "results"
DATASET_PATH = RESULTS_DIR / "airfrans_dataset.csv"
CURVES_PATH = RESULTS_DIR / "airfrans_wall_curves.npz"
REPORT_PATH = RESULTS_DIR / "airfrans_surface_gate.md"
GATE_PATH = RESULTS_DIR / "airfrans_surface_gate.json"
FIG_DIR = RESULTS_DIR / "figures" / "surface"

LIMITS = {"Cl": 0.02, "Cd": 0.05}
TAIL_FACTOR = 3.0
TAIL_FRACTION = 0.05
N_WORST = 10


def criterion(true: np.ndarray, integrated: np.ndarray, limit: float) -> dict:
    rel = np.abs((integrated - true) / true)
    tail = float((rel > TAIL_FACTOR * limit).mean())
    median = float(np.median(rel))
    return {"median_rel": median, "mean_rel": float(rel.mean()), "p95_rel": float(np.quantile(rel, 0.95)),
            "median_abs": float(np.median(np.abs(integrated - true))),
            "tail_fraction": tail, "limit": limit, "tail_limit": TAIL_FACTOR * limit,
            "pass": bool(median <= limit and tail <= TAIL_FRACTION)}


def parity_figure(df: pd.DataFrame, cl_int: np.ndarray, cd_int: np.ndarray):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, true, pred, label in ((axes[0], df.Cl, cl_int, "C_L"), (axes[1], df.Cd, cd_int, "C_D")):
        ax.scatter(true, pred, s=6, color=SERIES[0], alpha=0.6, linewidths=0)
        lo, hi = float(min(true.min(), pred.min())), float(max(true.max(), pred.max()))
        ax.plot([lo, hi], [lo, hi], color=REFERENCE, lw=1)
        ax.set_xlabel(f"stored {label}")
        ax.set_ylabel(f"integrated from wall {label}")
        ax.set_title(f"{label}: wall integral vs AirfRANS")
    rel = (cd_int - df.Cd) / df.Cd
    axes[2].scatter(df.alpha_deg, 100 * rel, s=6, color=SERIES[1], alpha=0.6, linewidths=0)
    axes[2].axhline(0, color=REFERENCE, lw=1)
    for s in (-1, 1):
        axes[2].axhline(100 * s * LIMITS["Cd"], color=INK_MUTED, lw=0.8, ls="--")
    axes[2].set_xlabel("α (deg)")
    axes[2].set_ylabel("C_D relative error (%)")
    axes[2].set_title("C_D error against α")
    fig.tight_layout()
    return save(fig, FIG_DIR / "gate_parity.png")


def curves_figure(df: pd.DataFrame, curves) -> Path:
    """Cp and Cf of three samples spanning α (lowest, median, highest)."""
    order = np.argsort(df.alpha_deg.to_numpy())
    picks = [order[0], order[len(order) // 2], order[-1]]
    x = curves["x_grid"]
    fig, (ax_cp, ax_cf) = plt.subplots(1, 2, figsize=(11, 3.8))
    for colour, i in zip(SERIES, picks):
        label = f"#{i}: α = {df.alpha_deg[i]:.1f}°, Re = {df.Re[i] / 1e6:.1f}×10⁶"
        for side, ls in (("upper", "-"), ("lower", "--")):
            ax_cp.plot(x, curves[f"Cp_{side}"][i], color=colour, ls=ls, lw=1.4,
                       label=label if side == "upper" else None)
            ax_cf.plot(x, 1e3 * curves[f"Cf_{side}"][i], color=colour, ls=ls, lw=1.4)
    ax_cp.invert_yaxis()
    ax_cp.set_xlabel("x/c")
    ax_cp.set_ylabel("Cp")
    ax_cp.set_title("Wall Cp (solid upper, dashed lower)")
    ax_cp.legend()
    ax_cf.axhline(0, color=REFERENCE, lw=0.8)
    ax_cf.set_xlabel("x/c")
    ax_cf.set_ylabel("Cf × 10³")
    ax_cf.set_title("Wall Cf, signed along LE → TE")
    fig.tight_layout()
    return save(fig, FIG_DIR / "gate_examples.png")


def main() -> int:
    df = pd.read_csv(DATASET_PATH, keep_default_na=False, na_values=[""], dtype={"source_name": str})
    curves = np.load(CURVES_PATH)
    assert (curves["sample_id"] == df.sample_id.to_numpy()).all()

    cl_int = (df.Cl_int_p + df.Cl_int_tau).to_numpy()
    cd_int = (df.Cd_int_p + df.Cd_int_tau).to_numpy()
    stats = {
        "Cl_pressure_only": criterion(df.Cl.to_numpy(), df.Cl_int_p.to_numpy(), LIMITS["Cl"]),
        "Cl": criterion(df.Cl.to_numpy(), cl_int, LIMITS["Cl"]),
        "Cd": criterion(df.Cd.to_numpy(), cd_int, LIMITS["Cd"]),
    }
    cp_pass = stats["Cl_pressure_only"]["pass"]
    cf_pass = cp_pass and stats["Cl"]["pass"] and stats["Cd"]["pass"]
    shear_share = float(np.median(df.Cd_int_tau / cd_int))

    df = df.assign(Cl_int=cl_int, Cd_int=cd_int,
                   rel_Cl=np.abs((cl_int - df.Cl) / df.Cl), rel_Cd=np.abs((cd_int - df.Cd) / df.Cd))
    worst = df.nlargest(N_WORST, "rel_Cd")[["sample_id", "alpha_deg", "Re", "t_max", "m_max",
                                           "Cl", "Cl_int", "Cd", "Cd_int", "rel_Cd"]]

    GATE_PATH.write_text(json.dumps({
        "cp_pass": cp_pass, "cf_pass": cf_pass, "n_samples": len(df),
        "median_shear_fraction_of_Cd": shear_share, "stats": stats,
        "worst_Cd_samples": worst.sample_id.astype(int).tolist(),
    }, indent=2))

    figs = [parity_figure(df, cl_int, cd_int), curves_figure(df, curves)]

    md = ["# Wall Cp / Cf feasibility gate", "",
          "C_L and C_D re-integrated from the extracted wall pressure and computed wall shear "
          "stress (AirfRANS' `Compute_coefficients`, ported in `scripts/airfrans/surface.py`), "
          f"compared with the stored coefficients over all {len(df)} samples.", "",
          f"Pass rule (D4): median relative error ≤ limit, and at most {TAIL_FRACTION:.0%} of samples "
          f"above {TAIL_FACTOR:g} × limit. Limits: C_L {LIMITS['Cl']:.0%}, C_D {LIMITS['Cd']:.0%}.", "",
          "| check | median rel. | mean rel. | 95th pct rel. | median abs. | share above tail limit | pass |",
          "|---|---|---|---|---|---|---|"]
    for name, s in stats.items():
        md.append(f"| {name} | {s['median_rel']:.2%} | {s['mean_rel']:.2%} | {s['p95_rel']:.2%} | "
                  f"{s['median_abs']:.2e} | {s['tail_fraction']:.1%} (> {s['tail_limit']:.0%}) | "
                  f"{'yes' if s['pass'] else 'no'} |")
    md += ["", f"Median share of C_D carried by the wall shear stress: {shear_share:.0%}.", "",
           f"**Cp: {'PASS' if cp_pass else 'FAIL'}. Cf: {'PASS' if cf_pass else 'FAIL'}.**", ""]
    if not cf_pass:
        md += ["Cf fails, so by decision D2 only Cp curves are modelled and shipped.", ""]
    md += ["The mean relative C_L error is inflated by samples with C_L near zero; the median "
           "and the absolute error are the meaningful figures.", "",
           f"## Worst {N_WORST} samples by C_D error (listed, not dropped)", "",
           "| " + " | ".join(worst.columns) + " |", "|" + "---|" * worst.shape[1],
           *("| " + " | ".join(f"{v:.4g}" for v in row) + " |" for row in worst.itertuples(index=False)),
           "",
           "## Figures", ""]
    md += [f"![{p.stem}](figures/surface/{p.name})" for p in figs]
    REPORT_PATH.write_text("\n".join(md) + "\n")

    for name, s in stats.items():
        (log.info if s["pass"] else log.warning)(
            "Gate — %s: median %.2f%%, tail %.1f%% → %s", name, 100 * s["median_rel"],
            100 * s["tail_fraction"], "PASS" if s["pass"] else "FAIL")
    log.info("Cp %s, Cf %s. Wrote %s and %s", "PASS" if cp_pass else "FAIL",
             "PASS" if cf_pass else "FAIL", REPORT_PATH.relative_to(PROJECT_ROOT),
             GATE_PATH.relative_to(PROJECT_ROOT))
    return 0 if cp_pass else 1


if __name__ == "__main__":
    sys.exit(main())
