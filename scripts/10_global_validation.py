#!/usr/bin/env python3
"""
Script: 10_global_validation.py
Stage:  10 — Evaluate and benchmark (AirfRANS)
Purpose: Score every task × family on its frozen test split, draw parity and
         residual plots, check GP calibration, and benchmark against the
         AirfRANS paper's field-model force metrics.

Outputs:
    results/airfrans_metrics.csv            one row per task × family × target
    results/airfrans_benchmark.md           our models vs the paper's four models
    results/figures/eval/parity_{task}_{Cl,Cd}.png
    results/figures/eval/residuals_{task}.png

This is the only stage that reads test indices. It fits nothing.

Usage:
    uv run python scripts/10_global_validation.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from matplotlib.ticker import FixedLocator, FormatStrFormatter, NullFormatter
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.airfrans.plotting import INK_MUTED, SERIES, plt, save  # noqa: E402
from scripts.surrogate.data import MODELS_DIR, TARGETS, TASKS, get_xy, load_dataset, task_frames  # noqa: E402
from scripts.surrogate.models import FAMILIES, FAMILY_LABELS, predict  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(message)s")
log = logging.getLogger(__name__)

RESULTS_DIR = PROJECT_ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures" / "eval"
METRICS_PATH = RESULTS_DIR / "airfrans_metrics.csv"
BENCH_PATH = RESULTS_DIR / "airfrans_benchmark.md"
PAPER_DIR = RESULTS_DIR / "reference" / "airfrans_paper" / "scores"
PAPER_MODELS = ["MLP", "GraphSAGE", "PointNet", "Graph U-Net"]
TARGETS_REPORTED = {"full": {"Cl_R2": 0.99, "Cd_R2": 0.95, "rho_D": 0.9}}


def rel_err(true: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """|(true − pred) / true| — the paper's force metric (Extrality/AirfRANS metrics.py:44)."""
    return np.abs((true - pred) / true)


def evaluate_task(task: str, df: pd.DataFrame) -> tuple[list[dict], dict]:
    _, test = task_frames(task, df)
    d = MODELS_DIR / task
    X = joblib.load(d / "preprocessor.joblib").transform(get_xy(test)[0])
    truth = {"Cl": test.Cl.to_numpy(), "Cd": test.Cd.to_numpy()}
    rows, preds = [], {}
    for family in FAMILIES:
        for target in TARGETS:
            model = joblib.load(d / f"{family}_{target}.joblib")
            mean, std = predict(model, family, X, return_std=True)
            y_pred = np.exp(mean) if target == "Cd" else mean
            y = truth[target]
            re_ = rel_err(y, y_pred)
            row = {"task": task, "family": family, "target": target, "n_test": len(y),
                   "R2": r2_score(y, y_pred),
                   "RMSE": float(np.sqrt(mean_squared_error(y, y_pred))),
                   "MAE": mean_absolute_error(y, y_pred),
                   "spearman": float(spearmanr(y, y_pred)[0]),
                   "rel_err_mean": float(re_.mean()), "rel_err_median": float(np.median(re_))}
            if family == "gp":
                # ±2σ band in model space (log Cd for Cd; exp is monotone, so coverage is identical)
                y_model = np.log(y) if target == "Cd" else y
                row["gp_coverage_2sigma"] = float(np.mean(np.abs(y_model - mean) <= 2 * std))
            rows.append(row)
            preds[(family, target)] = y_pred
    return rows, {"test": test, "preds": preds}


def plot_parity(task: str, target: str, test: pd.DataFrame, preds: dict, metrics: pd.DataFrame) -> Path:
    y = test[target].to_numpy()
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.6), sharex=True, sharey=True)
    lo, hi = y.min(), y.max()
    pad = 0.05 * (hi - lo)
    for ax, family in zip(axes, FAMILIES):
        p = preds[(family, target)]
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=INK_MUTED, lw=1)
        ax.scatter(y, p, s=10, color=SERIES[0], alpha=0.75, linewidths=0)
        m = metrics[(metrics.task == task) & (metrics.family == family) & (metrics.target == target)].iloc[0]
        ax.set_title(f"{FAMILY_LABELS[family]}   R² {m.R2:.4f}   ρ {m.spearman:.3f}", fontsize=10)
        ax.set_xlabel(f"CFD {target}")
    axes[0].set_ylabel(f"predicted {target}")
    if target == "Cd":
        ticks = [t for t in (0.006, 0.008, 0.01, 0.015, 0.02, 0.03, 0.04, 0.06) if lo - pad <= t <= hi + pad]
        for ax in axes:
            ax.set_xscale("log")
            ax.set_yscale("log")
            for axis in (ax.xaxis, ax.yaxis):
                axis.set_major_locator(FixedLocator(ticks))
                axis.set_major_formatter(FormatStrFormatter("%g"))
                axis.set_minor_formatter(NullFormatter())
    fig.suptitle(f"Parity — task '{task}', {len(y)} test cases", y=1.02)
    fig.tight_layout()
    return save(fig, FIG_DIR / f"parity_{task}_{target}.png")


def plot_residuals(task: str, test: pd.DataFrame, preds: dict) -> Path:
    fig, axes = plt.subplots(4, 4, figsize=(13, 10), sharex="col")
    cols = [("Cl", "alpha_deg"), ("Cl", "Re"), ("Cd", "alpha_deg"), ("Cd", "Re")]
    for i, family in enumerate(FAMILIES):
        for j, (target, x) in enumerate(cols):
            ax = axes[i, j]
            y = test[target].to_numpy()
            r = preds[(family, target)] - y
            if target == "Cd":
                r = r / y * 100
            ax.scatter(test[x] / (1e6 if x == "Re" else 1), r, s=8, color=SERIES[0], alpha=0.7, linewidths=0)
            ax.axhline(0, color=INK_MUTED, lw=0.8)
            if i == 0:
                ax.set_title(f"{target} residual vs {'α' if x == 'alpha_deg' else 'Re'}")
            if i == 3:
                ax.set_xlabel("α (deg)" if x == "alpha_deg" else "Re (×10⁶)")
            if j == 0:
                ax.set_ylabel(f"{FAMILY_LABELS[family]}\npred − CFD")
            if j == 2:
                ax.set_ylabel("(pred − CFD)/CFD (%)")
    fig.suptitle(f"Test residuals — task '{task}' (Cl absolute, Cd relative)", y=1.0)
    fig.tight_layout()
    return save(fig, FIG_DIR / f"residuals_{task}.png")


def load_paper(task: str, kind: str) -> dict:
    return json.loads((PAPER_DIR / task / f"score_{kind}.json").read_text())


def benchmark_md(metrics: pd.DataFrame) -> str:
    md = ["# AirfRANS benchmark — scalar coefficient surrogates vs the paper's field models", "",
          "Our models regress C_L and C_D directly from (α, Re, section geometry); the paper's models "
          "predict the flow field and integrate forces from it. Metrics follow the paper exactly: "
          "Spearman's ρ between true and predicted coefficients over the test set, and the mean "
          "relative error `|(true − pred)/true|` (Extrality/AirfRANS `metrics.py`, `rel_err`). The "
          "relative error is a **raw ratio, not a percentage**: 0.05 = 5%.", "",
          "**Test sets.** Our splits are the official AirfRANS memberships (read from the dataset "
          "card; see `splits/README.md`) and QA excluded no rows, so every test set is identical to "
          "the paper's. Our numbers are one deterministic fit per model (seed 42); the paper's are "
          "means over 5 trained copies.", "",
          "**Which paper numbers.** Primary: the corrected results of arXiv:2212.07564 v3, "
          "Appendix N Table 27 (`score_WMSE.json`), after the authors' 26 May 2023 disclaimer that "
          "the main-text models had been trained with a plain MSE loss. The original main-text "
          "numbers (`score_MSE.json`, Tables 3 and 5) are given in brackets. Both files are cached in "
          "`results/reference/airfrans_paper/`.", ""]
    for task in TASKS:
        corr, orig = load_paper(task, "WMSE"), load_paper(task, "MSE")
        md += [f"## Task `{task}`", "",
               "| model | ρ_D | ρ_L | mean rel. err. C_D | mean rel. err. C_L | median rel. err. C_D | median rel. err. C_L |",
               "|---|---|---|---|---|---|---|"]
        for family in FAMILIES:
            m = metrics[(metrics.task == task) & (metrics.family == family)].set_index("target")
            md.append(f"| **ours — {FAMILY_LABELS[family]}** | {m.loc['Cd', 'spearman']:.3f} | "
                      f"{m.loc['Cl', 'spearman']:.3f} | {m.loc['Cd', 'rel_err_mean']:.4f} | "
                      f"{m.loc['Cl', 'rel_err_mean']:.3f} | {m.loc['Cd', 'rel_err_median']:.4f} | "
                      f"{m.loc['Cl', 'rel_err_median']:.4f} |")
        for k, name in enumerate(PAPER_MODELS):
            c_r, o_r = corr["spearman_coef_mean"][k], orig["spearman_coef_mean"][k]
            c_e, o_e = corr["mean_score_force"][k], orig["mean_score_force"][k]
            md.append(f"| paper — {name} | {c_r[0]:.3f} [{o_r[0]:.3f}] | {c_r[1]:.3f} [{o_r[1]:.3f}] | "
                      f"{c_e[0]:.3f} [{o_e[0]:.3f}] | {c_e[1]:.3f} [{o_e[1]:.3f}] | — | — |")
        md.append("")
    md += ["## Our models, all test metrics", "",
           "| task | family | target | R² | RMSE | MAE | Spearman | mean rel. err. | median rel. err. | GP ±2σ coverage |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in metrics.iterrows():
        cov = f"{r.gp_coverage_2sigma:.3f}" if pd.notna(r.get("gp_coverage_2sigma")) else "—"
        md.append(f"| {r.task} | {FAMILY_LABELS[r.family]} | {r.target} | {r.R2:.4f} | {r.RMSE:.3g} | "
                  f"{r.MAE:.3g} | {r.spearman:.4f} | {r.rel_err_mean:.4f} | {r.rel_err_median:.4f} | {cov} |")
    t = TARGETS_REPORTED["full"]
    full = metrics[metrics.task == "full"]
    md += ["", "## Reporting targets (task `full`, not gates)", "",
           f"Cl R² ≥ {t['Cl_R2']}, Cd R² ≥ {t['Cd_R2']}, Spearman ρ_D ≥ {t['rho_D']}.", "",
           "| family | Cl R² | Cd R² | ρ_D | all met |", "|---|---|---|---|---|"]
    for family in FAMILIES:
        m = full[full.family == family].set_index("target")
        ok = (m.loc["Cl", "R2"] >= t["Cl_R2"] and m.loc["Cd", "R2"] >= t["Cd_R2"]
              and m.loc["Cd", "spearman"] >= t["rho_D"])
        md.append(f"| {FAMILY_LABELS[family]} | {m.loc['Cl', 'R2']:.4f} | {m.loc['Cd', 'R2']:.4f} | "
                  f"{m.loc['Cd', 'spearman']:.4f} | {'yes' if ok else 'no'} |")
    md += ["", "Mean relative error on C_L is dominated by test cases with C_L near zero (small "
           "denominators), which is why the median is also reported.", "",
           "Figures: `results/figures/eval/parity_{task}_{Cl,Cd}.png`, "
           "`results/figures/eval/residuals_{task}.png`.", ""]
    return "\n".join(md)


def main() -> int:
    df = load_dataset()
    all_rows = []
    for task in TASKS:
        rows, out = evaluate_task(task, df)
        all_rows += rows
        m = pd.DataFrame(rows)
        for target in TARGETS:
            plot_parity(task, target, out["test"], out["preds"], m)
        plot_residuals(task, out["test"], out["preds"])
        for r in rows:
            log.info("[%s] %-3s %s  R² %.4f  ρ %.4f  rel-err mean %.4f median %.4f", task, r["family"],
                     r["target"], r["R2"], r["spearman"], r["rel_err_mean"], r["rel_err_median"])
    metrics = pd.DataFrame(all_rows)
    metrics.to_csv(METRICS_PATH, index=False, float_format="%.6g")
    BENCH_PATH.write_text(benchmark_md(metrics))
    log.info("Wrote %s and %s", METRICS_PATH.relative_to(PROJECT_ROOT), BENCH_PATH.relative_to(PROJECT_ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
