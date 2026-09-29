#!/usr/bin/env python3
"""
Script: 10_global_validation.py
Stage:  10 — Evaluate and benchmark (AirfRANS)
Purpose: Score every task × family on its frozen test split, draw parity and
         residual plots, check GP calibration, and benchmark against the
         AirfRANS paper's field-model force and surface metrics.

Outputs:
    results/airfrans_metrics.csv            one row per task × family × target
    results/airfrans_curve_metrics.csv      one row per task × family × curve quantity
    results/airfrans_benchmark.md           our models vs the paper's four models
    results/figures/eval/parity_{task}_{Cl,Cd}.png
    results/figures/eval/residuals_{task}.png
    results/figures/curves/examples_{task}.png   best / median / worst test curves
    results/figures/curves/naca0012_reference.png

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

from scripts.airfrans.plotting import INK_MUTED, REFERENCE, SERIES, plt, save  # noqa: E402
from scripts.airfrans.reference import REF_DIR, read_tecplot_zones  # noqa: E402
from scripts.surrogate.curve_eval import (  # noqa: E402
    NativeWall,
    curve_metrics,
    integrated_coefficients,
    paper_surface_errors,
)
from scripts.surrogate.curves import load_curves, predict_curves, split_sides  # noqa: E402
from scripts.surrogate.data import MODELS_DIR, TARGETS, TASKS, get_xy, load_dataset, task_frames  # noqa: E402
from scripts.surrogate.inference import predict_surface  # noqa: E402
from scripts.surrogate.models import FAMILIES, FAMILY_LABELS, predict  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(message)s")
log = logging.getLogger(__name__)

RESULTS_DIR = PROJECT_ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures" / "eval"
CURVE_FIG_DIR = RESULTS_DIR / "figures" / "curves"
METRICS_PATH = RESULTS_DIR / "airfrans_metrics.csv"
CURVE_METRICS_PATH = RESULTS_DIR / "airfrans_curve_metrics.csv"
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


def curve_quantities(task: str) -> list[str]:
    return list(json.loads((MODELS_DIR / task / "envelope.json").read_text()).get("curve_info", {}))


def evaluate_curves(task: str, df: pd.DataFrame, native: NativeWall) -> tuple[list[dict], dict]:
    """Curve metrics, the paper's surface metrics and curve-integrated C_L / C_D for one task."""
    _, test = task_frames(task, df)
    d = MODELS_DIR / task
    X = joblib.load(d / "preprocessor.joblib").transform(get_xy(test)[0])
    x_grid, curves = load_curves()
    ids = test.sample_id.to_numpy()
    quantities = curve_quantities(task)
    q_inf = 0.5 * test.U_inf.to_numpy() ** 2
    alpha = np.radians(test.alpha_deg.to_numpy())
    rows, preds = [], {}
    for family in FAMILIES:
        for q in quantities:
            pca = joblib.load(d / "curves" / f"pca_{q}.joblib")
            model = joblib.load(d / "curves" / f"{family}_{q}.joblib")
            mean, std = predict_curves(pca, model, family, X, return_std=True)
            preds[(family, q)] = (mean, std)
            rows.append({"task": task, "family": family, "quantity": q, "n_test": len(ids),
                         "n_modes": pca.n_components_,
                         **curve_metrics(q, curves[q][ids], mean, std, x_grid)})
        cp = preds[(family, "Cp")][0]
        cf = preds[(family, "Cf")][0] if "Cf" in quantities else None
        paper = paper_surface_errors(native, ids, q_inf, x_grid, cp, cf)
        cd_cl = integrated_coefficients(native, ids, alpha, test.U_inf.to_numpy(), x_grid, cp, cf)
        consistency = {"curve_int_Cl_rel_err_median": float(np.median(np.abs((cd_cl[:, 1] - test.Cl) / test.Cl))),
                       "curve_int_Cd_rel_err_median": float(np.median(np.abs((cd_cl[:, 0] - test.Cd) / test.Cd)))}
        for r in rows:
            if r["family"] == family and r["quantity"] == "Cp":
                r.update(paper)
                r.update(consistency)
    return rows, {"test": test, "preds": preds, "curves": curves, "x_grid": x_grid, "quantities": quantities}


def plot_curve_examples(task: str, out: dict, metrics: pd.DataFrame) -> Path:
    """Best / median / worst test cases by Cp RMSE of the family with the lowest Cp RMSE."""
    cp_rows = metrics[(metrics.task == task) & (metrics.quantity == "Cp")]
    family = cp_rows.sort_values("RMSE").family.iloc[0]
    test, x = out["test"], out["x_grid"]
    ids = test.sample_id.to_numpy()
    true_cp = out["curves"]["Cp"][ids]
    err = np.sqrt(np.mean((out["preds"][(family, "Cp")][0] - true_cp) ** 2, axis=1))
    order = np.argsort(err)
    picks = {"best": order[0], "median": order[len(order) // 2], "worst": order[-1]}
    quantities = out["quantities"]
    fig, axes = plt.subplots(len(quantities), 3, figsize=(13, 3.6 * len(quantities)), squeeze=False)
    for j, (label, k) in enumerate(picks.items()):
        for i, q in enumerate(quantities):
            ax = axes[i, j]
            mean, std = out["preds"][(family, q)]
            scale = 1e3 if q == "Cf" else 1
            for side, ls in (("upper", "-"), ("lower", "--")):
                t = split_sides(out["curves"][q][ids[k]][None])[side][0] * scale
                pr = split_sides(mean[k][None])[side][0] * scale
                ax.plot(x, t, color=REFERENCE, ls=ls, lw=1.4, label="CFD" if side == "upper" else None)
                ax.plot(x, pr, color=SERIES[0], ls=ls, lw=1.4,
                        label=FAMILY_LABELS[family] if side == "upper" else None)
                if std is not None:
                    sd = split_sides(std[k][None])[side][0] * scale
                    ax.fill_between(x, pr - 2 * sd, pr + 2 * sd, color=SERIES[0], alpha=0.12, lw=0)
            if q == "Cp":
                ax.invert_yaxis()
                r = test.iloc[k]
                ax.set_title(f"{label}: #{int(r.sample_id)}, α {r.alpha_deg:.1f}°, Re {r.Re / 1e6:.1f}×10⁶",
                             fontsize=10)
            else:
                ax.axhline(0, color=INK_MUTED, lw=0.8)
            ax.set_ylabel("Cp" if q == "Cp" else "Cf × 10³")
            ax.set_xlabel("x/c")
    axes[0, 0].legend()
    fig.suptitle(f"Wall curves — task '{task}' (solid upper, dashed lower; band ±2σ)", y=1.01)
    fig.tight_layout()
    return save(fig, CURVE_FIG_DIR / f"examples_{task}.png")


def _zone_alpha(zone: str) -> float:
    """α from a Ladson zone title such as 'Re=6 million, alpha=10.0254, free transition'."""
    return float(zone.split("alpha=")[1].split(",")[0])


def plot_naca0012_reference() -> Path:
    """GP (full task) Cp / Cf for NACA 0012 at Re 6×10⁶ against TMR CFL3D SST and experiment."""
    tmr_cp = read_tecplot_zones(REF_DIR / "n0012cp_cfl3d_sst.dat")
    tmr_cf = read_tecplot_zones(REF_DIR / "n0012cf_cfl3d_sst.dat")
    ladson = read_tecplot_zones(REF_DIR / "CP_Ladson.dat")
    gregory = read_tecplot_zones(REF_DIR / "CP_Gregory_expdata.dat")
    alphas = [0, 10, 15]
    has_cf = "Cf" in curve_quantities("full")
    fig, axes = plt.subplots(2 if has_cf else 1, 3, figsize=(13, 7.2 if has_cf else 3.8), squeeze=False)
    for j, a in enumerate(alphas):
        pred = predict_surface(a, 6e6, "0012", family="gp", task="full")
        ax = axes[0, j]
        for side, ls in (("upper", "-"), ("lower", "--")):
            s = pred[pred.surface == side]
            ax.plot(s.x_c, s.Cp, color=SERIES[0], ls=ls, lw=1.4, label="GP (ours)" if side == "upper" else None)
            ax.fill_between(s.x_c, s.Cp_lo, s.Cp_hi, color=SERIES[0], alpha=0.12, lw=0)
        ax.plot(*tmr_cp[f"alpha={a}"].T, color=REFERENCE, lw=1, label="TMR CFL3D SST")
        lad = next(v for k, v in ladson.items()
                   if k.startswith("Re=6 million") and abs(_zone_alpha(k) - a) < 0.1)
        ax.scatter(*lad.T, s=10, color=SERIES[1], label="Ladson Re 6×10⁶", zorder=3)
        if f"alpha={a}" in gregory:
            ax.scatter(*gregory[f"alpha={a}"].T, s=10, color=SERIES[2], marker="^",
                       label="Gregory Re 2.9×10⁶", zorder=3)
        ax.invert_yaxis()
        ax.set_title(f"NACA 0012, α = {a}°, Re = 6×10⁶")
        ax.set_xlabel("x/c")
        ax.set_ylabel("Cp")
        if has_cf:
            ax = axes[1, j]
            s = pred[pred.surface == "upper"]
            ax.plot(s.x_c, 1e3 * s.Cf, color=SERIES[0], lw=1.4, label="GP upper (ours)")
            ax.fill_between(s.x_c, 1e3 * s.Cf_lo, 1e3 * s.Cf_hi, color=SERIES[0], alpha=0.12, lw=0)
            ax.plot(tmr_cf[f"alpha={a}, upper surface"][:, 0], 1e3 * tmr_cf[f"alpha={a}, upper surface"][:, 1],
                    color=REFERENCE, lw=1, label="TMR CFL3D SST upper")
            ax.axhline(0, color=INK_MUTED, lw=0.8)
            ax.set_ylim(-2, 12)
            ax.set_xlabel("x/c")
            ax.set_ylabel("Cf × 10³")
    axes[0, 0].legend()
    if has_cf:
        axes[1, 0].legend()
    fig.tight_layout()
    return save(fig, CURVE_FIG_DIR / "naca0012_reference.png")


def curves_benchmark_md(cm: pd.DataFrame) -> list[str]:
    has_cf = "Cf" in set(cm.quantity)
    md = ["", "# Surface curves — Cp(x/c) and Cf(x/c)", "",
          "Our curve models predict PCA mode weights of the 202-station wall curves from the same six "
          "features; the paper's models predict the whole flow field. The paper's surface metrics are "
          "`mean_rel_p` and `mean_rel_wss` (x, y components): the mean over wall nodes of "
          "`|(true − pred)/true|`, averaged over test samples (`metrics.py`, `Results_test`). We "
          "evaluate them on the same native wall nodes by interpolating our curves to each node's x/c "
          "(p = Cp·½U∞², τ = Cf·½U∞²·t). The ratio is huge wherever p or τ crosses zero, so our own "
          "curve metrics follow.", ""]
    for task in TASKS:
        corr, orig = load_paper(task, "WMSE"), load_paper(task, "MSE")
        md += [f"## Task `{task}` — paper surface metrics", "",
               "| model | mean_rel_p | mean_rel_wss x | mean_rel_wss y |", "|---|---|---|---|"]
        t = cm[(cm.task == task) & (cm.quantity == "Cp")]
        for _, r in t.iterrows():
            wx = f"{r.mean_rel_wss_x:.3f}" if has_cf else "—"
            wy = f"{r.mean_rel_wss_y:.3f}" if has_cf else "—"
            md.append(f"| **ours — {FAMILY_LABELS[r.family]}** | {r.mean_rel_p:.3f} | {wx} | {wy} |")
        for k, name in enumerate(PAPER_MODELS):
            c_w, o_w = corr["mean_rel_wss"][k], orig["mean_rel_wss"][k]
            md.append(f"| paper — {name} | {corr['mean_rel_p'][k]:.3f} [{orig['mean_rel_p'][k]:.3f}] | "
                      f"{c_w[0]:.3f} [{o_w[0]:.3f}] | {c_w[1]:.3f} [{o_w[1]:.3f}] |")
        md.append("")
    md += ["## Our curve metrics", "",
           "| task | family | quantity | modes | RMSE | R² (station mean) | ±2σ coverage | suction-peak MAE | "
           "separation detected | separation x/c MAE | C_L from curves (median rel.) | C_D from curves (median rel.) |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    fmt = lambda v, f: "—" if pd.isna(v) else format(v, f)  # noqa: E731
    for _, r in cm.iterrows():
        md.append(f"| {r.task} | {FAMILY_LABELS[r.family]} | {r.quantity} | {r.n_modes} | {r.RMSE:.3g} | "
                  f"{r.R2_station_mean:.4f} | {fmt(r.get('coverage_2sigma'), '.3f')} | "
                  f"{fmt(r.get('suction_peak_MAE'), '.3f')} | {fmt(r.get('separation_detect_accuracy'), '.3f')} | "
                  f"{fmt(r.get('separation_xc_MAE'), '.3f')} | {fmt(r.get('curve_int_Cl_rel_err_median'), '.4f')} | "
                  f"{fmt(r.get('curve_int_Cd_rel_err_median'), '.4f')} |")
    md += ["", "Cf values are raw coefficients (≈10⁻³). The separation-detected column is the share of test cases "
           "where the prediction agrees with CFD on whether the upper surface has Cf < 0 between x/c = 0.02 "
           "and 0.98; the x/c error is over cases where both separate.", "",
           "Figures: `results/figures/curves/examples_{task}.png`, `results/figures/curves/naca0012_reference.png`.",
           ""]
    return md


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
    bench = benchmark_md(metrics)

    curve_tasks = [t for t in TASKS if curve_quantities(t)]
    if curve_tasks:
        native = NativeWall()
        curve_rows = []
        for task in curve_tasks:
            rows, out = evaluate_curves(task, df, native)
            curve_rows += rows
            plot_curve_examples(task, out, pd.DataFrame(rows))
            for r in rows:
                log.info("[%s] %-3s %s curves  RMSE %.4g  R² %.4f", task, r["family"], r["quantity"],
                         r["RMSE"], r["R2_station_mean"])
        cm = pd.DataFrame(curve_rows)
        cm.to_csv(CURVE_METRICS_PATH, index=False, float_format="%.6g")
        if "full" in curve_tasks:
            plot_naca0012_reference()
        bench += "\n".join(curves_benchmark_md(cm))
        log.info("Wrote %s", CURVE_METRICS_PATH.relative_to(PROJECT_ROOT))
    else:
        log.warning("no curve models found; run scripts/09_train_surrogates.py --outputs curves")
    BENCH_PATH.write_text(bench)
    log.info("Wrote %s and %s", METRICS_PATH.relative_to(PROJECT_ROOT), BENCH_PATH.relative_to(PROJECT_ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
