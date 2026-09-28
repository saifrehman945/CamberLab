#!/usr/bin/env python3
"""
Script: 21_qa_airfrans.py
Stage:  AirfRANS Phase 4 — data quality and physics validation
Purpose: Hard-check every row of results/airfrans_dataset.csv, report soft
         physics checks and outliers, and compare near-NACA0012 cases with
         Ladson's experiment and the NASA TMR CFL3D SST solution.

Rows are never edited or dropped here. Hard-check outcomes go to a separate
file, results/airfrans_qa_flags.csv (sample_id, qa_pass, qa_reason), which
the split and training stages join. Report: results/airfrans_qa.md; figures:
results/figures/qa/. Exits non-zero if more than 2% of rows fail (Gate 4).

Usage:
    uv run python scripts/21_qa_airfrans.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import cho_solve
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.airfrans.plotting import INK_MUTED, REFERENCE, SERIES, plt, save  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(message)s")
log = logging.getLogger(__name__)

SEED = 42
RESULTS_DIR = PROJECT_ROOT / "results"
DATASET_PATH = RESULTS_DIR / "airfrans_dataset.csv"
FLAGS_PATH = RESULTS_DIR / "airfrans_qa_flags.csv"
REPORT_PATH = RESULTS_DIR / "airfrans_qa.md"
FIG_DIR = RESULTS_DIR / "figures" / "qa"
REF_DIR = PROJECT_ROOT / "validation_data" / "common"

FEATURES = ["alpha_deg", "log10_Re", "t_max", "x_tmax", "m_max", "x_m"]
MAX_FAIL_FRACTION = 0.02
OUTLIER_Z = 5.0
SLOPE_RANGE = (0.09, 0.12)   # dCl/dα per degree
REF_ALPHA_TOL = 0.5          # deg: a case is compared with TMR only within this of a TMR α


# --------------------------------------------------------------------------
# Hard checks
# --------------------------------------------------------------------------

def hard_checks(df: pd.DataFrame) -> pd.DataFrame:
    reasons = [[] for _ in range(len(df))]
    numeric = df[["alpha_deg", "Re", "t_max", "x_tmax", "m_max", "x_m", "Cl", "Cd"]]
    for i in np.where(~np.isfinite(numeric.to_numpy()).all(axis=1))[0]:
        reasons[i].append("non-finite value")
    for i in np.where(~((df.Cd > 0) & (df.Cd < 0.3)))[0]:
        reasons[i].append(f"Cd={df.Cd.iloc[i]:.4g} outside (0, 0.3)")
    for i in np.where(~(df.Cl.abs() < 2.5))[0]:
        reasons[i].append(f"|Cl|={abs(df.Cl.iloc[i]):.4g} >= 2.5")
    key = pd.DataFrame({
        "alpha": df.alpha_deg.round(3), "re": (df.Re / 1e3).round(0),
        **{f: df[f].round(5) for f in ["t_max", "x_tmax", "m_max", "x_m"]},
    })
    dup = key.duplicated(keep="first")
    for i in np.where(dup)[0]:
        first = df.sample_id[(key == key.iloc[i]).all(axis=1)].iloc[0]
        reasons[i].append(f"duplicate of sample {first}")
    return pd.DataFrame({"sample_id": df.sample_id,
                         "qa_pass": [not r for r in reasons],
                         "qa_reason": ["; ".join(r) for r in reasons]})


# --------------------------------------------------------------------------
# Soft checks
# --------------------------------------------------------------------------

def lift_slope(df: pd.DataFrame) -> dict:
    sym = df[(df.m_max.abs() < 0.005) & (df.alpha_deg.abs() <= 8)]
    slope, intercept = np.polyfit(sym.alpha_deg, sym.Cl, 1)
    per_case = (sym.Cl / sym.alpha_deg)[sym.alpha_deg.abs() > 1.0]
    return {"n": len(sym), "slope": slope, "intercept": intercept,
            "per_case_median": float(per_case.median()),
            "per_case_p5": float(per_case.quantile(0.05)),
            "per_case_p95": float(per_case.quantile(0.95)),
            "in_range": SLOPE_RANGE[0] <= slope <= SLOPE_RANGE[1], "data": sym}


def zero_lift_angle(df: pd.DataFrame, slope: float) -> dict:
    cam = df[(df.m_max > 0.005) & (df.alpha_deg.abs() <= 8)].copy()
    cam["alpha0"] = cam.alpha_deg - cam.Cl / slope
    return {"n": len(cam), "frac_negative": float((cam.alpha0 < 0).mean()),
            "median": float(cam.alpha0.median()), "max": float(cam.alpha0.max()),
            "n_nonnegative": int((cam.alpha0 >= 0).sum()), "data": cam}


def loo_outliers(df: pd.DataFrame, target: np.ndarray, name: str) -> pd.DataFrame:
    """Leave-one-out standardised residuals of a GP fitted on all rows.

    Uses the closed form (Rasmussen & Williams §5.4.2): with K the fitted
    covariance (noise included) and a = K⁻¹y, the LOO residual of row i is
    a_i / [K⁻¹]_ii with variance 1 / [K⁻¹]_ii, so z_i = a_i / sqrt([K⁻¹]_ii).
    Hyperparameters are fitted once on all rows (standard approximation to
    refitting 1000 times). QA only: nothing here feeds model training.
    """
    X = StandardScaler().fit_transform(df[FEATURES].to_numpy())
    kernel = (ConstantKernel(1.0) * Matern(length_scale=np.ones(len(FEATURES)), nu=2.5)
              + WhiteKernel(1e-3))
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                  n_restarts_optimizer=2, random_state=SEED).fit(X, target)
    k_inv_diag = np.diag(cho_solve((gp.L_, True), np.eye(len(X))))
    z = gp.alpha_.ravel() / np.sqrt(k_inv_diag)
    log.info("LOO GP (%s): kernel %s", name, gp.kernel_)
    return pd.DataFrame({"sample_id": df.sample_id, f"z_{name}": z})


# --------------------------------------------------------------------------
# Reference data
# --------------------------------------------------------------------------

def read_tecplot_zones(path: Path) -> dict[str, np.ndarray]:
    zones: dict[str, list] = {}
    current = "default"
    for line in path.read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("#") or s.lower().startswith("variables"):
            continue
        if s.lower().startswith("zone"):
            current = s.split("=", 1)[1].strip().strip('"') if "=" in s else f"zone{len(zones)}"
            continue
        zones.setdefault(current, []).append([float(v) for v in s.split()])
    return {k: np.asarray(v) for k, v in zones.items()}


def reference_comparison(df: pd.DataFrame) -> dict:
    ladson = read_tecplot_zones(REF_DIR / "CLCD_Ladson_expdata.dat")
    tmr = read_tecplot_zones(REF_DIR / "n0012clcd_cfl3d_sst.dat")["default"]
    near = df[(df.m_max.abs() < 0.003) & df.t_max.between(0.11, 0.13)].copy()

    def tmr_cl(alpha: float) -> float:
        # TMR SST Cl is linear in α between its 0° and 10° points; for |α| outside
        # that use the odd symmetry of NACA0012 (Cl(−α) = −Cl(α)) and, past 10°, the 10–15° chord.
        return float(np.sign(alpha) * np.interp(abs(alpha), tmr[:, 0], tmr[:, 1]))

    rows = []
    for _, r in near.iterrows():
        j = int(np.argmin(np.abs(tmr[:, 0] - r.alpha_deg)))
        if abs(tmr[j, 0] - r.alpha_deg) <= REF_ALPHA_TOL:
            rows.append({"sample_id": int(r.sample_id), "alpha_deg": r.alpha_deg, "Re": r.Re,
                         "t_max": r.t_max, "Cl": r.Cl, "Cd": r.Cd,
                         "tmr_alpha": tmr[j, 0], "tmr_Cl_at_alpha": tmr_cl(r.alpha_deg),
                         "tmr_Cd": tmr[j, 2],
                         "dCl": r.Cl - tmr_cl(r.alpha_deg), "dCd": r.Cd - tmr[j, 2],
                         "dCd_rel": (r.Cd - tmr[j, 2]) / tmr[j, 2]})
    # Ladson: mean of the three grit zones, interpolated at each case's α (α ≤ 16°, pre-stall)
    grid = np.linspace(-4, 16, 81)
    lad = np.mean([np.column_stack([np.interp(grid, z[:, 0], z[:, 1]), np.interp(grid, z[:, 0], z[:, 2])])
                   for z in ladson.values()], axis=0)
    near["ladson_Cl"] = np.interp(near.alpha_deg, grid, lad[:, 0])
    near["ladson_Cd"] = np.interp(near.alpha_deg, grid, lad[:, 1])
    return {"near": near, "tmr_rows": pd.DataFrame(rows), "ladson": ladson, "tmr": tmr}


def plot_reference(ref: dict) -> Path:
    near, ladson, tmr = ref["near"], ref["ladson"], ref["tmr"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, col, k in [(axes[0], "Cl", 1), (axes[1], "Cd", 2)]:
        for zi, (name, z) in enumerate(ladson.items()):
            ax.plot(z[:, 0], z[:, k], "^", ms=4, mfc="none", color=REFERENCE,
                    label="Ladson exp. (Re 6e6, 3 grits)" if zi == 0 else None)
        ax.plot(tmr[:, 0], tmr[:, k], "s", ms=7, color=SERIES[1], label="NASA TMR CFL3D SST (Re 6e6)")
        sc = ax.scatter(near.alpha_deg, near[col], c=near.Re / 1e6, cmap="Blues", vmin=0, vmax=6.5,
                        s=40, edgecolors=SERIES[0], linewidths=0.8, label="AirfRANS, near-NACA0012")
        ax.set_xlabel("α (deg)")
        ax.set_ylabel(col)
        ax.set_xlim(-6, 18)
    axes[1].set_ylim(0, 0.03)
    axes[0].legend(loc="upper left")
    cb = fig.colorbar(sc, ax=axes, shrink=0.8, pad=0.01)
    cb.set_label("AirfRANS Re (×10⁶)", color=INK_MUTED)
    fig.suptitle(f"Near-NACA0012 AirfRANS cases (n = {len(near)}; |m| < 0.003, 0.11 ≤ t ≤ 0.13)", y=1.0)
    return save(fig, FIG_DIR / "reference_naca0012.png")


def plot_lift(sl: dict, zl: dict) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    d = sl["data"]
    axes[0].scatter(d.alpha_deg, d.Cl, s=10, color=SERIES[0], alpha=0.7)
    a = np.linspace(-8, 8, 2)
    axes[0].plot(a, sl["slope"] * a + sl["intercept"], color=REFERENCE, lw=1.2,
                 label=f"fit: {sl['slope']:.4f} /deg")
    axes[0].plot(a, 2 * np.pi * np.radians(a), color=SERIES[1], lw=1.2, ls="--",
                 label="thin aerofoil 2π/rad (0.110 /deg)")
    axes[0].set(xlabel="α (deg)", ylabel="Cl", title=f"Symmetric-ish sections (|m| < 0.005), n = {sl['n']}")
    axes[0].legend()
    c = zl["data"]
    axes[1].scatter(c.m_max, c.alpha0, s=10, color=SERIES[2], alpha=0.7)
    axes[1].axhline(0, color=INK_MUTED, lw=0.8)
    axes[1].set(xlabel="m_max (geometric chord)", ylabel="estimated α₀ (deg)",
                title=f"Zero-lift angle, cambered sections, n = {zl['n']}")
    return save(fig, FIG_DIR / "lift_slope_zero_lift.png")


def plot_distributions(df: pd.DataFrame) -> Path:
    cols = ["alpha_deg", "Re", "t_max", "m_max", "x_m", "Cl", "Cd"]
    fig, axes = plt.subplots(2, 4, figsize=(12, 5))
    for ax, col in zip(axes.ravel(), cols):
        ax.hist(df[col], bins=40, color=SERIES[0], edgecolor="white", linewidth=0.5)
        ax.set_title(col)
    axes.ravel()[-1].axis("off")
    fig.tight_layout()
    return save(fig, FIG_DIR / "distributions.png")


# --------------------------------------------------------------------------

def md_table(df: pd.DataFrame, floatfmt: str = ".4g") -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(format(v, floatfmt) if isinstance(v, (float, np.floating))
                                      else str(v) for v in r) + " |")
    return "\n".join(out)


def main() -> int:
    df = pd.read_csv(DATASET_PATH, keep_default_na=False, na_values=[""], dtype={"source_name": str})
    log.info("QA on %d rows", len(df))

    flags = hard_checks(df)
    flags.to_csv(FLAGS_PATH, index=False)
    n_fail = int((~flags.qa_pass).sum())
    frac = n_fail / len(df)
    log.info("Hard checks: %d / %d rows fail (%.2f%%)", n_fail, len(df), 100 * frac)

    sl = lift_slope(df)
    zl = zero_lift_angle(df, sl["slope"])
    z_cl = loo_outliers(df, df.Cl.to_numpy(), "Cl")
    z_cd = loo_outliers(df, np.log(df.Cd.to_numpy()), "logCd")
    z = z_cl.merge(z_cd, on="sample_id")
    outliers = df.merge(z, on="sample_id")
    outliers = outliers[(outliers.z_Cl.abs() > OUTLIER_Z) | (outliers.z_logCd.abs() > OUTLIER_Z)]
    ref = reference_comparison(df)
    figs = [plot_distributions(df), plot_lift(sl, zl), plot_reference(ref)]

    md = ["# AirfRANS data QA (Phase 4)", "",
          f"Dataset: `results/airfrans_dataset.csv`, {len(df)} rows. Rows are never edited; "
          "hard-check outcomes live in `results/airfrans_qa_flags.csv`.", "",
          "## Hard checks (exclude from training)", "",
          "- Cd in (0, 0.3); |Cl| < 2.5; all numeric values finite",
          "- no duplicate (α, Re, geometry) rows (α to 1e-3°, Re to 1e3, features to 1e-5)", "",
          f"**Failing rows: {n_fail} / {len(df)} ({100 * frac:.2f}%)** — Gate 4 limit is "
          f"{100 * MAX_FAIL_FRACTION:.0f}%.", ""]
    if n_fail:
        md += [md_table(df.merge(flags, on="sample_id")[~flags.qa_pass.to_numpy()]
                        [["sample_id", "alpha_deg", "Re", "t_max", "m_max", "Cl", "Cd", "qa_reason"]]), ""]

    md += ["## Soft checks (report only)", "",
           "### Lift-curve slope", "",
           f"Sections with |m_max| < 0.005 and |α| ≤ 8°: n = {sl['n']}. Pooled linear fit "
           f"dCl/dα = **{sl['slope']:.4f} /deg** (intercept {sl['intercept']:.4f}); expected "
           f"{SLOPE_RANGE[0]}–{SLOPE_RANGE[1]} → **{'PASS' if sl['in_range'] else 'OUTSIDE RANGE'}**. "
           f"Per-case Cl/α (|α| > 1°): median {sl['per_case_median']:.4f}, 5–95% "
           f"[{sl['per_case_p5']:.4f}, {sl['per_case_p95']:.4f}]. The pool mixes thickness and Re, "
           "so the spread is physical, not noise.", "",
           "### Zero-lift angle of cambered sections", "",
           f"Sections with m_max > 0.005, |α| ≤ 8° (n = {zl['n']}): α₀ ≈ α − Cl / (pooled slope). "
           f"Negative for **{100 * zl['frac_negative']:.1f}%** (median {zl['median']:.2f}°, "
           f"max {zl['max']:.2f}°; {zl['n_nonnegative']} non-negative).", "",
           f"### Outliers (leave-one-out GP, |z| > {OUTLIER_Z:g})", "",
           "One anisotropic Matérn-5/2 GP per target (Cl, log Cd) on all rows, closed-form LOO "
           "residuals. Flagged rows are listed, not dropped.", "",
           f"z_Cl: max |z| = {z.z_Cl.abs().max():.2f}; z_logCd: max |z| = {z.z_logCd.abs().max():.2f}. "
           f"**{len(outliers)} rows flagged.**", ""]
    if len(outliers):
        md += [md_table(outliers[["sample_id", "alpha_deg", "Re", "t_max", "x_tmax", "m_max", "x_m",
                                  "Cl", "Cd", "z_Cl", "z_logCd"]]), ""]

    near, tr = ref["near"], ref["tmr_rows"]
    md += ["### Reference comparison: near-NACA0012 cases", "",
           f"Selection |m_max| < 0.003, 0.11 ≤ t_max ≤ 0.13: **{len(near)} cases** "
           f"(α {near.alpha_deg.min():.2f}° to {near.alpha_deg.max():.2f}°, "
           f"Re {near.Re.min() / 1e6:.2f}–{near.Re.max() / 1e6:.2f}×10⁶).", "",
           "References: Ladson, NASA TM 4074 (Re 6×10⁶, tripped; mean of the 80/120/180-grit "
           "zones, interpolated in α) and NASA TMR CFL3D k-ω SST (Re 6×10⁶, α = 0°, 10°, 15°). "
           "AirfRANS cases are at their own Re and thickness, so this is evidence of consistency, "
           "not a like-for-like validation.", ""]
    if len(near):
        v = near.assign(dCl_ladson=near.Cl - near.ladson_Cl,
                        dCd_ladson_rel=(near.Cd - near.ladson_Cd) / near.ladson_Cd)
        md += [md_table(v[["sample_id", "alpha_deg", "Re", "t_max", "Cl", "ladson_Cl", "dCl_ladson",
                           "Cd", "ladson_Cd", "dCd_ladson_rel"]]), "",
               f"vs Ladson: median |ΔCl| = {v.dCl_ladson.abs().median():.4f}; median ΔCd/Cd = "
               f"{100 * v.dCd_ladson_rel.median():+.1f}%.", ""]
    if len(tr):
        md += [f"Cases within ±{REF_ALPHA_TOL}° of a TMR SST angle:", "", md_table(tr), "",
               f"vs TMR SST: mean |ΔCl| = {tr.dCl.abs().mean():.4f}, mean |ΔCd| = "
               f"{tr.dCd.abs().mean():.5f} ({100 * tr.dCd_rel.abs().mean():.1f}% of reference Cd).", ""]
        md += ["ΔCl uses TMR Cl interpolated to the case's own α (linear pre-stall); ΔCd uses the "
               "nearest TMR angle, so it also carries up to ±0.5° of α mismatch.", ""]
    else:
        md += [f"No near-NACA0012 case lies within ±{REF_ALPHA_TOL}° of a TMR SST angle "
               "(0°, 10°, 15°), so |ΔCl|, |ΔCd| vs TMR cannot be reported directly.", ""]

    md += ["## Figures", ""] + [f"- `{p.relative_to(PROJECT_ROOT)}`" for p in figs] + [""]
    gate = frac <= MAX_FAIL_FRACTION
    md += [f"**Gate 4: {'PASS' if gate else 'FAIL'}** ({n_fail} hard failures, {100 * frac:.2f}%).", ""]
    REPORT_PATH.write_text("\n".join(md))
    log.info("Wrote %s", REPORT_PATH.relative_to(PROJECT_ROOT))
    log.info("Lift slope %.4f/deg (%s); α0<0 for %.1f%%; %d LOO outliers; %d near-0012 cases",
             sl["slope"], "ok" if sl["in_range"] else "OUT", 100 * zl["frac_negative"],
             len(outliers), len(near))
    (log.info if gate else log.error)("Gate 4: %s", "PASS" if gate else "FAIL")
    return 0 if gate else 1


if __name__ == "__main__":
    sys.exit(main())
