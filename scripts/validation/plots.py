#!/usr/bin/env python3
"""
Module: scripts/validation/plots.py
Purpose: Generate per-regime validation plots. All figures are written under
         validation/regime_<X>/figures/ and have consistent styling.

Required figures:
    cl_cd_comparison.png   ← bar charts CFD vs every reference, per case
    cp_comparison.png      ← Cp(x/c) overlay CFD vs reference, per case
    residual_history.png   ← residuals over iterations, per case
    mesh_convergence.png   ← Cl, Cd vs cell count for the mesh ladder

The module exposes a single high-level entry point `generate_regime_plots()`
plus low-level helpers callable from tests or notebooks.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.validation.parsers import (
    load_naca4412_cp,
    load_residual_history,
    load_tmr_cp,
)
from scripts.validation.compare import load_openfoam_cp

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
VALIDATION_DATA_DIR = PROJECT_ROOT / "validation_data"
VALIDATION_ROOT = PROJECT_ROOT / "validation"

REF_COLOR = "#2ca02c"
CFD_COLOR = "#1f77b4"
SECONDARY_COLOR = "#ff7f0e"
TERTIARY_COLOR = "#9467bd"


# ---------------------------------------------------------------------------
# Cl / Cd bar plot
# ---------------------------------------------------------------------------

def plot_cl_cd_comparison(comparisons: list[dict], output_path: Path) -> None:
    """One row per case; each row shows Cl and Cd bars (CFD + every reference)."""
    n = len(comparisons)
    if n == 0:
        log.warning("plot_cl_cd_comparison: no cases to plot")
        return
    fig, axes = plt.subplots(n, 2, figsize=(11, 3.0 * n), squeeze=False)

    for row, comp in enumerate(comparisons):
        ax_cl, ax_cd = axes[row]
        _render_clcd_row(ax_cl, ax_cd, comp)

    fig.suptitle("Cl / Cd: CFD vs reference", fontsize=11, y=1.0)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("wrote %s", output_path)


def _render_clcd_row(ax_cl, ax_cd, comp: dict) -> None:
    labels_cl = ["CFD"] + [r["label"] for r in comp["references"] if r.get("Cl_ref") is not None]
    values_cl = [comp["Cl_cfd"]] + [r["Cl_ref"] for r in comp["references"] if r.get("Cl_ref") is not None]
    colors_cl = [CFD_COLOR] + _palette(len(values_cl) - 1)
    bars_cl = ax_cl.bar(range(len(values_cl)), values_cl, color=colors_cl)
    ax_cl.set_xticks(range(len(values_cl)))
    ax_cl.set_xticklabels(labels_cl, rotation=20, ha="right", fontsize=8)
    ax_cl.set_ylabel("Cl")
    ax_cl.set_title(
        f"{comp['case_id']}  α={comp['alpha_deg']:.2f}°  Re={comp['Re']:.1e}",
        fontsize=9,
    )
    for bar, v in zip(bars_cl, values_cl):
        ax_cl.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.4f}",
                   ha="center", va="bottom", fontsize=7)
    ax_cl.grid(axis="y", alpha=0.3)

    labels_cd = ["CFD"] + [r["label"] for r in comp["references"] if r.get("Cd_ref") is not None]
    values_cd = [comp["Cd_cfd"]] + [r["Cd_ref"] for r in comp["references"] if r.get("Cd_ref") is not None]
    colors_cd = [CFD_COLOR] + _palette(len(values_cd) - 1)
    bars_cd = ax_cd.bar(range(len(values_cd)), values_cd, color=colors_cd)
    ax_cd.set_xticks(range(len(values_cd)))
    ax_cd.set_xticklabels(labels_cd, rotation=20, ha="right", fontsize=8)
    ax_cd.set_ylabel("Cd")
    ax_cd.set_title("Cd", fontsize=9)
    for bar, v in zip(bars_cd, values_cd):
        ax_cd.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.5f}",
                   ha="center", va="bottom", fontsize=7)
    ax_cd.grid(axis="y", alpha=0.3)


def _palette(n: int) -> list[str]:
    base = [REF_COLOR, SECONDARY_COLOR, TERTIARY_COLOR, "#8c564b", "#e377c2"]
    if n <= len(base):
        return base[:n]
    return base + plt.cm.tab10(np.linspace(0, 1, n - len(base))).tolist()


# ---------------------------------------------------------------------------
# Cp comparison plot
# ---------------------------------------------------------------------------

def plot_cp_comparison(
    case_dir: Path,
    reference_case: dict,
    output_path: Path,
) -> bool:
    """Overlay OpenFOAM Cp with every Cp reference declared in metadata."""
    refs = reference_case.get("references", {})
    fig, ax = plt.subplots(figsize=(8, 5.5))

    plotted = False
    try:
        cp_cfd = load_openfoam_cp(case_dir)
        ax.plot(cp_cfd["x_c"], cp_cfd["Cp"], "-", color=CFD_COLOR, label="CFD (foamRun)", lw=1.6)
        plotted = True
    except FileNotFoundError as exc:
        log.info("no CFD Cp for %s (%s)", reference_case["case_id"], exc)

    ref_styles = [
        ("experimental_primary",  REF_COLOR,       "s", "experimental"),
        ("cfd_reference",         SECONDARY_COLOR, "^", "CFD reference"),
        ("experimental_secondary",TERTIARY_COLOR,  "o", "exp (secondary)"),
    ]
    for role, color, marker, fallback_label in ref_styles:
        entry = refs.get(role)
        if not entry or "file_cp" not in entry:
            continue
        ref_path = VALIDATION_DATA_DIR / entry["file_cp"]
        if not ref_path.exists():
            continue
        if "naca4412" in str(ref_path).lower():
            df = load_naca4412_cp(ref_path)
        else:
            df = load_tmr_cp(ref_path, entry.get("label", fallback_label))
            zone = entry.get("cp_zone")
            if zone:
                import re
                m = re.search(r"alpha\s*=\s*(-?\d+(?:\.\d+)?)", zone)
                if m:
                    target = float(m.group(1))
                    df = df[np.isclose(df["alpha_deg"], target, atol=0.5)]
        if df.empty:
            continue
        ax.plot(df["x_c"], df["Cp"], marker, color=color, ms=5,
                label=entry.get("label", fallback_label), alpha=0.85)
        plotted = True

    if not plotted:
        plt.close(fig)
        return False

    ax.invert_yaxis()
    ax.set_xlabel("x/c")
    ax.set_ylabel("Cp")
    ax.set_title(
        f"{reference_case['case_id']}  α={reference_case['alpha_deg']:.2f}°  "
        f"Re={reference_case['Re']:.1e}",
        fontsize=10,
    )
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.3)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("wrote %s", output_path)
    return True


# ---------------------------------------------------------------------------
# Residual history plot
# ---------------------------------------------------------------------------

def plot_residual_history(case_dir: Path, output_path: Path) -> bool:
    """Plot initial residuals for each solved field over iteration."""
    log_candidates = [
        case_dir / "log.foamRun",
        case_dir / "log.simpleFoam",
    ] + list(case_dir.glob("log.*Foam*")) + list(case_dir.glob("log.*foamRun*"))
    log_path = next((p for p in log_candidates if p.exists()), None)
    if log_path is None:
        log.info("no solver log found in %s", case_dir)
        return False

    df = load_residual_history(log_path)
    if df.empty or df.shape[1] <= 1:
        log.info("residual log %s has no parsed residuals", log_path)
        return False

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for col in df.columns:
        if col == "step":
            continue
        ax.semilogy(df["step"], df[col].clip(lower=1e-12), label=col, lw=1.2)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Initial residual")
    ax.set_title(f"Residual history — {case_dir.name}", fontsize=10)
    ax.grid(which="both", alpha=0.3)
    ax.legend(loc="best", fontsize=8, ncol=2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("wrote %s", output_path)
    return True


# ---------------------------------------------------------------------------
# Mesh convergence plot
# ---------------------------------------------------------------------------

def plot_mesh_convergence(
    mesh_records: list[dict],
    reference: dict | None,
    output_path: Path,
) -> None:
    """Plot Cl and Cd vs cell count for a mesh-refinement ladder.

    mesh_records: list of {"label": ..., "cells": int, "Cl": float, "Cd": float}.
    reference: {"label": ..., "Cl": float, "Cd": float} for the canonical value
               (drawn as a horizontal line).
    """
    if not mesh_records:
        log.warning("plot_mesh_convergence: empty mesh_records")
        return
    df = pd.DataFrame(mesh_records).sort_values("cells")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))

    axes[0].plot(df["cells"], df["Cl"], "o-", color=CFD_COLOR, lw=1.8, ms=7)
    axes[0].set_xlabel("Cells")
    axes[0].set_ylabel("Cl")
    axes[0].set_title("Cl vs grid")
    axes[0].set_xscale("log")
    axes[0].grid(which="both", alpha=0.3)

    axes[1].plot(df["cells"], df["Cd"], "o-", color=CFD_COLOR, lw=1.8, ms=7)
    axes[1].set_xlabel("Cells")
    axes[1].set_ylabel("Cd")
    axes[1].set_title("Cd vs grid")
    axes[1].set_xscale("log")
    axes[1].grid(which="both", alpha=0.3)

    if reference:
        if reference.get("Cl") is not None:
            axes[0].axhline(reference["Cl"], color=REF_COLOR, ls="--", lw=1.2,
                            label=reference.get("label", "reference"))
            axes[0].legend(loc="best", fontsize=8)
        if reference.get("Cd") is not None:
            axes[1].axhline(reference["Cd"], color=REF_COLOR, ls="--", lw=1.2,
                            label=reference.get("label", "reference"))
            axes[1].legend(loc="best", fontsize=8)

    for ax, row in zip(axes, ["Cl", "Cd"]):
        for _, r in df.iterrows():
            ax.annotate(r["label"], (r["cells"], r[row]),
                        xytext=(4, 6), textcoords="offset points",
                        fontsize=7, color="#555555")

    fig.suptitle("Mesh independence study", fontsize=11)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("wrote %s", output_path)


# ---------------------------------------------------------------------------
# High-level orchestration
# ---------------------------------------------------------------------------

def generate_regime_plots(
    regime: str,
    case_dirs: dict[str, Path],
    comparisons: list[dict],
    metadata: dict,
    mesh_records: list[dict] | None = None,
) -> dict[str, Path]:
    """Produce all regime-level plots and return paths written.

    Parameters
    ----------
    regime : 'A' | 'B' | 'C' | 'D'
    case_dirs : mapping case_id → OpenFOAM case directory
    comparisons : output of compare.compare_clcd() per case
    metadata : the regime's metadata.json dict
    mesh_records : optional list for mesh_convergence plot
    """
    figures_dir = VALIDATION_ROOT / f"regime_{regime.upper()}" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    written: dict[str, Path] = {}

    cl_cd_path = figures_dir / "cl_cd_comparison.png"
    plot_cl_cd_comparison(comparisons, cl_cd_path)
    written["cl_cd_comparison"] = cl_cd_path

    cp_paths = []
    for ref_case in metadata["validation_cases"]:
        cid = ref_case["case_id"]
        case_dir = case_dirs.get(cid)
        if case_dir is None:
            continue
        out = figures_dir / f"cp_comparison__{cid}.png"
        ok = plot_cp_comparison(case_dir, ref_case, out)
        if ok:
            cp_paths.append(out)
    written["cp_comparisons"] = cp_paths

    res_paths = []
    for cid, case_dir in case_dirs.items():
        out = figures_dir / f"residual_history__{cid}.png"
        ok = plot_residual_history(case_dir, out)
        if ok:
            res_paths.append(out)
    written["residual_histories"] = res_paths

    if mesh_records:
        # Use the first case's primary experimental reference as the "truth" line
        ref0 = metadata["validation_cases"][0].get("references", {}).get("experimental_primary")
        plot_mesh_convergence(
            mesh_records,
            ref0 if ref0 else None,
            figures_dir / "mesh_convergence.png",
        )
        written["mesh_convergence"] = figures_dir / "mesh_convergence.png"

    return written
