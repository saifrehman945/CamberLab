#!/usr/bin/env python3
"""
Module: scripts/validation/report.py
Purpose: Render a markdown validation report per regime, summarizing per-case
         Cl/Cd/Cp errors, mesh-convergence findings, and an overall pass/fail
         decision. Written to validation/regime_<X>/report.md.
"""

from __future__ import annotations

import datetime as _dt
import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)


def render_regime_report(
    regime: str,
    metadata: dict,
    comparisons: list[dict],
    tolerance_results: list[dict],
    figures: dict[str, Path | list[Path]],
    output_path: Path,
) -> Path:
    """Write a markdown report. Returns the path written."""
    lines: list[str] = []
    lines.append(f"# Regime {regime.upper()} Validation Report")
    lines.append("")
    lines.append(f"Generated: {_dt.datetime.utcnow().isoformat(timespec='seconds')} UTC")
    lines.append("")

    # Overview
    lines.append("## Overview")
    lines.append("")
    lines.append(f"- **Name:** {metadata['name']}")
    lines.append(f"- **Physics:** {metadata['physics']}")
    ds = metadata.get("design_space", {})
    lines.append(
        f"- **Design space:** α ∈ {ds.get('alpha_deg')}, "
        f"Re ∈ {ds.get('Re')}, t/c ∈ {ds.get('thickness')}"
    )
    rec = metadata.get("cfd_recipe", {})
    lines.append(
        f"- **CFD recipe:** {rec.get('turbulence_model')} / "
        f"{rec.get('wall_treatment')} (y+={rec.get('y_plus_target')})"
    )
    lines.append("")

    # Tolerance gate
    overall_pass = all(t["pass"] for t in tolerance_results) if tolerance_results else False
    lines.append("## Tolerance gate")
    lines.append("")
    tol = metadata.get("tolerances", {})
    lines.append(
        f"Acceptance: |ΔCl| ≤ {tol.get('delta_Cl_pct')}%, "
        f"|ΔCd| ≤ {tol.get('delta_Cd_pct')}% (vs experimental primary)."
    )
    lines.append("")
    lines.append(f"**Overall result: {'✅ PASS' if overall_pass else '❌ FAIL'}**")
    lines.append("")

    # Per-case results
    lines.append("## Per-case results")
    lines.append("")
    if not comparisons:
        lines.append("_No comparison data — cases were not run._")
    else:
        rows = []
        for comp, tres in zip(comparisons, tolerance_results):
            row = {
                "case_id":    comp["case_id"],
                "α_deg":      f"{comp['alpha_deg']:.2f}",
                "Re":         f"{comp['Re']:.2e}",
                "Cl_cfd":     f"{comp['Cl_cfd']:.4f}",
                "Cd_cfd":     f"{comp['Cd_cfd']:.5f}",
                "ΔCl%":       _fmt_pct(tres.get("Cl_err_pct")),
                "ΔCd%":       _fmt_pct(tres.get("Cd_err_pct")),
                "primary_ref": tres.get("primary_label", "—"),
                "pass":       "✅" if tres["pass"] else "❌",
            }
            rows.append(row)
        df = pd.DataFrame(rows)
        lines.append(df.to_markdown(index=False))
    lines.append("")

    # Figures
    lines.append("## Figures")
    lines.append("")
    if figures.get("cl_cd_comparison"):
        lines.append(f"- Cl/Cd comparison: `{_rel(figures['cl_cd_comparison'], output_path)}`")
    for cp_path in figures.get("cp_comparisons", []) or []:
        lines.append(f"- Cp overlay: `{_rel(cp_path, output_path)}`")
    for res_path in figures.get("residual_histories", []) or []:
        lines.append(f"- Residuals: `{_rel(res_path, output_path)}`")
    if figures.get("mesh_convergence"):
        lines.append(f"- Mesh convergence: `{_rel(figures['mesh_convergence'], output_path)}`")
    lines.append("")

    # Citations
    lines.append("## Citations")
    lines.append("")
    for cite in metadata.get("citations", []):
        lines.append(f"- {cite}")
    lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    log.info("wrote report → %s", output_path)
    return output_path


def _fmt_pct(value) -> str:
    if value is None or (isinstance(value, float) and (value != value)):
        return "—"
    try:
        return f"{value:+.2f}"
    except (TypeError, ValueError):
        return "—"


def _rel(p, anchor: Path) -> str:
    if isinstance(p, list):
        return ", ".join(_rel(item, anchor) for item in p)
    p = Path(p)
    try:
        return str(p.relative_to(anchor.parent))
    except ValueError:
        return str(p)
