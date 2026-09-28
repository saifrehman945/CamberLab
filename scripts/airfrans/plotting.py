"""
Module: scripts/airfrans/plotting.py
Purpose: One matplotlib style for every AirfRANS report figure.

Colours are the validated reference categorical palette (light mode). Only the
first three slots are safe together in scatter-type plots; four-family
comparisons use small multiples instead of four overlaid colours.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]   # blue, orange, aqua, yellow
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#8a8985"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"
REFERENCE = "#52514e"      # reference data (experiment / other CFD): neutral ink

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_SECONDARY, "axes.titlecolor": INK,
    "axes.titlesize": 11, "axes.labelsize": 9.5, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": INK_MUTED, "ytick.color": INK_MUTED,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "legend.frameon": False, "legend.fontsize": 8.5, "text.color": INK,
    "lines.linewidth": 2.0, "lines.markersize": 5, "savefig.dpi": 150,
    "savefig.bbox": "tight", "font.size": 9.5,
})


def save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path
