"""
CamberLab — Streamlit front end for the AirfRANS coefficient surrogate.

Usage:
    uv run streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.airfrans.geometry import naca_coordinates, parse_naca  # noqa: E402
from scripts.surrogate.inference import (  # noqa: E402
    curve_quantities,
    has_curve_models,
    load_envelope,
    naca_features,
    predict_curve,
    predict_surface,
)
from scripts.surrogate.models import FAMILIES, FAMILY_LABELS  # noqa: E402

RESULTS_DIR = PROJECT_ROOT / "results"
METRICS_PATH = RESULTS_DIR / "airfrans_metrics.csv"
TASK = "full"

SERIES = "#2a78d6"
SERIES_BAND = "rgba(42, 120, 214, 0.16)"
TRAINING = "#eb6834"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e4e3df"

st.set_page_config(page_title="CamberLab", layout="wide")


@st.cache_data(show_spinner=False)
def load_metrics() -> pd.DataFrame:
    return pd.read_csv(METRICS_PATH) if METRICS_PATH.exists() else pd.DataFrame()


def airfoil_figure(naca: str) -> go.Figure:
    xy = naca_coordinates(naca, n=301)
    xy = np.vstack([xy, xy[:1]])
    fig = go.Figure(go.Scatter(x=xy[:, 0], y=xy[:, 1], fill="toself", mode="lines",
                               line=dict(color=SERIES, width=2), fillcolor=SERIES_BAND,
                               hoverinfo="skip", showlegend=False))
    fig.add_hline(y=0, line_dash="dot", line_color="#8a8985", line_width=1)
    fig.update_layout(height=170, margin=dict(l=8, r=8, t=8, b=8),
                      xaxis=dict(visible=False, range=[-0.03, 1.03], scaleanchor="y", scaleratio=1),
                      yaxis=dict(visible=False), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)")
    return fig


def style_plot(fig: go.Figure, x_title: str, y_title: str, title: str, height: int = 400) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color=INK), x=0.02, xanchor="left"),
        height=height, margin=dict(l=60, r=24, t=50, b=90),
        paper_bgcolor="#ffffff", plot_bgcolor="#ffffff", hovermode="x unified",
        legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5,
                    font=dict(size=12, color=INK_SECONDARY)),
        font=dict(size=13, color=INK),
    )
    for upd in (fig.update_xaxes, fig.update_yaxes):
        upd(showgrid=True, gridcolor=GRID, linecolor=GRID, zerolinecolor="#c9c8c3",
            tickfont=dict(size=12, color=INK_SECONDARY))
    fig.update_xaxes(title=dict(text=x_title, font=dict(size=13, color=INK_SECONDARY)))
    fig.update_yaxes(title=dict(text=y_title, font=dict(size=13, color=INK_SECONDARY)))
    return fig


def curve_figure(df: pd.DataFrame, y: str, title: str, y_title: str, family: str) -> go.Figure:
    fig = go.Figure()
    lo, hi = f"{y}_lo", f"{y}_hi"
    if lo in df:
        fig.add_trace(go.Scatter(x=pd.concat([df.alpha_deg, df.alpha_deg[::-1]]),
                                 y=pd.concat([df[hi], df[lo][::-1]]), fill="toself",
                                 fillcolor=SERIES_BAND, line=dict(width=0), hoverinfo="skip",
                                 name="±2σ band"))
    fig.add_trace(go.Scatter(x=df.alpha_deg, y=df[y], mode="lines", line=dict(color=SERIES, width=2.5),
                             name=f"{FAMILY_LABELS[family]} surrogate",
                             hovertemplate=f"α %{{x:.2f}}°<br>{y_title} %{{y:.5g}}<extra></extra>"))
    return style_plot(fig, "Angle of attack (deg)", y_title, title)


def polar_figure(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Scatter(x=df.Cd, y=df.Cl, mode="lines", line=dict(color=SERIES, width=2.5),
                               name="drag polar", customdata=df.alpha_deg,
                               hovertemplate="α %{customdata:.2f}°<br>Cd %{x:.5f}<br>Cl %{y:.4f}<extra></extra>"))
    i = int(np.argmax(df.L_over_D))
    fig.add_trace(go.Scatter(x=[df.Cd.iloc[i]], y=[df.Cl.iloc[i]], mode="markers",
                             marker=dict(size=11, color=TRAINING), name=f"max L/D ({df.L_over_D.iloc[i]:.0f})",
                             hoverinfo="skip"))
    fig.update_layout(hovermode="closest")
    return style_plot(fig, "Cd", "Cl", "Drag polar")


def surface_figure(surf: pd.DataFrame, q: str, alpha: float) -> go.Figure:
    """Cp or Cf against x/c, upper solid and lower dashed, with ±2σ bands where available."""
    scale = 1e3 if q == "Cf" else 1.0
    fig = go.Figure()
    for side, dash in (("upper", "solid"), ("lower", "dash")):
        s = surf[surf.surface == side]
        if f"{q}_lo" in s:
            fig.add_trace(go.Scatter(x=pd.concat([s.x_c, s.x_c[::-1]]),
                                     y=pd.concat([s[f"{q}_hi"], s[f"{q}_lo"][::-1]]) * scale,
                                     fill="toself", fillcolor=SERIES_BAND, line=dict(width=0),
                                     hoverinfo="skip", name="±2σ band", showlegend=side == "upper"))
        fig.add_trace(go.Scatter(x=s.x_c, y=s[q] * scale, mode="lines", name=f"{side} surface",
                                 line=dict(color=SERIES if side == "upper" else TRAINING, width=2.5, dash=dash),
                                 hovertemplate=f"x/c %{{x:.3f}}<br>{q} %{{y:.4g}}<extra>{side}</extra>"))
    if q == "Cp":
        fig.update_yaxes(autorange="reversed")
        title, y_title = f"Pressure coefficient at α = {alpha:g}°", "Cp"
    else:
        fig.add_hline(y=0, line_color="#8a8985", line_width=1)
        title, y_title = f"Skin friction at α = {alpha:g}° (Cf < 0: reversed flow)", "Cf × 10³"
    return style_plot(fig, "x/c", y_title, title)


def model_error_note(metrics: pd.DataFrame, family: str) -> list[str]:
    if metrics.empty:
        return ["No evaluation metrics file — run scripts/10_global_validation.py."]
    out = []
    for target in ("Cl", "Cd"):
        m = metrics[(metrics.task == TASK) & (metrics.family == family) & (metrics.target == target)]
        if len(m):
            r = m.iloc[0]
            out.append(f"**{target}** on the {int(r.n_test)} held-out AirfRANS test cases: "
                       f"R² {r.R2:.4f}, RMSE {r.RMSE:.3g}, median relative error {100 * r.rel_err_median:.2f}%.")
    return out


# --------------------------------------------------------------------------
# Page
# --------------------------------------------------------------------------

envelope = load_envelope(TASK)["envelope"]
metrics = load_metrics()

st.title("CamberLab")
st.caption("Lift, drag and surface pressure / skin friction of NACA 4- and 5-digit aerofoils, "
           "predicted in milliseconds by surrogates "
           "trained on 800 AirfRANS steady RANS (k-ω SST) simulations.")

with st.sidebar:
    st.header("Inputs")
    naca = st.text_input("NACA code (4 or 5 digits)", value="2412", max_chars=9).strip()
    re = st.slider("Reynolds number", min_value=2.0e6, max_value=6.0e6, value=3.0e6, step=1.0e5, format="%.1e")
    alpha_min, alpha_max = st.slider("AoA sweep (deg)", -5.0, 15.0, (-5.0, 15.0), step=0.5)
    n_points = st.slider("Sweep points", 21, 161, 81, step=10)
    surface_alpha = st.slider("AoA for surface curves (deg)", -5.0, 15.0, 4.0, step=0.5)
    family = st.selectbox("Model family", FAMILIES, index=FAMILIES.index("gp"),
                          format_func=lambda f: {"gp": "GP — Gaussian process", "rf": "RF — random forest",
                                                 "mlp": "MLP — neural network", "krg": "KRG — Kriging (SMT)"}[f])
    try:
        code = parse_naca(naca)["code"]
        valid = True
    except ValueError as exc:
        valid = False
        st.error(str(exc))
    if valid:
        try:
            geo = naca_features(code)
            st.subheader(f"NACA {code}")
            st.plotly_chart(airfoil_figure(code), width="stretch", config={"displayModeBar": False})
            st.caption(f"t/c {geo['t_max']:.3f} at {geo['x_tmax']:.2f}c · camber {geo['m_max']:.4f} "
                       f"at {geo['x_m']:.2f}c (geometric chord)")
        except ValueError as exc:
            valid = False
            st.error(str(exc))

if not valid:
    st.info("Enter a valid NACA 4-digit (e.g. 0012, 2412) or 5-digit (e.g. 23012, 23112) code.")
    st.stop()

alphas = np.linspace(alpha_min, alpha_max, n_points)
df = predict_curve(alphas, re, code, family, TASK)
warnings = df.attrs["warnings"]
outside = [w for w in warnings if "outside" in w]
if outside:
    st.warning("**Outside the training envelope — these predictions are extrapolations.**\n\n"
               + "\n".join(f"- {w}" for w in outside))
for w in warnings:
    if w not in outside:
        st.info(w)

best = df.loc[df.L_over_D.idxmax()]
kpi = st.columns(5)
kpi[0].metric("Best L/D", f"{best.L_over_D:.1f}")
kpi[1].metric("AoA at best L/D", f"{best.alpha_deg:.1f}°")
kpi[2].metric("Max Cl in sweep", f"{df.Cl.max():.3f}")
kpi[3].metric("Min Cd in sweep", f"{df.Cd.min():.5f}")
zero = np.interp(0.0, df.Cl, df.alpha_deg) if df.Cl.min() < 0 < df.Cl.max() else np.nan
kpi[4].metric("Zero-lift α", "—" if np.isnan(zero) else f"{zero:.2f}°")

c1, c2 = st.columns(2)
c1.plotly_chart(curve_figure(df, "Cl", "Lift curve", "Cl", family), width="stretch")
c2.plotly_chart(curve_figure(df, "Cd", "Drag curve", "Cd", family), width="stretch")
c3, c4 = st.columns(2)
c3.plotly_chart(polar_figure(df), width="stretch")
c4.plotly_chart(curve_figure(df, "L_over_D", "Efficiency", "L/D", family), width="stretch")
quantities = curve_quantities(TASK)
if quantities and not has_curve_models(family, TASK):
    st.caption(f"Surface Cp / Cf curves are not bundled for {FAMILY_LABELS[family]} (only GP and MLP "
               "curve models are committed); train them with `scripts/09_train_surrogates.py "
               "--outputs curves`.")
elif quantities:
    surf = predict_surface(surface_alpha, re, code, family, TASK)
    cols = st.columns(len(quantities))
    for col, q in zip(cols, quantities):
        col.plotly_chart(surface_figure(surf, q, surface_alpha), width="stretch")
    if "Cf" not in quantities:
        st.caption("Skin friction is not shown: the computed wall shear stress did not reproduce "
                   "AirfRANS' drag closely enough (results/airfrans_surface_gate.md).")
if family not in ("gp", "krg"):
    st.caption("Uncertainty bands are shown for the GP and KRG families only.")

st.subheader("Model error context")
for line in model_error_note(metrics, family):
    st.markdown(line)
st.caption(f"Training envelope: α {envelope['alpha_deg'][0]:.1f}° to {envelope['alpha_deg'][1]:.1f}°, "
           f"Re {envelope['Re'][0] / 1e6:.2f}–{envelope['Re'][1] / 1e6:.2f}×10⁶, "
           f"t/c {envelope['t_max'][0]:.3f}–{envelope['t_max'][1]:.3f}, "
           f"camber ≤ {envelope['m_max'][1]:.3f}. Fully turbulent SST: no laminar-turbulent "
           "transition; steady RANS is least reliable near stall.")

st.subheader("Prediction table")
cols = [c for c in ["alpha_deg", "Cl", "Cl_std", "Cd", "Cd_std", "L_over_D"] if c in df]
table = df[cols].astype(float).round(6)
st.dataframe(table, width="stretch", hide_index=True)
st.download_button("Download predictions CSV", data=table.to_csv(index=False).encode("utf-8"),
                   file_name=f"naca{code}_Re{re:.2e}_{family}.csv", mime="text/csv")

st.divider()
st.caption("Data: AirfRANS (Bonnet et al., NeurIPS 2022 Datasets & Benchmarks, arXiv:2212.07564), "
           "via PLAID-datasets/AirfRANS_clipped, © Safran, licensed under the Open Database "
           "License (ODbL 1.0).")
