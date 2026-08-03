from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.mesh.regime_parameters import classify_regime  # noqa: E402
from scripts.surrogate.data import REGIMES  # noqa: E402
from scripts.surrogate.inference import (  # noqa: E402
    FAMILIES,
    OutOfDistributionError,
    predict_blended_with_uncertainty,
    predict_untrained_regime_extrapolated,
    predict_with_uncertainty,
)

MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
REGIME_BOUNDS_PATH = MODELS_DIR / "regime_bounds.json"
DATASET_PATH = RESULTS_DIR / "dataset_clean.csv"
METRICS_PATH = RESULTS_DIR / "surrogate_metrics.csv"

FLOW_PHYSICS_LABELS = {
    "A": "Attached turbulent flow",
    "B": "Near-stall separated flow",
    "C": "Transitional low-Reynolds-number flow",
    "D": "High-Reynolds-number attached flow",
}

STATUS_COLORS = {
    "validated": "#2563eb",
    "unvalidated": "#b45309",
    "extrapolated": "#b91c1c",
    "blended": "#6d28d9",
    "skipped": "#64748b",
}


st.set_page_config(page_title="CamberLab ", layout="wide")


@st.cache_data(show_spinner=False)
def load_artifacts() -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    if not REGIME_BOUNDS_PATH.exists():
        st.error("models/regime_bounds.json not found. Run scripts/09_train_surrogates.py first.")
        st.stop()

    bounds = json.loads(REGIME_BOUNDS_PATH.read_text(encoding="utf-8"))
    dataset = pd.read_csv(DATASET_PATH) if DATASET_PATH.exists() else pd.DataFrame()
    metrics = pd.read_csv(METRICS_PATH) if METRICS_PATH.exists() else pd.DataFrame()
    return bounds, dataset, metrics


def naca4_coords(camber: float, camber_position: float, thickness: float, n: int = 220) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = np.linspace(0.0, 1.0, n)
    yt = 5.0 * thickness * (
        0.2969 * np.sqrt(x)
        - 0.1260 * x
        - 0.3516 * x**2
        + 0.2843 * x**3
        - 0.1036 * x**4
    )
    if camber <= 0.0 or camber_position <= 0.0:
        yc = np.zeros_like(x)
        dyc = np.zeros_like(x)
    else:
        p = camber_position
        yc = np.where(
            x < p,
            camber / p**2 * (2.0 * p * x - x**2),
            camber / (1.0 - p) ** 2 * ((1.0 - 2.0 * p) + 2.0 * p * x - x**2),
        )
        dyc = np.where(
            x < p,
            2.0 * camber / p**2 * (p - x),
            2.0 * camber / (1.0 - p) ** 2 * (p - x),
        )
    theta = np.arctan(dyc)
    xu = x - yt * np.sin(theta)
    yu = yc + yt * np.cos(theta)
    xl = x + yt * np.sin(theta)
    yl = yc - yt * np.cos(theta)
    xp = np.concatenate([xu[::-1], xl[1:]])
    yp = np.concatenate([yu[::-1], yl[1:]])
    return xp, yp, x, yc


def airfoil_figure(camber: float, camber_position: float, thickness: float, unsupported: bool) -> go.Figure:
    x, y, xc, yc = naca4_coords(camber, camber_position, thickness)
    line_color = "#b91c1c" if unsupported else "#1f77b4"
    fill_color = "rgba(185, 28, 28, 0.16)" if unsupported else "rgba(31, 119, 180, 0.18)"
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            fill="toself",
            mode="lines",
            line=dict(color=line_color, width=2),
            fillcolor=fill_color,
            hoverinfo="skip",
            showlegend=False,
        )
    )
    if unsupported:
        fig.add_trace(
            go.Scatter(
                x=xc,
                y=yc,
                mode="lines",
                line=dict(color="#b91c1c", width=1.5, dash="dash"),
                hoverinfo="skip",
                showlegend=False,
            )
        )
    fig.add_hline(y=0, line_dash="dot", line_color="#94a3b8", line_width=1)
    fig.update_layout(
        height=180,
        margin=dict(l=8, r=8, t=8, b=8),
        xaxis=dict(visible=False, range=[-0.03, 1.03], scaleanchor="y", scaleratio=1),
        yaxis=dict(visible=False, range=[-0.18, 0.18]),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def metric_value(metrics: pd.DataFrame, family: str, target: str, status: str) -> str:
    if metrics.empty:
        return "No validation metrics file."
    rows = metrics[
        (metrics["family"] == family)
        & (metrics["target"] == target)
        & (metrics["slice"] == "global")
    ]
    if rows.empty:
        return "No validation metric for this model."
    row = rows.iloc[0]
    return f"Global RMSE {row['RMSE']:.4g}, MAE {row['MAE']:.4g}; row status: {status}."


def physics_status(regime: str, bounds: dict) -> tuple[str, str]:
    if regime not in bounds:
        return "untrained", "No CFD training rows for this flow-physics range in the persisted model."
    if not bounds[regime].get("validated", True):
        return "unvalidated", "Trained from rows included despite failing the strict convergence gate."
    return "validated", "Trained from validated CFD rows."


def _single_target_prediction(
    alpha: float,
    re: float,
    thickness: float,
    family: str,
    target: str,
    mode_regime: str | None,
    bounds: dict,
    allow_untrained: bool,
) -> dict:
    resolved = mode_regime or classify_regime(alpha, re, thickness)
    try:
        if mode_regime is None:
            return predict_blended_with_uncertainty(alpha, re, thickness, family=family, target=target)
        return predict_with_uncertainty(
            alpha, re, thickness, family=family, target=target, regime=mode_regime
        )
    except OutOfDistributionError as exc:
        if allow_untrained and resolved not in bounds:
            return predict_untrained_regime_extrapolated(
                alpha, re, thickness, resolved, family=family, target=target
            )
        raise exc


def predict_point(
    alpha: float,
    re: float,
    thickness: float,
    family: str,
    mode_regime: str | None,
    bounds: dict,
    allow_untrained: bool,
) -> dict:
    resolved = mode_regime or classify_regime(alpha, re, thickness)
    cl = _single_target_prediction(alpha, re, thickness, family, "Cl", mode_regime, bounds, allow_untrained)
    cd = _single_target_prediction(alpha, re, thickness, family, "Cd", mode_regime, bounds, allow_untrained)
    value_cd = cd["value"]
    l_over_d = cl["value"] / value_cd if value_cd > 0 else np.nan
    if cl["std"] is not None and cd["std"] is not None and cl["value"] != 0 and value_cd > 0:
        l_over_d_std = abs(l_over_d) * np.sqrt((cl["std"] / cl["value"]) ** 2 + (cd["std"] / value_cd) ** 2)
    else:
        l_over_d_std = None
    status = "extrapolated" if "extrapolated" in {cl["status"], cd["status"]} else cl["status"]
    if status != "extrapolated" and cl.get("weights"):
        status = cl["status"]

    weights = cl.get("weights")
    if weights:
        regime_label = " + ".join(
            f"{FLOW_PHYSICS_LABELS[r]}:{w:.2f}" for r, w in sorted(weights.items())
        )
    else:
        regime_label = FLOW_PHYSICS_LABELS[resolved]

    return {
        "alpha_deg": alpha,
        "flow_physics": regime_label,
        "status": status,
        "Cl": cl["value"],
        "Cl_std": cl["std"],
        "Cd": value_cd,
        "Cd_std": cd["std"],
        "L_over_D": l_over_d,
        "L_over_D_std": l_over_d_std,
    }


def sweep_predictions(
    alphas: np.ndarray,
    re: float,
    thickness: float,
    family: str,
    mode_regime: str | None,
    bounds: dict,
    allow_untrained: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    skipped = []
    for alpha in alphas:
        try:
            rows.append(
                predict_point(
                    float(alpha), re, thickness, family, mode_regime, bounds, allow_untrained
                )
            )
        except OutOfDistributionError as exc:
            skipped.append(
                {
                    "alpha_deg": float(alpha),
                    "flow_physics": FLOW_PHYSICS_LABELS[
                        mode_regime or classify_regime(float(alpha), re, thickness)
                    ],
                    "status": "skipped",
                    "reason": str(exc),
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(skipped)


def add_prediction_trace(fig: go.Figure, df: pd.DataFrame, y_col: str, std_col: str, name: str) -> None:
    fig.add_trace(
        go.Scatter(
            x=df["alpha_deg"],
            y=df[y_col],
            mode="lines+markers",
            line=dict(color="#1f77b4", width=3),
            marker=dict(size=6),
            name=name,
            showlegend=True,
        )
    )
    if std_col in df and df[std_col].notna().any():
        upper = df[y_col] + 1.96 * df[std_col]
        lower = df[y_col] - 1.96 * df[std_col]
        fig.add_trace(
            go.Scatter(
                x=pd.concat([df["alpha_deg"], df["alpha_deg"][::-1]]),
                y=pd.concat([upper, lower[::-1]]),
                fill="toself",
                fillcolor="rgba(31, 119, 180, 0.20)",
                line=dict(color="rgba(255,255,255,0)"),
                hoverinfo="skip",
                name="95% uncertainty band",
                showlegend=True,
            )
        )


def style_plot(fig: go.Figure, x_title: str, y_title: str, title: str, height: int = 430) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, font=dict(size=17, color="#0f172a"), x=0.02, xanchor="left"),
        height=height,
        margin=dict(l=64, r=28, t=54, b=104),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.22,
            xanchor="center",
            x=0.5,
            bgcolor="#ffffff",
            bordercolor="#cbd5e1",
            borderwidth=1,
            font=dict(size=12, color="#1e293b"),
            itemwidth=30,
        ),
        font=dict(family="Arial", size=13, color="#0f172a"),
    )
    fig.update_xaxes(
        title=dict(text=x_title, font=dict(size=13, color="#1e293b")),
        tickfont=dict(size=12, color="#334155"),
        showgrid=True,
        gridcolor="#dbe3ef",
        linecolor="#94a3b8",
        zerolinecolor="#94a3b8",
    )
    fig.update_yaxes(
        title=dict(text=y_title, font=dict(size=13, color="#1e293b")),
        tickfont=dict(size=12, color="#334155"),
        showgrid=True,
        gridcolor="#dbe3ef",
        linecolor="#94a3b8",
        zerolinecolor="#94a3b8",
    )
    return fig


def add_status_shading(fig: go.Figure, df: pd.DataFrame) -> None:
    if df.empty or "status" not in df:
        return
    alpha = df["alpha_deg"].to_numpy(dtype=float)
    if len(alpha) < 2:
        pad = 0.25
    else:
        pad = float(np.nanmedian(np.diff(np.sort(alpha)))) / 2.0

    start_idx = 0
    statuses = df["status"].tolist()
    shaded_statuses: set[str] = set()
    for idx in range(1, len(df) + 1):
        if idx < len(df) and statuses[idx] == statuses[start_idx]:
            continue
        status = statuses[start_idx]
        if status != "validated":
            shaded_statuses.add(status)
            fig.add_vrect(
                x0=float(df.iloc[start_idx]["alpha_deg"]) - pad,
                x1=float(df.iloc[idx - 1]["alpha_deg"]) + pad,
                fillcolor=STATUS_COLORS.get(status, "#64748b"),
                opacity=0.11,
                layer="below",
                line_width=0,
            )
        start_idx = idx

    status_labels = {
        "unvalidated": "Low-confidence flow range",
        "extrapolated": "Untrained-flow extrapolation",
        "blended": "Blended trained flow ranges",
    }
    for status in sorted(shaded_statuses):
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(
                    size=10,
                    symbol="square",
                    color=STATUS_COLORS.get(status, "#64748b"),
                    opacity=0.72,
                ),
                name=status_labels.get(status, status),
                showlegend=True,
                hoverinfo="skip",
            )
        )


def curve_figure(df: pd.DataFrame, y_col: str, std_col: str, title: str, y_title: str) -> go.Figure:
    fig = go.Figure()
    add_status_shading(fig, df)
    add_prediction_trace(fig, df, y_col, std_col, "CFD surrogate")
    return style_plot(fig, "Angle of attack (deg)", y_title, title)


def polar_figure(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["Cd"],
            y=df["Cl"],
            mode="lines+markers",
            line=dict(color="#7c3aed", width=3),
            marker=dict(size=6),
            name="CFD surrogate",
            showlegend=True,
            text=df["alpha_deg"].map(lambda a: f"alpha={a:.2f} deg"),
            hovertemplate="%{text}<br>Cd=%{x:.5f}<br>Cl=%{y:.5f}<extra></extra>",
        )
    )
    return style_plot(fig, "Cd", "Cl", "Drag polar")


def overlay_training_points(fig: go.Figure, dataset: pd.DataFrame, re: float, thickness: float, y_col: str) -> None:
    if dataset.empty or y_col not in dataset:
        return
    if not {"Re", "thickness", "alpha_deg"}.issubset(dataset.columns):
        return
    re_band = max(0.08 * re, 75_000.0)
    t_band = 0.01
    nearby = dataset[
        (dataset["Re"].sub(re).abs() <= re_band)
        & (dataset["thickness"].sub(thickness).abs() <= t_band)
    ]
    if nearby.empty:
        return
    fig.add_trace(
        go.Scatter(
            x=nearby["alpha_deg"],
            y=nearby[y_col],
            mode="markers",
            marker=dict(size=7, color="#f59e0b", symbol="circle-open", line=dict(width=2)),
            name="nearby CFD rows",
            showlegend=True,
        )
    )


def status_summary(df: pd.DataFrame, skipped: pd.DataFrame) -> list[str]:
    """Terse notes only — the plots already shade which region is untrained."""
    notes = []
    if not df.empty and (df["status"] == "extrapolated").any():
        notes.append("The selected range has a part of untrained data.")
    if not df.empty and (df["status"] == "unvalidated").any():
        notes.append("The selected range has a part of unvalidated data.")
    if not skipped.empty:
        notes.append(f"{len(skipped)} sweep point(s) fell outside the trained envelope and were skipped.")
    return notes


def flow_physics_segments(
    alphas: np.ndarray,
    re: float,
    thickness: float,
    mode_regime: str | None,
    bounds: dict,
) -> pd.DataFrame:
    """Group the sweep into contiguous AoA bands, one row per flow-physics range."""
    labels = [mode_regime or classify_regime(float(a), re, thickness) for a in alphas]
    rows = []
    start = 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[start]:
            regime = labels[start]
            status, status_text = physics_status(regime, bounds)
            rows.append(
                {
                    "AoA band (deg)": f"{alphas[start]:.1f} – {alphas[i - 1]:.1f}",
                    "Flow physics": FLOW_PHYSICS_LABELS[regime],
                    "Training status": status,
                    "Notes": status_text,
                }
            )
            start = i
    return pd.DataFrame(rows)


bounds, dataset, metrics = load_artifacts()

st.title("CamberLab")
st.caption("Flow-physics-aware RANS surrogate for steady state symmetric NACA 4-digit airfoils.")

with st.sidebar:
    st.header("Inputs")
    camber_percent = st.slider("Camber, m/c (%)", 0, 6, 0, step=1)
    if camber_percent == 0:
        camber_position_tenths = st.slider("Camber position, p/c (tenths)", 0, 9, 0, step=1, disabled=True)
    else:
        camber_position_tenths = st.slider("Camber position, p/c (tenths)", 1, 9, 4, step=1)
    thickness_percent = st.slider("Thickness, t/c (%)", 8, 24, 12, step=1)
    camber = camber_percent / 100.0
    camber_position = camber_position_tenths / 10.0 if camber_position_tenths else 0.0
    thickness = thickness_percent / 100.0
    re = st.number_input("Reynolds number", min_value=300_000.0, max_value=5_000_000.0, value=2_000_000.0, step=100_000.0, format="%.0f")
    alpha_min, alpha_max = st.slider("AoA sweep (deg)", 0.0, 16.0, (0.0, 16.0), step=0.5)
    n_points = st.slider("Sweep points", 21, 161, 61, step=10)
    family = st.selectbox("Model family", FAMILIES, index=FAMILIES.index("gp"))
    mode_options = {"Auto classification": None, **{FLOW_PHYSICS_LABELS[r]: r for r in REGIMES}}
    mode_label = st.selectbox("Flow physics mode", list(mode_options), index=0)
    mode_regime = mode_options[mode_label]
    allow_untrained = st.checkbox("Allow untrained-flow extrapolation", value=True)

    camber_unsupported = camber_percent != 0 or camber_position_tenths != 0
    naca_code = f"{camber_percent}{camber_position_tenths}{thickness_percent:02d}"
    st.subheader(f"NACA {naca_code}")
    st.plotly_chart(
        airfoil_figure(camber, camber_position, thickness, camber_unsupported),
        use_container_width=True,
        config={"displayModeBar": False},
    )
    if camber_unsupported:
        st.markdown(
            """
            <div style="border:1px solid #b91c1c;background:#fef2f2;color:#7f1d1d;
                        padding:10px 12px;border-radius:6px;font-size:0.9rem;">
            Camber is not supported by the trained surrogate. Predictions below still use
            the symmetric NACA 00xx model at the selected thickness.
            </div>
            """,
            unsafe_allow_html=True,
        )

alphas = np.linspace(alpha_min, alpha_max, n_points)

if camber_unsupported:
    st.error(
        "Cambered NACA geometry is shown only as an unsupported preview. "
        f"The surrogate prediction uses symmetric NACA 00{thickness_percent:02d}."
    )

pred_df, skipped_df = sweep_predictions(
    alphas, re, thickness, family, mode_regime, bounds, allow_untrained
)

if pred_df.empty:
    st.error("No sweep points are inside the current app prediction policy.")
    if not skipped_df.empty:
        st.dataframe(skipped_df, use_container_width=True)
    st.stop()

ld_clean = pred_df["L_over_D"].replace([np.inf, -np.inf], np.nan)
best_idx = ld_clean.idxmax() if ld_clean.notna().any() else pred_df.index[0]
best = pred_df.loc[best_idx]
max_cl = pred_df.loc[pred_df["Cl"].idxmax()]
min_cd = pred_df.loc[pred_df["Cd"].idxmin()]

# The sweep itself is the query, so the KPIs summarise the whole curve.
sweep_status = (
    "extrapolated"
    if (pred_df["status"] == "extrapolated").any()
    else "unvalidated"
    if (pred_df["status"] == "unvalidated").any()
    else str(pred_df["status"].mode().iat[0])
)

def kpi_note(row: pd.Series) -> str | None:
    """Flag a KPI whose winning sweep point is not CFD-backed."""
    if row["status"] == "extrapolated":
        return "extrapolated"
    if row["status"] == "unvalidated":
        return "unvalidated"
    return None


kpi = st.columns(5)
kpi[0].metric("Best L/D", f"{best['L_over_D']:.1f}", delta=kpi_note(best), delta_color="off")
kpi[1].metric("AoA at best L/D", f"{best['alpha_deg']:.1f} deg")
kpi[2].metric("Max Cl", f"{max_cl['Cl']:.4f}", delta=kpi_note(max_cl), delta_color="off")
kpi[3].metric("AoA at max Cl", f"{max_cl['alpha_deg']:.1f} deg")
kpi[4].metric("Min Cd", f"{min_cd['Cd']:.5f}", delta=kpi_note(min_cd), delta_color="off")

for note in status_summary(pred_df, skipped_df):
    st.warning(note)

plot_col_1, plot_col_2 = st.columns(2)
fig_cl = curve_figure(pred_df, "Cl", "Cl_std", "Lift curve", "Cl")
overlay_training_points(fig_cl, dataset, re, thickness, "Cl_mean")
plot_col_1.plotly_chart(fig_cl, use_container_width=True)

fig_cd = curve_figure(pred_df, "Cd", "Cd_std", "Drag curve", "Cd")
overlay_training_points(fig_cd, dataset, re, thickness, "Cd_mean")
plot_col_2.plotly_chart(fig_cd, use_container_width=True)

plot_col_3, plot_col_4 = st.columns(2)
plot_col_3.plotly_chart(curve_figure(pred_df, "L_over_D", "L_over_D_std", "Efficiency curve", "L/D"), use_container_width=True)
plot_col_4.plotly_chart(polar_figure(pred_df), use_container_width=True)

st.subheader("Flow physics across the AoA sweep")
st.dataframe(
    flow_physics_segments(alphas, re, thickness, mode_regime, bounds),
    use_container_width=True,
    hide_index=True,
)

st.subheader("Model Error Context")
err_cols = st.columns(2)
err_cols[0].write(metric_value(metrics, family, "Cl", sweep_status))
err_cols[1].write(metric_value(metrics, family, "Cd", sweep_status))

st.subheader("Prediction Table")
table_cols = ["alpha_deg", "flow_physics", "status", "Cl", "Cl_std", "Cd", "Cd_std", "L_over_D", "L_over_D_std"]
display_df = pred_df[table_cols].copy()
for col in ["Cl", "Cl_std", "Cd", "Cd_std", "L_over_D", "L_over_D_std"]:
    display_df[col] = display_df[col].astype(float).round(6)
st.dataframe(display_df, use_container_width=True)

csv = display_df.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download predictions CSV",
    data=csv,
    file_name=f"naca_00{thickness_percent:02d}_{family}_predictions.csv",
    mime="text/csv",
)

if not skipped_df.empty:
    with st.expander("Skipped sweep points"):
        st.dataframe(skipped_df, use_container_width=True)
