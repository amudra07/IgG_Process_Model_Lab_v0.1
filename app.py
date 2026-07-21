"""Thermicra high-concentration IgG formulation model — v0.3.1.

Decision-support prototype. The response surfaces are empirical and must not be
used as validated manufacturing specifications.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(
    page_title="Thermicra IgG Model v0.3.1",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)


COLORS = {
    "navy": "#122C3A",
    "teal": "#087E8B",
    "mint": "#DDF4EF",
    "amber": "#F4A261",
    "red": "#D1495B",
    "paper": "#F4F7F8",
    "ink": "#17313B",
}

FACTOR_RANGES = {
    "Trehalose": (0.0, 30.0),
    "Sucrose": (0.0, 30.0),
    "HPβCD": (0.0, 30.0),
    "PVP K12": (0.0, 25.0),
}

# Quadratic BBD surrogate in coded units. The interface intentionally keeps
# these implementation details out of view; the standalone report documents
# the model and its evidence base.
CONC_COEF = {
    "b0": 475.0,
    "linear": np.array([28.0, 46.0, 72.0, -18.0]),
    "square": np.array([-20.0, -34.0, -42.0, -24.0]),
    "interaction": {
        (0, 1): 18.0,
        (0, 2): 34.0,
        (0, 3): 10.0,
        (1, 2): 48.0,
        (1, 3): 8.0,
        (2, 3): 12.0,
    },
}

HMW_COEF = {
    "b0": 4.10,
    "linear": np.array([-0.18, -0.22, -0.28, -0.94]),
    "square": np.array([0.30, 0.38, 0.32, 0.20]),
    "interaction": {
        (0, 1): -0.12,
        (0, 2): -0.24,
        (0, 3): -0.08,
        (1, 2): -0.34,
        (1, 3): -0.10,
        (2, 3): -0.16,
    },
}

@dataclass(frozen=True)
class Prediction:
    concentration: float
    hmw: float


def coded_values(values: dict[str, float]) -> np.ndarray:
    result = []
    for factor, value in values.items():
        low, high = FACTOR_RANGES[factor]
        result.append(2.0 * (value - low) / (high - low) - 1.0)
    return np.asarray(result, dtype=float)


def quadratic(x: np.ndarray, coef: dict) -> np.ndarray:
    value = coef["b0"] + np.tensordot(coef["linear"], x, axes=(0, 0))
    value = value + np.tensordot(coef["square"], x**2, axes=(0, 0))
    for (i, j), beta in coef["interaction"].items():
        value = value + beta * x[i] * x[j]
    return value


def pvp_observed_hmw(pvp: float) -> float:
    return float(np.interp(pvp, [0, 15, 20, 25], [5.17, 3.03, 3.21, 1.66]))


def predict(values: dict[str, float]) -> Prediction:
    x = coded_values(values)
    concentration = float(np.clip(quadratic(x, CONC_COEF), 250, 650))
    hmw_bbd = float(quadratic(x, HMW_COEF))

    # v0.3.1 correction: blend the legacy BBD surface with the actual PVP study.
    # The evidence gets greater weight inside the tested 15–25 mg/mL interval.
    pvp = values["PVP K12"]
    evidence_weight = 0.65 if pvp >= 15 else 0.25
    hmw = float(np.clip((1 - evidence_weight) * hmw_bbd + evidence_weight * pvp_observed_hmw(pvp), 0.4, 9.0))
    return Prediction(
        concentration=concentration,
        hmw=hmw,
    )


def make_heatmap(
    x_factor: str,
    y_factor: str,
    fixed: dict[str, float],
    response: str,
) -> go.Figure:
    x_values = np.linspace(*FACTOR_RANGES[x_factor], 55)
    y_values = np.linspace(*FACTOR_RANGES[y_factor], 55)
    z = np.zeros((len(y_values), len(x_values)))
    for row, y_value in enumerate(y_values):
        for col, x_value in enumerate(x_values):
            values = dict(fixed)
            values[x_factor] = float(x_value)
            values[y_factor] = float(y_value)
            pred = predict(values)
            z[row, col] = pred.concentration if response == "concentration" else pred.hmw

    if response == "concentration":
        colorscale = [
            [0.00, "#13293D"], [0.18, "#146C94"], [0.38, "#19A7CE"],
            [0.55, "#A7D129"], [0.72, "#F4D35E"], [0.86, "#EE964B"], [1.00, "#C1121F"],
        ]
        title, unit, zmin, zmax = "Empirical achievable IgG concentration", "mg/mL", 250, 650
    else:
        colorscale = [
            [0.00, "#0B6E4F"], [0.24, "#69B578"], [0.47, "#F2E863"],
            [0.68, "#F4A261"], [0.84, "#E76F51"], [1.00, "#9B2226"],
        ]
        title, unit, zmin, zmax = "Predicted final HMW", "%", 1.0, 6.5

    fig = go.Figure(
        go.Contour(
            x=x_values,
            y=y_values,
            z=z,
            colorscale=colorscale,
            zmin=zmin,
            zmax=zmax,
            contours={"showlabels": True, "labelfont": {"size": 11, "color": "white"}},
            colorbar={"title": unit, "thickness": 14},
            hovertemplate=f"{x_factor}: %{{x:.1f}} mg/mL<br>{y_factor}: %{{y:.1f}} mg/mL<br>{title}: %{{z:.2f}} {unit}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[fixed[x_factor]],
            y=[fixed[y_factor]],
            mode="markers",
            marker={"size": 13, "color": "white", "line": {"width": 3, "color": COLORS["navy"]}},
            name="Current formulation",
            hovertemplate="Current formulation<extra></extra>",
            showlegend=False,
        )
    )
    fig.update_layout(
        title={"text": title, "x": 0.02, "xanchor": "left"},
        xaxis_title=f"{x_factor} (mg/mL feed)",
        yaxis_title=f"{y_factor} (mg/mL feed)",
        margin=dict(l=30, r=25, t=65, b=25),
        height=520,
        paper_bgcolor="white",
        plot_bgcolor="white",
        font={"family": "Arial, sans-serif", "color": COLORS["ink"]},
    )
    return fig


def metric_card(label: str, value: str, color: str = COLORS["teal"]) -> None:
    st.markdown(
        f"""
        <div class="metric-card" style="border-top-color:{color}">
          <div class="metric-label">{label}</div>
          <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


st.markdown(
    f"""
    <style>
      .stApp {{ background: {COLORS['paper']}; }}
      [data-testid="stSidebar"] {{ background: #EAF3F3; border-right:1px solid #D8E5E7; }}
      [data-testid="stSidebar"] h2 {{ color:{COLORS['navy']}; }}
      .hero {{ background: linear-gradient(120deg,{COLORS['navy']},#075F68); padding:24px 30px;
              border-radius:18px; color:white; margin-bottom:18px; box-shadow:0 12px 35px #17313b18; }}
      .hero h1 {{ color:white; margin:0 0 5px; font-size:2rem; }}
      .hero p {{ color:#D8ECEC; margin:0; }}
      .version {{ display:inline-block; background:#DDF4EF; color:#075F68; padding:4px 9px;
                  border-radius:999px; font-weight:800; font-size:.72rem; margin-bottom:10px; }}
      .metric-card {{ background:white; border-radius:14px; padding:17px 20px; min-height:112px;
                      border:1px solid #DCE7E9; border-top:5px solid; box-shadow:0 7px 20px #17313b0d; }}
      .metric-label {{ color:#60747C; font-size:.76rem; font-weight:800; letter-spacing:.06em; text-transform:uppercase; }}
      .metric-value {{ color:{COLORS['ink']}; font-size:2rem; font-weight:850; margin-top:8px; }}
      div[data-testid="stPlotlyChart"] {{ background:white; border:1px solid #DCE7E9;
                                         border-radius:15px; padding:5px; box-shadow:0 7px 20px #17313b0b; }}
      #MainMenu, footer {{ visibility:hidden; }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <div class="version">MODEL v0.3.1</div>
      <h1>High-Concentration IgG Formulation Model</h1>
      <p>Explore how four excipients influence achievable IgG concentration and aggregation.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Model inputs")
    st.caption("Excipient concentration in the aqueous feed")
    values = {
        "Trehalose": st.slider("Trehalose (mg/mL)", 0.0, 30.0, 15.0, 0.5),
        "Sucrose": st.slider("Sucrose (mg/mL)", 0.0, 30.0, 15.0, 0.5),
        "HPβCD": st.slider("HPβCD (mg/mL)", 0.0, 30.0, 15.0, 0.5),
        "PVP K12": st.slider("PVP K12 (mg/mL)", 0.0, 25.0, 20.0, 0.5),
    }
    st.divider()
    st.subheader("Heatmap axes")
    x_factor = st.selectbox("Horizontal axis", list(FACTOR_RANGES), index=1)
    y_options = [factor for factor in FACTOR_RANGES if factor != x_factor]
    y_factor = st.selectbox("Vertical axis", y_options, index=min(1, len(y_options) - 1))
    st.divider()
    st.caption("Other process variables are fixed at the current optimized conditions. Vehicle: MCT.")

prediction = predict(values)

metric_cols = st.columns(2)
with metric_cols[0]:
    metric_card("Predicted IgG concentration", f"{prediction.concentration:.0f} mg/mL", COLORS["teal"])
with metric_cols[1]:
    metric_card(
        "Predicted aggregation (HMW)",
        f"{prediction.hmw:.2f}%",
        COLORS["red"] if prediction.hmw > 5 else COLORS["teal"],
    )

st.markdown("### Response surfaces")
fixed = dict(values)
chart_cols = st.columns(2)
with chart_cols[0]:
    st.plotly_chart(make_heatmap(x_factor, y_factor, fixed, "concentration"), width="stretch")
with chart_cols[1]:
    st.plotly_chart(make_heatmap(x_factor, y_factor, fixed, "hmw"), width="stretch")

fixed_text = " · ".join(
    f"{factor}: {value:.1f} mg/mL" for factor, value in values.items() if factor not in {x_factor, y_factor}
)
st.caption(f"Fixed in this surface view — {fixed_text}")

st.divider()
st.caption(
    "Thermicra IgG model v0.3.1 · Research-use modelling prototype"
)
