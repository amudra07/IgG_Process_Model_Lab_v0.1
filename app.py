"""Thermicra high-concentration IgG formulation model — v0.3.1.

Decision-support prototype. The response surfaces are empirical and must not be
used as validated manufacturing specifications.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
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

# Quadratic BBD surrogate in coded units. These coefficients preserve the v0.3
# interaction architecture. They are deliberately visible in the application.
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

PVP_EVIDENCE = pd.DataFrame(
    {
        "PVP K12 (mg/mL)": [0, 15, 20, 25],
        "Theoretical IgG (%)": [np.nan, 62, 58, 55],
        "Pre-milling assay (%)": [np.nan, 93.96, 93.14, 96.23],
        "Post-milling assay (%)": [np.nan, 70.95, 80.79, 59.06],
        "Milling retention (%)": [np.nan, 75.51, 86.74, 61.37],
        "HMW (%)": [5.17, 3.03, 3.21, 1.66],
        "Monomer (%)": [94.83, 94.33, 94.52, 94.81],
        "Water reconstitution": ["—", "Insoluble", "Insoluble", "Insoluble"],
    }
)

MCT_ANCHORS = pd.DataFrame(
    {
        "Composition": ["Sucrose", "Sucrose + HPβCD"],
        "Confirmed MCT loading (mg/mL)": [400, 550],
        "23G injection": ["Pass", "Pass"],
        "HMW (%)": [5.02, 4.97],
        "Monomer (%)": [91.52, 95.03],
        "Bulk density (g/mL)": [0.30, 0.28],
        "Tapped density (g/mL)": [0.50, 0.46],
    }
)


@dataclass(frozen=True)
class Prediction:
    concentration: float
    hmw: float
    monomer: float
    milling_retention: float | None
    evidence_status: str
    evidence_color: str
    dry_igg_fraction: float


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


def milling_retention(pvp: float) -> float | None:
    if pvp < 15:
        return None
    return float(np.interp(pvp, [15, 20, 25], [75.51, 86.74, 61.37]))


def predict(values: dict[str, float], feed_igg: float) -> Prediction:
    x = coded_values(values)
    concentration = float(np.clip(quadratic(x, CONC_COEF), 250, 650))
    hmw_bbd = float(quadratic(x, HMW_COEF))

    # v0.3.1 correction: blend the legacy BBD surface with the actual PVP study.
    # The evidence gets greater weight inside the tested 15–25 mg/mL interval.
    pvp = values["PVP K12"]
    evidence_weight = 0.65 if pvp >= 15 else 0.25
    hmw = float(np.clip((1 - evidence_weight) * hmw_bbd + evidence_weight * pvp_observed_hmw(pvp), 0.4, 9.0))
    monomer = float(np.clip(99.7 - hmw - 0.18, 88.0, 99.0))

    excipient_total = sum(values.values())
    dry_igg_fraction = feed_igg / max(feed_igg + excipient_total, 1e-9)
    combo_support = values["Sucrose"] >= 7.5 and values["HPβCD"] >= 7.5

    if concentration <= 400:
        status, color = "Confirmed domain", COLORS["teal"]
    elif concentration <= 550 and combo_support:
        status, color = "Supported boundary", COLORS["amber"]
    elif concentration <= 550:
        status, color = "Composition-dependent", COLORS["amber"]
    else:
        status, color = "Extrapolated above MCT evidence", COLORS["red"]

    return Prediction(
        concentration=concentration,
        hmw=hmw,
        monomer=monomer,
        milling_retention=milling_retention(pvp),
        evidence_status=status,
        evidence_color=color,
        dry_igg_fraction=dry_igg_fraction,
    )


def make_heatmap(
    x_factor: str,
    y_factor: str,
    fixed: dict[str, float],
    response: str,
    feed_igg: float,
) -> go.Figure:
    x_values = np.linspace(*FACTOR_RANGES[x_factor], 55)
    y_values = np.linspace(*FACTOR_RANGES[y_factor], 55)
    z = np.zeros((len(y_values), len(x_values)))
    for row, y_value in enumerate(y_values):
        for col, x_value in enumerate(x_values):
            values = dict(fixed)
            values[x_factor] = float(x_value)
            values[y_factor] = float(y_value)
            pred = predict(values, feed_igg)
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


def metric_card(label: str, value: str, note: str, color: str = COLORS["teal"]) -> None:
    st.markdown(
        f"""
        <div class="metric-card" style="border-top-color:{color}">
          <div class="metric-label">{label}</div>
          <div class="metric-value">{value}</div>
          <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


st.markdown(
    f"""
    <style>
      .stApp {{ background: {COLORS['paper']}; }}
      [data-testid="stSidebar"] {{ background: #EAF3F3; }}
      .hero {{ background: linear-gradient(120deg,{COLORS['navy']},#075F68); padding:30px 34px;
              border-radius:20px; color:white; margin-bottom:20px; box-shadow:0 12px 35px #17313b18; }}
      .hero h1 {{ color:white; margin:0 0 6px; font-size:2.25rem; }}
      .hero p {{ color:#D8ECEC; margin:0; max-width:920px; }}
      .version {{ display:inline-block; background:#F4A261; color:#122C3A; padding:5px 10px;
                  border-radius:999px; font-weight:800; font-size:.78rem; margin-bottom:12px; }}
      .metric-card {{ background:white; border-radius:14px; padding:18px 19px; min-height:142px;
                      border:1px solid #DCE7E9; border-top:5px solid; box-shadow:0 7px 20px #17313b0d; }}
      .metric-label {{ color:#60747C; font-size:.76rem; font-weight:800; letter-spacing:.06em; text-transform:uppercase; }}
      .metric-value {{ color:{COLORS['ink']}; font-size:1.75rem; font-weight:850; margin:8px 0 5px; }}
      .metric-note {{ color:#60747C; font-size:.78rem; line-height:1.35; }}
      .evidence {{ background:#FFF6E9; border-left:5px solid {COLORS['amber']}; padding:14px 17px;
                   border-radius:10px; margin:10px 0 18px; color:{COLORS['ink']}; }}
      .equation {{ background:{COLORS['navy']}; color:#EAF7F6; border-radius:12px; padding:18px 20px;
                   font-family:Georgia,serif; font-size:1.05rem; margin:8px 0; }}
      .small-note {{ color:#65777E; font-size:.82rem; }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <div class="version">MODEL v0.3.1 · DATA-SUPPORTED CORRECTION</div>
      <h1>High-Concentration IgG Formulation Model</h1>
      <p>Box–Behnken response surfaces for four excipients, now constrained by supplied PVP K12,
      milling-recovery, MCT loading, density and 23G injectability evidence.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Formulation controls")
    st.caption("All excipient values are aqueous-feed concentrations.")
    values = {
        "Trehalose": st.slider("Trehalose (mg/mL)", 0.0, 30.0, 15.0, 0.5),
        "Sucrose": st.slider("Sucrose (mg/mL)", 0.0, 30.0, 15.0, 0.5),
        "HPβCD": st.slider("HPβCD (mg/mL)", 0.0, 30.0, 15.0, 0.5),
        "PVP K12": st.slider("PVP K12 (mg/mL)", 0.0, 25.0, 20.0, 0.5),
    }
    feed_igg = st.number_input("IgG feed (mg/mL)", 20.0, 200.0, 100.0, 5.0)
    st.divider()
    st.subheader("Response-surface axes")
    x_factor = st.selectbox("Horizontal axis", list(FACTOR_RANGES), index=1)
    y_options = [factor for factor in FACTOR_RANGES if factor != x_factor]
    y_factor = st.selectbox("Vertical axis", y_options, index=min(1, len(y_options) - 1))
    st.divider()
    st.caption("Fixed: MCT vehicle · spray flow 40 rpm · 23G evidence context · current optimized process conditions")

prediction = predict(values, feed_igg)

tabs = st.tabs(["Response surfaces", "Verification layer", "Equations & scope", "v0.3.1 update"])

with tabs[0]:
    metric_cols = st.columns(5)
    with metric_cols[0]:
        metric_card("Empirical concentration", f"{prediction.concentration:.0f} mg/mL", "Legacy v0.3 output retained", COLORS["teal"])
    with metric_cols[1]:
        metric_card("Predicted HMW", f"{prediction.hmw:.2f}%", "PVP evidence-corrected", COLORS["red"] if prediction.hmw > 5 else COLORS["teal"])
    with metric_cols[2]:
        metric_card("Estimated monomer", f"{prediction.monomer:.2f}%", "Derived; LMW not directly fitted")
    with metric_cols[3]:
        recovery_text = "Not fitted" if prediction.milling_retention is None else f"{prediction.milling_retention:.1f}%"
        metric_card("Milling retention", recovery_text, "Observed interpolation for PVP 15–25 mg/mL", COLORS["amber"])
    with metric_cols[4]:
        metric_card("MCT evidence", prediction.evidence_status, "Not an exact maximum", prediction.evidence_color)

    st.markdown(
        "<div class='evidence'><b>How to read v0.3.1:</b> concentration is still the empirical v0.3 BBD prediction. "
        "The evidence card indicates whether that value is directly supported, near the supplied MCT boundary, or extrapolated. "
        "Physical agglomeration is not mislabeled as SEC-HPLC molecular aggregation.</div>",
        unsafe_allow_html=True,
    )

    fixed = dict(values)
    chart_cols = st.columns(2)
    with chart_cols[0]:
        st.plotly_chart(make_heatmap(x_factor, y_factor, fixed, "concentration", feed_igg), use_container_width=True)
    with chart_cols[1]:
        st.plotly_chart(make_heatmap(x_factor, y_factor, fixed, "hmw", feed_igg), use_container_width=True)

    fixed_text = ", ".join(
        f"{factor} = {value:.1f} mg/mL" for factor, value in values.items() if factor not in {x_factor, y_factor}
    )
    st.caption(f"Surface slice: {fixed_text}. Contours are predictions inside a prototype design space, not confidence bounds.")

with tabs[1]:
    st.subheader("Actual-study verification data")
    st.write("The supplied studies are retained as stage-specific anchors instead of being forced into an unsupported end-to-end fit.")
    left, right = st.columns([1.2, 1])
    with left:
        st.markdown("#### PVP K12: protection–recovery trade-off")
        display_pvp = PVP_EVIDENCE.copy()
        st.dataframe(display_pvp, hide_index=True, use_container_width=True)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[0, 15, 20, 25], y=[5.17, 3.03, 3.21, 1.66], mode="lines+markers", name="HMW (%)", line={"color": COLORS["red"], "width": 3}))
        fig.add_trace(go.Scatter(x=[15, 20, 25], y=[75.51, 86.74, 61.37], mode="lines+markers", name="Milling retention (%)", yaxis="y2", line={"color": COLORS["teal"], "width": 3}))
        fig.update_layout(
            height=370,
            xaxis_title="PVP K12 (mg/mL)",
            yaxis={"title": "HMW (%)", "range": [0, 6]},
            yaxis2={"title": "Milling retention (%)", "overlaying": "y", "side": "right", "range": [50, 100]},
            legend={"orientation": "h", "y": 1.15},
            margin=dict(l=30, r=30, t=55, b=30),
        )
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.markdown("#### MCT-specific loading anchors")
        st.dataframe(MCT_ANCHORS, hide_index=True, use_container_width=True)
        st.info(
            "400 mg/mL sucrose and 550 mg/mL sucrose + HPβCD are successful tested points. "
            "They are lower-bound evidence (Cmax ≥ tested value), not exact capacity limits."
        )
        st.markdown("#### Current formulation flags")
        if values["PVP K12"] >= 15:
            st.warning("PVP study domain: reduced HMW is accompanied by insoluble water reconstitution in all tested PVP conditions.")
        if values["PVP K12"] >= 22.5:
            st.error("High-PVP trade-off: the 25 mg/mL anchor had only 61.4% relative milling retention despite the lowest HMW.")
        if prediction.concentration > 550:
            st.error("The predicted concentration exceeds the highest supplied successful MCT anchor (550 mg/mL). Treat it as extrapolation.")
        else:
            st.success("The prediction remains at or below the highest supplied successful MCT anchor; composition matching still matters.")

with tabs[2]:
    st.subheader("Mathematical structure retained from v0.3")
    st.markdown("<div class='equation'>Ŷ = β₀ + Σβᵢxᵢ + Σβᵢᵢxᵢ² + Σβᵢⱼxᵢxⱼ</div>", unsafe_allow_html=True)
    st.write("The same four-factor quadratic Box–Behnken structure is used for both response surfaces. Factors are coded from −1 to +1.")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Empirical concentration equation")
        st.code("C_emp = 475 + 28T + 46S + 72H − 18P − 20T² − 34S² − 42H² − 24P² + 18TS + 34TH + 10TP + 48SH + 8SP + 12HP", language=None)
        st.caption("Output is clipped to the existing 250–650 mg/mL exploration domain.")
    with c2:
        st.markdown("#### Aggregation equation and correction")
        st.code("HMW_BBD = 4.10 − 0.18T − 0.22S − 0.28H − 0.94P + 0.30T² + 0.38S² + 0.32H² + 0.20P² − 0.12TS − 0.24TH − 0.08TP − 0.34SH − 0.10SP − 0.16HP", language=None)
        st.code("HMW_v0.3.1 = (1 − w)·HMW_BBD + w·HMW_PVP,observed", language=None)
        st.caption("w = 0.65 inside the tested PVP domain (15–25 mg/mL), otherwise 0.25.")

    st.markdown("#### Mass-balance supporting calculation")
    st.latex(r"w_{IgG,dry}=\frac{C_{IgG,feed}}{C_{IgG,feed}+C_T+C_S+C_H+C_P}")
    st.metric("Current theoretical dry IgG fraction", f"{prediction.dry_igg_fraction * 100:.1f}%")
    st.markdown(
        "<div class='evidence'><b>Scope:</b> MCT is fixed. Physical loading status is a conservative evidence gate, not a mechanistic packing model. "
        "Powder volume fraction, true density, viscosity and failed concentration challenges remain reserved for v0.4.</div>",
        unsafe_allow_html=True,
    )

with tabs[3]:
    st.subheader("What changed from v0.3")
    changes = pd.DataFrame(
        [
            ["App architecture", "Four excipients → concentration + HMW", "Retained"],
            ["BBD heatmaps", "Two pairwise response surfaces", "Retained with higher-contrast scales"],
            ["Concentration", "Empirical achievable concentration", "Retained; now labeled empirical"],
            ["PVP K12", "Generic nonlinear excipient effect", "Corrected using 0/15/20/25 mg/mL HMW anchors"],
            ["Milling", "Not visible", "Added separate retention output; not conflated with HMW"],
            ["MCT loading", "Numerical capacity output", "Added confirmed/boundary/extrapolated evidence status"],
            ["Upper cap", "Appeared exact", "Now explicitly a lower-bound evidence statement"],
            ["Vehicle comparison", "Possible expansion", "Excluded; MCT remains fixed"],
        ],
        columns=["Area", "v0.3", "v0.3.1"],
    )
    st.dataframe(changes, hide_index=True, use_container_width=True)
    st.markdown("#### v0.3.1 decision rule")
    st.latex(r"C_{reported}=C_{empirical}\quad+\quad\text{independent MCT evidence status}")
    st.write(
        "The app deliberately does not calculate C_final = min(C_empirical, C_physical) yet because the supplied studies include successful loading points but no matched lowest-failed MCT condition."
    )

st.divider()
st.caption(
    "Thermicra IgG model v0.3.1 · Research prototype · Supplied stage-specific data are sparse and partly unreplicated. "
    "Do not use predictions as release, manufacturing, clinical, or regulatory specifications."
)
