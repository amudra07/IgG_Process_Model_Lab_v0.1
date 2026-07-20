from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from model import (
    DRYING_LEVELS,
    EA_RATIO_LEVELS,
    FIXED_PROCESS,
    HARDENING_LEVELS,
    MOCK_RANGES,
    SPRAY_LEVELS,
    add_derived_features,
    default_batch,
    generate_mock_data,
    make_candidates,
    pareto_mask,
    sensitivity_frame,
    train_mock_models,
)


st.set_page_config(
    page_title="IgG Process Model Lab",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .stApp { background: #f6f8f7; }
      [data-testid="stSidebar"] { background: #102f2b; }
      [data-testid="stSidebar"] * { color: #f5fbf8; }
      .hero {
        padding: 1.35rem 1.55rem; border-radius: 18px;
        background: linear-gradient(120deg, #123f39, #1f6b5e);
        color: white; margin-bottom: 1rem;
        box-shadow: 0 8px 24px rgba(20, 67, 58, .15);
      }
      .hero h1 { margin: 0; font-size: 2rem; }
      .hero p { margin: .45rem 0 0; opacity: .86; }
      .notice {
        border-left: 5px solid #dc9d22; background: #fff8e7;
        padding: .85rem 1rem; border-radius: 8px; color: #5b461a;
      }
      .stage-label {
        color: #1f6b5e; text-transform: uppercase; letter-spacing: .08em;
        font-size: .76rem; font-weight: 700; margin-top: .2rem;
      }
      [data-testid="stMetric"] {
        background: white; border: 1px solid #dfe8e4; padding: .75rem;
        border-radius: 12px;
      }
      .fixed-box {
        padding: .8rem 1rem; background: #e9f2ef; border-radius: 10px;
        color: #214d45; font-size: .9rem;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_pipeline():
    mock_data = generate_mock_data()
    models = train_mock_models(mock_data)
    return mock_data, models


mock_data, models = load_pipeline()
defaults = default_batch()


st.markdown(
    """
    <div class="hero">
      <h1>IgG Process Model Lab</h1>
      <p>Explore a spray-dry → hardening → final-drying → MCT suspension workflow.</p>
    </div>
    <div class="notice">
      <b>Pipeline prototype:</b> all formulation ranges, response equations, training
      batches, uncertainty bands, and recommendations are synthetic. They demonstrate
      model behavior and must not be used as experimental or manufacturing evidence.
    </div>
    """,
    unsafe_allow_html=True,
)


with st.sidebar:
    st.markdown("## Model status")
    st.success("Synthetic pipeline active")
    st.caption(f"Training data: {len(mock_data):,} mock batches")
    st.caption("Concentration capacity calibrated to 2 supplied formulation observations")
    st.markdown("---")
    st.markdown("### Fixed conditions")
    st.markdown(
        f"""
        <div class="fixed-box">
          Inlet / outlet: <b>25 °C</b><br>
          Hardening: <b>25 °C</b><br>
          PS80 in ethyl acetate: <b>40 mg/mL</b><br>
          Feed batch volume: <b>{FIXED_PROCESS['feed_batch_volume_ml']:.0f} mL</b><br>
          Viscosity endpoint: <b>25 °C, 100 s⁻¹</b>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.caption("Version 0.1 · Mock-data exploration only")


def formulation_inputs() -> dict[str, float]:
    st.markdown('<div class="stage-label">01 · Feed formulation</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        feed_igg = st.slider("IgG in feed (mg/mL)", 45.0, 75.0, defaults["feed_igg_mg_ml"], 1.0)
        sucrose = st.slider("Sucrose (mg/mL)", 0.0, 30.0, defaults["sucrose_mg_ml"], 0.5)
    with c2:
        trehalose = st.slider("Trehalose (mg/mL)", 0.0, 30.0, defaults["trehalose_mg_ml"], 0.5)
        hpbcd = st.slider("HPBCD (mg/mL)", 0.0, 30.0, defaults["hpbcd_mg_ml"], 0.5)
    with c3:
        pvp = st.slider("PVP (mg/mL)", 0.0, 10.0, defaults["pvp_mg_ml"], 0.5)
        feed_visc = st.slider(
            "Initial feed viscosity (mPa·s)", 10.0, 80.0, defaults["feed_viscosity_mpas"], 1.0
        )

    st.markdown('<div class="stage-label">02 · Process</div>', unsafe_allow_html=True)
    p1, p2, p3, p4 = st.columns(4)
    with p1:
        spray = st.radio("Spray flow", SPRAY_LEVELS, horizontal=True, format_func=lambda x: f"{x} rpm")
    with p2:
        hardening = st.select_slider("Hardening time", options=HARDENING_LEVELS, value=60, format_func=lambda x: f"{x} min")
    with p3:
        ea_ratio = st.select_slider(
            "Ethyl acetate : powder ratio",
            options=EA_RATIO_LEVELS,
            value=0,
            format_func=lambda x: {-1: "Low (−1)", 0: "Center (0)", 1: "High (+1)"}[x],
        )
    with p4:
        drying = st.radio("Final drying", DRYING_LEVELS, index=1, horizontal=True, format_func=lambda x: f"{x} h")

    st.markdown('<div class="stage-label">03 · Baseline quality and MCT loading</div>', unsafe_allow_html=True)
    b1, b2, b3, b4 = st.columns(4)
    with b1:
        feed_hmw = st.number_input("Feed HMW area%", 0.0, 10.0, defaults["feed_hmw_pct"], 0.05)
    with b2:
        feed_monomer = st.number_input("Feed monomer area%", 80.0, 100.0, defaults["feed_monomer_pct"], 0.05)
    with b3:
        powder_added = st.number_input(
            "Dried powder added (mg)", 100.0, 5_000.0, float(defaults["powder_added_mg"]), 25.0
        )
    with b4:
        mct_volume = st.number_input("MCT volume (mL)", 0.10, 5.00, defaults["mct_volume_ml"], 0.05)

    return {
        "feed_igg_mg_ml": feed_igg,
        "sucrose_mg_ml": sucrose,
        "trehalose_mg_ml": trehalose,
        "hpbcd_mg_ml": hpbcd,
        "pvp_mg_ml": pvp,
        "feed_viscosity_mpas": feed_visc,
        "spray_flow_rpm": spray,
        "hardening_time_min": hardening,
        "ea_powder_ratio_code": ea_ratio,
        "drying_time_h": drying,
        "feed_hmw_pct": feed_hmw,
        "feed_monomer_pct": feed_monomer,
        "powder_added_mg": powder_added,
        "mct_volume_ml": mct_volume,
    }


tabs = st.tabs(["Batch explorer", "Variable effects", "Pareto lab", "Model notes"])

with tabs[0]:
    batch = formulation_inputs()
    frame = pd.DataFrame([batch])
    enriched = add_derived_features(frame)
    prediction = models.predict(frame).iloc[0]

    st.markdown("### Predicted outputs")
    r1, r2, r3, r4 = st.columns(4)
    r1.metric(
        "Predicted achievable IgG",
        f"{prediction['predicted_achievable_igg_mg_ml']:.0f} mg/mL",
        f"nominal mass balance {prediction['nominal_igg_mg_ml']:.0f} mg/mL",
        delta_color="off",
    )
    r2.metric(
        "Aggregation / HMW",
        f"{prediction['final_hmw_pct']:.2f}%",
        f"mock P05–P95 {prediction['final_hmw_p05']:.2f}–{prediction['final_hmw_p95']:.2f}%",
        delta_color="off",
    )
    r3.metric(
        "Monomer",
        f"{prediction['final_monomer_pct']:.2f}%",
        f"mock P05–P95 {prediction['final_monomer_p05']:.2f}–{prediction['final_monomer_p95']:.2f}%",
        delta_color="off",
    )
    r4.metric(
        "MCT suspension viscosity",
        f"{prediction['final_viscosity_mpas']:,.0f} mPa·s",
        f"mock P05–P95 {prediction['final_viscosity_p05']:,.0f}–{prediction['final_viscosity_p95']:,.0f}",
        delta_color="off",
    )

    if not 500 <= prediction["predicted_achievable_igg_mg_ml"] <= 700:
        st.warning("Interaction-adjusted achievable IgG is outside the 500–700 mg/mL exploration domain.")
    if not bool(prediction["powder_available_check"]):
        st.error("Selected powder load exceeds the synthetic powder available from this fixed-size mock batch.")

    with st.expander("Mass-balance details", expanded=False):
        mass_table = pd.DataFrame(
            {
                "Quantity": [
                    "Total feed solids",
                    "Initial IgG mass",
                    "Theoretical IgG dry fraction",
                    "Mock spray-dried powder recovered",
                    "Mock final powder after hardening/drying",
                    "Powder loaded into MCT",
                    "Sucrose–HPBCD interaction index",
                    "Calibrated assay recovery",
                    "Interaction-adjusted loading capacity",
                ],
                "Value": [
                    f"{enriched.loc[0, 'initial_dry_solids_mass_g']:.2f} g",
                    f"{enriched.loc[0, 'initial_igg_mass_g']:.2f} g",
                    f"{100 * enriched.loc[0, 'igg_dry_fraction']:.1f}%",
                    f"{enriched.loc[0, 'mock_spray_powder_mass_g']:.2f} g",
                    f"{enriched.loc[0, 'mock_final_powder_mass_g']:.2f} g",
                    f"{batch['powder_added_mg'] / 1_000:.2f} g",
                    f"{prediction['sucrose_hpbcd_interaction_norm']:.2f} × reference",
                    f"{prediction['calibrated_assay_recovery_pct']:.1f}%",
                    f"{prediction['interaction_capacity_mg_ml']:.0f} mg/mL",
                ],
            }
        )
        st.dataframe(mass_table, width="stretch", hide_index=True)
        st.caption(
            "Nominal concentration uses mass balance. Achievable concentration additionally uses a literature-informed sucrose/HPBCD interaction term calibrated to the supplied 400 and 550 mg/mL observations."
        )
    with st.expander("How the excipient interaction equation works", expanded=False):
        st.markdown(
            r"""
            The model calculates a saturable sucrose interaction and a threshold-like
            HPBCD interfacial-protection term:

            $$\theta_{Suc}=\frac{C_{Suc}}{K_{D,Suc}+C_{Suc}}$$

            $$\theta_{HP}=\frac{C_{HP}^{n}}{K_{50,HP}^{n}+C_{HP}^{n}}$$

            $$I_{Suc\times HP}=\theta_{Suc}\theta_{HP}$$

            The normalized combination term modifies the formulation-enabled loading
            capacity. The current constants use $K_{D,Suc}=88.63$ mM at 293 K,
            $K_{50,HP}\approx2.5$ mM, and a fitted combination coefficient of 0.422.
            The latter is target-process calibration, not a universal literature constant.
            """
        )

with tabs[1]:
    st.subheader("One-variable sensitivity explorer")
    st.caption("All other inputs remain at the anonymized center-point batch.")
    effect_options = {
        "Sucrose concentration": ("sucrose_mg_ml", np.linspace(0, 30, 31)),
        "Trehalose concentration": ("trehalose_mg_ml", np.linspace(0, 30, 31)),
        "HPBCD concentration": ("hpbcd_mg_ml", np.linspace(0, 30, 31)),
        "PVP concentration": ("pvp_mg_ml", np.linspace(0, 10, 21)),
        "Initial feed viscosity": ("feed_viscosity_mpas", np.linspace(10, 80, 31)),
        "Spray flow": ("spray_flow_rpm", np.array(SPRAY_LEVELS)),
        "Hardening time": ("hardening_time_min", np.array(HARDENING_LEVELS)),
        "Ethyl acetate : powder ratio": ("ea_powder_ratio_code", np.array(EA_RATIO_LEVELS)),
        "Final drying time": ("drying_time_h", np.array(DRYING_LEVELS)),
        "Powder added": ("powder_added_mg", np.linspace(500, 2500, 31)),
    }
    selected_label = st.selectbox("Variable to sweep", list(effect_options))
    variable, sweep_values = effect_options[selected_label]
    sensitivity = sensitivity_frame(models, defaults, variable, sweep_values)

    response_label = st.radio(
        "Response",
        ["Aggregation / HMW", "Monomer", "Viscosity", "Achievable IgG", "Nominal IgG"],
        horizontal=True,
    )
    response_map = {
        "Aggregation / HMW": "final_hmw_pct",
        "Monomer": "final_monomer_pct",
        "Viscosity": "final_viscosity_mpas",
        "Achievable IgG": "predicted_achievable_igg_mg_ml",
        "Nominal IgG": "nominal_igg_mg_ml",
    }
    response_col = response_map[response_label]
    chart_data = sensitivity[[variable, response_col]].set_index(variable)
    st.line_chart(chart_data, width="stretch")
    st.dataframe(
        sensitivity[[variable, "predicted_achievable_igg_mg_ml", "nominal_igg_mg_ml", "final_hmw_pct", "final_monomer_pct", "final_viscosity_mpas"]]
        .round(3),
        width="stretch",
        hide_index=True,
    )
    st.info("These curves reflect the synthetic equation and surrogate model—not a measured causal relationship.")

with tabs[2]:
    st.subheader("Pareto candidate explorer")
    st.caption(
        "Frontier objectives: minimize HMW, maximize monomer, and minimize viscosity. Interaction-adjusted achievable IgG is constrained to the 500–700 mg/mL model domain."
    )
    candidate_count = st.slider("Number of mock candidates", 500, 3_000, 1_500, 250)
    candidates = make_candidates(candidate_count)
    candidate_predictions = models.predict(candidates)
    candidate_results = pd.concat([candidates, candidate_predictions], axis=1)
    in_domain = candidate_results["predicted_achievable_igg_mg_ml"].between(500, 700)
    candidate_results = candidate_results.loc[in_domain].reset_index(drop=True)
    objectives = np.column_stack(
        [
            candidate_results["final_hmw_pct"],
            -candidate_results["final_monomer_pct"],
            np.log(candidate_results["final_viscosity_mpas"]),
        ]
    )
    frontier = candidate_results.loc[pareto_mask(objectives)].copy()
    frontier["set"] = "Pareto frontier"
    candidate_results["set"] = "Dominated candidate"

    st.metric("Non-dominated candidates", f"{len(frontier):,}", f"of {len(candidate_results):,} in-domain")
    plot_frame = pd.concat(
        [
            candidate_results.sample(min(800, len(candidate_results)), random_state=17),
            frontier,
        ],
        ignore_index=True,
    )
    st.scatter_chart(
        plot_frame,
        x="final_hmw_pct",
        y="final_viscosity_mpas",
        color="set",
        size="final_monomer_pct",
        width="stretch",
    )
    st.caption("Bubble size represents predicted monomer percentage.")

    display_cols = [
        "candidate_id",
        "feed_igg_mg_ml",
        "sucrose_mg_ml",
        "trehalose_mg_ml",
        "hpbcd_mg_ml",
        "pvp_mg_ml",
        "spray_flow_rpm",
        "hardening_time_min",
        "ea_powder_ratio_code",
        "drying_time_h",
        "predicted_achievable_igg_mg_ml",
        "nominal_igg_mg_ml",
        "final_hmw_pct",
        "final_monomer_pct",
        "final_viscosity_mpas",
    ]
    sort_by = st.selectbox(
        "Sort frontier for review",
        ["final_hmw_pct", "final_monomer_pct", "final_viscosity_mpas", "predicted_achievable_igg_mg_ml"],
        format_func=lambda x: {
            "final_hmw_pct": "Lowest HMW",
            "final_monomer_pct": "Highest monomer",
            "final_viscosity_mpas": "Lowest viscosity",
            "predicted_achievable_igg_mg_ml": "Achievable IgG",
        }[x],
    )
    ascending = sort_by != "final_monomer_pct"
    st.dataframe(
        frontier.sort_values(sort_by, ascending=ascending)[display_cols].round(3),
        width="stretch",
        hide_index=True,
    )

with tabs[3]:
    st.subheader("Defined output specification")
    st.markdown(
        """
        | Output | Definition | Model method |
        |---|---|---|
        | Achievable IgG concentration | Minimum of nominal mass-balance loading and interaction-adjusted formulation capacity, mg/mL | Literature-informed equation calibrated to two supplied observations |
        | Nominal IgG concentration | Theoretical IgG mass divided by added MCT volume, mg/mL | Deterministic mass balance retained as an intermediate |
        | Aggregation | SEC-HPLC HMW area% after final drying | Synthetic surrogate predicts process-induced change from feed |
        | Monomer | SEC-HPLC monomer area% after final drying | Synthetic surrogate predicts process-induced change from feed |
        | Viscosity | Dried powder dispersed in MCT, measured at 25 °C and 100 s⁻¹, mPa·s | Log-viscosity synthetic surrogate |
        """
    )
    st.subheader("Key assumptions")
    st.markdown(
        """
        - Sucrose, trehalose, HPBCD, PVP, and feed IgG ranges remain prototype values.
        - Sucrose uses a reported mAb interaction constant of 88.63 mM at 293 K; transferability to this IgG must be tested.
        - HPBCD uses a provisional molecular weight of 1,400 g/mol and a 2.5 mM interfacial-protection scale; the exact grade must replace these assumptions.
        - The 0.422 sucrose–HPBCD combination coefficient is calibrated to the supplied 400/550 mg/mL results, not taken from literature.
        - Inlet, outlet, and hardening temperatures are fixed at 25 °C.
        - PS80 is fixed at 40 mg/mL in ethyl acetate.
        - Ethyl-acetate-to-powder ratio is coded as low/center/high because physical values are not yet confirmed.
        - Spray flow is evaluated only at 20 and 40 rpm; interpolation is unsupported.
        - Nominal concentration assumes proportional, non-selective recovery and uses added MCT volume rather than true final suspension volume.
        - The tree-to-tree P05–P95 range is a mock model-disagreement band, not a validated prediction interval.
        """
    )
    st.subheader("Required before experimental use")
    st.markdown(
        """
        1. Replace prototype formulation ranges, HPBCD molecular weight, and ethyl-acetate ratio codes with approved physical values.
        2. Standardize SEC-HPLC, UV/protein assay, and rheology sample preparation.
        3. Collect paired feed and post-drying SEC results for each batch.
        4. Record actual powder recovery at spray drying and final drying stages.
        5. Fit only low-complexity, uncertainty-aware models to the first <15 real batches.
        6. Perform prospective confirmation batches before using recommendations for decisions.
        """
    )
