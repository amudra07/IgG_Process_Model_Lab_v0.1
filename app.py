from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from model import (
    DRYING_LEVELS,
    EA_RATIO_LEVELS,
    FIXED_PROCESS,
    HARDENING_LEVELS,
    MOCK_RANGES,
    ULTRASONIC_POWER_LEVELS,
    FEED_FLOW_LEVELS,
    add_derived_features,
    default_batch,
    generate_mock_data,
    make_candidates,
    pareto_mask,
    sensitivity_frame,
    train_mock_models,
    experimental_capacity_surface,
)

APP_VERSION = "0.3.2"
PREDICTION_SCHEMA = "0.3.2-modular-ultrasonic-power-feed-flow-20260722"
POWER_EVIDENCE_PATH = Path(__file__).resolve().parent / "data" / "ultrasonic_power_observations.csv"


def _with_v03_concentration(frame: pd.DataFrame, prediction: pd.DataFrame) -> pd.DataFrame:
    """Attach v0.3 empirical-capacity outputs at one stable schema boundary.

    Streamlit deployments can briefly retain an older cached ModelBundle while a
    new app file is starting. Keeping the deterministic concentration layer here
    makes that transition safe and guarantees one prediction schema for every tab.
    """
    result = prediction.copy()
    total_excipient = frame[[
        "sucrose_mg_ml", "trehalose_mg_ml", "hpbcd_mg_ml", "pvp_mg_ml"
    ]].sum(axis=1)
    igg_fraction = frame["feed_igg_mg_ml"] / (frame["feed_igg_mg_ml"] + total_excipient).clip(lower=1e-9)
    nominal = frame["powder_added_mg"] * igg_fraction / frame["mct_volume_ml"].clip(lower=1e-9)

    capacity, support = experimental_capacity_surface(
        frame["sucrose_mg_ml"], frame["hpbcd_mg_ml"]
    )

    result["nominal_igg_mg_ml"] = nominal.to_numpy()
    result["interaction_capacity_mg_ml"] = capacity
    result["experimental_capacity_support"] = support
    result["predicted_achievable_igg_mg_ml"] = np.minimum(nominal, capacity)
    return result


class PredictionSchemaAdapter:
    """Normalize model bundles to the v0.3 output schema."""

    def __init__(self, model_bundle):
        self.model_bundle = model_bundle

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        return _with_v03_concentration(frame, self.model_bundle.predict(frame))


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
def load_pipeline(prediction_schema: str):
    mock_data = generate_mock_data()
    models = PredictionSchemaAdapter(train_mock_models(mock_data))
    return mock_data, models


mock_data, models = load_pipeline(PREDICTION_SCHEMA)
defaults = default_batch()


st.markdown(
    """
    <div class="hero">
      <h1>IgG Process Model Lab</h1>
      <p>Explore formulation → ultrasonic spray → hardening/drying → MCT loading as connected unit-operation models.</p>
    </div>
    <div class="notice">
      <b>Evidence-aware prototype:</b> mass balances are calculated; six local
      sucrose–HPBCD points, PVP observations, power observations, and loading anchors
      are experimental evidence; remaining process responses are provisional synthetic
      relationships. Outputs must not be used as manufacturing evidence.
    </div>
    """,
    unsafe_allow_html=True,
)


with st.sidebar:
    st.markdown("## Model status")
    st.success("v0.3.2 modular pipeline active")
    st.caption(f"Training data: {len(mock_data):,} mock batches")
    st.caption("Capacity surface uses 6 quantitative supplied rows")
    st.caption("Broader screen: 25 transcribed rows, missing values preserved")
    st.caption("PVP HMW evidence: 4 supplied anchors")
    st.caption("Ultrasonic power and feed flow are separate")
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
    st.caption(f"Version {APP_VERSION} · Mock-data exploration only")


MODEL_SCOPES = (
    "Full process simulator",
    "Formulation stability",
    "Ultrasonic spray drying",
    "Hardening and final drying",
    "Milling and MCT loading",
)


def _enabled(scope: str, stage: str) -> bool:
    return scope == "Full process simulator" or scope == stage


def formulation_inputs(scope: str) -> dict[str, float]:
    formulation_enabled = _enabled(scope, "Formulation stability")
    spray_enabled = _enabled(scope, "Ultrasonic spray drying")
    hardening_enabled = _enabled(scope, "Hardening and final drying")
    loading_enabled = _enabled(scope, "Milling and MCT loading")
    st.markdown('<div class="stage-label">01 · Feed formulation</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        feed_igg = st.slider("IgG in feed (mg/mL)", 45.0, 75.0, defaults["feed_igg_mg_ml"], 1.0, disabled=not formulation_enabled)
        sucrose = st.slider("Sucrose (mg/mL)", 0.0, 10.0, defaults["sucrose_mg_ml"], 0.25, disabled=not formulation_enabled)
    with c2:
        trehalose = st.slider("Trehalose (mg/mL)", 0.0, 30.0, defaults["trehalose_mg_ml"], 0.5, disabled=not formulation_enabled)
        hpbcd = st.slider("HPBCD (mg/mL)", 0.0, 15.0, defaults["hpbcd_mg_ml"], 0.25, disabled=not formulation_enabled)
    with c3:
        pvp = st.slider("PVP (mg/mL)", 0.0, 25.0, defaults["pvp_mg_ml"], 0.5, disabled=not (formulation_enabled or loading_enabled))
        feed_visc = st.slider(
            "Initial feed viscosity (mPa·s)", 10.0, 80.0, defaults["feed_viscosity_mpas"], 1.0,
            disabled=not (formulation_enabled or spray_enabled),
        )

    st.markdown('<div class="stage-label">02 · Process</div>', unsafe_allow_html=True)
    p1, p2, p3, p4, p5 = st.columns(5)
    with p1:
        power = st.radio("Ultrasonic power", ULTRASONIC_POWER_LEVELS, horizontal=True, format_func=lambda x: f"{x}%", disabled=not spray_enabled)
    with p2:
        feed_flow = st.radio("Feed flow", FEED_FLOW_LEVELS, horizontal=True, format_func=lambda x: f"{x} rpm", disabled=not spray_enabled)
    with p3:
        hardening = st.select_slider("Hardening time", options=HARDENING_LEVELS, value=60, format_func=lambda x: f"{x} min", disabled=not hardening_enabled)
    with p4:
        ea_ratio = st.select_slider(
            "Ethyl acetate : powder ratio",
            options=EA_RATIO_LEVELS,
            value=0,
            format_func=lambda x: {-1: "Low (−1)", 0: "Center (0)", 1: "High (+1)"}[x],
            disabled=not hardening_enabled,
        )
    with p5:
        drying = st.radio("Final drying", DRYING_LEVELS, index=1, horizontal=True, format_func=lambda x: f"{x} h", disabled=not hardening_enabled)

    st.markdown('<div class="stage-label">03 · Baseline quality and MCT loading</div>', unsafe_allow_html=True)
    b1, b2, b3, b4 = st.columns(4)
    with b1:
        feed_hmw = st.number_input("Feed HMW area%", 0.0, 10.0, defaults["feed_hmw_pct"], 0.05, disabled=not (formulation_enabled or spray_enabled))
    with b2:
        feed_monomer = st.number_input("Feed monomer area%", 80.0, 100.0, defaults["feed_monomer_pct"], 0.05, disabled=not (formulation_enabled or spray_enabled))
    with b3:
        powder_added = st.number_input(
            "Dried powder added (mg)", 100.0, 5_000.0, float(defaults["powder_added_mg"]), 25.0,
            disabled=not loading_enabled,
        )
    with b4:
        mct_volume = st.number_input("MCT volume (mL)", 0.10, 5.00, defaults["mct_volume_ml"], 0.05, disabled=not loading_enabled)

    return {
        "feed_igg_mg_ml": feed_igg,
        "sucrose_mg_ml": sucrose,
        "trehalose_mg_ml": trehalose,
        "hpbcd_mg_ml": hpbcd,
        "pvp_mg_ml": pvp,
        "feed_viscosity_mpas": feed_visc,
        "ultrasonic_power_pct": power,
        "feed_flow_rpm": feed_flow,
        "hardening_time_min": hardening,
        "ea_powder_ratio_code": ea_ratio,
        "drying_time_h": drying,
        "feed_hmw_pct": feed_hmw,
        "feed_monomer_pct": feed_monomer,
        "powder_added_mg": powder_added,
        "mct_volume_ml": mct_volume,
    }


scope = st.selectbox(
    "Model scope",
    MODEL_SCOPES,
    help="Only variables belonging to the selected unit-operation model remain adjustable; the full simulator keeps every stage connected.",
)
if scope != "Full process simulator":
    st.info(f"{scope} scope active. Inputs from other stages are locked at the reference condition.")

tabs = st.tabs(["Batch explorer", "Variable effects", "Stage model status", "Pareto lab"])

with tabs[0]:
    batch = formulation_inputs(scope)
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
        f"process-induced change {prediction['delta_hmw_pct']:+.2f}%",
        delta_color="off",
    )
    r3.metric(
        "Monomer",
        f"{prediction['final_monomer_pct']:.2f}%",
        f"process-induced change {prediction['delta_monomer_pct']:+.2f}%",
        delta_color="off",
    )
    r4.metric(
        "MCT suspension viscosity",
        f"{prediction['final_viscosity_mpas']:,.0f} mPa·s",
        "Provisional synthetic relationship",
        delta_color="off",
    )

    st.markdown("### Supporting evidence outputs")
    s1, s2, s3 = st.columns(3)
    milling_value = prediction["milling_recovery_pct"]
    s1.metric(
        "Milling recovery",
        f"{milling_value:.1f}%" if np.isfinite(milling_value) else "Not calibrated",
        "PVP study: 15–25 mg/mL" if np.isfinite(milling_value) else "No matched assay anchor",
        delta_color="off",
    )
    lower_bound = prediction["mct_confirmed_lower_bound_mg_ml"]
    lower_bound_text = (
        f"confirmed ≥{lower_bound:.0f} mg/mL"
        if np.isfinite(lower_bound)
        else "no composition-matched anchor"
    )
    s2.metric(
        "MCT loading evidence",
        prediction["mct_loading_status"],
        lower_bound_text,
        delta_color="off",
    )
    s3.metric(
        "Water reconstitution",
        prediction["reconstitution_status"],
        "Evidence status, not a fitted response",
        delta_color="off",
    )

    if not 250 <= prediction["predicted_achievable_igg_mg_ml"] <= 650:
        st.warning("Predicted achievable IgG is outside the 250–650 mg/mL v0.3 exploration domain.")
    if prediction["experimental_capacity_support"] < 0.35:
        st.warning("Low experimental support: this sucrose/HPBCD combination is far from the six quantitative v0.3 anchors.")
    if not bool(prediction["powder_available_check"]):
        st.error("Selected powder load exceeds the synthetic powder available from this fixed-size mock batch.")
    if prediction["mct_loading_status"] in {"Boundary", "Extrapolated", "Uncertain composition"}:
        st.warning(
            "MCT loading is not directly confirmed for this selected composition and loading. "
            "The reported 400 and 550 mg/mL successes are lower-bound anchors, not exact maxima."
        )

    st.markdown("### Evidence and spray-block status")
    evidence_table = pd.DataFrame(
        {
            "Output / block": [
                "Nominal IgG",
                "Attainable concentration",
                "Final HMW / monomer",
                "Spray recovery",
                "Spray particle size",
                "Spray-specific aggregation coefficient",
                "Ultrasonic nozzle risk",
            ],
            "Evidence class": [
                "Mass balance",
                prediction["capacity_evidence_status"],
                prediction["quality_evidence_status"],
                "Provisional assumption",
                "Not yet modelled",
                "Not yet calibrated",
                "Observed process warning",
            ],
            "Result / status": [
                f"{prediction['nominal_igg_mg_ml']:.0f} mg/mL",
                f"{prediction['predicted_achievable_igg_mg_ml']:.0f} mg/mL",
                f"ΔHMW {prediction['delta_hmw_pct']:+.2f}%; Δmonomer {prediction['delta_monomer_pct']:+.2f}%",
                f"{prediction['provisional_spray_recovery_pct']:.1f}%",
                prediction["particle_size_calibration_status"],
                prediction["spray_agg_calibration_status"],
                prediction["thermal_clogging_risk"],
            ],
        }
    )
    st.dataframe(evidence_table, width="stretch", hide_index=True)
    st.caption(
        f"Nearest formulation distance: {prediction['nearest_formulation_distance']:.2f} scaled units; "
        f"supporting quantitative formulation observations: {int(prediction['supporting_formulation_observations'])}. "
        "Distance is an evidence descriptor, not a confidence interval."
    )

    with st.expander("Mass-balance details", expanded=False):
        mass_table = pd.DataFrame(
            {
                "Quantity": [
                    "Total feed solids",
                    "Initial IgG mass",
                    "Theoretical IgG dry fraction",
                    "Provisional spray-dried powder recovered",
                    "Provisional final powder after hardening/drying",
                    "Powder loaded into MCT",
                    "Experimental-surface support",
                    "Empirical loading capacity",
                    "MCT confirmed lower bound",
                    "MCT loading evidence status",
                ],
                "Value": [
                    f"{enriched.loc[0, 'initial_dry_solids_mass_g']:.2f} g",
                    f"{enriched.loc[0, 'initial_igg_mass_g']:.2f} g",
                    f"{100 * enriched.loc[0, 'igg_dry_fraction']:.1f}%",
                    f"{enriched.loc[0, 'provisional_spray_powder_mass_g']:.2f} g",
                    f"{enriched.loc[0, 'provisional_final_powder_mass_g']:.2f} g",
                    f"{batch['powder_added_mg'] / 1_000:.2f} g",
                    f"{100 * prediction['experimental_capacity_support']:.0f}% proximity",
                    f"{prediction['interaction_capacity_mg_ml']:.0f} mg/mL",
                    f"≥{lower_bound:.0f} mg/mL" if np.isfinite(lower_bound) else "Not composition-matched",
                    prediction["mct_loading_status"],
                ],
            }
        )
        st.dataframe(mass_table, width="stretch", hide_index=True)
        st.caption(
            "Nominal concentration uses mass balance. Achievable concentration is capped by an empirical surface fitted to six supplied sucrose/HPBCD concentration combinations."
        )
    with st.expander("How the excipient interaction equation works", expanded=False):
        st.markdown(
            r"""
            Version 0.3 replaces the one-direction interaction bonus with bounded
            inverse-distance interpolation over the six quantitative experimental rows:

            $$\hat C_{cap}(x)=\frac{\sum_i w_i(x)C_i}{\sum_i w_i(x)}$$

            $$w_i(x)=\left[d_i(x)^2+0.04\right]^{-2}$$

            where $x=(C_{Suc},C_{HPBCD})$ and scaled distance uses 2.5 mg/mL on each
            axis. Exact experimental combinations return the observed capacity exactly.
            Rows with unreported component concentrations are displayed as qualitative
            evidence but are not assigned invented numerical doses.
            """
        )

with tabs[1]:
    st.subheader("One-variable sensitivity explorer")
    st.caption("All other inputs remain at the anonymized center-point batch.")
    effect_options = {
        "Sucrose concentration": ("sucrose_mg_ml", np.linspace(0, 10, 41)),
        "Trehalose concentration": ("trehalose_mg_ml", np.linspace(0, 30, 31)),
        "HPBCD concentration": ("hpbcd_mg_ml", np.linspace(0, 15, 61)),
        "PVP concentration": ("pvp_mg_ml", np.linspace(0, 25, 51)),
        "Initial feed viscosity": ("feed_viscosity_mpas", np.linspace(10, 80, 31)),
        "Ultrasonic power": ("ultrasonic_power_pct", np.array(ULTRASONIC_POWER_LEVELS)),
        "Feed flow": ("feed_flow_rpm", np.array(FEED_FLOW_LEVELS)),
        "Hardening time": ("hardening_time_min", np.array(HARDENING_LEVELS)),
        "Ethyl acetate : powder ratio": ("ea_powder_ratio_code", np.array(EA_RATIO_LEVELS)),
        "Final drying time": ("drying_time_h", np.array(DRYING_LEVELS)),
        "Powder added": ("powder_added_mg", np.linspace(500, 2500, 31)),
    }
    scope_variables = {
        "Formulation stability": {"sucrose_mg_ml", "trehalose_mg_ml", "hpbcd_mg_ml", "pvp_mg_ml", "feed_viscosity_mpas"},
        "Ultrasonic spray drying": {"ultrasonic_power_pct", "feed_flow_rpm", "feed_viscosity_mpas"},
        "Hardening and final drying": {"hardening_time_min", "ea_powder_ratio_code", "drying_time_h"},
        "Milling and MCT loading": {"pvp_mg_ml", "powder_added_mg"},
    }
    if scope != "Full process simulator":
        allowed = scope_variables[scope]
        effect_options = {
            label: item for label, item in effect_options.items() if item[0] in allowed
        }
    selected_label = st.selectbox("Variable to sweep", list(effect_options))
    variable, sweep_values = effect_options[selected_label]
    sensitivity = sensitivity_frame(models, defaults, variable, sweep_values)

    response_label = st.radio(
        "Response",
        ["Aggregation / HMW", "Monomer", "Viscosity", "Achievable IgG", "Nominal IgG", "Milling recovery"],
        horizontal=True,
    )
    response_map = {
        "Aggregation / HMW": "final_hmw_pct",
        "Monomer": "final_monomer_pct",
        "Viscosity": "final_viscosity_mpas",
        "Achievable IgG": "predicted_achievable_igg_mg_ml",
        "Nominal IgG": "nominal_igg_mg_ml",
        "Milling recovery": "milling_recovery_pct",
    }
    response_col = response_map[response_label]
    chart_data = sensitivity[[variable, response_col]].set_index(variable)
    st.line_chart(chart_data, width="stretch")
    st.dataframe(
        sensitivity[[variable, "predicted_achievable_igg_mg_ml", "nominal_igg_mg_ml", "final_hmw_pct", "final_monomer_pct", "final_viscosity_mpas", "milling_recovery_pct", "mct_loading_status"]]
        .round(3),
        width="stretch",
        hide_index=True,
    )
    st.info(
        "These curves show local empirical interpolation where anchors exist and provisional "
        "synthetic relationships elsewhere. They are not measured causal effects."
    )

with tabs[2]:
    st.subheader("Modular unit-operation model status")
    st.caption("The output state of each block becomes the input state of the next block.")
    stage_table = pd.DataFrame(
        {
            "Unit-operation model": [
                "1 · Formulation stability",
                "2 · Ultrasonic spray drying",
                "3 · EA hardening / final drying",
                "4 · Milling / MCT loading",
            ],
            "Active variables": [
                "Sucrose, trehalose, HPBCD, PVP",
                "Ultrasonic power, feed flow, feed viscosity, total solids",
                "EA:powder ratio, hardening time, drying time",
                "PVP, powder loading, MCT volume; future particle size/moisture",
            ],
            "Primary outputs": [
                "Local attainable concentration; final-quality anchors",
                "Recovery and drying severity; future moisture/particle size/ΔHMW",
                "Provisional recovery and combined quality change",
                "Milling recovery, concentration, viscosity, loading status",
            ],
            "Current evidence": [
                "6 quantitative sucrose–HPBCD anchors + 4 PVP HMW anchors",
                "Power observations support heating/clogging warning; physical CQAs pending",
                "Synthetic/provisional until matched intermediate measurements exist",
                "3 PVP milling anchors + 2 composition-specific loading lower bounds",
            ],
        }
    )
    st.dataframe(stage_table, width="stretch", hide_index=True)
    st.markdown(
        r"""
        **Spray-drying transformation block**

        $$m_{powder,out}=\eta_{SD}\,m_{dry\ solids,in}$$

        $$X_{moisture,out}=X_{eq}+(X_{moisture,in}-X_{eq})e^{-K_{dry,SD}}$$

        $$M_{out}=M_{in}e^{-K_{agg,SD}}$$

        In v0.3.2, $\eta_{SD}$ and $K_{dry,SD}$ are provisional severity relationships.
        $K_{agg,SD}$ remains unavailable because SEC-HPLC immediately before and after spraying
        has not yet been supplied. It must not be interpreted from the final HMW response.
        """
    )
    st.warning(
        "The model does not yet convert ultrasonic power into particle size. Higher power → "
        "stronger atomization → potentially smaller droplets remains a mechanistic hypothesis "
        "until droplet or particle-size measurements are paired with each batch."
    )
    st.markdown("### Supplied ultrasonic-power observations")
    power_evidence = pd.read_csv(POWER_EVIDENCE_PATH)
    st.dataframe(
        power_evidence[
            [
                "composition",
                "ultrasonic_power_pct",
                "feed_flow_rpm",
                "suspension_mg_ml",
                "aggregation_pct",
                "interpretation",
            ]
        ],
        width="stretch",
        hide_index=True,
    )
    st.caption(
        "These four rows support local comparison and the 40% heating/clogging warning. "
        "They are not used to fit a universal power coefficient."
    )

with tabs[3]:
    st.subheader("Pareto candidate explorer")
    if scope != "Full process simulator":
        st.info("Pareto search always evaluates the connected full-process prototype, independent of the selected single-stage input scope.")
    st.caption(
        "Frontier objectives: minimize HMW, maximize monomer, and minimize viscosity. Achievable IgG is constrained to the 250–650 mg/mL v0.3 domain."
    )
    candidate_count = st.slider("Number of mock candidates", 500, 3_000, 1_500, 250)
    candidates = make_candidates(candidate_count)
    candidate_predictions = models.predict(candidates)
    candidate_results = pd.concat([candidates, candidate_predictions], axis=1)
    in_domain = candidate_results["predicted_achievable_igg_mg_ml"].between(250, 650)
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
        "ultrasonic_power_pct",
        "feed_flow_rpm",
        "hardening_time_min",
        "ea_powder_ratio_code",
        "drying_time_h",
        "predicted_achievable_igg_mg_ml",
        "nominal_igg_mg_ml",
        "final_hmw_pct",
        "final_monomer_pct",
        "final_viscosity_mpas",
        "milling_recovery_pct",
        "mct_loading_status",
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
