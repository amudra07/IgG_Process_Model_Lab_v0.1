import pathlib
import sys

import numpy as np
import pandas as pd


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model import (  # noqa: E402
    FEED_FLOW_LEVELS,
    PVP_HMW_ANCHORS,
    PVP_MILLING_RECOVERY_ANCHORS,
    ULTRASONIC_POWER_LEVELS,
    V03_CAPACITY_ANCHORS,
    add_derived_features,
    default_batch,
    experimental_quality_surface,
    generate_mock_data,
    make_candidates,
    pareto_mask,
    train_mock_models,
    pvp_milling_recovery,
    pvp_observed_hmw,
)


def test_mock_pipeline_outputs_are_finite_and_bounded():
    data = generate_mock_data(n=160, seed=11)
    bundle = train_mock_models(data, seed=12)
    inputs = data.iloc[:12][
        [
            "feed_igg_mg_ml",
            "sucrose_mg_ml",
            "trehalose_mg_ml",
            "hpbcd_mg_ml",
            "pvp_mg_ml",
            "feed_viscosity_mpas",
            "ultrasonic_power_pct",
            "feed_flow_rpm",
            "hardening_time_min",
            "ea_powder_ratio_code",
            "drying_time_h",
            "feed_hmw_pct",
            "feed_monomer_pct",
            "powder_added_mg",
            "mct_volume_ml",
        ]
    ]
    prediction = bundle.predict(inputs)
    numeric = prediction.select_dtypes(include="number").drop(
        columns=[
            "milling_recovery_pct",
            "mct_confirmed_lower_bound_mg_ml",
            "spray_agg_severity_k",
        ]
    )
    assert np.isfinite(numeric).all().all()
    assert np.array_equal(
        prediction["milling_recovery_pct"].isna().to_numpy(),
        (~inputs["pvp_mg_ml"].between(15.0, 25.0)).to_numpy(),
    )
    assert prediction["nominal_igg_mg_ml"].between(250, 650).all()
    assert (prediction["predicted_achievable_igg_mg_ml"] <= prediction["nominal_igg_mg_ml"]).all()
    assert (prediction["final_hmw_pct"] >= 0).all()
    assert (prediction["final_monomer_pct"] <= 100).all()
    assert (prediction["final_viscosity_mpas"] > 0).all()


def test_nominal_concentration_is_mass_balance_not_ml():
    batch = default_batch()
    frame = add_derived_features(pd.DataFrame([batch]))
    expected = (
        batch["powder_added_mg"]
        * frame.loc[0, "igg_dry_fraction"]
        / batch["mct_volume_ml"]
    )
    assert np.isclose(frame.loc[0, "nominal_igg_mg_ml"], expected)


def test_capacity_surface_reproduces_all_six_v03_anchors():
    rows = []
    for sucrose, hpbcd, _capacity, _assay, _hmw, _monomer in V03_CAPACITY_ANCHORS:
        batch = default_batch()
        batch.update(
            sucrose_mg_ml=sucrose,
            trehalose_mg_ml=0.0,
            hpbcd_mg_ml=hpbcd,
            pvp_mg_ml=0.0,
            powder_added_mg=2_000.0,
            mct_volume_ml=1.0,
        )
        rows.append(batch)
    result = add_derived_features(pd.DataFrame(rows))
    assert np.allclose(
        result["interaction_capacity_mg_ml"].to_numpy(),
        V03_CAPACITY_ANCHORS[:, 2],
    )
    assert np.allclose(result["experimental_capacity_support"], 1.0)

    assay, hmw, monomer, support = experimental_quality_surface(
        V03_CAPACITY_ANCHORS[:, 0], V03_CAPACITY_ANCHORS[:, 1]
    )
    assert np.allclose(assay, V03_CAPACITY_ANCHORS[:, 3])
    assert np.allclose(hmw, V03_CAPACITY_ANCHORS[:, 4])
    assert np.allclose(monomer, V03_CAPACITY_ANCHORS[:, 5])
    assert np.allclose(support, 1.0)


def test_pareto_filter_removes_known_dominated_point():
    points = np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [0.5, 3.0, 1.0]])
    mask = pareto_mask(points)
    assert mask.tolist() == [True, False, True]


def test_candidate_generator_has_expected_process_levels():
    candidates = make_candidates(n=80, seed=17)
    assert set(candidates["ultrasonic_power_pct"].unique()).issubset(set(ULTRASONIC_POWER_LEVELS))
    assert set(candidates["feed_flow_rpm"].unique()).issubset(set(FEED_FLOW_LEVELS))
    assert set(candidates["hardening_time_min"].unique()).issubset({10, 30, 60, 180, 300})
    assert set(candidates["drying_time_h"].unique()).issubset({8, 24, 48})


def test_v031_pvp_hmw_and_milling_anchors_are_reproduced():
    assert np.allclose(
        pvp_observed_hmw(PVP_HMW_ANCHORS[:, 0]),
        PVP_HMW_ANCHORS[:, 1],
    )
    assert np.allclose(
        pvp_milling_recovery(PVP_MILLING_RECOVERY_ANCHORS[:, 0]),
        PVP_MILLING_RECOVERY_ANCHORS[:, 1],
    )
    assert np.isnan(pvp_milling_recovery(np.array([10.0]))[0])


def test_v031_full_process_prediction_adds_evidence_outputs():
    data = generate_mock_data(n=160, seed=21)
    bundle = train_mock_models(data, seed=22)
    batch = default_batch()
    batch["pvp_mg_ml"] = 20.0
    result = bundle.predict(pd.DataFrame([batch])).iloc[0]
    assert np.isclose(result["milling_recovery_pct"], 86.7)
    assert result["reconstitution_status"] == "Insoluble in supplied PVP study"
    assert result["mct_loading_status"] == "Uncertain composition"
    assert 0.4 <= result["final_hmw_pct"] <= 9.0


def test_v032_separates_power_from_feed_flow_and_keeps_spray_damage_uncalibrated():
    low = default_batch()
    high = default_batch()
    low.update(ultrasonic_power_pct=20, feed_flow_rpm=40)
    high.update(ultrasonic_power_pct=40, feed_flow_rpm=40)
    enriched = add_derived_features(pd.DataFrame([low, high]))
    assert enriched.loc[1, "atomization_severity_index"] > enriched.loc[0, "atomization_severity_index"]
    assert "clogging" in enriched.loc[1, "thermal_clogging_risk"]
    assert enriched["spray_agg_severity_k"].isna().all()


def test_v032_exact_formulation_anchor_is_labelled_as_evidence_not_confidence():
    data = generate_mock_data(n=120, seed=31)
    bundle = train_mock_models(data, seed=32)
    batch = default_batch()
    batch.update(sucrose_mg_ml=2.5, hpbcd_mg_ml=5.0)
    result = bundle.predict(pd.DataFrame([batch])).iloc[0]
    assert result["capacity_evidence_status"] == "Exact experimental anchor"
    assert result["supporting_formulation_observations"] == 6
    assert np.isclose(result["delta_hmw_pct"], result["final_hmw_pct"] - batch["feed_hmw_pct"])
