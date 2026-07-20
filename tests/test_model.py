import pathlib
import sys

import numpy as np
import pandas as pd


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model import (  # noqa: E402
    REFERENCE_HPBCD_MG_ML,
    REFERENCE_SUCROSE_MG_ML,
    add_derived_features,
    default_batch,
    generate_mock_data,
    make_candidates,
    pareto_mask,
    train_mock_models,
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
            "spray_flow_rpm",
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
    assert np.isfinite(prediction.select_dtypes(include="number")).all().all()
    assert prediction["nominal_igg_mg_ml"].between(450, 750).all()
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


def test_interaction_capacity_reproduces_supplied_calibration_points():
    sucrose = default_batch()
    sucrose.update(
        feed_igg_mg_ml=60.0,
        sucrose_mg_ml=REFERENCE_SUCROSE_MG_ML,
        trehalose_mg_ml=0.0,
        hpbcd_mg_ml=0.0,
        pvp_mg_ml=0.0,
        powder_added_mg=2_000.0,
        mct_volume_ml=1.0,
    )
    combination = dict(sucrose)
    combination["hpbcd_mg_ml"] = REFERENCE_HPBCD_MG_ML

    result = add_derived_features(pd.DataFrame([sucrose, combination]))
    assert np.isclose(result.loc[0, "interaction_capacity_mg_ml"], 400.0)
    assert np.isclose(result.loc[1, "interaction_capacity_mg_ml"], 550.0)
    assert np.isclose(result.loc[0, "calibrated_assay_recovery_fraction"], 0.887)
    assert np.isclose(result.loc[1, "calibrated_assay_recovery_fraction"], 0.926)
    assert np.isclose(result.loc[1, "sucrose_hpbcd_interaction_norm"], 1.0)


def test_pareto_filter_removes_known_dominated_point():
    points = np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [0.5, 3.0, 1.0]])
    mask = pareto_mask(points)
    assert mask.tolist() == [True, False, True]


def test_candidate_generator_has_expected_process_levels():
    candidates = make_candidates(n=80, seed=17)
    assert set(candidates["spray_flow_rpm"].unique()).issubset({20, 40})
    assert set(candidates["hardening_time_min"].unique()).issubset({10, 30, 60, 180, 300})
    assert set(candidates["drying_time_h"].unique()).issubset({8, 24, 48})
