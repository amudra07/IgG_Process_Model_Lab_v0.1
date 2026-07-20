import pathlib
import sys

import numpy as np
import pandas as pd


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model import (  # noqa: E402
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


def test_pareto_filter_removes_known_dominated_point():
    points = np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [0.5, 3.0, 1.0]])
    mask = pareto_mask(points)
    assert mask.tolist() == [True, False, True]


def test_candidate_generator_has_expected_process_levels():
    candidates = make_candidates(n=80, seed=17)
    assert set(candidates["spray_flow_rpm"].unique()).issubset({20, 40})
    assert set(candidates["hardening_time_min"].unique()).issubset({10, 30, 60, 180, 300})
    assert set(candidates["drying_time_h"].unique()).issubset({8, 24, 48})

