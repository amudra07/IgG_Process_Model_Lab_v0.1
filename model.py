"""Synthetic hybrid model for an IgG spray-dry/hardening workflow.

This module is deliberately a pipeline prototype.  The equations and generated
responses are illustrative and must not be interpreted as experimental evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor


RANDOM_SEED = 20260720


MOCK_RANGES = {
    "feed_igg_mg_ml": (45.0, 75.0),
    "sucrose_mg_ml": (0.0, 50.0),
    "trehalose_mg_ml": (0.0, 50.0),
    "hpbcd_mg_ml": (0.0, 75.0),
    "pvp_mg_ml": (0.0, 15.0),
    "feed_viscosity_mpas": (10.0, 80.0),
    "feed_hmw_pct": (0.30, 1.50),
    "feed_monomer_pct": (97.0, 99.4),
    "powder_added_mg": (500.0, 2_500.0),
    "mct_volume_ml": (0.75, 1.50),
}

SPRAY_LEVELS = (20, 40)
HARDENING_LEVELS = (10, 30, 60, 180, 300)
DRYING_LEVELS = (8, 24, 48)
EA_RATIO_LEVELS = (-1, 0, 1)

FIXED_PROCESS = {
    "inlet_temperature_c": 25.0,
    "outlet_temperature_c": 25.0,
    "hardening_temperature_c": 25.0,
    "ps80_hardening_mg_ml": 40.0,
    "feed_batch_volume_ml": 100.0,
}


QUALITY_FEATURES = [
    "feed_igg_mg_ml",
    "sucrose_mg_ml",
    "trehalose_mg_ml",
    "hpbcd_mg_ml",
    "pvp_mg_ml",
    "feed_viscosity_mpas",
    "spray_flow_rpm",
    "hardening_time_min",
    "log_hardening_time",
    "ea_powder_ratio_code",
    "drying_time_h",
    "total_solids_mg_ml",
    "stabilizer_to_igg_ratio",
]

VISCOSITY_FEATURES = QUALITY_FEATURES + [
    "powder_added_mg",
    "mct_volume_ml",
    "powder_loading_g_ml",
    "nominal_igg_mg_ml",
]


def _clip(value: np.ndarray, low: float, high: float) -> np.ndarray:
    return np.minimum(np.maximum(value, low), high)


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add deterministic formulation and mass-balance descriptors."""
    out = df.copy()
    excipient_cols = [
        "sucrose_mg_ml",
        "trehalose_mg_ml",
        "hpbcd_mg_ml",
        "pvp_mg_ml",
    ]
    out["total_excipient_mg_ml"] = out[excipient_cols].sum(axis=1)
    out["total_solids_mg_ml"] = (
        out["feed_igg_mg_ml"] + out["total_excipient_mg_ml"]
    )
    out["igg_dry_fraction"] = out["feed_igg_mg_ml"] / out[
        "total_solids_mg_ml"
    ].clip(lower=1e-9)
    out["stabilizer_to_igg_ratio"] = out["total_excipient_mg_ml"] / out[
        "feed_igg_mg_ml"
    ].clip(lower=1e-9)
    out["log_hardening_time"] = np.log1p(out["hardening_time_min"])
    out["powder_loading_g_ml"] = (
        out["powder_added_mg"] / 1_000.0 / out["mct_volume_ml"].clip(lower=1e-9)
    )
    out["nominal_igg_mg_ml"] = (
        out["powder_added_mg"]
        * out["igg_dry_fraction"]
        / out["mct_volume_ml"].clip(lower=1e-9)
    )

    feed_volume = FIXED_PROCESS["feed_batch_volume_ml"]
    out["initial_igg_mass_g"] = out["feed_igg_mg_ml"] * feed_volume / 1_000.0
    out["initial_dry_solids_mass_g"] = (
        out["total_solids_mg_ml"] * feed_volume / 1_000.0
    )

    # These are mock process-yield equations for demonstrating mass-balance flow.
    solids_centered = (out["total_solids_mg_ml"] - 130.0) / 100.0
    spray_yield = (
        0.78
        + 0.045 * (out["spray_flow_rpm"] == 40).astype(float)
        + 0.025 * np.tanh(solids_centered)
    )
    out["mock_spray_recovery_fraction"] = _clip(spray_yield.to_numpy(), 0.60, 0.93)

    hardening_penalty = (
        0.010 * np.abs(out["ea_powder_ratio_code"])
        + 0.018 * np.maximum(np.log1p(out["hardening_time_min"] / 60.0), 0)
    )
    out["mock_hardening_recovery_fraction"] = _clip(
        (0.965 - hardening_penalty).to_numpy(), 0.84, 0.98
    )
    drying_loss = 0.018 + 0.010 * (out["drying_time_h"] / 48.0)
    out["mock_drying_recovery_fraction"] = _clip(
        (1.0 - drying_loss).to_numpy(), 0.94, 0.99
    )

    out["mock_spray_powder_mass_g"] = (
        out["initial_dry_solids_mass_g"] * out["mock_spray_recovery_fraction"]
    )
    out["mock_final_powder_mass_g"] = (
        out["mock_spray_powder_mass_g"]
        * out["mock_hardening_recovery_fraction"]
        * out["mock_drying_recovery_fraction"]
    )
    out["powder_available_check"] = (
        out["powder_added_mg"] / 1_000.0 <= out["mock_final_powder_mass_g"]
    )
    return out


def _synthetic_truth(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Generate internally consistent, explicitly non-experimental responses."""
    out = add_derived_features(df)

    sugar_protection = np.log1p(
        (out["sucrose_mg_ml"] + out["trehalose_mg_ml"] + 0.75 * out["hpbcd_mg_ml"])
        / 45.0
    )
    pvp_optimum = np.exp(-((out["pvp_mg_ml"] - 6.0) / 5.0) ** 2)
    shear_stress = (out["spray_flow_rpm"] == 40).astype(float)
    hardening_distance = np.abs(
        np.log1p(out["hardening_time_min"]) - np.log1p(60.0)
    )
    drying_short = np.maximum((24.0 - out["drying_time_h"]) / 16.0, 0)
    drying_long = np.maximum((out["drying_time_h"] - 24.0) / 24.0, 0)
    viscosity_stress = np.maximum((out["feed_viscosity_mpas"] - 30.0) / 50.0, 0)

    delta_hmw = (
        0.32
        + 0.16 * shear_stress
        + 0.16 * hardening_distance
        + 0.13 * np.abs(out["ea_powder_ratio_code"])
        + 0.13 * drying_short
        + 0.10 * drying_long
        + 0.18 * viscosity_stress
        - 0.21 * sugar_protection
        - 0.07 * pvp_optimum
        + rng.normal(0.0, 0.055, len(out))
    )
    delta_hmw = _clip(delta_hmw.to_numpy(), 0.02, 3.5)

    non_hmw_loss = (
        0.10
        + 0.045 * shear_stress
        + 0.065 * drying_long
        + 0.035 * hardening_distance
        + rng.normal(0.0, 0.035, len(out))
    )
    non_hmw_loss = _clip(np.asarray(non_hmw_loss), 0.02, 1.2)
    delta_monomer = -(delta_hmw + non_hmw_loss)

    out["delta_hmw_pct"] = delta_hmw
    out["final_hmw_pct"] = _clip(
        out["feed_hmw_pct"].to_numpy() + delta_hmw, 0.02, 10.0
    )
    provisional_monomer = out["feed_monomer_pct"].to_numpy() + delta_monomer
    out["final_monomer_pct"] = np.minimum(
        _clip(provisional_monomer, 85.0, 99.7),
        99.8 - out["final_hmw_pct"].to_numpy(),
    )
    out["delta_monomer_pct"] = (
        out["final_monomer_pct"] - out["feed_monomer_pct"]
    )

    loading_scaled = out["powder_loading_g_ml"] / 2.0
    morphology_factor = (
        0.10 * shear_stress
        + 0.09 * np.abs(out["ea_powder_ratio_code"])
        + 0.07 * hardening_distance
        - 0.0015 * out["hpbcd_mg_ml"]
    )
    log_viscosity = (
        np.log(24.0)
        + 1.05 * loading_scaled
        + 1.35 * loading_scaled**2
        + 0.0045 * (out["feed_viscosity_mpas"] - 25.0)
        + morphology_factor
        + rng.normal(0.0, 0.08, len(out))
    )
    out["log_final_viscosity"] = log_viscosity
    out["final_viscosity_mpas"] = _clip(
        np.exp(log_viscosity.to_numpy()), 10.0, 100_000.0
    )
    return out


def generate_mock_data(n: int = 700, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Generate anonymized mock batches spanning the provisional design space."""
    rng = np.random.default_rng(seed)
    feed_igg = rng.uniform(*MOCK_RANGES["feed_igg_mg_ml"], n)
    sucrose = rng.uniform(*MOCK_RANGES["sucrose_mg_ml"], n)
    trehalose = rng.uniform(*MOCK_RANGES["trehalose_mg_ml"], n)
    hpbcd = rng.uniform(*MOCK_RANGES["hpbcd_mg_ml"], n)
    pvp = rng.uniform(*MOCK_RANGES["pvp_mg_ml"], n)

    base_viscosity = (
        9.0
        + 0.30 * feed_igg
        + 0.08 * (sucrose + trehalose + hpbcd)
        + 0.75 * pvp
    )
    feed_viscosity = _clip(
        base_viscosity + rng.normal(0.0, 4.5, n),
        *MOCK_RANGES["feed_viscosity_mpas"],
    )
    feed_hmw = rng.uniform(*MOCK_RANGES["feed_hmw_pct"], n)
    feed_monomer_upper = np.minimum(
        MOCK_RANGES["feed_monomer_pct"][1], 99.75 - feed_hmw
    )
    feed_monomer = rng.uniform(MOCK_RANGES["feed_monomer_pct"][0], feed_monomer_upper)

    df = pd.DataFrame(
        {
            "batch_id": [f"MOCK-{i + 1:04d}" for i in range(n)],
            "feed_igg_mg_ml": feed_igg,
            "sucrose_mg_ml": sucrose,
            "trehalose_mg_ml": trehalose,
            "hpbcd_mg_ml": hpbcd,
            "pvp_mg_ml": pvp,
            "feed_viscosity_mpas": feed_viscosity,
            "spray_flow_rpm": rng.choice(SPRAY_LEVELS, n),
            "hardening_time_min": rng.choice(HARDENING_LEVELS, n),
            "ea_powder_ratio_code": rng.choice(EA_RATIO_LEVELS, n),
            "drying_time_h": rng.choice(DRYING_LEVELS, n),
            "feed_hmw_pct": feed_hmw,
            "feed_monomer_pct": feed_monomer,
            "mct_volume_ml": rng.uniform(*MOCK_RANGES["mct_volume_ml"], n),
        }
    )
    interim = add_derived_features(
        df.assign(powder_added_mg=np.full(n, 1_000.0))
    )
    target_concentration = rng.uniform(500.0, 700.0, n)
    df["powder_added_mg"] = (
        target_concentration
        * df["mct_volume_ml"]
        / interim["igg_dry_fraction"]
    )
    return _synthetic_truth(df, rng)


def _tree_quantiles(
    model: ExtraTreesRegressor,
    x: pd.DataFrame,
    lower: float = 0.05,
    upper: float = 0.95,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_values = x.to_numpy()
    tree_predictions = np.column_stack(
        [estimator.predict(x_values) for estimator in model.estimators_]
    )
    return (
        np.mean(tree_predictions, axis=1),
        np.quantile(tree_predictions, lower, axis=1),
        np.quantile(tree_predictions, upper, axis=1),
    )


@dataclass
class ModelBundle:
    hmw_delta_model: ExtraTreesRegressor
    monomer_delta_model: ExtraTreesRegressor
    log_viscosity_model: ExtraTreesRegressor

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        enriched = add_derived_features(frame)

        hmw_delta, hmw_lo_delta, hmw_hi_delta = _tree_quantiles(
            self.hmw_delta_model, enriched[QUALITY_FEATURES]
        )
        mon_delta, mon_lo_delta, mon_hi_delta = _tree_quantiles(
            self.monomer_delta_model, enriched[QUALITY_FEATURES]
        )
        log_visc, log_visc_lo, log_visc_hi = _tree_quantiles(
            self.log_viscosity_model, enriched[VISCOSITY_FEATURES]
        )

        hmw = np.maximum(enriched["feed_hmw_pct"].to_numpy() + hmw_delta, 0.0)
        monomer = enriched["feed_monomer_pct"].to_numpy() + mon_delta
        monomer = np.minimum(monomer, 99.8 - hmw)

        result = pd.DataFrame(index=enriched.index)
        result["nominal_igg_mg_ml"] = enriched["nominal_igg_mg_ml"]
        result["final_hmw_pct"] = hmw
        result["final_hmw_p05"] = np.maximum(
            enriched["feed_hmw_pct"].to_numpy() + hmw_lo_delta, 0.0
        )
        result["final_hmw_p95"] = np.maximum(
            enriched["feed_hmw_pct"].to_numpy() + hmw_hi_delta, 0.0
        )
        result["final_monomer_pct"] = monomer
        result["final_monomer_p05"] = np.maximum(
            enriched["feed_monomer_pct"].to_numpy() + mon_lo_delta, 0.0
        )
        result["final_monomer_p95"] = np.minimum(
            enriched["feed_monomer_pct"].to_numpy() + mon_hi_delta, 100.0
        )
        result["final_viscosity_mpas"] = np.exp(log_visc)
        result["final_viscosity_p05"] = np.exp(log_visc_lo)
        result["final_viscosity_p95"] = np.exp(log_visc_hi)
        result["powder_available_check"] = enriched["powder_available_check"].to_numpy()
        return result


def train_mock_models(data: pd.DataFrame, seed: int = RANDOM_SEED) -> ModelBundle:
    """Train flexible surrogate models on the synthetic responses."""
    params = dict(
        n_estimators=240,
        min_samples_leaf=3,
        max_features=0.85,
        random_state=seed,
        n_jobs=-1,
    )
    hmw_model = ExtraTreesRegressor(**params).fit(
        data[QUALITY_FEATURES], data["delta_hmw_pct"]
    )
    monomer_model = ExtraTreesRegressor(**params).fit(
        data[QUALITY_FEATURES], data["delta_monomer_pct"]
    )
    viscosity_model = ExtraTreesRegressor(**params).fit(
        data[VISCOSITY_FEATURES], data["log_final_viscosity"]
    )
    return ModelBundle(hmw_model, monomer_model, viscosity_model)


def default_batch() -> dict[str, float]:
    """Return an anonymized center-point batch for the interactive explorer."""
    batch = {
        "feed_igg_mg_ml": 60.0,
        "sucrose_mg_ml": 25.0,
        "trehalose_mg_ml": 25.0,
        "hpbcd_mg_ml": 35.0,
        "pvp_mg_ml": 6.0,
        "feed_viscosity_mpas": 38.0,
        "spray_flow_rpm": 20,
        "hardening_time_min": 60,
        "ea_powder_ratio_code": 0,
        "drying_time_h": 24,
        "feed_hmw_pct": 0.70,
        "feed_monomer_pct": 98.70,
        "mct_volume_ml": 1.0,
    }
    temp = add_derived_features(
        pd.DataFrame([{**batch, "powder_added_mg": 1_000.0}])
    )
    igg_fraction = float(temp.loc[0, "igg_dry_fraction"])
    batch["powder_added_mg"] = 600.0 / igg_fraction
    return batch


def make_candidates(n: int = 2_000, seed: int = RANDOM_SEED + 11) -> pd.DataFrame:
    """Generate candidate conditions inside the mock formulation domain."""
    rng = np.random.default_rng(seed)
    frame = generate_mock_data(n=n, seed=seed)
    # Candidate recommendations use noiseless inputs; synthetic observed outputs
    # are discarded and replaced by surrogate-model predictions in the app.
    keep = [
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
    candidates = frame[keep].copy()
    candidates.insert(0, "candidate_id", [f"C-{i + 1:04d}" for i in range(n)])
    return candidates.sample(frac=1.0, random_state=int(rng.integers(0, 1_000_000))).reset_index(drop=True)


def pareto_mask(values: np.ndarray) -> np.ndarray:
    """Return mask for rows that are non-dominated under minimization."""
    points = np.asarray(values, dtype=float)
    efficient = np.ones(points.shape[0], dtype=bool)
    for i, point in enumerate(points):
        if not efficient[i]:
            continue
        active = np.where(efficient)[0]
        worse = np.all(points[active] >= point, axis=1) & np.any(
            points[active] > point, axis=1
        )
        efficient[active[worse]] = False
    return efficient


def sensitivity_frame(
    bundle: ModelBundle,
    base_batch: dict[str, float],
    variable: str,
    values: Iterable[float],
) -> pd.DataFrame:
    rows = []
    for value in values:
        row = base_batch.copy()
        row[variable] = value
        rows.append(row)
    frame = pd.DataFrame(rows)
    prediction = bundle.predict(frame)
    return pd.concat(
        [frame[[variable]].reset_index(drop=True), prediction.reset_index(drop=True)],
        axis=1,
    )
