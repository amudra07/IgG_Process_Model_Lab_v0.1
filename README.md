# IgG Process Model Lab

Interactive Streamlit prototype for a high-concentration IgG formulation process:

`feed formulation → spray drying → particle hardening → final drying → MCT suspension`

## Prototype status (v0.3.2 in the original v0.3 format)

The process-response models, ranges, uncertainty bands, and recommendations are
synthetic. The concentration module is now a hybrid: deterministic mass balance
plus a bounded empirical sucrose/HPBCD capacity surface fitted to six supplied
rows with reported component concentrations. A broader 25-row formulation
screen is included as evidence, with missing concentrations preserved rather
than imputed. This is calibration, not independent validation.

Version 0.3.2 keeps the complete v0.3 interface and end-to-end simulator while
separating the predictive workflow into four unit-operation scopes. The former
20/40 `spray_flow_rpm` field is corrected into independent ultrasonic power and
feed-flow controls. The app distinguishes calculations, real local anchors,
provisional assumptions, and outputs that are not yet calibrated. Soluplus is
not an active model variable in v0.3.2.

## Modular model scopes

1. Formulation stability: sucrose, trehalose, HPBCD, and PVP.
2. Ultrasonic spray drying: ultrasonic power, feed flow, viscosity, and total solids.
3. EA hardening/final drying: EA:powder ratio, hardening time, and drying time.
4. Milling/MCT loading: PVP, powder loading, and MCT volume.

Selecting one scope locks unrelated variables at the reference condition. The
full process simulator keeps all stages connected.

## Defined outputs

- Nominal IgG concentration: deterministic mass balance using theoretical IgG
  fraction, powder mass added, and added MCT volume.
- Predicted achievable IgG concentration: the smaller of nominal concentration
  and the empirical formulation capacity.
- Aggregation: final SEC-HPLC HMW area% plus process-induced ΔHMW from the feed baseline.
- Monomer: final SEC-HPLC monomer area% plus process-induced Δmonomer from the feed baseline.
- Viscosity: powder dispersed in MCT, measured at 25 °C and 100 s⁻¹.
- Milling recovery: interpolated only within supplied 15, 20, and 25 mg/mL PVP
  observations; otherwise reported as not calibrated.
- MCT loading evidence: supported, boundary, extrapolated, or composition-
  uncertain relative to successful 400 and 550 mg/mL anchors. These are lower
  bounds, not exact maximum concentrations.
- Ultrasonic spray block: provisional recovery and drying-severity descriptors;
  spray-specific aggregation and particle size remain explicitly uncalibrated.

## Run locally

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## GitHub / Streamlit deployment

Replace all repository files from this v0.3 package together, especially both
`app.py` and `model.py`, then reboot the Streamlit app. The app includes a
versioned cache key and a prediction-schema adapter so an older cached v0.1
model cannot cause missing v0.3 concentration fields during startup.

macOS/Linux:

```bash
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Files

- `app.py`: interactive user interface.
- `model.py`: mock-data generator, mass balance, surrogate models, sensitivity,
  and Pareto calculations.
- `tests/test_model.py`: smoke and invariant tests.
- `LITERATURE_BASIS.md`: evidence, model translation, and transfer limitations.
- `CHANGELOG_v0.3.2.md`: implemented changes and remaining calibration gates.
- `Summary Prototype v0.3.2 IgG model.html`: standalone architecture summary.
- `data/ultrasonic_power_observations.csv`: supplied 20/40% power observations.
- `requirements.txt`: Python dependencies.

## Current model domain

- Ultrasonic transducer power: 20% or 40%.
- Feed flow: 20, 40, or 60 rpm.
- Hardening time: 10, 30, 60, 180, or 300 minutes.
- Final pressure-drying time: 8, 24, or 48 hours at room temperature.
- Hardening medium: ethyl acetate with PS80 fixed at 40 mg/mL.
- Ethyl-acetate-to-powder ratio: provisional coded low/center/high levels.
- Achievable IgG exploration domain: 250–650 mg/mL.

## Empirical capacity equation

The earlier one-direction sucrose×HPBCD bonus is replaced by inverse-distance
interpolation over six quantitative supplied observations:

```text
weight_i = 1 / (scaled_distance_i^2 + 0.04)^2
capacity = sum(weight_i * observed_capacity_i) / sum(weight_i)
achievable concentration = min(nominal mass-balance concentration, capacity)
```

Exact experimental combinations reproduce their reported capacity. The result
is bounded to 250–650 mg/mL, and the app reports distance-based experimental
support, nearest-anchor distance, supporting-observation count, and
interpolation/extrapolation status. None is a validated confidence interval.

## Ultrasonic spray transformation block

The spray stage is represented as a reactor-like material transformation:

```text
powder_out = spray_recovery × dry_solids_in
moisture_out = equilibrium_moisture + (moisture_in - equilibrium_moisture) × exp(-K_dry)
monomer_out = monomer_in × exp(-K_agg)
```

The current recovery and `K_dry` relationships are provisional severity
descriptors. `K_agg` is deliberately returned as not calibrated because no
paired SEC-HPLC measurement immediately before and after spraying is available.
The higher-power/smaller-droplet relationship also remains a mechanistic
hypothesis until droplet or particle-size measurements are supplied.

The four supplied 20%/40% power observations are stored in
`data/ultrasonic_power_observations.csv`. They support the observed 40% nozzle-
heating/clogging warning but are not used as a universal atomization coefficient.

## Evidence display

The v0.3.1 synthetic P05–P95 bands were removed from the interface because
variation among synthetic trees is not experimental uncertainty. v0.3.2 labels
each result as mass balance, empirical anchor/local interpolation, provisional
assumption, or not calibrated.

## Replacement with real data

The first real-data cycle is expected to contain fewer than 15 batches. Do not
use deep learning for that dataset. Begin with mass-balance equations plus a
regularized linear or Gaussian-process model, use leave-one-batch-out validation,
and retain feed SEC-HPLC results as baseline predictors. To identify the spray-
specific aggregation coefficient, collect SEC-HPLC immediately before and after
spraying; otherwise only combined downstream damage can be estimated.
