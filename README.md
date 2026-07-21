# IgG Process Model Lab

Interactive Streamlit prototype for a high-concentration IgG formulation process:

`feed formulation → spray drying → particle hardening → final drying → MCT suspension`

## Prototype status (v0.3.1 in the original v0.3 format)

The process-response models, ranges, uncertainty bands, and recommendations are
synthetic. The concentration module is now a hybrid: deterministic mass balance
plus a bounded empirical sucrose/HPBCD capacity surface fitted to six supplied
rows with reported component concentrations. A broader 25-row formulation
screen is included as evidence, with missing concentrations preserved rather
than imputed. This is calibration, not independent validation.

Version 0.3.1 keeps the complete v0.3 process interface and calculation stack.
It adds a PVP-specific HMW evidence blend, milling-recovery interpolation inside
the measured 15--25 mg/mL PVP range, reconstitution evidence status, and a
conservative MCT loading-support classification. It does not replace the app
with the later four-factor Box--Behnken interface.

## Defined outputs

- Nominal IgG concentration: deterministic mass balance using theoretical IgG
  fraction, powder mass added, and added MCT volume.
- Predicted achievable IgG concentration: the smaller of nominal concentration
  and the empirical formulation capacity.
- Aggregation: SEC-HPLC HMW area% after final drying.
- Monomer: SEC-HPLC monomer area% after final drying.
- Viscosity: powder dispersed in MCT, measured at 25 °C and 100 s⁻¹.
- Milling recovery: interpolated only within supplied 15, 20, and 25 mg/mL PVP
  observations; otherwise reported as not calibrated.
- MCT loading evidence: supported, boundary, extrapolated, or composition-
  uncertain relative to successful 400 and 550 mg/mL anchors. These are lower
  bounds, not exact maximum concentrations.

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
- `requirements.txt`: Python dependencies.

## Current model domain

- Spray flow: 20 or 40 rpm.
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
support. This support score is not a validated confidence interval.

## Replacement with real data

The first real-data cycle is expected to contain fewer than 15 batches. Do not
use deep learning for that dataset. Begin with mass-balance equations plus a
regularized linear or Gaussian-process model, use leave-one-batch-out validation,
and retain feed SEC-HPLC results as baseline predictors.
