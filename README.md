# IgG Process Model Lab

Interactive Streamlit prototype for a high-concentration IgG formulation process:

`feed formulation → spray drying → particle hardening → final drying → MCT suspension`

## Prototype status (v0.2)

The process-response models, ranges, uncertainty bands, and recommendations are
synthetic. The concentration module is now a hybrid: deterministic mass balance
plus a literature-informed sucrose/HPBCD interaction equation calibrated to the
two supplied observations (400 and 550 mg/mL). This is a calibration check, not
independent validation, and must not be used as manufacturing evidence.

## Defined outputs

- Nominal IgG concentration: deterministic mass balance using theoretical IgG
  fraction, powder mass added, and added MCT volume.
- Predicted achievable IgG concentration: the smaller of nominal concentration
  and the interaction-adjusted formulation capacity.
- Aggregation: SEC-HPLC HMW area% after final drying.
- Monomer: SEC-HPLC monomer area% after final drying.
- Viscosity: powder dispersed in MCT, measured at 25 °C and 100 s⁻¹.

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

Replace all repository files from this v0.2 package together, especially both
`app.py` and `model.py`, then reboot the Streamlit app. The app includes a
versioned cache key and a prediction-schema adapter so an older cached v0.1
model cannot cause missing v0.2 concentration fields during startup.

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
- Nominal IgG suspension concentration: 500–700 mg/mL.

## Interaction equation

Sucrose is represented by a saturable binding term and HPBCD by a threshold-like
interfacial-protection term:

```text
theta_suc = C_suc / (88.63 mM + C_suc)
theta_HP  = C_HP^2 / ((2.5 mM)^2 + C_HP^2)
I         = (theta_suc * theta_HP) / I_reference
capacity  = 400 * (w_IgG * assay_recovery)/(0.81 * 0.887) * (1 + beta * I)
achievable concentration = min(nominal mass-balance concentration, capacity)
```

`beta = 0.42246` is fitted so that the reference sucrose/HPBCD formulation gives
550 mg/mL. Exact feed concentrations and the grade-specific HPBCD molecular
weight are still needed to replace the anonymized reference normalization.

## Replacement with real data

The first real-data cycle is expected to contain fewer than 15 batches. Do not
use deep learning for that dataset. Begin with mass-balance equations plus a
regularized linear or Gaussian-process model, use leave-one-batch-out validation,
and retain feed SEC-HPLC results as baseline predictors.
