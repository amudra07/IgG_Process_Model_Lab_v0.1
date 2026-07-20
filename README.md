# IgG Process Model Lab

Interactive Streamlit prototype for a high-concentration IgG formulation process:

`feed formulation → spray drying → particle hardening → final drying → MCT suspension`

## Important limitation

Every training batch, formulation range, response equation, uncertainty band, and
recommendation in this prototype is synthetic. The application demonstrates a
modeling workflow; it is not an experimentally validated formulation model and
must not be used as manufacturing evidence.

## Defined outputs

- Nominal IgG concentration: deterministic mass balance using theoretical IgG
  fraction, powder mass added, and added MCT volume.
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
- `requirements.txt`: Python dependencies.

## Current model domain

- Spray flow: 20 or 40 rpm.
- Hardening time: 10, 30, 60, 180, or 300 minutes.
- Final pressure-drying time: 8, 24, or 48 hours at room temperature.
- Hardening medium: ethyl acetate with PS80 fixed at 40 mg/mL.
- Ethyl-acetate-to-powder ratio: provisional coded low/center/high levels.
- Nominal IgG suspension concentration: 500–700 mg/mL.

## Replacement with real data

The first real-data cycle is expected to contain fewer than 15 batches. Do not
use deep learning for that dataset. Begin with mass-balance equations plus a
regularized linear or Gaussian-process model, use leave-one-batch-out validation,
and retain feed SEC-HPLC results as baseline predictors.

