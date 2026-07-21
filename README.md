# Thermicra IgG Formulation Model v0.3.1

Interactive Streamlit prototype for exploring four-excipient Box–Behnken response surfaces for empirical achievable IgG concentration and final HMW.

## What v0.3.1 changes

- Preserves the v0.3 concentration and HMW response-surface architecture.
- Adds supplied PVP K12 HMW anchors at 0, 15, 20 and 25 mg/mL.
- Adds milling retention as a separate output.
- Adds MCT-specific evidence labels for confirmed, boundary and extrapolated predictions.
- Keeps MCT fixed and does not introduce a vehicle-ranking model.
- Avoids treating the highest successful MCT loading as an exact physical maximum.

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

This is an empirical research prototype, not a validated manufacturing model.
