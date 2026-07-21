# Thermicra IgG Formulation Model v0.3.1

Interactive Streamlit prototype for exploring four-excipient Box–Behnken response surfaces for empirical achievable IgG concentration and final HMW.

## What v0.3.1 changes

- Preserves the v0.3 concentration and HMW response-surface architecture.
- Adds supplied PVP K12 HMW anchors at 0, 15, 20 and 25 mg/mL.
- Keeps the Streamlit interface limited to four excipient inputs, two prediction outputs and two heatmaps.
- Moves equations, milling-recovery evidence, MCT support limits and update details to the standalone HTML report.
- Keeps MCT fixed and does not introduce a vehicle-ranking model.
- Fixes other process conditions at the current optimized settings.

## Interface scope

The app intentionally contains no mathematical equations, verification tables, update log or detailed scientific report. See `Thermicra_v0.3.1_update_card.html` for those details.

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

This is an empirical research prototype, not a validated manufacturing model.
