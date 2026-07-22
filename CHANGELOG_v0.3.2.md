# Thermicra IgG Process Model v0.3.2

## Approved changes implemented

- Preserved the original v0.3 visual layout and complete end-to-end simulator.
- Added four selectable unit-operation scopes; unrelated controls lock at the
  reference condition during a stage-specific study.
- Corrected the former 20/40 spray-flow field into independent ultrasonic
  transducer power (20/40%) and feed flow (20/40/60 rpm).
- Kept temperature separate from ultrasonic power and added the observed 40%
  nozzle-heating/clogging warning.
- Added a reactor-like spray transformation block with separate provisional
  recovery and drying-severity outputs.
- Left spray-specific aggregation severity and particle-size relationships
  explicitly uncalibrated pending post-spray SEC-HPLC and particle measurements.
- Exposed ΔHMW and Δmonomer relative to feed quality.
- Replaced synthetic P05–P95 bands with nearest-anchor distance, six-observation
  count, interpolation/extrapolation status, and evidence classification.
- Retained the six-point sucrose–HPBCD surface as local interpolation rather
  than a universal capacity mechanism.
- Kept PVP aggregation, milling recovery, reconstitution, and MCT loading as
  separate evidence relationships.
- Added the supplied ultrasonic-power observations as a traceable CSV without
  fitting a universal power coefficient.
- Updated tests, user documentation, literature basis, and standalone HTML summary.

## Explicitly excluded

- Soluplus was not added as an active formulation variable or predictive model
  input. Historical screen rows remain in the evidence CSV only and are not
  used quantitatively.

## Calibration gates still open

- Immediate pre-/post-spray SEC-HPLC for `K_agg,SD`.
- Matched power, flow, feed properties, recovery, moisture, particle-size,
  morphology, deposition, and nozzle-temperature results.
- Replicates for statistical uncertainty and held-out validation batches.
