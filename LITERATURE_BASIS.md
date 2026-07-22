# Evidence basis for the v0.3.2 modular hybrid model in the original v0.3 architecture

## v0.3.2 supplied study update

The complete v0.3 process model is retained. The new study adds four PVP HMW
anchors (0, 15, 20, and 25 mg/mL), three milling-recovery anchors (15, 20, and
25 mg/mL), a categorical water-reconstitution observation, and successful MCT
loading anchors at 400 mg/mL for sucrose and 550 mg/mL for sucrose + HPBCD.

The HMW anchors are blended with the existing v0.3 quality prediction. Milling
recovery is interpolated only within its measured domain. The MCT successes are
reported as confirmed lower bounds rather than exact maximum concentrations.
This prevents the new evidence from being overextended into a fitted physical-
failure probability or a universal vehicle-capacity equation.

The supplied ultrasonic-spray table also establishes that transducer power (%)
and feed flow (rpm) are independent controls. Earlier source code incorrectly
used the 20/40 values as spray flow. v0.3.2 separates 20/40% ultrasonic power
from 20/40/60 rpm feed flow. The 40% observations support a nozzle-heating and
clogging warning; they do not yet establish a particle-size or kinetic equation.

Soluplus is not activated as a model variable in v0.3.2.

## Modular unit-operation strategy

The full process map is retained, but influence analysis is separated into four
models: formulation stability, ultrasonic spray drying, hardening/final drying,
and milling/MCT loading. Only variables from the selected unit operation are
adjustable in that scope. The full simulator connects the output material state
from one block to the next.

The Fiedler et al. (2023) strategy is used as an architectural lesson: learn
physical spray-drying responses using lower-cost surrogate experiments, then
bridge selected conditions to actual protein. It does not provide Thermicra
coefficients. Accordingly, recovery, moisture, particle size, morphology, and
deposition belong to the physical spray model, whereas IgG aggregation requires
actual-IgG input/output SEC-HPLC measurements.

## v0.3 challenge result

The new supplied screen contradicts a universally positive sucrose–HPBCD
interaction. At 5 mg/mL HPBCD, reported suspension capacity was 600 mg/mL with
no sucrose, 400 mg/mL with 1.25 mg/mL sucrose, and 500 mg/mL with 2.5 or
5 mg/mL sucrose. Version 0.3 therefore replaces the prior directional bonus with
a bounded empirical surface. Journal mechanisms remain explanatory priors for
quality responses, not concentration multipliers.

## Scientific interpretation

The supplied 400 mg/mL (sucrose) and 550 mg/mL (sucrose + HPBCD) results show
that a composition-only mass balance is not sufficient to describe achievable
MCT suspension loading. The prototype therefore separates:

1. nominal concentration from mass balance; and
2. formulation-enabled capacity from excipient interactions.

Published results are used as mechanistic priors, not as directly pooled
training rows, because protein identity, concentration, drying method, stress,
and measurement endpoint differ from the target process.

## Evidence mapped into the model

| Evidence | Model use | Limitation |
|---|---|---|
| Zhang et al. (2023) reported saturable plus nonspecific mAb–sucrose interactions, including an average sucrose Kd of 88.63 mM at 293 K. | Sucrose occupancy term `C/(Kd+C)`. | Different mAbs and solution conditions; Kd is a prior, not a target-process estimate. |
| Serno et al. (2010) reported complete suppression of agitation-induced IgG aggregation at 2.5 mM HPBCD, primarily through competition at the air–water interface. | A Hill term centered at 2.5 mM represents threshold-like interfacial protection. | Aggregation under agitation is not the same endpoint as achievable loading in MCT. |
| Härtl et al. (2013) found concentration-, stress-, and antibody-dependent HPBCD effects, including possible thermal destabilization. | Prevents assuming a universally positive linear HPBCD effect. | No universal coefficient can be transferred. |
| Tam et al. (2025) found that HPBCD combined with trehalose inhibited trehalose recrystallization and improved stability of spray-dried mAb formulations versus trehalose alone. | Supports explicit combination features and future trehalose×HPBCD testing. | Protein:excipient ratio and formulation conditions govern the effect. |
| Jiang et al. (2021) reported that PVP reduced turbidity and increased IgG recovery in a spray-layering study, with sugars providing further benefit. | Retains PVP as a potential main effect and interaction input. | Spray layering is not the current spray-dry/hardening process. |
| Fiedler et al. (2023) used staged surrogate experiments to map spray-drying CQAs before protein verification. | Supports a distinct spray transformation block and staged surrogate-to-protein calibration. | Lactose/HSA results cannot supply target-process IgG coefficients. |

## Primary sources

- Zhang et al., *mAbs* (2023): https://doi.org/10.1080/19420862.2023.2212416
- Serno et al., *Journal of Pharmaceutical Sciences* (2010): https://doi.org/10.1002/jps.21931
- Härtl et al., *Journal of Pharmaceutical Sciences* (2013): https://doi.org/10.1002/jps.23729
- Tam et al., *Molecular Pharmaceutics* (2025): https://doi.org/10.1021/acs.molpharmaceut.5c00639
- Jiang et al., *International Journal of Pharmaceutics* (2021): https://pubmed.ncbi.nlm.nih.gov/34748814/
- Fiedler et al., *International Journal of Pharmaceutics* (2023): https://doi.org/10.1016/j.ijpharm.2023.123133
- Ramezani et al., *Drug Development and Industrial Pharmacy* (2017): https://doi.org/10.1080/03639045.2017.1293679

## What the two supplied observations establish

The sucrose-only reference fixes the baseline capacity at 400 mg/mL. The
sucrose+HPBCD reference fixes the combination coefficient so the model returns
550 mg/mL. Because the same observations were used to estimate the equation,
agreement at those two points is calibration, not predictive validation.

## Data needed before claiming predictive accuracy

- Exact sucrose and HPBCD feed concentrations for both supplied formulations.
- HPBCD grade, substitution range, and certificate molecular weight.
- Replicate achievable-loading measurements and a written endpoint definition.
- Intermediate sucrose/HPBCD ratios, plus sucrose-only and HPBCD-only controls.
- Trehalose and PVP main-effect and interaction arms.
- Assay recovery, moisture, bulk/tapped density, particle-size distribution, and
  MCT viscosity at 25 °C and 100 s^-1.
- Feed and post-drying SEC-HPLC HMW/monomer for each batch.
- Immediate post-spray SEC-HPLC HMW/monomer to isolate `K_agg,SD`.
- Matched ultrasonic power, feed flow, nozzle temperature, recovery, moisture,
  particle size, morphology, and deposition measurements for each spray batch.
- A held-out set of batches that is not used to fit coefficients.

## Evidence and uncertainty display

Synthetic tree-to-tree P05–P95 ranges are not experimental confidence
intervals and are therefore removed from the v0.3.2 interface. The app instead
reports nearest-anchor distance, number of quantitative formulation anchors,
interpolation versus extrapolation, and evidence class. Replicate batches are
required before statistical confidence or prediction intervals can be claimed.
