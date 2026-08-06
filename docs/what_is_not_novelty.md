# What Is Not Novelty

This file enumerates methodological choices and comparisons that are
established practice. None of these may be presented as original contributions
of this project.

## Validation protocol

- **Rolling-origin cross-validation** (also called expanding-window
  cross-validation): standard temporal validation design; no novelty claim.
- **Train-only preprocessing within each fold**: required hygiene, not novelty.

## Post-processing methodology

- **Model Output Statistics (MOS)**: operational since the 1970s.
- **Ensemble MOS (EMOS)**: Gneiting et al. (2005) and subsequent literature.
- **Bias correction of NWP/AI model output**: decades of operational use.
- **Residual regression on NWP output**: standard downscaling/correction.

## Machine learning estimators

- **LightGBM, XGBoost, or gradient-boosting** for forecast correction: widely
  published in air quality post-processing.
- **Linear regression with lagged predictors**: classical MOS.

## Evaluation design

- **Station-scale evaluation of NWP or AI model output**: standard operational
  meteorology and air quality science.
- **Comparison of raw vs corrected model output**: prior published comparisons
  exist for Aurora and comparable AI models.

## Observed model behaviour

- **Aurora smoothing or underestimation of peak concentrations**: expected and
  documented behaviour of probabilistic AI weather models trained with L2/MAE
  loss; not a new finding.

## Detection methodology

- **Fixed or percentile-based thresholds for episode detection**: standard air
  quality practice (WHO, EU directive thresholds; percentile surrogates).

## Regional scope

- **Comunitat Valenciana / Mediterranean Spain**: prior regional air quality
  studies and operational post-processing work exist; regional application is
  not itself a novelty.

## Compound comparisons

- **Aurora raw vs Aurora corrected**: straightforward comparison; prior
  published work covers analogous set-ups for other AI NWP models.
- **Aurora raw vs CAMS raw**: head-to-head benchmark, not a methodological
  contribution.

---

*Any element on this list that appears in a draft manuscript must be cited
as prior art, not presented as a project contribution.*
