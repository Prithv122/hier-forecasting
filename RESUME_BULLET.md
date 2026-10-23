# Resume Bullets — Hierarchical Forecasting

Form: **action → technical specifics → measured outcome.**

---

## Bullets

- Pre-registered and ran a 12-origin rolling backtest of 4 base forecasters (ETS, global
  LightGBM, hybrid, seasonal naive) × 5 reconcilers on Australian tourism's 111-series hierarchy;
  found MinT reconciliation made the national total 9% worse (MASE 0.990 → 1.083, 95% CI
  +0.042 to +0.147) while improving regions by under 1%.
- Implemented bottom-up, top-down, OLS, WLS and MinT-shrink reconciliation from the linear algebra
  (Schäfer–Strimmer covariance shrinkage), matching Nixtla's `hierarchicalforecast` to 4e-9, with a
  leakage test that scrambles all post-origin data and requires byte-identical forecasts.
- Built a seeded synthetic hierarchy that isolates error correlation; an independent review caught
  a confound in it (unequal noise persistence), fixed by a logged amendment before any result was
  read.

## Which roles this supports

- [x] Data Scientist / ML
- [ ] AI Engineer (LLM/NLP/CV)
- [ ] Data Engineer
- [x] Data Analyst / Python Developer

## Keywords this project earns

Hierarchical forecasting, forecast reconciliation, MinT, time-series cross-validation
(rolling-origin backtest), MASE / RMSSE, ETS (statsforecast), LightGBM global models, direct
multi-step forecasting, bootstrap confidence intervals, pre-registration, Python, pandas, numpy.
