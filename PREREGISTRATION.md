# Pre-registration

Written and committed before any forecaster, reconciler or backtest code exists, and before any
data is downloaded. The frozen values are also in `src/hierfc/spec.py`. Any change after this
commit is an amendment: dated, given a reason, appended at the end of this file, and committed
**before** the run it governs.

## 1. Question

Under an expanding-window rolling-origin backtest, does reconciling base forecasts lower the
error at **every** level of a hierarchy compared with the unreconciled base forecasts, or does it
help the top and hurt the bottom?

## 2. Data

| Arm | Role | Hierarchy | Series | Frequency |
|---|---|---|---|---|
| `tourism` | **confirmatory** (H1, H2) | total / state / zone / region | 1 + 7 + 27 + 76 = 111 | monthly, 1998–2016 |
| `synth-correlated` | mechanism (H3) | total / region / store / category | 1 + 5 + 20 + 60 = 86 | monthly, 144 months |
| `synth-independent` | control (H3) | same | 86 | same |

- **Tourism:** Australian domestic visitor nights (Tourism Research Australia, National Visitor
  Survey), the data of Wickramasuriya, Athanasopoulos & Hyndman (2019). Bottom series are the 76
  regions, each summed over travel purpose; the purpose dimension is not used. Licence and exact
  source are recorded in the README before the first download. The data is never committed.
- **Synthetic:** `synth.py`, seed `SYNTH_SEED = 31`. Each bottom series is a positive level ×
  trend × monthly seasonal pattern, times `exp(noise)`. The noise sd is the same (0.15) in both
  configs; in `correlated` 45% of its variance comes from an AR(1) shock shared by the region and
  25% from one shared by the store, in `independent` none does. Same signal, same seed, so the
  configs differ in error correlation only.

## 3. Backtest

- Horizon `h = 12` months; seasonal period 12.
- 12 origins, 3 months apart, expanding window; the last origin is 12 months before the end of
  the data. Each origin refits every forecaster on data up to and including the origin only.
- Base forecasters, each forecasting **all** nodes (aggregates included):
  - `snaive` — seasonal naive.
  - `ets` — `statsforecast` AutoETS, season length 12, per series.
  - `lgbm` — one global LightGBM model per horizon step (direct strategy) over all nodes; each
    series scaled by the mean of its last 12 training months; features are scaled lags, target
    month and level id.
  - `hybrid` — ETS plus a global LightGBM model per horizon step fitted to ETS in-sample
    residuals (scaled), added back to the ETS forecast.
- Reconcilers (all of the form `ỹ = S G ŷ`, own linear algebra):
  - `bu` bottom-up; `td` top-down with average historical proportions of the training window;
  - `ols` (W = I); `wls_struct` (W = diag(S·1));
  - `mint_shrink` — W = the Schäfer–Strimmer shrinkage estimate of the covariance of the base
    forecaster's in-sample one-step residuals (Wickramasuriya et al. 2019). Columns with any
    missing residual are dropped before estimation.
- Reconciled forecasts are **not** clipped at zero (clipping breaks coherence). Negative values
  are counted and reported.

## 4. Metrics

- MASE and RMSSE per series per origin, averaged over the 12 steps. Scale: the in-sample error of
  the lag-12 seasonal naive on that origin's training window (mean absolute error for MASE, mean
  squared error for RMSSE).
- A series whose scale is 0 at an origin is excluded at that origin; the count is reported.
- Reported **per level**, never pooled across levels: the mean over the level's series and origins.
- MASE is the decision metric. RMSSE is reported alongside it as a check.

## 5. Inference

- Paired difference per (series, origin): `d = MASE(candidate) − MASE(reference)`; negative is
  better for the candidate. Level estimate = mean of `d`.
- 95% percentile bootstrap, B = 2000, seed 20261023. The resampling unit is the series (all of its
  origins kept together), since origins of one series overlap and are not independent. A level
  with fewer than 5 series (the total level; nothing else here) is resampled over its
  (series, origin) pairs instead; that interval ignores the overlap between origins and is labelled
  as such.
- "Better" at a level means the whole interval is below 0; "worse" means it is above 0; otherwise
  "no clear difference".

## 6. Hypotheses (confirmatory)

**H1 — reconciliation helps at every level.** Data `tourism`, base `ets`, candidate `mint_shrink`,
reference the unreconciled `ets`.
- **Supported** if MinT-shrink is better at ≥ 3 of the 4 levels and worse at none.
- **Helps the top, hurts the bottom** if it is better at the total level and worse at the bottom
  level.
- Otherwise **mixed**, reported level by level.

**H2 — the hybrid beats pure ETS at the bottom.** Data `tourism`, bottom level, unreconciled
forecasts, candidate `hybrid`, reference `ets`. **Supported** if better; **refuted** if worse;
otherwise **tie**.

**H3 — MinT helps more when errors are correlated.** Bottom level, base `ets`. Gain = mean `d` of
`mint_shrink` vs unreconciled `ets`. **Supported** if gain(`synth-correlated`) − gain(`synth-independent`)
has a 95% interval entirely below 0 (series resampled independently within each dataset);
**refuted** if entirely above 0; otherwise **no clear difference**.

The null results (reconciliation helps only the aggregate, the hybrid ties ETS, correlation makes
no difference) are publishable outcomes and are reported in the same place and the same detail.

## 7. Deviation from the approved brief

The brief's H1 compared "the best of the five reconcilers" with its base. Picking the best on the
same test origins that the interval is computed on biases the test towards success. H1 therefore
fixes MinT-shrink in advance, as the method the theory expects to be best. The other four
reconcilers, the other three base forecasters and the two synthetic arms are reported in full as
**descriptive** tables, with intervals but no verdicts.

## 8. Correctness checks (tests, not results)

- Every reconciled forecast is coherent: `max |ỹ − S·ỹ_bottom| ≤ 1e-8` (relative to scale).
- MinT with W = I equals OLS; reconciling already-coherent forecasts changes nothing for the
  projection methods (`ols`, `wls_struct`, `mint_shrink`).
- Top-down proportions are non-negative and sum to 1.
- Own reconcilers match `hierarchicalforecast` (test-only dependency) on the same inputs.
- No leakage: changing any value after the origin leaves every forecast unchanged, for every
  forecaster.

## Amendments

### A1 — 2026-10-23, before any result was read

An independent code review (fresh context) found that the two synthetic configs did not differ in
error correlation only, as §2 claims. The shared shocks were AR(1) with coefficient 0.6 but the
idiosyncratic noise was white, so `correlated` had persistent noise (lag-1 autocorrelation 0.42)
and `independent` did not (−0.00). Persistence affects ETS accuracy and MinT's gain on its own, so
H3 would have mixed two effects.

1. **Change:** the idiosyncratic noise is now AR(1) with the same coefficient (`shock_ar = 0.6`) as
   the shared shocks, each with unit stationary variance. Both configs now have the same noise
   variance and the same lag-1 autocorrelation per series (0.6) and differ only in cross-series
   correlation. A test pins this.
2. **Clarification, no change in behaviour:** in §3, "columns with any missing residual" means
   *time points* at which any node's residual is missing. The `lgbm` and `hybrid` bases have no
   residual for their first 24 months, so MinT estimates their covariance on the remaining months.
3. **Clarification:** the shrinkage estimator centres each node's residuals and divides by T − 1
   (the `hierarchicalforecast` convention, which the tests match). The original paper's reference
   code uses uncentred residuals divided by T. The ETS residual means are about 2% of their sd on
   tourism, so the difference is negligible; it is recorded rather than changed.
4. **Clarification:** the coherence tolerance in the code is aligned with §8 (1e-8, relative to the
   largest absolute forecast).
5. **Known limitation, no change:** the hybrid learns ETS one-step in-sample residuals, which are
   close to white noise when ETS fits well. H2 may therefore come out "tie" because of this design,
   not because ML residual correction cannot help. The README will say so next to H2.

Full backtests had been started on all three datasets before this amendment, into a scratch
directory outside the repo, to check run time. Their scores were not opened and are discarded. All
reported results come from runs at or after the commit that adds this amendment.
