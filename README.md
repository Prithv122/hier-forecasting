# hier-forecasting

> Does forecast reconciliation make every level of a hierarchy more accurate, or only coherent?

[![CI](https://github.com/Prithv122/hier-forecasting/actions/workflows/ci.yml/badge.svg)](https://github.com/Prithv122/hier-forecasting/actions/workflows/ci.yml)

**Live demo:** not deployed (a batch study that runs locally)
**Stack:** Python 3.12, uv, statsforecast (AutoETS), LightGBM, numpy, pandas. Reconciliation is own
linear algebra, cross-checked in tests against `hierarchicalforecast`.

---

## 1. The problem

Demand and capacity plans are made at several levels at once: the country, each state, each region.
Forecast every level on its own and the numbers do not add up, so planners adjust them by hand.
Reconciliation methods force the forecasts to add up. The textbook claim is that the best of them
(MinT) also makes the forecasts more accurate. The question here is narrower and more useful to a
planner: **does reconciliation lower the error at every level, or does it help the totals and hurt
the regions people actually staff?**

## 2. The data

| | Tourism (confirmatory) | Synthetic (mechanism check) |
|---|---|---|
| Source | Australian domestic visitor nights, Tourism Research Australia (National Visitor Survey), the data of Wickramasuriya, Athanasopoulos & Hyndman (2019), as redistributed by Nixtla (`TourismLarge`) | `src/hierfc/synth.py`, seed 31 |
| Hierarchy | total → 7 states → 27 zones → 76 regions (111 series; travel purpose summed out) | total → 5 regions → 20 stores → 60 categories (86 series) |
| Size | 228 months, 1998-01 to 2016-12 | 144 months, two configs |
| Licence | Upstream content is CC BY 4.0 per the Austrade site disclaimer. The redistributed copy states no licence of its own, so the data is **downloaded at run time and never committed**. Archive pinned by sha256 | MIT, part of this repo |

The two synthetic configs share the same signal and, per series, the same noise variance and
persistence (every noise component is AR(1) with coefficient 0.6). In `correlated`, 70% of the noise
variance comes from shocks shared by a region or a store; in `independent`, none does. So they
differ in cross-series error correlation only, which is the condition MinT is supposed to exploit.
(The first version had white idiosyncratic noise, which made the arms differ in persistence too; a
review caught it and Amendment A1 fixed it before any result was read.) **Synthetic results are labelled synthetic wherever they appear.**

## 3. Architecture

```mermaid
flowchart LR
    K[keys: state / zone / region] --> S[summing matrix S]
    D[bottom series] --> A[aggregate to all 111 nodes]
    S --> A
    A --> O[12 expanding-window origins]
    O --> F[base forecasts for every node<br/>snaive · ETS · LightGBM · hybrid]
    F --> R[reconcile ỹ = S G ŷ<br/>BU · TD · OLS · WLS · MinT-shrink]
    R --> M[MASE / RMSSE per series per origin]
    M --> B[per-level tables + series-cluster bootstrap]
    B --> REP[results/REPORT.md<br/>mechanical H1–H3 verdicts]
```

| Module | Role |
|---|---|
| `contracts.py`, `spec.py` | Shared interfaces and the frozen protocol values |
| `hierarchy.py` | Builds `S` from a keys frame; aggregates bottom series |
| `reconcile.py` | The five reconcilers as `G` matrices, Schäfer–Strimmer shrinkage |
| `base/` | Seasonal naive, AutoETS, direct global LightGBM, ETS + LightGBM hybrid |
| `backtest.py` | Origins, strict history slicing, coherence check on every reconciled matrix |
| `metrics.py` | MASE, RMSSE, per-level tables, paired bootstrap |
| `report.py` | Writes `results/REPORT.md` with verdicts computed by the pre-registered rules |
| `tourism.py`, `synth.py`, `datasets.py` | The three data arms |

## 4. Key decisions & tradeoffs

| Decision | Chose | Over | Why |
|---|---|---|---|
| Reconciliation code | Own `G`-matrix linear algebra (~150 lines) | `hierarchicalforecast` at run time | The methods are the subject of the study, so they are written out and tested; the library is kept as a test-only cross-check (all five agree to 1e-6 in the tests; MinT to 4e-9 in practice) |
| H1's candidate | MinT-shrink, fixed in advance | "Best of the five reconcilers" (the approved brief) | Choosing the best on the same origins the interval is computed on biases the test towards success. Logged as a deviation in `PREREGISTRATION.md` §7 |
| Error metric | MASE per level, RMSSE as a check | MAPE, or one pooled number | Tourism regions have zero months, which breaks MAPE; aggregates are orders of magnitude larger than leaves, so a pooled score is just the total's score |
| Bootstrap unit | Whole series (all origins together) | Each (series, origin) pair | Origins 3 months apart with a 12-month horizon overlap, so their errors are not independent. The total level has one series and falls back to pairs, labelled as such |
| Global LightGBM | One model per horizon step over all nodes, each series scaled by its last-12-month mean | One model per series; recursive forecasting | 111 series × ~180 months is too little data per series; direct models avoid feeding predictions back in |
| Negative forecasts | Not clipped, counted | Clipping at zero | Clipping a reconciled forecast breaks the coherence that reconciliation exists to provide |

## 5. Results

Every number below is from [`results/REPORT.md`](results/REPORT.md), generated by
`hier-forecast report` from the score files of one run of the code at `3739dad` (12 origins,
2013-03 to 2015-12 for tourism, h = 12). MASE: lower is better, 1.0 is the in-sample seasonal naive.
Intervals are 95% bootstrap intervals of the paired difference (candidate − reference).

**Headline: on Australian tourism, MinT-shrink reconciliation of ETS forecasts made the national
total worse by 9% (MASE 0.990 → 1.083), changed the states by nothing measurable, and improved the
zones and regions by under 1%.** The worry this study set out to test ("helps the top, hurts the
bottom") is the reverse of what happened.

### Confirmatory (pre-registered; verdicts computed mechanically)

**H1 — does MinT-shrink lower ETS error at every level? Verdict: Mixed.**

| Level (tourism) | Series | ETS base | ETS + MinT-shrink | Difference [95% interval] | Reading |
|---|---|---|---|---|---|
| Total | 1 | 0.990 | 1.083 | +0.093 [+0.042, +0.147]† | worse |
| State | 7 | 0.943 | 0.947 | +0.004 [−0.020, +0.038] | no clear difference |
| Zone | 27 | 0.910 | 0.904 | −0.006 [−0.014, −0.000] | better |
| Region | 76 | 0.838 | 0.831 | −0.007 [−0.011, −0.004] | better |

† The total level has one series, so its interval resamples (series, origin) pairs and ignores
the overlap between origins; it is narrower than it should be. RMSSE moves the same way at every
level (total 0.948 → 1.023, region 0.784 → 0.776).

**H2 — does the ETS + LightGBM hybrid beat pure ETS at the regions? Verdict: Tie.** MASE 0.841 vs
0.838, difference +0.003 [−0.008, +0.013]. This hybrid learns ETS one-step in-sample residuals,
which are close to white noise when ETS fits well, so the tie may reflect that design rather than
the limits of ML residual correction (pre-registered as a known limitation in Amendment A1).

**H3 — does MinT help more when bottom errors are correlated? Verdict: No clear difference
(synthetic).** MinT's bottom-level gain over ETS was −0.001 with correlated errors and +0.000 with
independent ones; the difference is −0.002 [−0.011, +0.008]. On these synthetic hierarchies MinT
barely moved the bottom level either way.

### Descriptive (no verdicts; read as patterns, not tests)

| Tourism, ETS base, difference vs base | Total | State | Zone | Region |
|---|---|---|---|---|
| Bottom-up | **+0.196**† | +0.034 | +0.011 | 0 (identical) |
| Top-down (historical proportions) | 0 (identical) | **+0.388** | **+0.286** | **+0.169** |
| OLS | **+0.012**† | **−0.023** | **−0.016** | −0.001 |
| Structural WLS | **+0.095**† | +0.005 | −0.006 | −0.003 |
| MinT-shrink | **+0.093**† | +0.004 | **−0.006** | **−0.007** |

Bold: interval excludes 0. Full intervals for every forecaster are in the report.

- **The cost of coherence lands on a level whose own forecast beats the sum of its parts.** On
  tourism the ETS forecast of the total is far better than the sum of the regional forecasts (MASE
  0.990 vs 1.186 for bottom-up), and every method that leans on the bottom (bottom-up, WLS, MinT)
  pulls the total towards that sum. OLS, which weights all levels equally, kept the total almost
  intact and improved states and zones.
- **The synthetic arms show the mirror image.** There, LightGBM's own forecast of the total was
  worse than the sum of its bottom forecasts (MASE 1.087 vs 0.654 bottom-up in
  `synth-independent`), and reconciliation fixed it: MinT took it to 0.625 (−0.462 [−0.602,
  −0.314]). Whether reconciliation "helps" a level depends on whether that level's own forecast is
  better or worse than what the rest of the hierarchy implies for it.
- **Top-down with historical proportions was the worst method below the total** for ETS,
  LightGBM and the hybrid on every dataset (tourism ETS: +0.17 to +0.39 MASE). The one exception is
  the seasonal naive at the tourism regions, where it helped (−0.074 [−0.133, −0.010]).
- **ETS beat the global LightGBM** at every tourism level (base MASE 0.990 / 0.943 / 0.910 / 0.838
  vs 1.179 / 1.006 / 0.943 / 0.857); the seasonal naive was worst below the total.

### What this means for a planner

If the national number is the one that gets budgeted, don't reconcile ETS forecasts with MinT and
then report the reconciled total: on this data it was 9% worse than the forecast you started with.
Either keep the base total and reconcile only below it, or use OLS, which here cost 1% at the total
and gained about 2% at the states and zones (descriptive, not a pre-registered test).

### Limits

- One real dataset, 12 origins over 2013–2015; a different period or hierarchy can differ.
- The total-level interval is too narrow (see †), though its lower bound is +0.042.
- Synthetic results describe the generator in `synth.py`, not real demand.
- The descriptive tables make 80 comparisons per dataset (5 reconcilers × 4 levels × 4 base
  forecasters) without correction; a few "excludes 0" readings are expected by chance.
- Two amendments were made: A1 (synthetic noise persistence) before any result was read, A2 (a
  floating-point display fix that did not touch H1–H3) after. Both are in `PREREGISTRATION.md`.

## 6. How to run

```bash
git clone https://github.com/Prithv122/hier-forecasting.git
cd hier-forecasting
uv sync
uv run pytest
```

Reproduce the study (about 1.3 MB download; the tourism backtest took about an hour on a shared laptop, each synthetic arm about 25 minutes):

```bash
uv run hier-forecast download                       # tourism archive into data/, sha256-checked
uv run hier-forecast backtest --dataset tourism
uv run hier-forecast backtest --dataset synth-correlated
uv run hier-forecast backtest --dataset synth-independent
uv run hier-forecast report                         # writes results/REPORT.md
```

`uv run hier-forecast backtest --smoke` runs a 15-series synthetic set in a couple of minutes.
`hier-forecast generate --config correlated --seed 31 --out DIR` writes a synthetic hierarchy to
CSV, and `hier-forecast reconcile` applies one reconciler to your own base forecasts.

## 7. What I'd change at 100× scale

- **MinT's covariance is the first thing to break.** At 11,000 series the residual covariance is
  11,000 × 11,000 estimated from ~200 time points, and the dense solve is cubic. Use the sparse
  structure of `S` with an iterative solver, or keep MinT for the top levels and reconcile within
  each state independently.
- **AutoETS per series is embarrassingly parallel but the backtest loop is serial.** Parallelise
  across origins and series (one process per origin), keeping LightGBM single-threaded for
  reproducibility.
- **The global LightGBM rebuilds its training set at every origin.** Build the lag matrix once and
  mask by origin; at scale this dominates run time.
- **Forecast dumps grow as origins × forecasters × reconcilers × nodes × h.** Write Parquet
  partitioned by origin instead of one gzipped CSV, and keep only scores in git.

---

## References

- Wickramasuriya, S. L., Athanasopoulos, G., & Hyndman, R. J. (2019). Optimal forecast
  reconciliation for hierarchical and grouped time series through trace minimization. *JASA*,
  114(526), 804–819. doi:10.1080/01621459.2018.1448825
- Schäfer, J., & Strimmer, K. (2005). A shrinkage approach to large-scale covariance matrix
  estimation. *Statistical Applications in Genetics and Molecular Biology*, 4(1).
- Hyndman, R. J., & Athanasopoulos, G. *Forecasting: Principles and Practice* (3rd ed.), chapter 11.
- Nixtla `hierarchicalforecast` (Apache-2.0), used only as a test cross-check; `datasetsforecast`
  (MIT) is the source of the archive URL.
