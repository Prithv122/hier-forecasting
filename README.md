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

The two synthetic configs share the same signal and the same total noise per series. In
`correlated`, 70% of the noise variance comes from shocks shared by a region or a store; in
`independent`, none does. They differ in error correlation and nothing else, which is the condition
MinT is supposed to exploit. **Synthetic results are labelled synthetic wherever they appear.**

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
| Reconciliation code | Own `G`-matrix linear algebra (~150 lines) | `hierarchicalforecast` at run time | The methods are the subject of the study, so they are written out and tested; the library is kept as a test-only cross-check (all five agree, MinT to 4e-9) |
| H1's candidate | MinT-shrink, fixed in advance | "Best of the five reconcilers" (the approved brief) | Choosing the best on the same origins the interval is computed on biases the test towards success. Logged as a deviation in `PREREGISTRATION.md` §7 |
| Error metric | MASE per level, RMSSE as a check | MAPE, or one pooled number | Tourism regions have zero months, which breaks MAPE; aggregates are orders of magnitude larger than leaves, so a pooled score is just the total's score |
| Bootstrap unit | Whole series (all origins together) | Each (series, origin) pair | Origins 3 months apart with a 12-month horizon overlap, so their errors are not independent. The total level has one series and falls back to pairs, labelled as such |
| Global LightGBM | One model per horizon step over all nodes, each series scaled by its last-12-month mean | One model per series; recursive forecasting | 111 series × ~180 months is too little data per series; direct models avoid feeding predictions back in |
| Negative forecasts | Not clipped, counted | Clipping at zero | Clipping a reconciled forecast breaks the coherence that reconciliation exists to provide |

## 5. Results

_Pending: filled from `results/REPORT.md` after the full run. Every number here will come from
that file and the `results/<dataset>/` files it reads._

## 6. How to run

```bash
git clone https://github.com/Prithv122/hier-forecasting.git
cd hier-forecasting
uv sync
uv run pytest
```

Reproduce the study (about 1.3 MB download; the full backtest takes tens of minutes on a laptop):

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
