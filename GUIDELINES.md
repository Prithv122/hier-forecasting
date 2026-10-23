# Hierarchical Forecasting — D5

**Tier:** 3 · **Category:** Analytics & data science · **Wave:** 4

The project's own engineering guide. Keep it short and current.

## What this is

A rolling-origin backtest that asks whether forecast reconciliation makes every level of a
hierarchy more accurate, or only coherent. Four base forecasters (seasonal naive, ETS, global
LightGBM, ETS + LightGBM hybrid) × five reconcilers (bottom-up, top-down, OLS, structural WLS,
MinT-shrink) on Australian tourism and two synthetic hierarchies, scored per level.

## Stack

Python 3.12, `uv`. numpy, pandas 3, statsforecast (AutoETS), LightGBM. Reconciliation is own
linear algebra; `hierarchicalforecast` is a test-only cross-check. Tests use pytest + hypothesis.

## Non-negotiables

- **The protocol is frozen** in `PREREGISTRATION.md` and `src/hierfc/spec.py`. A change is an
  amendment appended to the pre-registration and committed before the run it governs.
- **No leakage.** A forecaster sees data up to and including its origin, nothing after. The
  property test that perturbs post-origin values must pass for every forecaster.
- **Metrics per level, never pooled.** Aggregates are orders of magnitude larger than leaves.
- **No number without a results file.** README numbers come from `results/<dataset>/` written by
  `hier-forecast backtest` and `hier-forecast report`, whose header records the data sha256 and
  library versions. Synthetic results are labelled synthetic wherever they appear.
- Real data is downloaded by `hier-forecast download`, never committed (`data/` is gitignored).

## Contracts

`src/hierfc/contracts.py` is the shared interface: `Hierarchy(S, names, level_of, levels)`, the
`Forecaster` protocol (`fit(y_hist, hierarchy)`, `forecast(h)`, `residuals()`) and long frames
`unique_id, ds, y | yhat | resid`. Reconcilers map a base matrix `(n_nodes, h)` to a coherent one.

| Module | Owns |
|---|---|
| `hierarchy.py` | `build_hierarchy(keys, levels)`, `aggregate(bottom, hierarchy)` |
| `reconcile.py` | `g_matrix(...)`, `reconcile(...)`, shrinkage covariance |
| `synth.py` | seeded synthetic hierarchy from a `SynthConfig` |
| `base/` | `naive`, `ets`, `lgbm`, `hybrid` forecasters + `make_forecaster(name)` |
| `backtest.py` | origins, per-origin fit → forecast → reconcile loop |
| `metrics.py` | MASE, RMSSE, per-level tables, paired bootstrap |
| `report.py` | `results/REPORT.md` with mechanical H1–H3 verdicts |
| `data/tourism.py` | download + load the tourism hierarchy |
| `cli.py` | `hier-forecast generate | download | backtest | reconcile | report` |

## Acceptance criteria

- [ ] Correctness checks of PREREGISTRATION.md §8 pass as tests
- [ ] Full backtest on all three datasets; H1–H3 answered mechanically by `report`
- [ ] Runs from a clean clone using only the documented steps
- [ ] Tests, lint and CI green; README results come from real runs

## Conventions

- `uv` for dependencies; `ruff check` + `ruff format`; `pytest`.
- No secrets; this project reads no environment variables.
- Deterministic: every random draw takes a seed from `spec.py`.

## Project-specific notes

- If the default temp directory is not writable, run pytest with `--basetemp=<a writable dir>`.
- Accuracy results do not depend on machine load, so the backtest needs no quiet machine.
