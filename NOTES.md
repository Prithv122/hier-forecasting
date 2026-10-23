# Build Notes — Hierarchical Forecasting

Working notes: what broke, what I tried, why X over Y.

---

## Log

### 2026-10-23 — contracts, pre-registration, build

- **Order:** scaffold → `PREREGISTRATION.md` → `contracts.py` + `spec.py`, each its own commit, before
  any model code. Modules then built in parallel against the contracts and integrated.
- **Deviation from the brief, logged before any code:** H1 tests MinT-shrink, fixed in advance, not
  "the best of five reconcilers" (selection on the test origins would bias the interval).
- **Data licence:** the tourism archive (Nixtla's redistribution of the MinT-paper data) states no
  licence. Upstream Austrade content is CC BY 4.0. The brief allowed an unverified licence on the
  condition that the data is downloaded, not committed; the README states the gap. Archive pinned by
  sha256 `6512d9aa…f704b`; the loader cross-checks our regional sums against the file's own state
  totals.
- **Hierarchy choice:** the source has 304 bottom series (region × purpose) and a grouped structure.
  Purpose is summed out to keep a strictly nested 4-level hierarchy (total / state / zone / region),
  so every reconciler applies without grouped-structure extensions.
- **Shrinkage convention:** centred residuals, covariance / (T−1), lambda from standardised
  residuals. Matches `hierarchicalforecast` 1.5.3 to ~1e-8 relative in lambda; MinT results agree to
  3.7e-9. Its NaN path uses pairwise-complete data; ours drops whole time points with any NaN, as
  pre-registered, so the two differ only when residuals have gaps.
- **LightGBM threads:** `n_jobs=4` was ~30× slower than `n_jobs=1` on small fits under load and not
  needed for reproducibility; fixed at 1. Parallelism belongs across origins.
- **statsforecast sorts series by id** in its output; mapping back is by name, with a shuffled-row
  test.
- **ETS fallback:** a series with a non-finite ETS forecast falls back to the seasonal naive. Counted
  per run in `meta.json` (`ets_fallback_series`) instead of being silent.
- **Leakage:** besides the fake-forecaster test, every real forecaster is run through the backtest
  twice, with all post-origin values scrambled the second time; forecasts are byte-identical.

---

## Rejected approaches

| Approach | Why rejected |
|---|---|
| `hierarchicalforecast` as the implementation | The reconcilers are the subject; kept as a test-only cross-check |
| Grouped hierarchy (geography × purpose) | Needs grouped-structure reconciliation and complicates "per level"; purpose summed out |
| Monash Tourism Monthly (CC BY 4.0) | No hierarchy labels, so nothing to reconcile |
| Top-down with forecast proportions | Not a fixed `G` matrix; average historical proportions keep all five methods linear |
| Clipping negative reconciled forecasts | Breaks coherence; negatives are counted instead |

## Open questions

- [ ] Do LightGBM in-sample residuals (too small, in-sample fit) distort MinT's covariance for the
      `lgbm` and `hybrid` bases? Descriptive arms only, but worth a sentence in the README.
