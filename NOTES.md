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

### 2026-10-23 — review, amendments, results

- **Independent review (fresh context):** 11 findings. HIGH: the synthetic arms differed in noise
  persistence as well as correlation (white idiosyncratic noise vs AR(1) shared shocks), which would
  have confounded H3. Fixed by Amendment A1 before any result was read; dry-run scores from before
  A1 were discarded unopened. Four mutants survived the first suite (full-series scale, pair
  resampling labelled "series", hybrid sign, stale LightGBM anchor); each now has a test that kills
  it. meta.json now records HEAD and a dirty flag at the start of a run.
- **Official runs:** all three datasets at `3739dad`, clean tree, HEAD unchanged during the run,
  0 excluded series, 0 ETS fallbacks. Wall time (machine shared with other work): tourism ETS 20 min,
  hybrid 36 min over 12 origins; the synthetic arms a third to a half of that.
- **Amendment A2 (after reading results, reporting only):** no-op reconcilers produced ~1e-15
  differences that the verdict function read as "better"/"worse". |d| < 1e-9 is now 0. Only those
  descriptive cells changed; H1–H3 byte-identical.
- **Surprise:** MinT hurt the total (+0.093 MASE), the reverse of the "helps top, hurts bottom"
  worry. bottom-up at the total was +0.196, so the tourism ETS total is much better than the sum of
  its regions, and every bottom-leaning method pays for that. OLS paid least at the total.
- **README errors caught while checking claims against the report:** "most accurate base forecast"
  (false in MASE terms; regions are 0.838 vs total 0.990), "top-down worst everywhere" (not for the
  seasonal naive at regions), "2–3%" (it was ~2%), incomplete bolding. All corrected before commit.

### Local CI record (hosted CI unavailable: Actions limit until about 2026-11-02)

Replayed `.github/workflows/ci.yml` step by step on a **fresh clone of `69a13a4`** with a fresh
install, on Windows 11 / Python 3.12.13 / uv 0.11.19 (the hosted runner is ubuntu-latest):

| Step | Command | Result |
|---|---|---|
| Install | `uv python install`; `uv sync --all-groups --locked` | OK (lock honoured) |
| Lint | `uv run ruff check .`; `uv run ruff format --check .` | clean; 36 files formatted |
| Test | `uv run pytest --cov=src --cov-report=term-missing` (+ `--basetemp` in a scratch dir, Windows only) | 141 passed, coverage 94% |
| Smoke (extra) | `uv run hier-forecast backtest --smoke` | 3 origins, wrote scores/meta |

**Reproduction from the clean clone:** `hier-forecast download` (sha256 matched), then
`hier-forecast backtest --dataset tourism`: the 31,968-row `scores.csv` is identical to the committed
one (max |ΔMASE| = 0.0). `hier-forecast report` on the committed scores gives a `REPORT.md` with an
empty `git diff` (only line endings differ on Windows).

Differences from the hosted runner: OS (Windows vs Ubuntu) and the `--basetemp` flag. Attribution
grep over all commit messages: 0 matches; every commit authored "Prithvi M".

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
