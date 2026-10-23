# Interview Prep — Hierarchical Forecasting

**Five questions, five answers.**

---

### Q1. Walk me through the architecture in 90 seconds.

A keys frame (state / zone / region) becomes a summing matrix `S`, so every node is a known sum of
the 76 regional series. The backtest takes 12 expanding-window origins, 3 months apart. At each
origin it hands only data up to the origin to four base forecasters: seasonal naive, AutoETS per
series, a global LightGBM with one model per horizon step, and an ETS + LightGBM hybrid. Each
forecasts all 111 nodes for 12 months. Five reconcilers turn those into coherent forecasts with
`ỹ = S G ŷ`: bottom-up, top-down, OLS, structural WLS and MinT-shrink, whose `G` uses a shrunk
covariance of the base forecaster's in-sample residuals. Every reconciled matrix is checked for
coherence. Scores are MASE and RMSSE per series per origin, then per level, and a series-cluster
bootstrap gives intervals. The report applies the pre-registered verdict rules mechanically.

### Q2. Why did you write the reconciliation yourself instead of using `hierarchicalforecast`?

Because the reconcilers are what the study is about. Every one of them is `G = (S'W⁻¹S)⁻¹S'W⁻¹`
with a different `W`, or a fixed `G` for bottom-up and top-down, so it is about 150 lines including
the Schäfer–Strimmer shrinkage. Writing them out meant I could test the properties that matter:
coherence, MinT with `W = I` equals OLS, projections leave coherent input unchanged. I kept the
library as a test-only cross-check: all five methods agree, MinT to 4e-9. The one place they
legitimately differ is missing residuals: the library uses pairwise-complete covariance, and I drop
whole time points, as pre-registered.

### Q3. What's the weakest part of this?

Three things. First, the total level has a single series, so its interval resamples
(series, origin) pairs, and origins 3 months apart with a 12-month horizon overlap. That interval
is too narrow; the lower bound of +0.042 is far enough from 0 that I believe the direction, but I
wouldn't quote the width. Second, one real dataset and 12 origins in 2013–2015. Third, the hybrid
learns ETS one-step in-sample residuals, which are close to white noise, so H2's tie says more about
that design than about ML residual correction in general. I wrote that down as a known limitation
before seeing the result.

### Q4. How do you know it works? What did you measure, against what baseline?

Correctness first: coherence of every reconciled forecast to 1e-8; agreement with an independent
library; and a leakage test that runs each real forecaster through the backtest twice, scrambling
every value after the origin the second time, and requires byte-identical forecasts. An independent
review also mutated key functions; four mutants passed the first test suite, so I added tests that
kill them (training-window scale, series-level bootstrap, hybrid formula, LightGBM anchor). Then
the result: against the unreconciled ETS forecasts, MinT-shrink made the tourism total worse, MASE
0.990 → 1.083 (+0.093 [+0.042, +0.147]), with no clear change at the states and gains under 1% at
zones (−0.006) and regions (−0.007). The hybrid tied ETS at the regions (+0.003 [−0.008, +0.013]).
On synthetic data, correlated errors made no clear difference to MinT's gain (−0.002 [−0.011,
+0.008]).

### Q5. MinT is supposed to be optimal. Why did it make the total worse?

MinT minimises the total variance of the reconciled errors summed over every node, under unbiased
base forecasts and a well-estimated covariance. It is not optimal for any single level. My reading
of why (an explanation, not something the study tested): on tourism,
the ETS forecast of the total is much better than the sum of the 76 regional forecasts (MASE 0.990
vs 1.186 bottom-up). MinT puts substantial weight on the bottom, since there are 76 regional series
against one total, and that pulls the total towards the noisier sum. The covariance is estimated from in-sample
one-step residuals, which understate 12-step errors, and shrinkage pulls it towards diagonal; both
make the weights less right for the total. The synthetic arms show the mirror image: where
LightGBM's total was worse than the sum of its parts (1.087 vs 0.654 bottom-up), MinT took it to
0.625. So "does reconciliation help a level" depends on whether that level's own forecast beats what
the rest of the hierarchy implies for it.

---

## 30-second pitch

Planners forecast totals and regions separately, the numbers don't add up, and reconciliation
methods like MinT promise to fix that and improve accuracy. I tested whether they improve accuracy
at every level, with the hypotheses fixed before any code: a 12-origin rolling backtest of four
base forecasters and five reconcilers on Australian tourism's 111-series hierarchy, scored per
level with a series-cluster bootstrap. MinT made the national total 9% worse and improved the
regions by under 1%, the reverse of the usual worry. The practical rule: reconcile below the total,
or use OLS, which kept the total within 1% and still gained about 2% at the states and zones.
