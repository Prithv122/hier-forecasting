"""Scoring on hand-computed examples and the paired bootstrap."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from hierfc.contracts import Hierarchy
from hierfc.metrics import (
    bootstrap_gain_difference,
    level_table,
    paired_bootstrap,
    scales,
    score,
)


def tiny_hierarchy() -> Hierarchy:
    return Hierarchy(
        S=np.array([[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]]),
        names=("total", "a", "b"),
        level_of=("total", "bottom", "bottom"),
        levels=("total", "bottom"),
    )


def make_scores(
    level: str, n_series: int, n_origins: int, delta, noise: float = 0.0, seed: int = 0
) -> pd.DataFrame:
    """Rows for (ets, base) and (ets, mint_shrink); the candidate is base + delta (+ noise)."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_series):
        for o in range(n_origins):
            ref = 1.0 + 0.1 * i + 0.01 * o
            cand = ref + delta + noise * rng.standard_normal()
            for rec, val in (("base", ref), ("mint_shrink", cand)):
                rows.append(
                    (
                        "ets",
                        rec,
                        f"s{i}",
                        level,
                        pd.Timestamp("2020-01-01") + pd.DateOffset(months=o),
                        val,
                    )
                )
    cols = ["forecaster", "reconciler", "unique_id", "level", "origin", "mase"]
    return pd.DataFrame(rows, columns=cols)


CAND, REF = ("ets", "mint_shrink"), ("ets", "base")


def test_scales_hand_computed():
    mae, mse = scales(np.array([1.0, 2, 3, 4, 5, 6]), season=2)
    assert (mae, mse) == (2.0, 4.0)
    y = np.array([[1.0, 2, 3, 4, 5, 6], [0, 0, 0, 0, 0, 0]])
    mae, mse = scales(y, season=2)
    assert mae.tolist() == [2.0, 0.0]
    assert mse.tolist() == [4.0, 0.0]
    with pytest.raises(ValueError):
        scales(np.arange(3.0), season=3)


def test_score_hand_computed_and_scale_zero_exclusion():
    hier = tiny_hierarchy()
    ds = pd.date_range("2020-01-01", periods=8, freq="MS")
    a = np.arange(1.0, 9.0)
    b = np.full(8, 5.0)  # constant: scale 0 at every origin
    y_all = pd.concat(
        [
            pd.DataFrame({"unique_id": n, "ds": ds, "y": v})
            for n, v in (("total", a + b), ("a", a), ("b", b))
        ],
        ignore_index=True,
    )
    origin = ds[5]  # train = first 6 months, h = 2 -> actuals months 7, 8
    fut = ds[6:8]
    fc = pd.concat(
        [
            pd.DataFrame(
                {
                    "origin": origin,
                    "forecaster": "f",
                    "reconciler": "base",
                    "unique_id": n,
                    "ds": fut,
                    "step": [1, 2],
                    "yhat": v,
                }
            )
            for n, v in (("total", [13.0, 13.0]), ("a", [8.0, 8.0]), ("b", [5.0, 5.0]))
        ],
        ignore_index=True,
    )
    out = score(fc, y_all, hier, season=2)
    assert set(out.columns) == {
        "origin",
        "forecaster",
        "reconciler",
        "unique_id",
        "level",
        "mase",
        "rmsse",
    }
    assert sorted(out["unique_id"]) == ["a", "total"]  # b excluded
    assert out.attrs["excluded"] == 1
    assert out.attrs["excluded_by_level"] == {"total": 0, "bottom": 1}
    row = out.set_index("unique_id").loc["a"]
    # errors are [1, 0]; scales are mae 2, mse 4
    assert row["mase"] == pytest.approx(0.5 / 2)
    assert row["rmsse"] == pytest.approx(math.sqrt(0.5 / 4))
    assert row["level"] == "bottom"
    assert out.set_index("unique_id").loc["total", "level"] == "total"


def test_level_table_orders_levels_and_averages():
    scores = pd.DataFrame(
        {
            "forecaster": "ets",
            "reconciler": "base",
            "level": ["total", "bottom", "bottom"],
            "mase": [1.0, 2.0, 4.0],
        }
    )
    table = level_table(scores)
    assert list(table.columns) == ["total", "bottom"]
    assert table.loc[("ets", "base"), "bottom"] == 3.0
    assert list(level_table(scores, levels=["bottom", "total"]).columns) == ["bottom", "total"]


def test_bootstrap_unit_depends_on_series_count():
    many = paired_bootstrap(
        make_scores("bottom", 8, 3, -0.5), candidate=CAND, reference=REF, level="bottom"
    )
    assert many["unit"] == "series"
    assert many["n_series"] == 8
    assert many["n_pairs"] == 24
    few = paired_bootstrap(
        make_scores("total", 1, 6, -0.5), candidate=CAND, reference=REF, level="total"
    )
    assert few["unit"] == "series-origin"
    assert few["n_series"] == 1
    assert few["n_pairs"] == 6
    four = paired_bootstrap(
        make_scores("zone", 4, 3, -0.5), candidate=CAND, reference=REF, level="zone"
    )
    assert four["unit"] == "series-origin"


@pytest.mark.parametrize(
    ("delta", "noise", "verdict"),
    [(-0.5, 0.05, "better"), (0.5, 0.05, "worse"), (0.0, 0.5, "no clear difference")],
)
def test_bootstrap_verdicts(delta, noise, verdict):
    scores = make_scores("bottom", 20, 4, delta, noise, seed=3)
    res = paired_bootstrap(scores, candidate=CAND, reference=REF, level="bottom")
    assert res["verdict"] == verdict
    assert res["lo"] <= res["mean"] <= res["hi"]
    if verdict != "no clear difference":
        assert res["mean"] == pytest.approx(delta, abs=0.05)


def test_bootstrap_is_deterministic_for_a_seed_and_fast():
    scores = make_scores("bottom", 76, 12, -0.1, 0.3, seed=1)
    a = paired_bootstrap(scores, candidate=CAND, reference=REF, level="bottom", seed=5)
    b = paired_bootstrap(scores, candidate=CAND, reference=REF, level="bottom", seed=5)
    c = paired_bootstrap(scores, candidate=CAND, reference=REF, level="bottom", seed=6)
    assert a == b
    assert (a["lo"], a["hi"]) != (c["lo"], c["hi"])


def test_bootstrap_inner_joins_pairs():
    scores = make_scores("bottom", 6, 3, -0.5)
    scores = scores.drop(
        scores[(scores["reconciler"] == "mint_shrink") & (scores["unique_id"] == "s0")].index
    )
    res = paired_bootstrap(scores, candidate=CAND, reference=REF, level="bottom")
    assert res["n_series"] == 5
    assert res["n_pairs"] == 15
    with pytest.raises(ValueError):
        paired_bootstrap(scores, candidate=CAND, reference=("ets", "nope"), level="bottom")


@pytest.mark.parametrize(
    ("delta_a", "delta_b", "verdict"),
    [(-0.5, -0.1, "better"), (-0.1, -0.5, "worse"), (-0.3, -0.3, "no clear difference")],
)
def test_gain_difference_verdicts(delta_a, delta_b, verdict):
    a = make_scores("bottom", 20, 3, delta_a, 0.2, seed=1)
    b = make_scores("bottom", 20, 3, delta_b, 0.2, seed=2)
    res = bootstrap_gain_difference(a, b, candidate=CAND, reference=REF, level="bottom")
    assert res["verdict"] == verdict
    assert res["mean"] == pytest.approx(res["mean_a"] - res["mean_b"])
    assert res["unit"] == "series"
    assert res["n_series"] == (20, 20)
    again = bootstrap_gain_difference(a, b, candidate=CAND, reference=REF, level="bottom")
    assert again == res


def test_score_scale_uses_only_the_training_window():
    """The MASE/RMSSE scale at an origin ignores everything after it.

    ``a`` has seasonal differences of 2 inside the training window and huge ones afterwards, so a
    scale taken over the full series would be very different.
    """
    hier = tiny_hierarchy()
    ds = pd.date_range("2020-01-01", periods=8, freq="MS")
    a = np.array([1.0, 2, 3, 4, 5, 6, 50, 100])
    b = np.array([2.0, 1, 4, 3, 6, 5, 7, 9])
    y_all = pd.concat(
        [
            pd.DataFrame({"unique_id": n, "ds": ds, "y": v})
            for n, v in (("total", a + b), ("a", a), ("b", b))
        ],
        ignore_index=True,
    )
    origin = ds[5]
    fc = pd.concat(
        [
            pd.DataFrame(
                {
                    "origin": origin,
                    "forecaster": "f",
                    "reconciler": "base",
                    "unique_id": n,
                    "ds": ds[6:8],
                    "step": [1, 2],
                    "yhat": v,
                }
            )
            for n, v in (("total", [60.0, 100.0]), ("a", [40.0, 90.0]), ("b", [7.0, 9.0]))
        ],
        ignore_index=True,
    )
    out = score(fc, y_all, hier, season=2).set_index("unique_id")
    # training diffs at lag 2: a -> [2,2,2,2] (mae 2, mse 4); b -> [2,2,2,2]; total -> [4,4,4,4]
    # (mae 4, mse 16); a scale over the whole series would be dominated by the later jumps.
    assert out.loc["a", "mase"] == pytest.approx(10.0 / 2.0)  # errors [-10, -10]
    assert out.loc["a", "rmsse"] == pytest.approx(math.sqrt(100.0 / 4.0))
    assert out.loc["b", "mase"] == 0.0
    assert out.loc["b", "rmsse"] == 0.0
    assert out.loc["total", "mase"] == pytest.approx(6.0 / 4.0)  # errors [3, -9]
    assert out.loc["total", "rmsse"] == pytest.approx(math.sqrt(45.0 / 16.0))
    assert out.attrs["excluded"] == 0


def clustered_scores(effects, n_origins, noise=0.01, drop=(), seed=0) -> pd.DataFrame:
    """Candidate minus reference is ``effects[i] + small noise`` for series i at every origin."""
    rng = np.random.default_rng(seed)
    rows = []
    for i, eff in enumerate(effects):
        for o in range(n_origins):
            if (i, o) in drop:
                continue
            ref = 1.0 + 0.1 * i
            cand = ref + eff + noise * rng.standard_normal()
            origin = pd.Timestamp("2020-01-01") + pd.DateOffset(months=o)
            for rec, val in (("base", ref), ("mint_shrink", cand)):
                rows.append(("ets", rec, f"s{i}", "bottom", origin, val))
    cols = ["forecaster", "reconciler", "unique_id", "level", "origin", "mase"]
    return pd.DataFrame(rows, columns=cols)


def test_series_bootstrap_resamples_whole_series_not_pairs():
    """With strong per-series effects the series-unit interval is far wider than a pair bootstrap.

    Resampling (series, origin) pairs while calling the unit "series" would break the within-series
    dependence and produce the narrow interval.
    """
    effects = [(-1.0) ** i * (0.5 + 0.1 * i) for i in range(20)]
    scores = clustered_scores(effects, n_origins=10, noise=0.01)
    kw = dict(candidate=CAND, reference=REF, level="bottom")
    by_series = paired_bootstrap(scores, **kw)
    by_pair = paired_bootstrap(scores, min_series=21, **kw)  # forces the pair unit
    assert by_series["unit"] == "series" and by_pair["unit"] == "series-origin"
    assert by_series["mean"] == by_pair["mean"]
    width_series = by_series["hi"] - by_series["lo"]
    width_pair = by_pair["hi"] - by_pair["lo"]
    # design effect: 10 origins per series -> about sqrt(10) wider; require at least 2x
    assert width_series > 2.0 * width_pair


def test_series_bootstrap_matches_an_independent_cluster_resample():
    """Replay the draws by hand: ratio of resampled sums to resampled counts, series as unit."""
    effects = [0.3 * i - 1.5 for i in range(12)]
    scores = clustered_scores(effects, n_origins=4, noise=0.1, drop={(2, 3), (5, 0), (5, 1)})
    B, seed, alpha = 400, 11, 0.1
    res = paired_bootstrap(
        scores, candidate=CAND, reference=REF, level="bottom", B=B, seed=seed, alpha=alpha
    )
    d = scores.pivot_table(index=["unique_id", "origin"], columns="reconciler", values="mase")
    diff = (d["mint_shrink"] - d["base"]).dropna()
    clusters = [g.to_numpy() for _, g in diff.groupby(level="unique_id")]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(clusters), size=(B, len(clusters)))
    boots = np.array(
        [sum(clusters[j].sum() for j in row) / sum(len(clusters[j]) for j in row) for row in idx]
    )
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    assert res["mean"] == pytest.approx(diff.mean())
    assert res["lo"] == pytest.approx(lo)
    assert res["hi"] == pytest.approx(hi)
    assert res["n_series"] == 12 and res["n_pairs"] == 45
