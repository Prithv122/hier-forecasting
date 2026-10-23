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
