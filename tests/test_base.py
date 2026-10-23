"""Base forecasters: schema, coverage, determinism, no leakage, scale guard."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hierfc import contracts, spec
from hierfc.base import ETSForecaster, HybridForecaster, LGBMForecaster, make_forecaster
from hierfc.base.features import (
    History,
    build_training_set,
    calendar_months,
    level_codes,
    series_scale,
)
from hierfc.base.naive import SeasonalNaive

T = 84
H = spec.H


def make_hierarchy() -> contracts.Hierarchy:
    """total / 2 regions / 4 leaves."""
    S = np.array(
        [
            [1, 1, 1, 1],
            [1, 1, 0, 0],
            [0, 0, 1, 1],
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
        dtype=float,
    )
    return contracts.Hierarchy(
        S=S,
        names=("total", "A", "B", "A/1", "A/2", "B/1", "B/2"),
        level_of=("total", "region", "region", "leaf", "leaf", "leaf", "leaf"),
        levels=("total", "region", "leaf"),
    )


def make_history(n: int = T, seed: int = 0, noise: float = 2.0) -> pd.DataFrame:
    """Monthly seasonal leaves with trend and noise; aggregates are exact sums."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    bottom = np.stack(
        [
            level + 0.3 * t + amp * np.sin(2 * np.pi * (t + phase) / 12) + rng.normal(0, noise, n)
            for level, amp, phase in [(60, 10, 0), (40, 6, 3), (80, 12, 6), (30, 5, 9)]
        ]
    )
    hier = make_hierarchy()
    full = hier.S @ bottom
    ds = pd.date_range("2015-01-01", periods=n, freq="MS")
    return contracts.from_matrix(full, hier, list(ds), contracts.Y)


@pytest.fixture(scope="module")
def hier() -> contracts.Hierarchy:
    return make_hierarchy()


@pytest.fixture(scope="module")
def hist() -> pd.DataFrame:
    return make_history()


@pytest.fixture(scope="module")
def fitted(hier, hist) -> dict[str, contracts.Forecaster]:
    return {name: make_forecaster(name).fit(hist, hier) for name in spec.FORECASTERS}


def expected_ds(hist: pd.DataFrame, h: int) -> pd.DatetimeIndex:
    return pd.date_range(hist["ds"].max() + pd.offsets.MonthBegin(1), periods=h, freq="MS")


@pytest.mark.parametrize("name", spec.FORECASTERS)
def test_forecast_schema_and_coverage(name, fitted, hier, hist):
    model = fitted[name]
    assert isinstance(model, contracts.Forecaster)
    assert model.name == name
    out = model.forecast(H)
    assert list(out.columns) == [contracts.ID, contracts.DS, contracts.YHAT]
    assert len(out) == hier.n_nodes * H
    assert not out.duplicated([contracts.ID, contracts.DS]).any()
    assert set(out[contracts.ID]) == set(hier.names)
    for _, g in out.groupby(contracts.ID):
        assert list(g[contracts.DS]) == list(expected_ds(hist, H))
    assert np.isfinite(out[contracts.YHAT]).all()


@pytest.mark.parametrize("name", spec.FORECASTERS)
def test_shorter_horizon(name, fitted, hier):
    out = fitted[name].forecast(3)
    assert len(out) == hier.n_nodes * 3


@pytest.mark.parametrize("name", spec.FORECASTERS)
def test_residual_schema(name, fitted, hier, hist):
    res = fitted[name].residuals()
    assert list(res.columns) == [contracts.ID, contracts.DS, contracts.RESID]
    assert len(res) == hier.n_nodes * T
    assert set(res[contracts.ID]) == set(hier.names)
    assert set(res[contracts.DS]) == set(hist[contracts.DS])
    assert res[contracts.RESID].notna().mean() > 0.5
    assert np.isfinite(res[contracts.RESID].dropna()).all()


def test_forecasts_are_sensible(fitted, hist):
    """Every forecaster stays in the neighbourhood of the series it was given."""
    last = hist[hist["ds"] >= hist["ds"].max() - pd.DateOffset(months=11)]
    level = last.groupby(contracts.ID)[contracts.Y].mean()
    for name, model in fitted.items():
        fc = model.forecast(H).groupby(contracts.ID)[contracts.YHAT].mean()
        ratio = (fc / level).to_numpy()
        assert ((ratio > 0.7) & (ratio < 1.5)).all(), name


def test_snaive_exact_values(hier):
    t = np.arange(30)
    ds = pd.date_range("2020-01-01", periods=30, freq="MS")
    bottom = np.stack([10.0 * (k + 1) + (t % 12) + 100 * (t // 12) for k in range(4)])
    frame = contracts.from_matrix(hier.S @ bottom, hier, list(ds), contracts.Y)
    model = SeasonalNaive().fit(frame, hier)
    mat, _ = contracts.to_matrix(model.forecast(14), hier, contracts.YHAT)
    full = hier.S @ bottom
    # Origin is index 29 (June of year 3): steps 1..12 repeat indices 18..29, then wrap.
    expected = full[:, 18:30]
    assert np.allclose(mat[:, :12], expected)
    assert np.allclose(mat[:, 12:], expected[:, :2])
    res, _ = contracts.to_matrix(model.residuals(), hier, contracts.RESID)
    assert np.isnan(res[:, :12]).all()
    assert np.allclose(res[:, 12:], full[:, 12:] - full[:, :-12])


def test_ets_reproduces_clean_seasonal_series(hier):
    n = 72
    t = np.arange(n)
    pattern = 20 + 8 * np.sin(2 * np.pi * t / 12)
    bottom = np.stack([pattern * (k + 1) for k in range(4)])
    ds = pd.date_range("2016-01-01", periods=n, freq="MS")
    frame = contracts.from_matrix(hier.S @ bottom, hier, list(ds), contracts.Y)
    model = ETSForecaster().fit(frame, hier)
    mat, _ = contracts.to_matrix(model.forecast(H), hier, contracts.YHAT)
    truth = hier.S @ np.stack(
        [(20 + 8 * np.sin(2 * np.pi * (n + np.arange(H)) / 12)) * (k + 1) for k in range(4)]
    )
    assert np.max(np.abs(mat - truth) / truth) < 0.03


def test_ets_maps_ids_by_name(hier):
    """A frame with rows shuffled and different series levels must keep names aligned."""
    frame = make_history(seed=5).sample(frac=1.0, random_state=1)
    model = ETSForecaster().fit(frame, hier)
    last = frame.sort_values(contracts.DS).groupby(contracts.ID).tail(12)
    level = last.groupby(contracts.ID)[contracts.Y].mean()
    fc = model.forecast(H).groupby(contracts.ID)[contracts.YHAT].mean()
    assert ((fc / level) > 0.7).all() and ((fc / level) < 1.5).all()
    assert fc["total"] > fc["A"] > fc["A/1"]


@pytest.mark.parametrize("name", ["lgbm", "hybrid"])
def test_deterministic_for_fixed_seed(name, hier, hist, fitted):
    again = make_forecaster(name).fit(hist, hier)
    pd.testing.assert_frame_equal(fitted[name].forecast(H), again.forecast(H))
    pd.testing.assert_frame_equal(fitted[name].residuals(), again.residuals())


@pytest.mark.parametrize("name", spec.FORECASTERS)
def test_post_origin_values_cannot_leak(name, hier, hist, fitted):
    """Wild values appended after the origin are cut off before fitting and change nothing."""
    origin = hist["ds"].max()
    future = hist[hist["ds"] > origin - pd.DateOffset(months=12)].copy()
    future["ds"] = future["ds"] + pd.DateOffset(months=12)
    future[contracts.Y] *= 1000.0
    extended = pd.concat([hist, future], ignore_index=True)
    truncated = extended[extended["ds"] <= origin]
    again = make_forecaster(name).fit(truncated, hier).forecast(H)
    pd.testing.assert_frame_equal(fitted[name].forecast(H), again)


def test_training_set_never_uses_a_target_past_the_window(hier, hist):
    history = History.from_frame(hist, hier)
    scale = series_scale(history.Y)[:, None]
    Z = history.Y / scale
    months = calendar_months(history, H)
    levels = level_codes(hier)
    n_lags = 24
    for k in range(1, H + 1):
        ts = build_training_set([Z], Z, months, levels, k, n_lags)
        assert ts.target_idx.max() <= history.T - 1
        assert ts.anchor.max() + k <= history.T - 1
        assert ts.anchor.min() >= n_lags - 1
        assert np.array_equal(ts.y, Z[ts.node, ts.target_idx])
        # month feature is the month of the target, not of the anchor
        assert np.array_equal(ts.X[:, n_lags], months[ts.target_idx])
        # the newest lag is the value at the anchor, the oldest n_lags - 1 steps back
        assert np.array_equal(ts.X[:, 0], Z[ts.node, ts.anchor])
        assert np.array_equal(ts.X[:, n_lags - 1], Z[ts.node, ts.anchor - (n_lags - 1)])


def test_training_set_too_short_raises(hier):
    Z = np.ones((hier.n_nodes, 30))
    with pytest.raises(ValueError):
        build_training_set([Z], Z, np.ones(42), level_codes(hier), 12, 24)


def test_make_forecaster_names_and_error():
    for name in spec.FORECASTERS:
        assert make_forecaster(name).name == name
    assert isinstance(make_forecaster("lgbm", seed=3), LGBMForecaster)
    assert isinstance(make_forecaster("hybrid", seed=3), HybridForecaster)
    with pytest.raises(ValueError):
        make_forecaster("prophet")


def test_series_scale_guard():
    Y = np.array(
        [
            [1.0, 2.0, 3.0] * 8,  # normal: mean of last 12 = 2
            [0.0] * 24,  # all zero -> 1.0
            [5.0] * 12 + [0.0] * 12,  # last 12 zero, fallback to mean |y| = 2.5
            [-3.0, 3.0] * 12,  # mean 0 -> mean |y| = 3
        ]
    )
    scale = series_scale(Y)
    assert np.allclose(scale, [2.0, 1.0, 2.5, 3.0])


@pytest.mark.parametrize("name", spec.FORECASTERS)
def test_all_zero_series_does_not_crash(name, hier):
    frame = make_history(seed=2)
    leaf = frame[contracts.ID] == "B/2"
    frame.loc[leaf, contracts.Y] = 0.0
    out = make_forecaster(name).fit(frame, hier).forecast(H)
    assert len(out) == hier.n_nodes * H
    assert np.isfinite(out[contracts.YHAT]).all()
    if name == "snaive":
        assert (out[out[contracts.ID] == "B/2"][contracts.YHAT] == 0).all()


def test_rejects_gappy_history(hier, hist):
    with pytest.raises(ValueError):
        SeasonalNaive().fit(hist.iloc[:-3], hier)
