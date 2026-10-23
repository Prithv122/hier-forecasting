"""Origins arithmetic, output schema, leakage and the coherence check, with fake components."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hierfc.backtest import COHERENCE_TOL, make_origins, run_backtest
from hierfc.contracts import Hierarchy, to_matrix

H = 6


def tiny_hierarchy() -> Hierarchy:
    return Hierarchy(
        S=np.array([[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]]),
        names=("total", "a", "b"),
        level_of=("total", "bottom", "bottom"),
        levels=("total", "bottom"),
    )


def make_y(n: int = 60, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ds = pd.date_range("2015-01-01", periods=n, freq="MS")
    a = 10 + rng.random(n)
    b = 20 + rng.random(n)
    return pd.concat(
        [
            pd.DataFrame({"unique_id": k, "ds": ds, "y": v})
            for k, v in (("total", a + b), ("a", a), ("b", b))
        ],
        ignore_index=True,
    )


class LastValue:
    """Forecasts each node's last value; residuals are first differences."""

    def __init__(self, name: str = "last") -> None:
        self.name = name

    def fit(self, y_hist, hierarchy):
        self.hist = y_hist
        self.y, self.ds = to_matrix(y_hist, hierarchy, "y")
        self.hierarchy = hierarchy
        return self

    def forecast(self, h):
        ds = pd.date_range(self.ds[-1] + pd.offsets.MonthBegin(1), periods=h, freq="MS")
        rows = [
            pd.DataFrame({"unique_id": n, "ds": ds, "yhat": self.y[i, -1]})
            for i, n in enumerate(self.hierarchy.names)
        ]
        return pd.concat(rows, ignore_index=True)

    def residuals(self):
        res = np.full_like(self.y, np.nan)
        res[:, 1:] = np.diff(self.y, axis=1)
        return pd.DataFrame(
            {
                "unique_id": np.repeat(list(self.hierarchy.names), len(self.ds)),
                "ds": np.tile(pd.DatetimeIndex(self.ds), len(self.hierarchy.names)),
                "resid": res.reshape(-1),
            }
        )


class MeanOfAll(LastValue):
    """Would expose leakage: forecasts the mean of every row it was given."""

    def fit(self, y_hist, hierarchy):
        super().fit(y_hist, hierarchy)
        self.y = np.tile(self.y.mean(axis=1, keepdims=True), (1, self.y.shape[1]))
        self.max_seen = y_hist["ds"].max()
        return self


def fake_reconcile(base, hierarchy, method, *, residuals=None, history=None):
    if method == "bu":
        return hierarchy.S @ base[-hierarchy.n_bottom :]
    if method == "broken":
        out = base.copy()
        out[0] += 1.0
        return out
    if method == "slightly_broken":  # relative violation of 1e-7: between 1e-8 and the old 1e-6
        out = hierarchy.S @ base[-hierarchy.n_bottom :]
        out[0] += 1e-7 * np.abs(out).max()
        return out
    raise AssertionError(method)


def run(y, forecasters=("last",), reconcilers=("base", "bu"), factory=LastValue, **kw):
    return run_backtest(
        y,
        tiny_hierarchy(),
        forecasters,
        reconcilers,
        h=H,
        n_origins=3,
        step=3,
        make_forecaster=factory,
        reconcile_fn=fake_reconcile,
        **kw,
    )


def test_make_origins_arithmetic():
    ds = pd.date_range("2000-01-01", periods=144, freq="MS")
    origins = make_origins(ds, h=12, n_origins=12, step=3)
    assert len(origins) == 12
    assert origins[-1] == ds[-13]  # origin + 12 months is the last date
    assert origins[0] == ds[-13 - 33]
    positions = [ds.get_loc(o) for o in origins]
    assert np.diff(positions).tolist() == [3] * 11
    assert origins == sorted(origins)


def test_make_origins_requires_36_months_and_contiguous_dates():
    ds = pd.date_range("2000-01-01", periods=60, freq="MS")
    with pytest.raises(ValueError, match="not enough history"):
        make_origins(ds, h=12, n_origins=12, step=3)
    assert len(make_origins(ds, h=12, n_origins=4, step=3)) == 4
    with pytest.raises(ValueError, match="contiguous"):
        make_origins(ds.delete(10), h=12, n_origins=2, step=3)


def test_output_schema_and_shape():
    y = make_y()
    fc, diag = run(y)
    assert list(fc.columns) == [
        "origin",
        "forecaster",
        "reconciler",
        "unique_id",
        "ds",
        "step",
        "yhat",
    ]
    assert len(fc) == 3 * 2 * 3 * H  # origins x reconcilers x nodes x steps
    assert set(fc["step"]) == set(range(1, H + 1))
    assert (fc["ds"] > fc["origin"]).all()
    assert fc["origin"].nunique() == 3
    base = fc[(fc["reconciler"] == "base")]
    last = y[y["ds"] == base["origin"].min()].set_index("unique_id")["y"]
    first_origin = base[base["origin"] == base["origin"].min()]
    expected = first_origin["unique_id"].map(last).to_numpy()
    assert (first_origin["yhat"].to_numpy() == expected).all()
    assert set(diag) >= {"negative_forecasts", "coherence_max_error", "wall_seconds", "origins"}
    assert diag["wall_seconds"]["last"] >= 0
    assert diag["coherence_max_error"]["last"]["bu"] <= 1e-9
    assert diag["negative_forecasts"]["last"] == {"base": 0, "bu": 0}


def test_progress_callback_is_called_per_origin():
    msgs: list[str] = []
    run(make_y(), progress=msgs.append)
    assert len(msgs) == 3


def test_negative_forecasts_are_counted():
    class Neg(LastValue):
        def forecast(self, h):
            fc = super().forecast(h)
            fc["yhat"] = -1.0
            return fc

    _, diag = run(make_y(), reconcilers=("base",), factory=Neg)
    assert diag["negative_forecasts"]["last"]["base"] == 3 * 3 * H


def test_no_leakage_perturbing_post_origin_values_changes_nothing():
    y = make_y()
    fc, _ = run(y, factory=MeanOfAll)
    origins = sorted(fc["origin"].unique())
    for cut in (origins[0], origins[-1]):
        perturbed = y.copy()
        perturbed.loc[perturbed["ds"] > cut, "y"] += 1000.0
        fc2, _ = run(perturbed, factory=MeanOfAll)
        for origin in (o for o in origins if o <= cut):
            a = fc[fc["origin"] == origin].reset_index(drop=True)
            b = fc2[fc2["origin"] == origin].reset_index(drop=True)
            pd.testing.assert_frame_equal(a, b)
        # the probe is sensitive: origins after the cut do see the perturbed rows
        for origin in (o for o in origins if o > cut):
            assert not fc[fc["origin"] == origin]["yhat"].equals(
                fc2[fc2["origin"] == origin]["yhat"]
            )


def test_forecaster_never_sees_rows_after_the_origin():
    seen: list[tuple[pd.Timestamp, pd.Timestamp]] = []

    class Spy(LastValue):
        def fit(self, y_hist, hierarchy):
            seen.append((y_hist["ds"].max(), y_hist["ds"].min()))
            return super().fit(y_hist, hierarchy)

    fc, _ = run(make_y(), factory=Spy)
    assert [m for m, _ in seen] == sorted(fc["origin"].unique())


def test_incoherent_reconciler_raises():
    with pytest.raises(ValueError, match="not coherent"):
        run(make_y(), reconcilers=("base", "broken"))


def test_coherence_tolerance_is_the_preregistered_one():
    """PREREGISTRATION section 8: relative tolerance 1e-8, so a 1e-7 violation must raise."""
    assert COHERENCE_TOL == 1e-8
    with pytest.raises(ValueError, match="not coherent"):
        run(make_y(), reconcilers=("base", "slightly_broken"))


def test_base_is_not_checked_for_coherence():
    class Incoherent(LastValue):
        def forecast(self, h):
            fc = super().forecast(h)
            fc.loc[fc["unique_id"] == "total", "yhat"] += 5.0
            return fc

    fc, diag = run(make_y(), reconcilers=("base", "bu"), factory=Incoherent)
    assert diag["coherence_max_error"]["last"]["base"] == 0.0
    assert diag["coherence_max_error"]["last"]["bu"] <= 1e-9
    assert len(fc) > 0
