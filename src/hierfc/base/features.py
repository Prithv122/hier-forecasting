"""Shared helpers for the base forecasters: history matrices, scaling, and the direct LightGBM.

The direct strategy trains one global regressor per horizon step ``k``. A training row is an
(series, anchor ``t``) pair: features are the scaled values at ``t, t-1, ..., t-n_lags+1`` plus the
calendar month of the TARGET ``t+k`` and the series' level id; the target is the scaled value at
``t+k``. Only anchors with ``t+k`` inside the fitted window are used, so no row ever looks past
the last month of the history.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from hierfc import contracts, spec


@dataclass(frozen=True)
class History:
    """The fitted history as a matrix ``Y`` of shape ``(n_nodes, T)`` in hierarchy order."""

    Y: np.ndarray
    ds: pd.DatetimeIndex
    hierarchy: contracts.Hierarchy

    @classmethod
    def from_frame(cls, y_hist: pd.DataFrame, hierarchy: contracts.Hierarchy) -> History:
        Y, ds = contracts.to_matrix(y_hist, hierarchy, contracts.Y)
        if not np.isfinite(Y).all():
            raise ValueError("history has gaps or non-finite values")
        index = pd.DatetimeIndex(ds)
        expected = pd.date_range(index[0], periods=len(index), freq="MS")
        if not index.equals(expected):
            raise ValueError("history must be consecutive month-start dates")
        return cls(Y=Y, ds=index, hierarchy=hierarchy)

    @property
    def T(self) -> int:
        return self.Y.shape[1]

    def future_ds(self, h: int) -> pd.DatetimeIndex:
        return pd.date_range(self.ds[-1] + pd.offsets.MonthBegin(1), periods=h, freq="MS")

    def forecast_frame(self, mat: np.ndarray) -> pd.DataFrame:
        ds = list(self.future_ds(mat.shape[1]))
        return contracts.from_matrix(mat, self.hierarchy, ds, contracts.YHAT)

    def resid_frame(self, mat: np.ndarray) -> pd.DataFrame:
        return contracts.from_matrix(mat, self.hierarchy, list(self.ds), contracts.RESID)


def series_scale(Y: np.ndarray, season: int = spec.SEASON) -> np.ndarray:
    """Per-series scale: the mean of the last ``season`` months.

    Guard: a non-positive mean falls back to the mean absolute value of the whole series, and a
    series that is identically zero gets 1.0, so the scale is always positive and finite.
    """
    scale = Y[:, -season:].mean(axis=1)
    fallback = np.abs(Y).mean(axis=1)
    scale = np.where(scale > 0, scale, fallback)
    return np.where(scale > 0, scale, 1.0)


def level_codes(hierarchy: contracts.Hierarchy) -> np.ndarray:
    """Integer level id per node: the position of its level in ``hierarchy.levels``."""
    return np.array([hierarchy.levels.index(lv) for lv in hierarchy.level_of], dtype=float)


def calendar_months(history: History, horizon: int) -> np.ndarray:
    """Month number (1-12) of every history month followed by ``horizon`` future months."""
    return np.concatenate([history.ds.month, history.future_ds(horizon).month]).astype(float)


@dataclass(frozen=True)
class TrainingSet:
    X: np.ndarray
    y: np.ndarray
    node: np.ndarray  # series index of each row
    anchor: np.ndarray  # time index t of each row's last lag
    target_idx: np.ndarray  # time index t+k of each row's target


def _design(
    blocks: list[np.ndarray],
    node: np.ndarray,
    anchor: np.ndarray,
    target_idx: np.ndarray,
    months: np.ndarray,
    levels: np.ndarray,
    n_lags: int,
) -> np.ndarray:
    lag_idx = anchor[:, None] - np.arange(n_lags)[None, :]
    cols = [block[node[:, None], lag_idx] for block in blocks]
    cols.append(months[target_idx][:, None])
    cols.append(levels[node][:, None])
    return np.hstack(cols)


def build_training_set(
    blocks: list[np.ndarray],
    target: np.ndarray,
    months: np.ndarray,
    levels: np.ndarray,
    k: int,
    n_lags: int,
) -> TrainingSet:
    """All (series, anchor) rows for horizon step ``k``; ``target_idx`` never exceeds ``T - 1``."""
    n_nodes, T = target.shape
    anchors = np.arange(n_lags - 1, T - k)
    if anchors.size == 0:
        raise ValueError(f"history of {T} months is too short for {n_lags} lags and step {k}")
    node = np.repeat(np.arange(n_nodes), anchors.size)
    anchor = np.tile(anchors, n_nodes)
    target_idx = anchor + k
    y = target[node, target_idx]
    keep = np.isfinite(y)
    node, anchor, target_idx, y = node[keep], anchor[keep], target_idx[keep], y[keep]
    X = _design(blocks, node, anchor, target_idx, months, levels, n_lags)
    return TrainingSet(X=X, y=y, node=node, anchor=anchor, target_idx=target_idx)


class DirectEnsemble:
    """One LightGBM regressor per horizon step ``k = 1..horizon``, trained across all series.

    ``blocks`` are ``(n_nodes, T)`` matrices of scaled lag features and ``target`` the scaled
    series to predict. ``months`` covers the history plus ``horizon`` future months. Predictions
    are made at the anchor ``T - 1`` (the origin) and are in scaled units.

    ``n_jobs=1`` by default: the models are tiny, a single thread is bitwise reproducible, and
    multi-threaded OpenMP stalls badly when the machine is busy (measured: 3 s vs 100 s). Run
    several origins in parallel instead.
    """

    def __init__(self, n_lags: int, horizon: int, seed: int, n_jobs: int = 1) -> None:
        self.n_lags, self.horizon, self.seed, self.n_jobs = n_lags, horizon, seed, n_jobs
        self.models: list[LGBMRegressor] = []

    def _regressor(self) -> LGBMRegressor:
        return LGBMRegressor(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=20,
            deterministic=True,
            force_row_wise=True,
            random_state=self.seed,
            n_jobs=self.n_jobs,
            verbose=-1,
        )

    def fit(
        self,
        blocks: list[np.ndarray],
        target: np.ndarray,
        months: np.ndarray,
        levels: np.ndarray,
    ) -> DirectEnsemble:
        self.blocks, self.months, self.levels = blocks, months, levels
        n_nodes, self.T = target.shape
        cat = [len(blocks) * self.n_lags, len(blocks) * self.n_lags + 1]  # month, level
        self.models = []
        self.fitted_one_step = np.full((n_nodes, self.T), np.nan)
        for k in range(1, self.horizon + 1):
            ts = build_training_set(blocks, target, months, levels, k, self.n_lags)
            model = self._regressor().fit(ts.X, ts.y, categorical_feature=cat)
            self.models.append(model)
            if k == 1:
                self.fitted_one_step[ts.node, ts.target_idx] = model.predict(ts.X)
        return self

    def predict(self, h: int) -> np.ndarray:
        """Scaled predictions at the origin for steps ``1..h``, shape ``(n_nodes, h)``."""
        if not 1 <= h <= self.horizon:
            raise ValueError(f"h must be in 1..{self.horizon}, got {h}")
        node = np.arange(self.blocks[0].shape[0])
        anchor = np.full(node.size, self.T - 1)
        out = np.empty((node.size, h))
        for k in range(1, h + 1):
            X = _design(
                self.blocks, node, anchor, anchor + k, self.months, self.levels, self.n_lags
            )
            out[:, k - 1] = self.models[k - 1].predict(X)
        return out
