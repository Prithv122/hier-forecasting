"""Scoring (MASE, RMSSE per series per origin) and the paired bootstrap (PREREGISTRATION.md 4-5).

Scores are long frames with columns ``origin, forecaster, reconciler, unique_id, level, mase,
rmsse``. Nothing here pools across levels: every table and interval is per level.

Excluded series: a series whose scaling error is 0 (or not finite) at an origin is dropped for that
origin. :func:`score` reports the count in ``df.attrs["excluded"]`` (distinct series-origin pairs)
and ``df.attrs["excluded_by_level"]``. ``attrs`` do not survive a CSV round trip, so callers that
persist scores must also persist the count.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from hierfc import spec
from hierfc.contracts import DS, ID, YHAT, Hierarchy, to_matrix

SCORE_COLUMNS = ["origin", "forecaster", "reconciler", ID, "level", "mase", "rmsse"]
VERDICTS = ("better", "worse", "no clear difference")


def scales(y_train: np.ndarray, season: int = spec.SEASON) -> tuple[Any, Any]:
    """In-sample errors of the lag-``season`` seasonal naive: ``(mae_scale, mse_scale)``.

    ``y_train`` is ``(T,)`` for one series or ``(n_series, T)``; the scales are taken along the last
    axis and are scalars or ``(n_series,)`` arrays accordingly.
    """
    y = np.asarray(y_train, dtype=float)
    if y.shape[-1] <= season:
        raise ValueError(f"need more than {season} training periods, got {y.shape[-1]}")
    err = y[..., season:] - y[..., :-season]
    return np.mean(np.abs(err), axis=-1), np.mean(err**2, axis=-1)


def score(
    forecasts: pd.DataFrame,
    y_all: pd.DataFrame,
    hierarchy: Hierarchy,
    season: int = spec.SEASON,
) -> pd.DataFrame:
    """MASE and RMSSE per (origin, forecaster, reconciler, series), averaged over the horizon.

    The scaling errors come from the training window of each origin (``ds <= origin``).
    """
    y_mat, y_ds = to_matrix(y_all, hierarchy, "y")
    y_index = pd.DatetimeIndex(y_ds)
    levels = np.array(hierarchy.level_of, dtype=object)
    ids = np.array(hierarchy.names, dtype=object)

    frames: list[pd.DataFrame] = []
    excluded_by_level: dict[str, int] = {lv: 0 for lv in hierarchy.levels}
    scale_cache: dict[pd.Timestamp, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    keys = ["origin", "forecaster", "reconciler"]
    for (origin, forecaster, reconciler), grp in forecasts.groupby(keys, sort=False):
        origin = pd.Timestamp(origin)
        if origin not in scale_cache:
            train = y_mat[:, y_index <= origin]
            mae_s, mse_s = scales(train, season)
            valid = np.isfinite(mae_s) & np.isfinite(mse_s) & (mae_s > 0) & (mse_s > 0)
            for lv in levels[~valid]:
                excluded_by_level[lv] += 1
            scale_cache[origin] = (mae_s, mse_s, valid)
        mae_s, mse_s, valid = scale_cache[origin]

        fc_mat, fc_ds = to_matrix(grp[[ID, DS, YHAT]], hierarchy, YHAT)
        pos = y_index.get_indexer(pd.DatetimeIndex(fc_ds))
        if (pos < 0).any():
            raise ValueError(f"forecast dates for origin {origin} are not in the actuals")
        err = fc_mat - y_mat[:, pos]
        with np.errstate(divide="ignore", invalid="ignore"):
            mase = np.mean(np.abs(err), axis=1) / mae_s
            rmsse = np.sqrt(np.mean(err**2, axis=1) / mse_s)
        frames.append(
            pd.DataFrame(
                {
                    "origin": origin,
                    "forecaster": forecaster,
                    "reconciler": reconciler,
                    ID: ids[valid],
                    "level": levels[valid],
                    "mase": mase[valid],
                    "rmsse": rmsse[valid],
                }
            )
        )

    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=SCORE_COLUMNS)
    out.attrs["excluded"] = int(sum(excluded_by_level.values()))
    out.attrs["excluded_by_level"] = excluded_by_level
    return out


def level_table(
    scores: pd.DataFrame, metric: str = "mase", levels: Sequence[str] | None = None
) -> pd.DataFrame:
    """Mean ``metric`` per (forecaster, reconciler) row and level column, levels top to bottom.

    ``levels`` fixes the column order; by default it is the order of first appearance in ``scores``
    (``score`` emits nodes top level first).
    """
    order = list(levels) if levels is not None else list(pd.unique(scores["level"]))
    present = set(scores["level"])
    order = [lv for lv in order if lv in present]
    table = scores.groupby(["forecaster", "reconciler", "level"])[metric].mean().unstack("level")
    return table[order]


def _verdict(lo: float, hi: float) -> str:
    if hi < 0:
        return "better"
    if lo > 0:
        return "worse"
    return "no clear difference"


def paired_differences(
    scores: pd.DataFrame,
    candidate: tuple[str, str],
    reference: tuple[str, str],
    level: str,
    metric: str = "mase",
) -> pd.Series:
    """``metric(candidate) - metric(reference)`` per (series, origin) at ``level``, inner-joined."""
    sub = scores[scores["level"] == level]

    def pick(key: tuple[str, str]) -> pd.Series:
        rows = sub[(sub["forecaster"] == key[0]) & (sub["reconciler"] == key[1])]
        series = rows.set_index([ID, "origin"])[metric]
        if not series.index.is_unique:
            raise ValueError(f"duplicate (series, origin) rows for {key} at level {level!r}")
        return series

    both = pd.concat([pick(candidate), pick(reference)], axis=1, join="inner").dropna()
    if both.empty:
        raise ValueError(f"no paired scores for {candidate} vs {reference} at level {level!r}")
    return both.iloc[:, 0] - both.iloc[:, 1]


def _bootstrap_means(
    d: pd.Series, B: int, rng: np.random.Generator, min_series: int
) -> tuple[float, np.ndarray, str, int]:
    """Point estimate, B resampled means, unit and series count for a series of paired ``d``."""
    by_series = d.groupby(level=0)
    sums = by_series.sum().to_numpy()
    counts = by_series.size().to_numpy().astype(float)
    n_series = len(sums)
    point = float(d.mean())
    if n_series >= min_series:
        idx = rng.integers(0, n_series, size=(B, n_series))
        boots = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
        return point, boots, "series", n_series
    values = d.to_numpy()
    idx = rng.integers(0, len(values), size=(B, len(values)))
    return point, values[idx].mean(axis=1), "series-origin", n_series


def paired_bootstrap(
    scores: pd.DataFrame,
    *,
    candidate: tuple[str, str],
    reference: tuple[str, str],
    level: str,
    metric: str = "mase",
    B: int = spec.BOOTSTRAP_B,
    seed: int = spec.BOOTSTRAP_SEED,
    min_series: int = spec.MIN_SERIES_FOR_SERIES_BOOTSTRAP,
    alpha: float = spec.ALPHA,
) -> dict[str, Any]:
    """Percentile bootstrap of the mean paired difference (candidate - reference) at one level.

    Unit is the series (all its origins kept) when the level has at least ``min_series`` series,
    otherwise the (series, origin) pair. ``candidate``/``reference`` are (forecaster, reconciler).
    Negative means the candidate is better.
    """
    d = paired_differences(scores, candidate, reference, level, metric)
    rng = np.random.default_rng(seed)
    point, boots, unit, n_series = _bootstrap_means(d, B, rng, min_series)
    lo, hi = (float(q) for q in np.quantile(boots, [alpha / 2, 1 - alpha / 2]))
    return {
        "mean": point,
        "lo": lo,
        "hi": hi,
        "unit": unit,
        "n_series": n_series,
        "n_pairs": len(d),
        "verdict": _verdict(lo, hi),
    }


def bootstrap_gain_difference(
    scores_a: pd.DataFrame,
    scores_b: pd.DataFrame,
    *,
    candidate: tuple[str, str],
    reference: tuple[str, str],
    level: str,
    metric: str = "mase",
    B: int = spec.BOOTSTRAP_B,
    seed: int = spec.BOOTSTRAP_SEED,
    min_series: int = spec.MIN_SERIES_FOR_SERIES_BOOTSTRAP,
    alpha: float = spec.ALPHA,
) -> dict[str, Any]:
    """Interval for gain(a) - gain(b), series resampled independently within each dataset.

    Gain is the mean paired difference (candidate - reference) at ``level``. "better" means the
    interval is entirely below 0, i.e. dataset a gains more than dataset b. ``n_series`` and
    ``n_pairs`` are ``(a, b)`` tuples; ``unit`` is the shared unit or ``"mixed"``.
    """
    d_a = paired_differences(scores_a, candidate, reference, level, metric)
    d_b = paired_differences(scores_b, candidate, reference, level, metric)
    rng = np.random.default_rng(seed)
    point_a, boots_a, unit_a, n_a = _bootstrap_means(d_a, B, rng, min_series)
    point_b, boots_b, unit_b, n_b = _bootstrap_means(d_b, B, rng, min_series)
    diff = boots_a - boots_b
    lo, hi = (float(q) for q in np.quantile(diff, [alpha / 2, 1 - alpha / 2]))
    return {
        "mean": point_a - point_b,
        "mean_a": point_a,
        "mean_b": point_b,
        "lo": lo,
        "hi": hi,
        "unit": unit_a if unit_a == unit_b else "mixed",
        "n_series": (n_a, n_b),
        "n_pairs": (len(d_a), len(d_b)),
        "verdict": _verdict(lo, hi),
    }
