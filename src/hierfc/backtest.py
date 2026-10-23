"""Expanding-window rolling-origin backtest (PREREGISTRATION.md section 3).

For every origin each forecaster is refitted on data with ``ds <= origin`` only, forecasts all nodes
``h`` steps ahead, and every reconciler is applied to that base forecast. The output is one long
frame: ``origin, forecaster, reconciler, unique_id, ds, step, yhat``.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import pandas as pd

from hierfc import spec
from hierfc.contracts import DS, ID, Hierarchy, Y, to_matrix

MIN_TRAIN = 36  # months of history required at the earliest origin
COHERENCE_TOL = 1e-6  # max |y~ - S y~_bottom| relative to max |y~|
FORECAST_COLUMNS = ["origin", "forecaster", "reconciler", ID, DS, "step", "yhat"]

MakeForecaster = Callable[[str], Any]
ReconcileFn = Callable[..., np.ndarray]


def make_origins(
    ds: Sequence[pd.Timestamp], h: int, n_origins: int, step: int, min_train: int = MIN_TRAIN
) -> list[pd.Timestamp]:
    """Origins in ascending order; the last one is ``h`` months before the final date.

    ``ds`` are the distinct monthly dates of the data. Raises if they are not contiguous months or
    if the earliest origin would have fewer than ``min_train`` months of history.
    """
    dates = pd.DatetimeIndex(sorted(set(pd.to_datetime(list(ds)))))
    periods = dates.to_period("M")
    if len(periods) > 1 and not (np.diff(periods.asi8) == 1).all():
        raise ValueError("dates must be contiguous monthly steps")
    if n_origins < 1 or step < 1 or h < 1:
        raise ValueError("h, n_origins and step must be positive")
    last = len(dates) - 1 - h
    first = last - (n_origins - 1) * step
    if first + 1 < min_train:
        raise ValueError(
            f"not enough history: earliest origin would have {first + 1} months of training "
            f"data, need at least {min_train} ({len(dates)} months, h={h}, "
            f"n_origins={n_origins}, step={step})"
        )
    return [dates[first + k * step] for k in range(n_origins)]


def coherence_error(rec: np.ndarray, hierarchy: Hierarchy) -> float:
    """Max |y~ - S y~_bottom| relative to the largest absolute reconciled value."""
    bottom = rec[-hierarchy.n_bottom :]
    scale = max(float(np.abs(rec).max()), 1e-12)
    return float(np.abs(rec - hierarchy.S @ bottom).max() / scale)


def run_backtest(
    y_all: pd.DataFrame,
    hierarchy: Hierarchy,
    forecasters: Sequence[str],
    reconcilers: Sequence[str],
    *,
    h: int = spec.H,
    n_origins: int = spec.N_ORIGINS,
    step: int = spec.ORIGIN_STEP,
    make_forecaster: MakeForecaster | None = None,
    reconcile_fn: ReconcileFn | None = None,
    progress: Callable[[str], None] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run the backtest. ``y_all`` is the long ``unique_id, ds, y`` frame of every node.

    ``make_forecaster(name)`` and ``reconcile_fn(base, hierarchy, method, residuals=, history=)``
    default to ``hierfc.base.make_forecaster`` and ``hierfc.reconcile.reconcile``; the reconciler
    named ``"base"`` leaves the forecast unchanged. Every reconciled matrix is checked for
    coherence and a violation raises.

    Returns ``(forecasts, diagnostics)``. Diagnostics hold, per forecaster and reconciler, the
    count of negative ``yhat`` and the worst coherence error, plus wall seconds per forecaster.
    """
    if make_forecaster is None:
        from hierfc.base import make_forecaster as _make

        make_forecaster = _make
    if reconcile_fn is None:
        from hierfc.reconcile import reconcile as _reconcile

        reconcile_fn = _reconcile
    say = progress or (lambda _msg: None)

    origins = make_origins(sorted(y_all[DS].unique()), h, n_origins, step)
    names = np.array(hierarchy.names, dtype=object)
    negatives = {f: dict.fromkeys(reconcilers, 0) for f in forecasters}
    coherence = {f: dict.fromkeys(reconcilers, 0.0) for f in forecasters}
    wall = dict.fromkeys(forecasters, 0.0)
    fallbacks = dict.fromkeys(forecasters, 0)  # series where ETS fell back to seasonal naive
    pieces: list[pd.DataFrame] = []

    for i, origin in enumerate(origins, start=1):
        # Strict slice: nothing after the origin is ever handed to a forecaster.
        hist = y_all.loc[y_all[DS] <= origin, [ID, DS, Y]].reset_index(drop=True)
        y_train, _ = to_matrix(hist, hierarchy, Y)
        for fname in forecasters:
            start = time.perf_counter()
            fc = make_forecaster(fname).fit(hist, hierarchy)
            base, fc_ds = to_matrix(fc.forecast(h), hierarchy, "yhat")
            resid, _ = to_matrix(fc.residuals(), hierarchy, "resid")
            wall[fname] += time.perf_counter() - start
            fallbacks[fname] += int(getattr(fc, "n_fallback", 0))
            if base.shape[1] != h or min(fc_ds) <= origin:
                raise ValueError(f"{fname}: forecast must cover {h} steps after origin {origin}")
            ds_arr = np.tile(pd.DatetimeIndex(fc_ds).to_numpy(), hierarchy.n_nodes)
            step_arr = np.tile(np.arange(1, h + 1), hierarchy.n_nodes)
            id_arr = np.repeat(names, h)
            for rname in reconcilers:
                if rname == "base":
                    rec = base
                else:
                    rec = reconcile_fn(base, hierarchy, rname, residuals=resid, history=y_train)
                    err = coherence_error(rec, hierarchy)
                    if err > COHERENCE_TOL:
                        raise ValueError(
                            f"{fname}/{rname} at origin {origin} is not coherent "
                            f"(relative error {err:.3g} > {COHERENCE_TOL:g})"
                        )
                    coherence[fname][rname] = max(coherence[fname][rname], err)
                negatives[fname][rname] += int((rec < 0).sum())
                pieces.append(
                    pd.DataFrame(
                        {
                            "origin": origin,
                            "forecaster": fname,
                            "reconciler": rname,
                            ID: id_arr,
                            DS: ds_arr,
                            "step": step_arr,
                            "yhat": rec.reshape(-1),
                        }
                    )
                )
        say(f"origin {i}/{len(origins)} ({origin:%Y-%m}) done")

    forecasts = pd.concat(pieces, ignore_index=True)
    diagnostics = {
        "origins": [f"{o:%Y-%m-%d}" for o in origins],
        "negative_forecasts": negatives,
        "coherence_max_error": coherence,
        "wall_seconds": wall,
        "ets_fallback_series": fallbacks,
    }
    return forecasts, diagnostics
