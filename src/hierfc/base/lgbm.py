"""Global LightGBM, direct strategy: one model per horizon step over every node."""

from __future__ import annotations

import numpy as np
import pandas as pd

from hierfc import contracts, spec
from hierfc.base.features import (
    DirectEnsemble,
    History,
    calendar_months,
    level_codes,
    series_scale,
)


class LGBMForecaster:
    """One global regressor per step ``k = 1..horizon`` trained on all nodes at once.

    Each series is divided by its scale (mean of its last 12 training months; see
    ``series_scale`` for the guard). Features are the last ``n_lags`` scaled values, the calendar
    month of the target and the node's level id.

    ``residuals()`` are the in-sample one-step errors of the ``k=1`` model, ``y - scale * pred``,
    over the training anchors, NaN for the first ``n_lags`` months. Known limitation: boosted
    trees fit their training data closely, so these residuals understate the true out-of-sample
    error, and a covariance estimated from them (MinT) is correspondingly too small.
    """

    name = "lgbm"

    def __init__(
        self, seed: int = spec.MODEL_SEED, n_lags: int = 24, horizon: int = spec.H
    ) -> None:
        self.seed, self.n_lags, self.horizon = seed, n_lags, horizon

    def fit(self, y_hist: pd.DataFrame, hierarchy: contracts.Hierarchy) -> LGBMForecaster:
        hist = self._hist = History.from_frame(y_hist, hierarchy)
        self._scale = series_scale(hist.Y)[:, None]
        Z = hist.Y / self._scale
        self._ens = DirectEnsemble(self.n_lags, self.horizon, self.seed).fit(
            [Z], Z, calendar_months(hist, self.horizon), level_codes(hierarchy)
        )
        return self

    def forecast(self, h: int) -> pd.DataFrame:
        return self._hist.forecast_frame(self._ens.predict(h) * self._scale)

    def residuals(self) -> pd.DataFrame:
        resid = self._hist.Y - self._scale * self._ens.fitted_one_step
        return self._hist.resid_frame(np.where(np.isfinite(resid), resid, np.nan))
