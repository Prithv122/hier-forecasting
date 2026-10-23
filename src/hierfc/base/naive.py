"""Seasonal naive: each month is forecast by the same month of the last observed season."""

from __future__ import annotations

import numpy as np
import pandas as pd

from hierfc import contracts, spec
from hierfc.base.features import History


class SeasonalNaive:
    """``yhat[T+j] = y[T+j-season]`` repeated, so steps beyond one season reuse the last one."""

    name = "snaive"

    def __init__(self, season: int = spec.SEASON) -> None:
        self.season = season

    def fit(self, y_hist: pd.DataFrame, hierarchy: contracts.Hierarchy) -> SeasonalNaive:
        self._hist = History.from_frame(y_hist, hierarchy)
        if self.season > self._hist.T:
            raise ValueError(f"need at least {self.season} months, got {self._hist.T}")
        return self

    def forecast(self, h: int) -> pd.DataFrame:
        Y = self._hist.Y
        last_season = Y[:, -self.season :]
        return self._hist.forecast_frame(last_season[:, np.arange(h) % self.season])

    def residuals(self) -> pd.DataFrame:
        Y = self._hist.Y
        resid = np.full_like(Y, np.nan)
        resid[:, self.season :] = Y[:, self.season :] - Y[:, : -self.season]
        return self._hist.resid_frame(resid)
