"""Per-series ETS selected by AICc (statsforecast ``AutoETS``)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import AutoETS

from hierfc import contracts, spec
from hierfc.base.features import History

_MODEL = "AutoETS"


class ETSForecaster:
    """AutoETS with ``season_length`` months, fitted independently per node.

    ``fit`` runs the whole job once, forecasting ``horizon`` steps ahead and keeping the in-sample
    fitted values; ``forecast(h)`` then slices, so it is free for any ``h <= horizon``. A series
    for which ETS returns a non-finite forecast falls back to the seasonal naive for that series.
    """

    name = "ets"

    def __init__(self, season: int = spec.SEASON, horizon: int = spec.H) -> None:
        self.season, self.horizon = season, horizon

    def fit(self, y_hist: pd.DataFrame, hierarchy: contracts.Hierarchy) -> ETSForecaster:
        hist = self._hist = History.from_frame(y_hist, hierarchy)
        n_nodes, T = hist.Y.shape
        df = pd.DataFrame(
            {
                contracts.ID: np.repeat(np.array(hierarchy.names, dtype=object), T),
                contracts.DS: np.tile(hist.ds.to_numpy(), n_nodes),
                contracts.Y: hist.Y.reshape(-1),
            }
        )
        sf = StatsForecast(models=[AutoETS(season_length=self.season)], freq="MS", n_jobs=1)
        fcst = sf.forecast(df=df, h=self.horizon, fitted=True)
        fitted = sf.forecast_fitted_values()
        # statsforecast returns series sorted by id, so map back by name, not by position.
        yhat = _by_name(fcst, hierarchy, self.horizon)
        fit_vals = _by_name(fitted, hierarchy, T)
        bad = ~np.isfinite(yhat).all(axis=1)
        self.n_fallback = int(bad.sum())  # reported in the backtest diagnostics
        if bad.any() and self.season <= T:
            last = hist.Y[:, -self.season :]
            yhat[bad] = last[bad][:, np.arange(self.horizon) % self.season]
        self._yhat = yhat
        self._resid = hist.Y - fit_vals
        return self

    @property
    def resid_matrix(self) -> np.ndarray:
        """In-sample ``y - fitted``, shape ``(n_nodes, T)``."""
        return self._resid

    def forecast_matrix(self, h: int) -> np.ndarray:
        if not 1 <= h <= self.horizon:
            raise ValueError(f"h must be in 1..{self.horizon}, got {h}")
        return self._yhat[:, :h]

    def forecast(self, h: int) -> pd.DataFrame:
        return self._hist.forecast_frame(self.forecast_matrix(h))

    def residuals(self) -> pd.DataFrame:
        resid = np.where(np.isfinite(self._resid), self._resid, np.nan)
        return self._hist.resid_frame(resid)


def _by_name(frame: pd.DataFrame, hierarchy: contracts.Hierarchy, n_periods: int) -> np.ndarray:
    wide = frame.pivot(index=contracts.ID, columns=contracts.DS, values=_MODEL)
    mat = wide.loc[list(hierarchy.names)].to_numpy(dtype=float)
    if mat.shape[1] != n_periods:
        raise ValueError(f"expected {n_periods} periods from ETS, got {mat.shape[1]}")
    return mat
