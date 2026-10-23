"""ETS plus a global LightGBM fitted to the ETS in-sample residuals."""

from __future__ import annotations

import numpy as np
import pandas as pd

from hierfc import contracts, spec
from hierfc.base.ets import ETSForecaster
from hierfc.base.features import (
    DirectEnsemble,
    History,
    calendar_months,
    level_codes,
    series_scale,
)


class HybridForecaster:
    """``forecast = ETS forecast + scale * predicted residual``.

    The residual model is the direct LightGBM of ``lgbm.py`` with target ``r[t+k] / scale`` where
    ``r`` is the ETS in-sample residual and ``scale`` is the same per-series scale of ``y`` used
    by the pure LightGBM forecaster. Features: scaled lags of ``r``, scaled lags of ``y``, target
    month and level id.

    ``residuals()`` are ETS residuals minus the in-sample ``k=1`` residual prediction, NaN for the
    first ``n_lags`` months. As with ``lgbm``, in-sample fits understate out-of-sample error.
    """

    name = "hybrid"

    def __init__(
        self,
        seed: int = spec.MODEL_SEED,
        n_lags: int = 24,
        horizon: int = spec.H,
        season: int = spec.SEASON,
    ) -> None:
        self.seed, self.n_lags, self.horizon, self.season = seed, n_lags, horizon, season

    def fit(self, y_hist: pd.DataFrame, hierarchy: contracts.Hierarchy) -> HybridForecaster:
        hist = self._hist = History.from_frame(y_hist, hierarchy)
        self._ets = ETSForecaster(self.season, self.horizon).fit(y_hist, hierarchy)
        self._scale = series_scale(hist.Y)[:, None]
        r = np.where(np.isfinite(self._ets.resid_matrix), self._ets.resid_matrix, np.nan)
        self._r = r
        r_scaled = r / self._scale
        self._ens = DirectEnsemble(self.n_lags, self.horizon, self.seed).fit(
            [r_scaled, hist.Y / self._scale],
            r_scaled,
            calendar_months(hist, self.horizon),
            level_codes(hierarchy),
        )
        return self

    def forecast(self, h: int) -> pd.DataFrame:
        mat = self._ets.forecast_matrix(h) + self._scale * self._ens.predict(h)
        return self._hist.forecast_frame(mat)

    def residuals(self) -> pd.DataFrame:
        resid = self._r - self._scale * self._ens.fitted_one_step
        return self._hist.resid_frame(np.where(np.isfinite(resid), resid, np.nan))
