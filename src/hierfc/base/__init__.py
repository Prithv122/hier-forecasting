"""Base forecasters: seasonal naive, ETS, global LightGBM and the ETS + LightGBM hybrid."""

from __future__ import annotations

from hierfc import contracts, spec
from hierfc.base.ets import ETSForecaster
from hierfc.base.hybrid import HybridForecaster
from hierfc.base.lgbm import LGBMForecaster
from hierfc.base.naive import SeasonalNaive

__all__ = [
    "ETSForecaster",
    "HybridForecaster",
    "LGBMForecaster",
    "SeasonalNaive",
    "make_forecaster",
]


def make_forecaster(name: str, seed: int = spec.MODEL_SEED) -> contracts.Forecaster:
    """Build the forecaster registered under ``name`` (one of ``spec.FORECASTERS``)."""
    if name == "snaive":
        return SeasonalNaive()
    if name == "ets":
        return ETSForecaster()
    if name == "lgbm":
        return LGBMForecaster(seed=seed)
    if name == "hybrid":
        return HybridForecaster(seed=seed)
    raise ValueError(f"unknown forecaster {name!r}; expected one of {spec.FORECASTERS}")
