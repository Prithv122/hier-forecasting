"""Frozen protocol values. Mirrors PREREGISTRATION.md; change only through a logged amendment."""

from __future__ import annotations

from dataclasses import dataclass

# Backtest -------------------------------------------------------------------------------------
H = 12  # forecast horizon, months
SEASON = 12  # seasonal period; also the lag of the MASE / RMSSE scaling naive
N_ORIGINS = 12  # expanding-window origins
ORIGIN_STEP = 3  # months between consecutive origins; the last origin is h months before the end

FORECASTERS = ("snaive", "ets", "lgbm", "hybrid")
RECONCILERS = ("base", "bu", "td", "ols", "wls_struct", "mint_shrink")

# Confirmatory tests ---------------------------------------------------------------------------
PRIMARY_DATASET = "tourism"
PRIMARY_BASE = "ets"
PRIMARY_RECONCILER = "mint_shrink"
BOOTSTRAP_B = 2000
BOOTSTRAP_SEED = 20261023
ALPHA = 0.05
# A level with fewer series than this is bootstrapped over (series, origin) pairs instead of
# over whole series; see PREREGISTRATION.md section 5.
MIN_SERIES_FOR_SERIES_BOOTSTRAP = 5

# Seeds ----------------------------------------------------------------------------------------
SYNTH_SEED = 31
MODEL_SEED = 7


@dataclass(frozen=True)
class SynthConfig:
    """Parameters of the synthetic hierarchy (total / region / store / category).

    Every bottom series gets the same total noise variance in both named configs; only how that
    variance is split between shared shocks and idiosyncratic noise differs, so the configs
    differ in error correlation and nothing else.
    """

    name: str
    n_regions: int
    stores_per_region: int
    cats_per_store: int
    n_periods: int
    start: str
    noise_sd: float  # total sd of the noise added to each bottom series
    region_share: float  # share of noise variance from a shock shared by the whole region
    store_share: float  # share from a shock shared by the store's categories
    shock_ar: float  # AR(1) coefficient of every noise component (Amendment A1)


SYNTH_CONFIGS: dict[str, SynthConfig] = {
    "correlated": SynthConfig(
        name="correlated",
        n_regions=5,
        stores_per_region=4,
        cats_per_store=3,
        n_periods=144,
        start="2012-01-01",
        noise_sd=0.15,
        region_share=0.45,
        store_share=0.25,
        shock_ar=0.6,
    ),
    "independent": SynthConfig(
        name="independent",
        n_regions=5,
        stores_per_region=4,
        cats_per_store=3,
        n_periods=144,
        start="2012-01-01",
        noise_sd=0.15,
        region_share=0.0,
        store_share=0.0,
        shock_ar=0.6,
    ),
    # Small set for the CLI smoke run and fast tests: 1 + 2 + 4 + 8 = 15 nodes.
    "smoke": SynthConfig(
        name="smoke",
        n_regions=2,
        stores_per_region=2,
        cats_per_store=2,
        n_periods=72,
        start="2018-01-01",
        noise_sd=0.15,
        region_share=0.45,
        store_share=0.25,
        shock_ar=0.6,
    ),
}

SYNTH_LEVELS = ("total", "region", "store", "category")
TOURISM_LEVELS = ("total", "state", "zone", "region")
DATASETS = ("synth-correlated", "synth-independent", "tourism")
