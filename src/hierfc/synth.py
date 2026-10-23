"""Seeded synthetic hierarchy (total / region / store / category) with tunable error correlation.

Each bottom series is ``level * trend(t) * seasonal(month) * exp(noise_t)``. The signal (level,
trend, seasonality) is drawn from its own random stream and the noise shocks from three further
streams, so two configs with the same seed share the signal exactly and differ only in how the
noise variance is split between shared and idiosyncratic shocks.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from hierfc import spec
from hierfc.contracts import DS, ID, Hierarchy, Y
from hierfc.hierarchy import aggregate, bottom_ids, build_hierarchy
from hierfc.spec import SynthConfig


def _ar1(rng: np.random.Generator, n_series: int, n_periods: int, phi: float) -> np.ndarray:
    """AR(1) shocks with unit stationary variance, shape ``(n_series, n_periods)``."""
    eta = rng.standard_normal((n_series, n_periods))
    out = np.empty_like(eta)
    out[:, 0] = eta[:, 0]
    scale = np.sqrt(1.0 - phi**2)
    for t in range(1, n_periods):
        out[:, t] = phi * out[:, t - 1] + scale * eta[:, t]
    return out


def _harmonics(rng: np.random.Generator, n: int, sd1: float, sd2: float) -> np.ndarray:
    """``n`` random 12-month log-seasonal profiles built from two harmonics."""
    m = np.arange(12)
    out = np.zeros((n, 12))
    for k, sd in ((1, sd1), (2, sd2)):
        a, b = rng.normal(0.0, sd, (2, n, 1))
        out += a * np.sin(2 * np.pi * k * m / 12) + b * np.cos(2 * np.pi * k * m / 12)
    return out


def generate(config: SynthConfig, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Bottom long frame (``unique_id, ds, y``) and its keys frame (region, store, category)."""
    if (
        config.region_share < 0
        or config.store_share < 0
        or config.region_share + config.store_share > 1
    ):
        raise ValueError("region_share and store_share must be >= 0 and sum to at most 1")
    n_reg, n_sto, n_cat, n_t = (
        config.n_regions,
        config.stores_per_region,
        config.cats_per_store,
        config.n_periods,
    )
    n_bottom = n_reg * n_sto * n_cat
    region = np.repeat(np.arange(n_reg), n_sto * n_cat)
    store = np.repeat(np.arange(n_reg * n_sto), n_cat)  # global store index

    keys = pd.DataFrame(
        {
            spec.SYNTH_LEVELS[1]: [f"R{r + 1}" for r in region],
            spec.SYNTH_LEVELS[2]: [f"S{s % n_sto + 1}" for s in store],
            spec.SYNTH_LEVELS[3]: [f"C{c + 1}" for c in np.tile(np.arange(n_cat), n_reg * n_sto)],
        }
    )

    sig_ss, reg_ss, sto_ss, idio_ss = np.random.SeedSequence(seed).spawn(4)

    # Signal: identical for every config with the same seed and shape.
    rng = np.random.default_rng(sig_ss)
    level = np.exp(rng.uniform(np.log(100.0), np.log(1000.0), n_bottom))
    slope = rng.normal(0.001, 0.002, n_bottom)  # log-trend per month, mild
    common = _harmonics(rng, 1, 0.15, 0.05)  # shared by every series
    reg_dev = _harmonics(rng, n_reg, 0.15, 0.05)  # shared within a region
    own_dev = rng.normal(0.0, 0.04, (n_bottom, 12))  # per-series perturbation
    log_season = common + reg_dev[region] + own_dev
    log_season -= log_season.mean(axis=1, keepdims=True)

    ds = pd.date_range(config.start, periods=n_t, freq="MS")
    t = np.arange(n_t)
    log_signal = (
        np.log(level)[:, None]
        + slope[:, None] * (t - (n_t - 1) / 2)[None, :]
        + log_season[:, ds.month.to_numpy() - 1]
    )

    # Noise: total variance noise_sd^2 split into region / store / idiosyncratic parts.
    rs, ss = config.region_share, config.store_share
    reg_shock = _ar1(np.random.default_rng(reg_ss), n_reg, n_t, config.shock_ar)
    sto_shock = _ar1(np.random.default_rng(sto_ss), n_reg * n_sto, n_t, config.shock_ar)
    idio = np.random.default_rng(idio_ss).standard_normal((n_bottom, n_t))
    noise = config.noise_sd * (
        np.sqrt(rs) * reg_shock[region]
        + np.sqrt(ss) * sto_shock[store]
        + np.sqrt(1.0 - rs - ss) * idio
    )

    y = np.exp(log_signal + noise)
    ids = bottom_ids(keys, spec.SYNTH_LEVELS).to_numpy(dtype=object)
    bottom = pd.DataFrame(
        {
            ID: np.repeat(ids, n_t),
            DS: np.tile(ds.to_numpy(dtype="datetime64[ns]"), n_bottom),
            Y: y.reshape(-1),
        }
    )
    return bottom, keys


def load_synth(config_name: str, seed: int) -> tuple[pd.DataFrame, Hierarchy]:
    """All-node long frame of a named synthetic config and its hierarchy."""
    if config_name not in spec.SYNTH_CONFIGS:
        raise ValueError(
            f"unknown synth config {config_name!r}; choose from {sorted(spec.SYNTH_CONFIGS)}"
        )
    bottom, keys = generate(spec.SYNTH_CONFIGS[config_name], seed)
    hierarchy = build_hierarchy(keys, spec.SYNTH_LEVELS)
    return aggregate(bottom, hierarchy), hierarchy
