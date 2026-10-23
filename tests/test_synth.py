from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from hierfc import spec
from hierfc.contracts import DS, ID, TOTAL, Y
from hierfc.synth import generate, load_synth

CORR = spec.SYNTH_CONFIGS["correlated"]
INDEP = spec.SYNTH_CONFIGS["independent"]
SEED = spec.SYNTH_SEED


def _log_matrix(bottom: pd.DataFrame) -> np.ndarray:
    return np.log(bottom.pivot(index=ID, columns=DS, values=Y).to_numpy())


def test_deterministic_for_a_seed() -> None:
    a, ka = generate(CORR, SEED)
    b, kb = generate(CORR, SEED)
    pd.testing.assert_frame_equal(a, b)
    pd.testing.assert_frame_equal(ka, kb)
    c, _ = generate(CORR, SEED + 1)
    assert not np.allclose(a[Y].to_numpy(), c[Y].to_numpy())


def test_shapes_and_positive() -> None:
    bottom, keys = generate(CORR, SEED)
    assert list(keys.columns) == list(spec.SYNTH_LEVELS[1:])
    assert len(keys) == 60
    assert len(bottom) == 60 * 144
    assert bottom[ID].nunique() == 60
    assert (bottom[Y] > 0).all() and np.isfinite(bottom[Y]).all()
    ds = pd.DatetimeIndex(np.sort(bottom[DS].unique()))
    assert len(ds) == 144
    assert ds[0] == pd.Timestamp(CORR.start)
    assert ds.is_month_start.all()
    assert (np.diff(ds.year * 12 + ds.month) == 1).all()


@pytest.mark.parametrize("name", ["correlated", "independent"])
def test_level_counts_match_spec(name: str) -> None:
    full, h = load_synth(name, SEED)
    assert [len(h.nodes_in(lv)) for lv in spec.SYNTH_LEVELS] == [1, 5, 20, 60]
    assert h.n_nodes == 86
    assert set(full[ID]) == set(h.names)
    assert len(full) == 86 * 144
    assert (full[Y] > 0).all()


def test_load_synth_aggregates_and_rejects_unknown_name() -> None:
    full, h = load_synth("smoke", SEED)
    wide = full.pivot(index=ID, columns=DS, values=Y)
    np.testing.assert_allclose(wide.loc[TOTAL], wide.loc[list(h.bottom_names)].sum(axis=0))
    with pytest.raises(ValueError, match="unknown synth config"):
        load_synth("nope", SEED)


def test_signal_identical_across_configs() -> None:
    quiet = [dataclasses.replace(c, noise_sd=0.0) for c in (CORR, INDEP)]
    (a, ka), (b, kb) = (generate(c, SEED) for c in quiet)
    pd.testing.assert_frame_equal(a, b)
    pd.testing.assert_frame_equal(ka, kb)


def test_noise_has_same_total_sd_but_different_correlation() -> None:
    # Longer series so the empirical correlation is tight.
    cfgs = [dataclasses.replace(c, n_periods=1200) for c in (CORR, INDEP)]
    signal, _ = generate(dataclasses.replace(cfgs[0], noise_sd=0.0), SEED)
    log_signal = _log_matrix(signal)
    ids = signal.pivot(index=ID, columns=DS, values=Y).index.to_numpy()
    region = np.array([i.split("/")[0] for i in ids])
    store_key = np.array([i.rsplit("/", 1)[0] for i in ids])

    stats = {}
    for cfg in cfgs:
        bottom, _ = generate(cfg, SEED)
        noise = _log_matrix(bottom) - log_signal
        assert noise.std() == pytest.approx(cfg.noise_sd, rel=0.05)
        corr = np.corrcoef(noise)
        same_region = region[:, None] == region[None, :]
        same_store = store_key[:, None] == store_key[None, :]
        off = ~np.eye(len(ids), dtype=bool)
        stats[cfg.name] = (
            corr[same_region & ~same_store & off].mean(),  # share a region only
            corr[same_store & off].mean(),  # share a store (and a region)
            corr[~same_region].mean(),  # different regions
        )
    c_reg, c_sto, c_far = stats["correlated"]
    i_reg, i_sto, i_far = stats["independent"]
    assert c_reg == pytest.approx(CORR.region_share, abs=0.05)
    assert c_sto == pytest.approx(CORR.region_share + CORR.store_share, abs=0.05)
    assert abs(c_far) < 0.05
    assert abs(i_reg) < 0.05 and abs(i_sto) < 0.05 and abs(i_far) < 0.05
    assert c_reg > i_reg + 0.3


def test_noise_persistence_is_the_same_in_both_configs() -> None:
    """Amendment A1: only cross-series correlation differs, not each series' autocorrelation."""
    cfgs = [dataclasses.replace(c, n_periods=1200) for c in (CORR, INDEP)]
    log_signal = _log_matrix(generate(dataclasses.replace(cfgs[0], noise_sd=0.0), SEED)[0])
    lag1 = {}
    for cfg in cfgs:
        noise = _log_matrix(generate(cfg, SEED)[0]) - log_signal
        centred = noise - noise.mean(axis=1, keepdims=True)
        lag1[cfg.name] = np.mean(
            (centred[:, 1:] * centred[:, :-1]).sum(axis=1) / (centred**2).sum(axis=1)
        )
    assert lag1["correlated"] == pytest.approx(CORR.shock_ar, abs=0.05)
    assert lag1["independent"] == pytest.approx(INDEP.shock_ar, abs=0.05)


def test_shares_must_be_valid() -> None:
    with pytest.raises(ValueError, match="share"):
        generate(dataclasses.replace(CORR, region_share=0.8, store_share=0.5), SEED)
