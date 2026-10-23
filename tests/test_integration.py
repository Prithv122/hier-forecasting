"""End-to-end checks with the real forecasters and reconcilers (PREREGISTRATION.md section 8)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hierfc import spec
from hierfc.backtest import run_backtest
from hierfc.contracts import DS, Y
from hierfc.synth import load_synth


@pytest.fixture(scope="module")
def smoke():
    return load_synth("smoke", spec.SYNTH_SEED)


@pytest.mark.parametrize("forecaster", spec.FORECASTERS)
def test_no_leakage_with_real_forecasters(smoke, forecaster) -> None:
    """Scrambling every value after the last origin leaves every forecast unchanged."""
    y_all, hierarchy = smoke
    kwargs = dict(h=spec.H, n_origins=2, step=spec.ORIGIN_STEP)
    clean, diag = run_backtest(y_all, hierarchy, [forecaster], spec.RECONCILERS, **kwargs)
    last_origin = pd.Timestamp(diag["origins"][-1])

    rng = np.random.default_rng(0)
    after = y_all[DS] > last_origin
    scrambled = y_all.copy()
    scrambled.loc[after, Y] = rng.uniform(0, 1e6, after.sum())
    dirty, _ = run_backtest(scrambled, hierarchy, [forecaster], spec.RECONCILERS, **kwargs)

    pd.testing.assert_frame_equal(clean, dirty)


def test_every_reconciler_is_coherent_on_real_forecasts(smoke) -> None:
    y_all, hierarchy = smoke
    _, diag = run_backtest(
        y_all, hierarchy, ["ets"], spec.RECONCILERS, h=spec.H, n_origins=1, step=spec.ORIGIN_STEP
    )
    for rname in spec.RECONCILERS[1:]:
        assert diag["coherence_max_error"]["ets"][rname] <= 1e-8
    assert diag["ets_fallback_series"]["ets"] == 0
