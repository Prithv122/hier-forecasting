from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from hierfc.contracts import Hierarchy
from hierfc.hierarchy import build_hierarchy
from hierfc.reconcile import METHODS, g_from_w, g_matrix, reconcile, shrink_cov

LEVELS = ("total", "state", "zone", "region")
H = 6
T = 60


def make_hierarchy(children: tuple[int, int, int] = (2, 2, 2)) -> Hierarchy:
    """Balanced tree: ``children[k]`` children under every node of level k."""
    a, b, c = children
    rows = [(f"s{i}", f"z{j}", f"r{k}") for i in range(a) for j in range(b) for k in range(c)]
    return build_hierarchy(pd.DataFrame(rows, columns=list(LEVELS[1:])), LEVELS)


def make_inputs(h: Hierarchy, seed: int = 0) -> dict[str, np.ndarray]:
    """Positive coherent history, incoherent base forecasts, and noisy residuals."""
    rng = np.random.default_rng(seed)
    bottom = rng.uniform(5, 50, (h.n_bottom, T))
    history = h.S @ bottom
    base = h.S @ rng.uniform(5, 50, (h.n_bottom, H)) + rng.normal(0, 3, (h.n_nodes, H))
    mix = rng.normal(0, 1, (h.n_nodes, h.n_nodes)) * 0.3 + np.eye(h.n_nodes)
    residuals = mix @ rng.normal(0, 1, (h.n_nodes, T))
    return {"history": history, "base": base, "residuals": residuals}


def kwargs_for(method: str, inp: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    if method == "td":
        return {"history": inp["history"]}
    if method == "mint_shrink":
        return {"residuals": inp["residuals"]}
    return {}


def incoherence(h: Hierarchy, y: np.ndarray) -> float:
    gap = np.abs(y - h.S @ y[-h.n_bottom :]).max()
    return float(gap / max(1.0, np.abs(y).max()))


@pytest.mark.parametrize("method", METHODS)
def test_every_method_is_coherent(method: str) -> None:
    h = make_hierarchy((3, 2, 2))
    inp = make_inputs(h)
    out = reconcile(inp["base"], h, method, **kwargs_for(method, inp))
    assert out.shape == inp["base"].shape
    assert incoherence(h, out) <= 1e-8


def test_base_is_a_copy() -> None:
    h = make_hierarchy()
    base = make_inputs(h)["base"]
    out = reconcile(base, h, "base")
    np.testing.assert_array_equal(out, base)
    out[0, 0] += 1.0
    assert out[0, 0] != base[0, 0]


def test_mint_with_identity_w_equals_ols() -> None:
    h = make_hierarchy((3, 2, 2))
    ols = g_matrix(h, "ols")
    np.testing.assert_allclose(g_from_w(h.S, np.eye(h.n_nodes)), ols, atol=1e-10)
    np.testing.assert_allclose(g_from_w(h.S, np.ones(h.n_nodes)), ols, atol=1e-10)


def test_g_from_w_matches_textbook_formula() -> None:
    h = make_hierarchy()
    rng = np.random.default_rng(1)
    a = rng.normal(size=(h.n_nodes, h.n_nodes))
    w = a @ a.T + h.n_nodes * np.eye(h.n_nodes)
    wi = np.linalg.inv(w)
    expected = np.linalg.inv(h.S.T @ wi @ h.S) @ h.S.T @ wi
    np.testing.assert_allclose(g_from_w(h.S, w), expected, atol=1e-9)
    np.testing.assert_allclose(g_matrix(h, "wls_struct"), g_from_w(h.S, np.diag(h.S.sum(axis=1))))
    # G S = I: the projection leaves coherent forecasts alone.
    np.testing.assert_allclose(g_from_w(h.S, w) @ h.S, np.eye(h.n_bottom), atol=1e-9)


@pytest.mark.parametrize("method", ["ols", "wls_struct", "mint_shrink"])
def test_projection_methods_keep_coherent_input(method: str) -> None:
    h = make_hierarchy((3, 2, 2))
    inp = make_inputs(h)
    coherent = h.S @ np.random.default_rng(5).uniform(5, 50, (h.n_bottom, H))
    out = reconcile(coherent, h, method, **kwargs_for(method, inp))
    np.testing.assert_allclose(out, coherent, rtol=1e-9, atol=1e-9)


def test_bottom_up_returns_base_bottom_unchanged() -> None:
    h = make_hierarchy()
    inp = make_inputs(h)
    out = reconcile(inp["base"], h, "bu")
    np.testing.assert_array_equal(out[-h.n_bottom :], inp["base"][-h.n_bottom :])
    np.testing.assert_allclose(out[0], inp["base"][-h.n_bottom :].sum(axis=0))


def test_top_down_proportions_nonnegative_sum_to_one() -> None:
    h = make_hierarchy((3, 2, 2))
    inp = make_inputs(h)
    g = g_matrix(h, "td", history=inp["history"])
    assert g.shape == (h.n_bottom, h.n_nodes)
    prop = g[:, 0]
    assert (prop >= 0).all()
    assert prop.sum() == pytest.approx(1.0, abs=1e-12)
    assert not g[:, 1:].any()
    out = reconcile(inp["base"], h, "td", history=inp["history"])
    np.testing.assert_allclose(out[0], inp["base"][0])  # total is kept
    np.testing.assert_allclose(out[-h.n_bottom :], np.outer(prop, inp["base"][0]))


def test_top_down_skips_zero_total_and_rejects_negatives() -> None:
    h = make_hierarchy()
    hist = make_inputs(h)["history"].copy()
    clean = g_matrix(h, "td", history=hist[:, 1:])
    hist[:, 0] = 0.0
    np.testing.assert_allclose(g_matrix(h, "td", history=hist), clean)
    neg = make_inputs(h)["history"].copy()
    neg[-1, 3] = -1.0
    with pytest.raises(ValueError, match="non-negative"):
        g_matrix(h, "td", history=neg)


def test_missing_or_misshaped_inputs_raise() -> None:
    h = make_hierarchy()
    inp = make_inputs(h)
    with pytest.raises(ValueError, match="history"):
        reconcile(inp["base"], h, "td")
    with pytest.raises(ValueError, match="residuals"):
        reconcile(inp["base"], h, "mint_shrink")
    with pytest.raises(ValueError, match="residuals"):
        reconcile(inp["base"], h, "mint_shrink", residuals=inp["residuals"][:-1])
    with pytest.raises(ValueError, match="history"):
        reconcile(inp["base"], h, "td", history=inp["history"][:-1])
    with pytest.raises(ValueError, match="base must be"):
        reconcile(inp["base"][:-1], h, "bu")
    with pytest.raises(ValueError, match="unknown"):
        reconcile(inp["base"], h, "nope")
    bad = inp["base"].copy()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        reconcile(bad, h, "ols")


def test_shrink_cov_symmetric_psd_lambda_in_unit_interval() -> None:
    rng = np.random.default_rng(3)
    for p, n in [(5, 40), (12, 20), (30, 15)]:  # includes n < p, where the sample cov is singular
        r = rng.normal(size=(p, n)) + rng.normal(size=(p, 1)) * rng.normal(size=(1, n))
        w, lam = shrink_cov(r)
        assert w.shape == (p, p)
        assert 0.0 <= lam <= 1.0
        np.testing.assert_allclose(w, w.T, atol=1e-12)
        assert np.linalg.eigvalsh(w).min() >= -1e-10
        # Diagonal is the sample variance (shrinkage only touches off-diagonals).
        np.testing.assert_allclose(np.diag(w), np.var(r, axis=1, ddof=1))
        np.testing.assert_allclose(
            w - np.diag(np.diag(w)),
            (1 - lam) * (np.cov(r) - np.diag(np.diag(np.cov(r)))),
            atol=1e-12,
        )


def test_shrink_cov_drops_nan_columns() -> None:
    rng = np.random.default_rng(4)
    r = rng.normal(size=(6, 30))
    dirty = r.copy()
    dirty[2, :5] = np.nan
    dirty[0, 10] = np.nan
    w_dirty, lam_dirty = shrink_cov(dirty)
    keep = np.ones(30, dtype=bool)
    keep[:5] = False
    keep[10] = False
    w_clean, lam_clean = shrink_cov(r[:, keep])
    np.testing.assert_array_equal(w_dirty, w_clean)
    assert lam_dirty == lam_clean
    assert not np.isnan(w_dirty).any()


def test_shrink_cov_needs_two_complete_columns() -> None:
    r = np.full((4, 5), np.nan)
    r[:, 0] = 1.0
    with pytest.raises(ValueError, match="at least 2"):
        shrink_cov(r)
    with pytest.raises(ValueError, match="2-D"):
        shrink_cov(np.ones(5))


def test_shrink_cov_constant_node_does_not_break() -> None:
    rng = np.random.default_rng(6)
    r = rng.normal(size=(4, 30))
    r[1] = 0.0
    w, lam = shrink_cov(r)
    assert np.isfinite(w).all() and 0.0 <= lam <= 1.0
    assert w[1, 1] == 0.0


@settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    children=st.tuples(st.integers(1, 3), st.integers(1, 3), st.integers(1, 3)),
    method=st.sampled_from(METHODS),
    seed=st.integers(0, 10_000),
)
def test_property_coherence_on_random_hierarchies(
    children: tuple[int, int, int], method: str, seed: int
) -> None:
    h = make_hierarchy(children)
    inp = make_inputs(h, seed)
    out = reconcile(inp["base"], h, method, **kwargs_for(method, inp))
    assert incoherence(h, out) <= 1e-8


# --- Cross-check against hierarchicalforecast (test-only dependency) ------------------------------


def _hf_inputs() -> tuple[Hierarchy, dict[str, np.ndarray], dict[str, np.ndarray]]:
    h = make_hierarchy((2, 2, 3))
    inp = make_inputs(h, seed=11)
    tags = {lv: np.array([h.index_of(n) for n in h.nodes_in(lv)]) for lv in h.levels}
    return h, inp, tags


def test_matches_hierarchicalforecast_bu_ols_wls() -> None:
    from hierarchicalforecast.methods import BottomUp, MinTrace

    h, inp, _ = _hf_inputs()
    base = inp["base"]
    ref = BottomUp().fit_predict(S=h.S, y_hat=base)["mean"]
    np.testing.assert_allclose(reconcile(base, h, "bu"), ref, rtol=1e-6, atol=1e-8)
    for method in ("ols", "wls_struct"):
        ref = MinTrace(method=method).fit_predict(S=h.S, y_hat=base)["mean"]
        np.testing.assert_allclose(reconcile(base, h, method), ref, rtol=1e-6, atol=1e-8)


def test_matches_hierarchicalforecast_top_down_average_proportions() -> None:
    from hierarchicalforecast.methods import TopDown

    h, inp, tags = _hf_inputs()
    ref = TopDown(method="average_proportions").fit_predict(
        S=h.S, y_hat=inp["base"], tags=tags, y_insample=inp["history"]
    )["mean"]
    out = reconcile(inp["base"], h, "td", history=inp["history"])
    np.testing.assert_allclose(out, ref, rtol=1e-6, atol=1e-8)


def test_matches_hierarchicalforecast_mint_shrink() -> None:
    from hierarchicalforecast.methods import MinTrace

    h, inp, _ = _hf_inputs()
    # The library takes in-sample actuals and fits; its residual is actual - fitted.
    ref = MinTrace(method="mint_shrink").fit_predict(
        S=h.S,
        y_hat=inp["base"],
        y_insample=inp["residuals"],
        y_hat_insample=np.zeros_like(inp["residuals"]),
    )["mean"]
    out = reconcile(inp["base"], h, "mint_shrink", residuals=inp["residuals"])
    np.testing.assert_allclose(out, ref, rtol=0, atol=1e-8)
