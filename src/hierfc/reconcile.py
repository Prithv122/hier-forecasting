"""Linear forecast reconciliation, ``y_tilde = S G y_hat``, in plain numpy.

Every method is a matrix ``G`` of shape ``(n_bottom, n_nodes)`` that maps base forecasts of all
nodes to bottom-level forecasts; ``S G`` is then applied to make the result coherent.
"""

from __future__ import annotations

import numpy as np

from hierfc.contracts import Hierarchy

METHODS = ("bu", "td", "ols", "wls_struct", "mint_shrink")


def shrink_cov(residuals: np.ndarray) -> tuple[np.ndarray, float]:
    """Schaefer-Strimmer shrinkage of the sample covariance towards its diagonal.

    ``residuals`` is ``(n_nodes, T)``. Time points (columns) holding any NaN are dropped first;
    fewer than 2 remaining columns is an error. Convention, chosen to match the installed
    ``hierarchicalforecast`` (1.5.3) to about 1e-8: each node's residuals are centred, the sample
    covariance divides by ``T - 1``, and lambda comes from the standardised residuals as
    ``sum_{i!=j} Var(r_ij) / sum_{i!=j} r_ij^2`` with ``Var(r_ij) = T/(T-1)^3 * sum_t (w_ijt -
    mean_t w_ij)^2`` and ``w_ijt = z_it z_jt``. Lambda is clipped to [0, 1] and
    ``W = lambda * diag(S) + (1 - lambda) * S``. A zero-variance node is left unstandardised (its
    correlations are 0). Returns ``(W, lambda)``.
    """
    r = np.asarray(residuals, dtype=float)
    if r.ndim != 2:
        raise ValueError(f"residuals must be 2-D (n_nodes, T), got shape {r.shape}")
    r = r[:, ~np.isnan(r).any(axis=0)]
    p, n = r.shape
    if n < 2:
        raise ValueError(f"need at least 2 residual time points without NaN, got {n}")
    x = r - r.mean(axis=1, keepdims=True)
    cov = x @ x.T / (n - 1)
    sd = np.sqrt(np.diag(cov))
    z = x / np.where(sd > 0, sd, 1.0)[:, None]
    w_bar = z @ z.T / n
    z2 = z**2
    var_r = n / (n - 1) ** 3 * (z2 @ z2.T - n * w_bar**2)
    corr = z @ z.T / (n - 1)
    off = ~np.eye(p, dtype=bool)
    num, den = var_r[off].sum(), (corr[off] ** 2).sum()
    lam = 0.0 if p == 1 else (1.0 if den <= 0 else float(np.clip(num / den, 0.0, 1.0)))
    w = (1.0 - lam) * cov
    w[np.diag_indices(p)] = np.diag(cov)
    return w, lam


def g_from_w(S: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Generalised least squares ``G = (S' W^-1 S)^-1 S' W^-1`` by solving, never inverting.

    ``W`` is a symmetric positive definite ``(n_nodes, n_nodes)`` matrix, or a 1-D vector holding
    the diagonal of one.
    """
    S = np.asarray(S, dtype=float)
    W = np.asarray(W, dtype=float)
    n = S.shape[0]
    try:
        if W.ndim == 1:
            if W.shape != (n,) or np.any(W <= 0):
                raise ValueError("a diagonal W must have n_nodes positive entries")
            w_inv_s = S / W[:, None]
        elif W.shape == (n, n):
            w_inv_s = np.linalg.solve((W + W.T) / 2.0, S)
        else:
            raise ValueError(f"W has shape {W.shape}, expected ({n}, {n}) or ({n},)")
        return np.linalg.solve(S.T @ w_inv_s, w_inv_s.T)
    except np.linalg.LinAlgError as err:
        raise ValueError("W is singular or ill-conditioned; cannot reconcile") from err


def _top_down_g(hierarchy: Hierarchy, history: np.ndarray) -> np.ndarray:
    """Average historical proportions: ``p_j = mean_t y_bottom_j,t / y_total,t``."""
    hist = np.asarray(history, dtype=float)
    if hist.ndim != 2 or hist.shape[0] != hierarchy.n_nodes:
        raise ValueError(f"history must be (n_nodes={hierarchy.n_nodes}, T), got {hist.shape}")
    bottom, total = hist[-hierarchy.n_bottom :], hist[0]
    use = np.isfinite(hist).all(axis=0) & (total != 0)
    if not use.any():
        raise ValueError("history has no time point with a non-zero total")
    if (bottom[:, use] < 0).any() or (total[use] < 0).any():
        raise ValueError("top-down proportions need non-negative history")
    prop = (bottom[:, use] / total[use]).mean(axis=1)
    g = np.zeros((hierarchy.n_bottom, hierarchy.n_nodes))
    g[:, 0] = prop / prop.sum()  # sums to 1 up to floating error
    return g


def g_matrix(
    hierarchy: Hierarchy,
    method: str,
    *,
    residuals: np.ndarray | None = None,
    history: np.ndarray | None = None,
) -> np.ndarray:
    """The ``(n_bottom, n_nodes)`` matrix ``G`` of a reconciliation method.

    ``history`` ``(n_nodes, T)`` is required by ``td``; ``residuals`` ``(n_nodes, T)`` by
    ``mint_shrink``. Both are in hierarchy order.
    """
    S, nb, n = hierarchy.S, hierarchy.n_bottom, hierarchy.n_nodes
    if method == "bu":
        g = np.zeros((nb, n))
        g[:, n - nb :] = np.eye(nb)
        return g
    if method == "td":
        if history is None:
            raise ValueError("method 'td' needs history=(n_nodes, T)")
        return _top_down_g(hierarchy, history)
    if method == "ols":
        return g_from_w(S, np.ones(n))
    if method == "wls_struct":
        return g_from_w(S, S.sum(axis=1))
    if method == "mint_shrink":
        if residuals is None:
            raise ValueError("method 'mint_shrink' needs residuals=(n_nodes, T)")
        res = np.asarray(residuals, dtype=float)
        if res.ndim != 2 or res.shape[0] != n:
            raise ValueError(f"residuals must be (n_nodes={n}, T), got {res.shape}")
        return g_from_w(S, shrink_cov(res)[0])
    raise ValueError(f"unknown reconciliation method {method!r}; choose from {METHODS}")


def reconcile(
    base: np.ndarray,
    hierarchy: Hierarchy,
    method: str,
    *,
    residuals: np.ndarray | None = None,
    history: np.ndarray | None = None,
) -> np.ndarray:
    """Reconcile base forecasts ``(n_nodes, h)`` into coherent ones, ``S @ G @ base``.

    ``method="base"`` returns a copy of ``base`` unchanged.
    """
    b = np.asarray(base, dtype=float)
    if b.ndim != 2 or b.shape[0] != hierarchy.n_nodes:
        raise ValueError(f"base must be (n_nodes={hierarchy.n_nodes}, h), got {b.shape}")
    if method == "base":
        return b.copy()
    if not np.isfinite(b).all():
        raise ValueError("base forecasts contain NaN or inf")
    g = g_matrix(hierarchy, method, residuals=residuals, history=history)
    return hierarchy.S @ (g @ b)
