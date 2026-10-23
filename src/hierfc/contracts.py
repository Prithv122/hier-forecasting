"""Shared contracts: the hierarchy object, the forecaster interface and the frame schemas.

Every module builds against this file. Changing a signature here is a deliberate decision,
recorded in NOTES.md, not a drive-by edit.

Frames are long format throughout:

* history / actuals: ``unique_id, ds, y``
* forecasts:         ``unique_id, ds, yhat``
* residuals:         ``unique_id, ds, resid`` (in-sample one-step errors, NaN allowed)

``unique_id`` is a node name from :class:`Hierarchy`. ``ds`` is a month-start timestamp.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd

ID, DS, Y, YHAT, RESID = "unique_id", "ds", "y", "yhat", "resid"
TOTAL = "total"
SEP = "/"


@dataclass(frozen=True)
class Hierarchy:
    """A strictly nested hierarchy and its summing matrix.

    ``names`` lists every node, top level first and bottom level last. Within a level, nodes are
    sorted lexicographically. Node names are paths: ``"total"``, ``"NSW"``, ``"NSW/Sydney"``,
    ``"NSW/Sydney/Inner"``. The bottom nodes are the last ``n_bottom`` entries of ``names`` and
    are the columns of ``S`` in the same order, so ``S[-n_bottom:]`` is the identity.
    """

    S: np.ndarray  # (n_nodes, n_bottom), 0/1 floats
    names: tuple[str, ...]  # length n_nodes
    level_of: tuple[str, ...]  # level name of each node, aligned with names
    levels: tuple[str, ...]  # level names, top to bottom; levels[0] == "total"

    @property
    def n_nodes(self) -> int:
        return self.S.shape[0]

    @property
    def n_bottom(self) -> int:
        return self.S.shape[1]

    @property
    def bottom_names(self) -> tuple[str, ...]:
        return self.names[-self.n_bottom :]

    def nodes_in(self, level: str) -> tuple[str, ...]:
        return tuple(n for n, lv in zip(self.names, self.level_of, strict=True) if lv == level)

    def index_of(self, name: str) -> int:
        return self.names.index(name)


@runtime_checkable
class Forecaster(Protocol):
    """A base forecaster. One instance is fitted per backtest origin.

    ``fit`` receives the history of ALL nodes up to and including the origin (never anything
    after it) and the hierarchy. ``forecast(h)`` returns ``h`` monthly steps after the last
    ``ds`` in the history for every node, in a frame with columns ``unique_id, ds, yhat``.
    ``residuals()`` returns in-sample one-step errors ``y - fitted`` for every node over the
    fitted history (``unique_id, ds, resid``); the MinT reconciler estimates its covariance
    from them. Leading NaNs are allowed (a seasonal naive has none for its first season).

    Implementations must be deterministic for a fixed ``seed``.
    """

    name: str

    def fit(self, y_hist: pd.DataFrame, hierarchy: Hierarchy) -> Forecaster: ...

    def forecast(self, h: int) -> pd.DataFrame: ...

    def residuals(self) -> pd.DataFrame: ...


def to_matrix(frame: pd.DataFrame, hierarchy: Hierarchy, value: str) -> tuple[np.ndarray, list]:
    """Pivot a long frame to ``(n_nodes, n_periods)`` in hierarchy order. Returns (matrix, ds)."""
    wide = frame.pivot(index=ID, columns=DS, values=value)
    missing = set(hierarchy.names) - set(wide.index)
    if missing:
        raise ValueError(f"frame is missing {len(missing)} nodes, e.g. {sorted(missing)[:3]}")
    wide = wide.loc[list(hierarchy.names)]
    return wide.to_numpy(dtype=float), list(wide.columns)


def from_matrix(mat: np.ndarray, hierarchy: Hierarchy, ds: list, value: str) -> pd.DataFrame:
    """Inverse of :func:`to_matrix`: ``(n_nodes, n_periods)`` back to a long frame."""
    if mat.shape != (hierarchy.n_nodes, len(ds)):
        raise ValueError(f"shape {mat.shape} != ({hierarchy.n_nodes}, {len(ds)})")
    return pd.DataFrame(
        {
            ID: np.repeat(np.array(hierarchy.names, dtype=object), len(ds)),
            DS: np.tile(np.array(ds, dtype="datetime64[ns]"), hierarchy.n_nodes),
            value: mat.reshape(-1),
        }
    )
