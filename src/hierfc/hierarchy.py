"""Build a :class:`~hierfc.contracts.Hierarchy` from bottom-series keys; aggregate bottom data."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from hierfc.contracts import DS, ID, SEP, TOTAL, Hierarchy, Y, from_matrix


def _paths(keys: pd.DataFrame, levels: Sequence[str]) -> list[np.ndarray]:
    """Node name of every keys row at each level below the total, as arrays of strings."""
    cols = list(levels[1:])
    if not cols:
        raise ValueError("levels must contain at least one level below 'total'")
    missing = [c for c in cols if c not in keys.columns]
    if missing:
        raise ValueError(f"keys is missing level columns {missing}")
    if keys.empty:
        raise ValueError("keys has no rows")
    labels = keys[cols]
    if labels.isna().any().any():
        raise ValueError("keys has missing labels")
    labels = labels.astype(str)
    for c in cols:
        if labels[c].str.contains(SEP, regex=False).any():
            raise ValueError(f"labels in column {c!r} must not contain {SEP!r}")
    out: list[np.ndarray] = []
    prefix = labels[cols[0]].to_numpy(dtype=object)
    out.append(prefix)
    for c in cols[1:]:
        prefix = np.array(
            [f"{p}{SEP}{x}" for p, x in zip(prefix, labels[c], strict=True)], dtype=object
        )
        out.append(prefix)
    return out


def bottom_ids(keys: pd.DataFrame, levels: Sequence[str]) -> pd.Series:
    """Bottom node name of every row of ``keys`` (same index as ``keys``)."""
    return pd.Series(_paths(keys, _check_levels(levels))[-1], index=keys.index, name=ID)


def _check_levels(levels: Sequence[str]) -> tuple[str, ...]:
    levels = tuple(levels)
    if not levels or levels[0] != TOTAL:
        raise ValueError(f"levels[0] must be {TOTAL!r}, got {levels[:1]}")
    if len(set(levels)) != len(levels):
        raise ValueError("level names must be unique")
    return levels


def build_hierarchy(keys: pd.DataFrame, levels: Sequence[str]) -> Hierarchy:
    """Strictly nested hierarchy from one row of labels per bottom series.

    ``levels[0]`` is ``"total"``; ``keys`` has one column per name in ``levels[1:]`` holding the
    series' label at that level. A node is the path of labels down to its level, so the same label
    under two parents gives two nodes. Duplicate bottom paths raise.
    """
    levels = _check_levels(levels)
    paths = _paths(keys, levels)
    bottom = paths[-1]
    if len(set(bottom)) != len(bottom):
        dup = pd.Series(bottom)[pd.Series(bottom).duplicated()].iloc[0]
        raise ValueError(f"duplicate bottom series path {dup!r}; the hierarchy is not nested")

    names: list[str] = [TOTAL]
    level_of: list[str] = [TOTAL]
    blocks = [np.ones((1, len(bottom)))]
    bottom_nodes = np.array(sorted(set(bottom)), dtype=object)
    bottom_pos = {p: j for j, p in enumerate(bottom_nodes)}
    # Column order of S is the sorted bottom nodes; express every level's membership in it.
    for lv, prefix in zip(levels[1:], paths, strict=True):
        nodes = sorted(set(prefix))
        node_pos = {p: i for i, p in enumerate(nodes)}
        block = np.zeros((len(nodes), len(bottom_nodes)))
        for p_node, p_bottom in zip(prefix, bottom, strict=True):
            block[node_pos[p_node], bottom_pos[p_bottom]] = 1.0
        blocks.append(block)
        names.extend(nodes)
        level_of.extend([lv] * len(nodes))
    return Hierarchy(
        S=np.vstack(blocks).astype(np.float64),
        names=tuple(names),
        level_of=tuple(level_of),
        levels=levels,
    )


def aggregate(bottom: pd.DataFrame, hierarchy: Hierarchy) -> pd.DataFrame:
    """Long frame of the bottom series (``unique_id, ds, y``) to a long frame of ALL nodes."""
    unknown = set(bottom[ID]) - set(hierarchy.bottom_names)
    if unknown:
        raise ValueError(f"frame has non-bottom nodes, e.g. {sorted(unknown)[:3]}")
    wide = bottom.pivot(index=ID, columns=DS, values=Y)
    missing = set(hierarchy.bottom_names) - set(wide.index)
    if missing:
        raise ValueError(
            f"frame is missing {len(missing)} bottom nodes, e.g. {sorted(missing)[:3]}"
        )
    wide = wide.loc[list(hierarchy.bottom_names)]
    if wide.isna().any().any():
        raise ValueError("a bottom series has a missing value or a missing ds")
    full = hierarchy.S @ wide.to_numpy(dtype=float)
    return from_matrix(full, hierarchy, list(wide.columns), Y)
