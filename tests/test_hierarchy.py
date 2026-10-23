from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hierfc.contracts import DS, ID, SEP, TOTAL, Y
from hierfc.hierarchy import aggregate, bottom_ids, build_hierarchy

LEVELS = ("total", "state", "zone", "region")


def make_keys() -> pd.DataFrame:
    rows = [
        ("VIC", "VICa", "Melbourne"),
        ("NSW", "NSWa", "Sydney"),
        ("NSW", "NSWa", "Newcastle"),
        ("NSW", "NSWb", "Albury"),
        ("VIC", "VICb", "Ballarat"),
        ("VIC", "VICb", "Geelong"),
    ]
    return pd.DataFrame(rows, columns=list(LEVELS[1:]))


def test_names_order_and_level_of() -> None:
    h = build_hierarchy(make_keys(), LEVELS)
    assert h.names == (
        "total",
        "NSW",
        "VIC",
        "NSW/NSWa",
        "NSW/NSWb",
        "VIC/VICa",
        "VIC/VICb",
        "NSW/NSWa/Newcastle",
        "NSW/NSWa/Sydney",
        "NSW/NSWb/Albury",
        "VIC/VICa/Melbourne",
        "VIC/VICb/Ballarat",
        "VIC/VICb/Geelong",
    )
    assert h.levels == LEVELS
    assert h.level_of[0] == TOTAL
    assert h.nodes_in("state") == ("NSW", "VIC")
    assert len(h.nodes_in("region")) == 6
    assert all(SEP in n for n in h.nodes_in("zone"))


def test_summing_matrix_shape_dtype_and_identity_block() -> None:
    h = build_hierarchy(make_keys(), LEVELS)
    assert h.S.shape == (13, 6)
    assert h.S.dtype == np.float64
    assert h.n_bottom == 6 and h.n_nodes == 13
    np.testing.assert_array_equal(h.S[-h.n_bottom :], np.eye(6))
    np.testing.assert_array_equal(h.S[0], np.ones(6))
    assert h.bottom_names == h.names[-6:]


def test_rows_sum_children_exactly() -> None:
    h = build_hierarchy(make_keys(), LEVELS)
    nsw = h.S[h.index_of("NSW")]
    members = {h.bottom_names[j] for j in np.flatnonzero(nsw)}
    assert members == {n for n in h.bottom_names if n.startswith("NSW/")}
    # Every level partitions the bottom series.
    for lv in LEVELS:
        rows = [h.index_of(n) for n in h.nodes_in(lv)]
        np.testing.assert_array_equal(h.S[rows].sum(axis=0), np.ones(6))


def test_duplicate_bottom_path_raises() -> None:
    keys = pd.concat([make_keys(), make_keys().iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        build_hierarchy(keys, LEVELS)


def test_same_label_under_different_parents_is_two_nodes() -> None:
    keys = pd.DataFrame(
        {"state": ["A", "A", "B", "B"], "zone": ["x", "y", "x", "y"], "region": ["r"] * 4}
    )
    h = build_hierarchy(keys, LEVELS)
    assert h.n_bottom == 4
    assert "A/x/r" in h.names and "B/x/r" in h.names


def test_bad_levels_and_keys_raise() -> None:
    keys = make_keys()
    with pytest.raises(ValueError, match="total"):
        build_hierarchy(keys, ("state", "zone", "region"))
    with pytest.raises(ValueError, match="missing level"):
        build_hierarchy(keys.drop(columns="zone"), LEVELS)
    bad = keys.copy()
    bad.loc[0, "zone"] = "a/b"
    with pytest.raises(ValueError, match="must not contain"):
        build_hierarchy(bad, LEVELS)
    bad = keys.copy()
    bad.loc[0, "zone"] = None
    with pytest.raises(ValueError, match="missing labels"):
        build_hierarchy(bad, LEVELS)


def test_bottom_ids_follow_keys_rows() -> None:
    keys = make_keys()
    ids = bottom_ids(keys, LEVELS)
    assert list(ids) == [
        "VIC/VICa/Melbourne",
        "NSW/NSWa/Sydney",
        "NSW/NSWa/Newcastle",
        "NSW/NSWb/Albury",
        "VIC/VICb/Ballarat",
        "VIC/VICb/Geelong",
    ]
    assert ids.index.equals(keys.index)
    assert set(ids) == set(build_hierarchy(keys, LEVELS).bottom_names)


def _bottom_frame(h, n_t: int = 5, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ds = pd.date_range("2020-01-01", periods=n_t, freq="MS")
    return pd.DataFrame(
        {
            ID: np.repeat(np.array(h.bottom_names, dtype=object), n_t),
            DS: np.tile(ds.to_numpy(dtype="datetime64[ns]"), h.n_bottom),
            Y: rng.uniform(1, 10, h.n_bottom * n_t),
        }
    )


def test_aggregate_sums_correctly() -> None:
    h = build_hierarchy(make_keys(), LEVELS)
    bottom = _bottom_frame(h)
    full = aggregate(bottom, h)
    assert len(full) == h.n_nodes * 5
    assert set(full[ID]) == set(h.names)
    wide = full.pivot(index=ID, columns=DS, values=Y)
    b = bottom.pivot(index=ID, columns=DS, values=Y)
    np.testing.assert_allclose(wide.loc[TOTAL], b.sum(axis=0))
    nsw = [n for n in h.bottom_names if n.startswith("NSW/")]
    np.testing.assert_allclose(wide.loc["NSW"], b.loc[nsw].sum(axis=0))
    nsw_a = [n for n in h.bottom_names if n.startswith("NSW/NSWa/")]
    np.testing.assert_allclose(wide.loc["NSW/NSWa"], b.loc[nsw_a].sum(axis=0))
    np.testing.assert_allclose(wide.loc[list(h.bottom_names)], b.loc[list(h.bottom_names)])


def test_aggregate_rejects_missing_inputs() -> None:
    h = build_hierarchy(make_keys(), LEVELS)
    bottom = _bottom_frame(h)
    with pytest.raises(ValueError, match="missing"):
        aggregate(bottom[bottom[ID] != h.bottom_names[0]], h)
    with pytest.raises(ValueError, match="missing value or a missing ds"):
        aggregate(bottom.drop(index=3), h)
    extra = bottom.copy()
    extra.loc[0, ID] = "NSW"
    with pytest.raises(ValueError, match="non-bottom"):
        aggregate(extra, h)
