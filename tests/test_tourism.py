"""Tourism loader on a synthetic stand-in of the source file (the real data is never committed)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hierfc import tourism
from hierfc.contracts import DS, ID, Y


def stand_in(n_months: int = 24) -> pd.DataFrame:
    """A wide frame shaped like the source: 76 regions x 4 purposes plus state totals."""
    rng = np.random.default_rng(0)
    regions = []
    for s in "ABCDEFG":
        for z in "ABCD":
            for r in "ABC":
                regions.append(s + z + r)
    regions = regions[:76]
    dates = pd.date_range("1998-01-01", periods=n_months, freq="MS").strftime("%Y-%m-%d")
    cols = {f"{r}{p}": rng.uniform(0, 100, n_months) for r in regions for p in tourism.PURPOSES}
    wide = pd.DataFrame(cols, index=dates)
    for s in sorted({r[0] for r in regions}):
        wide[f"{s}All"] = wide[[c for c in cols if c[0] == s]].sum(axis=1)
    wide["TotalAll"] = wide[list(cols)].sum(axis=1)
    return wide


def test_parse_wide_sums_purposes_into_regions() -> None:
    wide = stand_in()
    bottom, keys = tourism.parse_wide(wide)
    assert list(bottom.columns) == [ID, DS, Y]
    assert len(keys) == 76 and list(keys.columns) == ["state", "zone", "region"]
    assert (keys["zone"].str[0] == keys["state"]).all()
    first = bottom[bottom[ID] == "A/AA/AAA"].sort_values(DS)
    expected = wide[[f"AAA{p}" for p in tourism.PURPOSES]].sum(axis=1).to_numpy()
    np.testing.assert_allclose(first[Y].to_numpy(), expected)
    assert bottom[DS].dt.is_month_start.all()
    tourism._check_state_totals(wide, bottom)


def test_state_total_mismatch_is_caught() -> None:
    wide = stand_in()
    bottom, _ = tourism.parse_wide(wide)
    wide["AAll"] += 1.0
    with pytest.raises(ValueError, match="state A"):
        tourism._check_state_totals(wide, bottom)


def test_wrong_number_of_bottom_series_raises() -> None:
    wide = stand_in().drop(columns=["AAAHol"])
    with pytest.raises(ValueError, match="304"):
        tourism.parse_wide(wide)


def test_download_rejects_a_file_with_the_wrong_hash(tmp_path) -> None:
    src = tmp_path / "fake.zip"
    src.write_bytes(b"not the archive")
    with pytest.raises(ValueError, match="sha256 mismatch"):
        tourism.download_tourism(tmp_path / "data", url=src.as_uri())
    assert not (tmp_path / "data" / tourism.ZIP_NAME).exists()


def test_load_without_download_says_what_to_run(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="hier-forecast download"):
        tourism.load_tourism(tmp_path)
