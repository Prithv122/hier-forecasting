"""Australian domestic tourism hierarchy: download, verify and load.

The source is the monthly visitor-nights hierarchy of Wickramasuriya, Athanasopoulos & Hyndman
(2019), 1998-2016, as redistributed by Nixtla (`TourismLarge` in `datasetsforecast`). Its 304
bottom series are region x travel purpose; this project sums over purpose, so the bottom level is
the 76 regions and the hierarchy is total / state / zone / region.

Series codes encode the geography: ``"AAB"`` is state ``A``, zone ``AA``, region ``AAB``, and a
fourth part names the purpose (``Hol``, ``Vis``, ``Bus``, ``Oth``). Labels are kept as codes.
"""

from __future__ import annotations

import hashlib
import io
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

from hierfc.contracts import DS, ID, SEP, Y

URL = "https://nixtla-public.s3.amazonaws.com/hierarchical-data/datasets.zip"
SHA256 = "6512d9aa80f111ee26480bc6f3f4eb3b5655d4ceecc384933100edc85adf704b"
MEMBER = "TourismLarge/data.csv"
ZIP_NAME = "nixtla-hierarchical-datasets.zip"
PURPOSES = ("Hol", "Vis", "Bus", "Oth")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def download_tourism(data_dir: Path, *, url: str = URL, timeout: float = 60.0) -> Path:
    """Download the archive into ``data_dir`` once and check its pinned sha256."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    target = data_dir / ZIP_NAME
    if target.exists() and _sha256(target) == SHA256:
        return target
    partial = target.with_suffix(".part")
    with urllib.request.urlopen(url, timeout=timeout) as resp, partial.open("wb") as fh:
        while block := resp.read(1 << 20):
            fh.write(block)
    got = _sha256(partial)
    if got != SHA256:
        partial.unlink()
        raise ValueError(f"sha256 mismatch for {url}: got {got}, pinned {SHA256}")
    partial.replace(target)
    return target


def parse_wide(wide: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Turn the wide source frame (dates x series codes) into regional bottom series and keys.

    Only the 304 region x purpose columns are used; aggregates in the file are ignored and
    rebuilt from the bottom, which the loader cross-checks against the file's own state totals.
    """
    bottom_cols = [c for c in wide.columns if len(c) == 6 and c[3:] in PURPOSES]
    if len(bottom_cols) != 304:
        raise ValueError(f"expected 304 region x purpose series, found {len(bottom_cols)}")
    regions = wide[bottom_cols].T.groupby(lambda code: code[:3]).sum().T
    codes = list(regions.columns)
    keys = pd.DataFrame(
        {"state": [c[0] for c in codes], "zone": [c[:2] for c in codes], "region": codes}
    )
    names = [SEP.join(row) for row in keys.itertuples(index=False)]
    long = (
        regions.set_axis(names, axis=1)
        .rename_axis(DS)
        .reset_index()
        .melt(id_vars=DS, var_name=ID, value_name=Y)
    )
    long[DS] = pd.to_datetime(long[DS])
    return long[[ID, DS, Y]].sort_values([ID, DS], ignore_index=True), keys


def load_tourism(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Return (bottom long frame, keys with state/zone/region, sha256 of the archive)."""
    path = Path(data_dir) / ZIP_NAME
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `hier-forecast download` first")
    sha = _sha256(path)
    if sha != SHA256:
        raise ValueError(f"{path} has sha256 {sha}, pinned {SHA256}")
    with zipfile.ZipFile(path) as zf:
        wide = pd.read_csv(io.BytesIO(zf.read(MEMBER)), index_col=0)
    bottom, keys = parse_wide(wide)
    _check_state_totals(wide, bottom)
    return bottom, keys, sha


def _check_state_totals(wide: pd.DataFrame, bottom: pd.DataFrame) -> None:
    """The file's own state totals (``AAll``...) must equal the sum of our regions."""
    rebuilt = bottom.assign(state=bottom[ID].str[0]).pivot_table(
        index=DS, columns="state", values=Y, aggfunc="sum"
    )
    for state in rebuilt.columns:
        source = wide[f"{state}All"].to_numpy()
        if abs(rebuilt[state].to_numpy() - source).max() > 1e-6 * max(1.0, abs(source).max()):
            raise ValueError(f"state {state}: regional sum does not match the file's total")
