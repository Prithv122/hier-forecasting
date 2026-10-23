"""Dataset loading: one entry point for the three arms (plus a small synthetic smoke set)."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pandas as pd

from hierfc import spec
from hierfc.contracts import DS, ID, Hierarchy

SMOKE = "synth-smoke"
SYNTH_CONFIG_OF = {
    "synth-correlated": "correlated",
    "synth-independent": "independent",
    SMOKE: "smoke",
}
ALL_NAMES = (*spec.DATASETS, SMOKE)


def bottom_frame(y_all: pd.DataFrame, hierarchy: Hierarchy) -> pd.DataFrame:
    """The bottom-level rows of ``y_all``, sorted by series then date."""
    bottom = y_all[y_all[ID].isin(set(hierarchy.bottom_names))]
    return bottom.sort_values([ID, DS], ignore_index=True)


def _frame_sha256(frame: pd.DataFrame) -> str:
    return hashlib.sha256(frame.to_csv(index=False).encode()).hexdigest()


def load_dataset(name: str, data_dir: Path) -> tuple[pd.DataFrame, Hierarchy, dict[str, Any]]:
    """Return ``(y_all, hierarchy, provenance)`` for ``name``.

    ``y_all`` is the long ``unique_id, ds, y`` frame of every node. Provenance records the dataset,
    the sha256 of its input (the source archive for tourism, the CSV bytes of the generated bottom
    frame for synthetic data, which is deterministic for a seed) and its size.
    """
    if name in SYNTH_CONFIG_OF:
        from hierfc.synth import load_synth

        config = SYNTH_CONFIG_OF[name]
        y_all, hierarchy = load_synth(config, spec.SYNTH_SEED)
        sha = _frame_sha256(bottom_frame(y_all, hierarchy))
        extra: dict[str, Any] = {"synthetic": True, "config": config, "seed": spec.SYNTH_SEED}
    elif name == "tourism":
        from hierfc.hierarchy import aggregate, build_hierarchy
        from hierfc.tourism import load_tourism

        bottom, keys, sha = load_tourism(Path(data_dir))
        hierarchy = build_hierarchy(keys, spec.TOURISM_LEVELS)
        y_all = aggregate(bottom, hierarchy)
        extra = {"synthetic": False}
    else:
        raise ValueError(f"unknown dataset {name!r}; choose from {', '.join(ALL_NAMES)}")

    y_all = y_all.copy()
    y_all[DS] = pd.to_datetime(y_all[DS])
    provenance = {
        "dataset": name,
        "sha256": sha,
        "n_series": hierarchy.n_nodes,
        "n_bottom": hierarchy.n_bottom,
        "n_periods": int(y_all[DS].nunique()),
        "first_ds": f"{y_all[DS].min():%Y-%m-%d}",
        "last_ds": f"{y_all[DS].max():%Y-%m-%d}",
        "levels": list(hierarchy.levels),
        **extra,
    }
    return y_all, hierarchy, provenance
