"""Mechanical H1-H3 verdicts on synthetic score frames, and the report file itself."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from hierfc import spec
from hierfc.report import h1_verdict, h2_verdict, h3_verdict, write_report

N_SERIES = {
    "total": 1,
    "state": 3,
    "zone": 6,
    "region": 12,
    "region_": 5,
    "store": 8,
    "category": 10,
}


def frame(levels, deltas: dict[tuple[str, str], dict[str, float]], n_origins: int = 3):
    """Scores where (f, base) has MASE 1 + 0.1 i and every other pair adds deltas[pair][level].

    Constant differences make each interval collapse onto the estimate, so verdicts are exact.
    A level mapped to the string key "alt" would alternate sign; see ``alternating``.
    """
    rows = []
    for level in levels:
        for i in range(N_SERIES.get(level, 6)):
            for o in range(n_origins):
                origin = pd.Timestamp("2020-01-01") + pd.DateOffset(months=o)
                base = 1.0 + 0.1 * i
                for (f, r), by_level in deltas.items():
                    delta = by_level.get(level, 0.0)
                    if isinstance(delta, tuple):  # alternating +-x: mean 0 with spread
                        delta = delta[0] if (i + o) % 2 == 0 else -delta[0]
                    rows.append((origin, f, r, f"{level}{i}", level, base + delta, base + delta))
    return pd.DataFrame(
        rows, columns=["origin", "forecaster", "reconciler", "unique_id", "level", "mase", "rmsse"]
    )


def tourism_scores(delta_by_level):
    return frame(
        spec.TOURISM_LEVELS,
        {("ets", "base"): {}, ("ets", "mint_shrink"): delta_by_level},
    )


LEVELS = list(spec.TOURISM_LEVELS)


def test_h1_supported():
    label, res = h1_verdict(
        tourism_scores({"total": -0.5, "state": -0.5, "zone": -0.5, "region": 0.0}), LEVELS
    )
    assert label == "Supported"
    assert [r["verdict"] for r in res.values()] == [
        "better",
        "better",
        "better",
        "no clear difference",
    ]


def test_h1_helps_the_top_hurts_the_bottom():
    label, res = h1_verdict(
        tourism_scores({"total": -0.5, "state": 0.0, "zone": 0.0, "region": 0.5}), LEVELS
    )
    assert label == "Helps the top, hurts the bottom"
    assert res["total"]["unit"] == "series-origin"
    assert res["region"]["unit"] == "series"


def test_h1_three_better_but_one_worse_is_not_supported():
    label, _ = h1_verdict(
        tourism_scores({"total": -0.5, "state": -0.5, "zone": -0.5, "region": 0.5}), LEVELS
    )
    assert label == "Helps the top, hurts the bottom"


def test_h1_mixed():
    label, _ = h1_verdict(
        tourism_scores({"total": 0.0, "state": -0.5, "zone": 0.0, "region": 0.0}), LEVELS
    )
    assert label == "Mixed"
    label, _ = h1_verdict(
        tourism_scores({"total": -0.5, "state": 0.5, "zone": 0.0, "region": 0.0}), LEVELS
    )
    assert label == "Mixed"


def test_h1_and_h2_not_run_without_the_needed_forecasts():
    only_base = frame(spec.TOURISM_LEVELS, {("ets", "base"): {}})
    assert h1_verdict(only_base, LEVELS)[0] == "not run"
    assert h2_verdict(only_base, LEVELS)[0] == "not run"


@pytest.mark.parametrize(("delta", "label"), [(-0.5, "Supported"), (0.5, "Refuted"), (0.0, "Tie")])
def test_h2(delta, label):
    scores = frame(
        spec.TOURISM_LEVELS,
        {("ets", "base"): {}, ("hybrid", "base"): {"region": delta, "total": -9.0}},
    )
    got, res = h2_verdict(scores, LEVELS)
    assert got == label
    assert list(res) == ["region"]  # bottom level only


def synth(delta):
    return frame(
        spec.SYNTH_LEVELS,
        {("ets", "base"): {}, ("ets", "mint_shrink"): {"category": delta}},
    )


@pytest.mark.parametrize(
    ("d_corr", "d_ind", "label"),
    [(-0.5, -0.1, "Supported"), (-0.1, -0.5, "Refuted"), (-0.3, -0.3, "No clear difference")],
)
def test_h3(d_corr, d_ind, label):
    got, res = h3_verdict(synth(d_corr), synth(d_ind), "category")
    assert got == label
    assert res["mean"] == pytest.approx(d_corr - d_ind)


def test_h3_not_run_when_a_synthetic_arm_is_missing():
    assert h3_verdict(synth(-0.5), None, "category") == ("not run", None)
    assert h3_verdict(None, None, "category") == ("not run", None)


def write_dataset(root, name, scores, extra_meta=None):
    folder = root / name
    folder.mkdir(parents=True)
    scores.to_csv(folder / "scores.csv", index=False)
    meta = {
        "dataset": name,
        "data": {
            "sha256": "abc123",
            "n_series": 111,
            "n_bottom": 76,
            "n_periods": 228,
            "first_ds": "1998-01-01",
            "last_ds": "2016-12-01",
        },
        "diagnostics": {
            "origins": ["2015-01-01", "2015-04-01"],
            "negative_forecasts": {"ets": {"base": 0, "mint_shrink": 7}},
        },
        "excluded": {"total": 4, "by_level": {}},
        "versions": {"numpy": "2.5.4"},
        "git_commit": "deadbeef",
    }
    meta.update(extra_meta or {})
    (folder / "meta.json").write_text(json.dumps(meta), encoding="utf-8")


def test_write_report_with_only_tourism(tmp_path):
    scores = frame(
        spec.TOURISM_LEVELS,
        {
            ("ets", "base"): {},
            ("ets", "mint_shrink"): {"total": -0.5, "state": -0.5, "zone": -0.5},
            ("hybrid", "base"): {"region": -0.5},
            ("ets", "bu"): {"region": 0.5},
        },
    )
    write_dataset(tmp_path, "tourism", scores)
    path = write_report(tmp_path)
    text = path.read_text(encoding="utf-8")
    assert path == tmp_path / "REPORT.md"
    assert "### H1" in text and "**Verdict: Supported**" in text
    assert "H2" in text and "**Verdict: Supported**" in text
    h3 = text.split("### H3")[1].split("## Provenance")[0]
    assert "not run" in h3
    assert "abc123" in text and "deadbeef" in text and "numpy 2.5.4" in text
    assert "### tourism: MASE" in text and "### tourism: RMSSE" in text
    assert "synthetic" not in text.split("## Provenance")[1].lower().replace(
        "synthetic datasets are labelled synthetic", ""
    )
    assert "bu" in text.split("## Descriptive")[1]
    assert text == write_report(tmp_path).read_text(encoding="utf-8")  # deterministic


def test_write_report_with_all_datasets_labels_synthetic(tmp_path):
    write_dataset(tmp_path, "tourism", tourism_scores({"total": -0.5, "state": 0.0}))
    write_dataset(tmp_path, "synth-correlated", synth(-0.5))
    write_dataset(tmp_path, "synth-independent", synth(-0.1))
    text = write_report(tmp_path).read_text(encoding="utf-8")
    assert "### synth-correlated (synthetic): MASE" in text
    assert "### synth-independent (synthetic)" in text
    h3 = text.split("### H3")[1].split("## Provenance")[0]
    assert "**Verdict: Supported**" in h3


def test_write_report_with_no_results(tmp_path):
    text = write_report(tmp_path).read_text(encoding="utf-8")
    assert text.count("**Verdict: not run**") == 3
