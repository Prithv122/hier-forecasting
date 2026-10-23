"""CLI parsing, the report command on a fixture results directory, and (if the real modules
exist) a smoke run."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from hierfc import cli, spec


def test_generate_args():
    args = cli.build_parser().parse_args(
        ["generate", "--config", "smoke", "--seed", "3", "--out", "x"]
    )
    assert (args.config, args.seed, args.out) == ("smoke", 3, "x")
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["generate", "--config", "nope", "--out", "x"])


def test_download_and_report_defaults():
    p = cli.build_parser()
    assert p.parse_args(["download"]).data_dir == "data"
    assert p.parse_args(["report"]).results == "results"


def test_backtest_args():
    p = cli.build_parser()
    args = p.parse_args(
        [
            "backtest",
            "--dataset",
            "tourism",
            "--forecasters",
            "snaive",
            "ets",
            "--reconcilers",
            "bu",
            "mint_shrink",
        ]
    )
    assert args.forecasters == ["snaive", "ets"]
    assert args.reconcilers == ["bu", "mint_shrink"]
    assert args.out is None and args.smoke is False
    assert p.parse_args(["backtest", "--smoke"]).smoke is True
    with pytest.raises(SystemExit):
        p.parse_args(["backtest", "--dataset", "nope"])
    with pytest.raises(SystemExit):
        p.parse_args(["backtest", "--forecasters", "arima"])


def test_reconcile_args():
    args = cli.build_parser().parse_args(
        [
            "reconcile",
            "--base",
            "b.csv",
            "--keys",
            "k.csv",
            "--levels",
            "total",
            "state",
            "--method",
            "mint_shrink",
            "--residuals",
            "r.csv",
            "--out",
            "o.csv",
        ]
    )
    assert args.levels == ["total", "state"]
    assert args.method == "mint_shrink"
    assert args.history is None
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(
            [
                "reconcile",
                "--base",
                "b",
                "--keys",
                "k",
                "--levels",
                "total",
                "--method",
                "base",
                "--out",
                "o",
            ]
        )


def test_no_command_is_an_error():
    with pytest.raises(SystemExit):
        cli.main([])


def test_backtest_without_dataset_or_smoke_exits():
    with pytest.raises(SystemExit):
        cli.main(["backtest"])


def test_library_versions_and_spec_values():
    versions = cli.library_versions()
    assert set(versions) == set(cli.LIBRARIES)
    assert versions["numpy"] != "not installed"
    values = cli.spec_values(3, ["snaive"], ["base"])
    assert values["N_ORIGINS"] == 3 and values["BOOTSTRAP_SEED"] == spec.BOOTSTRAP_SEED
    commit = cli.git_commit()
    assert commit is None or len(commit) == 40


def test_report_command_on_fixture_results(tmp_path):
    rows = []
    for level in spec.TOURISM_LEVELS:
        for i in range(6):
            for o in range(2):
                for rec, delta in (("base", 0.0), ("mint_shrink", -0.5)):
                    rows.append(
                        (
                            pd.Timestamp("2020-01-01") + pd.DateOffset(months=o),
                            "ets",
                            rec,
                            f"{level}{i}",
                            level,
                            1.0 + 0.1 * i + delta,
                            1.0 + delta,
                        )
                    )
    folder = tmp_path / "tourism"
    folder.mkdir()
    pd.DataFrame(
        rows, columns=["origin", "forecaster", "reconciler", "unique_id", "level", "mase", "rmsse"]
    ).to_csv(folder / "scores.csv", index=False)
    (folder / "meta.json").write_text(json.dumps({"dataset": "tourism"}), encoding="utf-8")
    assert cli.main(["report", "--results", str(tmp_path)]) == 0
    text = (tmp_path / "REPORT.md").read_text(encoding="utf-8")
    assert "**Verdict: Supported**" in text


def test_smoke_backtest_writes_results(tmp_path):
    for module in ("hierarchy", "reconcile", "synth", "base"):
        pytest.importorskip(f"hierfc.{module}")
    code = cli.main(["backtest", "--smoke", "--forecasters", "snaive", "--out", str(tmp_path)])
    assert code == 0
    folder = tmp_path / "synth-smoke"
    scores = pd.read_csv(folder / "scores.csv")
    assert {"mase", "rmsse", "level"} <= set(scores.columns)
    assert (folder / "forecasts.csv.gz").exists()
    meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
    assert meta["data"]["sha256"] and meta["n_origins"] == 3
    assert set(meta["versions"]) == set(cli.LIBRARIES)
    assert cli.main(["report", "--results", str(tmp_path)]) == 0


def test_generate_writes_bottom_and_keys(tmp_path):
    pytest.importorskip("hierfc.synth")
    assert cli.main(["generate", "--config", "smoke", "--out", str(tmp_path)]) == 0
    bottom = pd.read_csv(tmp_path / "bottom.csv")
    keys = pd.read_csv(tmp_path / "keys.csv")
    assert list(bottom.columns) == ["unique_id", "ds", "y"]
    assert list(keys.columns) == list(spec.SYNTH_LEVELS[1:])
    assert len(keys) == bottom["unique_id"].nunique() == 8
