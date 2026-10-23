"""``hier-forecast`` command line: generate | download | backtest | reconcile | report."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Sequence
from importlib import metadata
from pathlib import Path
from typing import Any

import pandas as pd

from hierfc import spec
from hierfc.contracts import DS, SEP, Y

LIBRARIES = ("numpy", "pandas", "statsforecast", "lightgbm")
SMOKE_ORIGINS = 3
SMOKE_FORECASTERS = ("snaive", "ets")


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def library_versions() -> dict[str, str]:
    versions = {}
    for lib in LIBRARIES:
        try:
            versions[lib] = metadata.version(lib)
        except metadata.PackageNotFoundError:
            versions[lib] = "not installed"
    return versions


# Untracked paths under these folders are run output and downloaded data, not code changes.
IGNORED_UNTRACKED = ("results/", "results-smoke/", "data/")


def _git_output(*args: str) -> str | None:
    """Stdout of ``git <args>`` run in the project checkout, or None when git is unavailable."""
    try:
        done = subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=Path(__file__).resolve().parent,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout


def git_commit() -> str | None:
    """Current commit of the project checkout, or None when git is unavailable."""
    out = _git_output("rev-parse", "HEAD")
    return (out or "").strip() or None


def git_dirty() -> bool | None:
    """Whether the checkout has uncommitted changes to code; None when git is unavailable.

    Untracked files under ``results/``, ``results-smoke/`` and ``data/`` do not count.
    """
    out = _git_output("status", "--porcelain")
    if out is None:
        return None
    for line in out.splitlines():
        if not line.strip():
            continue
        path = line[3:].strip().replace("\\", "/")
        if line.startswith("??") and path.startswith(IGNORED_UNTRACKED):
            continue
        return True
    return False


def spec_values(n_origins: int, forecasters: Sequence[str], reconcilers: Sequence[str]) -> dict:
    return {
        "H": spec.H,
        "SEASON": spec.SEASON,
        "N_ORIGINS": n_origins,
        "ORIGIN_STEP": spec.ORIGIN_STEP,
        "BOOTSTRAP_B": spec.BOOTSTRAP_B,
        "BOOTSTRAP_SEED": spec.BOOTSTRAP_SEED,
        "ALPHA": spec.ALPHA,
        "SYNTH_SEED": spec.SYNTH_SEED,
        "MODEL_SEED": spec.MODEL_SEED,
        "FORECASTERS": list(forecasters),
        "RECONCILERS": list(reconcilers),
    }


# Commands -------------------------------------------------------------------------------------


def cmd_generate(args: argparse.Namespace) -> int:
    from hierfc.datasets import bottom_frame
    from hierfc.synth import load_synth

    y_all, hierarchy = load_synth(args.config, args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    bottom = bottom_frame(y_all, hierarchy)
    parts = [name.split(SEP) for name in hierarchy.bottom_names]
    keys = pd.DataFrame(parts, columns=list(hierarchy.levels[1:]))
    bottom.to_csv(out / "bottom.csv", index=False)
    keys.to_csv(out / "keys.csv", index=False)
    _log(f"wrote {out / 'bottom.csv'} and {out / 'keys.csv'} ({hierarchy.n_bottom} series)")
    return 0


def cmd_download(args: argparse.Namespace) -> int:
    from hierfc.tourism import download_tourism

    path = download_tourism(Path(args.data_dir))
    _log(f"tourism archive at {path}")
    return 0


def cmd_backtest(args: argparse.Namespace) -> int:
    from hierfc.backtest import run_backtest
    from hierfc.datasets import SMOKE, load_dataset
    from hierfc.metrics import score

    name = SMOKE if args.smoke else args.dataset
    if name is None:
        raise SystemExit("backtest needs --dataset or --smoke")
    # Record the code that runs: the state at the start, not whatever HEAD is when it finishes.
    commit_start = git_commit()
    dirty = git_dirty()
    n_origins = SMOKE_ORIGINS if args.smoke else spec.N_ORIGINS
    forecasters = args.forecasters or (SMOKE_FORECASTERS if args.smoke else spec.FORECASTERS)
    reconcilers = args.reconcilers or spec.RECONCILERS
    out_root = Path(args.out or ("results-smoke" if args.smoke else "results"))

    _log(f"loading {name}")
    y_all, hierarchy, provenance = load_dataset(name, Path(args.data_dir))
    _log(f"{provenance['n_series']} nodes, {provenance['n_periods']} months")
    forecasts, diagnostics = run_backtest(
        y_all,
        hierarchy,
        list(forecasters),
        list(reconcilers),
        n_origins=n_origins,
        progress=_log,
    )
    scores = score(forecasts, y_all, hierarchy)

    out = out_root / name
    out.mkdir(parents=True, exist_ok=True)
    scores.to_csv(out / "scores.csv", index=False)
    forecasts.to_csv(out / "forecasts.csv.gz", index=False)
    commit_end = git_commit()
    if commit_end != commit_start:
        _log(
            f"warning: HEAD changed during the run ({commit_start} -> {commit_end}); "
            "git_commit is the commit the run started from"
        )
    meta: dict[str, Any] = {
        "dataset": name,
        "data": provenance,
        "h": spec.H,
        "n_origins": n_origins,
        "forecasters": list(forecasters),
        "reconcilers": list(reconcilers),
        "excluded": {
            "total": scores.attrs.get("excluded", 0),
            "by_level": scores.attrs.get("excluded_by_level", {}),
        },
        "diagnostics": diagnostics,
        "versions": library_versions(),
        "git_commit": commit_start,
        "git_dirty": dirty,
        "spec": spec_values(n_origins, forecasters, reconcilers),
    }
    if commit_end != commit_start:
        meta["git_commit_end"] = commit_end
    (out / "meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    _log(f"wrote scores.csv, forecasts.csv.gz, meta.json to {out}")
    return 0


def cmd_reconcile(args: argparse.Namespace) -> int:
    from hierfc.contracts import from_matrix, to_matrix
    from hierfc.hierarchy import build_hierarchy
    from hierfc.reconcile import reconcile

    hierarchy = build_hierarchy(pd.read_csv(args.keys), tuple(args.levels))
    base, ds = to_matrix(pd.read_csv(args.base, parse_dates=[DS]), hierarchy, "yhat")
    residuals = history = None
    if args.residuals:
        residuals, _ = to_matrix(pd.read_csv(args.residuals, parse_dates=[DS]), hierarchy, "resid")
    if args.history:
        history, _ = to_matrix(pd.read_csv(args.history, parse_dates=[DS]), hierarchy, Y)
    rec = reconcile(base, hierarchy, args.method, residuals=residuals, history=history)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    from_matrix(rec, hierarchy, ds, "yhat").to_csv(out, index=False)
    _log(f"wrote {out}")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    from hierfc.report import write_report

    path = write_report(Path(args.results))
    _log(f"wrote {path}")
    return 0


# Parser ---------------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    from hierfc.datasets import ALL_NAMES

    parser = argparse.ArgumentParser(
        prog="hier-forecast", description="Hierarchical forecasting and reconciliation backtest."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="write a synthetic hierarchy as bottom.csv + keys.csv")
    gen.add_argument("--config", choices=sorted(spec.SYNTH_CONFIGS), required=True)
    gen.add_argument("--seed", type=int, default=spec.SYNTH_SEED)
    gen.add_argument("--out", required=True, help="output directory")
    gen.set_defaults(func=cmd_generate)

    dl = sub.add_parser("download", help="download the tourism data (never committed)")
    dl.add_argument("--data-dir", default="data")
    dl.set_defaults(func=cmd_download)

    bt = sub.add_parser("backtest", help="run the rolling-origin backtest on one dataset")
    bt.add_argument("--dataset", choices=ALL_NAMES)
    bt.add_argument("--data-dir", default="data")
    bt.add_argument("--out", help="output root (default: results; results-smoke with --smoke)")
    bt.add_argument("--forecasters", nargs="+", choices=spec.FORECASTERS)
    bt.add_argument("--reconcilers", nargs="+", choices=spec.RECONCILERS)
    bt.add_argument(
        "--smoke",
        action="store_true",
        help=f"synth-smoke, {SMOKE_ORIGINS} origins, snaive + ets, all reconcilers",
    )
    bt.set_defaults(func=cmd_backtest)

    rc = sub.add_parser("reconcile", help="reconcile one base forecast file with one method")
    rc.add_argument("--base", required=True, help="CSV: unique_id, ds, yhat for every node")
    rc.add_argument("--keys", required=True, help="CSV of node keys, one column per level")
    rc.add_argument("--levels", nargs="+", required=True, help="level names, top to bottom")
    rc.add_argument("--method", required=True, choices=[r for r in spec.RECONCILERS if r != "base"])
    rc.add_argument("--residuals", help="CSV: unique_id, ds, resid (mint_shrink)")
    rc.add_argument("--history", help="CSV: unique_id, ds, y (td)")
    rc.add_argument("--out", required=True)
    rc.set_defaults(func=cmd_reconcile)

    rp = sub.add_parser("report", help="write results/REPORT.md with the H1-H3 verdicts")
    rp.add_argument("--results", default="results")
    rp.set_defaults(func=cmd_report)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
