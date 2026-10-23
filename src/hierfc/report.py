"""Write ``results/REPORT.md``: provenance, per-level tables and the mechanical H1-H3 verdicts.

Every verdict is computed by the exact rules of PREREGISTRATION.md section 6 from the score files
under ``results/<dataset>/``. The output has no timestamps beyond what ``meta.json`` records, so the
same inputs always give the same file.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from hierfc import spec
from hierfc.metrics import bootstrap_gain_difference, level_table, paired_bootstrap

ORDER = ("tourism", "synth-correlated", "synth-independent")
NOT_RUN = "not run"
CANDIDATE = (spec.PRIMARY_BASE, spec.PRIMARY_RECONCILER)
REFERENCE = (spec.PRIMARY_BASE, "base")
DAGGER = "†"

Result = dict[str, Any]


def dataset_levels(name: str, scores: pd.DataFrame) -> list[str]:
    """Levels of a dataset, top to bottom (from the spec, else order of first appearance)."""
    expected = spec.TOURISM_LEVELS if name == "tourism" else spec.SYNTH_LEVELS
    present = set(scores["level"])
    if present <= set(expected):
        return [lv for lv in expected if lv in present]
    return list(pd.unique(scores["level"]))


def _has(scores: pd.DataFrame, *keys: tuple[str, str]) -> bool:
    pairs = set(zip(scores["forecaster"], scores["reconciler"], strict=True))
    return all(k in pairs for k in keys)


def h1_verdict(scores: pd.DataFrame, levels: list[str]) -> tuple[str, dict[str, Result]]:
    """H1: MinT-shrink vs unreconciled ETS on tourism, per level."""
    if not _has(scores, CANDIDATE, REFERENCE):
        return NOT_RUN, {}
    results = {
        lv: paired_bootstrap(scores, candidate=CANDIDATE, reference=REFERENCE, level=lv)
        for lv in levels
    }
    verdicts = [r["verdict"] for r in results.values()]
    n_better, n_worse = verdicts.count("better"), verdicts.count("worse")
    if n_better >= 3 and n_worse == 0:
        label = "Supported"
    elif results[levels[0]]["verdict"] == "better" and results[levels[-1]]["verdict"] == "worse":
        label = "Helps the top, hurts the bottom"
    else:
        label = "Mixed"
    return label, results


def h2_verdict(scores: pd.DataFrame, levels: list[str]) -> tuple[str, dict[str, Result]]:
    """H2: hybrid vs ETS at the bottom level of tourism, unreconciled forecasts."""
    candidate, reference = ("hybrid", "base"), ("ets", "base")
    if not _has(scores, candidate, reference):
        return NOT_RUN, {}
    result = paired_bootstrap(scores, candidate=candidate, reference=reference, level=levels[-1])
    label = {"better": "Supported", "worse": "Refuted"}.get(result["verdict"], "Tie")
    return label, {levels[-1]: result}


def h3_verdict(
    scores_corr: pd.DataFrame | None, scores_ind: pd.DataFrame | None, level: str
) -> tuple[str, Result | None]:
    """H3: gain(correlated) - gain(independent) of MinT-shrink over ETS at the bottom level."""
    if scores_corr is None or scores_ind is None:
        return NOT_RUN, None
    if not (_has(scores_corr, CANDIDATE, REFERENCE) and _has(scores_ind, CANDIDATE, REFERENCE)):
        return NOT_RUN, None
    result = bootstrap_gain_difference(
        scores_corr, scores_ind, candidate=CANDIDATE, reference=REFERENCE, level=level
    )
    label = {"better": "Supported", "worse": "Refuted"}.get(
        result["verdict"], "No clear difference"
    )
    return label, result


# Markdown helpers -----------------------------------------------------------------------------


def _table(headers: list[str], rows: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return [*lines, ""]


def _num(x: float) -> str:
    return f"{x:+.3f}"


def _estimate(r: Result) -> str:
    mark = DAGGER if r["unit"] == "series-origin" else ""
    return f"{_num(r['mean'])} [{_num(r['lo'])}, {_num(r['hi'])}]{mark} {r['verdict']}"


def _label(name: str) -> str:
    return f"{name} (synthetic)" if name.startswith("synth") else name


def _level_rows(results: dict[str, Result]) -> list[list[str]]:
    return [
        [
            lv,
            _num(r["mean"]),
            f"[{_num(r['lo'])}, {_num(r['hi'])}]",
            r["unit"] + (f" ({r['n_series']} series)"),
            r["verdict"],
        ]
        for lv, r in results.items()
    ]


LEVEL_HEADERS = ["level", "mean d", "95% interval", "resampling unit", "verdict"]
UNIT_NOTE = (
    f"{DAGGER} resampled over (series, origin) pairs because the level has fewer than "
    f"{spec.MIN_SERIES_FOR_SERIES_BOOTSTRAP} series; the interval ignores the overlap between "
    "origins."
)


def _verdict_sections(data: dict[str, tuple[pd.DataFrame, dict]]) -> list[str]:
    out = ["## Hypotheses (confirmatory)", ""]
    out += [
        "Paired difference d = metric(candidate) - metric(reference) on MASE; negative favours "
        f"the candidate. {spec.BOOTSTRAP_B} bootstrap draws, seed {spec.BOOTSTRAP_SEED}, "
        f"{int((1 - spec.ALPHA) * 100)}% percentile interval.",
        "",
    ]

    out += ["### H1 - reconciliation helps at every level (tourism, ETS, MinT-shrink vs base)", ""]
    if "tourism" in data:
        scores = data["tourism"][0]
        levels = dataset_levels("tourism", scores)
        label, results = h1_verdict(scores, levels)
        out += [f"**Verdict: {label}**", ""]
        if results:
            out += [*_table(LEVEL_HEADERS, _level_rows(results)), UNIT_NOTE, ""]
    else:
        out += [f"**Verdict: {NOT_RUN}** (no tourism results)", ""]

    out += ["### H2 - the hybrid beats pure ETS at the bottom (tourism, unreconciled)", ""]
    if "tourism" in data:
        scores = data["tourism"][0]
        levels = dataset_levels("tourism", scores)
        label, results = h2_verdict(scores, levels)
        out += [f"**Verdict: {label}**", ""]
        if results:
            out += _table(LEVEL_HEADERS, _level_rows(results))
    else:
        out += [f"**Verdict: {NOT_RUN}** (no tourism results)", ""]

    out += [
        "### H3 - MinT helps more when errors are correlated (bottom level, ETS, synthetic)",
        "",
    ]
    corr = data.get("synth-correlated", (None, {}))[0]
    ind = data.get("synth-independent", (None, {}))[0]
    level = dataset_levels("synth-correlated", corr)[-1] if corr is not None else ""
    label, r = h3_verdict(corr, ind, level)
    out += [f"**Verdict: {label}**", ""]
    if r is None:
        out += [f"**{NOT_RUN}**: needs both synthetic datasets.", ""]
    else:
        out += _table(
            ["quantity", "value"],
            [
                [f"gain, synth-correlated ({level})", _num(r["mean_a"])],
                [f"gain, synth-independent ({level})", _num(r["mean_b"])],
                ["difference (correlated - independent)", _num(r["mean"])],
                ["95% interval", f"[{_num(r['lo'])}, {_num(r['hi'])}]"],
                ["resampling unit", f"{r['unit']} (series: {r['n_series']})"],
            ],
        )
    return out


def _provenance_section(name: str, scores: pd.DataFrame, meta: dict) -> list[str]:
    data = meta.get("data", {})
    origins = meta.get("diagnostics", {}).get("origins") or meta.get("origins") or []
    excluded = meta.get("excluded", {})
    out = [f"### {_label(name)}", ""]
    facts = [
        ("data sha256", data.get("sha256", "n/a")),
        (
            "series",
            f"{data.get('n_series', 'n/a')} nodes, {data.get('n_bottom', 'n/a')} bottom, "
            f"{data.get('n_periods', 'n/a')} months ({data.get('first_ds', '?')} to "
            f"{data.get('last_ds', '?')})",
        ),
        (
            "origins",
            f"{len(origins)} ({origins[0]} to {origins[-1]}), h = {meta.get('h', spec.H)}"
            if origins
            else "n/a",
        ),
        (
            "excluded series-origin pairs (scale 0)",
            str(excluded.get("total", "n/a")) if isinstance(excluded, dict) else str(excluded),
        ),
        ("library versions", ", ".join(f"{k} {v}" for k, v in meta.get("versions", {}).items())),
        ("code commit", str(meta.get("git_commit") or "n/a")),
    ]
    out += [f"- **{k}:** {v}" for k, v in facts] + [""]
    neg = meta.get("diagnostics", {}).get("negative_forecasts", {})
    if neg:
        table = pd.DataFrame(neg).T
        rows = [[f, *[str(int(table.loc[f, r])) for r in table.columns]] for f in table.index]
        out += [
            "Negative forecast values (count over all nodes, steps and origins; not clipped):",
            "",
        ]
        out += _table(["forecaster", *table.columns], rows)
    return out


def _metric_table(name: str, scores: pd.DataFrame, metric: str) -> list[str]:
    table = level_table(scores, metric, dataset_levels(name, scores))
    rows = [[f, r, *[f"{v:.3f}" for v in vals]] for (f, r), vals in table.iterrows()]
    return _table(["forecaster", "reconciler", *table.columns], rows)


def _descriptive_table(name: str, scores: pd.DataFrame) -> list[str]:
    levels = dataset_levels(name, scores)
    rows: list[list[str]] = []
    for fc in [f for f in spec.FORECASTERS if f in set(scores["forecaster"])]:
        if not _has(scores, (fc, "base")):
            continue
        for rec in [r for r in spec.RECONCILERS if r != "base"]:
            if not _has(scores, (fc, rec)):
                continue
            cells = [
                _estimate(
                    paired_bootstrap(scores, candidate=(fc, rec), reference=(fc, "base"), level=lv)
                )
                for lv in levels
            ]
            rows.append([fc, rec, *cells])
    if not rows:
        return ["No reconciled forecasts to compare.", ""]
    return _table(["forecaster", "reconciler", *levels], rows)


def build_report(data: dict[str, tuple[pd.DataFrame, dict]]) -> str:
    """The report text for ``{dataset: (scores, meta)}``."""
    out = [
        "# Hierarchical forecasting: results",
        "",
        "Generated by `hier-forecast report` from the score files under `results/`. Accuracy is "
        "reported per level, never pooled. MASE is the decision metric; RMSSE is a check. "
        "Synthetic datasets are labelled synthetic.",
        "",
    ]
    out += _verdict_sections(data)

    out += ["## Provenance", ""]
    for name in ORDER:
        if name in data:
            out += _provenance_section(name, *data[name])
    if not data:
        out += ["No results found.", ""]

    out += ["## Mean accuracy per level", ""]
    for name in ORDER:
        if name not in data:
            continue
        scores = data[name][0]
        out += [f"### {_label(name)}: MASE", "", *_metric_table(name, scores, "mase")]
        out += [f"### {_label(name)}: RMSSE", "", *_metric_table(name, scores, "rmsse")]

    out += [
        "## Descriptive: every reconciler against its own unreconciled base",
        "",
        "Cells are mean d on MASE [95% interval] and the interval's reading. No verdicts are drawn "
        "from this table; the hypotheses above are the only confirmatory tests.",
        "",
    ]
    for name in ORDER:
        if name in data:
            out += [f"### {_label(name)}", "", *_descriptive_table(name, data[name][0])]
    out += [UNIT_NOTE, ""]
    return "\n".join(out)


def write_report(results_dir: Path) -> Path:
    """Read the available ``<dataset>/scores.csv`` and ``meta.json`` and write ``REPORT.md``."""
    results_dir = Path(results_dir)
    data: dict[str, tuple[pd.DataFrame, dict]] = {}
    for name in ORDER:
        scores_path = results_dir / name / "scores.csv"
        if not scores_path.exists():
            continue
        scores = pd.read_csv(scores_path, parse_dates=["origin"])
        meta_path = results_dir / name / "meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        data[name] = (scores, meta)
    path = results_dir / "REPORT.md"
    path.write_text(build_report(data), encoding="utf-8", newline="\n")
    return path
