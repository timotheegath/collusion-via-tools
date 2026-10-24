from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt

from experiments.common.plotting.io_utils import (
    ensure_dir,
    finite,
    mean,
    sem,
    sanitize_filename,
    write_csv,
)
from experiments.common.plotting.logging_utils import log_saved_plot
from experiments.common.plotting.style import apply_default_style

logger = logging.getLogger(__name__)

_ADVERSARY_BAR_COLOR = "#4c78a8"
_BENIGN_BAR_COLOR = "#f58518"
_SERIES_COLORS = (
    "#4c78a8",
    "#009e73",
    "#7570b3",
    "#56b4e9",
    "#666666",
    "#332288",
)
_SERIES_MARKERS = ("o", "^", "D", "v", "P", "X", "<", ">", "h", "H")

_STYLE = {
    "font.size": 24,
    "axes.labelsize": 24,
    "xtick.labelsize": 22,
    "ytick.labelsize": 24,
    "legend.fontsize": 24,
}


def _series_style(series_index: int) -> Tuple[str, str]:
    i = int(series_index)
    return (
        _SERIES_COLORS[i % len(_SERIES_COLORS)],
        _SERIES_MARKERS[i % len(_SERIES_MARKERS)],
    )


def _series_groups(
    rows: List[Dict[str, Any]],
) -> List[Tuple[int, str, List[Dict[str, Any]]]]:
    grouped: Dict[int, List[Dict[str, Any]]] = {}
    labels: Dict[int, str] = {}
    for row in rows:
        raw_index = row.get("series_index")
        try:
            index = int(raw_index) if raw_index not in (None, "") else 0
        except (TypeError, ValueError):
            index = 0
        grouped.setdefault(index, []).append(row)
        label = str(row.get("series_label") or "").strip()
        if label and index not in labels:
            labels[index] = label
    multi = len(grouped) > 1
    return [
        (
            index,
            labels.get(index) or ("Main sweep" if not multi else f"Series {index + 1}"),
            grouped[index],
        )
        for index in sorted(grouped)
    ]


def _multi_legend(ax: Any, n_entries: int, *, fontsize: float = 12.0) -> None:
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=min(2, max(1, n_entries)),
        fontsize=fontsize,
        frameon=False,
        handlelength=1.5,
        columnspacing=1.2,
        borderaxespad=0.2,
    )


def _adv_counts_for_plot(
    main: List[Dict[str, Any]],
    base: List[Dict[str, Any]],
    adv_counts: Sequence[int],
    *,
    multi: bool,
) -> List[int]:
    if not multi:
        return sorted({int(a) for a in adv_counts})
    counts = {
        count
        for count in (_as_int(r.get("adversary_count")) for r in main)
        if count is not None
    }
    if base:
        counts.add(0)
    return sorted(counts)


def _as_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _norm_role(value: Any) -> str:
    return str(value) if value is not None else "None"


def _adv_label(adversary_count: int) -> str:
    """X tick label for an adversary count value (0 is the benign baseline)."""
    return "Benign" if int(adversary_count) == 0 else str(int(adversary_count))


def _bar_color(adversary_count: int) -> str:
    return _BENIGN_BAR_COLOR if int(adversary_count) == 0 else _ADVERSARY_BAR_COLOR


def _aggregate_metric(
    rows: List[Dict[str, Any]],
    *,
    metric_key: str,
    default_when_missing: Optional[float] = None,
) -> Tuple[Optional[float], Optional[float], int]:
    ys = finite([r.get(metric_key) for r in rows])
    if not ys and default_when_missing is not None and rows:
        ys = [float(default_when_missing)]
    if not ys:
        return None, None, 0
    return mean(ys), sem(ys), len(ys)


def _filter_rows(
    rows: List[Dict[str, Any]],
    *,
    num_agents: int,
    target_role: str,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for r in rows:
        if _as_int(r.get("num_agents")) != int(num_agents):
            continue
        if _norm_role(r.get("target_role")) != str(target_role):
            continue
        out.append(r)
    return out


def _filter_adv_count(
    rows: List[Dict[str, Any]], adversary_count: int
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for r in rows:
        if _as_int(r.get("adversary_count")) != int(adversary_count):
            continue
        out.append(r)
    return out


def _rows_for_adversary_count(
    main_rows: List[Dict[str, Any]],
    benign_rows: List[Dict[str, Any]],
    adversary_count: int,
) -> List[Dict[str, Any]]:
    """
    Rows for a single adversary count value.

    adversary_count == 0 is the benign baseline: it is taken from the main rows
    when present, otherwise it falls back to the separately-run benign sweep
    (benign_rows), when one was provided.
    """
    rows = _filter_adv_count(main_rows, int(adversary_count))
    if not rows and int(adversary_count) == 0:
        rows = _filter_adv_count(benign_rows, 0)
    return rows


def _default_when_missing(metric_key: str, adversary_count: int) -> Optional[float]:
    # Coalition regret is undefined for the benign baseline (no coalition): treat as 0.
    if int(adversary_count) == 0 and metric_key == "coalition_reward_regret":
        return 0.0
    return None


def _bar_by_adversary_count(
    *,
    run_rows: List[Dict[str, Any]],
    benign_rows: List[Dict[str, Any]],
    metric_key: str,
    metric_label: str,
    num_agents: int,
    target_role: str,
    adv_counts: Sequence[int],
    out_path: Path,
    emit_artifacts: bool = True,
) -> None:
    """Plot one pooled bar per count, or grouped bars in multi-series mode."""
    apply_default_style(plt)
    plt.rcParams.update(_STYLE)
    ensure_dir(out_path.parent)
    main = _filter_rows(run_rows, num_agents=num_agents, target_role=target_role)
    base = _filter_rows(benign_rows, num_agents=num_agents, target_role=target_role)
    series = _series_groups(main)
    multi = len(series) > 1
    single_index, single_label = series[0][:2] if series else (0, "Main sweep")
    # plot_overview computes the multi-series union and requests one chart per
    # non-zero count; keep each such chart scoped to its own count plus benign.
    counts = sorted({int(a) for a in adv_counts})
    bars: List[Dict[str, Any]] = []
    artifact_rows: List[Dict[str, Any]] = []
    series_catalog = json.dumps([[index, label] for index, label, _ in series])

    for ac in counts:
        if not multi:
            candidates = [
                (
                    _rows_for_adversary_count(main, base, int(ac)),
                    _bar_color(int(ac)),
                    "s" if int(ac) == 0 else "o",
                    "Benign baseline" if int(ac) == 0 else single_label,
                    None if int(ac) == 0 else single_index,
                )
            ]
            width = 0.8
        else:
            candidates = [
                (
                    _filter_adv_count(selected, int(ac)),
                    _series_style(index)[0],
                    _series_style(index)[1],
                    label,
                    index,
                )
                for index, label, selected in series
            ]
            if int(ac) == 0 and base:
                candidates.append(
                    (
                        _filter_adv_count(base, 0),
                        _BENIGN_BAR_COLOR,
                        "s",
                        "Benign baseline",
                        None,
                    )
                )
            width = 0.8 / len(candidates)

        for candidate_i, (selected, color, marker, label, index) in enumerate(
            candidates
        ):
            if not selected:
                continue
            m, e, _ = _aggregate_metric(
                selected,
                metric_key=metric_key,
                default_when_missing=_default_when_missing(metric_key, int(ac)),
            )
            if m is None:
                continue
            offset = (
                (candidate_i - (len(candidates) - 1) / 2.0) * width if multi else 0.0
            )
            bars.append(
                {
                    "ac": int(ac),
                    "x": float(ac) + offset,
                    "height": float(m),
                    "error": float(e or 0.0),
                    "color": color,
                    "label": label,
                    "group": "baseline" if index is None else "main",
                    "width": width if multi else 0.8,
                }
            )
            default = _default_when_missing(metric_key, int(ac))
            for row in selected:
                value = row.get(metric_key)
                if value is None:
                    if default is None:
                        continue
                    value = float(default)
                artifact_rows.append(
                    {
                        "plot_type": "bar_by_adversary_count",
                        "plot_metric_key": metric_key,
                        "plot_metric_label": metric_label,
                        "num_agents": num_agents,
                        "target_role": target_role,
                        "adversary_count": int(ac),
                        "group": "baseline" if index is None else "main",
                        "series_index": "" if index is None else index,
                        "series_label": label,
                        "plot_series_json": series_catalog,
                        "value": value,
                        "run_id": row.get("run_id"),
                        "seed": row.get("seed"),
                    }
                )
    if not bars:
        return

    fig, ax = plt.subplots(figsize=(9.2, 4.4))
    ax.grid(False)
    if not multi:
        ax.bar(
            [int(b["ac"]) for b in bars],
            [float(b["height"]) for b in bars],
            yerr=[float(b["error"]) for b in bars],
            capsize=3,
            color=[str(b["color"]) for b in bars],
            alpha=0.9,
        )
    else:
        for b in bars:
            x = float(b["x"])
            color = str(b["color"])
            ax.bar(
                [x],
                [float(b["height"])],
                yerr=[float(b["error"])],
                width=float(b["width"]),
                capsize=3,
                color=color,
                alpha=0.9,
            )
            label = str(b.get("label") or "")
            raw_values = finite(
                [
                    r["value"]
                    for r in artifact_rows
                    if int(r["adversary_count"]) == int(b["ac"])
                    and str(r.get("group")) == str(b["group"])
                    and (
                        b["group"] == "baseline" or str(r.get("series_label")) == label
                    )
                ]
            )
            ax.scatter([x] * len(raw_values), raw_values, s=16, alpha=0.25, color=color)
        for index, label, _ in series:
            color, marker = _series_style(index)
            ax.scatter([], [], s=38, color=color, marker=marker, label=label)
        has_baseline = any(b["group"] == "baseline" for b in bars)
        if has_baseline:
            ax.scatter(
                [],
                [],
                s=34,
                color=_BENIGN_BAR_COLOR,
                marker="s",
                label="Benign baseline",
            )
        _multi_legend(ax, len(series) + int(has_baseline))

    for b in bars:
        if abs(float(b["height"])) < 1e-12:
            ax.plot(
                [float(b["x"])],
                [0.0],
                marker="_",
                markersize=18,
                color=str(b["color"]),
                zorder=6,
            )
    present = sorted({int(b["ac"]) for b in bars})
    ax.set_xticks(present)
    ax.set_xticklabels([_adv_label(ac) for ac in present], rotation=0, ha="center")
    ax.set_xlabel("Adversary Count")
    ax.set_ylabel(metric_label)
    fig.tight_layout()
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    log_saved_plot(out_path, logger=logger)
    plt.close(fig)
    if not emit_artifacts:
        return

    csv_path = out_path.with_suffix(".csv")
    script_path = out_path.with_name(out_path.stem + "__replot.py")
    write_csv(csv_path, artifact_rows)
    script = f"""\
from __future__ import annotations
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_COLORS = ("#4c78a8", "#009e73", "#7570b3", "#56b4e9", "#666666", "#332288")
_MARKERS = ("o", "^", "D", "v", "P", "X", "<", ">", "h", "H")

def _float(x: Any) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None

def _finite(xs: List[Any]) -> List[float]:
    return [f for f in (_float(x) for x in xs) if f is not None and math.isfinite(f)]

def _mean(xs: List[float]) -> float:
    return float(sum(xs) / len(xs))

def _sem(xs: List[float]) -> float:
    if len(xs) <= 1:
        return 0.0
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) / math.sqrt(len(xs))

def _series(rows: List[Dict[str, Any]]) -> List[Tuple[int, str, List[Dict[str, Any]]]]:
    grouped: Dict[int, List[Dict[str, Any]]] = {{}}
    labels: Dict[int, str] = {{}}
    for row in rows:
        try:
            index = int(row.get("series_index") or 0)
        except Exception:
            index = 0
        grouped.setdefault(index, []).append(row)
        label = str(row.get("series_label") or "").strip()
        if label:
            labels.setdefault(index, label)
    multi = len(grouped) > 1
    return [(i, labels.get(i) or ("Main sweep" if not multi else f"Series {{i + 1}}"), grouped[i]) for i in sorted(grouped)]

def main() -> None:
    here = Path(__file__).resolve().parent
    with (here / {csv_path.name!r}).open("r", encoding="utf-8", newline="") as f:
        rows = [dict(r) for r in csv.DictReader(f)]
    if not rows:
        raise SystemExit("No rows in CSV")
    metric_label = str(rows[0].get("plot_metric_label") or {metric_label!r})
    sweep_rows = [r for r in rows if (r.get("group") or "main") not in ("baseline", "benign")]
    series = _series(sweep_rows)
    catalog = json.loads(rows[0].get("plot_series_json") or "[]")
    if len(catalog) > 1:
        present = {{index: selected for index, _, selected in series}}
        series = [(int(index), str(label), present.get(int(index), [])) for index, label in catalog]
    multi = len(series) > 1
    advs = sorted({{int(float(r.get("adversary_count") or 0)) for r in rows}})
    fig, ax = plt.subplots(figsize=(9.2, 4.4))
    ax.grid(False)
    bars: List[Dict[str, Any]] = []
    if not multi:
        for ac in advs:
            selected = [r for r in rows if int(float(r.get("adversary_count") or 0)) == ac]
            vals = _finite([r.get("value") for r in selected])
            if not vals:
                continue
            color = "#f58518" if ac == 0 else "#4c78a8"
            bars.append({{"ac": ac, "x": float(ac), "height": _mean(vals), "color": color}})
            ax.scatter([float(ac)] * len(vals), vals, s=16, alpha=0.25, color=color)
        ax.bar([b["x"] for b in bars], [b["height"] for b in bars], yerr=[_sem(_finite([r.get("value") for r in rows if int(float(r.get("adversary_count") or 0)) == b["ac"]])) for b in bars], capsize=3, color=[b["color"] for b in bars], alpha=0.9)
    else:
        has_baseline = False
        for ac in advs:
            candidates = [(index, label, [r for r in selected if int(float(r.get("adversary_count") or 0)) == ac], _COLORS[index % len(_COLORS)]) for index, label, selected in series]
            if ac == 0:
                baseline = [r for r in rows if (r.get("group") or "main") in ("baseline", "benign") and int(float(r.get("adversary_count") or 0)) == 0]
                if baseline:
                    candidates.append((None, "Benign baseline", baseline, "#f58518"))
            width = 0.8 / len(candidates)
            for ci, (index, label, selected, color) in enumerate(candidates):
                vals = _finite([r.get("value") for r in selected])
                if not vals:
                    continue
                x = float(ac) + (ci - (len(candidates) - 1) / 2) * width
                bars.append({{"ac": ac, "x": x, "height": _mean(vals), "color": color}})
                ax.bar([x], [_mean(vals)], yerr=[_sem(vals)], width=width, capsize=3, color=color, alpha=0.9)
                ax.scatter([x] * len(vals), vals, s=16, alpha=0.25, color=color)
                if index is None:
                    has_baseline = True
        for index, label, _ in series:
            ax.scatter([], [], s=38, color=_COLORS[index % len(_COLORS)], marker=_MARKERS[index % len(_MARKERS)], label=label)
        if has_baseline:
            ax.scatter([], [], s=34, color="#f58518", marker="s", label="Benign baseline")
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=min(2, len(series) + int(has_baseline)), fontsize=12, frameon=False, handlelength=1.5, columnspacing=1.2, borderaxespad=0.2)
    for b in bars:
        if abs(float(b["height"])) < 1e-12:
            ax.plot([b["x"]], [0.0], marker="_", markersize=18, color=b["color"], zorder=6)
    present = sorted({{int(b["ac"]) for b in bars}})
    ax.set_xticks(present)
    ax.set_xticklabels(["Benign" if ac == 0 else str(ac) for ac in present], rotation=0, ha="center")
    ax.set_xlabel("Adversary Count")
    ax.set_ylabel(metric_label)
    fig.tight_layout()
    fig.savefig(here / {out_path.name!r}, format="pdf", bbox_inches="tight")
    plt.close(fig)

if __name__ == "__main__":
    main()
"""
    script_path.write_text(script, encoding="utf-8")


def _lines_over_adversary_count(
    *,
    run_rows: List[Dict[str, Any]],
    benign_rows: List[Dict[str, Any]],
    metric_key: str,
    metric_label: str,
    num_agents: int,
    target_role: str,
    adv_counts: Sequence[int],
    out_path: Path,
    emit_artifacts: bool = True,
) -> None:
    """Plot pooled lines, or one line per sweep with a benign reference point."""
    apply_default_style(plt)
    plt.rcParams.update(_STYLE)
    ensure_dir(out_path.parent)
    main = _filter_rows(run_rows, num_agents=num_agents, target_role=target_role)
    base = _filter_rows(benign_rows, num_agents=num_agents, target_role=target_role)
    series = _series_groups(main)
    multi = len(series) > 1
    series_catalog = json.dumps([[index, label] for index, label, _ in series])
    counts = _adv_counts_for_plot(main, base, adv_counts, multi=multi)
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    ax.grid(False)

    if not multi:
        xs: List[int] = []
        ys: List[float] = []
        es: List[float] = []
        for ac in counts:
            selected = _rows_for_adversary_count(main, base, int(ac))
            m, e, _ = _aggregate_metric(
                selected,
                metric_key=metric_key,
                default_when_missing=_default_when_missing(metric_key, int(ac)),
            )
            if m is None:
                continue
            xs.append(int(ac))
            ys.append(float(m))
            es.append(float(e or 0.0))
        if xs:
            ax.errorbar(
                xs,
                ys,
                yerr=es,
                marker="o",
                linewidth=3.0,
                markersize=8,
                elinewidth=2.0,
                capthick=2.0,
                capsize=3,
                color=_ADVERSARY_BAR_COLOR,
            )
    else:
        for index, label, selected in series:
            xs = []
            ys = []
            es = []
            for ac in counts:
                if int(ac) == 0:
                    continue
                m, e, _ = _aggregate_metric(
                    _filter_adv_count(selected, int(ac)), metric_key=metric_key
                )
                if m is None:
                    continue
                xs.append(int(ac))
                ys.append(float(m))
                es.append(float(e or 0.0))
            color, marker = _series_style(index)
            if xs:
                ax.errorbar(
                    xs,
                    ys,
                    yerr=es,
                    marker=marker,
                    linewidth=2.6,
                    markersize=8,
                    elinewidth=2.0,
                    capthick=2.0,
                    capsize=3,
                    color=color,
                )
            ax.scatter([], [], s=40, color=color, marker=marker, label=label)
        baseline = _rows_for_adversary_count(main, base, 0)
        m, e, _ = _aggregate_metric(
            baseline,
            metric_key=metric_key,
            default_when_missing=_default_when_missing(metric_key, 0),
        )
        has_baseline = m is not None
        if has_baseline:
            ax.errorbar(
                [0],
                [float(m)],
                yerr=[float(e or 0.0)],
                marker="s",
                linestyle="None",
                linewidth=2.0,
                markersize=8,
                elinewidth=2.0,
                capthick=2.0,
                capsize=3,
                color=_BENIGN_BAR_COLOR,
            )
            ax.scatter(
                [],
                [],
                s=36,
                color=_BENIGN_BAR_COLOR,
                marker="s",
                label="Benign baseline",
            )
        _multi_legend(ax, len(series) + int(has_baseline))
    ax.set_xlabel("Adversary Count")
    ax.set_ylabel(metric_label)
    ax.set_xticks(counts)
    fig.tight_layout()
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    log_saved_plot(out_path, logger=logger)
    plt.close(fig)
    if not emit_artifacts:
        return

    csv_path = out_path.with_suffix(".csv")
    script_path = out_path.with_name(out_path.stem + "__replot.py")
    artifact_rows: List[Dict[str, Any]] = []

    def _emit(selected: List[Dict[str, Any]], ac: int, index: int, label: str) -> None:
        default = _default_when_missing(metric_key, int(ac))
        for row in selected:
            value = row.get(metric_key)
            if value is None:
                if default is None:
                    continue
                value = float(default)
            artifact_rows.append(
                {
                    "plot_type": "lines_over_adversary_count",
                    "plot_metric_key": metric_key,
                    "plot_metric_label": metric_label,
                    "num_agents": num_agents,
                    "target_role": target_role,
                    "adversary_count": int(ac),
                    "group": "baseline" if int(ac) == 0 else "main",
                    "series_index": "" if int(ac) == 0 else index,
                    "series_label": "Benign baseline" if int(ac) == 0 else label,
                    "plot_series_json": series_catalog,
                    "value": value,
                    "run_id": row.get("run_id"),
                    "seed": row.get("seed"),
                }
            )

    if not multi:
        index, label = series[0][:2] if series else (0, "Main sweep")
        for ac in counts:
            _emit(_rows_for_adversary_count(main, base, int(ac)), int(ac), index, label)
    else:
        for ac in counts:
            if int(ac) == 0:
                _emit(_rows_for_adversary_count(main, base, 0), 0, 0, "Benign baseline")
                continue
            for index, label, selected in series:
                _emit(_filter_adv_count(selected, int(ac)), int(ac), index, label)
    write_csv(csv_path, artifact_rows)
    script = f"""\
from __future__ import annotations
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_COLORS = ("#4c78a8", "#009e73", "#7570b3", "#56b4e9", "#666666", "#332288")
_MARKERS = ("o", "^", "D", "v", "P", "X", "<", ">", "h", "H")

def _float(x: Any) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None

def _finite(xs: List[Any]) -> List[float]:
    return [f for f in (_float(x) for x in xs) if f is not None and math.isfinite(f)]

def _mean(xs: List[float]) -> float:
    return float(sum(xs) / len(xs))

def _sem(xs: List[float]) -> float:
    if len(xs) <= 1:
        return 0.0
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) / math.sqrt(len(xs))

def _series(rows: List[Dict[str, Any]]) -> List[Tuple[int, str, List[Dict[str, Any]]]]:
    grouped: Dict[int, List[Dict[str, Any]]] = {{}}
    labels: Dict[int, str] = {{}}
    for row in rows:
        try:
            index = int(row.get("series_index") or 0)
        except Exception:
            index = 0
        grouped.setdefault(index, []).append(row)
        label = str(row.get("series_label") or "").strip()
        if label:
            labels.setdefault(index, label)
    multi = len(grouped) > 1
    return [(i, labels.get(i) or ("Main sweep" if not multi else f"Series {{i + 1}}"), grouped[i]) for i in sorted(grouped)]

def main() -> None:
    here = Path(__file__).resolve().parent
    with (here / {csv_path.name!r}).open("r", encoding="utf-8", newline="") as f:
        rows = [dict(r) for r in csv.DictReader(f)]
    if not rows:
        raise SystemExit("No rows in CSV")
    metric_label = str(rows[0].get("plot_metric_label") or {metric_label!r})
    sweep_rows = [r for r in rows if (r.get("group") or "main") not in ("baseline", "benign")]
    series = _series(sweep_rows)
    catalog = json.loads(rows[0].get("plot_series_json") or "[]")
    if len(catalog) > 1:
        present = {{index: selected for index, _, selected in series}}
        series = [(int(index), str(label), present.get(int(index), [])) for index, label in catalog]
    multi = len(series) > 1
    advs = sorted({{int(float(r.get("adversary_count") or 0)) for r in rows}})
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    ax.grid(False)
    if not multi:
        xs: List[int] = []
        ys: List[float] = []
        es: List[float] = []
        for ac in advs:
            vals = _finite([r.get("value") for r in rows if int(float(r.get("adversary_count") or 0)) == ac])
            if not vals:
                continue
            xs.append(ac)
            ys.append(_mean(vals))
            es.append(_sem(vals))
        if xs:
            ax.errorbar(xs, ys, yerr=es, marker="o", linewidth=3.0, markersize=8, elinewidth=2.0, capthick=2.0, capsize=3, color="#4c78a8")
    else:
        for index, label, selected in series:
            xs = []
            ys = []
            es = []
            for ac in advs:
                if ac == 0:
                    continue
                vals = _finite([r.get("value") for r in selected if int(float(r.get("adversary_count") or 0)) == ac])
                if not vals:
                    continue
                xs.append(ac)
                ys.append(_mean(vals))
                es.append(_sem(vals))
            color, marker = _COLORS[index % len(_COLORS)], _MARKERS[index % len(_MARKERS)]
            if xs:
                ax.errorbar(xs, ys, yerr=es, marker=marker, linewidth=2.6, markersize=8, elinewidth=2.0, capthick=2.0, capsize=3, color=color)
            ax.scatter([], [], s=40, color=color, marker=marker, label=label)
        vals = _finite([r.get("value") for r in rows if int(float(r.get("adversary_count") or 0)) == 0])
        if vals:
            ax.errorbar([0], [_mean(vals)], yerr=[_sem(vals)], marker="s", linestyle="None", linewidth=2.0, markersize=8, elinewidth=2.0, capthick=2.0, capsize=3, color="#f58518")
            ax.scatter([], [], s=36, color="#f58518", marker="s", label="Benign baseline")
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=min(2, len(series) + int(bool(vals))), fontsize=12, frameon=False, handlelength=1.5, columnspacing=1.2, borderaxespad=0.2)
    ax.set_xlabel("Adversary Count")
    ax.set_ylabel(metric_label)
    ax.set_xticks(advs)
    fig.tight_layout()
    fig.savefig(here / {out_path.name!r}, format="pdf", bbox_inches="tight")
    plt.close(fig)

if __name__ == "__main__":
    main()
"""
    script_path.write_text(script, encoding="utf-8")


@dataclass(frozen=True)
class OverviewSpec:
    adv_counts: Tuple[int, ...] = (0, 2, 4, 8)


def plot_overview(
    *,
    run_rows: List[Dict[str, Any]],
    benign_run_rows: Optional[List[Dict[str, Any]]],
    out_dir: Path,
    spec: OverviewSpec = OverviewSpec(),
) -> None:
    """
    Produces:
      - bar plots by adversary count (per adv_count in {2,4,8}, with the benign
        baseline bar at adversary_count 0 overlaid from the benign sweep)
      - line plots over adversary_count (0,2,4,8), one line per (target_role, num_agents)
    Stratified by:
      - target_role (departmental vs Resource_Provisioner)
      - num_agents (requested sweep value)
    """
    ensure_dir(out_dir)
    benign_run_rows = benign_run_rows or []

    roles = sorted(
        {
            _norm_role(r.get("target_role"))
            for r in run_rows
            if r.get("target_role") is not None
        }
    )
    num_agents_vals = sorted(
        {int(r.get("num_agents")) for r in run_rows if r.get("num_agents") is not None}
    )

    if not roles or not num_agents_vals:
        return

    for role in roles:
        role_dir = out_dir / "by_target_role" / sanitize_filename(role)
        ensure_dir(role_dir)

        for n in num_agents_vals:
            n_dir = role_dir / f"n{int(n)}"
            ensure_dir(n_dir)
            role_main = _filter_rows(run_rows, num_agents=int(n), target_role=role)
            role_base = _filter_rows(
                benign_run_rows, num_agents=int(n), target_role=role
            )
            role_series = _series_groups(role_main)
            adv_counts = _adv_counts_for_plot(
                role_main,
                role_base,
                spec.adv_counts,
                multi=len(role_series) > 1,
            )

            # Line plots across adversary counts for this role + n.
            _lines_over_adversary_count(
                run_rows=run_rows,
                benign_rows=benign_run_rows,
                metric_key="joint_reward",
                metric_label="Joint Reward",
                num_agents=int(n),
                target_role=role,
                adv_counts=adv_counts,
                out_path=n_dir / "joint_reward_vs_adv_count.pdf",
            )
            _lines_over_adversary_count(
                run_rows=run_rows,
                benign_rows=benign_run_rows,
                metric_key="coalition_minus_noncoalition_avg",
                metric_label="Coalition Advantage",
                num_agents=int(n),
                target_role=role,
                adv_counts=adv_counts,
                out_path=n_dir / "coalition_advantage_vs_adv_count.pdf",
            )
            _lines_over_adversary_count(
                run_rows=run_rows,
                benign_rows=benign_run_rows,
                metric_key="coalition_minus_noncoalition_hospital_avg",
                metric_label="Coalition Advantage",
                num_agents=int(n),
                target_role=role,
                adv_counts=adv_counts,
                out_path=n_dir / "coalition_advantage_hospital_level_vs_adv_count.pdf",
            )
            _lines_over_adversary_count(
                run_rows=run_rows,
                benign_rows=benign_run_rows,
                metric_key="coalition_minus_noncoalition_hospital_avg_actual",
                metric_label="Coalition Advantage",
                num_agents=int(n),
                target_role=role,
                adv_counts=adv_counts,
                out_path=n_dir
                / "coalition_advantage_hospital_level_actual_vs_adv_count.pdf",
            )
            _lines_over_adversary_count(
                run_rows=run_rows,
                benign_rows=benign_run_rows,
                metric_key="coalition_minus_noncoalition_avg_hospital_adjusted",
                metric_label="Coalition Advantage",
                num_agents=int(n),
                target_role=role,
                adv_counts=adv_counts,
                out_path=n_dir
                / "coalition_advantage_hospital_adjusted_vs_adv_count.pdf",
            )
            _lines_over_adversary_count(
                run_rows=run_rows,
                benign_rows=benign_run_rows,
                metric_key="coalition_minus_noncoalition_avg_hospital_adjusted_normalized",
                metric_label="Coalition Advantage",
                num_agents=int(n),
                target_role=role,
                adv_counts=adv_counts,
                out_path=n_dir
                / "coalition_advantage_hospital_adjusted_normalized_vs_adv_count.pdf",
            )
            _lines_over_adversary_count(
                run_rows=run_rows,
                benign_rows=benign_run_rows,
                metric_key="coalition_minus_noncoalition_hospital_avg_hospital_adjusted_normalized",
                metric_label="Coalition Advantage",
                num_agents=int(n),
                target_role=role,
                adv_counts=adv_counts,
                out_path=n_dir
                / "coalition_advantage_hospital_level_hospital_adjusted_normalized_vs_adv_count.pdf",
            )
            _lines_over_adversary_count(
                run_rows=run_rows,
                benign_rows=benign_run_rows,
                metric_key="joint_reward_regret",
                metric_label="Overall Regret",
                num_agents=int(n),
                target_role=role,
                adv_counts=adv_counts,
                out_path=n_dir / "overall_regret_vs_adv_count.pdf",
            )
            _lines_over_adversary_count(
                run_rows=run_rows,
                benign_rows=benign_run_rows,
                metric_key="coalition_reward_regret",
                metric_label="Coalition Regret",
                num_agents=int(n),
                target_role=role,
                adv_counts=adv_counts,
                out_path=n_dir / "coalition_regret_vs_adv_count.pdf",
            )
            _lines_over_adversary_count(
                run_rows=run_rows,
                benign_rows=benign_run_rows,
                metric_key="coalition_reward_regret_hospital_adjusted",
                metric_label="Coalition Regret",
                num_agents=int(n),
                target_role=role,
                adv_counts=adv_counts,
                out_path=n_dir / "coalition_regret_hospital_adjusted_vs_adv_count.pdf",
            )
            _lines_over_adversary_count(
                run_rows=run_rows,
                benign_rows=benign_run_rows,
                metric_key="agent_resource_failure_penalty_total",
                metric_label="Resource Failure Penalty",
                num_agents=int(n),
                target_role=role,
                adv_counts=adv_counts,
                out_path=n_dir
                / "resource_failure_penalty_total_agent_attributed_vs_adv_count.pdf",
            )

            # Bar plots by adversary count: the benign baseline (adversary_count 0,
            # overlaid from the benign sweep when available) plus the adversary bar.
            for ac in [a for a in adv_counts if a != 0]:
                _bar_by_adversary_count(
                    run_rows=run_rows,
                    benign_rows=benign_run_rows,
                    metric_key="joint_reward",
                    metric_label="Joint Reward",
                    num_agents=int(n),
                    target_role=role,
                    adv_counts=(0, int(ac)),
                    out_path=n_dir / f"joint_reward_adv{int(ac)}.pdf",
                )
                _bar_by_adversary_count(
                    run_rows=run_rows,
                    benign_rows=benign_run_rows,
                    metric_key="coalition_minus_noncoalition_avg",
                    metric_label="Coalition Advantage",
                    num_agents=int(n),
                    target_role=role,
                    adv_counts=(0, int(ac)),
                    out_path=n_dir / f"coalition_advantage_adv{int(ac)}.pdf",
                )
                _bar_by_adversary_count(
                    run_rows=run_rows,
                    benign_rows=benign_run_rows,
                    metric_key="coalition_minus_noncoalition_hospital_avg",
                    metric_label="Coalition Advantage",
                    num_agents=int(n),
                    target_role=role,
                    adv_counts=(0, int(ac)),
                    out_path=n_dir
                    / f"coalition_advantage_hospital_level_adv{int(ac)}.pdf",
                )
                _bar_by_adversary_count(
                    run_rows=run_rows,
                    benign_rows=benign_run_rows,
                    metric_key="coalition_minus_noncoalition_hospital_avg_actual",
                    metric_label="Coalition Advantage",
                    num_agents=int(n),
                    target_role=role,
                    adv_counts=(0, int(ac)),
                    out_path=n_dir
                    / f"coalition_advantage_hospital_level_actual_adv{int(ac)}.pdf",
                )
                _bar_by_adversary_count(
                    run_rows=run_rows,
                    benign_rows=benign_run_rows,
                    metric_key="coalition_minus_noncoalition_avg_hospital_adjusted",
                    metric_label="Coalition Advantage",
                    num_agents=int(n),
                    target_role=role,
                    adv_counts=(0, int(ac)),
                    out_path=n_dir
                    / f"coalition_advantage_hospital_adjusted_adv{int(ac)}.pdf",
                )
                _bar_by_adversary_count(
                    run_rows=run_rows,
                    benign_rows=benign_run_rows,
                    metric_key="coalition_minus_noncoalition_avg_hospital_adjusted_normalized",
                    metric_label="Coalition Advantage",
                    num_agents=int(n),
                    target_role=role,
                    adv_counts=(0, int(ac)),
                    out_path=n_dir
                    / f"coalition_advantage_hospital_adjusted_normalized_adv{int(ac)}.pdf",
                )
                _bar_by_adversary_count(
                    run_rows=run_rows,
                    benign_rows=benign_run_rows,
                    metric_key="coalition_minus_noncoalition_hospital_avg_hospital_adjusted_normalized",
                    metric_label="Coalition Advantage",
                    num_agents=int(n),
                    target_role=role,
                    adv_counts=(0, int(ac)),
                    out_path=n_dir
                    / f"coalition_advantage_hospital_level_hospital_adjusted_normalized_adv{int(ac)}.pdf",
                )

                _bar_by_adversary_count(
                    run_rows=run_rows,
                    benign_rows=benign_run_rows,
                    metric_key="hospital_holding_cost_total",
                    metric_label="Holding Cost",
                    num_agents=int(n),
                    target_role=role,
                    adv_counts=(0, int(ac)),
                    out_path=n_dir / f"holding_cost_total_adv{int(ac)}.pdf",
                )
                _bar_by_adversary_count(
                    run_rows=run_rows,
                    benign_rows=benign_run_rows,
                    metric_key="hospital_resource_failure_penalty_total",
                    metric_label="Resource Failure Penalty",
                    num_agents=int(n),
                    target_role=role,
                    adv_counts=(0, int(ac)),
                    out_path=n_dir / f"resource_failure_penalty_total_adv{int(ac)}.pdf",
                )
                _bar_by_adversary_count(
                    run_rows=run_rows,
                    benign_rows=benign_run_rows,
                    metric_key="agent_resource_failure_penalty_total",
                    metric_label="Resource Failure Penalty",
                    num_agents=int(n),
                    target_role=role,
                    adv_counts=(0, int(ac)),
                    out_path=n_dir
                    / f"resource_failure_penalty_total_agent_attributed_adv{int(ac)}.pdf",
                )
                _bar_by_adversary_count(
                    run_rows=run_rows,
                    benign_rows=benign_run_rows,
                    metric_key="hospital_missed_step_penalty_total",
                    metric_label="Missed-Step Penalty",
                    num_agents=int(n),
                    target_role=role,
                    adv_counts=(0, int(ac)),
                    out_path=n_dir / f"missed_step_penalty_total_adv{int(ac)}.pdf",
                )
                _bar_by_adversary_count(
                    run_rows=run_rows,
                    benign_rows=benign_run_rows,
                    metric_key="joint_reward_regret",
                    metric_label="Overall Regret",
                    num_agents=int(n),
                    target_role=role,
                    adv_counts=(0, int(ac)),
                    out_path=n_dir / f"overall_regret_adv{int(ac)}.pdf",
                )
                _bar_by_adversary_count(
                    run_rows=run_rows,
                    benign_rows=benign_run_rows,
                    metric_key="joint_reward_regret_normalized",
                    metric_label="Overall Regret",
                    num_agents=int(n),
                    target_role=role,
                    adv_counts=(0, int(ac)),
                    out_path=n_dir / f"overall_regret_normalized_adv{int(ac)}.pdf",
                )
                _bar_by_adversary_count(
                    run_rows=run_rows,
                    benign_rows=benign_run_rows,
                    metric_key="coalition_reward_regret",
                    metric_label="Coalition Regret",
                    num_agents=int(n),
                    target_role=role,
                    adv_counts=(0, int(ac)),
                    out_path=n_dir / f"coalition_regret_adv{int(ac)}.pdf",
                )
                _bar_by_adversary_count(
                    run_rows=run_rows,
                    benign_rows=benign_run_rows,
                    metric_key="coalition_reward_regret_hospital_adjusted",
                    metric_label="Coalition Regret",
                    num_agents=int(n),
                    target_role=role,
                    adv_counts=(0, int(ac)),
                    out_path=n_dir
                    / f"coalition_regret_hospital_adjusted_adv{int(ac)}.pdf",
                )
