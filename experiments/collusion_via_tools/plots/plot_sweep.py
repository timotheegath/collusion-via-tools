from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt

from experiments.common.plotting.io_utils import (
    ensure_dir,
    finite,
    groupby,
    mean,
    sem,
    write_csv,
)
from experiments.common.plotting.logging_utils import log_saved_plot
from experiments.common.plotting.style import apply_default_style

logger = logging.getLogger(__name__)

_STYLE = {
    # Paper-friendly defaults (reasonable in LaTeX figures).
    "font.size": 26,
    "axes.labelsize": 26,
    "xtick.labelsize": 26,
    "ytick.labelsize": 26,
    "legend.fontsize": 26,
}

# Cool/neutral colorblind-safe hues keep every sweep distinct from the
# baseline's reserved orange. Squares are reserved for baseline references.
_SERIES_COLORS = (
    "#4c78a8",  # established steel blue
    "#009e73",  # Okabe-Ito bluish green
    "#7570b3",  # ColorBrewer violet
    "#56b4e9",  # Okabe-Ito sky blue
    "#666666",  # neutral graphite
    "#332288",  # Tol deep indigo
)
_SERIES_MARKERS = ("o", "^", "D", "v", "P", "X", "<", ">", "h", "H")


def _series_style(series_index: int) -> Tuple[str, str]:
    i = int(series_index)
    return (
        _SERIES_COLORS[i % len(_SERIES_COLORS)],
        _SERIES_MARKERS[i % len(_SERIES_MARKERS)],
    )


def _series_groups(
    rows: List[Dict[str, Any]],
) -> List[Tuple[int, str, List[Dict[str, Any]]]]:
    """Return sweep series in index order, with a legacy single-series fallback."""
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


def _series_offsets(n_series: int, has_baseline: bool) -> List[float]:
    """Evenly fill the historical 0.56-wide group, baseline last."""
    count = int(n_series) + int(bool(has_baseline))
    if count <= 1:
        return [0.0]
    slot_width = 0.56 / count
    return [(i - (count - 1) / 2.0) * slot_width for i in range(count)]


def _add_violin(
    ax: Any,
    *,
    pos: float,
    ys: List[float],
    color: str,
    width: float = 0.28,
    alpha: float = 0.35,
) -> None:
    if not ys:
        return
    parts = ax.violinplot(
        [ys],
        positions=[pos],
        widths=width,
        showmeans=False,
        showmedians=True,
        showextrema=False,
    )
    for b in parts.get("bodies", []):
        b.set_facecolor(color)
        b.set_edgecolor(color)
        b.set_alpha(alpha)
    if parts.get("cmedians") is not None:
        parts["cmedians"].set_color(color)
        parts["cmedians"].set_linewidth(1.6)


def _unique_non_null(rows: List[Dict[str, Any]], key: str) -> List[Any]:
    seen = []
    for r in rows:
        v = r.get(key)
        if v is None:
            continue
        if v not in seen:
            seen.append(v)
    return seen


def _infer_x_key(run_rows: List[Dict[str, Any]]) -> Tuple[str, str]:
    """
    Picks a sweep axis based on what varies in the data.

    Returns (x_key, x_label).
    """
    if len(_unique_non_null(run_rows, "adversary_count")) > 1:
        return ("adversary_count", "Adversary Count")
    if len(_unique_non_null(run_rows, "target_role")) > 1:
        return ("target_role", "Target Role")
    if len(_unique_non_null(run_rows, "num_agents")) > 1:
        return ("num_agents", "Number of Agents")
    return ("run_id", "Run")


def _is_numeric(values: Iterable[Any]) -> bool:
    for v in values:
        if v is None:
            continue
        try:
            float(v)
        except Exception:
            return False
    return True


def _plot_metric_by_x(
    run_rows: List[Dict[str, Any]],
    *,
    x_key: str,
    x_label: str,
    y_key: str,
    y_label: str,
    out_path: Path,
    baseline_rows: Optional[List[Dict[str, Any]]] = None,
    baseline_label: str = "Benign baseline",
    emit_artifacts: bool = True,
) -> None:
    apply_default_style(plt)
    plt.rcParams.update(_STYLE)
    ensure_dir(out_path.parent)

    xs = [r.get(x_key) for r in run_rows]
    if not xs:
        return
    numeric_x = _is_numeric(xs)
    series = _series_groups(run_rows)
    multi = len(series) > 1
    has_baseline = bool(baseline_rows)
    offsets = _series_offsets(len(series), has_baseline)
    violin_width = min(0.28, 0.56 / len(offsets) * 0.78)

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.grid(False)
    baseline_color = "#f58518"
    main_color = "#4c78a8"

    def _draw(
        rows: List[Dict[str, Any]],
        *,
        pos: float,
        error_pos: float,
        color: str,
        marker: str,
        single: bool = False,
    ) -> None:
        ys = finite([r.get(y_key) for r in rows])
        if not ys:
            return
        _add_violin(
            ax,
            pos=pos,
            ys=ys,
            color=color,
            width=0.28 if single else violin_width,
        )
        ax.errorbar(
            [error_pos],
            [mean(ys)],
            yerr=[sem(ys)],
            fmt=marker,
            ms=5.5 if marker == "s" else 6,
            color="black" if single and marker == "o" else color,
            capsize=3,
            zorder=6 if marker == "s" else 5,
        )

    if numeric_x:
        rows_for_x = list(run_rows) + list(baseline_rows or []) if multi else run_rows
        by_x_all = groupby(rows_for_x, (x_key,))
        x_vals = sorted(
            [x for x in by_x_all if x[0] is not None], key=lambda t: float(t[0])
        )
        if not multi:
            by_x = groupby(run_rows, (x_key,))
            for x_t in x_vals:
                x = x_t[0]
                _draw(
                    by_x.get((x,), []),
                    pos=float(x) - 0.11,
                    error_pos=float(x),
                    color=main_color,
                    marker="o",
                    single=True,
                )
        else:
            for series_i, (index, _, series_rows) in enumerate(series):
                color, marker = _series_style(index)
                by_x = groupby(series_rows, (x_key,))
                for x_t in x_vals:
                    x = x_t[0]
                    pos = float(x) + offsets[series_i]
                    _draw(
                        by_x.get((x,), []),
                        pos=pos,
                        error_pos=pos,
                        color=color,
                        marker=marker,
                    )
        if baseline_rows:
            by_x_base = groupby(baseline_rows, (x_key,))
            offset = offsets[-1] if multi else 0.11
            base_x_vals = [x for x in by_x_base if x[0] is not None]
            for x_t in sorted(base_x_vals, key=lambda t: float(t[0])):
                x = x_t[0]
                pos = float(x) + offset
                _draw(
                    by_x_base[(x,)],
                    pos=pos,
                    error_pos=pos if multi else float(x),
                    color=baseline_color,
                    marker="s",
                    single=not multi,
                )
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
    else:
        cats = [str(v) for v in _unique_non_null(run_rows, x_key)]
        if not cats:
            return
        if not multi:
            by_x = groupby(run_rows, (x_key,))
            for i, category in enumerate(cats):
                _draw(
                    by_x.get((category,), []),
                    pos=float(i) - 0.11,
                    error_pos=float(i),
                    color=main_color,
                    marker="o",
                    single=True,
                )
        else:
            for series_i, (index, _, series_rows) in enumerate(series):
                color, marker = _series_style(index)
                by_x = groupby(series_rows, (x_key,))
                for i, category in enumerate(cats):
                    pos = float(i) + offsets[series_i]
                    _draw(
                        by_x.get((category,), []),
                        pos=pos,
                        error_pos=pos,
                        color=color,
                        marker=marker,
                    )
        if baseline_rows:
            baseline_rows_for_plot = baseline_rows
            base_cats = [str(v) for v in _unique_non_null(baseline_rows, x_key)]
            if len(base_cats) == 1 and len(cats) > 1:
                baseline_rows_for_plot = []
                for category in cats:
                    for row in baseline_rows:
                        patched = dict(row)
                        patched[x_key] = category
                        baseline_rows_for_plot.append(patched)
            by_x_base = groupby(baseline_rows_for_plot, (x_key,))
            offset = offsets[-1] if multi else 0.11
            for i, category in enumerate(cats):
                pos = float(i) + offset
                _draw(
                    by_x_base.get((category,), []),
                    pos=pos,
                    error_pos=pos if multi else float(i),
                    color=baseline_color,
                    marker="s",
                    single=not multi,
                )
        ax.set_xticks(range(len(cats)))
        rot = 0 if max((len(c) for c in cats), default=0) <= 12 else 90
        ax.set_xticklabels(cats, rotation=rot, ha="right" if rot else "center")
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

    if multi:
        for index, label, _ in series:
            color, marker = _series_style(index)
            ax.scatter([], [], s=36, color=color, label=label, marker=marker)
    if baseline_rows:
        ax.scatter([], [], s=28, color=baseline_color, label=baseline_label, marker="s")
        if not multi:
            ax.scatter([], [], s=28, color=main_color, label="Main sweep", marker="o")
    if multi:
        _multi_legend(ax, len(series) + int(has_baseline))
    elif baseline_rows:
        ax.legend(loc="best")

    if multi and not numeric_x:
        fig.subplots_adjust(left=0.16, right=0.98, bottom=0.30, top=0.70)
    else:
        fig.tight_layout()
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    log_saved_plot(out_path, logger=logger)
    plt.close(fig)

    if not emit_artifacts:
        return

    csv_path = out_path.with_suffix(".csv")
    script_path = out_path.with_name(out_path.stem + "__replot.py")
    baseline_expanded = False
    baseline_rows_for_plot = baseline_rows
    if not numeric_x and baseline_rows:
        base_cats = [str(v) for v in _unique_non_null(baseline_rows, x_key)]
        cats = [str(v) for v in _unique_non_null(run_rows, x_key)]
        if len(base_cats) == 1 and len(cats) > 1:
            baseline_expanded = True
            baseline_rows_for_plot = []
            for category in cats:
                for row in baseline_rows:
                    patched = dict(row)
                    patched[x_key] = category
                    baseline_rows_for_plot.append(patched)

    series_by_identity = {
        id(row): (index, label)
        for index, label, series_rows in series
        for row in series_rows
    }
    series_catalog = json.dumps([[index, label] for index, label, _ in series])

    def _point_rows(rows: List[Dict[str, Any]], *, group: str) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for row in rows:
            x = row.get(x_key)
            y = row.get(y_key)
            if x is None or y is None:
                continue
            index, label = series_by_identity.get(id(row), (0, "Main sweep"))
            out.append(
                {
                    "group": group,
                    "series_index": index if group == "main" else "",
                    "series_label": label if group == "main" else baseline_label,
                    "plot_series_json": series_catalog,
                    "plot_x_key": x_key,
                    "plot_y_key": y_key,
                    "plot_x_label": x_label,
                    "plot_y_label": y_label,
                    "baseline_label": baseline_label if group == "baseline" else "",
                    "baseline_expanded": bool(baseline_expanded)
                    if group == "baseline"
                    else False,
                    x_key: x,
                    y_key: y,
                    "run_id": row.get("run_id"),
                    "seed": row.get("seed"),
                    "target_role": row.get("target_role"),
                    "adversary_count": row.get("adversary_count"),
                    "num_agents": row.get("num_agents"),
                }
            )
        return out

    rows_out = _point_rows(run_rows, group="main")
    if baseline_rows_for_plot:
        rows_out += _point_rows(list(baseline_rows_for_plot), group="baseline")
    write_csv(csv_path, rows_out)

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

def _group(rows: List[Dict[str, Any]], key: str) -> Dict[Any, List[Dict[str, Any]]]:
    out: Dict[Any, List[Dict[str, Any]]] = {{}}
    for row in rows:
        out.setdefault(row.get(key), []).append(row)
    return out

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

def _numeric(values: List[Any]) -> bool:
    for value in values:
        if value is not None and _float(value) is None:
            return False
    return True

def main() -> None:
    here = Path(__file__).resolve().parent
    with (here / {csv_path.name!r}).open("r", encoding="utf-8", newline="") as f:
        rows = [dict(r) for r in csv.DictReader(f)]
    if not rows:
        raise SystemExit("No rows in CSV")
    x_key = rows[0].get("plot_x_key") or {x_key!r}
    y_key = rows[0].get("plot_y_key") or {y_key!r}
    x_label = rows[0].get("plot_x_label") or {x_label!r}
    y_label = rows[0].get("plot_y_label") or {y_label!r}
    main_rows = [r for r in rows if r.get("group") == "main"]
    base_rows = [r for r in rows if r.get("group") == "baseline"]
    series = _series(main_rows)
    catalog = json.loads(rows[0].get("plot_series_json") or "[]")
    if len(catalog) > 1:
        present = {{index: selected for index, _, selected in series}}
        series = [(int(index), str(label), present.get(int(index), [])) for index, label in catalog]
    multi = len(series) > 1
    count = len(series) + int(bool(base_rows))
    offsets = [(i - (count - 1) / 2) * 0.56 / count for i in range(count)]
    width = min(0.28, 0.56 / count * 0.78)
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.grid(False)

    def draw(selected: List[Dict[str, Any]], pos: float, error_pos: float, color: str, marker: str, single: bool = False) -> None:
        vals = _finite([r.get(y_key) for r in selected])
        if not vals:
            return
        parts = ax.violinplot([vals], positions=[pos], widths=0.28 if single else width, showmeans=False, showmedians=True, showextrema=False)
        for body in parts.get("bodies", []):
            body.set_facecolor(color)
            body.set_edgecolor(color)
            body.set_alpha(0.35)
        if parts.get("cmedians") is not None:
            parts["cmedians"].set_color(color)
            parts["cmedians"].set_linewidth(1.6)
        ax.errorbar([error_pos], [_mean(vals)], yerr=[_sem(vals)], fmt=marker, ms=5.5 if marker == "s" else 6, color="black" if single and marker == "o" else color, capsize=3, zorder=6 if marker == "s" else 5)

    numeric_x = _numeric([r.get(x_key) for r in main_rows])
    if numeric_x:
        source = main_rows + base_rows if multi else main_rows
        x_vals = sorted(_group(source, x_key), key=lambda x: float(x))
        if not multi:
            grouped = _group(main_rows, x_key)
            for x in x_vals:
                draw(grouped[x], float(x) - 0.11, float(x), "#4c78a8", "o", True)
        else:
            for si, (index, _, selected) in enumerate(series):
                grouped = _group(selected, x_key)
                color, marker = _COLORS[index % len(_COLORS)], _MARKERS[index % len(_MARKERS)]
                for x in x_vals:
                    pos = float(x) + offsets[si]
                    draw(grouped.get(x, []), pos, pos, color, marker)
        if base_rows:
            grouped = _group(base_rows, x_key)
            offset = offsets[-1] if multi else 0.11
            for x in sorted(grouped, key=float):
                pos = float(x) + offset
                draw(grouped[x], pos, pos if multi else float(x), "#f58518", "s", not multi)
    else:
        cats = list(dict.fromkeys(str(r.get(x_key)) for r in main_rows))
        if not multi:
            grouped = _group(main_rows, x_key)
            for i, category in enumerate(cats):
                draw(grouped.get(category, []), i - 0.11, i, "#4c78a8", "o", True)
        else:
            for si, (index, _, selected) in enumerate(series):
                grouped = _group(selected, x_key)
                color, marker = _COLORS[index % len(_COLORS)], _MARKERS[index % len(_MARKERS)]
                for i, category in enumerate(cats):
                    pos = i + offsets[si]
                    draw(grouped.get(category, []), pos, pos, color, marker)
        if base_rows:
            grouped = _group(base_rows, x_key)
            offset = offsets[-1] if multi else 0.11
            for i, category in enumerate(cats):
                pos = i + offset
                draw(grouped.get(category, []), pos, pos if multi else float(i), "#f58518", "s", not multi)
        rot = 0 if max(map(len, cats), default=0) <= 12 else 90
        ax.set_xticks(range(len(cats)))
        ax.set_xticklabels(cats, rotation=rot, ha="right" if rot else "center")
    ax.set_xlabel(str(x_label))
    ax.set_ylabel(str(y_label))
    if multi:
        for index, label, _ in series:
            ax.scatter([], [], s=36, color=_COLORS[index % len(_COLORS)], marker=_MARKERS[index % len(_MARKERS)], label=label)
    if base_rows:
        label = base_rows[0].get("series_label") or base_rows[0].get("baseline_label") or "Benign baseline"
        ax.scatter([], [], s=28, color="#f58518", marker="s", label=label)
        if not multi:
            ax.scatter([], [], s=28, color="#4c78a8", marker="o", label="Main sweep")
    if multi:
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=min(2, len(series) + int(bool(base_rows))), fontsize=12, frameon=False, handlelength=1.5, columnspacing=1.2, borderaxespad=0.2)
    elif base_rows:
        ax.legend(loc="best")
    if multi and not numeric_x:
        fig.subplots_adjust(left=0.16, right=0.98, bottom=0.30, top=0.70)
    else:
        fig.tight_layout()
    fig.savefig(here / {out_path.name!r}, format="pdf", bbox_inches="tight")
    plt.close(fig)

if __name__ == "__main__":
    main()
"""
    script_path.write_text(script, encoding="utf-8")


def plot_distance_effects(
    agent_rows: List[Dict[str, Any]],
    *,
    out_path: Path,
    emit_artifacts: bool = True,
) -> None:
    """Plot non-adversary suspicion and attribution by adversary distance."""
    apply_default_style(plt)
    plt.rcParams.update(_STYLE)
    ensure_dir(out_path.parent)

    rows = [
        r
        for r in agent_rows
        if str(r.get("role") or "") != "adversary"
        and r.get("distance_to_nearest_adversary") is not None
    ]
    if not rows:
        return
    series = _series_groups(rows)
    multi = len(series) > 1

    def _rates(
        selected: List[Dict[str, Any]],
    ) -> Tuple[List[int], List[float], List[float]]:
        by_d = groupby(selected, ("distance_to_nearest_adversary",))
        ds = sorted(int(k[0]) for k in by_d if k[0] is not None)
        susp_rates: List[float] = []
        corr_rates: List[float] = []
        for distance in ds:
            bucket = by_d.get((distance,), [])
            denom = len(bucket)
            susp = sum(1 for r in bucket if bool(r.get("suspects_foul_play")))
            corr = sum(1 for r in bucket if bool(r.get("correctly_blames_adversary")))
            susp_rates.append(susp / denom if denom else 0.0)
            corr_rates.append(corr / denom if denom else 0.0)
        return ds, susp_rates, corr_rates

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.grid(False)
    if not multi:
        ds, susp_rates, corr_rates = _rates(rows)
        if not ds:
            return
        ax.plot(
            ds,
            susp_rates,
            marker="o",
            linewidth=3.0,
            markersize=8,
            label="Suspicion rate (non-adversary)",
        )
        ax.plot(
            ds,
            corr_rates,
            marker="o",
            linewidth=3.0,
            markersize=8,
            label="Correct attribution rate (non-adversary)",
        )
        ax.legend(loc="best")
    else:
        for index, label, selected in series:
            ds, susp_rates, corr_rates = _rates(selected)
            color, marker = _series_style(index)
            ax.plot(
                ds,
                susp_rates,
                marker=marker,
                color=color,
                linewidth=2.4,
                markersize=7,
                linestyle="-",
                label=f"{label} — suspicion",
            )
            ax.plot(
                ds,
                corr_rates,
                marker=marker,
                color=color,
                linewidth=2.4,
                markersize=7,
                linestyle="--",
                label=f"{label} — correct attribution",
            )
        _multi_legend(ax, 2 * len(series), fontsize=10.5)
    ax.set_xlabel("Distance to Nearest Adversary")
    ax.set_ylabel("Rate")
    ax.set_ylim(0.0, 1.05)
    fig.tight_layout()
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    log_saved_plot(out_path, logger=logger)
    plt.close(fig)

    if not emit_artifacts:
        return
    csv_path = out_path.with_suffix(".csv")
    script_path = out_path.with_name(out_path.stem + "__replot.py")
    artifact_rows: List[Dict[str, Any]] = []
    for index, label, selected in series:
        for row in selected:
            artifact_rows.append(
                {
                    "group": "main",
                    "series_index": index,
                    "series_label": label,
                    "distance_to_nearest_adversary": row.get(
                        "distance_to_nearest_adversary"
                    ),
                    "suspects_foul_play": bool(row.get("suspects_foul_play")),
                    "correctly_blames_adversary": bool(
                        row.get("correctly_blames_adversary")
                    ),
                    "run_id": row.get("run_id"),
                    "seed": row.get("seed"),
                }
            )
    write_csv(csv_path, artifact_rows)
    script = f"""\
from __future__ import annotations
import csv
from pathlib import Path
from typing import Any, Dict, List, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_COLORS = ("#4c78a8", "#009e73", "#7570b3", "#56b4e9", "#666666", "#332288")
_MARKERS = ("o", "^", "D", "v", "P", "X", "<", ">", "h", "H")

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

def _bool(value: Any) -> bool:
    return str(value).strip().lower() in ("1", "true", "yes")

def _rates(rows: List[Dict[str, Any]]) -> Tuple[List[int], List[float], List[float]]:
    grouped: Dict[int, List[Dict[str, Any]]] = {{}}
    for row in rows:
        grouped.setdefault(int(float(row["distance_to_nearest_adversary"])), []).append(row)
    ds = sorted(grouped)
    susp = [sum(_bool(r.get("suspects_foul_play")) for r in grouped[d]) / len(grouped[d]) for d in ds]
    corr = [sum(_bool(r.get("correctly_blames_adversary")) for r in grouped[d]) / len(grouped[d]) for d in ds]
    return ds, susp, corr

def main() -> None:
    here = Path(__file__).resolve().parent
    with (here / {csv_path.name!r}).open("r", encoding="utf-8", newline="") as f:
        rows = [dict(r) for r in csv.DictReader(f)]
    if not rows:
        raise SystemExit("No rows in CSV")
    series = _series(rows)
    multi = len(series) > 1
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.grid(False)
    if not multi:
        ds, susp, corr = _rates(rows)
        ax.plot(ds, susp, marker="o", linewidth=3.0, markersize=8, label="Suspicion rate (non-adversary)")
        ax.plot(ds, corr, marker="o", linewidth=3.0, markersize=8, label="Correct attribution rate (non-adversary)")
        ax.legend(loc="best")
    else:
        for index, label, selected in series:
            ds, susp, corr = _rates(selected)
            color, marker = _COLORS[index % len(_COLORS)], _MARKERS[index % len(_MARKERS)]
            ax.plot(ds, susp, marker=marker, color=color, linewidth=2.4, markersize=7, linestyle="-", label=f"{{label}} — suspicion")
            ax.plot(ds, corr, marker=marker, color=color, linewidth=2.4, markersize=7, linestyle="--", label=f"{{label}} — correct attribution")
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2, fontsize=10.5, frameon=False, handlelength=1.5, columnspacing=1.2, borderaxespad=0.2)
    ax.set_xlabel("Distance to Nearest Adversary")
    ax.set_ylabel("Rate")
    ax.set_ylim(0.0, 1.05)
    fig.tight_layout()
    fig.savefig(here / {out_path.name!r}, format="pdf", bbox_inches="tight")
    plt.close(fig)

if __name__ == "__main__":
    main()
"""
    script_path.write_text(script, encoding="utf-8")


def plot_sweep_metrics(
    *,
    run_rows: List[Dict[str, Any]],
    agent_rows: List[Dict[str, Any]],
    out_dir: Path,
    baseline_run_rows: Optional[List[Dict[str, Any]]] = None,
    baseline_agent_rows: Optional[List[Dict[str, Any]]] = None,
    baseline_label: str = "Benign baseline",
) -> None:
    ensure_dir(out_dir)
    x_key, x_label = _infer_x_key(run_rows)

    # Some metrics are undefined in benign runs (no coalition). For plotting, treat
    # coalition advantage as 0 when adversary_count == 0.
    baseline_run_rows_coalition_adv = baseline_run_rows
    if baseline_run_rows:
        patched: List[Dict[str, Any]] = []
        for r in baseline_run_rows:
            rr = dict(r)
            try:
                ac = int(rr.get("adversary_count") or 0)
            except Exception:
                ac = 0
            if ac == 0 and rr.get("coalition_minus_noncoalition_avg") is None:
                rr["coalition_minus_noncoalition_avg"] = 0.0
            if (
                ac == 0
                and rr.get("coalition_minus_noncoalition_avg_hospital_adjusted") is None
            ):
                rr["coalition_minus_noncoalition_avg_hospital_adjusted"] = 0.0
            patched.append(rr)
        baseline_run_rows_coalition_adv = patched

    baseline_run_rows_coalition_regret = baseline_run_rows
    if baseline_run_rows:
        patched: List[Dict[str, Any]] = []
        for r in baseline_run_rows:
            rr = dict(r)
            try:
                ac = int(rr.get("adversary_count") or 0)
            except Exception:
                ac = 0
            if ac == 0 and rr.get("coalition_reward_regret") is None:
                rr["coalition_reward_regret"] = 0.0
            if ac == 0 and rr.get("coalition_reward_regret_hospital_adjusted") is None:
                rr["coalition_reward_regret_hospital_adjusted"] = 0.0
            patched.append(rr)
        baseline_run_rows_coalition_regret = patched

    # Core outcomes
    _plot_metric_by_x(
        run_rows,
        x_key=x_key,
        x_label=x_label,
        y_key="joint_reward",
        y_label="Joint Reward",
        out_path=out_dir / "joint_reward.pdf",
        baseline_rows=baseline_run_rows,
        baseline_label=baseline_label,
    )
    _plot_metric_by_x(
        run_rows,
        x_key=x_key,
        x_label=x_label,
        y_key="joint_reward_regret",
        y_label="Overall Regret",
        out_path=out_dir / "overall_regret.pdf",
        baseline_rows=baseline_run_rows,
        baseline_label=baseline_label,
    )
    _plot_metric_by_x(
        run_rows,
        x_key=x_key,
        x_label=x_label,
        y_key="resource_failures_total",
        y_label="Total Resource Failures",
        out_path=out_dir / "resource_failures_total.pdf",
        baseline_rows=baseline_run_rows,
        baseline_label=baseline_label,
    )
    _plot_metric_by_x(
        run_rows,
        x_key=x_key,
        x_label=x_label,
        y_key="failed_patients_count",
        y_label="Failed Patients",
        out_path=out_dir / "failed_patients_count.pdf",
        baseline_rows=baseline_run_rows,
        baseline_label=baseline_label,
    )
    _plot_metric_by_x(
        run_rows,
        x_key=x_key,
        x_label=x_label,
        y_key="converged_patients",
        y_label="Converged Patients",
        out_path=out_dir / "converged_patients.pdf",
        baseline_rows=baseline_run_rows,
        baseline_label=baseline_label,
    )
    _plot_metric_by_x(
        run_rows,
        x_key=x_key,
        x_label=x_label,
        y_key="meeting_outcome_variables_assigned",
        y_label="Variables Assigned",
        out_path=out_dir / "variables_assigned.pdf",
        baseline_rows=baseline_run_rows,
        baseline_label=baseline_label,
    )

    # Belief / judge metrics
    _plot_metric_by_x(
        run_rows,
        x_key=x_key,
        x_label=x_label,
        y_key="suspicion_rate_non_adversary",
        y_label="Suspicion Rate (Non-adversary)",
        out_path=out_dir / "suspicion_rate_non_adversary.pdf",
        baseline_rows=baseline_run_rows,
        baseline_label=baseline_label,
    )
    _plot_metric_by_x(
        run_rows,
        x_key=x_key,
        x_label=x_label,
        y_key="correct_attribution_rate_non_adversary",
        y_label="Correct Attribution Rate (Non-adversary)",
        out_path=out_dir / "correct_attribution_rate_non_adversary.pdf",
        baseline_rows=baseline_run_rows,
        baseline_label=baseline_label,
    )
    _plot_metric_by_x(
        run_rows,
        x_key=x_key,
        x_label=x_label,
        y_key="avg_distance_to_adversary_suspicious_agents",
        y_label="Avg Distance to Adversary (Suspicious Non-adversaries)",
        out_path=out_dir / "avg_distance_suspicious.pdf",
        baseline_rows=baseline_run_rows,
        baseline_label=baseline_label,
    )

    # Communication proxies
    _plot_metric_by_x(
        run_rows,
        x_key=x_key,
        x_label=x_label,
        y_key="total_messages",
        y_label="Total Messages",
        out_path=out_dir / "total_messages.pdf",
        baseline_rows=baseline_run_rows,
        baseline_label=baseline_label,
    )
    _plot_metric_by_x(
        run_rows,
        x_key=x_key,
        x_label=x_label,
        y_key="complaint_messages",
        y_label="Complaint Messages",
        out_path=out_dir / "complaint_messages.pdf",
        baseline_rows=baseline_run_rows,
        baseline_label=baseline_label,
    )

    # Graph metrics (key logged fields)
    _plot_metric_by_x(
        run_rows,
        x_key=x_key,
        x_label=x_label,
        y_key="graph_density",
        y_label="Graph Density",
        out_path=out_dir / "graph_density.pdf",
        baseline_rows=baseline_run_rows,
        baseline_label=baseline_label,
    )
    _plot_metric_by_x(
        run_rows,
        x_key=x_key,
        x_label=x_label,
        y_key="graph_avg_degree",
        y_label="Graph Avg Degree",
        out_path=out_dir / "graph_avg_degree.pdf",
        baseline_rows=baseline_run_rows,
        baseline_label=baseline_label,
    )
    _plot_metric_by_x(
        run_rows,
        x_key=x_key,
        x_label=x_label,
        y_key="graph_avg_clustering",
        y_label="Graph Avg Clustering",
        out_path=out_dir / "graph_avg_clustering.pdf",
        baseline_rows=baseline_run_rows,
        baseline_label=baseline_label,
    )

    # Coalition advantage (if present)
    _plot_metric_by_x(
        run_rows,
        x_key=x_key,
        x_label=x_label,
        y_key="coalition_minus_noncoalition_avg",
        y_label="Coalition Advantage",
        out_path=out_dir / "coalition_advantage.pdf",
        baseline_rows=baseline_run_rows_coalition_adv,
        baseline_label=baseline_label,
    )
    _plot_metric_by_x(
        run_rows,
        x_key=x_key,
        x_label=x_label,
        y_key="coalition_minus_noncoalition_avg_hospital_adjusted",
        y_label="Coalition Advantage",
        out_path=out_dir / "coalition_advantage_hospital_adjusted.pdf",
        baseline_rows=baseline_run_rows_coalition_adv,
        baseline_label=baseline_label,
    )
    _plot_metric_by_x(
        run_rows,
        x_key=x_key,
        x_label=x_label,
        y_key="coalition_minus_noncoalition_avg_hospital_adjusted_normalized",
        y_label="Coalition Advantage",
        out_path=out_dir / "coalition_advantage_hospital_adjusted_normalized.pdf",
        baseline_rows=baseline_run_rows_coalition_adv,
        baseline_label=baseline_label,
    )
    _plot_metric_by_x(
        run_rows,
        x_key=x_key,
        x_label=x_label,
        y_key="coalition_reward_regret",
        y_label="Coalition Regret",
        out_path=out_dir / "coalition_regret.pdf",
        baseline_rows=baseline_run_rows_coalition_regret,
        baseline_label=baseline_label,
    )
    _plot_metric_by_x(
        run_rows,
        x_key=x_key,
        x_label=x_label,
        y_key="coalition_reward_regret_hospital_adjusted",
        y_label="Coalition Regret",
        out_path=out_dir / "coalition_regret_hospital_adjusted.pdf",
        baseline_rows=baseline_run_rows_coalition_regret,
        baseline_label=baseline_label,
    )
    _plot_metric_by_x(
        run_rows,
        x_key=x_key,
        x_label=x_label,
        y_key="coalition_reward_avg",
        y_label="Coalition Avg Reward",
        out_path=out_dir / "coalition_reward_avg.pdf",
        baseline_rows=baseline_run_rows,
        baseline_label=baseline_label,
    )
    _plot_metric_by_x(
        run_rows,
        x_key=x_key,
        x_label=x_label,
        y_key="noncoalition_reward_avg",
        y_label="Non-coalition Avg Reward",
        out_path=out_dir / "noncoalition_reward_avg.pdf",
        baseline_rows=baseline_run_rows,
        baseline_label=baseline_label,
    )

    # Distance effects (aggregated across runs)
    plot_distance_effects(agent_rows, out_path=out_dir / "distance_effects.pdf")

    _plot_metric_by_x(
        run_rows,
        x_key=x_key,
        x_label=x_label,
        y_key="hospital_holding_cost_total",
        y_label="Holding Cost",
        out_path=out_dir / "holding_cost_total.pdf",
        baseline_rows=baseline_run_rows,
        baseline_label=baseline_label,
    )
    _plot_metric_by_x(
        run_rows,
        x_key=x_key,
        x_label=x_label,
        y_key="hospital_resource_failure_penalty_total",
        y_label="Resource Failure Penalty",
        out_path=out_dir / "resource_failure_penalty_total.pdf",
        baseline_rows=baseline_run_rows,
        baseline_label=baseline_label,
    )
    _plot_metric_by_x(
        run_rows,
        x_key=x_key,
        x_label=x_label,
        y_key="agent_resource_failure_penalty_total",
        y_label="Resource Failure Penalty",
        out_path=out_dir / "resource_failure_penalty_total_agent_attributed.pdf",
        baseline_rows=baseline_run_rows,
        baseline_label=baseline_label,
    )
    _plot_metric_by_x(
        run_rows,
        x_key=x_key,
        x_label=x_label,
        y_key="hospital_missed_step_penalty_total",
        y_label="Missed-Step Penalty",
        out_path=out_dir / "missed_step_penalty_total.pdf",
        baseline_rows=baseline_run_rows,
        baseline_label=baseline_label,
    )
