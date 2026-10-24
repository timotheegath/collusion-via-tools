from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from experiments.common.plotting.io_utils import (
    ensure_dir,
    infer_labels_from_sweep_dir,
    sanitize_filename,
    write_csv,
    write_json,
)
from experiments.common.plotting.logging_utils import configure_basic_logging
from experiments.common.plotting.load_runs import load_runs

from .build_tables import build_tables
from .plot_overview import plot_overview
from .plot_sweep import plot_sweep_metrics


def _parse_seeds_spec(spec: str) -> Set[int]:
    """
    Accepts comma-separated ints and inclusive ranges like:
      - "0-9"
      - "0-9,11,15-16"
      - "1,2,3"
    """
    out: Set[int] = set()
    s = (spec or "").strip()
    if not s:
        return out
    parts = [p.strip() for p in s.split(",") if p.strip()]
    for p in parts:
        if "-" in p:
            lo_s, _, hi_s = p.partition("-")
            lo = int(lo_s.strip())
            hi = int(hi_s.strip())
            if hi < lo:
                lo, hi = hi, lo
            out.update(range(lo, hi + 1))
        else:
            out.add(int(p))
    return out


def _filter_rows_by_seeds(rows: List[dict], seeds: Optional[Set[int]]) -> List[dict]:
    if not seeds:
        return rows
    out = []
    for r in rows:
        try:
            seed = int(r.get("seed")) if r.get("seed") is not None else None
        except Exception:
            seed = None
        if seed is None:
            continue
        if seed in seeds:
            out.append(r)
    return out


def _filter_rows_by_int(rows: List[dict], key: str, value: int) -> List[dict]:
    out = []
    for r in rows:
        raw = r.get(key)
        if raw is None:
            continue
        try:
            if int(raw) == int(value):
                out.append(r)
        except Exception:
            continue
    return out


def _filter_rows_by_str(rows: List[dict], key: str, value: str) -> List[dict]:
    return [r for r in rows if str(r.get(key)) == str(value)]


def _compute_series_labels(
    sweep_dirs: List[Path],
    metas: List[dict],
    fallback_model_labels: List[Optional[str]],
) -> List[str]:
    """
    Build a unique, human-readable legend label per source sweep.

    Default label is the sweep dir basename. When several selected sweeps share
    the same basename (their legend entries would be indistinguishable), every
    colliding sweep is prefixed with its model label, e.g.
    ``glm5.3-flash/baseline_sweep``. Any remaining collisions (same basename
    *and* same model label) get a numeric suffix (`` (2)``, `` (3)``, ...) so
    the returned list is guaranteed to be unique.
    """
    basenames = [sd.name for sd in sweep_dirs]
    basename_counts: Dict[str, int] = {}
    for b in basenames:
        basename_counts[b] = basename_counts.get(b, 0) + 1

    model_labels: List[str] = []
    for i, meta in enumerate(metas):
        label = str((meta or {}).get("model_label") or "").strip()
        if not label:
            label = str(fallback_model_labels[i] or "").strip()
        model_labels.append(label)

    labels: List[str] = []
    for basename, model_label in zip(basenames, model_labels):
        if basename_counts.get(basename, 0) > 1 and model_label:
            labels.append(f"{model_label}/{basename}")
        else:
            labels.append(basename)

    seen: Dict[str, int] = {}
    unique: List[str] = []
    for label in labels:
        n = seen.get(label, 0) + 1
        seen[label] = n
        unique.append(label if n == 1 else f"{label} ({n})")
    return unique


def _tag_rows_with_series(
    rows: List[dict], run_dir_to_series: Dict[str, Tuple[int, str]]
) -> None:
    """Add ``series_index`` / ``series_label`` columns to each row in-place."""
    for row in rows:
        idx, label = run_dir_to_series.get(str(row.get("run_dir")), (0, "Main sweep"))
        row["series_index"] = idx
        row["series_label"] = label


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate plots for an collusion_via_tools sweep directory."
    )
    sweep_group = parser.add_mutually_exclusive_group(required=True)
    sweep_group.add_argument(
        "--sweep-dir",
        action="append",
        type=str,
        default=None,
        help=(
            "Path like experiments/collusion_via_tools/outputs/<tag>/<ts>/runs/<model>/<sweep_name>. "
            "Pass multiple times to combine several sweeps into a single plot run "
            "(rows from all sweeps are merged and plotted together)."
        ),
    )
    sweep_group.add_argument(
        "--model-runs-dir",
        type=str,
        default=None,
        help=(
            "Path like experiments/collusion_via_tools/outputs/<tag>/<ts>/runs/<model>. "
            "When set, the script finds both sweep dirs inside it."
        ),
    )
    parser.add_argument(
        "--sweep-name",
        type=str,
        default="collusion_via_tools_sweep",
        help="Sweep directory name under --model-runs-dir (default: collusion_via_tools_sweep).",
    )
    parser.add_argument(
        "--benign-sweep-name",
        type=str,
        default="benign_baseline_sweep",
        help="Benign baseline sweep directory name under --model-runs-dir (default: benign_baseline_sweep).",
    )
    parser.add_argument(
        "--benign-sweep-dir",
        "--baseline-sweep-dir",
        dest="benign_sweep_dir",
        type=str,
        default=None,
        help=(
            "Optional sweep dir for a separately-run benign baseline (same sweep layout). "
            "When provided, baseline points are overlaid in the sweep plots."
        ),
    )
    parser.add_argument(
        "--benign-label",
        type=str,
        default="Benign baseline",
        help="Legend label for the benign/baseline overlay.",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help='Optional seed filter like "0-9" or "0-9,11,15-16".',
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory (default: experiments/collusion_via_tools/plots_outputs/<tag>/<ts>/<model>/<sweep_name>)",
    )
    args = parser.parse_args(argv)

    if args.model_runs_dir:
        model_dir = Path(args.model_runs_dir).expanduser().resolve()
        sweep_dir = model_dir / str(args.sweep_name)
        sweep_dirs: List[Path] = [sweep_dir]
        # If caller didn't explicitly pass a benign dir, infer it as a sibling sweep under model_dir.
        if not args.benign_sweep_dir:
            inferred_benign = model_dir / str(args.benign_sweep_name)
            if inferred_benign.exists():
                args.benign_sweep_dir = str(inferred_benign)
    else:
        sweep_dirs = [Path(p).expanduser().resolve() for p in (args.sweep_dir or [])]

    for sd in sweep_dirs:
        if not sd.exists():
            raise FileNotFoundError(f"sweep_dir does not exist: {sd}")

    # Load and merge all sweep dirs. Dedupe by run_id so re-listing the same
    # sweep (or two sweeps that share a run_id) doesn't double-count rows.
    all_runs: List = []
    sources: List[Tuple[Path, dict, int]] = []
    seen_run_ids: Set[str] = set()
    run_dir_to_series_index: Dict[str, int] = {}
    series_run_dirs: List[Set[str]] = []
    for series_index, sd in enumerate(sweep_dirs):
        runs, meta = load_runs(sd)
        kept = 0
        kept_dirs: Set[str] = set()
        for r in runs:
            rid = str(getattr(r, "run_id", "") or r.run_dir.name)
            if rid in seen_run_ids:
                continue
            seen_run_ids.add(rid)
            all_runs.append(r)
            run_dir_to_series_index[str(r.run_dir)] = series_index
            kept_dirs.add(str(r.run_dir))
            kept += 1
        series_run_dirs.append(kept_dirs)
        sources.append((sd, meta, kept))
    tables = build_tables(all_runs)
    meta = sources[-1][1] if sources else {}

    # Series tagging: map each run dir to the sweep it was loaded from so rows
    # from different source sweeps can be plotted as distinguishable series.
    model_label_by_run_dir: Dict[str, Optional[str]] = {}
    for row in tables.run_rows:
        rd = str(row.get("run_dir"))
        if rd not in model_label_by_run_dir:
            model_label_by_run_dir[rd] = row.get("model_label")
    fallback_model_labels: List[Optional[str]] = []
    for kept_dirs in series_run_dirs:
        fallback: Optional[str] = None
        for rd in sorted(kept_dirs):
            candidate = model_label_by_run_dir.get(rd)
            if candidate:
                fallback = str(candidate)
                break
        fallback_model_labels.append(fallback)

    series_labels = _compute_series_labels(
        sweep_dirs, [m for _, m, _ in sources], fallback_model_labels
    )
    run_dir_to_series: Dict[str, Tuple[int, str]] = {
        rd: (idx, series_labels[idx]) for rd, idx in run_dir_to_series_index.items()
    }
    _tag_rows_with_series(tables.run_rows, run_dir_to_series)
    _tag_rows_with_series(tables.agent_rows, run_dir_to_series)

    seed_filter = _parse_seeds_spec(args.seeds) if args.seeds else None
    if seed_filter:
        tables = type(tables)(
            run_rows=_filter_rows_by_seeds(tables.run_rows, seed_filter),
            target_rows=tables.target_rows,
            agent_rows=_filter_rows_by_seeds(tables.agent_rows, seed_filter),
        )

    benign_tables = None
    benign_meta = None
    if args.benign_sweep_dir:
        benign_sweep_dir = Path(args.benign_sweep_dir).expanduser().resolve()
        benign_runs, benign_meta = load_runs(benign_sweep_dir)
        benign_tables = build_tables(benign_runs)
        if seed_filter:
            benign_tables = type(benign_tables)(
                run_rows=_filter_rows_by_seeds(benign_tables.run_rows, seed_filter),
                target_rows=benign_tables.target_rows,
                agent_rows=_filter_rows_by_seeds(benign_tables.agent_rows, seed_filter),
            )

    if args.out_dir:
        out_dir = Path(args.out_dir).expanduser().resolve()
    else:
        labels = infer_labels_from_sweep_dir(sweep_dirs[0])
        if len(sweep_dirs) == 1:
            out_dir = (
                Path("experiments/collusion_via_tools/plots_outputs")
                / labels.experiment_tag
                / labels.timestamp
                / labels.model_label
                / labels.sweep_name
            )
        else:
            # Combined sweeps: keep the timestamp from the first source and
            # stash everything under a single "combined" subdir so the
            # cross-sweep CSVs and plots land in one place.
            out_dir = (
                Path("experiments/collusion_via_tools/plots_outputs")
                / labels.experiment_tag
                / labels.timestamp
                / "combined"
            )
    ensure_dir(out_dir)

    # Diagnostics: detect mismatches between requested sweep params (from run_id)
    # and realized values (from run_config).
    mismatch_counts = {"num_agents": 0, "adversary_count": 0}
    for r in tables.run_rows:
        try:
            req_n = (
                int(r.get("num_agents")) if r.get("num_agents") is not None else None
            )
            act_n = (
                int(r.get("actual_num_agents"))
                if r.get("actual_num_agents") is not None
                else None
            )
        except Exception:
            req_n, act_n = None, None
        if req_n is not None and act_n is not None and req_n != act_n:
            mismatch_counts["num_agents"] += 1
        try:
            req_a = (
                int(r.get("adversary_count"))
                if r.get("adversary_count") is not None
                else None
            )
            act_a = (
                int(r.get("actual_adversary_count"))
                if r.get("actual_adversary_count") is not None
                else None
            )
        except Exception:
            req_a, act_a = None, None
        if req_a is not None and act_a is not None and req_a != act_a:
            mismatch_counts["adversary_count"] += 1

    if mismatch_counts["num_agents"] or mismatch_counts["adversary_count"]:
        import logging

        logger = logging.getLogger(__name__)
        logger.warning("Found requested-vs-actual mismatches: %s", mismatch_counts)

    write_json(
        out_dir / "meta.json",
        {
            "meta": meta,
            "sources": [
                {
                    "sweep_dir": str(sd),
                    "n_runs": kept,
                    "meta": m,
                    "series_index": idx,
                    "series_label": series_labels[idx],
                }
                for idx, (sd, m, kept) in enumerate(sources)
            ],
            "benign_meta": benign_meta,
            "mismatches": mismatch_counts,
            "counts": {
                "runs": len(all_runs),
                "run_rows": len(tables.run_rows),
                "agent_rows": len(tables.agent_rows),
                "benign_runs": len(benign_tables.run_rows) if benign_tables else 0,
            },
        },
        sort_keys=True,
    )
    write_csv(out_dir / "run_rows.csv", tables.run_rows)
    write_csv(out_dir / "agent_rows.csv", tables.agent_rows)
    if benign_tables:
        write_csv(out_dir / "benign_run_rows.csv", benign_tables.run_rows)
        write_csv(out_dir / "benign_agent_rows.csv", benign_tables.agent_rows)

    sweep_out = out_dir / "sweep"
    ensure_dir(sweep_out)
    plot_sweep_metrics(
        run_rows=tables.run_rows,
        agent_rows=tables.agent_rows,
        out_dir=sweep_out,
        baseline_run_rows=benign_tables.run_rows if benign_tables else None,
        baseline_agent_rows=benign_tables.agent_rows if benign_tables else None,
        baseline_label=str(args.benign_label),
    )

    overview_out = out_dir / "overview"
    ensure_dir(overview_out)
    plot_overview(
        run_rows=tables.run_rows,
        benign_run_rows=benign_tables.run_rows if benign_tables else None,
        out_dir=overview_out,
    )

    # Stratified: separate plot sets per sweep "combination".
    # Combination here means (target_role, num_agents), plotted over adversary_count.
    combos = sorted(
        {
            (
                str(r.get("target_role")),
                int(r.get("num_agents")),
            )
            for r in tables.run_rows
            if r.get("target_role") is not None and r.get("num_agents") is not None
        }
    )
    if combos:
        by_combo_root = sweep_out / "by_combo"
        ensure_dir(by_combo_root)
        for role, n_agents in combos:
            combo_run_rows = [
                r
                for r in tables.run_rows
                if str(r.get("target_role")) == role
                and int(r.get("num_agents")) == int(n_agents)
            ]
            combo_agent_rows = [
                r
                for r in tables.agent_rows
                if str(r.get("target_role")) == role
                and int(r.get("num_agents")) == int(n_agents)
            ]
            if not combo_run_rows:
                continue
            combo_out = by_combo_root / sanitize_filename(role) / f"n{int(n_agents)}"
            ensure_dir(combo_out)
            plot_sweep_metrics(
                run_rows=combo_run_rows,
                agent_rows=combo_agent_rows,
                out_dir=combo_out,
                baseline_run_rows=_filter_rows_by_int(
                    benign_tables.run_rows, "num_agents", int(n_agents)
                )
                if benign_tables
                else None,
                baseline_agent_rows=_filter_rows_by_int(
                    benign_tables.agent_rows, "num_agents", int(n_agents)
                )
                if benign_tables
                else None,
                baseline_label=str(args.benign_label),
            )

    # Stratified: separate plot sets per number of agents.
    agent_counts = sorted(
        {
            int(r.get("num_agents"))
            for r in tables.run_rows
            if r.get("num_agents") is not None
        }
    )
    if agent_counts:
        by_agents_root = sweep_out / "by_num_agents"
        ensure_dir(by_agents_root)
        for n_agents in agent_counts:
            n_run_rows = _filter_rows_by_int(tables.run_rows, "num_agents", n_agents)
            n_agent_rows = _filter_rows_by_int(
                tables.agent_rows, "num_agents", n_agents
            )
            if not n_run_rows:
                continue
            n_out = by_agents_root / f"n{n_agents}"
            ensure_dir(n_out)
            plot_sweep_metrics(
                run_rows=n_run_rows,
                agent_rows=n_agent_rows,
                out_dir=n_out,
                baseline_run_rows=_filter_rows_by_int(
                    benign_tables.run_rows, "num_agents", n_agents
                )
                if benign_tables
                else None,
                baseline_agent_rows=_filter_rows_by_int(
                    benign_tables.agent_rows, "num_agents", n_agents
                )
                if benign_tables
                else None,
                baseline_label=str(args.benign_label),
            )

    # NOTE: per-run-config (arm) stratification was removed; runs are now
    # stratified by target_role / num_agents (see by_combo / by_num_agents above).


if __name__ == "__main__":
    configure_basic_logging()
    raise SystemExit(main())
