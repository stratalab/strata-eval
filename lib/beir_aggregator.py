"""BEIR result aggregation — multi-run stats, significance tests, table generation.

Reads v3 raw result files, computes per-dataset and cross-dataset statistics,
runs paired significance tests between configurations, and outputs aggregated
JSON plus Markdown/LaTeX tables.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_raw_runs(
    results_dir: str | Path,
    experiment: str,
    configuration_name: str,
) -> list[dict]:
    """Load all v3 raw result files matching the given experiment + config."""
    raw_dir = Path(results_dir) / "beir" / "raw"
    if not raw_dir.exists():
        return []

    runs = []
    prefix = f"{experiment}-{configuration_name}-run"
    for path in sorted(raw_dir.glob("*.json")):
        if not path.name.startswith(prefix):
            continue
        with open(path) as f:
            data = json.load(f)
        if data.get("schema_version") != 3:
            continue
        if data.get("experiment") != experiment:
            continue
        if data.get("configuration_name") != configuration_name:
            continue
        runs.append(data)
    return runs


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _stats(values: list[float]) -> dict:
    """Compute summary statistics for a list of values."""
    arr = np.array(values, dtype=float)
    n = len(arr)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    ci_half = 1.96 * std / math.sqrt(n) if n > 0 else 0.0
    return {
        "mean": round(mean, 6),
        "std": round(std, 6),
        "ci95_lo": round(mean - ci_half, 6),
        "ci95_hi": round(mean + ci_half, 6),
        "median": round(float(np.median(arr)), 6),
        "min": round(float(np.min(arr)), 6),
        "max": round(float(np.max(arr)), 6),
        "values": [round(v, 6) for v in arr.tolist()],
    }


_AGG_METRICS = [
    "ndcg_at_10", "ndcg_at_100",
    "recall_at_10", "recall_at_100",
    "map_at_10", "mrr_at_10", "precision_at_10",
]

_AGG_TIMING = ["avg_latency_ms", "qps"]


def aggregate_runs(runs: list[dict]) -> dict:
    """Aggregate multiple v3 raw runs into a summary dict."""
    if not runs:
        raise ValueError("No runs to aggregate")

    first = runs[0]
    dataset_names = sorted(first["datasets"].keys())

    aggregated: dict[str, dict] = {}
    for ds_name in dataset_names:
        ds_agg: dict[str, dict] = {}
        for metric in _AGG_METRICS:
            values = [
                r["datasets"][ds_name]["metrics"].get(metric, 0.0)
                for r in runs if ds_name in r["datasets"]
            ]
            if values:
                ds_agg[metric] = _stats(values)
        for timing_key in _AGG_TIMING:
            values = [
                r["datasets"][ds_name]["timing"].get(timing_key, 0.0)
                for r in runs if ds_name in r["datasets"]
            ]
            if values:
                ds_agg[timing_key] = _stats(values)
        aggregated[ds_name] = ds_agg

    # Cross-dataset average for primary metrics
    avg_across: dict[str, dict] = {}
    for metric in _AGG_METRICS:
        # For each run, compute the mean across datasets, then aggregate
        per_run_means = []
        for r in runs:
            ds_values = [
                r["datasets"][ds]["metrics"].get(metric, 0.0)
                for ds in dataset_names if ds in r["datasets"]
            ]
            if ds_values:
                per_run_means.append(float(np.mean(ds_values)))
        if per_run_means:
            avg_across[metric] = _stats(per_run_means)

    return {
        "schema_version": 3,
        "experiment": first["experiment"],
        "configuration_name": first["configuration_name"],
        "num_runs": len(runs),
        "seeds": [r.get("seed", 0) for r in runs],
        "git_commit": first.get("git_commit", "unknown"),
        "strata_version": first.get("strata_version", "unknown"),
        "aggregated": aggregated,
        "average_across_datasets": avg_across,
        "comparisons": {},
    }


# ---------------------------------------------------------------------------
# Significance testing
# ---------------------------------------------------------------------------

def compare_configurations(
    agg_a: dict,
    agg_b: dict,
    *,
    runs_a: list[dict] | None = None,
    runs_b: list[dict] | None = None,
    label: str = "vs_baseline",
) -> dict:
    """Compare two aggregated results. Uses per-query paired t-test when
    raw runs are provided, otherwise falls back to comparing means."""
    from scipy.stats import ttest_rel

    dataset_names = sorted(
        set(agg_a["aggregated"].keys()) & set(agg_b["aggregated"].keys())
    )

    comparisons: dict[str, dict] = {}
    for ds_name in dataset_names:
        a_ndcg = agg_a["aggregated"][ds_name].get("ndcg_at_10", {})
        b_ndcg = agg_b["aggregated"][ds_name].get("ndcg_at_10", {})
        mean_a = a_ndcg.get("mean", 0.0)
        mean_b = b_ndcg.get("mean", 0.0)
        delta = mean_a - mean_b
        relative_pct = (delta / mean_b * 100) if mean_b != 0 else 0.0

        p_value = None
        effect_d = None
        test_name = None

        # Try per-query paired t-test if raw runs available
        if runs_a and runs_b and len(runs_a) == len(runs_b):
            scores_a = _collect_per_query(runs_a, ds_name)
            scores_b = _collect_per_query(runs_b, ds_name)
            common_qids = sorted(set(scores_a.keys()) & set(scores_b.keys()))
            if len(common_qids) >= 2:
                vec_a = np.array([scores_a[q] for q in common_qids])
                vec_b = np.array([scores_b[q] for q in common_qids])
                t_stat, p_value = ttest_rel(vec_a, vec_b)
                p_value = float(p_value)
                pooled_std = float(np.std(vec_a - vec_b, ddof=1))
                if pooled_std > 0:
                    effect_d = round(float(np.mean(vec_a - vec_b)) / pooled_std, 4)
                test_name = "paired_t"

        comp = {
            "ndcg_at_10_delta": round(delta, 6),
            "ndcg_at_10_relative_pct": round(relative_pct, 2),
        }
        if p_value is not None:
            comp["p_value"] = round(p_value, 6)
            comp["test"] = test_name
            comp["effect_size_d"] = effect_d
            comp["significant_at_005"] = p_value < 0.05
            comp["significant_at_001"] = p_value < 0.01
            comp["significant_at_0001"] = p_value < 0.001
        comparisons[ds_name] = comp

    return {label: comparisons}


def _collect_per_query(runs: list[dict], dataset: str) -> dict[str, float]:
    """Average per-query nDCG@10 across runs for a dataset."""
    all_scores: dict[str, list[float]] = {}
    for r in runs:
        ds = r.get("datasets", {}).get(dataset, {})
        pq = ds.get("per_query_ndcg10", {})
        for qid, score in pq.items():
            all_scores.setdefault(qid, []).append(float(score))
    return {qid: float(np.mean(scores)) for qid, scores in all_scores.items()}


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_aggregated(agg: dict, results_dir: str | Path) -> Path:
    """Write aggregated JSON to results/beir/aggregated/."""
    out_dir = Path(results_dir) / "beir" / "aggregated"
    out_dir.mkdir(parents=True, exist_ok=True)

    from datetime import date
    filename = (
        f"{agg['experiment']}-{agg['configuration_name']}"
        f"-aggregated-{date.today().isoformat()}.json"
    )
    path = out_dir / filename

    fd, tmp_path = tempfile.mkstemp(dir=out_dir, suffix=".json.tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(agg, f, indent=2)
        os.rename(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise

    print(f"Aggregated result saved to {path}")
    return path


# ---------------------------------------------------------------------------
# Table generation
# ---------------------------------------------------------------------------

def generate_main_table(
    aggregated_configs: list[dict],
    *,
    fmt: str = "markdown",
    baselines: dict | None = None,
) -> str:
    """Generate the main results table (configs × datasets, nDCG@10)."""
    if not aggregated_configs:
        return ""

    # Collect all dataset names across configs
    all_datasets: list[str] = []
    for agg in aggregated_configs:
        for ds in agg.get("aggregated", {}):
            if ds not in all_datasets:
                all_datasets.append(ds)
    all_datasets.sort()

    rows: list[tuple[str, dict[str, float]]] = []

    # Add baseline rows if provided
    if baselines:
        from benchmarks.beir.config import PYSERINI_BASELINES
        for bl_name, bl_key in [("Pyserini BM25 flat", "bm25_flat"), ("Pyserini BM25 mf", "bm25_mf")]:
            ds_scores = {}
            for ds in all_datasets:
                bl = PYSERINI_BASELINES.get(ds, {}).get(bl_key, {})
                ds_scores[ds] = bl.get("NDCG@10", 0.0)
            rows.append((bl_name, ds_scores))

    # Add each configuration
    for agg in aggregated_configs:
        name = agg["configuration_name"]
        ds_scores = {}
        for ds in all_datasets:
            ds_agg = agg.get("aggregated", {}).get(ds, {})
            ndcg = ds_agg.get("ndcg_at_10", {})
            ds_scores[ds] = ndcg.get("mean", 0.0)
        rows.append((name, ds_scores))

    if fmt == "latex":
        return _table_latex(all_datasets, rows)
    return _table_markdown(all_datasets, rows)


def _table_markdown(datasets: list[str], rows: list[tuple[str, dict[str, float]]]) -> str:
    # Short dataset names for header
    short = {d: d.replace("-", "").replace("_", "")[:8] for d in datasets}

    header = "| System | " + " | ".join(short[d] for d in datasets) + " | Avg |"
    sep = "|---|" + "|".join("---:" for _ in datasets) + "|---:|"
    lines = [header, sep]

    for name, scores in rows:
        vals = [scores.get(d, 0.0) for d in datasets]
        avg = sum(vals) / len(vals) if vals else 0.0
        cells = " | ".join(f"{v:.3f}" for v in vals)
        lines.append(f"| {name} | {cells} | {avg:.3f} |")

    return "\n".join(lines)


def _table_latex(datasets: list[str], rows: list[tuple[str, dict[str, float]]]) -> str:
    short = {d: d.replace("-", "\\mbox{-}").replace("_", "\\_")[:12] for d in datasets}
    col_spec = "l" + "c" * len(datasets) + "c"

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{BEIR nDCG@10 Results}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        "System & " + " & ".join(short[d] for d in datasets) + r" & Avg \\",
        r"\midrule",
    ]
    for name, scores in rows:
        vals = [scores.get(d, 0.0) for d in datasets]
        avg = sum(vals) / len(vals) if vals else 0.0
        escaped = name.replace("_", r"\_").replace("+", r"{+}")
        cells = " & ".join(f"{v:.3f}" for v in vals)
        lines.append(f"{escaped} & {cells} & {avg:.3f} \\\\")

    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines)


def generate_ablation_table(
    aggregated_configs: list[dict],
    *,
    fmt: str = "markdown",
) -> str:
    """Generate an additive ablation table showing delta vs. first config."""
    if len(aggregated_configs) < 2:
        return ""

    baseline = aggregated_configs[0]
    bl_avg = _avg_ndcg(baseline)

    if fmt == "latex":
        lines = [
            r"\begin{table}[t]", r"\centering",
            r"\caption{Ablation Study (Additive)}", r"\begin{tabular}{lccc}",
            r"\toprule",
            r"Configuration & nDCG@10 Avg & Delta & Relative \\",
            r"\midrule",
        ]
        for agg in aggregated_configs:
            name = agg["configuration_name"].replace("_", r"\_").replace("+", r"{+}")
            avg = _avg_ndcg(agg)
            delta = avg - bl_avg
            rel = (delta / bl_avg * 100) if bl_avg != 0 else 0
            d_str = f"+{delta:.4f}" if delta >= 0 else f"{delta:.4f}"
            r_str = f"+{rel:.1f}\\%" if delta >= 0 else f"{rel:.1f}\\%"
            if agg is baseline:
                d_str = "---"
                r_str = "---"
            lines.append(f"{name} & {avg:.4f} & {d_str} & {r_str} \\\\")
        lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
        return "\n".join(lines)

    # Markdown
    lines = [
        "| Configuration | nDCG@10 Avg | Delta | Relative |",
        "|---|---:|---:|---:|",
    ]
    for agg in aggregated_configs:
        name = agg["configuration_name"]
        avg = _avg_ndcg(agg)
        delta = avg - bl_avg
        rel = (delta / bl_avg * 100) if bl_avg != 0 else 0
        d_str = f"+{delta:.4f}" if delta >= 0 else f"{delta:.4f}"
        r_str = f"+{rel:.1f}%" if delta >= 0 else f"{rel:.1f}%"
        if agg is baseline:
            d_str = "---"
            r_str = "---"
        lines.append(f"| {name} | {avg:.4f} | {d_str} | {r_str} |")
    return "\n".join(lines)


def _avg_ndcg(agg: dict) -> float:
    """Mean nDCG@10 across all datasets in an aggregated result."""
    cross = agg.get("average_across_datasets", {}).get("ndcg_at_10", {})
    if "mean" in cross:
        return cross["mean"]
    # Fallback: compute from per-dataset
    vals = []
    for ds_agg in agg.get("aggregated", {}).values():
        ndcg = ds_agg.get("ndcg_at_10", {})
        if "mean" in ndcg:
            vals.append(ndcg["mean"])
    return float(np.mean(vals)) if vals else 0.0


def save_tables(
    aggregated_configs: list[dict],
    results_dir: str | Path,
    experiment: str,
) -> list[Path]:
    """Generate and save Markdown + LaTeX tables."""
    table_dir = Path(results_dir) / "beir" / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    for fmt, ext in [("markdown", "md"), ("latex", "tex")]:
        main = generate_main_table(aggregated_configs, fmt=fmt, baselines=True)
        if main:
            p = table_dir / f"{experiment}-main-results.{ext}"
            p.write_text(main)
            paths.append(p)
            print(f"  Table: {p}")

        ablation = generate_ablation_table(aggregated_configs, fmt=fmt)
        if ablation:
            p = table_dir / f"{experiment}-additive-ablation.{ext}"
            p.write_text(ablation)
            paths.append(p)
            print(f"  Table: {p}")

    return paths
