#!/usr/bin/env python3
"""Unified CLI with subcommands for all benchmark suites.

Usage:
    python run.py beir --dataset nfcorpus --mode hybrid
    python run.py ycsb --workload a --records 100000
    python run.py ann --dataset sift-128-euclidean
    python run.py graphalytics --algorithm bfs --dataset example-directed
    python run.py download --bench ann --dataset sift-128-euclidean
    python run.py report --format latex
    python run.py aggregate --experiment ablation --config bm25 hybrid
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from benchmarks import get_benchmarks
from benchmarks.base import BaseBenchmark
from lib import download as dl_mod
from lib import report as report_mod
from lib.recorder import ResultRecorder

ROOT = Path(__file__).resolve().parent


def _run_aggregate(parsed: argparse.Namespace) -> None:
    """Aggregate BEIR multi-run results and optionally generate tables."""
    from lib.beir_aggregator import (
        load_raw_runs,
        aggregate_runs,
        save_aggregated,
        compare_configurations,
        save_tables,
    )

    results_dir = parsed.output_dir
    experiment = parsed.experiment
    all_aggs: list[dict] = []
    all_raw: dict[str, list[dict]] = {}

    for config_name in parsed.config:
        runs = load_raw_runs(results_dir, experiment, config_name)
        if not runs:
            print(f"No v3 runs found for experiment={experiment}, config={config_name}")
            continue
        print(f"Found {len(runs)} run(s) for {config_name}")
        agg = aggregate_runs(runs)
        all_raw[config_name] = runs
        all_aggs.append(agg)

    # Significance testing against baseline
    if parsed.compare_to and len(all_aggs) >= 1:
        baseline_runs = load_raw_runs(results_dir, experiment, parsed.compare_to)
        if baseline_runs:
            baseline_agg = aggregate_runs(baseline_runs)
            for agg in all_aggs:
                if agg["configuration_name"] == parsed.compare_to:
                    continue
                comps = compare_configurations(
                    agg, baseline_agg,
                    runs_a=all_raw.get(agg["configuration_name"]),
                    runs_b=baseline_runs,
                    label=f"vs_{parsed.compare_to}",
                )
                agg["comparisons"].update(comps)

    # Save aggregated results
    for agg in all_aggs:
        save_aggregated(agg, results_dir)

    # Tables
    if parsed.tables and all_aggs:
        save_tables(all_aggs, results_dir, experiment)

    if not all_aggs:
        print("No results to aggregate.")
        sys.exit(1)


def main(argv: list[str] | None = None) -> None:
    # Backward compat: if first arg looks like old-style BEIR invocation
    # (starts with --dataset), inject "beir" subcommand.
    args = argv if argv is not None else sys.argv[1:]
    if args and args[0] == "--dataset":
        args = ["beir"] + args

    parser = argparse.ArgumentParser(
        prog="strata-eval",
        description="Comprehensive benchmarks for StrataDB",
    )
    parser.add_argument(
        "--output-dir", type=str, default=str(ROOT / "results"),
        help="Directory for result JSON files (default: results/)",
    )
    parser.add_argument(
        "--strata-bin", type=str, default=None,
        help="Path to the strata CLI binary (default: auto-detect via PATH / STRATA_BIN)",
    )

    subparsers = parser.add_subparsers(dest="command")

    # Register each benchmark as a subcommand
    benchmarks = get_benchmarks()
    bench_instances: dict[str, BaseBenchmark] = {}
    for name, cls in sorted(benchmarks.items()):
        sub = subparsers.add_parser(name, help=f"Run {name} benchmarks")
        instance = cls()
        instance.register_args(sub)
        bench_instances[name] = instance

    # Download subcommand
    dl_parser = subparsers.add_parser("download", help="Download benchmark datasets")
    dl_mod.register_args(dl_parser)

    # Report subcommand
    report_parser = subparsers.add_parser("report", help="Generate benchmark reports")
    report_mod.register_args(report_parser)

    # Aggregate subcommand (BEIR multi-run aggregation)
    agg_parser = subparsers.add_parser("aggregate", help="Aggregate BEIR multi-run results")
    agg_parser.add_argument(
        "--experiment", type=str, required=True,
        help="Experiment name to aggregate",
    )
    agg_parser.add_argument(
        "--config", nargs="+", required=True,
        help="Configuration name(s) to aggregate",
    )
    agg_parser.add_argument(
        "--compare-to", type=str, default=None,
        help="Baseline configuration for significance testing",
    )
    agg_parser.add_argument(
        "--tables", action="store_true", default=False,
        help="Generate Markdown + LaTeX tables",
    )

    parsed = parser.parse_args(args)

    # If --strata-bin was provided, propagate it via environment variable
    if getattr(parsed, "strata_bin", None):
        import os
        os.environ["STRATA_BIN"] = parsed.strata_bin

    if parsed.command is None:
        parser.print_help()
        return

    if parsed.command == "download":
        dl_mod.run_download(parsed)
        return

    if parsed.command == "report":
        # Wire --output-dir through to report if --results-dir not explicitly set
        if not hasattr(parsed, "results_dir") or parsed.results_dir == "results":
            parsed.results_dir = parsed.output_dir
        report_mod.run_report(parsed)
        return

    if parsed.command == "aggregate":
        _run_aggregate(parsed)
        return

    # Run a benchmark
    bench = bench_instances.get(parsed.command)
    if bench is None:
        parser.print_help()
        return

    if not bench.validate(parsed):
        print(f"Validation failed for {parsed.command}. Check prerequisites.")
        sys.exit(1)

    try:
        results = bench.run(parsed)
    except NotImplementedError as e:
        print(f"\n{parsed.command}: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR running {parsed.command}: {e}", file=sys.stderr)
        sys.exit(1)

    if results:
        recorder = ResultRecorder(category=parsed.command)
        for r in results:
            recorder.record(r)
        recorder.save(parsed.output_dir)


if __name__ == "__main__":
    main()
