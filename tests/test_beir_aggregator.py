"""Tests for lib/beir_aggregator.py — multi-run aggregation and tables."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from lib.beir_aggregator import (
    _stats,
    aggregate_runs,
    _collect_per_query,
    compare_configurations,
    load_raw_runs,
    generate_main_table,
    generate_ablation_table,
    _avg_ndcg,
)


def _make_run(
    experiment: str = "test",
    config: str = "bm25",
    seed: int = 42,
    ndcg10: float = 0.35,
    per_query: dict | None = None,
    datasets: list[str] | None = None,
) -> dict:
    """Build a minimal v3 raw run dict for testing."""
    ds_list = datasets or ["nfcorpus"]
    ds = {}
    for name in ds_list:
        ds[name] = {
            "corpus_size": 3633,
            "num_queries": 3,
            "metrics": {
                "ndcg_at_10": ndcg10,
                "recall_at_100": 0.25,
                "map_at_10": 0.12,
                "mrr_at_10": 0.55,
                "precision_at_10": 0.20,
            },
            "per_query_ndcg10": per_query or {"q1": ndcg10, "q2": ndcg10 - 0.1, "q3": ndcg10 + 0.1},
            "timing": {"avg_latency_ms": 0.06, "qps": 16000},
        }
    return {
        "schema_version": 3,
        "experiment": experiment,
        "configuration_name": config,
        "seed": seed,
        "git_commit": "abc1234",
        "strata_version": "0.7.0",
        "datasets": ds,
    }


class TestStats(unittest.TestCase):
    def test_single_value(self):
        s = _stats([0.5])
        self.assertAlmostEqual(s["mean"], 0.5)
        self.assertAlmostEqual(s["std"], 0.0)
        self.assertEqual(s["min"], s["max"])
        self.assertEqual(len(s["values"]), 1)

    def test_multiple_values(self):
        s = _stats([0.3, 0.4, 0.5])
        self.assertAlmostEqual(s["mean"], 0.4, places=4)
        self.assertGreater(s["std"], 0)
        self.assertAlmostEqual(s["min"], 0.3)
        self.assertAlmostEqual(s["max"], 0.5)
        self.assertLessEqual(s["ci95_lo"], s["mean"])
        self.assertGreaterEqual(s["ci95_hi"], s["mean"])

    def test_identical_values(self):
        s = _stats([0.5, 0.5, 0.5])
        self.assertAlmostEqual(s["std"], 0.0)
        self.assertAlmostEqual(s["ci95_lo"], s["ci95_hi"])


class TestAggregateRuns(unittest.TestCase):
    def test_single_run(self):
        runs = [_make_run()]
        agg = aggregate_runs(runs)
        self.assertEqual(agg["num_runs"], 1)
        self.assertIn("nfcorpus", agg["aggregated"])
        self.assertAlmostEqual(
            agg["aggregated"]["nfcorpus"]["ndcg_at_10"]["mean"], 0.35, places=4
        )

    def test_multiple_runs(self):
        runs = [
            _make_run(ndcg10=0.30, seed=1),
            _make_run(ndcg10=0.35, seed=2),
            _make_run(ndcg10=0.40, seed=3),
        ]
        agg = aggregate_runs(runs)
        self.assertEqual(agg["num_runs"], 3)
        self.assertAlmostEqual(
            agg["aggregated"]["nfcorpus"]["ndcg_at_10"]["mean"], 0.35, places=4
        )

    def test_empty_runs_raises(self):
        with self.assertRaises(ValueError):
            aggregate_runs([])

    def test_union_of_datasets(self):
        """If run 1 has ds A and run 2 has ds A+B, both should appear."""
        run1 = _make_run(datasets=["nfcorpus"])
        run2 = _make_run(datasets=["nfcorpus", "scifact"])
        agg = aggregate_runs([run1, run2])
        self.assertIn("nfcorpus", agg["aggregated"])
        self.assertIn("scifact", agg["aggregated"])
        # scifact only in 1 run
        self.assertEqual(len(agg["aggregated"]["scifact"]["ndcg_at_10"]["values"]), 1)

    def test_cross_dataset_average(self):
        runs = [_make_run(datasets=["nfcorpus", "scifact"])]
        agg = aggregate_runs(runs)
        self.assertIn("ndcg_at_10", agg["average_across_datasets"])


class TestCollectPerQuery(unittest.TestCase):
    def test_averages_across_runs(self):
        runs = [
            _make_run(per_query={"q1": 0.3, "q2": 0.5}),
            _make_run(per_query={"q1": 0.5, "q2": 0.7}),
        ]
        scores = _collect_per_query(runs, "nfcorpus")
        self.assertAlmostEqual(scores["q1"], 0.4)
        self.assertAlmostEqual(scores["q2"], 0.6)

    def test_missing_dataset(self):
        runs = [_make_run()]
        scores = _collect_per_query(runs, "nonexistent")
        self.assertEqual(scores, {})


class TestCompareConfigurations(unittest.TestCase):
    def test_basic_comparison(self):
        runs_a = [_make_run(config="hybrid", ndcg10=0.40, per_query={"q1": 0.4, "q2": 0.5, "q3": 0.3})]
        runs_b = [_make_run(config="bm25", ndcg10=0.35, per_query={"q1": 0.35, "q2": 0.45, "q3": 0.25})]
        agg_a = aggregate_runs(runs_a)
        agg_b = aggregate_runs(runs_b)
        result = compare_configurations(
            agg_a, agg_b, runs_a=runs_a, runs_b=runs_b, label="vs_bm25",
        )
        self.assertIn("vs_bm25", result)
        comp = result["vs_bm25"]["nfcorpus"]
        self.assertGreater(comp["ndcg_at_10_delta"], 0)
        self.assertIn("p_value", comp)
        self.assertEqual(comp["test"], "paired_t")

    def test_unequal_run_counts(self):
        """Should work even with different run counts for A and B."""
        runs_a = [_make_run(ndcg10=0.40, seed=i) for i in range(3)]
        runs_b = [_make_run(ndcg10=0.35, seed=i) for i in range(5)]
        agg_a = aggregate_runs(runs_a)
        agg_b = aggregate_runs(runs_b)
        result = compare_configurations(
            agg_a, agg_b, runs_a=runs_a, runs_b=runs_b,
        )
        # Should still produce a p_value (per-query paired test)
        comp = result["vs_baseline"]["nfcorpus"]
        self.assertIn("p_value", comp)


class TestLoadRawRuns(unittest.TestCase):
    def test_loads_matching_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_dir = Path(tmpdir) / "beir" / "raw"
            raw_dir.mkdir(parents=True)

            run = _make_run(experiment="exp1", config="bm25")
            path = raw_dir / "exp1-bm25-run1-2026-03-15T10-00-00-abc1234.json"
            path.write_text(json.dumps(run))

            # Non-matching file
            other = raw_dir / "exp1-hybrid-run1-2026-03-15T10-00-00-abc1234.json"
            other.write_text(json.dumps(_make_run(experiment="exp1", config="hybrid")))

            loaded = load_raw_runs(tmpdir, "exp1", "bm25")
            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded[0]["configuration_name"], "bm25")

    def test_empty_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            loaded = load_raw_runs(tmpdir, "exp1", "bm25")
            self.assertEqual(loaded, [])


class TestTableGeneration(unittest.TestCase):
    def _make_agg(self, config: str, ndcg10: float) -> dict:
        runs = [_make_run(config=config, ndcg10=ndcg10)]
        return aggregate_runs(runs)

    def test_markdown_table(self):
        aggs = [self._make_agg("bm25", 0.35), self._make_agg("hybrid", 0.42)]
        table = generate_main_table(aggs, fmt="markdown")
        self.assertIn("| System |", table)
        self.assertIn("bm25", table)
        self.assertIn("hybrid", table)

    def test_latex_table(self):
        aggs = [self._make_agg("bm25", 0.35)]
        table = generate_main_table(aggs, fmt="latex")
        self.assertIn(r"\begin{table}", table)
        self.assertIn(r"\end{table}", table)

    def test_empty_configs(self):
        self.assertEqual(generate_main_table([]), "")

    def test_ablation_table(self):
        aggs = [self._make_agg("bm25", 0.35), self._make_agg("hybrid", 0.42)]
        table = generate_ablation_table(aggs, fmt="markdown")
        self.assertIn("bm25", table)
        self.assertIn("+", table)  # delta should show +

    def test_ablation_single_config(self):
        """Ablation with only 1 config should return empty string."""
        aggs = [self._make_agg("bm25", 0.35)]
        self.assertEqual(generate_ablation_table(aggs), "")


class TestAvgNdcg(unittest.TestCase):
    def test_from_cross_dataset(self):
        agg = {"average_across_datasets": {"ndcg_at_10": {"mean": 0.45}}, "aggregated": {}}
        self.assertAlmostEqual(_avg_ndcg(agg), 0.45)

    def test_fallback_from_per_dataset(self):
        agg = {
            "average_across_datasets": {},
            "aggregated": {
                "nfcorpus": {"ndcg_at_10": {"mean": 0.3}},
                "scifact": {"ndcg_at_10": {"mean": 0.5}},
            },
        }
        self.assertAlmostEqual(_avg_ndcg(agg), 0.4)

    def test_empty(self):
        self.assertAlmostEqual(_avg_ndcg({"average_across_datasets": {}, "aggregated": {}}), 0.0)


if __name__ == "__main__":
    unittest.main()
