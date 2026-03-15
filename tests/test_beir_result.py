"""Tests for lib/beir_result.py — v3 result builder."""

from __future__ import annotations

import json
import os
import tempfile
import unittest

from lib.beir_result import BeirRunResult, components_from_mode, _DEFAULT_COMPONENTS


class TestComponentsFromMode(unittest.TestCase):
    def test_keyword_mode(self):
        c = components_from_mode("keyword")
        self.assertTrue(c["bm25"])
        self.assertFalse(c["vectors"])
        self.assertFalse(c["expand"])
        self.assertFalse(c["rerank"])
        self.assertIsNone(c["embedding_model"])

    def test_hybrid_mode(self):
        c = components_from_mode("hybrid", embed_model="miniLM")
        self.assertTrue(c["bm25"])
        self.assertTrue(c["vectors"])
        self.assertEqual(c["embedding_model"], "miniLM")
        self.assertFalse(c["expand"])
        self.assertFalse(c["rerank"])

    def test_hybrid_with_expand(self):
        c = components_from_mode("hybrid", expand=True)
        self.assertTrue(c["vectors"])
        self.assertTrue(c["expand"])
        self.assertEqual(c["expand_types"], ["lex", "vec", "hyde"])
        self.assertFalse(c["rerank"])

    def test_hybrid_llm_mode(self):
        c = components_from_mode("hybrid-llm")
        self.assertTrue(c["vectors"])
        self.assertTrue(c["expand"])
        self.assertTrue(c["rerank"])

    def test_returns_independent_copy(self):
        """Two calls should return independent dicts."""
        c1 = components_from_mode("keyword")
        c2 = components_from_mode("keyword")
        c1["bm25"] = False
        self.assertTrue(c2["bm25"])

    def test_does_not_mutate_defaults(self):
        """components_from_mode must not mutate _DEFAULT_COMPONENTS."""
        original_expand_types = list(_DEFAULT_COMPONENTS["expand_types"])
        components_from_mode("hybrid", expand=True)
        self.assertEqual(_DEFAULT_COMPONENTS["expand_types"], original_expand_types)


class TestBeirRunResult(unittest.TestCase):
    def test_empty_result_structure(self):
        r = BeirRunResult(experiment="test", configuration_name="bm25")
        d = r.to_dict()
        self.assertEqual(d["schema_version"], 3)
        self.assertEqual(d["experiment"], "test")
        self.assertEqual(d["configuration_name"], "bm25")
        self.assertEqual(d["run_index"], 1)
        self.assertEqual(d["seed"], 42)
        self.assertIsInstance(d["hardware"], dict)
        self.assertIsInstance(d["configuration"], dict)
        self.assertEqual(d["datasets"], {})

    def test_add_dataset(self):
        r = BeirRunResult()
        r.add_dataset(
            "nfcorpus",
            corpus_size=3633,
            num_queries=323,
            metrics={"ndcg_at_10": 0.35},
            per_query_ndcg10={"q1": 0.4, "q2": 0.3},
            timing={"avg_latency_ms": 0.5, "qps": 2000},
        )
        d = r.to_dict()
        self.assertIn("nfcorpus", d["datasets"])
        ds = d["datasets"]["nfcorpus"]
        self.assertEqual(ds["corpus_size"], 3633)
        self.assertEqual(ds["metrics"]["ndcg_at_10"], 0.35)
        self.assertEqual(len(ds["per_query_ndcg10"]), 2)

    def test_add_dataset_with_baselines(self):
        r = BeirRunResult()
        r.add_dataset(
            "scifact",
            corpus_size=5000,
            num_queries=300,
            metrics={"ndcg_at_10": 0.65},
            per_query_ndcg10={},
            timing={},
            baselines={
                "pyserini_bm25_flat": {"ndcg_at_10": 0.672},
                "pyserini_bm25_mf": {"ndcg_at_10": 0.665},
            },
        )
        d = r.to_dict()
        self.assertEqual(d["baselines"]["pyserini_bm25_flat"]["scifact"]["ndcg_at_10"], 0.672)

    def test_set_components(self):
        r = BeirRunResult()
        r.set_components({"bm25": True, "vectors": True})
        d = r.to_dict()
        self.assertEqual(d["configuration"]["components"]["vectors"], True)

    def test_save_and_load(self):
        r = BeirRunResult(experiment="test-exp", configuration_name="bm25", run_index=1)
        r.add_dataset(
            "nfcorpus",
            corpus_size=100,
            num_queries=10,
            metrics={"ndcg_at_10": 0.3},
            per_query_ndcg10={"q1": 0.5},
            timing={"qps": 100},
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = r.save(tmpdir)
            self.assertTrue(path.exists())
            self.assertTrue(path.name.startswith("test-exp-bm25-run1-"))
            with open(path) as f:
                loaded = json.load(f)
            self.assertEqual(loaded["schema_version"], 3)
            self.assertIn("nfcorpus", loaded["datasets"])

    def test_git_state_captured_at_construction(self):
        """git_branch and git_dirty should be captured once, not re-evaluated."""
        r = BeirRunResult()
        d1 = r.to_dict()
        d2 = r.to_dict()
        # Both calls should return the same git state
        self.assertEqual(d1["git_branch"], d2["git_branch"])
        self.assertEqual(d1["git_dirty"], d2["git_dirty"])

    def test_default_config_name(self):
        r = BeirRunResult(configuration_name=None)
        self.assertEqual(r.configuration_name, "unknown")


class TestDeriveConfigName(unittest.TestCase):
    """Test the runner's _derive_config_name helper."""

    def test_keyword(self):
        from benchmarks.beir.runner import _derive_config_name
        import argparse
        args = argparse.Namespace(config_name=None, expand=False, rerank=False)
        self.assertEqual(_derive_config_name("keyword", args), "keyword")

    def test_hybrid_with_expand(self):
        from benchmarks.beir.runner import _derive_config_name
        import argparse
        args = argparse.Namespace(config_name=None, expand=True, rerank=False)
        self.assertEqual(_derive_config_name("hybrid", args), "hybrid+expand")

    def test_hybrid_llm(self):
        from benchmarks.beir.runner import _derive_config_name
        import argparse
        args = argparse.Namespace(config_name=None, expand=False, rerank=False)
        self.assertEqual(_derive_config_name("hybrid-llm", args), "hybrid+expand+rerank")

    def test_explicit_name_overrides(self):
        from benchmarks.beir.runner import _derive_config_name
        import argparse
        args = argparse.Namespace(config_name="my-custom", expand=True, rerank=True)
        self.assertEqual(_derive_config_name("hybrid", args), "my-custom")


if __name__ == "__main__":
    unittest.main()
