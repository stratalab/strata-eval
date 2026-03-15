# BEIR Benchmark & Ablation Study Result Format

This document defines the standard result format for BEIR benchmark runs and ablation studies in strata-eval. The format is designed to support the Strata research paper with reproducible, publication-ready results.

## Reporting Standard

Following the BEIR benchmark standard (Thakur et al., NeurIPS 2021 Datasets & Benchmarks):

- **Primary metric**: nDCG@10 per dataset, averaged across all datasets
- **Secondary metrics**: Recall@100, MAP@10, MRR@10, Precision@10
- **Per-query scores**: Stored for statistical significance testing
- **Multiple runs**: 3-5 runs per configuration with different seeds
- **Statistical significance**: Paired t-test or Wilcoxon signed-rank, with Bonferroni correction for multiple comparisons

---

## Paper Table Formats

### Main Results Table

Rows = systems/configurations, columns = BEIR datasets. Primary metric: nDCG@10.

```
| System          | NFCorpus | SciFact | FiQA  | TREC-COVID | ... | Avg   |
|-----------------|----------|---------|-------|------------|-----|-------|
| Pyserini BM25   | 0.325    | 0.665   | 0.236 | 0.656      | ... | 0.XXX |
| Strata BM25     | 0.XXX    | 0.XXX   | 0.XXX | 0.XXX      | ... | 0.XXX |
| Strata Hybrid   | 0.XXX    | 0.XXX   | 0.XXX | 0.XXX      | ... | 0.XXX |
| + Expansion     | 0.XXX    | 0.XXX   | 0.XXX | 0.XXX      | ... | 0.XXX |
| + Reranking     | 0.XXX    | 0.XXX   | 0.XXX | 0.XXX      | ... | 0.XXX |
| Full Pipeline   | 0.XXX    | 0.XXX   | 0.XXX | 0.XXX      | ... | 0.XXX |
```

Bold = best per column. Significance markers: `*` p<0.05, `**` p<0.01, `***` p<0.001 (vs previous row / baseline).

### Ablation Table (Additive)

Shows cumulative effect of adding each component:

```
| Configuration         | Components Enabled                | nDCG@10 Avg | Delta   | p-value |
|-----------------------|-----------------------------------|-------------|---------|---------|
| BM25                  | BM25 only                         | 0.XXX       | —       | —       |
| + BM25F               | + field-aware scoring             | 0.XXX       | +0.0XX  | <0.01   |
| + Sparse Expansion    | + write-time expansion            | 0.XXX       | +0.0XX  | <0.05   |
| + Vectors (RRF)       | + HNSW + RRF fusion               | 0.XXX       | +0.0XX  | <0.01   |
| + Query Expansion     | + LLM expansion (lex/vec/hyde)    | 0.XXX       | +0.0XX  | <0.05   |
| + Cross-Encoder       | + cross-encoder reranking         | 0.XXX       | +0.0XX  | <0.05   |
| + Graph Boost         | + graph proximity boost           | 0.XXX       | +0.0XX  | <0.05   |
| Full Pipeline         | all components                    | 0.XXX       | +0.0XX  | <0.001  |
```

### Ablation Table (Removal)

Shows effect of removing each component from the full pipeline:

```
| Configuration         | Component Removed          | nDCG@10 Avg | Delta   | p-value |
|-----------------------|----------------------------|-------------|---------|---------|
| Full Pipeline         | —                          | 0.XXX       | —       | —       |
| - Reranking           | cross-encoder reranking    | 0.XXX       | -0.0XX  | <0.01   |
| - Expansion           | LLM query expansion        | 0.XXX       | -0.0XX  | <0.05   |
| - Vectors             | HNSW + RRF (BM25 only)    | 0.XXX       | -0.0XX  | <0.01   |
| - Sparse Expansion    | write-time expansion       | 0.XXX       | -0.0XX  | <0.05   |
| - Graph Boost         | graph proximity boost      | 0.XXX       | -0.0XX  | n.s.    |
```

### Latency Table

```
| Configuration     | Avg Latency (ms) | p50    | p95    | p99    | QPS     |
|-------------------|-------------------|--------|--------|--------|---------|
| BM25 only         | X.X               | X.X    | X.X    | X.X    | XX,XXX  |
| Hybrid            | X.X               | X.X    | X.X    | X.X    | XX,XXX  |
| + Expansion       | XXX.X             | XXX.X  | XXX.X  | XXX.X  | X,XXX   |
| + Reranking       | XXX.X             | XXX.X  | XXX.X  | XXX.X  | X,XXX   |
| Full Pipeline     | X,XXX.X           | X,XXX  | X,XXX  | X,XXX  | XXX     |
```

---

## Result JSON Schema (v3)

### Per-Run Result File

Each benchmark run produces a single JSON file. File naming: `{experiment}-{configuration}-run{N}-{timestamp}-{git_commit}.json`

```json
{
  "schema_version": 3,

  "experiment": "ablation-phase2",
  "configuration_name": "hybrid+expand",
  "run_index": 1,
  "seed": 42,

  "timestamp": "2026-03-15T10:00:00Z",
  "git_commit": "abc1234def5678",
  "git_branch": "main",
  "git_dirty": false,
  "strata_version": "0.7.0",

  "hardware": {
    "cpu": "AMD Ryzen 9 7950X",
    "cores": 32,
    "ram_gb": 64,
    "gpu": "RTX 4070 Super",
    "os": "Linux 6.17.0"
  },

  "configuration": {
    "components": {
      "bm25": true,
      "bm25f": false,
      "proximity_scoring": false,
      "sparse_expansion": false,
      "vectors": true,
      "embedding_model": "miniLM-L6-v2",
      "rrf_k": 60,
      "original_weight": 2.0,
      "top_rank_bonus": [0.05, 0.02],
      "expand": true,
      "expand_types": ["lex", "vec", "hyde"],
      "expand_model": "qwen3:1.7b",
      "strong_signal_skip": true,
      "strong_signal_threshold": 0.85,
      "strong_signal_gap": 0.15,
      "rerank": false,
      "rerank_model": null,
      "rerank_candidates": 20,
      "cross_encoder": false,
      "cross_encoder_model": null,
      "graph_aware": false,
      "graph_weight": 0.0,
      "max_hops": 0
    },
    "tokenizer": {
      "stemmer": "porter",
      "stopwords": "lucene33",
      "preserve_compounds": false
    },
    "scorer": {
      "type": "bm25",
      "k1": 0.9,
      "b": 0.4,
      "title_boost": 1.2,
      "recency_boost": 0.0
    }
  },

  "datasets": {
    "nfcorpus": {
      "corpus_size": 3633,
      "num_queries": 323,

      "metrics": {
        "ndcg_at_10": 0.3462,
        "ndcg_at_100": 0.3208,
        "recall_at_10": 0.1651,
        "recall_at_100": 0.3258,
        "map_at_10": 0.1281,
        "mrr_at_10": 0.5568,
        "precision_at_10": 0.2557
      },

      "per_query_ndcg10": {
        "PLAIN-2": 0.4521,
        "PLAIN-7": 0.1234,
        "PLAIN-12": 0.8901
      },

      "timing": {
        "index_time_s": 0.35,
        "search_time_s": 0.02,
        "avg_latency_ms": 0.06,
        "p50_latency_ms": 0.04,
        "p95_latency_ms": 0.12,
        "p99_latency_ms": 0.31,
        "qps": 16836.3,

        "stage_latencies_ms": {
          "bm25": 0.03,
          "vector_embed": 1.2,
          "vector_search": 0.8,
          "rrf_fusion": 0.001,
          "expand": 12.5,
          "rerank": 0.0,
          "graph_boost": 0.0,
          "blend": 0.0
        }
      }
    }
  },

  "baselines": {
    "pyserini_bm25_flat": {
      "nfcorpus": { "ndcg_at_10": 0.322, "recall_at_100": 0.246 }
    },
    "pyserini_bm25_mf": {
      "nfcorpus": { "ndcg_at_10": 0.325, "recall_at_100": 0.250 }
    }
  }
}
```

### Field Descriptions

| Field | Type | Description |
|-------|------|-------------|
| `schema_version` | int | Always `3` for this format |
| `experiment` | string | Groups related runs (e.g., "baseline-v0.7", "ablation-phase2") |
| `configuration_name` | string | Human-readable name for this config (e.g., "hybrid+expand") |
| `run_index` | int | 1-indexed run number within an experiment |
| `seed` | int | Random seed for reproducibility |
| `configuration.components` | object | Exactly which pipeline stages were enabled — the ablation knobs |
| `configuration.tokenizer` | object | Tokenizer settings (stemmer, stopwords, compound handling) |
| `configuration.scorer` | object | Scoring function and parameters |
| `datasets.*.metrics` | object | Standard BEIR metrics at cutoff points |
| `datasets.*.per_query_ndcg10` | object | Per-query nDCG@10 for significance testing |
| `datasets.*.timing.stage_latencies_ms` | object | Per-component latency breakdown |
| `baselines` | object | Published baseline numbers for comparison |

---

## Aggregated Result Schema

After N runs per configuration, an aggregation step produces a summary file. File naming: `{experiment}-{configuration}-aggregated-{timestamp}.json`

```json
{
  "schema_version": 3,
  "experiment": "ablation-phase2",
  "configuration_name": "hybrid+expand",
  "num_runs": 5,
  "seeds": [42, 43, 44, 45, 46],
  "git_commit": "abc1234def5678",
  "strata_version": "0.7.0",

  "aggregated": {
    "nfcorpus": {
      "ndcg_at_10": {
        "mean": 0.3462,
        "std": 0.0012,
        "ci95_lo": 0.3441,
        "ci95_hi": 0.3483,
        "median": 0.3460,
        "min": 0.3445,
        "max": 0.3479,
        "values": [0.3451, 0.3460, 0.3462, 0.3479, 0.3458]
      },
      "recall_at_100": {
        "mean": 0.3258,
        "std": 0.0008,
        "ci95_lo": 0.3245,
        "ci95_hi": 0.3271
      },
      "avg_latency_ms": {
        "mean": 0.06,
        "std": 0.003,
        "p50": 0.04,
        "p95": 0.12,
        "p99": 0.31
      }
    }
  },

  "average_across_datasets": {
    "ndcg_at_10": {
      "mean": 0.XXXX,
      "std": 0.XXXX,
      "ci95_lo": 0.XXXX,
      "ci95_hi": 0.XXXX
    }
  },

  "comparisons": {
    "vs_bm25_baseline": {
      "nfcorpus": {
        "ndcg_at_10_delta": 0.0242,
        "ndcg_at_10_relative_pct": 7.45,
        "p_value": 0.003,
        "test": "paired_t",
        "effect_size_d": 0.45,
        "significant_at_005": true,
        "significant_at_001": true,
        "significant_at_0001": false
      }
    },
    "vs_hybrid_baseline": {
      "nfcorpus": {
        "ndcg_at_10_delta": 0.0081,
        "ndcg_at_10_relative_pct": 2.40,
        "p_value": 0.041,
        "test": "paired_t",
        "effect_size_d": 0.22,
        "significant_at_005": true,
        "significant_at_001": false,
        "significant_at_0001": false
      }
    }
  }
}
```

### Statistical Methods

| Method | When to Use | Implementation |
|--------|-------------|----------------|
| Paired t-test | Per-query nDCG@10 differences between two systems | `scipy.stats.ttest_rel` on per-query score vectors |
| Wilcoxon signed-rank | When normality assumption is violated | `scipy.stats.wilcoxon` as alternative to paired t |
| Cohen's d | Effect size for improvements | `(mean_A - mean_B) / pooled_std` |
| 95% CI | Confidence intervals on mean metrics | `mean +/- 1.96 * std / sqrt(n)` |
| Bonferroni correction | Multiple comparisons across datasets | `alpha / num_datasets` for family-wise error rate |
| Holm-Bonferroni | Less conservative multiple comparison correction | Sequential rejection with ordered p-values |

---

## Ablation Configuration Matrix

Each row in the ablation study is a named configuration. The full matrix:

### Additive Ablation (build up from BM25)

| Config Name | bm25 | bm25f | sparse_exp | vectors | expand | rerank | cross_enc | graph |
|-------------|-------|-------|------------|---------|--------|--------|-----------|-------|
| `bm25` | Yes | — | — | — | — | — | — | — |
| `bm25f` | Yes | Yes | — | — | — | — | — | — |
| `bm25f+sparse` | Yes | Yes | Yes | — | — | — | — | — |
| `hybrid` | Yes | Yes | Yes | Yes | — | — | — | — |
| `hybrid+expand` | Yes | Yes | Yes | Yes | Yes | — | — | — |
| `hybrid+xenc` | Yes | Yes | Yes | Yes | — | — | Yes | — |
| `hybrid+expand+xenc` | Yes | Yes | Yes | Yes | Yes | — | Yes | — |
| `hybrid+expand+llm-rerank` | Yes | Yes | Yes | Yes | Yes | Yes | — | — |
| `full` | Yes | Yes | Yes | Yes | Yes | — | Yes | Yes |

### Removal Ablation (remove from full pipeline)

| Config Name | Removed Component | Expected Effect |
|-------------|-------------------|-----------------|
| `full` | — | Baseline (best) |
| `full-graph` | graph boost | Measures graph contribution |
| `full-xenc` | cross-encoder | Measures reranker contribution |
| `full-expand` | query expansion | Measures expansion contribution |
| `full-vectors` | HNSW + RRF | Measures vector contribution |
| `full-sparse` | write-time expansion | Measures sparse expansion contribution |
| `full-bm25f` | field-aware scoring | Measures BM25F contribution |

### Parameter Sensitivity

| Sweep | Values | Metric |
|-------|--------|--------|
| RRF k | [20, 40, 60, 80, 100] | nDCG@10 |
| Original weight | [1.0, 1.5, 2.0, 2.5, 3.0] | nDCG@10 |
| BM25 k1 | [0.5, 0.9, 1.2, 1.5, 2.0] | nDCG@10 |
| BM25 b | [0.2, 0.4, 0.6, 0.75, 0.9] | nDCG@10 |
| Embedding model | [miniLM-L6, miniLM-L12, nomic, bge-small] | nDCG@10 |
| Rerank candidates | [10, 20, 30, 50] | nDCG@10 |
| Graph max_hops | [1, 2, 3] | nDCG@10 |
| Graph weight | [0.1, 0.2, 0.3, 0.5] | nDCG@10 |

---

## BEIR Datasets

All 15 BEIR datasets used, grouped by domain:

| Dataset | Domain | Docs | Queries | Notes |
|---------|--------|------|---------|-------|
| msmarco | Web passages | 8.84M | 43 | Keyword-only (small query set) |
| nfcorpus | Nutrition/medical | 3.6K | 323 | Small, domain-specific |
| scifact | Scientific claims | 5K | 300 | Precision-focused (fact verification) |
| arguana | Argument retrieval | 8.7K | 1,406 | Counterargument retrieval |
| scidocs | Scientific papers | 25K | 1,000 | Citation prediction |
| trec-covid | COVID-19 literature | 171K | 50 | Biomedical, small query set |
| fiqa | Financial QA | 57K | 648 | Domain-specific QA |
| quora | Community QA | 523K | 10,000 | Duplicate question detection |
| webis-touche2020 | Argument retrieval | 382K | 49 | Argumentative search |
| cqadupstack | Stack Exchange | 457K | 13,145 | 12 subforums, macro-averaged |
| fever | Fact verification | 5.42M | 6,666 | Large-scale verification |
| climate-fever | Climate claims | 5.42M | 1,535 | Climate-specific FEVER |
| nq | Natural Questions | 2.68M | 3,452 | Google search questions |
| hotpotqa | Multi-hop QA | 5.23M | 7,405 | Multi-hop reasoning |
| dbpedia-entity | Entity linking | 4.63M | 400 | Entity search |

---

## File Organization

```
results/
  beir/
    raw/
      ablation-phase2-bm25-run1-2026-03-15T10-00-00-abc1234.json
      ablation-phase2-bm25-run2-2026-03-15T10-05-00-abc1234.json
      ablation-phase2-bm25-run3-2026-03-15T10-10-00-abc1234.json
      ablation-phase2-hybrid-run1-2026-03-15T11-00-00-abc1234.json
      ...
    aggregated/
      ablation-phase2-bm25-aggregated-2026-03-15.json
      ablation-phase2-hybrid-aggregated-2026-03-15.json
      ablation-phase2-hybrid+expand-aggregated-2026-03-15.json
      ...
    tables/
      ablation-phase2-main-results.md
      ablation-phase2-main-results.tex
      ablation-phase2-additive-ablation.md
      ablation-phase2-additive-ablation.tex
      ablation-phase2-removal-ablation.md
      ablation-phase2-removal-ablation.tex
      ablation-phase2-latency.md
      ablation-phase2-latency.tex
```

---

## Migration from Schema v2

Current strata-eval results use a legacy format (no `schema_version` field) or schema v2 (`BenchmarkResult` dataclass). To migrate:

1. Existing results remain in `results/` (legacy) and `scratch/results/` (older)
2. New BEIR runs use schema v3 and write to `results/beir/raw/`
3. The report generator handles both formats (already supports legacy detection)
4. No need to re-run old benchmarks — legacy results serve as historical reference

Key additions in v3 over v2:
- `configuration.components` — per-component toggle tracking (the ablation knobs)
- `run_index` + `seed` — multi-run support
- `per_query_ndcg10` — required (was optional in legacy)
- `timing.stage_latencies_ms` — per-component latency breakdown
- `timing.p50/p95/p99_latency_ms` — latency percentiles
- `configuration.tokenizer` and `configuration.scorer` — full parameter capture
