"""BEIR result schema v3 — per-run result builder and writer.

Each run produces a single JSON file containing results for all evaluated
datasets, full pipeline configuration, per-query scores for significance
testing, and timing breakdowns.
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from .system_info import capture_hardware, get_sdk_version, git_branch, git_is_dirty, git_short_commit


def _json_default(obj: object) -> object:
    type_name = type(obj).__module__
    if type_name == "numpy":
        import numpy as np
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


# Default component configuration (all off except bm25).
_DEFAULT_COMPONENTS = {
    "bm25": True,
    "bm25f": False,
    "proximity_scoring": False,
    "sparse_expansion": False,
    "vectors": False,
    "embedding_model": None,
    "rrf_k": 60,
    "original_weight": 2.0,
    "top_rank_bonus": [0.05, 0.02],
    "expand": False,
    "expand_types": [],
    "expand_model": None,
    "strong_signal_skip": True,
    "strong_signal_threshold": 0.85,
    "strong_signal_gap": 0.15,
    "rerank": False,
    "rerank_model": None,
    "rerank_candidates": 20,
    "cross_encoder": False,
    "cross_encoder_model": None,
    "graph_aware": False,
    "graph_weight": 0.0,
    "max_hops": 0,
}

_DEFAULT_TOKENIZER = {
    "stemmer": "porter",
    "stopwords": "lucene33",
    "preserve_compounds": False,
}

_DEFAULT_SCORER = {
    "type": "bm25",
    "k1": 0.9,
    "b": 0.4,
    "title_boost": 1.2,
    "recency_boost": 0.0,
}


def components_from_mode(
    mode: str,
    *,
    embed_model: str = "miniLM",
    expand: bool = False,
    rerank: bool = False,
) -> dict:
    """Derive component flags from the search mode and CLI flags."""
    components = dict(_DEFAULT_COMPONENTS)
    if mode in ("hybrid", "hybrid-llm"):
        components["vectors"] = True
        components["embedding_model"] = embed_model
    if expand or mode == "hybrid-llm":
        components["expand"] = True
        components["expand_types"] = ["lex", "vec", "hyde"]
        components["expand_model"] = os.environ.get("STRATA_MODEL_NAME")
    if rerank or mode == "hybrid-llm":
        components["rerank"] = True
        components["rerank_model"] = os.environ.get("STRATA_MODEL_NAME")
    return components


class BeirRunResult:
    """Accumulates per-dataset BEIR results and writes a v3 JSON file."""

    def __init__(
        self,
        experiment: str = "default",
        configuration_name: str | None = None,
        run_index: int = 1,
        seed: int = 42,
    ) -> None:
        self.experiment = experiment
        self.configuration_name = configuration_name or "unknown"
        self.run_index = run_index
        self.seed = seed

        now = datetime.now(timezone.utc)
        self._timestamp = now.isoformat()
        self._timestamp_slug = now.strftime("%Y-%m-%dT%H-%M-%S")
        self._git_commit = git_short_commit() or "unknown"

        self._hardware = capture_hardware()
        self._strata_version = get_sdk_version()

        self._configuration: dict = {
            "components": dict(_DEFAULT_COMPONENTS),
            "tokenizer": dict(_DEFAULT_TOKENIZER),
            "scorer": dict(_DEFAULT_SCORER),
        }

        self._datasets: dict[str, dict] = {}
        self._baselines: dict[str, dict] = {
            "pyserini_bm25_flat": {},
            "pyserini_bm25_mf": {},
        }

    def set_components(self, components: dict) -> None:
        self._configuration["components"] = components

    def set_tokenizer(self, tokenizer: dict) -> None:
        self._configuration["tokenizer"] = tokenizer

    def set_scorer(self, scorer: dict) -> None:
        self._configuration["scorer"] = scorer

    def add_dataset(
        self,
        name: str,
        *,
        corpus_size: int,
        num_queries: int,
        metrics: dict,
        per_query_ndcg10: dict[str, float],
        timing: dict,
        baselines: dict | None = None,
    ) -> None:
        self._datasets[name] = {
            "corpus_size": corpus_size,
            "num_queries": num_queries,
            "metrics": metrics,
            "per_query_ndcg10": per_query_ndcg10,
            "timing": timing,
        }
        if baselines:
            for baseline_name, ds_metrics in baselines.items():
                self._baselines.setdefault(baseline_name, {})[name] = ds_metrics

    def to_dict(self) -> dict:
        hw = self._hardware
        return {
            "schema_version": 3,
            "experiment": self.experiment,
            "configuration_name": self.configuration_name,
            "run_index": self.run_index,
            "seed": self.seed,
            "timestamp": self._timestamp,
            "git_commit": self._git_commit,
            "git_branch": git_branch(),
            "git_dirty": git_is_dirty(),
            "strata_version": self._strata_version,
            "hardware": {
                "cpu": hw.cpu,
                "cores": hw.cores,
                "ram_gb": hw.ram_gb,
                "os": hw.os,
            },
            "configuration": self._configuration,
            "datasets": self._datasets,
            "baselines": self._baselines,
        }

    def save(self, output_dir: str | Path) -> Path:
        output_dir = Path(output_dir) / "beir" / "raw"
        output_dir.mkdir(parents=True, exist_ok=True)

        filename = (
            f"{self.experiment}-{self.configuration_name}"
            f"-run{self.run_index}"
            f"-{self._timestamp_slug}"
            f"-{self._git_commit}.json"
        )
        path = output_dir / filename

        fd, tmp_path = tempfile.mkstemp(dir=output_dir, suffix=".json.tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(self.to_dict(), f, indent=2, default=_json_default)
            os.rename(tmp_path, path)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

        print(f"\nv3 result saved to {path}")
        return path
