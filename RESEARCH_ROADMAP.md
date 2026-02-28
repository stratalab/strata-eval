# Strata Research Roadmap

This document describes the research program for Strata — not individual papers,
but a sustained body of work with a shared benchmarking harness, cumulative evidence,
and a coherent thesis. The goal is a research identity that is expensive to replicate
and impossible to dismiss as a feature catalog.

---

## Table of Contents

1. [Core Thesis](#core-thesis)
   - [Why a Database?](#why-a-database)
   - [What "Speculative Execution" Means](#what-speculative-execution-means)
2. [What Is Architecturally Novel](#what-is-architecturally-novel)
   - [The Core Technical Problem](#the-core-technical-problem)
3. [Research Arcs](#research-arcs)
   - [Arc A: Database Architecture](#arc-a-database-architecture)
   - [Arc B: Retrieval Model](#arc-b-retrieval-model-enabled-by-that-architecture)
   - [Arc C: Co-located Inference](#arc-c-co-located-inference-as-a-systems-primitive)
   - [Arc D: Databases for Agents](#arc-d-databases-designed-for-agents)
4. [The Killer Experiment](#the-killer-experiment)
5. [Benchmarking Methodology](#benchmarking-methodology)
   - [How Research Teams Benchmark](#how-research-teams-benchmark)
   - [Raw Benchmarking](#raw-benchmarking)
   - [Comparative Benchmarking](#comparative-benchmarking)
   - [Statistical Rigor](#statistical-rigor)
6. [Benchmark Infrastructure](#benchmark-infrastructure)
   - [The Harness as an Independent Contribution](#the-harness-as-an-independent-contribution)
   - [Experiment Configuration](#experiment-configuration)
   - [Results Storage](#results-storage)
   - [Comparison Engine](#comparison-engine)
7. [Sequencing and Dependencies](#sequencing-and-dependencies)
8. [References](#references)

---

## Core Thesis

The research program makes one falsifiable claim:

> AI-era workflows — retrieval, inference, mutation, and speculation — are better
> served by a database where these operations share one transactional and memory
> reality, enabling branch-scoped experimentation and zero-serialization pipelines
> that are impractical in polyglot architectures.

Specifically: compared to a best-effort polyglot stack, Strata reduces end-to-end
pipeline latency (by eliminating serialization boundaries), reduces engineering
complexity (by replacing orchestration code with database operations), and uniquely
enables branch-scoped speculative computation (which has no polyglot equivalent
that avoids full reindexing).

This is not an argument that many primitives can coexist. Reviewers will mentally
reduce that to "DuckDB + FAISS + llama.cpp glued together." The argument is that
**sharing physical layout, lifecycle, and transactional semantics** enables an
execution model that is impractical when these capabilities live in separate
processes.

The strongest expression of that execution model is **branching as a queryable
execution dimension**, not a storage trick. Git-like copy-on-write branching
applied to vector indexes, graphs, event materializations, embeddings, and search
configurations turns the database into a substrate for continuous speculative
computation. Most databases treat snapshots as durability artifacts. Strata treats
them as an experimentation substrate — for agent simulation, RAG evaluation, index
A/B testing, and recursive reasoning sandboxes.

### Why a Database?

A predictable reviewer objection: "This is an application runtime with embedded
storage, not a database."

The answer: speculative computation is a first-class database workload. It must be
**durable** (branches survive crashes), **replayable** (event log reconstructs any
state), **branch-isolated** (concurrent experiments do not interfere), **index-
consistent** (all indexes reflect the same logical snapshot), and **queryable**
(branches are addressed and searched like any other data). These are database
guarantees — durability, concurrency, consistency, indexing, introspection — not
application-layer concerns.

An agent framework with embedded storage does not provide branch-isolated index
consistency or cross-primitive transactional atomicity. A database does.

### What "Speculative Execution" Means

We use "speculative execution" in a precise operational sense:

- **Speculative branches** are cheap COW forks of all state and indexes.
- Branches can be **queried and mutated independently** with snapshot-consistent
  indexes (BM25, HNSW, property graph all reflect the same logical snapshot).
- **Merges** reconcile changes under defined conflict rules (last-writer-wins per
  key, with conflict metadata preserved for application-level resolution).
- The system exposes **branch lineage metadata** to queries (parent, creation time,
  delta size, divergence point).

This is not filesystem snapshotting. The hard problem is maintaining index
consistency across branch boundaries — see [invariants below](#the-core-technical-problem).

Every paper in this program explores a consequence of that shared physical reality.
The later agent and recursive-reasoning work should feel like inevitable consequences
of the architecture, not bolted-on features.

**Compressed thesis for Paper 1 abstract:**

> Strata is a database designed for continuous speculative computation. It unifies
> heterogeneous primitives under a single write-ahead log and MVCC branching model,
> so retrieval, indexing, inference, and mutation execute in one address space with
> snapshot-consistent indexes. This enables branch-scoped experimentation and zero-
> serialization RAG pipelines that are impractical in polyglot stacks, while
> remaining competitive with embedded baselines on standard microbenchmarks.

---

## What Is Architecturally Novel

The novelty is not the primitives. Each has standalone equivalents. The novelty is
that they share a unified execution and storage model:

| Architectural Property | What It Means | Why It Matters |
|---|---|---|
| One WAL / event log | All primitives (KV, JSON, events, state, graph, vectors) write to the same log | Atomic cross-primitive operations without distributed transactions |
| One MVCC / branching model | COW branches span all primitives simultaneously | Speculative execution: branch, experiment, discard or merge |
| Zero serialization boundaries | Search, inference, and mutation share memory | No IPC, no JSON marshaling, no network round-trips between retrieval and generation |
| Shared memory residency | Vector indexes, inverted indexes, and graph structures co-reside | Cross-primitive search is a pointer chase, not an API call |
| Unified write-path hooks | Auto-embedding triggers on any primitive write | Write-time indexing without external pipelines |

Standalone systems that cover individual capabilities:

| Capability | Standalone Equivalents |
|---|---|
| KV with CAS | Redis, RocksDB |
| JSON documents | MongoDB, SQLite JSON |
| Append-only hash-chained events | Kafka, EventStoreDB |
| State cells with CAS | etcd, Consul |
| Property graph with ontology | Neo4j, DGraph |
| Dense vectors (Segmented HNSW) | Pinecone, Qdrant, FAISS |
| BM25 inverted index | Elasticsearch, Tantivy |
| Local LLM inference (llama.cpp) | Ollama, vLLM |
| Copy-on-write branching | Git (but for data) |

Each primitive alone is well-understood. The contribution is unifying these under
a single execution and storage model, and demonstrating emergent capabilities —
cross-primitive search, zero-serialization RAG, branched experimentation,
write-time auto-embedding — that arise from that unification and do not exist
when these are separate systems.

### The Core Technical Problem

If a reviewer asks *"What is the single hardest systems problem you solved?"*
the answer must be one sentence:

> Snapshot-consistent copy-on-write branching across heterogeneous indexes
> (inverted lists, HNSW graphs, and property graphs) under a unified WAL,
> without reindexing.

This is the publishable systems meat. Spell out the invariants the system
guarantees:

1. **Branch snapshot invariant.** All primitives and indexes in branch B reflect
   the same logical snapshot S(B). A query on branch B never sees a document in
   the BM25 index that is absent from the HNSW graph, or vice versa.

2. **Cross-primitive atomicity.** A transaction that updates a JSON document and
   emits an event is either visible everywhere (KV, JSON, BM25, HNSW, event log)
   or nowhere. There is no window where one index reflects the write and another
   does not.

3. **Index coherency invariant.** BM25 posting lists and HNSW graph membership
   correspond to the same set of live documents at every snapshot. Branching does
   not break this — a branch inherits coherent indexes and maintains coherency
   through its own mutations.

Without stated invariants, the paper reads like "we built stuff." With them, it
reads like "we solved a systems problem with formally stated correctness
guarantees."

**Important framing note:** Avoid "nothing does this" claims in papers. There is
always an obscure VLDB paper or a 2018 prototype. Frame novelty as: *"unifying
previously disjoint techniques under a single execution and storage model, with
formally stated snapshot-consistency invariants."*

---

## Research Arcs

The 12 individual paper ideas from the original roadmap compress into four
conceptual arcs. Externally these read as a coherent research identity, not a
sequence of disconnected outputs. Internally, each arc may produce 1-3 papers
depending on depth and results.

### Arc A: Database Architecture

**The foundation paper.** Everything else cites this.

**Venue:** VLDB, OSDI, or SOSP (systems)

**Thesis:** A database designed for continuous speculative computation — where
branching, auto-embedding, cross-primitive search, and event materialization are
first-class operations sharing one WAL and one MVCC model — achieves competitive
performance on each primitive's standard benchmark while enabling an execution
model that is impractical with polyglot stacks.

**Why this framing matters:** "We integrated X + Y + Z" gets rejected at systems
venues unless the paper articulates a systems principle that required architectural
rethinking, not just colocation. The principle here is: *all primitives share
physical layout, lifecycle, and transactional semantics so that speculative
execution (branching) extends across the entire data substrate.*

Paper 1 must clearly articulate what had to be rethought architecturally —
specifically the unified WAL, the COW branching model that spans heterogeneous
indexes, and the write-path hooks that make auto-embedding possible. If it does
not, reviewers will reduce it to a feature bundle.

**Sub-contributions (may be sections of Paper 1, or standalone papers if deep enough):**

1. **Copy-on-write branching across heterogeneous indexes.** Git-like branching
   applied to KV, vectors, graphs, BM25 indexes simultaneously. The hard problem
   is maintaining index consistency across branch boundaries — a BM25 inverted
   index and an HNSW graph must both reflect the same COW snapshot. This is a
   publishable database architecture result on its own.

2. **Auto-embedding as a write-path primitive.** Every KV set, JSON insert, and
   event append is immediately vector-indexed via an in-process embedding model.
   This trades write amplification for guaranteed query freshness, eliminating
   batch embedding pipelines. The economics (write cost vs. freshness value)
   determine whether this is a section of Paper 1 or its own paper.

3. **Event-sourced multi-primitive materialization.** Events project into KV, JSON,
   graph, and state primitives — all auto-embedded and searchable. Replay from the
   immutable event log guarantees consistency. This connects event sourcing to
   multi-modal materialization in a way no existing system does.

**Evaluation structure:**

1. **Microbenchmarks** — Each primitive vs. its standalone equivalent
   - KV: YCSB workloads A-F vs. Redis, RocksDB, SQLite
   - Vectors: ann-benchmarks protocol (recall@K vs. QPS) vs. FAISS, hnswlib
   - Graph: LDBC Graphalytics (BFS, WCC, PageRank, CDLP, LCC, SSSP) vs. Neo4j
     embedded, SQLite recursive CTEs
   - BM25: BEIR nDCG@10 vs. Tantivy, SQLite FTS5
   - JSON: document insert/query throughput vs. MongoDB, SQLite JSON1
   - Events: append throughput, hash-chain verification speed

2. **Macrobenchmarks** — Cross-primitive workflows
   - "Ingest 1M documents, auto-embed, hybrid search" — full write+embed+search
   - "Branch, modify 10K entries, search on branch, merge" — COW overhead
   - "Cross-primitive search across KV + JSON + Events" — unified retrieval
   - **The killer experiment** (see [below](#the-killer-experiment))

3. **Resource efficiency** — Memory, disk, CPU footprint
   - One Strata instance vs. the polyglot stack (Redis + Elasticsearch + Neo4j +
     Kafka + Qdrant) running the same workload
   - Quantify the "convergence tax" — measure overhead from sharing vs. savings
     from eliminated serialization

4. **Branching cost** — What speculative execution actually costs
   - Branch creation latency at varying database sizes
   - Write amplification on branches (COW efficiency)
   - Merge performance and conflict resolution
   - Compare: PostgreSQL snapshots, Redis BGSAVE, ZFS filesystem snapshots

**Baselines:**
- SQLite (embedded relational)
- RocksDB (embedded KV)
- DuckDB (embedded analytical)
- LanceDB (embedded vectors)
- The polyglot stack: Redis + Elasticsearch + Neo4j + Kafka + Qdrant

**Key result posture:**

For microbenchmarks: *competitive within a bounded overhead for most workloads,
with honest reporting of where we lose and why.* Each standalone system is
purpose-built for its workload; Strata pays a convergence tax on some primitives
and gains on others. The paper explains the tradeoff, not just the wins.

For macrobenchmarks: *wins end-to-end where serialization and orchestration
overhead dominate.* The branched configuration search (killer experiment) is the
result no polyglot stack can replicate.

**Fairness rubric** (commit to this in the methodology section to preempt reviewer
attacks on comparison validity):
- Same durability level (fsync policy matched across all systems)
- Same hardware, same OS, same kernel tuning
- Same dataset format and content
- Published configs for every baseline (readers can reproduce)
- Baselines tuned with best-of-N configs (report sensitivity to tuning)
- Report Strata losses alongside wins — full matrix, no cherry-picking

---

### Arc B: Retrieval Model Enabled by That Architecture

**What retrieval looks like when search, vectors, graph, and inference share one
address space.**

**Venue:** SIGIR, ECIR, CIKM (information retrieval)

**Unifying concept: Typed Retrieval Plans.** A query is compiled into a plan with
typed operators — route, fuse, early-exit, boost — that execute over co-resident
indexes. This avoids the "bag of heuristics" critique by giving every technique
a principled role in a retrieval compiler:

| Operator | What It Does | Why It's Principled |
|---|---|---|
| `route(expansion_type)` | Lex expansions → BM25; vec/HyDE → HNSW | Type-directed dispatch, not broadcast |
| `fuse(position_weighted)` | RRF + reranker signals weighted by rank position | Statistically motivated: top-3 vs. tail behave differently |
| `early_exit(strong_signal)` | BM25 probe skips expensive pipeline when confidence is high | Latency optimization with bounded quality loss |
| `boost(graph_proximity)` | Graph neighborhood score added post-fusion | Structural signal for entity-centric queries |
| `decompose(ontology)` | LLM reads ontology, emits typed sub-queries | Query compilation, not ad-hoc expansion |

**Thesis:** Compiling queries into typed retrieval plans over co-resident indexes
outperforms both fixed-weight fusion and untyped expansion — and the shared-memory
architecture makes the full plan cheap enough to execute on every query.

**Sub-contributions (1-2 papers):**

1. **Typed hybrid search with position-aware blending.**
   - `route()`: typed expansion routing. Lex expansions go to BM25 only, vec/HyDE
     go to HNSW only. Existing hybrid systems broadcast expansions to all indexes.
   - `fuse()`: position-aware blending. RRF and reranker signals weighted
     differently by rank position (top-3: trust RRF more; rank 11+: trust
     reranker more).
   - `early_exit()`: BM25 probe that skips the expensive pipeline when confidence
     is high. Latency optimization with quality preservation analysis.

2. **Graph-augmented hybrid retrieval.**
   - `boost()`: graph proximity score after RRF fusion
   - `decompose()`: ontology-guided query decomposition (LLM reads ontology,
     emits typed sub-queries that compile into a retrieval plan)
   - Graph-context reranking (enriching reranker snippets with graph neighborhood)
   - Three-signal blending (RRF + reranker + graph proximity, position-aware)
   - Key experiment: show that graph augmentation helps most on queries requiring
     structural reasoning (multi-hop, entity-centric) and least on simple keyword
     queries. Characterize when the graph signal adds value.

3. **Segmented HNSW internals** (if the implementation has a clean publishable
   result — segmentation strategy, incremental build, or memory layout).

**Evaluation:**
- BEIR benchmark suite (18 datasets), nDCG@10 headline metric
- Ablation: BM25 only -> +vectors -> +RRF -> +typed expansion -> +reranking
  (fixed blend) -> +position-aware blend -> +graph boost -> +strong signal skip
- Quality-latency Pareto chart
- ann-benchmarks protocol for vector index (recall@K vs. QPS Pareto curves)

**Baselines:**
- BM25 (Lucene/Tantivy)
- DPR (dense passage retrieval)
- ColBERT v2 (late interaction)
- Elasticsearch ELSER (learned sparse)
- Cohere Rerank + vector search (cloud RAG baseline)
- Microsoft GraphRAG
- qmd (local hybrid, architecturally similar)

**Existing infrastructure:** BEIR benchmarks already implemented in strata-eval
with keyword and hybrid results across 15 datasets.

---

### Arc C: Co-located Inference as a Systems Primitive

**What happens when inference is not a service call but a database operation.**

**Venue:** VLDB, EuroSys, MLSys (systems)

**This is primarily a systems paper.** The core contribution is the latency
economics of co-location. RAG quality is a secondary validation: *we don't regress
quality while improving latency and reducing engineering complexity.* If framed as
an ML paper about RAG quality, it competes with GPT-4-class systems on their home
turf. If framed as a systems paper about co-location economics, it occupies
uncontested ground.

**Thesis:** Co-locating LLM inference inside the database process — sharing memory
with indexes, eliminating serialization between retrieval and generation — yields
measurable latency improvements and enables a single-call RAG primitive (db.ask())
that achieves comparable answer quality to framework-based RAG with significantly
lower end-to-end latency and zero integration code.

**Sub-contributions:**

1. **Inference co-location as a systems primitive.**
   - Latency breakdown: search -> embed -> expand -> rerank -> generate pipeline
     measured in-process (Strata) vs. localhost inference server vs. remote API
   - Quantify serialization overhead, network round-trips, context switching
   - Token throughput: tokens/sec for generation, embedding operations
   - Memory efficiency: shared memory vs. separate heaps
   - **Key systems result:** per-stage latency attribution showing where
     serialization dominates and where compute dominates

2. **Native RAG with zero serialization overhead.**
   - A single-call RAG primitive (db.ask()) — search and generation in the same
     process, no integration code
   - RAG quality as validation (not contribution): answer accuracy/F1,
     faithfulness, citation precision — showing *parity*, not superiority
   - End-to-end latency: query in -> answer out, breakdown by component
   - Datasets: Natural Questions, TriviaQA, FinanceBench, HotpotQA

**Baselines:**
- Strata with external Ollama (same machine, network hop)
- Strata with cloud API (OpenAI/Anthropic)
- LangChain + Ollama + ChromaDB (standard stack)
- LangChain + Pinecone + GPT-4 (cloud RAG)
- LlamaIndex + ChromaDB + GPT-4
- vLLM + Qdrant (optimized separate services)
- Closed-book LLM (no retrieval, lower bound)

---

### Arc D: Databases Designed for Agents

**What a database looks like when its primary consumer is an LLM, not a human
writing SQL.**

**Venue:** CHI, UIST, NeurIPS (HCI/AI)

**Thesis:** Systematic design principles for LLM-consumable database APIs —
introspection, actionable errors, ontology-validated state machines, and
branch-scoped recursive execution — measurably improve agent task completion
rates and reduce error recovery cycles. The branch-as-sandbox model enables
recursive reasoning that is impossible with append-only databases.

This arc consolidates the agent-first API design, graph-validated state machines,
and recursive query execution stories into one coherent HCI/systems boundary
narrative.

**Sub-contributions:**

1. **Agent-first API design principles.**
   - describe() introspection, actionable errors, explain mode, progressive
     capability disclosure
   - Quantitative A/B test: agent performance with vs. without each feature
   - Metrics: task completion rate, API calls to complete task, error recovery
     rate, time to first successful query
   - Test across Claude, GPT-4, and open-source models to show generalization

2. **Database-native state machines via graph ontology.**
   - Property graph ontologies (object types = states, link types = transitions)
     define and validate state machines
   - Rich error messages include valid transitions and natural-language suggestions
   - Agent success rate on stateful tasks with FSM validation + rich errors vs.
     without
   - Compare: application-layer FSM (invisible to agent), PostgreSQL triggers,
     Strata graph-validated FSM

3. **Recursive query execution over indexed primitives.**
   - RLMs (Recursive Language Models) operating over indexed database primitives
     (BM25, vectors, graph, ontology) vs. RLMs over raw text
   - Branch-scoped recursive exploration: agents branch, explore speculatively,
     discard or merge findings
   - Key experiment: Strata's indexed primitives shift the model-size frontier —
     reducing the model size needed to solve structured multi-hop tasks under
     fixed compute budgets. Let the numbers determine the magnitude; do not
     pre-commit to a specific model-size ratio in the paper.

**Reproducibility safeguards for agent evaluations** (preempts HCI/AI reviewer
attacks on prompt dependence, model drift, and small sample sizes):
- Fixed tasks with deterministic grading (no subjective evaluation)
- Published tool-call traces for every experiment
- Multiple models (Claude, GPT-4, open-source 8B/70B) to show generalization
- Multiple seeds per model (minimum 5 runs)
- Strict budgets: max tool calls, max tokens, max wall-clock time per task
- Task suite versioned and frozen before experiments run (pre-registration)

**Baselines:**
- Standard tool-use agent with generic database (PostgreSQL, Redis)
- RLM reference implementation over raw text
- LangChain agent with separate vector store + KV store
- Application-layer FSM with generic error messages

---

## The Killer Experiment

Every paper needs thorough microbenchmarks and ablations. But the experiment that
makes reviewers remember the paper is one that **only Strata can run** — and that
would be absurd to attempt with a polyglot stack.

**Branched Configuration Search:**

> Index a corpus once. Fork 30 branches. Apply small per-branch deltas (schema
> tweak, ontology change, retrieval config, partial re-embed). Evaluate each
> branch. Converge on an optimal configuration in minutes.

The key insight that makes this uncopyable: branches **reuse physical structures**
from the base. The polyglot equivalent requires 30 separate full ingestion +
indexing cycles because there is no shared base state.

**Experiment design:**

1. **Base state:** Index a BEIR corpus with full BM25 + HNSW + graph indexes.
   This is done once.

2. **Fork 30 branches.** Each branch applies a small delta:
   ```
   5 embedding models  x  3 BM25 (k1, b) settings  x  2 ontology schemas
   = 30 configurations, each on its own COW branch
   ```
   Per-branch work: re-embed ~10K documents with a different model, adjust BM25
   parameters, swap ontology schema. The remaining corpus and indexes are shared
   via COW — no copying, no reindexing.

3. **Evaluate each branch** on BEIR nDCG@10.

4. **Polyglot baseline:** 30 separate Elasticsearch + Qdrant + model server
   deployments, each requiring full corpus ingestion and index building from
   scratch.

**Metrics (not just wall clock):**
- **Total CPU-seconds** across all branches (captures actual compute, not just
  parallelism)
- **Total bytes written** (COW efficiency — branches write only deltas)
- **Time-to-first-query on a new branch** (how fast a branch becomes searchable)
- **Wall-clock time** to evaluate all 30 configurations
- **nDCG@10 variance** across branches (confirms index isolation — branches
  produce different results)

**Why "just run 30 Docker containers" is not equivalent:**
- Docker containers replicate the full dataset and rebuild all indexes per config.
  Strata branches share the base and write only deltas.
- Docker requires orchestration code to manage 30 deployments. Strata requires
  `branch create config-17`.
- Docker ingestion is O(N * corpus_size). Strata branching is O(N * delta_size).

This demonstrates:
- Why convergence matters (one system, not 30 deployments)
- Why branching matters (shared base state, not full reindex)
- Why embedded inference matters (no external model server per branch)
- Why this is a new execution model, not a feature bundle

Without a "this would be absurd elsewhere" demonstration, reviewers will reduce
everything to microbenchmark comparisons where Strata inevitably loses to
purpose-built systems on their home turf.

---

## Benchmarking Methodology

### How Research Teams Benchmark

Research evaluation has two fundamental questions:

1. **"Does it work?"** — Raw/intrinsic benchmarking. Absolute performance on
   standardized tasks.
2. **"Does it work better?"** — Comparative benchmarking. Performance relative to
   baselines and state-of-the-art.

Different research communities have different evaluation cultures:

| Community | Cares About | Standard Benchmarks |
|---|---|---|
| Systems (VLDB, OSDI) | Throughput, latency, scalability, resource efficiency | YCSB, TPC, ann-benchmarks |
| IR (SIGIR, ECIR) | Retrieval quality on standardized datasets | BEIR, MTEB, MS MARCO |
| AI/ML (NeurIPS, ACL) | Task accuracy, comparison to SotA models | NQ, TriviaQA, HotpotQA, MMLU |
| HCI (CHI, UIST) | User/agent task completion, usability | Custom task suites with user studies |

### Raw Benchmarking

#### Standardized Benchmark Suites

Never invent your own evaluation dataset if a community-standard one exists.
Reviewers immediately distrust results on custom datasets.

- **BEIR** — 18 retrieval datasets spanning diverse domains. nDCG@10 is the
  headline metric. Gold standard for IR papers.
- **MTEB** — Massive text embedding benchmark. For evaluating embedding quality.
- **ann-benchmarks** — Standard protocol for approximate nearest neighbor indexes.
  Recall@K vs. QPS Pareto curves.
- **YCSB** — Yahoo Cloud Serving Benchmark. Standard for KV store performance
  (workloads A-F covering different read/write ratios).
- **LDBC Graphalytics** — Standard for graph algorithm performance (BFS, WCC,
  PageRank, CDLP, LCC, SSSP).
- **Domain-specific:** FinanceBench (financial QA), MedQA (medical),
  HotpotQA/MuSiQue (multi-hop reasoning).

#### Metrics — Never Report Just One

**Retrieval quality:**

| Metric | What It Measures |
|---|---|
| nDCG@10 | Ranking quality (graded relevance) |
| MAP | Average precision across recall levels |
| MRR | How quickly you find the first relevant result |
| Recall@100 | Coverage — did you find all relevant docs? |
| Precision@1 | Is the top result correct? |

**System performance:**

| Metric | What It Measures |
|---|---|
| p50/p95/p99 latency | Tail latency matters, not just average |
| Throughput (QPS, ops/s) | Queries or operations per second at saturation |
| Index build time | Write-path cost |
| Memory footprint | Practical deployment constraint |
| Index size on disk | Storage cost |

**RAG quality:**

| Metric | What It Measures |
|---|---|
| Answer accuracy / F1 | Is the answer correct? |
| Faithfulness | Does the answer only use retrieved context? |
| Citation precision | Are source citations accurate? |
| Answer relevance | Does it actually address the question? |

#### Experimental Protocol

This separates papers that get accepted from those that get desk-rejected:

1. **Pre-registered experiments.** Decide what you are measuring and why before
   running anything. Write the evaluation section outline first. This prevents
   cherry-picking results.

2. **Controlled variables.** Change exactly one thing per experiment. If testing
   graph-augmented retrieval, the embedding model, chunk size, BM25 parameters,
   and everything else must be identical between control and treatment.

3. **Hardware specification.** Every paper includes a hardware table:
   ```
   CPU: AMD EPYC 7763, 64 cores
   RAM: 256 GB DDR4
   GPU: NVIDIA A100 80GB (if applicable)
   Storage: NVMe SSD
   OS: Ubuntu 22.04
   ```

4. **Multiple runs with statistics.** Never report a single number. Run each
   experiment 3-5 times minimum. Report mean +/- standard deviation, or median
   with interquartile range.

5. **Warm-up runs.** Discard the first 1-2 runs to avoid cold cache effects.

6. **Fixed seeds.** Set random seeds for reproducibility. Document them.

### Comparative Benchmarking

#### Baseline Selection

Three categories of baselines are required:

**Simple baselines (sanity check):**
- BM25 only (no neural anything)
- TF-IDF + cosine similarity
- Random retrieval (establishes floor)

These prove your problem is not trivially solvable. If BM25 alone gets 95% nDCG,
a fancy graph-augmented pipeline needs to justify its complexity.

**Strong baselines (state of the art):**
- The current best system on the benchmark
- The system most similar to yours architecturally
- The system from the most cited recent paper in the area

**Ablation baselines (your system minus components):**

This is the most important category. Remove one component at a time from the full
system and measure the degradation:

```
Full system:     BM25 + vectors + expansion + reranking + graph boost
Ablation 1:      BM25 + vectors + expansion + reranking              (no graph)
Ablation 2:      BM25 + vectors + expansion              + graph     (no reranking)
Ablation 3:      BM25 + vectors             + reranking  + graph     (no expansion)
Ablation 4:      BM25           + expansion + reranking  + graph     (no vectors)
Ablation 5:             vectors + expansion + reranking  + graph     (no BM25)
```

This proves each component contributes. If removing a component does not change
results, you cannot claim it matters.

#### Fair Comparison Practices

- **Use the competitor's best configuration.** Comparing your tuned system against
  an untuned baseline is the fastest way to get rejected. Use published optimal
  hyperparameters, or tune baselines with the same effort as your own system.

- **Same data, same splits.** Everyone uses the same train/dev/test split. For
  BEIR, these are fixed.

- **Same compute budget (when relevant).** If your system uses 10x the compute,
  that is a critical detail. Consider a quality-per-dollar analysis.

- **Report losses, not just wins.** Every system has datasets where it
  underperforms. A credible paper shows the full matrix and explains why certain
  datasets favor the baseline.

#### The Results Table

Centerpiece of every evaluation section:

```
Table 1: nDCG@10 on BEIR benchmark (mean of 3 runs)

Dataset      BM25   DPR    ColBERT  ELSER  Strata  Strata+G
-------      ----   ---    -------  -----  ------  --------
MS MARCO     0.228  0.311  0.344    0.338  0.341   0.352*
NFCorpus     0.325  0.189  0.338    0.341  0.346   0.371*
SciFact      0.665  0.318  0.671    0.688  0.692   0.701
FiQA         0.236  0.112  0.356    0.348  0.361   0.359
...
Average      0.381  0.247  0.412    0.419  0.425   0.441*

* = statistically significant improvement (p < 0.05, paired t-test)
Strata+G = Strata with graph-augmented retrieval
```

Every cell is filled. Losses are shown. Significance is marked.

### Statistical Rigor

- Run each experiment 3-5 times minimum
- Report mean +/- standard deviation
- Use paired t-test or Wilcoxon signed-rank test for significance
- Report effect size (Cohen's d) when claiming improvements
- p < 0.05 threshold, but report exact p-values
- For multiple comparisons, apply Bonferroni or Holm-Bonferroni correction

---

## Benchmark Infrastructure

### The Harness as an Independent Contribution

The benchmarking harness itself is a significant asset. It provides reproducible
cross-modal benchmarking — the ability to run DB, IR, and ML experiments on
shared datasets with shared methodology and compare results longitudinally. This
is closer to what made TPC benchmarks influential than to a typical paper's eval
script.

Consider releasing the harness (strata-eval) before most papers. This:
- Builds credibility with reviewers ("they open-sourced the harness, we can verify")
- Invites community contributions and scrutiny
- Establishes Strata's evaluation methodology as a reference point
- Creates citation opportunities independent of any single paper

The harness already implements: BEIR (15 datasets), YCSB (6 workloads),
ann-benchmarks (3 datasets), LDBC Graphalytics (6 algorithms), result recording
with hardware metadata, and report generation. All benchmarks execute via direct
CLI invocation with no Python overhead in timing loops.

### Experiment Configuration

Each experiment is a declarative config, not a script with hardcoded parameters:

```yaml
experiment:
  name: "graph-boost-ablation-v3"
  date: "2026-02-27"
  hypothesis: "Graph proximity boost improves nDCG@10 on entity-centric datasets"

dataset:
  name: "nfcorpus"
  source: "beir"
  split: "test"

system:
  base: "strata"
  version: "0.6.0"
  commit: "abc123"

  components:
    bm25: true
    vectors: true
    expansion: true
    reranking: true
    graph_boost: true          # the variable being tested
    graph_weight: 0.3

  fixed_params:
    bm25_k1: 1.2
    bm25_b: 0.75
    embed_model: "nomic-embed-text-v1.5"
    expansion_model: "qwen3:8b"
    rerank_model: "qwen3:8b"
    top_k: 10
    rrf_k: 60

hardware:
  cpu: "AMD Ryzen 9 7950X"
  ram: "64GB"
  gpu: "none"
  storage: "NVMe"

runs: 5
seed_base: 42
warmup_runs: 1
```

### Results Storage

Every run produces a structured result artifact:

```json
{
  "experiment": "graph-boost-ablation-v3",
  "dataset": "nfcorpus",
  "run": 3,
  "seed": 44,
  "timestamp": "2026-02-27T14:23:01Z",
  "git_commit": "abc123",
  "metrics": {
    "ndcg@10": 0.371,
    "map": 0.298,
    "mrr": 0.412,
    "recall@100": 0.634,
    "precision@1": 0.389
  },
  "latency": {
    "p50_ms": 12.3,
    "p95_ms": 45.1,
    "p99_ms": 89.7
  },
  "system": {
    "version": "0.6.0",
    "components_enabled": ["bm25", "vectors", "expansion", "reranking", "graph_boost"]
  },
  "hardware": {
    "cpu": "AMD Ryzen 9 7950X",
    "ram_gb": 64,
    "gpu": "none"
  }
}
```

### Comparison Engine

After all runs complete, automated pipeline:

1. Aggregate results across runs (mean, std, confidence intervals)
2. Run significance tests (paired t-test per dataset, Bonferroni correction)
3. Generate LaTeX results tables
4. Generate plots (bar charts with error bars, Pareto curves, radar charts)
5. Flag regressions against previous experiment runs
6. Produce a reproducibility package (configs, seeds, model versions, dataset
   download scripts)

---

## Sequencing and Dependencies

```
Arc A: Architecture (Paper 1) ──────────────────────────┐
  │                                                      │
  │   Sub-contributions that may be sections             │
  │   or standalone papers:                              │
  │   - COW branching across heterogeneous indexes       │
  │   - Auto-embedding as write-path primitive            │
  │   - Event-sourced multi-primitive materialization     │
  │                                                      │
  ├──► Arc B: Retrieval Model                            │
  │      - Typed retrieval plans (query compiler)        │
  │      - Graph-augmented plan operators                │
  │      - (Segmented HNSW internals, if publishable)    │
  │                                                      │
  ├──► Arc C: Co-located Inference                       │
  │      - Inference co-location latency economics       │
  │      - Native RAG (db.ask())                         │
  │                                                      │
  └──► Arc D: Databases for Agents                       │
         - Agent-first API design principles             │
         - Graph-validated state machines                 │
         - Recursive query execution (branch-scoped)     │
                                                         │
     Killer Experiment (branched config search) ◄────────┘
```

**Phase 1 (now):** Arc A — Architecture. Build the harness. Establish the thesis.
The branched configuration search experiment is the centerpiece.

**Phase 2 (after harness):** Arcs B and C — independent papers that each evaluate
one consequence of the shared architecture. Can be written in parallel. These
reuse 80% of the harness infrastructure.

**Phase 3 (after roadmap features ship):** Arc D — depends on features from issues
#1269-#1274 and the RFCs. Also depends on Arc B results (the retrieval model that
agents use).

---

## References

### Strata Issues
- [#1269](https://github.com/strata-ai-labs/strata-core/issues/1269) — Hybrid search pipeline
- [#1270](https://github.com/strata-ai-labs/strata-core/issues/1270) — Graph-Augmented Hybrid Search
- [#1272](https://github.com/strata-ai-labs/strata-core/issues/1272) — Recursive Query Execution
- [#1273](https://github.com/strata-ai-labs/strata-core/issues/1273) — Native RAG: db.ask()
- [#1274](https://github.com/strata-ai-labs/strata-core/issues/1274) — Agent-First API Design

### Strata RFCs
- RFC: Event Projections to Other Primitives
- RFC: Graph-Validated State Transitions
- RFC: State-Event Audit Trail

### Benchmarking Standards
- [BEIR](https://github.com/beir-cellar/beir) — Heterogeneous benchmark for information retrieval (Thakur et al., 2021)
- [MTEB](https://github.com/embeddings-benchmark/mteb) — Massive Text Embedding Benchmark
- [ann-benchmarks](https://ann-benchmarks.com/) — Benchmarking approximate nearest neighbor algorithms
- [YCSB](https://github.com/brianfrankcooper/YCSB) — Yahoo Cloud Serving Benchmark (Cooper et al., 2010)
- [LDBC Graphalytics](https://graphalytics.org/) — Graph algorithm benchmarking (Iosup et al., 2016)

### Related Academic Work
- [GraphRAG](https://arxiv.org/abs/2404.16130) — Microsoft, knowledge graph enhanced RAG
- [RLM](https://arxiv.org/abs/2512.24601) — Recursive Language Models (Zhang, Kraska, Khattab, MIT)
- [PageIndex](https://github.com/VectifyAI/PageIndex) — Vectorless reasoning-based RAG
- [qmd](https://github.com/tobi/qmd) — Local hybrid search with typed expansion
- [Elasticsearch ELSER](https://www.elastic.co/guide/en/elasticsearch/reference/current/semantic-search-elser.html) — Learned sparse encoder
- Reciprocal Rank Fusion — Cormack, Clarke, Buettcher (2009)
