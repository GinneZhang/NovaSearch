# AsterScope: Enterprise Copilot & Intelligent Retrieval Engine

<p align="center">
  <a href="README.md">[English]</a> | <a href="README_CN.md">[中文文档]</a>
</p>

AsterScope is a production-hardened enterprise knowledge retrieval and reasoning system implementing **"Tri-Engine Fusion"**: **LLM Semantics + Hybrid Multimodal RAG + Knowledge Graph Reasoning**.

Our objective is to deliver highly accurate, explainable, and hallucination-resistant Copilot experiences for enterprise SOPs, compliance documents, and structured knowledge.

---

## System Architecture & Core Pillars

### 1. Query Understanding & Context Memory

- **Structuring**: LLM-driven query rewriting, intent recognition, and structured semantic graph generation (triplet extraction).
- **Decomposition**: Multi-hop task breakdown via `DependencyGraph` with Plan-and-Execute sub-task tracking.
- **Planner-Critic Loop**: Iterative plan → execute → evaluate → re-plan cycle with `PlannerCritic`.
- **Clarification**: Stateful clarification loop with SUSPENDED/REPLANNING states persisted via Redis `StateManager`.
- **Context**: Semantic cross-session thread linking and memory via Redis + SentenceTransformers.

### 2. Hybrid Multimodal Retrieval

- **Tri-Retrieval Fusion**: Sparse (PostgreSQL FTS / Elasticsearch) + Dense (PGVector `all-MiniLM-L6-v2`) + Vision (CLIP `clip-ViT-B-32`) + Structural (Neo4j Cypher).
- **Deep Table Embeddings**: Semantic "Table Summary + Column Description" headers instead of raw Markdown embedding.
- **Table-Specific Retrieval**: `TableRetriever` with `metadata.type == 'table'` filtering and structured value extraction.
- **Long-Context Processing**: LlamaIndex-backed `LongContextProcessor` for documents exceeding 50k chars.
- **Cross-Modal Search**: `/ask_vision` maps images to the shared CLIP vector space.
- **Advanced Chunking**: Sliding Window + Embedding Clustering.
- **Reranking**: Cross-Encoder, with pluggable ColBERT and MonoT5.

### 3. Knowledge Graph Reasoning (KG)

- **Probabilistic Entity Linking**: Vector similarity with confidence scoring; ambiguous matches (conf < 0.8) trigger proactive user clarification.
- **Cypher Linting & Validation**: Schema-aware validator rejects write operations and unknown labels before execution.
- **Structured Cypher Objects**: `CypherResult` dataclass with nodes, edges, properties, and path existence validation.
- **Symbolic Path Validator**: Checks label/rel existence against live schema before Cypher execution.
- **Hardened Ontology Alignment**: `OntologyManager` maps triplet terms to canonical schema classes via embedding similarity at strict >0.9 confidence. Unmapped terms trigger Clarification.
- **Dynamic Schema Introspection**: Live `CALL db.labels()` / `CALL db.relationshipTypes()` with property sampling.
- **Self-Healing Cypher**: Error-feedback retry loop (max 2 retries) auto-repairs invalid Cypher.
- **Factual Grounding**: KG constraints injected into generative context to suppress LLM hallucinations.

### 4. Controlled LLM Generation

- Source-grounded QA with mandatory origin tracing.
- **Pre-Flight Hallucination Interceptor**: Two-stage buffered generation for high-stakes queries; ConsistencyEvaluator validates before streaming (threshold > 0.8).
- **Symbolic Proof Engine**: `SymbolicValidator` verifies LLM answers against retrieved graph paths (A → REL → B). Contradictions trigger a hard block (score < 1.0).
- Fail-closed `ConsistencyEvaluator` (GPT-4 Turbo) with hard timeout.
- Iterative ReAct reasoning loop with clarification exhaustion.

### 5. Observability & Evaluation

- `MetricsCollector` with per-engine latency tracking, error rates, and LLM-as-Judge scores.
- `LatencyTimer` context manager for automatic operation timing.
- **OpenTelemetry / Jaeger**: Cross-service `trace_id` propagation via `core/tracing.py`. Auto-instruments FastAPI with Jaeger and OTLP exporters.
- RAGAS-inspired benchmark harness (Faithfulness, Answer Relevancy, Context Precision).
- Concurrent load testing (10 threads) and failure injection resilience tests.

---

## Tech Stack

| Component | Implemented Technology |
| :--- | :--- |
| **LLMs & Reasoning** | OpenAI GPT-4 / GPT-4 Turbo, GPT-3.5 Turbo |
| **Retrieval & Rerank** | PGVector, PostgreSQL FTS, Elasticsearch, Cross-Encoder, ColBERT, MonoT5 |
| **Vector Models** | `all-MiniLM-L6-v2` (text), `clip-ViT-B-32` (vision) |
| **Databases** | PostgreSQL + PGVector, Redis, Neo4j |
| **Frameworks** | FastAPI, LangChain, LlamaIndex (long-context) |
| **Observability** | `MetricsCollector`, OpenTelemetry, Jaeger |
| **Infrastructure** | K8s (HPA + Helm + Velero + External Secrets), Docker Compose |

---

## Roadmap

The following capabilities are **planned but not yet implemented**:

| Capability | Status |
| :--- | :--- |
| **PEFT / LoRA fine-tuning** | Planned |
| **BGE / Contriever / E5 embeddings** | Planned |
| **Claude 3 / Anthropic integration** | Planned |
| **Distributed tracing (Jaeger/OTel)** | **Active — core/tracing.py** |
| **Production Kubernetes** | **Active — HPA + Helm** |

---

## Project Structure

```bash
├── api/                  # FastAPI endpoints
├── core/                 # Configs, Memory, Auth, Observability
├── ingestion/            # ETL, Chunking, Multimodal parsers
│   ├── chunking/         # Embedding clustering
│   ├── graph_build/      # NER, Entity linking, Neo4j
│   └── long_context.py   # LlamaIndex hierarchical indexing
├── retrieval/            # Hybrid search coordinators
│   ├── dense/            # PGVector, FAISS, TableRetriever
│   ├── sparse/           # PostgreSQL FTS, Elasticsearch
│   ├── reranker/         # ColBERT, MonoT5, Cross-Encoder
│   └── graph/            # CypherGenerator (structured), EntityLinker
├── agent/                # State Machine, PlannerCritic, Query Parser
├── scripts/              # Warmup, demo scripts
├── tests/                # Unit, Integration, Benchmark, Resilience
├── docs/                 # Production checklist, Release notes
├── deploy/               # K8s manifests, Helm chart
├── docker-compose.yml    # Local infrastructure
├── requirements.txt      # Dependencies
└── api/main.py           # Application entry point
```

---

## Getting Started

### 1. Prerequisites

- Python 3.10+
- Docker + Docker Compose
- Git

### 2. Clone & Setup

```bash
git clone https://github.com/GinneZhang/AsterScope.git
cd AsterScope
python -m venv .venv
source .venv/bin/activate      # macOS/Linux
# .venv\Scripts\Activate.ps1   # Windows PowerShell
pip install -r requirements.txt
cp .env.example .env
# Fill in .env with your keys
```

### 3. Start Infrastructure

```bash
docker-compose up -d
docker ps   # Verify containers
```

### 4. Run the Warmup

**CRITICAL**: Run the warmup script before starting the API to eliminate first-query latency.

```bash
python scripts/warmup.py
```

This script will:

1. Verify all database connections (PostgreSQL, Redis, Neo4j, Elasticsearch)
2. Pre-load `all-MiniLM-L6-v2` (text embeddings)
3. Pre-load `clip-ViT-B-32` (vision embeddings)
4. Pre-load Cross-Encoder reranker
5. Pre-load MonoT5 reranker (if `RERANKER_TYPE=monot5`)
6. Pre-load spaCy NER model

Expected output:
```
[WARMUP] PostgreSQL — connected
[WARMUP] Redis — connected
[WARMUP] Neo4j — connected
[WARMUP] SentenceTransformer (all-MiniLM-L6-v2) — loaded in 2.1s
[WARMUP] CLIP (clip-ViT-B-32) — loaded in 3.4s
[WARMUP] Cross-Encoder (ms-marco-MiniLM-L-6-v2) — loaded in 1.8s
[WARMUP] All critical systems ready. AsterScope is warm!
```

### 5. Launch the API Server

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

> Entry point is `api.main:app`, not `main:app`.

### 6. Open API Docs

```text
http://localhost:8000/docs      # Swagger UI
http://localhost:8000/health    # Health Check
```

---

## Ingest a Document

```bash
curl -X POST "http://localhost:8000/ingest" \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Acme Corp Policy 404",
    "section": "Section 3.1",
    "document_text": "Employees possessing material non-public information are prohibited from trading company stock..."
  }'
```

Ingestion pipeline:
1. Raw text → semantic chunks
2. Chunks → embeddings (PGVector)
3. Chunks → Neo4j graph nodes

---

## Ask a Question

```bash
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What happens if I violate the insider trading policy?",
    "session_id": "your-session-id",
    "top_k": 5
  }'
```

The `/ask` endpoint returns a streaming NDJSON response with intermediate thoughts and the final grounded answer.

---

## Search by Image (True Multimodal RAG)

```bash
curl -X POST "http://localhost:8000/ask_vision" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/query_image.jpg" \
  -F "top_k=5"
```

Embeds your image into a 512-dimensional CLIP vector and retrieves the closest text chunks.

---

## Run Tests

```bash
# Full test suite
pytest tests/ -v --ignore=tests/benchmark_rag.py --ignore=tests/load_test.py

# Benchmarks (Ragas-based)
python tests/benchmark_rag.py

# HotpotQA 100-sample (generation contexts -> Ragas)
BENCHMARK_NAME=hotpotqa BENCHMARK_SAMPLE_SIZE=100 BENCHMARK_SPLIT=validation BENCHMARK_CONTEXT_MODE=generation python tests/benchmark_rag.py

# HotpotQA 100-sample (retrieval contexts -> Ragas)
BENCHMARK_NAME=hotpotqa BENCHMARK_SAMPLE_SIZE=100 BENCHMARK_SPLIT=validation BENCHMARK_CONTEXT_MODE=retrieval python tests/benchmark_rag.py

# SQuAD 100-sample (generation contexts -> Ragas)
BENCHMARK_NAME=squad BENCHMARK_SAMPLE_SIZE=100 BENCHMARK_SPLIT=validation BENCHMARK_CONTEXT_MODE=generation python tests/benchmark_rag.py

# SQuAD 2.0 100-sample (retrieval contexts -> Ragas)
BENCHMARK_NAME=squad_v2 BENCHMARK_SAMPLE_SIZE=100 BENCHMARK_SPLIT=validation BENCHMARK_CONTEXT_MODE=retrieval python tests/benchmark_rag.py

# Load test
python tests/load_test.py
```

The benchmark runner now uses Ragas as the primary evaluation framework. Official metrics are reported through Ragas, while AsterScope-specific chain/debug fields are preserved as side-channel diagnostics. See [docs/ragas_evaluation_migration.md](docs/ragas_evaluation_migration.md).

---

## Evidence Quality Controls

AsterScope now supports explicit feature flags for multi-hop evidence orchestration and ablation:

```bash
ENABLE_EARLY_SECOND_HOP=true|false
ENABLE_BRIDGE_PLANNER=true|false
ENABLE_DUAL_HEAD_SCORING=true|false
ENABLE_RAW_LEXICAL_RECALL=true|false
ENABLE_RETRIEVAL_DEBUG=true|false
```

When `ENABLE_RETRIEVAL_DEBUG=true`, `/ask` metadata includes stage-level retrieval diagnostics (candidate counts, source-type/evidence-role mix, dedup impact, bridge/support coverage) so you can trace where evidence quality degrades.

For a full architecture audit and rationale, see [docs/retrieval_evidence_audit.md](docs/retrieval_evidence_audit.md).

---

## Troubleshooting

| Issue | Solution |
| :--- | :--- |
| `OPENAI_API_KEY missing` | Add valid key to `.env` |
| Neo4j auth failed | Match `.env` credentials with Docker |
| PostgreSQL connection | Verify container is running |
| Redis connection | Verify port 6379 |
| spaCy model missing | `python -m spacy download en_core_web_sm` |
| Slow first startup | Run `python scripts/warmup.py` first |

---

## Recommended Startup Flow

```bash
git clone https://github.com/GinneZhang/AsterScope.git
cd AsterScope
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # Fill in keys
docker-compose up -d
python scripts/warmup.py          # Pre-load models
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

Then:
1. Open `http://localhost:8000/docs`
2. Call `/ingest` to add documents
3. Call `/ask` to query

---

**Version**: v1.0.0 | **License**: MIT | **Built with ❤️ by Ginne**
