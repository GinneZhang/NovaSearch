# 🚀 NovaSearch v1.0.0 — Release Notes

**[English](release_notes_v1.0.0.md) | [中文文档](release_notes_v1.0.0_cn.md)**

**Release Date**: 2026-03-12
**Commit**: `feat(prod): phase 14 terminal deterministic reasoning and 6-doc bilingual sync`

---

## Overview

NovaSearch v1.0.0 is the first production-ready release of the Enterprise Copilot & Intelligent Retrieval Engine. This version represents **12 phases of iterative architectural evolution**, progressing from a functional MVP to a production-hardened, enterprise-grade platform.

---

## 🏗️ Architecture Highlights

### Tri-Engine Fusion
The core of NovaSearch is the "Tri-Engine Fusion" architecture combining LLM semantics, hybrid multimodal RAG, and knowledge graph reasoning.

### Phase 1–4: Foundation

- FastAPI streaming API with NDJSON response format
- PGVector dense retrieval + PostgreSQL FTS sparse retrieval
- Neo4j knowledge graph with Text-to-Cypher
- Redis-backed semantic memory for multi-turn conversations
- Proactive clarification prompting for ambiguous queries
- Entity normalization and graph population

### Phase 5–6: Reranking & Multimodal

- **MonoT5 Reranker**: Pluggable transformer-based reranking alongside Cross-Encoder and ColBERT
- **CLIP Multimodal**: True cross-modal search via `clip-ViT-B-32` joint embedding space
- **Knowledge Graph Constraints**: Symbolic grounding to suppress LLM hallucinations

### Phase 7–8: Hardening & ReAct

- All missing files materialized (vision_search, monot5_reranker, etc.)
- **Iterative ReAct Loop**: Reason → Act → Observe → Re-reason (max 3 iterations)
- **Self-Healing Cypher**: Error-feedback repair loop (max 2 retries)

### Phase 9–10: Query Graphs & Table Extraction

- Structured semantic triplet extraction from user queries
- Table-aware PDF/DOCX extraction via pdfplumber / python-docx
- Fail-closed ConsistencyEvaluator (GPT-4 Turbo)
- README honest reconciliation — aspirational tech moved to Roadmap

### Phase 11: Total Alignment

- **Formal Agent State Machine**: `ConversationState` + `DependencyGraph` for Plan-and-Execute flows
- **Vector Entity Linker**: Embedding similarity against Neo4j Entity nodes
- **Dynamic Schema Introspection**: Live `CALL db.labels()` / `CALL db.relationshipTypes()`
- **Table-Specific Retrieval**: Dedicated `TableRetriever` with metadata filtering
- **RAGAS Benchmark Harness**: Faithfulness, Answer Relevancy, Context Precision

### Phase 12: Terminal Alignment

- **Planner-Critic Loop**: Plan → Execute → Evaluate → Re-plan with `PlannerCritic`
- **Cypher Linting & Validation**: Schema-aware pre-execution validator
- **Probabilistic Entity Linking**: Confidence < 0.8 triggers clarification
- **Deep Table Embeddings**: Semantic "Table Summary + Column Description" headers
- **Observability Hooks**: `MetricsCollector` + `LatencyTimer` per-engine telemetry
- **Failure Injection Tests**: 9 resilience tests (timeouts, invalid JSON, write rejection)
- **LlamaIndex Integration**: Long-context hierarchical indexing for documents > 50k chars

---

## 🛠 Tech Stack

| Component | Technology |
| :--- | :--- |
| **LLMs** | GPT-4 / GPT-4 Turbo, GPT-3.5 Turbo |
| **Retrieval** | PGVector, PostgreSQL FTS, Elasticsearch |
| **Reranking** | Cross-Encoder, ColBERT, MonoT5 |
| **Vision** | CLIP `clip-ViT-B-32` |
| **Graph** | Neo4j (Cypher linting + self-healing) |
| **Memory** | Redis `StateManager` + SentenceTransformers |
| **Frameworks** | FastAPI, LangChain, LlamaIndex |
| **Observability** | `MetricsCollector` (latency, errors, LLM scores) |
| **Infrastructure** | K8s (HPA, liveness/readiness probes), Helm, Docker Compose |

---

## 🔒 Phase 13: Hardened Runtime

- **Pre-Flight Hallucination Interceptor**: Two-stage generator buffers full response, validates with ConsistencyEvaluator (score > 0.8), and safety-blocks failed responses.
- **Deterministic Semantic Planning**: QueryGraphParser validates triplets against live Neo4j schema; unknown terms auto-aligned to nearest valid match.
- **Structured Cypher Objects**: `CypherResult` dataclass with nodes, edges, properties, and validation status. Symbolic path validator checks label/rel existence before execution.
- **Deep Table Schema Summary**: `generate_schema_summary()` infers column types from data; `extract_structured_answer()` routes table queries through LLM extraction.
- **K8s Production Deployment**: HPA for API (2-10) and Retrieval (2-8) pods. Liveness probes: `/health` (API), PG+Redis pings (Retrieval). Helm chart for standardized deployment.

---

## 🔒 Phase 14: Sovereign Runtime

- **Hardened Ontology Alignment**: `OntologyManager` maps triplet terms to canonical schema via embedding similarity (threshold > 0.9). Unmapped terms trigger Clarification instead of guessing.
- **Symbolic Proof Engine**: `SymbolicValidator` with two-layer verification — structural pre-check + GPT-4 Turbo proof layer. Hard-blocks answers that contradict Knowledge Graph facts (score < 1.0). Fail-closed on errors.
- **OpenTelemetry / Jaeger Integration**: Cross-service `trace_id` propagation via `core/tracing.py`. Jaeger + OTLP exporters with FastAPI auto-instrumentation and `@traced` decorator.
- **Enterprise Ops**: Helm chart with External Secrets Operator (AWS Secrets Manager), Velero backup annotations (daily 2AM schedule), and tracing configuration.

---

## 📊 Test Results

```
21 passed, 0 failed
- 3 Cypher generation tests
- 3 E2E integration tests (health, ingest, ask)
- 3 Memory tests
- 3 Cypher linter resilience tests
- 2 Consistency evaluator resilience tests
- 1 Entity linker resilience test
- 3 Observability resilience tests
- 3 Additional unit tests
```

---

## 🔮 Roadmap

| Feature | Status |
| :--- | :--- |
| PEFT / LoRA fine-tuning | Planned |
| BGE / Contriever embeddings | Planned |
| Distributed tracing (Jaeger/OTel) | **Active — core/tracing.py** |
| Production Kubernetes | **Active — HPA + Helm** |

---

**Built with ❤️ by the NovaSearch Team**
