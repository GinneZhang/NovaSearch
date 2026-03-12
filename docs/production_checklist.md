# 🚀 NovaSearch v1.0.0 — Production Checklist

**[English](production_checklist.md) | [中文文档](production_checklist_cn.md)**

> **Purpose**: Complete verification checklist before deploying NovaSearch to a production environment.

---

## 1. Infrastructure

- [ ] **PostgreSQL + PGVector** — Running with `pgvector` extension enabled
- [ ] **Redis** — Running on configured port with AUTH if exposed
- [ ] **Neo4j** — Running with authentication and APOC plugin installed
- [ ] **Elasticsearch** — Running (optional, for hybrid sparse retrieval)
- [ ] **Docker Compose** — All containers healthy (`docker ps`)

---

## 2. Environment Variables

- [ ] `OPENAI_API_KEY` — Valid key with GPT-4 Turbo access
- [ ] `POSTGRES_*` vars match Docker Compose
- [ ] `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD` — Match container
- [ ] `REDIS_HOST`, `REDIS_PORT` — Correct
- [ ] `API_ENV=production` — Set to production mode
- [ ] `LOG_LEVEL=WARNING` — Reduce log noise

---

## 3. Model Pre-loading

Run the warmup script before accepting traffic:

```bash
python scripts/warmup.py
```

This pre-loads:

- [ ] `all-MiniLM-L6-v2` — SentenceTransformer text embeddings
- [ ] `clip-ViT-B-32` — CLIP vision-text embeddings
- [ ] `cross-encoder/ms-marco-MiniLM-L-6-v2` — Cross-Encoder reranker
- [ ] `castorini/monot5-base-msmarco` — MonoT5 reranker (if `RERANKER_TYPE=monot5`)
- [ ] `en_core_web_sm` — spaCy NER model

---

## 4. Functional Verification

- [ ] `GET /health` returns `{"status": "ok"}`
- [ ] `POST /ingest` with a test document succeeds
- [ ] `POST /ask` returns grounded answer with sources
- [ ] `POST /ask_vision` with test image returns matches
- [ ] Multi-turn memory works with `session_id`
- [ ] Consistency evaluator blocks hallucinated outputs

---

## 5. Tests

```bash
# Run full test suite
pytest tests/ -v --ignore=tests/benchmark_rag.py --ignore=tests/load_test.py

# Run benchmarks
python tests/benchmark_rag.py

# Run load test
python tests/load_test.py
```

- [ ] All unit/integration tests pass
- [ ] Benchmark scores meet minimum thresholds
- [ ] Load test P95 latency < 5s under 10 concurrent threads

---

## 6. Security

- [ ] `.env` is in `.gitignore`
- [ ] No API keys committed to repository
- [ ] Cypher linter enforces read-only queries
- [ ] ConsistencyEvaluator is fail-closed
- [ ] CORS policy is restrictive for production

---

## 7. Observability

- [ ] `MetricsCollector` is active (check `core/observability.py`)
- [ ] Per-engine latency is being logged
- [ ] Error rates are tracked
- [ ] Consistency scores are being recorded

---

## 8. Pre-Flight Guardrails (Phase 13)

- [ ] `PREFLIGHT_MODE` env var set (`auto`, `always`, or `never`)
- [ ] High-stakes queries trigger buffered validation
- [ ] Safety-block response confirmed for failed pre-flight checks
- [ ] ConsistencyEvaluator threshold > 0.8 enforced

---

## 9. Schema Validation (Phase 13)

- [ ] QueryGraphParser validates triplets against live Neo4j schema
- [ ] CypherGenerator returns `CypherResult` structured objects
- [ ] Symbolic path validator rejects unknown labels/relationships
- [ ] Table Schema Summary headers are generated for table chunks

---

## 10. Kubernetes Deployment (Phase 13)

- [ ] `deploy/k8s/api-deployment.yaml` applied
- [ ] `deploy/k8s/retrieval-deployment.yaml` applied
- [ ] HPA autoscaling verified (API: 2-10, Retrieval: 2-8)
- [ ] Liveness probes pass (`/health` for API, PG+Redis check for Retrieval)
- [ ] Readiness probes pass
- [ ] Helm chart deployed via `helm install novasearch deploy/helm/novasearch/`

---

## 11. Ontology Alignment (Phase 14)

- [ ] `OntologyManager` loads canonical labels, rels, and properties from Neo4j
- [ ] Embedding index built for schema terms (`all-MiniLM-L6-v2`)
- [ ] Unmapped triplet terms (confidence < 0.9) trigger Clarification response
- [ ] `ONTOLOGY_CONFIDENCE_THRESHOLD` env var set (default: 0.9)

---

## 12. Symbolic Proof Engine (Phase 14)

- [ ] `SymbolicValidator` structural pre-check operational
- [ ] GPT-4 Turbo proof layer validates answers against graph facts
- [ ] Contradictions (score < 1.0) trigger hard block
- [ ] Fail-closed on proof engine errors

---

## 13. OpenTelemetry / Jaeger (Phase 14)

- [ ] `OTEL_EXPORTER_OTLP_ENDPOINT` or `OTEL_EXPORTER_JAEGER_ENDPOINT` configured
- [ ] `core/tracing.py` initialized at application startup
- [ ] FastAPI auto-instrumented with `FastAPIInstrumentor`
- [ ] Cross-service `trace_id` propagation verified in logs
- [ ] Jaeger UI accessible for trace visualization

---

**Version**: v1.0.0 (Phase 14 — Sovereign Runtime)
**Date**: 2026-03-12
