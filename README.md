# 🚀 NovaSearch: Enterprise Copilot & Intelligent Retrieval Engine

NovaSearch is a strong foundation for an enterprise knowledge retrieval and reasoning system. It implements a **"Tri-Engine Fusion"** approach: **LLM Semantics + Hybrid Multimodal RAG + Knowledge Graph Reasoning**.

Our objective is to deliver highly accurate, explainable, and hallucination-resistant Copilot experiences for enterprise SOPs, compliance documents, and structured knowledge.

## 🧠 System Architecture & Core Pillars

1. **Query Understanding & Context Memory**
   - **Structuring**: LLM-driven query rewriting, intent recognition, and structured semantic graph generation (triplet extraction).
   - **Decomposition**: Multi-hop task breakdown via `DependencyGraph` with Plan-and-Execute sub-task tracking.
   - **Clarification**: Stateful clarification loop persisted across API calls via Redis `StateManager`.
   - **Context**: Semantic cross-session thread linking and memory state management via Redis + SentenceTransformers.

2. **Hybrid Multimodal Retrieval**
   - **Tri-Retrieval Fusion**: Integrating Sparse (PostgreSQL FTS / Elasticsearch), Dense (PGVector with `all-MiniLM-L6-v2`), Multimodal Vision (OpenAI CLIP `clip-ViT-B-32`) and Structural (Neo4j Cypher Graph Traversal) pathways.
   - **Table-Specific Retrieval**: Dedicated `TableRetriever` filtering on `metadata.type == 'table'` chunks with table-aware reranking.
   - **Table-Aware Parsing**: PDF and DOCX table extraction via pdfplumber and python-docx, formatted as Markdown grids.
   - **Cross-Modal Search**: The `/ask_vision` endpoint maps images to the shared CLIP vector space for retrieval.
   - **Advanced Chunking**: Sliding Window + Embedding Clustering to preserve semantic context boundaries.
   - **Reranking**: Cross-Encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`), with pluggable ColBERT and MonoT5.

3. **Knowledge Graph Reasoning (KG)**
   - **Vector Entity Linking**: Embedding similarity search against Neo4j Entity nodes for precision disambiguation.
   - **Dynamic Schema Introspection**: `CypherGenerator` fetches live schema via `CALL db.labels()` / `CALL db.relationshipTypes()` before generating queries.
   - **Self-Healing Cypher**: Error-feedback retry loop (max 2 retries) auto-repairs invalid Cypher.
   - **Factual Grounding**: KG constraints injected into generative context to suppress LLM hallucinations.

4. **Controlled LLM Generation**
   - Source-grounded QA with mandatory origin tracing for all outputs.
   - Real-time consistency scoring via fail-closed `ConsistencyEvaluator` (GPT-4 Turbo).
   - Iterative ReAct reasoning loop with clarification exhaustion.

5. **Quantitative Evaluation**
   - RAGAS-inspired benchmark harness measuring Faithfulness, Answer Relevancy, and Context Precision.
   - Concurrent load testing (10 threads) for latency and throughput measurement.

## 🛠 Current Tech Stack

| Component | Implemented Technology |
| :--- | :--- |
| **LLMs & Reasoning** | OpenAI GPT-4 / GPT-4 Turbo, GPT-3.5 Turbo |
| **Retrieval & Rerank** | PGVector, PostgreSQL FTS, Elasticsearch, Cross-Encoder, ColBERT, MonoT5 |
| **Vector Models** | `all-MiniLM-L6-v2` (text), `clip-ViT-B-32` (vision) |
| **Databases** | PostgreSQL + PGVector, Redis, Neo4j |
| **Frameworks** | FastAPI, LangChain |
| **Infrastructure** | Docker Compose (PostgreSQL, Redis, Neo4j, Elasticsearch) |

## 🗺️ Roadmap / Future Architecture

The following capabilities are **planned but not yet implemented**:

| Capability | Status |
| :--- | :--- |
| **PEFT / LoRA fine-tuning** | Planned — no local fine-tuning pipeline exists yet |
| **LlamaIndex orchestration** | Planned — currently using custom LangChain + FastAPI orchestration |
| **BGE / Contriever / E5 embeddings** | Planned — currently using `all-MiniLM-L6-v2` for text |
| **Claude 3 / Anthropic integration** | Planned — API key support exists but not wired into generation |
| **Full observability harness** | Planned — basic logging in place, no distributed tracing |
| **Production Kubernetes deployment** | Planned — currently Docker Compose only |

## 📁 Project Structure

```bash
├── api/                  # FastAPI endpoints, streaming routers, schemas
├── core/                 # App configs, Context memory (Redis), auth
├── ingestion/            # ETL pipelines, Sliding Window chunking, Multimodal parsers
│   ├── chunking/         # Embedding clustering & metadata injection logic
│   └── graph_build/      # NER, Entity linking, Neo4j population
├── retrieval/            # Hybrid search coordinators
│   ├── dense/            # Vector DB interfaces (PGVector, FAISS, TableRetriever)
│   ├── sparse/           # Keyword indexing (PostgreSQL FTS, Elasticsearch)
│   ├── reranker/         # ColBERT, MonoT5, Cross-Encoder
│   └── graph/            # CypherGenerator (self-healing), EntityLinker (vector-based)
├── agent/                # LLM reasoning loops, State Machine, Query Parser
├── tests/                # Unit/Integration/Benchmark test suites
├── docker-compose.yml    # Local Core Infra (Postgres, Redis, Neo4j, Elasticsearch)
├── requirements.txt      # Dependencies
└── api/main.py           # Application entry point
```

## 🚀 Getting Started (Local Development)

### 1. Prerequisites
Make sure you have the following installed locally:

- Python 3.10+
- Docker + Docker Compose
- Git

---

### 2. Clone the repository

```bash
git clone https://github.com/GinneZhang/NovaSearch.git
cd NovaSearch
```

---

### 3. Create and activate a virtual environment

**macOS / Linux**

```bash
python -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell)**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

---

### 4. Install dependencies

Install runtime dependencies:

```bash
pip install -r requirements.txt
```

If you also want to run tests:

```bash
pip install -r requirements_dev.txt
```

---

### 5. Create your `.env` file

Copy the template:

```bash
cp .env.example .env
```

Then open `.env` and fill in the required values.

Example:

```env
# ==========================================
# NovaSearch Environment Variables
# ==========================================

# 1. LLM / API Keys
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# 2. PostgreSQL
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres_secure_password
POSTGRES_DB=novasearch
POSTGRES_HOST=localhost
POSTGRES_PORT=5432

# 3. Redis
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=

# 4. Neo4j
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=neo4j_secure_password

# 5. App settings
API_ENV=development
LOG_LEVEL=DEBUG
```

#### Required values to check carefully

- `OPENAI_API_KEY` is required for query rewriting, intent classification, and answer generation.
- `POSTGRES_PASSWORD` must match the password used by Docker Compose.
- `NEO4J_PASSWORD` must match the password used by Docker Compose.
- `REDIS_PASSWORD` can stay empty if you use the current local Docker setup.

---

### 6. Start local infrastructure

This project depends on:

- PostgreSQL with pgvector
- Redis
- Neo4j

Start them with:

```bash
docker-compose up -d
```

Check that the containers are running:

```bash
docker ps
```

Optional: inspect logs if something fails

```bash
docker-compose logs -f
```

---

### 7. Important first-run note

On the first run, NovaSearch may automatically download:

- spaCy model: `en_core_web_sm`
- SentenceTransformer embedding model: `all-MiniLM-L6-v2`
- Cross-encoder reranker model: `cross-encoder/ms-marco-MiniLM-L-6-v2`

So the first startup can take longer than usual and requires internet access.

---

### 8. Launch the API server

Run the app from the repository root with:

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

> Note: the entrypoint is `api.main:app`, not `main:app`.

At startup, the app will automatically:

- initialize PostgreSQL tables and the pgvector extension
- create Neo4j constraints
- initialize the copilot agent, chunker, and KG builder

---

### 9. Open the API docs

After startup, visit:

```text
http://localhost:8000/docs
```

Health check endpoint:

```text
http://localhost:8000/health
```

---

## 📥 Ingest a document

Before asking questions, you should ingest at least one document.

Example:

```bash
curl -X POST "http://localhost:8000/ingest" \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Acme Corp Policy 404",
    "section": "Section 3.1",
    "document_text": "Employees possessing material non-public information are prohibited from trading company stock..."
  }'
```

What happens during ingestion:

1. The raw text is split into semantic chunks.
2. Chunk embeddings are generated.
3. Chunks are stored in PostgreSQL / pgvector.
4. Document and chunk relationships are written into Neo4j.

---

## 💬 Ask a question

Once at least one document has been ingested, query the system:

```bash
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What happens if I violate the insider trading policy?",
    "top_k": 5
  }'
```

You can also pass a `session_id` to enable multi-turn memory:

```bash
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Who approves exceptions?",
    "session_id": "your-session-id-here",
    "top_k": 5
  }'
```

The `/ask` endpoint returns a streaming NDJSON response with intermediate status/thought messages and the final grounded answer.

---

## 🖼️ Search by Image (True Multimodal RAG)

You can query the common CLIP vector space using an image via the `/ask_vision` endpoint:

```bash
curl -X POST "http://localhost:8000/ask_vision" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/query_image.jpg" \
  -F "top_k=5"
```

This will embed your image into a 512-dimensional vector and retrieve the closest text chunks or graphs ingested previously.

---

## 🧪 Run tests

If you installed dev dependencies, you can run:

```bash
pytest tests/ -v
```

Before running tests, make sure:

- Docker services are up.
- Your `.env` is configured.
- Your `OPENAI_API_KEY` is available if you want full end-to-end behavior.

---

## 🛠 Troubleshooting

### 1. `OPENAI_API_KEY missing`
Your `.env` is missing a valid OpenAI API key.

### 2. Neo4j authentication failed
Make sure `NEO4J_USER` / `NEO4J_PASSWORD` in `.env` match the values used by Docker Compose.

### 3. PostgreSQL connection failed
Make sure the Postgres container is running and that your `.env` values match the container config.

### 4. Redis connection failed
Make sure the Redis container is running on port `6379`.

### 5. spaCy model missing
If automatic download fails, install it manually:

```bash
python -m spacy download en_core_web_sm
```

### 6. First startup is very slow
This is expected if HuggingFace or spaCy models are being downloaded for the first time.

---

## ✅ Recommended local startup flow

From a clean clone, the usual flow is:

```bash
git clone https://github.com/GinneZhang/NovaSearch.git
cd NovaSearch
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# fill in .env
docker-compose up -d
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

Then:

1. Open `http://localhost:8000/docs`
2. Call `/ingest`
3. Call `/ask`

