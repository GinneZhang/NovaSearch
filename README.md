# 🚀 NovaSearch: Enterprise Copilot & Intelligent Retrieval Engine

NovaSearch is a next-generation enterprise knowledge retrieval and reasoning system. Moving beyond traditional Keyword matching (BM25) and naive RAG architectures, NovaSearch utilizes a **"Tri-Engine Fusion"** approach: **LLM Semantics + Hybrid Multimodal RAG + Knowledge Graph Reasoning**. 

Our primary objective is to deliver highly accurate, explainable, and hallucination-free Copilot experiences for complex enterprise SOPs, medical guidelines, and legal frameworks.

## 🧠 System Architecture & Core Pillars

Our pipeline is designed for extreme reliability, logical reasoning, and factual grounding:

1. **Query Understanding & Context Memory**
   - **Structuring**: LLM-driven query rewriting, intent recognition, and structured semantic graph generation.
   - **Decomposition**: Multi-hop task breakdown and proactive clarification prompting for ambiguous queries.
   - **Context**: Agent-based cross-session thread linking and memory state management via Redis.

2. **Hybrid Multimodal Retrieval**
   - **Tri-Retrieval Fusion**: Integrating Sparse (BM25/Elasticsearch), Dense (FAISS/BGE for text/image/table vectors), and Structural (Neo4j SPARQL) pathways.
   - **Advanced Chunking**: Utilizing **Sliding Window + Embedding Clustering** to preserve semantic context boundaries.
   - **Reranking**: Deep contextual re-ranking via ColBERT or MonoT5.

3. **Knowledge Graph Reasoning (KG)**
   - **Disambiguation**: Precision Entity Linking and NER mapping queries to graph nodes.
   - **Multi-hop Traversal**: Symbolic + Neural graph traversal to unearth implicit relationships.
   - **Factual Grounding**: Utilizing KG constraints to structurally suppress LLM hallucinations.

4. **Controlled LLM Generation**
   - Source-grounded QA (mandatory origin tracing for all outputs).
   - Real-time consistency scoring and format alignment (Summaries, Tables, Decision Trees) via Streaming LLM.

## 🛠 Tech Stack Blueprint

| Component | Selected Technology |
| :--- | :--- |
| **LLMs & Reasoning** | GPT-4, Claude 3, LLaMA/Mistral + PEFT/LoRA |
| **Retrieval & Rerank** | FAISS, Elasticsearch, BM25, ColBERT |
| **Vector Models** | BGE, E5, Contriever, OpenAI Embedding |
| **Databases** | PostgreSQL + PGVector, Redis, Neo4j |
| **Frameworks** | FastAPI, LlamaIndex, LangChain |
| **Infrastructure** | Docker Compose (Local & Production Ready) |

## 📁 Project Structure

```bash
├── api/                  # FastAPI endpoints, streaming routers, schemas
├── core/                 # App configs, Context memory (Redis), auth
├── ingestion/            # ETL pipelines, Sliding Window chunking, Multimodal parsers
│   ├── chunking/         # Embedding clustering & metadata injection logic
│   └── graph_build/      # NER, Entity linking, Neo4j population
├── retrieval/            # Hybrid search coordinators
│   ├── dense/            # Vector DB interfaces (PGVector/FAISS)
│   ├── sparse/           # Keyword indexing
│   └── reranker/         # ColBERT/MonoT5 integration
├── agent/                # LLM reasoning loops, tool-use, CoT clarification
├── tests/                # Unit/Integration test suites
├── docker-compose.yml    # Local Core Infra (Postgres, Redis, Neo4j)
├── requirements.txt      # Dependencies
└── main.py               # Application entry point
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

## 🧪 Run tests

If you installed dev dependencies, you can run:

```bash
pytest tests/test_e2e.py -v
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

