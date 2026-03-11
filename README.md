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
| **Infrastructure** | Docker Compose (Local), Kubernetes (Production) |

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

## 🚀 Getting Started (Local Development)

### 1. Prerequisites
* Python 3.10+
* Docker & Docker Compose
* Git

### 2. Installation
git clone https://github.com/YourUsername/NovaSearch.git
cd NovaSearch
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt

### 3. Environment Setup
Copy the configuration template and populate your API Keys (OpenAI, Anthropic) and DB credentials:
cp .env.example .env

### 4. Spin up Core Infrastructure
Start the local persistence layer (PostgreSQL w/ pgvector, Redis, Neo4j) via Docker:
docker-compose up -d

### 5. Launch the Application
uvicorn main:app --reload --host 0.0.0.0 --port 8000
Interactive API documentation available at: http://localhost:8000/doc.

