# 🚀 NovaSearch: Enterprise Copilot & Intelligent Retrieval Engine
# 🚀 NovaSearch：企业级智能检索与推理引擎

NovaSearch is a production-hardened enterprise knowledge retrieval and reasoning system implementing **"Tri-Engine Fusion"**: **LLM Semantics + Hybrid Multimodal RAG + Knowledge Graph Reasoning**.

NovaSearch 是一套生产级企业知识检索与推理系统，采用**「三引擎融合」**架构：**LLM 语义 + 混合多模态 RAG + 知识图谱推理**。

Our objective is to deliver highly accurate, explainable, and hallucination-resistant Copilot experiences for enterprise SOPs, compliance documents, and structured knowledge.

我们的目标是为企业 SOP、合规文档和结构化知识提供高精度、可解释、抗幻觉的 Copilot 体验。

---

## 🧠 System Architecture & Core Pillars / 系统架构与核心支柱

### 1. Query Understanding & Context Memory / 查询理解与上下文记忆

- **Structuring / 结构化**: LLM-driven query rewriting, intent recognition, and structured semantic graph generation (triplet extraction). / LLM 驱动的查询重写、意图识别和结构化语义图谱生成（三元组提取）。
- **Decomposition / 分解**: Multi-hop task breakdown via `DependencyGraph` with Plan-and-Execute sub-task tracking. / 通过 `DependencyGraph` 实现多跳任务分解与规划执行子任务跟踪。
- **Planner-Critic Loop / 规划-评判循环**: Iterative plan → execute → evaluate → re-plan cycle with `PlannerCritic`. / 使用 `PlannerCritic` 实现迭代「规划 → 执行 → 评估 → 再规划」循环。
- **Clarification / 澄清**: Stateful clarification loop with SUSPENDED/REPLANNING states persisted via Redis `StateManager`. / 具有 SUSPENDED/REPLANNING 状态的有状态澄清循环，通过 Redis `StateManager` 持久化。
- **Context / 上下文**: Semantic cross-session thread linking and memory via Redis + SentenceTransformers. / 基于 Redis + SentenceTransformers 的跨会话语义记忆。

### 2. Hybrid Multimodal Retrieval / 混合多模态检索

- **Tri-Retrieval Fusion / 三路融合**: Sparse (PostgreSQL FTS / Elasticsearch) + Dense (PGVector `all-MiniLM-L6-v2`) + Vision (CLIP `clip-ViT-B-32`) + Structural (Neo4j Cypher). / 稀疏 + 稠密 + 视觉 + 结构化四路检索融合。
- **Deep Table Embeddings / 深度表格嵌入**: Semantic "Table Summary + Column Description" headers instead of raw Markdown embedding. / 语义化「表格摘要 + 列描述」标题，替代原始 Markdown 嵌入。
- **Table-Specific Retrieval / 表格专用检索**: `TableRetriever` with `metadata.type == 'table'` filtering and structured value extraction. / 基于 `metadata.type == 'table'` 的过滤与结构化值提取。
- **Long-Context Processing / 长上下文处理**: LlamaIndex-backed `LongContextProcessor` for documents exceeding 50k chars. / LlamaIndex 驱动的 `LongContextProcessor`，处理超过 5 万字符的文档。
- **Cross-Modal Search / 跨模态搜索**: `/ask_vision` maps images to the shared CLIP vector space. / `/ask_vision` 将图像映射到共享 CLIP 向量空间。
- **Advanced Chunking / 高级分块**: Sliding Window + Embedding Clustering. / 滑动窗口 + 嵌入聚类。
- **Reranking / 重排序**: Cross-Encoder, with pluggable ColBERT and MonoT5. / 交叉编码器，可插拔 ColBERT 和 MonoT5。

### 3. Knowledge Graph Reasoning (KG) / 知识图谱推理

- **Probabilistic Entity Linking / 概率实体链接**: Vector similarity with confidence scoring; ambiguous matches (conf < 0.8) trigger proactive user clarification. / 向量相似度置信度评分；模糊匹配（置信度 < 0.8）触发主动用户澄清。
- **Cypher Linting & Validation / Cypher 校验与验证**: Schema-aware validator rejects write operations and unknown labels before execution. / Schema 感知验证器在执行前拒绝写操作和未知标签。
- **Structured Cypher Objects / 结构化 Cypher 对象**: `CypherResult` dataclass with nodes, edges, properties, and path existence validation. / `CypherResult` 数据类，包含节点、边、属性和路径验证。
- **Symbolic Path Validator / 符号路径验证**: Checks label/rel existence against live schema before Cypher execution. / Cypher 执行前检查标签/关系是否存在于实时 Schema。
- **Dynamic Schema Introspection / 动态 Schema 内省**: Live `CALL db.labels()` / `CALL db.relationshipTypes()` with property sampling. / 实时 Schema 查询与属性取样。
- **Self-Healing Cypher / 自修复 Cypher**: Error-feedback retry loop (max 2 retries) auto-repairs invalid Cypher. / 错误反馈重试循环（最多 2 次）自动修复无效查询。
- **Factual Grounding / 事实约束**: KG constraints injected into generative context to suppress LLM hallucinations. / 知识图谱约束注入生成上下文以抑制 LLM 幻觉。

### 4. Controlled LLM Generation / 受控 LLM 生成

- Source-grounded QA with mandatory origin tracing. / 溯源式问答，强制标注信息来源。
- **Pre-Flight Hallucination Interceptor / 预检幻觉拦截器**: Two-stage buffered generation for high-stakes queries; ConsistencyEvaluator validates before streaming (threshold > 0.8). / 高风险查询的两阶段缓冲生成；一致性评估器在流式传输前验证（阈值 > 0.8）。
- Fail-closed `ConsistencyEvaluator` (GPT-4 Turbo) with hard timeout. / 失败封锁的 `ConsistencyEvaluator`（GPT-4 Turbo），带硬超时。
- Iterative ReAct reasoning loop with clarification exhaustion. / 迭代 ReAct 推理循环，带澄清穷尽机制。

### 5. Observability & Evaluation / 可观测性与评估

- `MetricsCollector` with per-engine latency tracking, error rates, and LLM-as-Judge scores. / `MetricsCollector` 按引擎追踪延迟、错误率和 LLM 评判分数。
- `LatencyTimer` context manager for automatic operation timing. / `LatencyTimer` 上下文管理器，自动计时。
- RAGAS-inspired benchmark harness (Faithfulness, Answer Relevancy, Context Precision). / RAGAS 风格基准测试工具（忠实度、答案相关性、上下文精度）。
- Concurrent load testing (10 threads) and failure injection resilience tests. / 并发负载测试（10 线程）与故障注入弹性测试。

---

## 🛠 Tech Stack / 技术栈

| Component / 组件 | Implemented Technology / 实现技术 |
| :--- | :--- |
| **LLMs & Reasoning** | OpenAI GPT-4 / GPT-4 Turbo, GPT-3.5 Turbo |
| **Retrieval & Rerank** | PGVector, PostgreSQL FTS, Elasticsearch, Cross-Encoder, ColBERT, MonoT5 |
| **Vector Models** | `all-MiniLM-L6-v2` (text / 文本), `clip-ViT-B-32` (vision / 视觉) |
| **Databases** | PostgreSQL + PGVector, Redis, Neo4j |
| **Frameworks** | FastAPI, LangChain, LlamaIndex (long-context / 长上下文) |
| **Observability / 可观测性** | `MetricsCollector` (per-engine latency, error rates, LLM scores) |
| **Infrastructure / 基础设施** | K8s (HPA + Helm), Docker Compose |

---

## 🗺️ Roadmap / Future Architecture / 路线图

The following capabilities are **planned but not yet implemented**:
以下功能**已规划但尚未实现**：

| Capability / 功能 | Status / 状态 |
| :--- | :--- |
| **PEFT / LoRA fine-tuning / 微调** | Planned / 计划中 |
| **BGE / Contriever / E5 embeddings / 嵌入** | Planned / 计划中 |
| **Claude 3 / Anthropic integration / 集成** | Planned / 计划中 |
| **Distributed tracing (Jaeger/OTel) / 分布式追踪** | Planned / 计划中 |
| **Production Kubernetes / 生产 K8s** | **Active — HPA + Helm / 已实现** |

---

## 📁 Project Structure / 项目结构

```bash
├── api/                  # FastAPI endpoints / FastAPI 端点
├── core/                 # Configs, Memory, Auth, Observability / 配置、记忆、认证、可观测性
├── ingestion/            # ETL, Chunking, Multimodal parsers / ETL、分块、多模态解析器
│   ├── chunking/         # Embedding clustering / 嵌入聚类
│   ├── graph_build/      # NER, Entity linking, Neo4j / NER、实体链接、Neo4j
│   └── long_context.py   # LlamaIndex hierarchical indexing / LlamaIndex 分层索引
├── retrieval/            # Hybrid search coordinators / 混合搜索协调器
│   ├── dense/            # PGVector, FAISS, TableRetriever / 稠密检索
│   ├── sparse/           # PostgreSQL FTS, Elasticsearch / 稀疏检索
│   ├── reranker/         # ColBERT, MonoT5, Cross-Encoder / 重排序
│   └── graph/            # CypherGenerator (structured), EntityLinker / 图谱推理
├── agent/                # State Machine, PlannerCritic, Query Parser / 状态机、规划器
├── scripts/              # Warmup, demo scripts / 预热与演示脚本
├── tests/                # Unit, Integration, Benchmark, Resilience / 测试
├── docs/                 # Production checklist, Release notes / 文档
├── deploy/               # K8s manifests, Helm chart / K8s 清单、Helm chart
├── docker-compose.yml    # Local infrastructure / 本地基础设施
├── requirements.txt      # Dependencies / 依赖
└── api/main.py           # Application entry point / 应用入口
```

---

## 🚀 Getting Started / 快速开始

### 1. Prerequisites / 前提条件

- Python 3.10+
- Docker + Docker Compose
- Git

### 2. Clone & Setup / 克隆与配置

```bash
git clone https://github.com/GinneZhang/NovaSearch.git
cd NovaSearch
python -m venv .venv
source .venv/bin/activate      # macOS/Linux
# .venv\Scripts\Activate.ps1   # Windows PowerShell
pip install -r requirements.txt
cp .env.example .env
# Fill in .env with your keys / 在 .env 中填入你的密钥
```

### 3. Start Infrastructure / 启动基础设施

```bash
docker-compose up -d
docker ps   # Verify containers / 验证容器
```

### 4. 🔥 Run the Warmup / 运行预热

**CRITICAL**: Run the warmup script before starting the API to eliminate first-query latency.
**重要**：在启动 API 前运行预热脚本以消除首次查询冷启动延迟。

```bash
python scripts/warmup.py
```

This script will:
该脚本将：

1. ✅ Verify all database connections (PostgreSQL, Redis, Neo4j, Elasticsearch) / 验证所有数据库连接
2. ✅ Pre-load `all-MiniLM-L6-v2` (text embeddings) / 预加载文本嵌入模型
3. ✅ Pre-load `clip-ViT-B-32` (vision embeddings) / 预加载视觉嵌入模型
4. ✅ Pre-load Cross-Encoder reranker / 预加载交叉编码器重排序模型
5. ✅ Pre-load MonoT5 reranker (if `RERANKER_TYPE=monot5`) / 预加载 MonoT5 重排序器（如已配置）
6. ✅ Pre-load spaCy NER model / 预加载 spaCy NER 模型

Expected output / 预期输出:
```
[WARMUP] ✅ PostgreSQL — connected
[WARMUP] ✅ Redis — connected
[WARMUP] ✅ Neo4j — connected
[WARMUP] ✅ SentenceTransformer (all-MiniLM-L6-v2) — loaded in 2.1s
[WARMUP] ✅ CLIP (clip-ViT-B-32) — loaded in 3.4s
[WARMUP] ✅ Cross-Encoder (ms-marco-MiniLM-L-6-v2) — loaded in 1.8s
[WARMUP] ✅ All critical systems ready. NovaSearch is warm!
```

### 5. Launch the API Server / 启动 API 服务

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

> Entry point is `api.main:app`, not `main:app`.
> 入口点为 `api.main:app`，非 `main:app`。

### 6. Open API Docs / 打开 API 文档

```text
http://localhost:8000/docs      # Swagger UI
http://localhost:8000/health    # Health Check / 健康检查
```

---

## 📥 Ingest a Document / 摄取文档

```bash
curl -X POST "http://localhost:8000/ingest" \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Acme Corp Policy 404",
    "section": "Section 3.1",
    "document_text": "Employees possessing material non-public information are prohibited from trading company stock..."
  }'
```

Ingestion pipeline / 摄取流程:
1. Raw text → semantic chunks / 原始文本 → 语义分块
2. Chunks → embeddings (PGVector) / 分块 → 嵌入向量
3. Chunks → Neo4j graph nodes / 分块 → Neo4j 图谱节点

---

## 💬 Ask a Question / 提问

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

`/ask` 端点返回流式 NDJSON 响应，包含中间推理过程和最终溯源答案。

---

## 🖼️ Search by Image / 图像搜索 (True Multimodal RAG / 真正的多模态 RAG)

```bash
curl -X POST "http://localhost:8000/ask_vision" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/query_image.jpg" \
  -F "top_k=5"
```

Embeds your image into a 512-dimensional CLIP vector and retrieves the closest text chunks.
将图像嵌入 512 维 CLIP 向量空间并检索最相关的文本分块。

---

## 🧪 Run Tests / 运行测试

```bash
# Full test suite / 完整测试套件 (21 tests)
pytest tests/ -v --ignore=tests/benchmark_rag.py --ignore=tests/load_test.py

# Benchmarks / 基准测试
python tests/benchmark_rag.py

# Load test / 负载测试
python tests/load_test.py
```

---

## 🛠 Troubleshooting / 故障排查

| Issue / 问题 | Solution / 解决方案 |
| :--- | :--- |
| `OPENAI_API_KEY missing` | Add valid key to `.env` / 在 `.env` 中添加有效密钥 |
| Neo4j auth failed / 认证失败 | Match `.env` credentials with Docker / 确保 `.env` 与 Docker 配置一致 |
| PostgreSQL connection / 连接失败 | Verify container is running / 确认容器正在运行 |
| Redis connection / 连接失败 | Verify port 6379 / 确认端口 6379 |
| spaCy model missing / 模型缺失 | `python -m spacy download en_core_web_sm` |
| Slow first startup / 首次启动慢 | Run `python scripts/warmup.py` first / 先运行预热脚本 |

---

## ✅ Recommended Startup Flow / 推荐启动流程

```bash
git clone https://github.com/GinneZhang/NovaSearch.git
cd NovaSearch
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # Fill in keys / 填入密钥
docker-compose up -d
python scripts/warmup.py          # 🔥 Pre-load models / 预加载模型
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

Then / 然后:
1. Open `http://localhost:8000/docs`
2. Call `/ingest` to add documents / 调用 `/ingest` 添加文档
3. Call `/ask` to query / 调用 `/ask` 进行查询

---

**Version / 版本**: v1.0.0 | **License / 许可**: MIT | **Built with ❤️ by Ginne**
