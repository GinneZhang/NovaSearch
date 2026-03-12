# 🚀 NovaSearch：企业级智能检索与推理引擎

**[English](README.md) | [中文文档](README_CN.md)**

NovaSearch 是一套生产级企业知识检索与推理系统，采用**「三引擎融合」**架构：**LLM 语义 + 混合多模态 RAG + 知识图谱推理**。

我们的目标是为企业 SOP、合规文档和结构化知识提供高精度、可解释、抗幻觉的 Copilot 体验。

---

## 🧠 系统架构与核心支柱

### 1. 查询理解与上下文记忆

- **结构化**：LLM 驱动的查询重写、意图识别和结构化语义图谱生成（三元组提取）。
- **分解**：通过 `DependencyGraph` 实现多跳任务分解与规划执行子任务跟踪。
- **规划-评判循环**：使用 `PlannerCritic` 实现迭代「规划 → 执行 → 评估 → 再规划」循环。
- **澄清**：具有 SUSPENDED/REPLANNING 状态的有状态澄清循环，通过 Redis `StateManager` 持久化。
- **上下文**：基于 Redis + SentenceTransformers 的跨会话语义记忆。

### 2. 混合多模态检索

- **三路融合**：稀疏（PostgreSQL FTS / Elasticsearch）+ 稠密（PGVector `all-MiniLM-L6-v2`）+ 视觉（CLIP `clip-ViT-B-32`）+ 结构化（Neo4j Cypher）四路检索融合。
- **深度表格嵌入**：语义化「表格摘要 + 列描述」标题，替代原始 Markdown 嵌入。
- **表格专用检索**：基于 `metadata.type == 'table'` 的 `TableRetriever`，支持过滤与结构化值提取。
- **长上下文处理**：LlamaIndex 驱动的 `LongContextProcessor`，处理超过 5 万字符的文档。
- **跨模态搜索**：`/ask_vision` 将图像映射到共享 CLIP 向量空间。
- **高级分块**：滑动窗口 + 嵌入聚类。
- **重排序**：交叉编码器，可插拔 ColBERT 和 MonoT5。

### 3. 知识图谱推理

- **概率实体链接**：向量相似度置信度评分；模糊匹配（置信度 < 0.8）触发主动用户澄清。
- **Cypher 校验与验证**：Schema 感知验证器在执行前拒绝写操作和未知标签。
- **结构化 Cypher 对象**：`CypherResult` 数据类，包含节点、边、属性和路径验证。
- **符号路径验证**：Cypher 执行前检查标签/关系是否存在于实时 Schema。
- **强化本体对齐**：`OntologyManager` 通过嵌入相似度（严格阈值 > 0.9）将三元组术语映射到规范 Schema 类。未映射的术语触发澄清请求。
- **动态 Schema 内省**：实时 `CALL db.labels()` / `CALL db.relationshipTypes()` 查询与属性取样。
- **自修复 Cypher**：错误反馈重试循环（最多 2 次）自动修复无效查询。
- **事实约束**：知识图谱约束注入生成上下文以抑制 LLM 幻觉。

### 4. 受控 LLM 生成

- 溯源式问答，强制标注信息来源。
- **预检幻觉拦截器**：高风险查询的两阶段缓冲生成；一致性评估器在流式传输前验证（阈值 > 0.8）。
- **符号证明引擎**：`SymbolicValidator` 验证 LLM 答案是否符合检索到的图谱路径（A → REL → B）。矛盾触发硬阻断（分数 < 1.0）。
- 失败封锁的 `ConsistencyEvaluator`（GPT-4 Turbo），带硬超时。
- 迭代 ReAct 推理循环，带澄清穷尽机制。

### 5. 可观测性与评估

- `MetricsCollector` 按引擎追踪延迟、错误率和 LLM 评判分数。
- `LatencyTimer` 上下文管理器，自动计时。
- **OpenTelemetry / Jaeger**：通过 `core/tracing.py` 实现跨服务 `trace_id` 传播。自动集成 FastAPI，支持 Jaeger 和 OTLP 导出器。
- RAGAS 风格基准测试工具（忠实度、答案相关性、上下文精度）。
- 并发负载测试（10 线程）与故障注入弹性测试。

---

## 🛠 技术栈

| 组件 | 实现技术 |
| :--- | :--- |
| **LLM 与推理** | OpenAI GPT-4 / GPT-4 Turbo, GPT-3.5 Turbo |
| **检索与重排序** | PGVector, PostgreSQL FTS, Elasticsearch, Cross-Encoder, ColBERT, MonoT5 |
| **向量模型** | `all-MiniLM-L6-v2`（文本）, `clip-ViT-B-32`（视觉） |
| **数据库** | PostgreSQL + PGVector, Redis, Neo4j |
| **框架** | FastAPI, LangChain, LlamaIndex（长上下文） |
| **可观测性** | `MetricsCollector`、OpenTelemetry、Jaeger |
| **基础设施** | K8s（HPA + Helm + Velero + External Secrets）, Docker Compose |

---

## 🗺️ 路线图

以下功能**已规划但尚未实现**：

| 功能 | 状态 |
| :--- | :--- |
| **PEFT / LoRA 微调** | 计划中 |
| **BGE / Contriever / E5 嵌入** | 计划中 |
| **Claude 3 / Anthropic 集成** | 计划中 |
| **分布式追踪（Jaeger/OTel）** | **已实现 — core/tracing.py** |
| **生产 K8s** | **已实现 — HPA + Helm** |

---

## 📁 项目结构

```bash
├── api/                  # FastAPI 端点
├── core/                 # 配置、记忆、认证、可观测性
├── ingestion/            # ETL、分块、多模态解析器
│   ├── chunking/         # 嵌入聚类
│   ├── graph_build/      # NER、实体链接、Neo4j
│   └── long_context.py   # LlamaIndex 分层索引
├── retrieval/            # 混合搜索协调器
│   ├── dense/            # PGVector, FAISS, TableRetriever
│   ├── sparse/           # PostgreSQL FTS, Elasticsearch
│   ├── reranker/         # ColBERT, MonoT5, Cross-Encoder
│   └── graph/            # CypherGenerator（结构化）, EntityLinker
├── agent/                # 状态机、规划器、查询解析器
├── scripts/              # 预热与演示脚本
├── tests/                # 单元、集成、基准、弹性测试
├── docs/                 # 生产检查清单、发布说明
├── deploy/               # K8s 清单、Helm chart
├── docker-compose.yml    # 本地基础设施
├── requirements.txt      # 依赖
└── api/main.py           # 应用入口
```

---

## 🚀 快速开始

### 1. 前提条件

- Python 3.10+
- Docker + Docker Compose
- Git

### 2. 克隆与配置

```bash
git clone https://github.com/GinneZhang/NovaSearch.git
cd NovaSearch
python -m venv .venv
source .venv/bin/activate      # macOS/Linux
# .venv\Scripts\Activate.ps1   # Windows PowerShell
pip install -r requirements.txt
cp .env.example .env
# 在 .env 中填入你的密钥
```

### 3. 启动基础设施

```bash
docker-compose up -d
docker ps   # 验证容器
```

### 4. 🔥 运行预热

**重要**：在启动 API 前运行预热脚本以消除首次查询冷启动延迟。

```bash
python scripts/warmup.py
```

该脚本将：

1. ✅ 验证所有数据库连接（PostgreSQL, Redis, Neo4j, Elasticsearch）
2. ✅ 预加载 `all-MiniLM-L6-v2`（文本嵌入模型）
3. ✅ 预加载 `clip-ViT-B-32`（视觉嵌入模型）
4. ✅ 预加载交叉编码器重排序模型
5. ✅ 预加载 MonoT5 重排序器（如已配置 `RERANKER_TYPE=monot5`）
6. ✅ 预加载 spaCy NER 模型

预期输出：
```
[WARMUP] ✅ PostgreSQL — connected
[WARMUP] ✅ Redis — connected
[WARMUP] ✅ Neo4j — connected
[WARMUP] ✅ SentenceTransformer (all-MiniLM-L6-v2) — loaded in 2.1s
[WARMUP] ✅ CLIP (clip-ViT-B-32) — loaded in 3.4s
[WARMUP] ✅ Cross-Encoder (ms-marco-MiniLM-L-6-v2) — loaded in 1.8s
[WARMUP] ✅ All critical systems ready. NovaSearch is warm!
```

### 5. 启动 API 服务

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

> 入口点为 `api.main:app`，非 `main:app`。

### 6. 打开 API 文档

```text
http://localhost:8000/docs      # Swagger UI
http://localhost:8000/health    # 健康检查
```

---

## 📥 摄取文档

```bash
curl -X POST "http://localhost:8000/ingest" \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Acme Corp Policy 404",
    "section": "Section 3.1",
    "document_text": "Employees possessing material non-public information are prohibited from trading company stock..."
  }'
```

摄取流程：
1. 原始文本 → 语义分块
2. 分块 → 嵌入向量（PGVector）
3. 分块 → Neo4j 图谱节点

---

## 💬 提问

```bash
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "如果我违反内幕交易政策会怎样？",
    "session_id": "your-session-id",
    "top_k": 5
  }'
```

`/ask` 端点返回流式 NDJSON 响应，包含中间推理过程和最终溯源答案。

---

## 🖼️ 图像搜索（真正的多模态 RAG）

```bash
curl -X POST "http://localhost:8000/ask_vision" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/query_image.jpg" \
  -F "top_k=5"
```

将图像嵌入 512 维 CLIP 向量空间并检索最相关的文本分块。

---

## 🧪 运行测试

```bash
# 完整测试套件（21 项测试）
pytest tests/ -v --ignore=tests/benchmark_rag.py --ignore=tests/load_test.py

# 基准测试
python tests/benchmark_rag.py

# 负载测试
python tests/load_test.py
```

---

## 🛠 故障排查

| 问题 | 解决方案 |
| :--- | :--- |
| `OPENAI_API_KEY missing` | 在 `.env` 中添加有效密钥 |
| Neo4j 认证失败 | 确保 `.env` 与 Docker 配置一致 |
| PostgreSQL 连接失败 | 确认容器正在运行 |
| Redis 连接失败 | 确认端口 6379 |
| spaCy 模型缺失 | 运行 `python -m spacy download en_core_web_sm` |
| 首次启动慢 | 先运行 `python scripts/warmup.py` 预热脚本 |

---

## ✅ 推荐启动流程

```bash
git clone https://github.com/GinneZhang/NovaSearch.git
cd NovaSearch
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # 填入密钥
docker-compose up -d
python scripts/warmup.py          # 🔥 预加载模型
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

然后：
1. 打开 `http://localhost:8000/docs`
2. 调用 `/ingest` 添加文档
3. 调用 `/ask` 进行查询

---

**版本**: v1.0.0 | **许可**: MIT | **由 Ginne 用心打造 ❤️**
