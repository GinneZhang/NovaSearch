# 🚀 NovaSearch v1.0.0 — Release Notes / 发布说明

**Release Date / 发布日期**: 2026-03-12
**Commit / 提交**: `release(v1.0.0): terminal alignment, bilingual docs, and warmup engine`

---

## Overview / 概述

NovaSearch v1.0.0 is the first production-ready release of the Enterprise Copilot & Intelligent Retrieval Engine. This version represents **12 phases of iterative architectural evolution**, progressing from a functional MVP to a production-hardened, enterprise-grade platform.

NovaSearch v1.0.0 是企业级知识检索与推理引擎的首个生产就绪版本。此版本代表了 **12 个阶段的迭代架构演进**，从功能性 MVP 发展为生产加固的企业级平台。

---

## 🏗️ Architecture Highlights / 架构亮点

### Tri-Engine Fusion / 三引擎融合
The core of NovaSearch is the "Tri-Engine Fusion" architecture combining LLM semantics, hybrid multimodal RAG, and knowledge graph reasoning.

NovaSearch 的核心是「三引擎融合」架构，结合 LLM 语义、混合多模态 RAG 和知识图谱推理。

### Phase 1–4: Foundation / 阶段 1–4：基础

- FastAPI streaming API with NDJSON response format / FastAPI 流式 API（NDJSON 响应格式）
- PGVector dense retrieval + PostgreSQL FTS sparse retrieval / PGVector 稠密检索 + PostgreSQL FTS 稀疏检索
- Neo4j knowledge graph with Text-to-Cypher / Neo4j 知识图谱与 Text-to-Cypher
- Redis-backed semantic memory for multi-turn conversations / 基于 Redis 的语义记忆，支持多轮对话
- Proactive clarification prompting for ambiguous queries / 主动澄清提示，处理模糊查询
- Entity normalization and graph population / 实体规范化与知识图谱填充

### Phase 5–6: Reranking & Multimodal / 阶段 5–6：重排序与多模态

- **MonoT5 Reranker**: Pluggable transformer-based reranking alongside Cross-Encoder and ColBERT / MonoT5 重排序器：可插拔的 Transformer 重排序，与交叉编码器和 ColBERT 并行
- **CLIP Multimodal**: True cross-modal search via `clip-ViT-B-32` joint embedding space / CLIP 多模态：通过 `clip-ViT-B-32` 联合嵌入空间实现真正的跨模态搜索
- **Knowledge Graph Constraints**: Symbolic grounding to suppress LLM hallucinations / 知识图谱约束：符号化约束抑制 LLM 幻觉

### Phase 7–8: Hardening & ReAct / 阶段 7–8：加固与 ReAct

- All missing files materialized (vision_search, monot5_reranker, etc.) / 所有缺失文件已实现
- **Iterative ReAct Loop**: Reason → Act → Observe → Re-reason (max 3 iterations) / 迭代 ReAct 循环：推理 → 执行 → 观察 → 再推理（最多 3 次迭代）
- **Self-Healing Cypher**: Error-feedback repair loop (max 2 retries) / 自修复 Cypher：错误反馈修复循环（最多 2 次重试）

### Phase 9–10: Query Graphs & Table Extraction / 阶段 9–10：查询图谱与表格提取

- Structured semantic triplet extraction from user queries / 从用户查询中提取结构化语义三元组
- Table-aware PDF/DOCX extraction via pdfplumber / python-docx / 表格感知的 PDF/DOCX 提取
- Fail-closed ConsistencyEvaluator (GPT-4 Turbo) / 失败封锁一致性评估器（GPT-4 Turbo）
- README honest reconciliation — aspirational tech moved to Roadmap / README 真实性对齐——愿景性技术移至路线图

### Phase 11: Total Alignment / 阶段 11：全面对齐

- **Formal Agent State Machine**: `ConversationState` + `DependencyGraph` for Plan-and-Execute flows / 正式 Agent 状态机：用于规划与执行流程
- **Vector Entity Linker**: Embedding similarity against Neo4j Entity nodes / 向量实体链接器：与 Neo4j 实体节点的嵌入相似度搜索
- **Dynamic Schema Introspection**: Live `CALL db.labels()` / `CALL db.relationshipTypes()` / 动态 Schema 内省
- **Table-Specific Retrieval**: Dedicated `TableRetriever` with metadata filtering / 表格专用检索路径
- **RAGAS Benchmark Harness**: Faithfulness, Answer Relevancy, Context Precision / RAGAS 基准测试工具

### Phase 12: Terminal Alignment / 阶段 12：终极对齐

- **Planner-Critic Loop**: Plan → Execute → Evaluate → Re-plan with `PlannerCritic` / 规划-评判循环
- **Cypher Linting & Validation**: Schema-aware pre-execution validator / Cypher 语法检查与验证：Schema 感知的预执行验证器
- **Probabilistic Entity Linking**: Confidence < 0.8 triggers clarification / 概率实体链接：置信度 < 0.8 触发澄清
- **Deep Table Embeddings**: Semantic "Table Summary + Column Description" headers / 深度表格嵌入：语义化「表格摘要 + 列描述」标题
- **Observability Hooks**: `MetricsCollector` + `LatencyTimer` per-engine telemetry / 可观测性钩子：按引擎遥测
- **Failure Injection Tests**: 9 resilience tests (timeouts, invalid JSON, write rejection) / 故障注入测试：9 项弹性测试
- **LlamaIndex Integration**: Long-context hierarchical indexing for documents > 50k chars / LlamaIndex 集成：大文档（> 5 万字符）的分层索引

---

## 🛠 Tech Stack / 技术栈

| Component / 组件 | Technology / 技术 |
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

## 🔒 Phase 13: Hardened Runtime / 阶段 13：硬化运行时

- **Pre-Flight Hallucination Interceptor / 预检幻觉拦截器**: Two-stage generator buffers full response, validates with ConsistencyEvaluator (score > 0.8), and safety-blocks failed responses. / 两阶段生成器缓冲完整响应，通过一致性评估器验证（分数 > 0.8），安全阻断未通过验证的响应。
- **Deterministic Semantic Planning / 确定性语义规划**: QueryGraphParser validates triplets against live Neo4j schema; unknown terms auto-aligned to nearest valid match. / 查询图谱解析器验证三元组；未知术语自动对齐到最近有效匹配。
- **Structured Cypher Objects / 结构化 Cypher 对象**: `CypherResult` dataclass with nodes, edges, properties, and validation status. Symbolic path validator checks label/rel existence before execution. / `CypherResult` 数据类，包含节点、边、属性和验证状态。
- **Deep Table Schema Summary / 深度表格 Schema 摘要**: `generate_schema_summary()` infers column types from data; `extract_structured_answer()` routes table queries through LLM extraction. / 从数据推断列类型；表格查询通过 LLM 提取路由。
- **K8s Production Deployment / K8s 生产部署**: HPA for API (2-10) and Retrieval (2-8) pods. Liveness probes: `/health` (API), PG+Redis pings (Retrieval). Helm chart for standardized deployment. / API 和检索 Pod 的 HPA 自动扩缩。存活探针。Helm chart 标准化部署。

---

## 📊 Test Results / 测试结果

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

## 🔮 Roadmap / 路线图

| Feature / 功能 | Status / 状态 |
| :--- | :--- |
| PEFT / LoRA fine-tuning / 微调 | Planned / 计划中 |
| BGE / Contriever embeddings / 嵌入 | Planned / 计划中 |
| Distributed tracing (Jaeger/OTel) / 分布式追踪 | Planned / 计划中 |
| Production Kubernetes / 生产 K8s | **Active — HPA + Helm / 已实现** |

---

**Built with ❤️ by the NovaSearch Team**
**由 NovaSearch 团队用心打造**
