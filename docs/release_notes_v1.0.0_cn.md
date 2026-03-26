# AsterScope v1.0.0 — 发布说明

<p align="center">
  <a href="release_notes_v1.0.0.md">[English]</a> | <a href="release_notes_v1.0.0_cn.md">[中文文档]</a>
</p>

**发布日期**：2026-03-12
**提交**：`feat(prod): phase 14 terminal deterministic reasoning and 6-doc bilingual sync`

---

## 概述

AsterScope v1.0.0 是企业级知识检索与推理引擎的首个生产就绪版本。此版本代表了 **12 个阶段的迭代架构演进**，从功能性 MVP 发展为生产加固的企业级平台。

---

## 🏗️ 架构亮点

### 三引擎融合
AsterScope 的核心是「三引擎融合」架构，结合 LLM 语义、混合多模态 RAG 和知识图谱推理。

### 阶段 1–4：基础

- FastAPI 流式 API（NDJSON 响应格式）
- PGVector 稠密检索 + PostgreSQL FTS 稀疏检索
- Neo4j 知识图谱与 Text-to-Cypher
- 基于 Redis 的语义记忆，支持多轮对话
- 主动澄清提示，处理模糊查询
- 实体规范化与知识图谱填充

### 阶段 5–6：重排序与多模态

- **MonoT5 重排序器**：可插拔的 Transformer 重排序，与交叉编码器和 ColBERT 并行
- **CLIP 多模态**：通过 `clip-ViT-B-32` 联合嵌入空间实现真正的跨模态搜索
- **知识图谱约束**：符号化约束抑制 LLM 幻觉

### 阶段 7–8：加固与 ReAct

- 所有缺失文件已实现（vision_search、monot5_reranker 等）
- **迭代 ReAct 循环**：推理 → 执行 → 观察 → 再推理（最多 3 次迭代）
- **自修复 Cypher**：错误反馈修复循环（最多 2 次重试）

### 阶段 9–10：查询图谱与表格提取

- 从用户查询中提取结构化语义三元组
- 表格感知的 PDF/DOCX 提取（pdfplumber / python-docx）
- 失败封锁一致性评估器（GPT-4 Turbo）
- README 真实性对齐——愿景性技术移至路线图

### 阶段 11：全面对齐

- **正式 Agent 状态机**：`ConversationState` + `DependencyGraph`，用于规划与执行流程
- **向量实体链接器**：与 Neo4j 实体节点的嵌入相似度搜索
- **动态 Schema 内省**：实时 `CALL db.labels()` / `CALL db.relationshipTypes()`
- **表格专用检索路径**：基于元数据过滤的 `TableRetriever`
- **RAGAS 基准测试工具**：忠实度、答案相关性、上下文精度

### 阶段 12：终极对齐

- **规划-评判循环**：使用 `PlannerCritic` 实现计划 → 执行 → 评估 → 再规划
- **Cypher 语法检查与验证**：Schema 感知的预执行验证器
- **概率实体链接**：置信度 < 0.8 触发澄清
- **深度表格嵌入**：语义化「表格摘要 + 列描述」标题
- **可观测性钩子**：`MetricsCollector` + `LatencyTimer` 按引擎遥测
- **故障注入测试**：9 项弹性测试（超时、无效 JSON、写操作拒绝）
- **LlamaIndex 集成**：大文档（> 5 万字符）的分层索引

---

## 🛠 技术栈

| 组件 | 技术 |
| :--- | :--- |
| **LLM** | GPT-4 / GPT-4 Turbo, GPT-3.5 Turbo |
| **检索** | PGVector, PostgreSQL FTS, Elasticsearch |
| **重排序** | Cross-Encoder, ColBERT, MonoT5 |
| **视觉** | CLIP `clip-ViT-B-32` |
| **图谱** | Neo4j（Cypher 语法检查 + 自修复） |
| **记忆** | Redis `StateManager` + SentenceTransformers |
| **框架** | FastAPI, LangChain, LlamaIndex |
| **可观测性** | `MetricsCollector`（延迟、错误率、LLM 评分） |
| **基础设施** | K8s（HPA、存活/就绪探针）, Helm, Docker Compose |

---

## 🔒 阶段 13：硬化运行时

- **预检幻觉拦截器**：两阶段生成器缓冲完整响应，通过一致性评估器验证（分数 > 0.8），安全阻断未通过验证的响应。
- **确定性语义规划**：查询图谱解析器验证三元组是否符合 Neo4j 实时 Schema；未知术语自动对齐到最近有效匹配。
- **结构化 Cypher 对象**：`CypherResult` 数据类，包含节点、边、属性和验证状态。符号路径验证器在执行前检查标签/关系是否存在。
- **深度表格 Schema 摘要**：`generate_schema_summary()` 从数据推断列类型；`extract_structured_answer()` 将表格查询通过 LLM 提取路由。
- **K8s 生产部署**：API（2-10）和检索（2-8）Pod 的 HPA 自动扩缩。存活探针：API `/health`，检索 PG+Redis 检查。Helm chart 标准化部署。

---

## 🔒 阶段 14：主权运行时

- **强化本体对齐**：`OntologyManager` 通过嵌入相似度（阈值 > 0.9）将三元组术语映射到规范 Schema。未映射的术语触发澄清请求而非猜测。
- **符号证明引擎**：`SymbolicValidator` 两层验证——结构预检查 + GPT-4 Turbo 证明层。矛盾知识图谱事实的答案触发硬阻断（分数 < 1.0）。错误时失败封锁。
- **OpenTelemetry / Jaeger 集成**：通过 `core/tracing.py` 实现跨服务 `trace_id` 传播。Jaeger + OTLP 导出器，FastAPI 自动集成和 `@traced` 装饰器。
- **企业运维**：Helm chart 集成 External Secrets Operator（AWS Secrets Manager）、Velero 备份注解（每日凌晨 2 点）和追踪配置。

---

## 📊 测试结果

```
21 项通过，0 项失败
- 3 项 Cypher 生成测试
- 3 项 E2E 集成测试（健康检查、摄取、问答）
- 3 项记忆测试
- 3 项 Cypher 校验弹性测试
- 2 项一致性评估弹性测试
- 1 项实体链接弹性测试
- 3 项可观测性弹性测试
- 3 项额外单元测试
```

---

## 🔮 路线图

| 功能 | 状态 |
| :--- | :--- |
| PEFT / LoRA 微调 | 计划中 |
| BGE / Contriever 嵌入 | 计划中 |
| 分布式追踪（Jaeger/OTel） | **已实现 — core/tracing.py** |
| 生产 Kubernetes | **已实现 — HPA + Helm** |

---

**由 AsterScope 团队用心打造 ❤️**
