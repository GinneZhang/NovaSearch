# 🚀 NovaSearch v1.0.0 — 生产环境检查清单

**[English](production_checklist.md) | [中文文档](production_checklist_cn.md)**

> **目的**：将 NovaSearch 部署到生产环境前的完整验证检查清单。

---

## 1. 基础设施

- [ ] **PostgreSQL + PGVector** — 确认已运行并启用 `pgvector` 扩展
- [ ] **Redis** — 确认正在指定端口运行，如公开访问则启用 AUTH
- [ ] **Neo4j** — 确认已运行并配置认证，安装 APOC 插件
- [ ] **Elasticsearch** — 确认已运行（可选，用于混合稀疏检索）
- [ ] **Docker Compose** — 所有容器健康运行（`docker ps`）

---

## 2. 环境变量

- [ ] `OPENAI_API_KEY` — 有效密钥，支持 GPT-4 Turbo
- [ ] `POSTGRES_*` 变量与 Docker Compose 配置一致
- [ ] `NEO4J_URI`、`NEO4J_USER`、`NEO4J_PASSWORD` — 与容器配置一致
- [ ] `REDIS_HOST`、`REDIS_PORT` — 配置正确
- [ ] `API_ENV=production` — 设置为生产模式
- [ ] `LOG_LEVEL=WARNING` — 降低日志级别

---

## 3. 模型预加载

在接收流量前运行预热脚本：

```bash
python scripts/warmup.py
```

预加载以下模型：

- [ ] `all-MiniLM-L6-v2` — 文本嵌入
- [ ] `clip-ViT-B-32` — 视觉-文本嵌入
- [ ] `cross-encoder/ms-marco-MiniLM-L-6-v2` — 交叉编码器重排序
- [ ] `castorini/monot5-base-msmarco` — MonoT5 重排序器（如已配置 `RERANKER_TYPE=monot5`）
- [ ] `en_core_web_sm` — spaCy 命名实体识别模型

---

## 4. 功能验证

- [ ] `GET /health` 返回 `{"status": "ok"}`
- [ ] `POST /ingest` 文档摄取成功
- [ ] `POST /ask` 返回溯源答案
- [ ] `POST /ask_vision` 图像搜索返回匹配
- [ ] 使用 `session_id` 的多轮记忆功能正常
- [ ] 一致性评估器阻断幻觉输出

---

## 5. 测试

```bash
# 运行完整测试套件
pytest tests/ -v --ignore=tests/benchmark_rag.py --ignore=tests/load_test.py

# 运行基准测试
python tests/benchmark_rag.py

# 运行负载测试
python tests/load_test.py
```

- [ ] 所有单元和集成测试通过
- [ ] 基准分数达到最低阈值
- [ ] 负载测试 P95 延迟 < 5 秒（10 并发线程）

---

## 6. 安全

- [ ] `.env` 已加入 `.gitignore`
- [ ] 无 API 密钥提交到仓库
- [ ] Cypher 校验器强制只读查询
- [ ] 一致性评估器采用失败封锁模式
- [ ] 生产环境 CORS 策略严格配置

---

## 7. 可观测性

- [ ] `MetricsCollector` 已激活（检查 `core/observability.py`）
- [ ] 按引擎记录延迟
- [ ] 错误率已跟踪
- [ ] 一致性评分已记录

---

## 8. 预检护栏（阶段 13）

- [ ] `PREFLIGHT_MODE` 环境变量已设置（`auto`、`always` 或 `never`）
- [ ] 高风险查询触发缓冲验证
- [ ] 确认预检失败时返回安全阻断响应
- [ ] 一致性评估阈值 > 0.8 强制执行

---

## 9. Schema 验证（阶段 13）

- [ ] 查询图谱解析器验证三元组是否符合 Neo4j 实时 Schema
- [ ] CypherGenerator 返回 `CypherResult` 结构化对象
- [ ] 符号路径验证拒绝未知标签/关系
- [ ] 表格 Schema 摘要标题已为表格分块生成

---

## 10. Kubernetes 部署（阶段 13）

- [ ] `deploy/k8s/api-deployment.yaml` 已部署
- [ ] `deploy/k8s/retrieval-deployment.yaml` 已部署
- [ ] HPA 自动扩缩已验证（API: 2-10, Retrieval: 2-8）
- [ ] 存活探针通过（API: `/health`，Retrieval: PG+Redis 检查）
- [ ] 就绪探针通过
- [ ] Helm chart 已部署：`helm install novasearch deploy/helm/novasearch/`

---

## 11. 本体对齐（阶段 14）

- [ ] `OntologyManager` 从 Neo4j 加载规范标签、关系和属性
- [ ] 为 Schema 术语构建嵌入索引（`all-MiniLM-L6-v2`）
- [ ] 未映射的三元组术语（置信度 < 0.9）触发澄清响应
- [ ] `ONTOLOGY_CONFIDENCE_THRESHOLD` 环境变量已设置（默认：0.9）

---

## 12. 符号证明引擎（阶段 14）

- [ ] `SymbolicValidator` 结构预检查运行正常
- [ ] GPT-4 Turbo 证明层验证答案是否符合图谱事实
- [ ] 矛盾（分数 < 1.0）触发硬阻断
- [ ] 证明引擎错误时采用失败封锁模式

---

## 13. OpenTelemetry / Jaeger（阶段 14）

- [ ] `OTEL_EXPORTER_OTLP_ENDPOINT` 或 `OTEL_EXPORTER_JAEGER_ENDPOINT` 已配置
- [ ] `core/tracing.py` 在应用启动时初始化
- [ ] FastAPI 已通过 `FastAPIInstrumentor` 自动集成
- [ ] 跨服务 `trace_id` 传播已在日志中验证
- [ ] Jaeger UI 可访问用于追踪可视化

---

**版本**: v1.0.0（阶段 14 — 主权运行时）
**日期**: 2026-03-12
