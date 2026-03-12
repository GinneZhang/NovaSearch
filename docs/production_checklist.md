# 🚀 NovaSearch v1.0.0 — Production Checklist / 生产环境检查清单

> **Purpose / 用途**: Complete verification checklist before deploying NovaSearch to a production environment.
>
> **目的**: 将 NovaSearch 部署到生产环境前的完整验证检查清单。

---

## 1. Infrastructure / 基础设施

- [ ] **PostgreSQL + PGVector** — Running with `pgvector` extension enabled
  - 确认 PostgreSQL + PGVector 已运行并启用 `pgvector` 扩展
- [ ] **Redis** — Running on configured port with AUTH if exposed
  - 确认 Redis 正在指定端口运行，如公开访问则启用 AUTH
- [ ] **Neo4j** — Running with authentication and APOC plugin installed
  - 确认 Neo4j 已运行并配置认证，安装 APOC 插件
- [ ] **Elasticsearch** — Running (optional, for hybrid sparse retrieval)
  - 确认 Elasticsearch 已运行（可选，用于混合稀疏检索）
- [ ] **Docker Compose** — All containers healthy (`docker ps`)
  - 所有容器健康运行

---

## 2. Environment Variables / 环境变量

- [ ] `OPENAI_API_KEY` — Valid key with GPT-4 Turbo access / 有效密钥，支持 GPT-4 Turbo
- [ ] `POSTGRES_*` vars match Docker Compose / 与 Docker Compose 配置一致
- [ ] `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD` — Match container / 与容器配置一致
- [ ] `REDIS_HOST`, `REDIS_PORT` — Correct / 配置正确
- [ ] `API_ENV=production` — Set to production mode / 设置为生产模式
- [ ] `LOG_LEVEL=WARNING` — Reduce log noise / 降低日志级别

---

## 3. Model Pre-loading / 模型预加载

Run the warmup script before accepting traffic:
在接收流量前运行预热脚本：

```bash
python scripts/warmup.py
```

This pre-loads:
预加载以下模型：

- [ ] `all-MiniLM-L6-v2` — SentenceTransformer text embeddings / 文本嵌入
- [ ] `clip-ViT-B-32` — CLIP vision-text embeddings / 视觉-文本嵌入
- [ ] `cross-encoder/ms-marco-MiniLM-L-6-v2` — Cross-Encoder reranker / 交叉编码器重排序
- [ ] `castorini/monot5-base-msmarco` — MonoT5 reranker (if `RERANKER_TYPE=monot5`) / MonoT5 重排序器
- [ ] `en_core_web_sm` — spaCy NER model / spaCy 命名实体识别模型

---

## 4. Functional Verification / 功能验证

- [ ] `GET /health` returns `{"status": "ok"}` / 健康检查返回正常
- [ ] `POST /ingest` with a test document succeeds / 文档摄取成功
- [ ] `POST /ask` returns grounded answer with sources / 问答返回溯源答案
- [ ] `POST /ask_vision` with test image returns matches / 图像搜索返回匹配
- [ ] Multi-turn memory works with `session_id` / 多轮记忆功能正常
- [ ] Consistency evaluator blocks hallucinated outputs / 一致性评估器阻断幻觉输出

---

## 5. Tests / 测试

```bash
# Run full test suite / 运行完整测试套件
pytest tests/ -v --ignore=tests/benchmark_rag.py --ignore=tests/load_test.py

# Run benchmarks / 运行基准测试
python tests/benchmark_rag.py

# Run load test / 运行负载测试
python tests/load_test.py
```

- [ ] All unit/integration tests pass / 所有单元和集成测试通过
- [ ] Benchmark scores meet minimum thresholds / 基准分数达到最低阈值
- [ ] Load test P95 latency < 5s under 10 concurrent threads / P95 延迟 < 5 秒

---

## 6. Security / 安全

- [ ] `.env` is in `.gitignore` / `.env` 已加入 `.gitignore`
- [ ] No API keys committed to repository / 无 API 密钥提交到仓库
- [ ] Cypher linter enforces read-only queries / Cypher 校验器强制只读查询
- [ ] ConsistencyEvaluator is fail-closed / 一致性评估器采用失败封锁模式
- [ ] CORS policy is restrictive for production / 生产环境 CORS 策略严格配置

---

## 7. Observability / 可观测性

- [ ] `MetricsCollector` is active (check `core/observability.py`) / 指标收集器已激活
- [ ] Per-engine latency is being logged / 按引擎记录延迟
- [ ] Error rates are tracked / 错误率已跟踪
- [ ] Consistency scores are being recorded / 一致性评分已记录

---

## 8. Pre-Flight Guardrails (Phase 13) / 预检护栏

- [ ] `PREFLIGHT_MODE` env var set (`auto`, `always`, or `never`) / 设置 `PREFLIGHT_MODE` 环境变量
- [ ] High-stakes queries trigger buffered validation / 高风险查询触发缓冲验证
- [ ] Safety-block response confirmed for failed pre-flight checks / 确认预检失败时返回安全阻断响应
- [ ] ConsistencyEvaluator threshold > 0.8 enforced / 一致性评估阈值 > 0.8 强制执行

---

## 9. Schema Validation (Phase 13) / Schema 验证

- [ ] QueryGraphParser validates triplets against live Neo4j schema / 查询图谱解析器验证三元组
- [ ] CypherGenerator returns `CypherResult` structured objects / CypherGenerator 返回结构化对象
- [ ] Symbolic path validator rejects unknown labels/relationships / 符号路径验证拒绝未知标签/关系
- [ ] Table Schema Summary headers are generated for table chunks / 表格 Schema 摘要标题已生成

---

## 10. Kubernetes Deployment (Phase 13) / K8s 部署

- [ ] `deploy/k8s/api-deployment.yaml` applied / 已部署 API 清单
- [ ] `deploy/k8s/retrieval-deployment.yaml` applied / 已部署检索清单
- [ ] HPA autoscaling verified (API: 2-10, Retrieval: 2-8) / HPA 自动扩缩已验证
- [ ] Liveness probes pass (`/health` for API, PG+Redis check for Retrieval) / 存活探针通过
- [ ] Readiness probes pass / 就绪探针通过
- [ ] Helm chart deployed via `helm install novasearch deploy/helm/novasearch/` / Helm chart 已部署

---

**Version / 版本**: v1.0.0 (Phase 13 hardened / 第 13 阶段加固)
**Date / 日期**: 2026-03-12

