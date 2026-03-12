#!/usr/bin/env python3
"""
NovaSearch Warmup Script / NovaSearch 预热脚本

Pre-loads all ML models and verifies database connections before the API
accepts its first query. This eliminates cold-start latency.

预加载所有 ML 模型并验证数据库连接，避免首次查询的冷启动延迟。

Usage / 用法:
    python scripts/warmup.py
"""

import os
import sys
import time
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [WARMUP] %(message)s")
logger = logging.getLogger("warmup")

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

PASS = "✅"
FAIL = "❌"
WARN = "⚠️"


def check_postgres():
    """Check PostgreSQL + PGVector connection. / 检查 PostgreSQL + PGVector 连接"""
    try:
        import psycopg2
        conn = psycopg2.connect(
            host=os.getenv("POSTGRES_HOST", "localhost"),
            port=int(os.getenv("POSTGRES_PORT", "5432")),
            dbname=os.getenv("POSTGRES_DB", "novasearch"),
            user=os.getenv("POSTGRES_USER", "postgres"),
            password=os.getenv("POSTGRES_PASSWORD", "")
        )
        cur = conn.cursor()
        cur.execute("SELECT 1")
        cur.close()
        conn.close()
        logger.info(f"{PASS} PostgreSQL — connected")
        return True
    except Exception as e:
        logger.error(f"{FAIL} PostgreSQL — {e}")
        return False


def check_redis():
    """Check Redis connection. / 检查 Redis 连接"""
    try:
        import redis
        r = redis.Redis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", "6379")),
            password=os.getenv("REDIS_PASSWORD", "") or None,
            socket_connect_timeout=5
        )
        r.ping()
        logger.info(f"{PASS} Redis — connected")
        return True
    except Exception as e:
        logger.error(f"{FAIL} Redis — {e}")
        return False


def check_neo4j():
    """Check Neo4j connection. / 检查 Neo4j 连接"""
    try:
        from neo4j import GraphDatabase
        uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        user = os.getenv("NEO4J_USER", "neo4j")
        password = os.getenv("NEO4J_PASSWORD", "")
        driver = GraphDatabase.driver(uri, auth=(user, password))
        with driver.session() as session:
            session.run("RETURN 1")
        driver.close()
        logger.info(f"{PASS} Neo4j — connected")
        return True
    except Exception as e:
        logger.error(f"{FAIL} Neo4j — {e}")
        return False


def check_elasticsearch():
    """Check Elasticsearch connection. / 检查 Elasticsearch 连接"""
    try:
        from elasticsearch import Elasticsearch
        es = Elasticsearch(
            os.getenv("ELASTICSEARCH_URL", "http://localhost:9200"),
            request_timeout=5
        )
        if es.ping():
            logger.info(f"{PASS} Elasticsearch — connected")
            return True
        else:
            logger.warning(f"{WARN} Elasticsearch — ping failed (non-critical)")
            return False
    except Exception as e:
        logger.warning(f"{WARN} Elasticsearch — {e} (non-critical)")
        return False


def preload_sentence_transformer():
    """Pre-load SentenceTransformer model. / 预加载 SentenceTransformer 模型"""
    try:
        from sentence_transformers import SentenceTransformer
        t0 = time.time()
        model = SentenceTransformer("all-MiniLM-L6-v2")
        # Warm the model with a test encode
        model.encode("NovaSearch warmup test")
        elapsed = time.time() - t0
        logger.info(f"{PASS} SentenceTransformer (all-MiniLM-L6-v2) — loaded in {elapsed:.1f}s")
        return True
    except Exception as e:
        logger.error(f"{FAIL} SentenceTransformer — {e}")
        return False


def preload_clip():
    """Pre-load CLIP model. / 预加载 CLIP 模型"""
    try:
        from sentence_transformers import SentenceTransformer
        t0 = time.time()
        model = SentenceTransformer("clip-ViT-B-32")
        model.encode("NovaSearch CLIP warmup")
        elapsed = time.time() - t0
        logger.info(f"{PASS} CLIP (clip-ViT-B-32) — loaded in {elapsed:.1f}s")
        return True
    except Exception as e:
        logger.error(f"{FAIL} CLIP — {e}")
        return False


def preload_cross_encoder():
    """Pre-load Cross-Encoder reranker. / 预加载交叉编码器重排序模型"""
    try:
        from sentence_transformers import CrossEncoder
        t0 = time.time()
        model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        model.predict([("warmup query", "warmup passage")])
        elapsed = time.time() - t0
        logger.info(f"{PASS} Cross-Encoder (ms-marco-MiniLM-L-6-v2) — loaded in {elapsed:.1f}s")
        return True
    except Exception as e:
        logger.error(f"{FAIL} Cross-Encoder — {e}")
        return False


def preload_monot5():
    """Pre-load MonoT5 reranker (if configured). / 预加载 MonoT5 重排序器（如已配置）"""
    reranker_type = os.getenv("RERANKER_TYPE", "cross_encoder").lower()
    if reranker_type != "monot5":
        logger.info(f"{WARN} MonoT5 — skipped (RERANKER_TYPE={reranker_type})")
        return True

    try:
        from transformers import T5ForConditionalGeneration, T5Tokenizer
        t0 = time.time()
        model_name = "castorini/monot5-base-msmarco"
        T5Tokenizer.from_pretrained(model_name)
        T5ForConditionalGeneration.from_pretrained(model_name)
        elapsed = time.time() - t0
        logger.info(f"{PASS} MonoT5 ({model_name}) — loaded in {elapsed:.1f}s")
        return True
    except Exception as e:
        logger.error(f"{FAIL} MonoT5 — {e}")
        return False


def preload_spacy():
    """Pre-load spaCy NER model. / 预加载 spaCy NER 模型"""
    try:
        import spacy
        t0 = time.time()
        nlp = spacy.load("en_core_web_sm")
        nlp("NovaSearch warmup test for named entity recognition.")
        elapsed = time.time() - t0
        logger.info(f"{PASS} spaCy (en_core_web_sm) — loaded in {elapsed:.1f}s")
        return True
    except OSError:
        logger.warning(f"{WARN} spaCy model not downloaded. Run: python -m spacy download en_core_web_sm")
        return False
    except Exception as e:
        logger.error(f"{FAIL} spaCy — {e}")
        return False


def main():
    logger.info("=" * 60)
    logger.info("NovaSearch Warmup — Starting / 开始预热")
    logger.info("=" * 60)

    total_start = time.time()
    results = {}

    # Phase 1: Database Connections / 数据库连接
    logger.info("\n--- Database Connections / 数据库连接 ---")
    results["PostgreSQL"] = check_postgres()
    results["Redis"] = check_redis()
    results["Neo4j"] = check_neo4j()
    results["Elasticsearch"] = check_elasticsearch()

    # Phase 2: Model Pre-loading / 模型预加载
    logger.info("\n--- Model Pre-loading / 模型预加载 ---")
    results["SentenceTransformer"] = preload_sentence_transformer()
    results["CLIP"] = preload_clip()
    results["Cross-Encoder"] = preload_cross_encoder()
    results["MonoT5"] = preload_monot5()
    results["spaCy"] = preload_spacy()

    # Summary / 汇总
    total_elapsed = time.time() - total_start
    logger.info("\n" + "=" * 60)
    logger.info("Warmup Summary / 预热摘要")
    logger.info("=" * 60)

    critical_failures = []
    for name, passed in results.items():
        status = PASS if passed else FAIL
        logger.info(f"  {status} {name}")
        if not passed and name in ("PostgreSQL", "Redis", "SentenceTransformer"):
            critical_failures.append(name)

    logger.info(f"\nTotal time / 总耗时: {total_elapsed:.1f}s")

    if critical_failures:
        logger.error(f"\n{FAIL} CRITICAL FAILURES / 关键失败: {', '.join(critical_failures)}")
        logger.error("NovaSearch may not function correctly. Fix these before starting the API.")
        logger.error("NovaSearch 可能无法正常工作。请在启动 API 前修复这些问题。")
        sys.exit(1)
    else:
        logger.info(f"\n{PASS} All critical systems ready. NovaSearch is warm! / 所有关键系统就绪。NovaSearch 已预热！")
        sys.exit(0)


if __name__ == "__main__":
    main()
