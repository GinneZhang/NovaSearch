"""
Hybrid Search Coordinator for NovaSearch.

This module orchestrates:
1. Dense Retrieval (via vector_search.py)
2. Sparse Retrieval (via keyword_search.py)
3. Graph Retrieval (Neo4j Contextual Expansion)
4. Reranking (via rrf_fusion.py)
"""

import os
import uuid
import json
import logging
import re
import math
from collections import Counter
from typing import List, Dict, Any, Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import spacy
import psycopg2
from neo4j import GraphDatabase

from retrieval.dense.vector_search import PGVectorDenseRetriever
from retrieval.dense.faiss_search import FAISSDenseRetriever
from retrieval.sparse.keyword_search import PostgresFTSSparseRetriever
from retrieval.sparse.elastic_search import ElasticSparseRetriever
from retrieval.reranker.rrf_fusion import reciprocal_rank_fusion
from retrieval.reranker.cross_encoder import CrossEncoderReranker
from retrieval.reranker.colbert_reranker import ColBERTReranker
from retrieval.reranker.monot5_reranker import MonoT5Reranker
from retrieval.graph.cypher_generator import CypherGenerator

logger = logging.getLogger(__name__)

_QUERY_STOPWORDS = {
    "what", "which", "who", "when", "where", "why", "how",
    "is", "was", "were", "are", "the", "a", "an", "of", "in",
    "on", "at", "to", "for", "from", "by", "with", "into",
    "does", "did", "do", "that", "this", "these", "those"
}


def _safe_numeric(value: Any, fallback: float = 0.0) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return fallback
    if math.isnan(numeric) or math.isinf(numeric):
        return fallback
    return numeric

class HybridSearchCoordinator:
    """
    Orchestrates dense, sparse, and graph search strategies, fusing the results
    to provide highly relevant, context-grounded chunks.
    """

    def __init__(
        self,
        embedding_model_name: str = "all-MiniLM-L6-v2",
        pg_dsn: str | None = None,
        neo4j_uri: str | None = None,
        neo4j_user: str | None = None,
        neo4j_password: str | None = None
    ):
        """
        Initializes the HybridSearchCoordinator with connections and retrievers.
        """
        # 1. Setup PostgreSQL (PGVector & FTS)
        self.pg_dsn = pg_dsn or os.getenv("DATABASE_URL", 
            f"dbname={os.getenv('POSTGRES_DB', 'novasearch')} "
            f"user={os.getenv('POSTGRES_USER', 'postgres')} "
            f"password={os.getenv('POSTGRES_PASSWORD', 'postgres_secure_password')} "
            f"host={os.getenv('POSTGRES_HOST', 'localhost')} "
            f"port={os.getenv('POSTGRES_PORT', '5432')}"
        )
        try:
            self.pg_conn = psycopg2.connect(self.pg_dsn)
            self.pg_conn.autocommit = True
            logger.info("Connected to PostgreSQL for Dense/Sparse search.")
        except Exception as e:
            logger.error("Failed to connect to PostgreSQL: %s", str(e))
            self.pg_conn = None

        # 2. Setup Retrievers & Reranker
        dense_backend = os.getenv("DENSE_BACKEND", os.getenv("VECTOR_STORE_TYPE", "pgvector")).lower()
        if dense_backend == "faiss":
            logger.info("Initializing FAISS local vector store...")
            self.dense_retriever = FAISSDenseRetriever(embedding_model_name, index_dir="faiss_index")
        else:
            logger.info("Initializing PGVector remote vector store...")
            self.dense_retriever = PGVectorDenseRetriever(self.pg_conn, embedding_model_name)
            
        sparse_backend = os.getenv("SPARSE_BACKEND", os.getenv("SPARSE_STORE_TYPE", "postgres")).lower()
        if sparse_backend in ["elastic", "elasticsearch"]:
            logger.info("Initializing Elasticsearch sparse store...")
            self.sparse_retriever = ElasticSparseRetriever()
        else:
            logger.info("Initializing Postgres FTS sparse store...")
            self.sparse_retriever = PostgresFTSSparseRetriever(self.pg_conn)
            
        reranker_type = os.getenv("RERANKER_TYPE", "crossencoder").lower()
        if reranker_type == "colbert":
            logger.info("Initializing ColBERT Reranker...")
            self.cross_encoder = ColBERTReranker()
        elif reranker_type == "monot5":
            logger.info("Initializing MonoT5 Reranker...")
            self.cross_encoder = MonoT5Reranker()
            # Graceful fallback check
            if not getattr(self.cross_encoder, "model", None):
                logger.warning("MonoT5 initialization failed. Falling back to CrossEncoder.")
                self.cross_encoder = CrossEncoderReranker()
        else:
            logger.info("Initializing CrossEncoder Reranker...")
            self.cross_encoder = CrossEncoderReranker()

        # 3. Setup Neo4j (Graph Retrieval)
        benchmark_mode = os.getenv("NOVASEARCH_BENCHMARK_MODE", "false").lower() in {"1", "true", "yes"}
        self.enable_graph_retrieval = os.getenv(
            "ENABLE_GRAPH_RETRIEVAL",
            "false" if benchmark_mode else "true"
        ).lower() in {"1", "true", "yes"}
        self.enable_raw_lexical_recall = os.getenv("ENABLE_RAW_LEXICAL_RECALL", "true").lower() in {"1", "true", "yes"}
        self.neo4j_uri = neo4j_uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.neo4j_user = neo4j_user or os.getenv("NEO4J_USER", "neo4j")
        self.neo4j_password = neo4j_password or os.getenv("NEO4J_PASSWORD", "neo4j_secure_password")
        
        if self.enable_graph_retrieval:
            try:
                self.neo4j_driver = GraphDatabase.driver(
                    self.neo4j_uri, auth=(self.neo4j_user, self.neo4j_password)
                )
                self.neo4j_driver.verify_connectivity()
                logger.info("Connected to Neo4j Knowledge Graph.")
            except Exception as e:
                logger.error("Failed to connect to Neo4j: %s", str(e))
                self.neo4j_driver = None
        else:
            logger.info("Graph retrieval disabled for current runtime mode.")
            self.neo4j_driver = None
            
        # 4. Setup NER for Graph Expansion
        if self.enable_graph_retrieval:
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                logger.warning("Spacy model not found. Continuing without spaCy NER.")
                self.nlp = None
        else:
            self.nlp = None
            
        # 5. Setup Dynamic Cypher Generator
        self.cypher_gen = CypherGenerator() if self.enable_graph_retrieval else None
        self.last_search_debug: Dict[str, Any] = {}

    def __del__(self):
        """Cleanup connections on destruction."""
        try:
            if hasattr(self, "pg_conn") and self.pg_conn and not self.pg_conn.closed:
                self.pg_conn.close()
        except Exception:
            pass
        try:
            close_fn = getattr(getattr(self, "neo4j_driver", None), "close", None)
            if callable(close_fn):
                close_fn()
        except Exception:
            pass

    def _extract_entities(self, query: str) -> List[str]:
        """Extracts Named Entities (ORG, PERSON, DATE, etc.) using spaCy."""
        if not hasattr(self, 'nlp') or not self.nlp:
            return []
        doc = self.nlp(query)
        # Filter for meaningful entities
        entities = [ent.text for ent in doc.ents if ent.label_ in ["ORG", "PERSON", "GPE", "PRODUCT", "LAW"]]
        return list(set(entities))

    @staticmethod
    def _normalize_terms(text: str) -> List[str]:
        return [token for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9_-]+", (text or "").lower()) if len(token) > 2]

    def _title_overlap(self, query: str, title: str) -> int:
        query_terms = set(self._normalize_terms(query))
        title_terms = set(self._normalize_terms(title))
        return len(query_terms & title_terms)

    def _is_multi_hop_query(self, query: str) -> bool:
        lowered = query.lower()
        quoted_phrases = re.findall(r"['\"]([^'\"]+)['\"]", query or "")
        entity_count = len(self._extract_entities(query))
        capitalized_spans = re.findall(r"(?:[A-Z][A-Za-z'`-]+(?:\s+[A-Z][A-Za-z'`-]+){1,5})", query or "")
        return (
            lowered.count(" of ") >= 1
            and (entity_count >= 1 or len(quoted_phrases) >= 1 or len(capitalized_spans) >= 2)
        )

    def _build_query_variants(self, query: str) -> List[str]:
        variants: List[str] = [query.strip()]
        entities = self._extract_entities(query)

        if entities:
            variants.extend(entity for entity in entities if entity and entity not in variants)
            combined_entities = " ".join(entities[:2]).strip()
            if combined_entities and combined_entities not in variants:
                variants.append(combined_entities)

        lowered = query.lower()
        cleaned = re.sub(r"\b(who|what|when|where|which|is|was|were|the|a|an)\b", " ", lowered, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        if cleaned and cleaned not in variants:
            variants.append(cleaned)

        return variants[:4]

    def _extract_query_subject_candidates(self, query: str) -> List[str]:
        subjects: List[str] = []
        seen = set()

        def _append(candidate: str):
            cleaned = (candidate or "").strip(" ?.,:;()[]{}\"'")
            lowered = cleaned.lower()
            if not cleaned or lowered in seen:
                return
            if len(cleaned.split()) > 8:
                return
            subjects.append(cleaned)
            seen.add(lowered)

        for quoted in re.findall(r"['\"]([^'\"]+)['\"]", query or ""):
            _append(quoted)

        for pattern in (
            r"\bof ([A-Z][A-Za-z'`-]+(?:\s+[A-Z][A-Za-z'`-]+){0,5})\??$",
            r"\bin the (?:film|book|song|album|series|event) ([A-Z][A-Za-z'`-]+(?:\s+[A-Z][A-Za-z'`()-]+){0,5})\??$",
            r"\bnarrator of ([A-Z][A-Za-z'`-]+(?:\s+[A-Z][A-Za-z'`()-]+){0,5})\??$",
        ):
            match = re.search(pattern, query or "")
            if match:
                _append(match.group(1))

        for span in re.findall(r"(?:[A-Z][A-Za-z'`-]+(?:\s+[A-Z][A-Za-z'`-]+){1,5})", query or ""):
            _append(span)

        return subjects[:6]

    def _extract_query_focus_terms(self, query: str) -> set:
        query_terms = set(self._normalize_terms(query))
        subject_terms = set()
        for subject in self._extract_query_subject_candidates(query):
            subject_terms.update(self._normalize_terms(subject))
        return {
            term for term in query_terms
            if term not in _QUERY_STOPWORDS and term not in subject_terms
        }

    def _extract_bridge_entities(self, query: str, hits: List[Dict[str, Any]]) -> List[str]:
        query_terms = set(self._normalize_terms(query))
        bridges: List[str] = []

        for hit in hits[:5]:
            candidates: List[str] = []
            title = hit.get("title") or ""
            if title:
                candidates.append(title)

            if hasattr(self, "nlp") and self.nlp:
                text = f"{title}\n{hit.get('chunk_text', '')}"
                try:
                    doc = self.nlp(text)
                    candidates.extend(
                        ent.text.strip() for ent in doc.ents
                        if ent.label_ in {"PERSON", "ORG", "GPE", "LOC", "PRODUCT", "EVENT"}
                    )
                except Exception:
                    pass
            else:
                text = hit.get("chunk_text", "")
                candidates.extend(re.findall(r"(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})", text))

            for candidate in candidates:
                candidate = candidate.strip(" .,:;()[]{}\"'")
                if not candidate:
                    continue
                candidate_terms = set(self._normalize_terms(candidate))
                if not candidate_terms:
                    continue
                if candidate_terms.issubset(query_terms):
                    continue
                if len(candidate_terms) == 1 and next(iter(candidate_terms)) in {"who", "what", "when", "where"}:
                    continue
                if candidate not in bridges:
                    bridges.append(candidate)

        return bridges[:6]

    def _follow_up_queries(self, query: str, hits: List[Dict[str, Any]]) -> List[str]:
        if not self._is_multi_hop_query(query):
            return []

        bridges = self._extract_bridge_entities(query, hits)
        query_terms = [
            term for term in self._normalize_terms(query)
            if term not in _QUERY_STOPWORDS
        ]
        follow_ups: List[str] = []

        for bridge in bridges:
            candidates = [bridge]
            if query_terms:
                candidates.append(f"{bridge} {' '.join(query_terms[:4])}".strip())
                if len(query_terms) > 2:
                    candidates.append(f"{bridge} {' '.join(query_terms[-3:])}".strip())
            for candidate in candidates:
                if candidate.lower() != query.lower() and candidate not in follow_ups:
                    follow_ups.append(candidate)

        return follow_ups[:6]

    def _collect_candidates(self, query_variants: List[str], fetch_k: int) -> List[Dict[str, Any]]:
        dense_hits: List[Dict[str, Any]] = []
        sparse_hits: List[Dict[str, Any]] = []

        per_variant_k = max(8, fetch_k // max(1, len(query_variants)))
        for variant in query_variants:
            for hit in self.dense_retriever.search(variant, top_k=per_variant_k):
                hit["query_variant"] = variant
                dense_hits.append(hit)
            for hit in self.sparse_retriever.search(variant, top_k=per_variant_k):
                hit["query_variant"] = variant
                sparse_hits.append(hit)
            if self.enable_raw_lexical_recall:
                for hit in self._raw_lexical_search(variant, top_k=per_variant_k):
                    hit["query_variant"] = variant
                    sparse_hits.append(hit)

        fused_hits = reciprocal_rank_fusion(dense_hits, sparse_hits)

        for hit in fused_hits:
            title = hit.get("title") or ""
            if title:
                hit["rrf_score"] = hit.get("rrf_score", 0.0) + (0.04 * self._title_overlap(query_variants[0], title))

        fused_hits.sort(key=lambda x: x.get("rrf_score", 0.0), reverse=True)
        return fused_hits[:fetch_k]

    def _raw_lexical_search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        if not self.pg_conn:
            return []

        terms = [
            term for term in self._normalize_terms(query)
            if term not in _QUERY_STOPWORDS
        ][:5]
        if not terms:
            return []

        text_expr = "lower(coalesce(d.title, '') || ' ' || c.chunk_text)"
        score_terms = []
        where_terms = []
        params: List[Any] = []
        for term in terms:
            pattern = f"%{term}%"
            score_terms.append(f"CASE WHEN {text_expr} LIKE %s THEN 1 ELSE 0 END")
            where_terms.append(f"{text_expr} LIKE %s")
            params.append(pattern)
            params.append(pattern)

        min_matches = min(2, len(terms))
        sql = f"""
            SELECT
                c.doc_id,
                c.index,
                c.chunk_text,
                d.title,
                ({' + '.join(score_terms)}) AS lexical_score
            FROM chunks c
            LEFT JOIN documents d ON d.id = c.doc_id
            WHERE ({' OR '.join(where_terms)})
            ORDER BY lexical_score DESC, c.index ASC
            LIMIT %s;
        """

        results: List[Dict[str, Any]] = []
        try:
            with self.pg_conn.cursor() as cur:
                cur.execute(sql, (*params, top_k * 4))
                for row in cur.fetchall():
                    score = _safe_numeric(row[4], 0.0)
                    if score < min_matches:
                        continue
                    title = row[3] or ""
                    score += 0.18 * self._title_overlap(query, title)
                    results.append({
                        "doc_id": row[0],
                        "chunk_index": row[1],
                        "chunk_text": row[2],
                        "title": title,
                        "score": score,
                        "source": "raw_lexical",
                    })
        except Exception as e:
            logger.warning("Raw lexical search failed: %s", str(e))
            self.pg_conn.rollback()

        results.sort(key=lambda hit: _safe_numeric(hit.get("score", 0.0)), reverse=True)
        return results[:top_k]

    def _merge_candidate_pools(self, candidate_groups: List[List[Dict[str, Any]]], fetch_k: int) -> List[Dict[str, Any]]:
        deduped_pool: Dict[tuple, Dict[str, Any]] = {}
        for group in candidate_groups:
            for hit in group:
                key = (hit.get("doc_id"), hit.get("chunk_index"))
                current_score = _safe_numeric(hit.get("rrf_score", hit.get("score", 0.0)))
                previous = deduped_pool.get(key)
                if previous is None:
                    deduped_pool[key] = hit
                    continue
                previous_score = _safe_numeric(previous.get("rrf_score", previous.get("score", 0.0)))
                if current_score > previous_score:
                    deduped_pool[key] = hit

        return sorted(
            deduped_pool.values(),
            key=lambda x: _safe_numeric(x.get("rrf_score", x.get("score", 0.0))),
            reverse=True
        )[:fetch_k]

    def _expand_candidate_documents(
        self,
        query: str,
        reranked_hits: List[Dict[str, Any]],
        fetch_k: int
    ) -> List[Dict[str, Any]]:
        if not self.pg_conn or not reranked_hits:
            return reranked_hits[:fetch_k]

        top_doc_ids: List[str] = []
        priority_doc_ids: List[str] = []
        retrieval_queries_by_doc: Dict[str, List[str]] = {}
        for hit in reranked_hits[: max(8, fetch_k // 2)]:
            doc_id = hit.get("doc_id")
            if not doc_id:
                continue
            if doc_id not in top_doc_ids:
                top_doc_ids.append(doc_id)
            retrieval_queries_by_doc.setdefault(doc_id, [])
            for retrieval_query in hit.get("retrieval_queries", []) or []:
                if retrieval_query not in retrieval_queries_by_doc[doc_id]:
                    retrieval_queries_by_doc[doc_id].append(retrieval_query)
            title = ((hit.get("graph_context") or {}).get("doc_title") or hit.get("title") or "").strip()
            title_terms = set(self._normalize_terms(title))
            retrieval_terms = set()
            for retrieval_query in retrieval_queries_by_doc[doc_id]:
                retrieval_terms.update(self._normalize_terms(retrieval_query))
            if (
                title_terms
                and retrieval_terms
                and title_terms & retrieval_terms
                and title_terms.isdisjoint(set(self._normalize_terms(query)))
                and doc_id not in priority_doc_ids
            ):
                priority_doc_ids.append(doc_id)

        if not top_doc_ids:
            return reranked_hits[:fetch_k]

        existing_keys = {
            (hit.get("doc_id"), hit.get("chunk_index"))
            for hit in reranked_hits
        }
        expanded_hits: List[Dict[str, Any]] = []
        try:
            with self.pg_conn.cursor() as cur:
                seeded_doc_ids = priority_doc_ids + [doc_id for doc_id in top_doc_ids if doc_id not in priority_doc_ids]
                for doc_id in seeded_doc_ids[:8]:
                    cur.execute(
                        """
                        SELECT c.doc_id, c.index, c.chunk_text, d.title
                        FROM chunks c
                        LEFT JOIN documents d ON d.id = c.doc_id
                        WHERE c.doc_id = %s
                        ORDER BY c.index ASC
                        LIMIT 6;
                        """,
                        (doc_id,)
                    )
                    for row in cur.fetchall():
                        key = (row[0], row[1])
                        if key in existing_keys:
                            continue
                        expanded_hits.append({
                            "doc_id": row[0],
                            "chunk_index": row[1],
                            "chunk_text": row[2],
                            "title": row[3] or "",
                            "score": 0.0,
                            "source": "doc_expansion",
                            "retrieval_queries": retrieval_queries_by_doc.get(doc_id, [])[:],
                        })
        except Exception as e:
            logger.warning("Document expansion failed: %s", str(e))
            self.pg_conn.rollback()
            return reranked_hits[:fetch_k]

        if not expanded_hits:
            return reranked_hits[:fetch_k]

        merged_hits = self._merge_candidate_pools([reranked_hits, expanded_hits], fetch_k=fetch_k)
        reranked_merged_hits = self.cross_encoder.rerank(query, merged_hits, top_k=fetch_k)
        return reranked_merged_hits[:fetch_k]

    def _infer_source_type(self, hit: Dict[str, Any]) -> str:
        source = str(hit.get("source") or "").strip().lower()
        if source:
            if source == "dynamic_cypher":
                return "symbolic"
            if source in {"doc_expansion", "graph_expansion"}:
                return "graph_expansion"
            if source in {"raw_lexical"}:
                return "lexical"
            return source

        source_tags = {
            str(tag).strip().lower()
            for tag in (hit.get("sources") or [])
            if str(tag).strip()
        }
        if {"dense", "sparse"} <= source_tags:
            return "hybrid"
        if "dense" in source_tags:
            return "dense"
        if "sparse" in source_tags:
            return "sparse"
        if hit.get("graph_context"):
            return "graph_enriched"
        return "unknown"

    def _is_near_duplicate_text(self, left_text: str, right_text: str) -> bool:
        left_terms = self._normalize_terms(left_text)[:40]
        right_terms = self._normalize_terms(right_text)[:40]
        if not left_terms or not right_terms:
            return False
        if left_terms[:20] == right_terms[:20]:
            return True
        left_set = set(left_terms)
        right_set = set(right_terms)
        overlap = len(left_set & right_set)
        union = len(left_set | right_set)
        return bool(union) and (overlap / union) >= 0.84

    def _should_use_symbolic_search(
        self,
        query: str,
        query_graph: Optional[List[Dict[str, str]]],
        reranked_hits: List[Dict[str, Any]],
    ) -> bool:
        if not self.enable_graph_retrieval or not self.neo4j_driver or not self.cypher_gen:
            return False

        has_structured_graph = bool(query_graph)
        likely_multi_hop = self._is_multi_hop_query(query)
        if not has_structured_graph and not likely_multi_hop:
            return False

        focus_terms = self._extract_query_focus_terms(query)
        top_hits = reranked_hits[: min(6, len(reranked_hits))]
        if not top_hits:
            return has_structured_graph

        focus_covered = 0
        for hit in top_hits:
            title = ((hit.get("graph_context") or {}).get("doc_title") or hit.get("title") or "").strip()
            text = hit.get("chunk_text", "") or ""
            if "\n" in text:
                text = text.split("\n", 1)[1]
            combined_terms = set(self._normalize_terms(title)) | set(self._normalize_terms(text))
            if focus_terms and combined_terms & focus_terms:
                focus_covered += 1

        focus_coverage_ratio = focus_covered / max(1, len(top_hits))
        if has_structured_graph and focus_coverage_ratio < 0.72:
            return True
        if likely_multi_hop and focus_coverage_ratio < 0.52:
            return True
        return False

    @staticmethod
    def _is_graph_like_source(source_type: str) -> bool:
        return source_type in {"graph_expansion", "graph_enriched", "symbolic"}

    def _annotate_source_calibration(
        self,
        candidates: List[Dict[str, Any]],
    ) -> Dict[tuple, Dict[str, float]]:
        by_source: Dict[str, List[tuple[tuple, float]]] = {}
        for hit in candidates:
            key = (hit.get("doc_id"), hit.get("chunk_index"))
            source_type = self._infer_source_type(hit)
            base_score = _safe_numeric(
                hit.get("final_rank_score"),
                _safe_numeric(
                    hit.get("cross_encoder_score"),
                    _safe_numeric(hit.get("rrf_score"), _safe_numeric(hit.get("score", 0.0))),
                ),
            )
            by_source.setdefault(source_type, []).append((key, base_score))

        calibration: Dict[tuple, Dict[str, float]] = {}
        for source_type, rows in by_source.items():
            ordered_rows = sorted(rows, key=lambda item: item[1], reverse=True)
            count = len(ordered_rows)
            mean_score = sum(score for _, score in ordered_rows) / max(1, count)
            variance = sum((score - mean_score) ** 2 for _, score in ordered_rows) / max(1, count)
            std_score = math.sqrt(variance)
            for index, (key, score) in enumerate(ordered_rows):
                if count <= 1:
                    rank_fraction = 1.0
                else:
                    rank_fraction = 1.0 - (index / max(1, count - 1))
                distribution_score = 0.5
                if std_score > 1e-6:
                    normalized = (score - mean_score) / (3 * std_score)
                    distribution_score = max(0.0, min(1.0, 0.5 + normalized))
                calibration[key] = {
                    "source_rank_fraction": round(rank_fraction, 4),
                    "distribution_score": round(distribution_score, 4),
                }

        return calibration

    def _select_role_aware_candidates(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int,
    ) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
        if not candidates:
            return [], {
                "duplicates_removed": 0,
                "source_type_mix": {},
                "evidence_role_mix": {},
                "bridge_coverage_score": 0.0,
                "support_coverage_score": 0.0,
            }

        query_terms = set(self._normalize_terms(query))
        focus_terms = self._extract_query_focus_terms(query)
        subject_terms = set()
        for subject in self._extract_query_subject_candidates(query):
            subject_terms.update(self._normalize_terms(subject))

        source_prior = {
            "hybrid": 0.24,
            "dense": 0.12,
            "sparse": 0.08,
            "lexical": 0.06,
            "graph_expansion": -0.02,
            "graph_enriched": 0.04,
            "symbolic": 0.02,
            "unknown": 0.0,
        }
        role_priority = {"direct": 4, "bridge": 3, "graph": 2, "symbolic": 2, "background": 1}
        source_calibration = self._annotate_source_calibration(candidates)

        rows: List[Dict[str, Any]] = []
        for position, hit in enumerate(candidates):
            title = ((hit.get("graph_context") or {}).get("doc_title") or hit.get("title") or "").strip()
            text = hit.get("chunk_text", "") or ""
            if "\n" in text:
                text = text.split("\n", 1)[1]
            title_terms = set(self._normalize_terms(title))
            text_terms = set(self._normalize_terms(text))
            combined_terms = title_terms | text_terms

            retrieval_terms = set()
            retrieval_queries = [
                rq.strip() for rq in (hit.get("retrieval_queries") or [])
                if isinstance(rq, str) and rq.strip()
            ]
            for retrieval_query in retrieval_queries:
                retrieval_terms.update(self._normalize_terms(retrieval_query))
            bridge_terms = retrieval_terms - query_terms

            query_overlap = len(combined_terms & query_terms)
            focus_overlap = len(combined_terms & focus_terms)
            bridge_overlap = len(combined_terms & bridge_terms)
            subject_overlap = len(combined_terms & subject_terms)

            base_score = _safe_numeric(
                hit.get("final_rank_score"),
                _safe_numeric(hit.get("cross_encoder_score"), _safe_numeric(hit.get("rrf_score"), _safe_numeric(hit.get("score", 0.0))))
            )
            source_type = self._infer_source_type(hit)
            calibration = source_calibration.get((hit.get("doc_id"), hit.get("chunk_index")), {})
            source_rank_fraction = float(calibration.get("source_rank_fraction", 0.5) or 0.5)
            distribution_score = float(calibration.get("distribution_score", 0.5) or 0.5)
            direct_signal = min(1.0, (0.14 * query_overlap) + (0.20 * focus_overlap) + (0.08 if subject_overlap else 0.0))
            bridge_signal = min(1.0, (0.18 * bridge_overlap) + (0.06 * len(title_terms & bridge_terms)) + (0.08 if bridge_terms else 0.0))
            graph_confidence = 0.0
            if source_type == "symbolic":
                graph_confidence = 0.78
            elif source_type in {"graph_expansion", "graph_enriched"}:
                shared_entities = ((hit.get("graph_context") or {}).get("shared_entities") or [])
                graph_confidence = min(0.55, 0.16 * len(shared_entities))

            role = "background"
            if source_type == "symbolic":
                role = "symbolic"
            elif bridge_signal >= max(0.30, direct_signal + 0.08):
                role = "bridge"
            elif direct_signal >= 0.42:
                role = "direct"
            elif source_type in {"graph_expansion", "graph_enriched"} and (direct_signal >= 0.2 or bridge_signal >= 0.2):
                role = "graph"

            background_penalty = 0.0
            if role == "background":
                background_penalty += 0.22
                if focus_terms and focus_overlap == 0:
                    background_penalty += 0.16
            if source_type == "symbolic" and (direct_signal + bridge_signal) < 0.24:
                background_penalty += 0.22
            if self._is_graph_like_source(source_type) and focus_overlap == 0 and bridge_overlap == 0:
                background_penalty += 0.14

            role_score = (
                base_score
                + (0.62 * direct_signal)
                + (0.48 * bridge_signal)
                + (0.34 * graph_confidence)
                + (0.18 * source_rank_fraction)
                + (0.14 * distribution_score)
                + source_prior.get(source_type, 0.0)
                - (0.015 * position)
                - background_penalty
            )

            rows.append({
                "hit": hit,
                "key": (hit.get("doc_id"), hit.get("chunk_index")),
                "title_key": title.lower() or f"{hit.get('doc_id')}_{hit.get('chunk_index')}",
                "title": title,
                "text": text,
                "source_type": source_type,
                "role": role,
                "direct_signal": direct_signal,
                "bridge_signal": bridge_signal,
                "source_rank_fraction": source_rank_fraction,
                "distribution_score": distribution_score,
                "score": role_score,
            })

        title_family_sources: Dict[str, set] = {}
        title_family_focus: Counter = Counter()
        title_family_supportive: Counter = Counter()
        for row in rows:
            title_family_sources.setdefault(row["title_key"], set()).add(row["source_type"])
            if row["direct_signal"] >= 0.22 or row["bridge_signal"] >= 0.22:
                title_family_focus[row["title_key"]] += 1
            if row["role"] in {"direct", "bridge", "graph", "symbolic"}:
                title_family_supportive[row["title_key"]] += 1

        graph_symbolic_candidates = 0
        graph_symbolic_kept = 0
        graph_symbolic_dropped = 0
        corroborated_graph_kept = 0
        drop_reasons = Counter()

        for row in rows:
            family_sources = title_family_sources.get(row["title_key"], set())
            family_support_count = int(title_family_supportive.get(row["title_key"], 0))
            corroboration_count = max(0, len(family_sources) - 1)
            if title_family_focus.get(row["title_key"], 0) > 1:
                corroboration_count += 1
            if row["direct_signal"] >= 0.42:
                corroboration_count += 1
            if row["bridge_signal"] >= 0.38:
                corroboration_count += 1
            is_graph_like = self._is_graph_like_source(row["source_type"])
            if is_graph_like:
                graph_symbolic_candidates += 1
            is_corroborated = (
                corroboration_count > 0
                or family_support_count > 1
                or (
                    row["direct_signal"] >= 0.42
                    and row["bridge_signal"] >= 0.18
                )
            )
            if is_graph_like and not is_corroborated:
                row["score"] -= 0.36
            if row["source_type"] == "symbolic" and not is_corroborated:
                row["score"] -= 0.24
            row["corroboration_count"] = corroboration_count
            row["is_corroborated"] = is_corroborated
            row["is_graph_like"] = is_graph_like

        rows.sort(key=lambda row: (row["score"], role_priority.get(row["role"], 0)), reverse=True)

        max_per_title = max(1, int(os.getenv("EVIDENCE_MAX_CHUNKS_PER_TITLE", "2")))
        role_caps = {
            "symbolic": max(1, min(2, max(1, top_k // 6))),
            "graph": max(1, max(1, top_k // 4)),
            "background": max(1, max(1, top_k // 5)),
        }
        source_caps = {
            "symbolic": 1,
            "graph_expansion": 1,
            "graph_enriched": 1,
        }

        selected_rows: List[Dict[str, Any]] = []
        selected_keys = set()
        title_counts: Dict[str, int] = {}
        role_counts = Counter()
        source_counts = Counter()
        selected_texts_by_title: Dict[str, List[str]] = {}
        duplicates_removed = 0

        def _can_take(row: Dict[str, Any]) -> bool:
            if row["key"] in selected_keys:
                drop_reasons["already_selected"] += 1
                return False
            if title_counts.get(row["title_key"], 0) >= max_per_title:
                drop_reasons["title_cap"] += 1
                return False
            capped = role_caps.get(row["role"])
            if capped is not None and role_counts.get(row["role"], 0) >= capped:
                drop_reasons["role_cap"] += 1
                return False
            source_cap = source_caps.get(row["source_type"])
            if source_cap is not None and source_counts.get(row["source_type"], 0) >= source_cap:
                drop_reasons["source_cap"] += 1
                return False
            if row["is_graph_like"] and not row["is_corroborated"]:
                drop_reasons["uncorroborated_graph"] += 1
                return False
            existing_texts = selected_texts_by_title.get(row["title_key"], [])
            if any(self._is_near_duplicate_text(row["text"], existing) for existing in existing_texts):
                drop_reasons["near_duplicate"] += 1
                return False
            return True

        for required_role in ("direct", "bridge"):
            for row in rows:
                if row["role"] != required_role:
                    continue
                if _can_take(row):
                    selected_rows.append(row)
                    selected_keys.add(row["key"])
                    title_counts[row["title_key"]] = title_counts.get(row["title_key"], 0) + 1
                    role_counts[row["role"]] += 1
                    source_counts[row["source_type"]] += 1
                    selected_texts_by_title.setdefault(row["title_key"], []).append(row["text"])
                    if row["is_graph_like"]:
                        graph_symbolic_kept += 1
                        if row["is_corroborated"]:
                            corroborated_graph_kept += 1
                    break

        for row in rows:
            if len(selected_rows) >= top_k:
                break
            if not _can_take(row):
                duplicates_removed += 1
                if row["is_graph_like"]:
                    graph_symbolic_dropped += 1
                continue
            selected_rows.append(row)
            selected_keys.add(row["key"])
            title_counts[row["title_key"]] = title_counts.get(row["title_key"], 0) + 1
            role_counts[row["role"]] += 1
            source_counts[row["source_type"]] += 1
            selected_texts_by_title.setdefault(row["title_key"], []).append(row["text"])
            if row["is_graph_like"]:
                graph_symbolic_kept += 1
                if row["is_corroborated"]:
                    corroborated_graph_kept += 1

        selected_rows = selected_rows[:top_k]
        selected_hits = []
        for row in selected_rows:
            annotated_hit = dict(row["hit"])
            annotated_hit["source_type"] = row["source_type"]
            annotated_hit["evidence_role"] = row["role"]
            annotated_hit["corroboration_count"] = row.get("corroboration_count", 0)
            annotated_hit["is_corroborated"] = row.get("is_corroborated", False)
            annotated_hit["source_rank_fraction"] = row.get("source_rank_fraction", 0.5)
            annotated_hit["distribution_score"] = row.get("distribution_score", 0.5)
            selected_hits.append(annotated_hit)
        source_mix = dict(Counter(row["source_type"] for row in selected_rows))
        role_mix = dict(Counter(row["role"] for row in selected_rows))
        bridge_coverage_score = sum(row["bridge_signal"] for row in selected_rows) / max(1, len(selected_rows))
        support_coverage_score = sum(row["direct_signal"] for row in selected_rows) / max(1, len(selected_rows))

        return selected_hits, {
            "duplicates_removed": duplicates_removed,
            "source_type_mix": source_mix,
            "evidence_role_mix": role_mix,
            "bridge_coverage_score": round(bridge_coverage_score, 4),
            "support_coverage_score": round(support_coverage_score, 4),
            "role_aware_pool_count": len(rows),
            "role_aware_selected_count": len(selected_rows),
            "graph_symbolic_candidates": graph_symbolic_candidates,
            "graph_symbolic_kept": graph_symbolic_kept,
            "graph_symbolic_dropped": graph_symbolic_dropped,
            "corroborated_graph_kept": corroborated_graph_kept,
            "selection_drop_reasons": dict(drop_reasons),
        }

    def collect_candidate_pool(
        self,
        query: str,
        top_k: int = 5,
        additional_queries: Optional[List[str]] = None,
        include_follow_ups: bool = True
    ) -> List[Dict[str, Any]]:
        fetch_k = max(40, top_k * 8)
        query_batch = [query]
        for extra_query in additional_queries or []:
            normalized = (extra_query or "").strip()
            if normalized and normalized not in query_batch:
                query_batch.append(normalized)

        candidate_groups = []
        for candidate_query in query_batch:
            base_variants = self._build_query_variants(candidate_query)
            fused_hits = self._collect_candidates(base_variants, fetch_k=fetch_k)
            for hit in fused_hits:
                hit.setdefault("retrieval_queries", [])
                hit["retrieval_queries"] = list({*hit["retrieval_queries"], candidate_query})
            candidate_groups.append(fused_hits)
        merged_pool = self._merge_candidate_pools(candidate_groups, fetch_k=fetch_k)

        if include_follow_ups:
            follow_ups = self._follow_up_queries(query, merged_pool[:10])
            if follow_ups:
                logger.info("Generated %d follow-up queries for multi-hop retrieval: %s", len(follow_ups), follow_ups)
                follow_up_groups = []
                for follow_up in follow_ups:
                    follow_up_hits = self._collect_candidates(self._build_query_variants(follow_up), fetch_k=fetch_k)
                    for hit in follow_up_hits:
                        hit.setdefault("retrieval_queries", [])
                        hit["retrieval_queries"] = list({*hit["retrieval_queries"], follow_up})
                    follow_up_groups.append(follow_up_hits)
                merged_pool = self._merge_candidate_pools([merged_pool, *follow_up_groups], fetch_k=fetch_k)

        final_pool = merged_pool[:fetch_k]
        self.last_search_debug = {
            "candidate_pool_count": len(final_pool),
            "additional_query_count": len(query_batch) - 1,
        }
        return final_pool

    def finalize_candidates(
        self,
        query: str,
        candidate_pool: List[Dict[str, Any]],
        top_k: int = 5,
        query_graph: Optional[List[Dict[str, str]]] = None
    ) -> List[Dict[str, Any]]:
        fetch_k = max(40, top_k * 8)
        candidate_pool = candidate_pool[:fetch_k]
        first_stage_count = len(candidate_pool)

        # 3. Second Stage Reranking (Cross-Encoder)
        reranked_hits = self.cross_encoder.rerank(query, candidate_pool, top_k=fetch_k)
        reranked_hits = self._expand_candidate_documents(query, reranked_hits, fetch_k=fetch_k)
        query_lower = query.lower()
        query_terms = set(self._normalize_terms(query))
        subject_terms = set()
        for subject in self._extract_query_subject_candidates(query):
            subject_terms.update(self._normalize_terms(subject))
        focus_terms = self._extract_query_focus_terms(query)
        for hit in reranked_hits:
            retrieval_queries = hit.get("retrieval_queries", []) or []
            extra_queries = []
            for retrieval_query in retrieval_queries:
                if not isinstance(retrieval_query, str):
                    continue
                normalized_query = retrieval_query.strip()
                lowered_query_text = normalized_query.lower()
                if not normalized_query or lowered_query_text == query_lower:
                    continue
                if any(marker in lowered_query_text for marker in ("title:", "snippet:", "evidence:", "{", "}")):
                    continue
                query_tokens = self._normalize_terms(normalized_query)
                if not query_tokens or len(query_tokens) > 10:
                    continue
                extra_queries.append(normalized_query)
            bridge_bonus = min(0.24, 0.08 * len(extra_queries))

            title = ((hit.get("graph_context") or {}).get("doc_title") or hit.get("title") or "").strip()
            title_terms = set(self._normalize_terms(title))
            chunk_text = hit.get("chunk_text", "") or ""
            if "\n" in chunk_text:
                chunk_text = chunk_text.split("\n", 1)[1]
            text_terms = set(self._normalize_terms(chunk_text))
            retrieval_terms = set()
            for rq in extra_queries:
                retrieval_terms.update(self._normalize_terms(rq))
            entity_anchor_overlap = len((title_terms | text_terms) & retrieval_terms)
            focus_overlap = len((title_terms | text_terms) & focus_terms)
            answer_bearing_overlap = len(text_terms & focus_terms)

            if entity_anchor_overlap:
                bridge_bonus += min(0.55, 0.18 * entity_anchor_overlap)
            if answer_bearing_overlap and extra_queries and entity_anchor_overlap:
                bridge_bonus += min(0.45, 0.16 * answer_bearing_overlap)
            if focus_overlap and extra_queries and entity_anchor_overlap:
                bridge_bonus += min(0.22, 0.08 * focus_overlap)

            # Penalize second-hop hits that neither anchor to bridge entity nor carry target focus.
            if extra_queries and entity_anchor_overlap == 0:
                bridge_bonus -= 0.22
            if (
                subject_terms
                and title_terms
                and (title_terms & subject_terms)
                and extra_queries
                and entity_anchor_overlap == 0
            ):
                bridge_bonus -= 0.14
            if extra_queries and entity_anchor_overlap > 0 and answer_bearing_overlap == 0:
                bridge_bonus -= 0.08
            if extra_queries and retrieval_terms and title_terms and title_terms.isdisjoint(retrieval_terms) and answer_bearing_overlap == 0:
                bridge_bonus -= 0.08

            base_score = _safe_numeric(
                hit.get("cross_encoder_score"),
                _safe_numeric(hit.get("rrf_score"), _safe_numeric(hit.get("score", 0.0)))
            )
            hit["bridge_entity_anchor_overlap"] = entity_anchor_overlap
            hit["bridge_focus_overlap"] = focus_overlap
            hit["answer_bearing_overlap"] = answer_bearing_overlap
            hit["bridge_bonus"] = bridge_bonus
            hit["final_rank_score"] = base_score + bridge_bonus
        reranked_hits.sort(
            key=lambda x: _safe_numeric(
                x.get("final_rank_score"),
                _safe_numeric(x.get("cross_encoder_score"), _safe_numeric(x.get("score", 0.0)))
            ),
            reverse=True
        )

        # 4. Graph Expansion with bounded contribution tracking.
        pre_graph_candidates = reranked_hits[: max(top_k * 3, 10)]
        grounded_candidates = self._graph_expansion(pre_graph_candidates, query=query)
        graph_expansion_enriched = sum(1 for hit in grounded_candidates if hit.get("graph_context"))

        # 5. Selective Symbolic Graph Retrieval.
        symbolic_triggered = self._should_use_symbolic_search(query, query_graph, reranked_hits)
        deep_hits: List[Dict[str, Any]] = []
        if symbolic_triggered:
            deep_hits = self._deep_graph_search(query, query_graph=query_graph)
            if deep_hits:
                logger.info("Injecting %d symbolic paths from Neo4j after selectivity gate.", len(deep_hits))

        combined_candidates = [*deep_hits, *grounded_candidates]
        pre_source_mix = dict(Counter(self._infer_source_type(hit) for hit in combined_candidates))

        # 6. Role-aware final evidence selection.
        selected_hits, role_debug = self._select_role_aware_candidates(
            query=query,
            candidates=combined_candidates,
            top_k=top_k,
        )
        self.last_search_debug = {
            **self.last_search_debug,
            "first_stage_candidates": first_stage_count,
            "reranked_candidates": len(reranked_hits),
            "pre_pack_count": len(combined_candidates),
            "graph_expansion_added": 0,
            "graph_expansion_enriched": graph_expansion_enriched,
            "dynamic_cypher_added": len(deep_hits),
            "symbolic_triggered": symbolic_triggered,
            "source_type_mix_pre": pre_source_mix,
            "source_type_mix": role_debug.get("source_type_mix", {}),
            "evidence_role_mix": role_debug.get("evidence_role_mix", {}),
            "duplicates_removed": role_debug.get("duplicates_removed", 0),
            "bridge_coverage_score": role_debug.get("bridge_coverage_score", 0.0),
            "support_coverage_score": role_debug.get("support_coverage_score", 0.0),
            "final_context_count": len(selected_hits),
            "post_pack_count": len(selected_hits),
            "role_aware_pool_count": role_debug.get("role_aware_pool_count", len(combined_candidates)),
            "role_aware_selected_count": role_debug.get("role_aware_selected_count", len(selected_hits)),
            "graph_symbolic_candidates": role_debug.get("graph_symbolic_candidates", 0),
            "graph_symbolic_kept": role_debug.get("graph_symbolic_kept", 0),
            "graph_symbolic_dropped": role_debug.get("graph_symbolic_dropped", 0),
            "corroborated_graph_kept": role_debug.get("corroborated_graph_kept", 0),
            "selection_drop_reasons": role_debug.get("selection_drop_reasons", {}),
        }

        return selected_hits[:top_k]

    def _graph_expansion(self, base_hits: List[Dict[str, Any]], query: str = "") -> List[Dict[str, Any]]:
        """
        Takes the top hits from Dense/Sparse, queries Neo4j for their parent
        document and adjacent chunks (1-hop), and injects this graph context.
        Also attempts to resolve Entities (NER) if mentioned in the query.
        """
        if not self.enable_graph_retrieval or not hasattr(self, 'neo4j_driver') or not self.neo4j_driver or not base_hits:
            return base_hits
            
        enriched_hits = []
        
        # Extract Entities for potential Graph Pathing
        entities = self._extract_entities(query)
        if entities:
            logger.info("NER detected entities for Graph Pathing: %s", entities)
            # The cypher handles entity matching efficiently directly parameterized
        
        with self.neo4j_driver.session() as session:
            for hit in base_hits:
                doc_id = hit["doc_id"]
                idx = hit["chunk_index"]
                
                cypher = """
                MATCH (d:Document {id: $doc_id})-[:HAS_CHUNK]->(c:Chunk {index: $idx})
                OPTIONAL MATCH (d)-[:HAS_CHUNK]->(prev:Chunk {index: $idx - 1})
                OPTIONAL MATCH (d)-[:HAS_CHUNK]->(next:Chunk {index: $idx + 1})
                
                // 2-Hop Entity Traversal: Find entities mentioned in this chunk, 
                // and then find OTHER chunks (in other docs) that mention the exact same entities.
                OPTIONAL MATCH (c)-[:MENTIONS]->(e:Entity)<-[:MENTIONS]-(cross_c:Chunk)<-[:HAS_CHUNK]-(cross_d:Document)
                WHERE e.name IN $entities AND cross_c.id <> c.id AND cross_d.id <> d.id
                
                RETURN 
                    d.title AS doc_title,
                    d.section AS doc_section,
                    c.chunk_text AS exact_hit_text,
                    prev.chunk_text AS prev_context,
                    next.chunk_text AS next_context,
                    collect(DISTINCT e.name) AS shared_entities,
                    collect(DISTINCT cross_d.title + ": " + cross_c.chunk_text)[0..3] AS cross_document_texts
                """
                
                try:
                    result = session.run(cypher, doc_id=doc_id, idx=idx, entities=entities).single()
                    if result:
                        graph_context = {
                            "doc_title": result["doc_title"],
                            "doc_section": result["doc_section"],
                            "prev_context": result["prev_context"],
                            "next_context": result["next_context"],
                            "shared_entities": result["shared_entities"],
                            "cross_document_texts": result["cross_document_texts"]
                        }
                        enriched_hit = {**hit, "graph_context": graph_context}
                        enriched_hits.append(enriched_hit)
                    else:
                        enriched_hits.append({**hit, "graph_context": None})
                except Exception as e:
                    logger.error("Graph Expansion error for Doc %s, Idx %s: %s", doc_id, idx, str(e))
                    enriched_hits.append({**hit, "graph_context": None})
                    
        return enriched_hits

    def _deep_graph_search(self, query: str, query_graph: Optional[List[Dict[str, str]]] = None) -> List[Dict[str, Any]]:
        """
        Attempts to generate and execute a dynamic Cypher query for multi-hop 
        relationship reasoning. Uses self-healing: if Cypher fails, the error
        is fed back to the LLM for repair.
        """
        if not self.enable_graph_retrieval or not self.neo4j_driver or not self.cypher_gen:
            return []
            
        # Enhance prompt with query graph if available
        context_query = query
        if query_graph:
            graph_json = json.dumps(query_graph)
            context_query = f"{query}\n[Semantic Context]: {graph_json}"
        
        # Define the executor function for self-healing validation
        _captured_records = []
        
        def _execute_cypher(cypher_str: str):
            """Validates Cypher by executing it. Raises on failure."""
            with self.neo4j_driver.session() as session:
                result = session.run(cypher_str)
                _captured_records.clear()
                for record in result:
                    _captured_records.append(record.data())
            
        # Use self-healing generation
        cypher = self.cypher_gen.generate_with_healing(context_query, executor_fn=_execute_cypher)
        if not cypher:
            return []
            
        logger.info(f"Validated Dynamic Cypher: {cypher}")
        graph_hits = []
        
        # Use the already-captured records from the healing validation
        if _captured_records:
            for data in _captured_records:
                text_content = " | ".join([str(v) for v in data.values()])
                graph_hits.append({
                    "id": f"graph_{uuid.uuid4().hex[:8]}",
                    "doc_id": "Knowledge Graph",
                    "chunk_index": -1,
                    "chunk_text": f"[Symbolic Reasoning]: {text_content}",
                    "score": 1.0,
                    "source": "dynamic_cypher",
                    "graph_context": {"doc_title": "Neo4j Symbolic Path", "doc_section": "Multi-hop reasoning"}
                })
        else:
            # Fallback: re-execute valid Cypher if records weren't captured
            try:
                with self.neo4j_driver.session() as session:
                    records = session.run(cypher)
                    for record in records:
                        data = record.data()
                        text_content = " | ".join([str(v) for v in data.values()])
                        graph_hits.append({
                            "id": f"graph_{uuid.uuid4().hex[:8]}",
                            "doc_id": "Knowledge Graph",
                            "chunk_index": -1,
                            "chunk_text": f"[Symbolic Reasoning]: {text_content}",
                            "score": 1.0,
                            "source": "dynamic_cypher",
                            "graph_context": {"doc_title": "Neo4j Symbolic Path", "doc_section": "Multi-hop reasoning"}
                        })
            except Exception as e:
                logger.error(f"Dynamic Cypher re-execution failed: {e}")
                return []
        
        if graph_hits:
            logger.info(f"Dynamic Cypher returned {len(graph_hits)} symbolic paths.")
        return graph_hits

    def search(
        self,
        query: str,
        top_k: int = 5,
        query_graph: Optional[List[Dict[str, str]]] = None,
        additional_queries: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Main orchestration method for Tri-Engine Fusion (Now with Cross-Encoder).
        """
        logger.info("Initiating Hybrid Search for query: '%s'", query)
        candidate_pool = self.collect_candidate_pool(
            query,
            top_k=top_k,
            additional_queries=additional_queries,
            include_follow_ups=True
        )
        final_grounded_results = self.finalize_candidates(
            query,
            candidate_pool,
            top_k=top_k,
            query_graph=query_graph
        )
        self.last_search_debug = {
            **self.last_search_debug,
            "candidate_pool_count": len(candidate_pool),
            "final_result_count": len(final_grounded_results),
        }
        
        logger.info("Hybrid Search Complete. Yielding %s results.", len(final_grounded_results))
        return final_grounded_results
