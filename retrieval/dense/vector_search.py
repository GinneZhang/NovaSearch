"""
Dense Retrieval Module for AsterScope.
"""
import logging
import re
import threading
from typing import List, Dict, Any
from abc import ABC, abstractmethod
from abc import ABC, abstractmethod

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    pass

try:
    from psycopg2.extras import DictCursor
except ImportError:
    pass

logger = logging.getLogger(__name__)


def _normalize_terms(text: str) -> List[str]:
    return [token for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9_-]+", (text or "").lower()) if len(token) > 2]


def _title_overlap_boost(query: str, title: str) -> float:
    query_terms = set(_normalize_terms(query))
    title_terms = set(_normalize_terms(title))
    if not query_terms or not title_terms:
        return 0.0
    overlap = len(query_terms & title_terms)
    if overlap == 0:
        return 0.0
    return min(0.35, overlap * 0.08)

class BaseDenseRetriever(ABC):
    """Abstract base class for dense vector retrievers."""
    
    @abstractmethod
    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        pass

class PGVectorDenseRetriever(BaseDenseRetriever):
    """Handles semantic vector similarity search via pgvector."""
    
    def __init__(self, pg_conn, embedding_model_name: str = "all-MiniLM-L6-v2"):
        self.pg_conn = pg_conn
        logger.info("Loading embedding model for Dense Search...")
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self._encode_lock = threading.Lock()
        
    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Performs vector similarity search on pgvector."""
        if not self.pg_conn:
            logger.warning("Postgres offline. Skipping dense search.")
            return []
            
        # 1. Embed query
        with self._encode_lock:
            query_vector = self.embedding_model.encode(query).tolist()
        
        sql = """
            SELECT 
                c.doc_id,
                c.index,
                c.chunk_text,
                d.title,
                1 - (embedding <=> %s::vector) AS similarity_score
            FROM chunks c
            LEFT JOIN documents d ON d.id = c.doc_id
            ORDER BY embedding <=> %s::vector
            LIMIT %s;
        """
        
        results = []
        try:
            with self.pg_conn.cursor(cursor_factory=DictCursor) as cur:
                vector_str = "[" + ",".join([str(x) for x in query_vector]) + "]"
                cur.execute(sql, (vector_str, vector_str, top_k))
                rows = cur.fetchall()
                for row in rows:
                    title = row["title"] or ""
                    base_score = float(row["similarity_score"])
                    boosted_score = base_score + _title_overlap_boost(query, title)
                    results.append({
                        "doc_id": row["doc_id"],
                        "chunk_index": row["index"],
                        "chunk_text": row["chunk_text"],
                        "title": title,
                        "score": boosted_score,
                        "base_score": base_score,
                        "source": "dense"
                    })
        except Exception as e:
            logger.error("Error during Dense Search: %s", str(e))
            self.pg_conn.rollback()
            
        return results
