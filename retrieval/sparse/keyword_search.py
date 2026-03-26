"""
Sparse Retrieval Module for AsterScope.
"""
import logging
import re
from typing import List, Dict, Any
from abc import ABC, abstractmethod
from abc import ABC, abstractmethod

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
    return min(0.75, overlap * 0.18)

class BaseSparseRetriever(ABC):
    """Abstract base class for sparse keyword retrievers."""
    
    @abstractmethod
    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        pass

class PostgresFTSSparseRetriever(BaseSparseRetriever):
    """Handles PostgreSQL Full-Text Search (Keyword Matching)."""
    
    def __init__(self, pg_conn):
        self.pg_conn = pg_conn

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Performs Keyword Matching search."""
        if not self.pg_conn:
             logger.warning("Postgres offline. Skipping sparse search.")
             return []
             
        sql = """
            SELECT 
                c.doc_id,
                c.index,
                c.chunk_text,
                d.title,
                ts_rank(
                    setweight(to_tsvector('english', coalesce(d.title, '')), 'A') ||
                    setweight(to_tsvector('english', c.chunk_text), 'B'),
                    websearch_to_tsquery('english', %s)
                ) AS rank_score
            FROM chunks c
            LEFT JOIN documents d ON d.id = c.doc_id
            WHERE (
                setweight(to_tsvector('english', coalesce(d.title, '')), 'A') ||
                setweight(to_tsvector('english', c.chunk_text), 'B')
            ) @@ websearch_to_tsquery('english', %s)
            ORDER BY rank_score DESC, c.index ASC
            LIMIT %s;
        """
        
        results = []
        try:
            with self.pg_conn.cursor(cursor_factory=DictCursor) as cur:
                cur.execute(sql, (query, query, top_k))
                rows = cur.fetchall()
                for row in rows:
                    title = row["title"] or ""
                    base_score = float(row["rank_score"])
                    boosted_score = base_score + _title_overlap_boost(query, title)
                    results.append({
                        "doc_id": row["doc_id"],
                        "chunk_index": row["index"],
                        "chunk_text": row["chunk_text"],
                        "title": title,
                        "score": boosted_score,
                        "base_score": base_score,
                        "source": "sparse"
                    })
        except Exception as e:
            logger.error("Error during Sparse Search: %s", str(e))
            self.pg_conn.rollback()
            
        return results
