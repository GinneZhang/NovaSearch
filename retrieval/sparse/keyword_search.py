"""
Sparse Retrieval Module for NovaSearch.
"""
import logging
from typing import List, Dict, Any
from abc import ABC, abstractmethod
from abc import ABC, abstractmethod

try:
    from psycopg2.extras import DictCursor
except ImportError:
    pass

logger = logging.getLogger(__name__)

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
                doc_id, index, chunk_text, 
                ts_rank(to_tsvector('english', chunk_text), websearch_to_tsquery('english', %s)) AS rank_score
            FROM chunks
            WHERE to_tsvector('english', chunk_text) @@ websearch_to_tsquery('english', %s)
            ORDER BY rank_score DESC
            LIMIT %s;
        """
        
        results = []
        try:
            with self.pg_conn.cursor(cursor_factory=DictCursor) as cur:
                cur.execute(sql, (query, query, top_k))
                rows = cur.fetchall()
                for row in rows:
                    results.append({
                        "doc_id": row["doc_id"],
                        "chunk_index": row["index"],
                        "chunk_text": row["chunk_text"],
                        "score": float(row["rank_score"]),
                        "source": "sparse"
                    })
        except Exception as e:
            logger.error("Error during Sparse Search: %s", str(e))
            self.pg_conn.rollback()
            
        return results
