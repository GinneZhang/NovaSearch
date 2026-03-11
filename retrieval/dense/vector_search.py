"""
Dense Retrieval Module for NovaSearch.
"""
import logging
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
        
    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Performs vector similarity search on pgvector."""
        if not self.pg_conn:
            logger.warning("Postgres offline. Skipping dense search.")
            return []
            
        # 1. Embed query
        query_vector = self.embedding_model.encode(query).tolist()
        
        sql = """
            SELECT 
                doc_id, index, chunk_text, 
                1 - (embedding <=> %s::vector) AS similarity_score
            FROM chunks
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
                    results.append({
                        "doc_id": row["doc_id"],
                        "chunk_index": row["index"],
                        "chunk_text": row["chunk_text"],
                        "score": float(row["similarity_score"]),
                        "source": "dense"
                    })
        except Exception as e:
            logger.error("Error during Dense Search: %s", str(e))
            self.pg_conn.rollback()
            
        return results
