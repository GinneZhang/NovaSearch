"""
Multimodal Vision Search Retriever (CLIP Integration).
"""

import os
import logging
import threading
from typing import List, Dict, Any, Union

try:
    import torch
    from PIL import Image
    from sentence_transformers import SentenceTransformer
except ImportError:
    pass

import psycopg2

logger = logging.getLogger(__name__)

class PGVectorVisionRetriever:
    """
    Handles true multimodal dense search using OpenAI CLIP (clip-ViT-B-32).
    Both images and text are embedded into the same 512-dimensional vector space.
    """
    
    def __init__(self, model_name: str = "clip-ViT-B-32", pg_conn=None):
        logger.info(f"Initializing PGVectorVisionRetriever with model {model_name}...")
        
        try:
            # SentenceTransformers cleanly wraps CLIP processing for both text and images
            self.model = SentenceTransformer(model_name)
            logger.info("CLIP model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load CLIP model: {e}")
            self.model = None
        self._encode_lock = threading.Lock()

        if pg_conn:
            self.pg_conn = pg_conn
            self._init_table()
        else:
            self._connect_db()
            self._init_table()

    def _connect_db(self):
        pg_dsn = os.getenv("DATABASE_URL", 
            f"dbname={os.getenv('POSTGRES_DB', 'asterscope')} "
            f"user={os.getenv('POSTGRES_USER', 'postgres')} "
            f"password={os.getenv('POSTGRES_PASSWORD', 'postgres_secure_password')} "
            f"host={os.getenv('POSTGRES_HOST', 'localhost')} "
            f"port={os.getenv('POSTGRES_PORT', '5432')}"
        )
        try:
            self.pg_conn = psycopg2.connect(pg_dsn)
        except Exception as e:
            logger.error("Failed to connect to PGVector for vision search: %s", str(e))
            self.pg_conn = None

    def _init_table(self):
        if not self.pg_conn:
            return
            
        try:
            with self.pg_conn.cursor() as cur:
                # clip-ViT-B-32 outputs 512-dimensional vectors
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS vision_embeddings (
                        id VARCHAR(255) PRIMARY KEY,
                        doc_id VARCHAR(255),
                        chunk_index INTEGER,
                        chunk_text TEXT,
                        embedding vector(512),
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """)
                # HNSW Index for L2 distance
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS vision_embeddings_hnsw_idx 
                    ON vision_embeddings 
                    USING hnsw (embedding vector_l2_ops);
                """)
            self.pg_conn.commit()
            logger.info("Vision embeddings table initialized.")
        except Exception as e:
            logger.error("Failed to initialize vision_embeddings table: %s", str(e))
            self.pg_conn.rollback()

    def embed(self, data: Union[str, "Image.Image"]) -> List[float]:
        """
        Embeds either a PIL Image or a text string into the CLIP vector space.
        """
        if not self.model:
            return [0.0] * 512
        try:
            with self._encode_lock:
                return self.model.encode(data).tolist()
        except Exception as e:
            logger.error(f"CLIP embedding failed: {e}")
            return [0.0] * 512

    def insert_vision_chunk(self, doc_id: str, chunk_index: int, chunk_text: str, embedding: List[float]):
        """
        Inserts a multimodal chunk into the PGVector database.
        """
        if not self.pg_conn:
            self._connect_db()
            self._init_table()
        if not self.pg_conn:
            logger.error("Still no pg_conn, cannot insert vision chunk.")
            return
            
        chunk_id = f"{doc_id}_{chunk_index}"
        emb_str = "[" + ",".join([str(x) for x in embedding]) + "]"
        
        try:
            with self.pg_conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO vision_embeddings (id, doc_id, chunk_index, chunk_text, embedding) 
                    VALUES (%s, %s, %s, %s, %s::vector)
                    ON CONFLICT (id) DO UPDATE SET 
                        chunk_text = EXCLUDED.chunk_text,
                        embedding = EXCLUDED.embedding,
                        created_at = CURRENT_TIMESTAMP;
                """, (chunk_id, doc_id, chunk_index, chunk_text, emb_str))
            self.pg_conn.commit()
        except Exception as e:
            logger.error("Failed to insert vision chunk: %s", str(e))
            self.pg_conn.rollback()

    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Performs an exact KNN search on the vision embeddings.
        Returns the top_k closest chunks.
        """
        if not self.pg_conn:
            self._connect_db()
            self._init_table()
        if not self.pg_conn:
            logger.error("Still no pg_conn, cannot search vision space.")
            return []
            
        emb_str = "[" + ",".join([str(x) for x in query_embedding]) + "]"
        
        try:
            with self.pg_conn.cursor() as cur:
                # Using Euclidean distance (<->) for similarity ordering
                cur.execute("""
                    SELECT id, doc_id, chunk_index, chunk_text, 1 - (embedding <-> %s::vector) as similarity_score
                    FROM vision_embeddings
                    ORDER BY embedding <-> %s::vector
                    LIMIT %s;
                """, (emb_str, emb_str, top_k))
                
                results = cur.fetchall()
                hits = []
                for row in results:
                    hits.append({
                        "id": row[0],
                        "doc_id": row[1],
                        "chunk_index": row[2],
                        "chunk_text": row[3],
                        "score": float(row[4]),
                        "source": "clip_vision"
                    })
                return hits
        except Exception as e:
            logger.error("Vision Search failed: %s", str(e))
            self.pg_conn.rollback()
            return []
