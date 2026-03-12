"""
FAISS Dense Retrieval Module for NovaSearch.
"""
import os
import json
import logging
from typing import List, Dict, Any

try:
    import faiss
    import numpy as np
except ImportError:
    pass

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    pass

from retrieval.dense.vector_search import BaseDenseRetriever

logger = logging.getLogger(__name__)

class FAISSDenseRetriever(BaseDenseRetriever):
    """Handles vector similarity search via local FAISS index."""
    
    def __init__(self, embedding_model_name: str = "all-MiniLM-L6-v2", index_dir: str = "faiss_index"):
        self.embedding_model_name = embedding_model_name
        self.index_dir = index_dir
        self.index_path = os.path.join(index_dir, "index.faiss")
        self.metadata_path = os.path.join(index_dir, "metadata.json")
        
        logger.info("Loading embedding model %s for FAISS...", embedding_model_name)
        try:
            self.embedding_model = SentenceTransformer(embedding_model_name)
            self.dim = self.embedding_model.get_sentence_embedding_dimension()
        except Exception as e:
            logger.error("FAISS couldn't load sentence transformer: %s", e)
            self.embedding_model = None
            self.dim = 384
            
        self.index = None
        self.metadata_store = []
        
        self._load_index()

    def _load_index(self):
        if not os.path.exists(self.index_dir):
            os.makedirs(self.index_dir)
            
        if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
            try:
                self.index = faiss.read_index(self.index_path)
                with open(self.metadata_path, 'r') as f:
                    self.metadata_store = json.load(f)
                logger.info("Loaded FAISS index with %d documents.", self.index.ntotal)
            except Exception as e:
                logger.error("Failed to load FAISS index: %s", e)
                self.index = faiss.IndexFlatL2(self.dim)
        else:
            self.index = faiss.IndexFlatL2(self.dim)
            self.metadata_store = []
            logger.info("Initialized new FAISS IndexFlatL2.")

    def _save_index(self):
        faiss.write_index(self.index, self.index_path)
        with open(self.metadata_path, 'w') as f:
            json.dump(self.metadata_store, f)

    def add_documents(self, doc_id: str, chunks: list):
        """Adds a list of chunks from ingestion pipeline into the local FAISS store."""
        if not self.index or not self.embedding_model:
            return
            
        texts = [c["chunk_text"] for c in chunks]
        embeddings = self.embedding_model.encode(texts)
        
        # FAISS expects float32
        embeddings = np.array(embeddings).astype('float32')
        self.index.add(embeddings)
        
        for c in chunks:
            self.metadata_store.append({
                "doc_id": doc_id,
                "chunk_index": c["chunk_metadata"].get("sequence_index", 0),
                "chunk_text": c["chunk_text"],
                "source": "dense_faiss"
            })
            
        self._save_index()
        logger.info("Added %d chunks to FAISS index.", len(chunks))

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        if not self.index or self.index.ntotal == 0 or not self.embedding_model:
            return []
            
        query_vector = self.embedding_model.encode([query])
        query_vector = np.array(query_vector).astype('float32')
        
        # FAISS returns distances and indices
        distances, indices = self.index.search(query_vector, top_k)
        
        results = []
        for i in range(len(indices[0])):
            idx = indices[0][i]
            if idx == -1:
                continue
            dist = distances[0][i]
            
            # Convert L2 distance to a basic similarity score (inverting so higher is better, scaled approx)
            score = 1.0 / (1.0 + float(dist))
            
            meta = self.metadata_store[idx]
            results.append({
                "doc_id": meta["doc_id"],
                "chunk_index": meta["chunk_index"],
                "chunk_text": meta["chunk_text"],
                "score": score,
                "source": "dense"
            })
            
        return results
