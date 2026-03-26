"""
Elasticsearch Sparse Retrieval Module for AsterScope.
"""
import os
import logging
from typing import List, Dict, Any

try:
    from elasticsearch import Elasticsearch
except ImportError:
    pass

from retrieval.sparse.keyword_search import BaseSparseRetriever

logger = logging.getLogger(__name__)

class ElasticSparseRetriever(BaseSparseRetriever):
    """Handles Sparse TF-IDF/BM25 Keyword Matching via Elasticsearch."""
    
    def __init__(self, index_name: str = "asterscope-chunks"):
        self.index_name = index_name
        es_url = os.getenv("ELASTIC_URL", "http://localhost:9200")
        
        try:
            self.client = Elasticsearch(es_url)
            if self.client.ping():
                logger.info("Connected to Elasticsearch cluster at %s", es_url)
                self._ensure_index()
            else:
                logger.warning("Elasticsearch ping failed. Node might be offline.")
                self.client = None
        except Exception as e:
            logger.error("Failed connecting to Elasticsearch: %s", str(e))
            self.client = None

    def _ensure_index(self):
        if not self.client.indices.exists(index=self.index_name):
            mapping = {
                "mappings": {
                    "properties": {
                        "doc_id": {"type": "keyword"},
                        "chunk_index": {"type": "integer"},
                        "chunk_text": {"type": "text"}
                    }
                }
            }
            self.client.indices.create(index=self.index_name, body=mapping)
            logger.info("Created Elasticsearch index: %s", self.index_name)

    def add_documents(self, doc_id: str, chunks: list):
        if not self.client:
            return
            
        try:
            from elasticsearch.helpers import bulk
            actions = []
            for c in chunks:
                seq_idx = c["chunk_metadata"].get("sequence_index", 0)
                chunk_id = f"{doc_id}_{seq_idx}"
                actions.append({
                    "_index": self.index_name,
                    "_id": chunk_id,
                    "_source": {
                        "doc_id": doc_id,
                        "chunk_index": seq_idx,
                        "chunk_text": c["chunk_text"]
                    }
                })
            success, _ = bulk(self.client, actions)
            self.client.indices.refresh(index=self.index_name)
            logger.info("Successfully added %d chunks to Elasticsearch.", success)
        except Exception as e:
            logger.error("Error inserting into Elasticsearch: %s", str(e))

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        if not self.client:
            return []
            
        try:
            res = self.client.search(
                index=self.index_name,
                body={
                    "query": {
                        "match": {
                            "chunk_text": query
                        }
                    },
                    "size": top_k
                }
            )
            
            results = []
            hits = res.get("hits", {}).get("hits", [])
            
            for hit in hits:
                source = hit.get("_source", {})
                results.append({
                    "doc_id": source.get("doc_id", "unknown"),
                    "chunk_index": source.get("chunk_index", 0),
                    "chunk_text": source.get("chunk_text", ""),
                    "score": hit.get("_score", 0.0),
                    "source": "sparse"
                })
            
            return results
        except Exception as e:
            logger.error("Elasticsearch search query failed: %s", str(e))
            return []
