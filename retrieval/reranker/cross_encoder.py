"""
Cross-Encoder Reranking Module for NovaSearch.
"""
import logging
from typing import List, Dict, Any
from abc import ABC, abstractmethod

try:
    from sentence_transformers import CrossEncoder
except ImportError:
    pass

logger = logging.getLogger(__name__)

class BaseReranker(ABC):
    """Abstract base class for cross-encoder or other rescoring models."""
    
    @abstractmethod
    def rerank(self, query: str, hits: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        pass

class CrossEncoderReranker(BaseReranker):
    """
    Advanced reranking using a HuggingFace Cross-Encoder.
    Unlike Bi-Encoders (dense vectors), Cross-Encoders process the query and 
    document simultaneously, capturing deep semantic interactions at a higher 
    computational cost.
    """
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        logger.info("Loading Cross-Encoder model: %s", model_name)
        try:
            self.model = CrossEncoder(model_name, max_length=512)
        except Exception as e:
            logger.error("Failed to load Cross-Encoder: %s", str(e))
            self.model = None

    def rerank(self, query: str, hits: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Takes a candidate list from the first-stage retrievers and rescores them.
        """
        if not self.model or not hits:
            logger.warning("Cross-Encoder offline or no hits. Returning original list.")
            return hits[:top_k]
            
        logger.info("Cross-Encoder rescoring %d candidate hits...", len(hits))
        
        # Prepare pairs: (Query, Document Text)
        pairs = []
        for hit in hits:
            title = (hit.get("title") or "").strip()
            chunk_text = hit.get("chunk_text", "")
            if title:
                doc_text = f"{title}\n{chunk_text}"
            else:
                doc_text = chunk_text
            pairs.append([query, doc_text])
        
        try:
            # Predict scores
            scores = self.model.predict(pairs)
            
            # Attach scores to original hits
            for i, hit in enumerate(hits):
                hit["cross_encoder_score"] = float(scores[i])
                
            # Sort descending by the new score
            hits.sort(key=lambda x: x.get("cross_encoder_score", -999.0), reverse=True)
            
            return hits[:top_k]
            
        except Exception as e:
            logger.error("Cross-Encoder scoring failed: %s", str(e))
            return hits[:top_k]
