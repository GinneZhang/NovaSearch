"""
Sliding Window Chunker logic for AsterScope Ingestion Module.
"""
import logging
from typing import List, Dict, Any

try:
    import tiktoken
except ImportError:
    raise ImportError("Please install tiktoken: pip install tiktoken")

logger = logging.getLogger(__name__)

class SlidingWindowChunker:
    """
    Splits document text into configurable overlapping chunks based on tokens using tiktoken.
    This acts as a drop-in replacement for SemanticChunker.
    """
    
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 128,
        model_name: str = "gpt-3.5-turbo",
        embedding_model_name: str = "all-MiniLM-L6-v2"
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.tokenizer = tiktoken.encoding_for_model(model_name)
        self.embedding_model_name = embedding_model_name
        
        # Also initialize the embedding model so it perfectly mimics SemanticChunker's properties if accessed
        try:
            from sentence_transformers import SentenceTransformer
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
        except ImportError:
            self.embedding_model = None

    def _inject_context(self, chunk_text: str, metadata: Dict[str, Any]) -> str:
        doc_id = metadata.get("doc_id", "Unknown ID")
        title = metadata.get("title", "Unknown Title")
        section = metadata.get("section", "General")
        prefix = f"[Doc: {title} | ID: {doc_id} | Section: {section}]\n"
        return prefix + chunk_text

    def chunk_document(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Processes a document into precise overlapping chunks, returning formatted chunks identical to SemanticChunker.
        """
        if not text or not text.strip():
            return []

        tokens = self.tokenizer.encode(text)
        chunks = []
        
        if len(tokens) == 0:
            return []

        step = max(self.chunk_size - self.chunk_overlap, 1)
        
        for idx_start in range(0, len(tokens), step):
            idx_end = idx_start + self.chunk_size
            chunk_tokens = tokens[idx_start:idx_end]
            raw_chunk = self.tokenizer.decode(chunk_tokens)
            
            contextualized_chunk = self._inject_context(raw_chunk, metadata)
            
            chunks.append({
                "chunk_text": contextualized_chunk,
                "chunk_metadata": metadata.copy(),
                "token_count": len(chunk_tokens)
            })

            if idx_end >= len(tokens):
                break

        return chunks
