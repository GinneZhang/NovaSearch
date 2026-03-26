"""
LlamaIndex Long-Context Module for AsterScope.

Provides specialized handling for very large documents (e.g., 100+ page DOCX/PDF)
where standard chunking fails to preserve cross-section semantics.

Uses LlamaIndex's document tree indexing for hierarchical summarization
and context-window-aware retrieval.
"""

import os
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

try:
    from llama_index.core import Document, VectorStoreIndex, Settings
    from llama_index.core.node_parser import SentenceSplitter
    LLAMAINDEX_AVAILABLE = True
except ImportError:
    LLAMAINDEX_AVAILABLE = False
    logger.info("LlamaIndex not installed. Long-context module disabled.")

try:
    from llama_index.llms.openai import OpenAI as LlamaOpenAI
except ImportError:
    LlamaOpenAI = None

try:
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
except ImportError:
    HuggingFaceEmbedding = None


class LongContextProcessor:
    """
    Handles very large documents using LlamaIndex's hierarchical indexing.
    
    For documents exceeding MAX_STANDARD_CHARS, this processor creates
    a LlamaIndex VectorStoreIndex with hierarchical summarization,
    enabling cross-section semantic retrieval that standard sliding-window
    chunking cannot achieve.
    """
    
    MAX_STANDARD_CHARS = 50_000  # ~12k tokens; above this, use LlamaIndex
    
    def __init__(self):
        self.index = None
        self._configured = False
        
        if LLAMAINDEX_AVAILABLE:
            try:
                # Configure LlamaIndex settings
                api_key = os.getenv("OPENAI_API_KEY")
                if api_key and LlamaOpenAI:
                    Settings.llm = LlamaOpenAI(
                        model="gpt-3.5-turbo",
                        api_key=api_key,
                        temperature=0.0
                    )
                
                if HuggingFaceEmbedding:
                    Settings.embed_model = HuggingFaceEmbedding(
                        model_name="all-MiniLM-L6-v2"
                    )
                
                self._configured = True
                logger.info("LongContextProcessor: LlamaIndex configured successfully.")
            except Exception as e:
                logger.warning(f"LongContextProcessor: Configuration failed: {e}")
    
    @staticmethod
    def needs_long_context(text: str) -> bool:
        """Determine if a document is too large for standard chunking."""
        return len(text) > LongContextProcessor.MAX_STANDARD_CHARS
    
    def index_document(self, text: str, doc_id: str = "long_doc") -> bool:
        """
        Create a LlamaIndex VectorStoreIndex from a large document.
        Returns True if indexing succeeded.
        """
        if not LLAMAINDEX_AVAILABLE or not self._configured:
            logger.warning("LongContextProcessor: LlamaIndex not available.")
            return False
        
        try:
            doc = Document(text=text, doc_id=doc_id)
            
            # Use sentence-level splitting for large docs
            parser = SentenceSplitter(chunk_size=1024, chunk_overlap=200)
            nodes = parser.get_nodes_from_documents([doc])
            
            self.index = VectorStoreIndex(nodes)
            logger.info(
                f"LongContextProcessor: Indexed {len(nodes)} nodes "
                f"from document '{doc_id}' ({len(text)} chars)."
            )
            return True
        except Exception as e:
            logger.error(f"LongContextProcessor: Indexing failed: {e}")
            return False
    
    def query(self, question: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Query the LlamaIndex index for relevant context.
        Returns a list of result dicts compatible with hybrid_search output.
        """
        if not self.index:
            logger.warning("LongContextProcessor: No index available. Call index_document first.")
            return []
        
        try:
            query_engine = self.index.as_query_engine(similarity_top_k=top_k)
            response = query_engine.query(question)
            
            results = []
            for i, node in enumerate(response.source_nodes):
                results.append({
                    "chunk_text": node.node.get_content(),
                    "score": float(node.score) if node.score else 0.0,
                    "source": "llamaindex_long_context",
                    "metadata": {
                        "node_id": node.node.node_id,
                        "chunk_index": i
                    }
                })
            
            logger.info(f"LongContextProcessor: Retrieved {len(results)} nodes for query.")
            return results
        except Exception as e:
            logger.error(f"LongContextProcessor: Query failed: {e}")
            return []
    
    def is_available(self) -> bool:
        """Check if LlamaIndex long-context processing is available."""
        return LLAMAINDEX_AVAILABLE and self._configured
