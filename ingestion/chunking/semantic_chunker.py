"""
Semantic Chunker logic for AsterScope Ingestion Module.

This module provides the SemanticChunker class which splits documents into atomic
sentences, embeds them, and uses Agglomerative Clustering to group conceptually 
similar, contiguous sentences into chunks. Finally, it injects document metadata 
into the text to maintain context.
"""

import logging
import re
import threading
from typing import List, Dict, Any
import numpy as np

try:
    import spacy
except ImportError:
    raise ImportError("Please install spacy: pip install spacy")

try:
    from sklearn.cluster import AgglomerativeClustering
except ImportError:
    raise ImportError("Please install scikit-learn: pip install scikit-learn")

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    raise ImportError("Please install sentence-transformers: pip install sentence-transformers")


logger = logging.getLogger(__name__)


class SemanticChunker:
    """
    Semantic Chunker that utilizes Embedding Clustering to group atomic sentences
    into coherent chunks while preserving semantic boundaries.
    """

    def __init__(
        self,
        embedding_model_name: str = "all-MiniLM-L6-v2",
        similarity_threshold: float = 0.5,
        max_tokens_per_chunk: int = 512,
        overlap_tokens: int = 128,
    ):
        """
        Initializes the SemanticChunker.

        Args:
            embedding_model_name (str): The huggingface model name for embeddings.
            similarity_threshold (float): Cosine distance threshold for clustering.
                                          Higher values allow more dissimilar sentences to merge.
            max_tokens_per_chunk (int): Maximum tokens allowed per chunk before forcing a split.
            overlap_tokens (int): Target number of tokens to overlap when max_tokens is reached.
        """
        self.embedding_model_name = embedding_model_name
        self.similarity_threshold = similarity_threshold
        self.max_tokens_per_chunk = max_tokens_per_chunk
        self.overlap_tokens = overlap_tokens
        self._model_lock = threading.RLock()
        
        # Load spaCy for atomic sentence splitting
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy en_core_web_sm unavailable. Falling back to regex sentence splitting.")
            self.nlp = None

        # Load embedding model
        logger.info("Loading embedding model: %s", embedding_model_name)
        self.embedding_model = SentenceTransformer(embedding_model_name)

    def _get_atomic_sentences(self, text: str) -> List[str]:
        """Splits raw text into atomic sentences using spaCy."""
        if self.nlp:
            doc = self.nlp(text)
            return [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        return [sent.strip() for sent in re.split(r"(?<=[.!?])\s+", text) if sent.strip()]

    def _get_embeddings(self, sentences: List[str]) -> np.ndarray:
        """Generates embeddings for a list of sentences."""
        with self._model_lock:
            return self.embedding_model.encode(sentences)

    def encode_text(self, text: str):
        """Thread-safe single-text embedding for downstream persistence."""
        with self._model_lock:
            return self.embedding_model.encode(text)

    def _count_tokens(self, text: str) -> int:
        """
        Counts the number of tokens in a string.
        Falls back to word splitting if the model's tokenizer isn't exposed.
        """
        if hasattr(self.embedding_model, "tokenizer"):
            # The underlying HF tokenizer is not safe for concurrent access on all backends.
            with self._model_lock:
                return len(self.embedding_model.tokenizer.encode(text, add_special_tokens=False))
        return len(text.split())

    def _inject_context(self, chunk_text: str, metadata: Dict[str, Any]) -> str:
        """Prepends metadata context to the chunk text."""
        doc_id = metadata.get("doc_id", "Unknown ID")
        title = metadata.get("title", "Unknown Title")
        section = metadata.get("section", "General")
        
        prefix = f"[Doc: {title} | ID: {doc_id} | Section: {section}]\n"
        return prefix + chunk_text

    def chunk_document(
        self, text: str, metadata: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Processes a document, applying semantic chunking, and returns formatted chunks.

        Args:
            text (str): The raw document text.
            metadata (Dict[str, Any]): Metadata for the document (doc_id, title, section).

        Returns:
            List[Dict[str, Any]]: A list of chunk dictionaries containing 'chunk_text', 
                                  'chunk_metadata', and 'token_count'.
        """
        # FastAPI serves ingestion in a threadpool, but spaCy + the embedding model
        # are both shared process-wide. Serializing chunk construction avoids flaky
        # benchmark-time failures under concurrent ingestion.
        with self._model_lock:
            if not text or not text.strip():
                return []

            # 1. Atomic Splitting
            sentences = self._get_atomic_sentences(text)
            if not sentences:
                return []

            if len(sentences) == 1:
                # Handle extremely short documents gracefully
                chunk_text = sentences[0]
                contextualized_text = self._inject_context(chunk_text, metadata)
                return [{
                    "chunk_text": contextualized_text,
                    "chunk_metadata": metadata.copy(),
                    "token_count": self._count_tokens(contextualized_text)
                }]

            # 2. Embedding
            embeddings = self._get_embeddings(sentences)

            # 3. Clustering
            # Create a connectivity matrix to enforce adjacency
            # (only contiguous sentences can be grouped into the same cluster).
            n_sentences = len(sentences)
            connectivity = np.zeros((n_sentences, n_sentences))
            for i in range(n_sentences - 1):
                connectivity[i, i + 1] = 1
                connectivity[i + 1, i] = 1

            clustering_model = AgglomerativeClustering(
                n_clusters=None,
                metric="cosine",
                linkage="average",
                distance_threshold=self.similarity_threshold,
                connectivity=connectivity
            )

            # Fit clustering model
            cluster_labels = clustering_model.fit_predict(embeddings)

            # 4. Group sentences by clusters and enforce max token limits
            chunks = []
            current_chunk_sentences = []
            current_cluster = cluster_labels[0]
            current_token_count = 0

            for i, sentence in enumerate(sentences):
                label = cluster_labels[i]
                sentence_tokens = self._count_tokens(sentence)

                # Start a new chunk if:
                # - The semantic cluster label changes (boundary detected)
                # - OR adding this sentence exceeds the token limit
                limit_reached = (current_token_count + sentence_tokens > self.max_tokens_per_chunk)

                if (label != current_cluster or limit_reached) and current_chunk_sentences:
                    # Finalize current chunk
                    raw_chunk = " ".join(current_chunk_sentences)
                    contextualized_chunk = self._inject_context(raw_chunk, metadata)
                    chunks.append({
                        "chunk_text": contextualized_chunk,
                        "chunk_metadata": metadata.copy(),
                        "token_count": self._count_tokens(contextualized_chunk)
                    })

                    # Calculate overlap if splitting due to token limit
                    overlap_sentences = []
                    overlap_token_count = 0
                    if limit_reached:
                        for prev_sentence in reversed(current_chunk_sentences):
                            prev_tokens = self._count_tokens(prev_sentence)
                            if overlap_token_count + prev_tokens <= self.overlap_tokens:
                                overlap_sentences.insert(0, prev_sentence)
                                overlap_token_count += prev_tokens
                            else:
                                break

                    # Reset for new chunk
                    current_chunk_sentences = overlap_sentences + [sentence]
                    current_cluster = label
                    current_token_count = overlap_token_count + sentence_tokens
                else:
                    current_chunk_sentences.append(sentence)
                    current_token_count += sentence_tokens
                    # Handle edge case where a single sentence might be longer than the token limit on its own
                    if not current_chunk_sentences:
                        current_cluster = label

            # Finalize the last chunk
            if current_chunk_sentences:
                raw_chunk = " ".join(current_chunk_sentences)
                contextualized_chunk = self._inject_context(raw_chunk, metadata)
                chunks.append({
                    "chunk_text": contextualized_chunk,
                    "chunk_metadata": metadata.copy(),
                    "token_count": self._count_tokens(contextualized_chunk)
                })

            return chunks
