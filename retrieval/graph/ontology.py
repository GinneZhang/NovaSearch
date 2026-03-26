"""
Hardened Ontology Manager for AsterScope.

Provides a formal mapping layer between extracted query triplets and the
canonical Neo4j schema. Replaces fuzzy string matching with embedding-based
similarity at a strict confidence threshold (>0.9). If a term cannot
be mapped with sufficient confidence, the system triggers a Clarification
request rather than guessing.
"""

import os
import re
import logging
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer, util as st_util
except ImportError:
    SentenceTransformer = None
    st_util = None

try:
    from neo4j import GraphDatabase
except ImportError:
    GraphDatabase = None


# --------------------------------------------------------------------------- #
#  Data Structures
# --------------------------------------------------------------------------- #

@dataclass
class OntologyMapping:
    """Result of mapping one term to the ontology."""
    original_term: str
    canonical_term: Optional[str] = None
    confidence: float = 0.0
    mapped: bool = False


@dataclass
class AlignmentResult:
    """Full alignment result for a set of triplets."""
    aligned_triplets: List[Dict[str, Any]] = field(default_factory=list)
    unmapped_terms: List[str] = field(default_factory=list)
    all_mapped: bool = True
    needs_clarification: bool = False
    clarification_message: Optional[str] = None


# --------------------------------------------------------------------------- #
#  Ontology Manager
# --------------------------------------------------------------------------- #

class OntologyManager:
    """
    Formal ontology alignment layer.

    Loads canonical classes (node labels, relationship types, property keys)
    from the live Neo4j schema and builds an embedding index. Incoming query
    triplet terms are mapped against this index at a strict confidence
    threshold (default 0.9).  Unmapped terms trigger a Clarification response
    instead of silent degradation.

    Usage
    -----
    >>> mgr = OntologyManager(neo4j_driver=driver)
    >>> result = mgr.align_triplets([
    ...     {"subject": "employee", "predicate": "WORKS_AT", "object": "Acme"}
    ... ])
    >>> if result.needs_clarification:
    ...     # Surface clarification_message to the user
    """

    CONFIDENCE_THRESHOLD = 0.9
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"

    def __init__(
        self,
        neo4j_driver=None,
        confidence_threshold: float = None,
        embedding_model: str = None,
    ):
        self.neo4j_driver = neo4j_driver
        self.threshold = confidence_threshold or self.CONFIDENCE_THRESHOLD
        self.model_name = embedding_model or self.EMBEDDING_MODEL

        # Schema vocabulary (populated lazily)
        self._labels: List[str] = []
        self._rel_types: List[str] = []
        self._properties: List[str] = []
        self._all_terms: List[str] = []

        # Embedding model + pre-computed term embeddings
        self._encoder: Optional[Any] = None
        self._term_embeddings: Optional[Any] = None
        self._loaded = False

    # ------------------------------------------------------------------ #
    #  Schema Loading
    # ------------------------------------------------------------------ #

    def _load_schema(self) -> None:
        """Introspect the live Neo4j schema for labels, rels, and properties."""
        if self._loaded:
            return

        if self.neo4j_driver is None or GraphDatabase is None:
            logger.warning("OntologyManager: No Neo4j driver — using empty schema.")
            self._loaded = True
            return

        try:
            with self.neo4j_driver.session() as session:
                self._labels = [
                    r["label"] for r in session.run("CALL db.labels()")
                ]
                self._rel_types = [
                    r["relationshipType"]
                    for r in session.run("CALL db.relationshipTypes()")
                ]
                prop_result = session.run("CALL db.propertyKeys()")
                self._properties = [r["propertyKey"] for r in prop_result]

            self._all_terms = list(
                set(self._labels + self._rel_types + self._properties)
            )
            logger.info(
                "OntologyManager: Loaded %d labels, %d rels, %d props from Neo4j.",
                len(self._labels),
                len(self._rel_types),
                len(self._properties),
            )
        except Exception as exc:
            logger.error("OntologyManager: Schema introspection failed: %s", exc)

        self._loaded = True

    # ------------------------------------------------------------------ #
    #  Embedding Index
    # ------------------------------------------------------------------ #

    def _ensure_encoder(self) -> None:
        """Lazy-load SentenceTransformer and pre-encode schema terms."""
        self._load_schema()

        if self._encoder is not None:
            return

        if SentenceTransformer is None:
            logger.warning(
                "OntologyManager: sentence-transformers not installed. "
                "Falling back to exact-match mode."
            )
            return

        try:
            self._encoder = SentenceTransformer(self.model_name)
            if self._all_terms:
                self._term_embeddings = self._encoder.encode(
                    self._all_terms, convert_to_tensor=True, show_progress_bar=False
                )
            logger.info(
                "OntologyManager: Encoded %d schema terms.", len(self._all_terms)
            )
        except Exception as exc:
            logger.error("OntologyManager: Encoder init failed: %s", exc)
            self._encoder = None

    # ------------------------------------------------------------------ #
    #  Single-Term Mapping
    # ------------------------------------------------------------------ #

    def map_term(self, term: str) -> OntologyMapping:
        """
        Map a single term to the canonical ontology.

        Returns an ``OntologyMapping`` with *mapped=True* only if the best
        match exceeds the confidence threshold.
        """
        self._ensure_encoder()

        if not self._all_terms:
            return OntologyMapping(original_term=term, confidence=0.0)

        # 1. Exact (case-insensitive) match — fast path
        term_lower = term.lower().strip()
        for canonical in self._all_terms:
            if canonical.lower() == term_lower:
                return OntologyMapping(
                    original_term=term,
                    canonical_term=canonical,
                    confidence=1.0,
                    mapped=True,
                )

        # 2. Embedding similarity — slow path
        if self._encoder is not None and self._term_embeddings is not None:
            try:
                query_emb = self._encoder.encode(
                    term, convert_to_tensor=True, show_progress_bar=False
                )
                scores = st_util.cos_sim(query_emb, self._term_embeddings)[0]
                best_idx = int(scores.argmax())
                best_score = float(scores[best_idx])

                if best_score >= self.threshold:
                    return OntologyMapping(
                        original_term=term,
                        canonical_term=self._all_terms[best_idx],
                        confidence=round(best_score, 4),
                        mapped=True,
                    )
                else:
                    logger.info(
                        "OntologyManager: '%s' best match '%s' (%.3f) below threshold %.2f",
                        term,
                        self._all_terms[best_idx],
                        best_score,
                        self.threshold,
                    )
                    return OntologyMapping(
                        original_term=term, confidence=round(best_score, 4)
                    )
            except Exception as exc:
                logger.error("OntologyManager: Embedding lookup failed: %s", exc)

        # 3. Substring fallback (case-insensitive) with lower confidence cap
        for canonical in self._all_terms:
            if term_lower in canonical.lower() or canonical.lower() in term_lower:
                return OntologyMapping(
                    original_term=term,
                    canonical_term=canonical,
                    confidence=0.75,  # Below threshold — will trigger clarification
                    mapped=False,
                )

        return OntologyMapping(original_term=term, confidence=0.0)

    # ------------------------------------------------------------------ #
    #  Triplet Alignment
    # ------------------------------------------------------------------ #

    def align_triplets(
        self, triplets: List[Dict[str, str]]
    ) -> AlignmentResult:
        """
        Align a list of semantic triplets against the canonical ontology.

        Each triplet is expected to have keys *subject*, *predicate*, *object*.
        Any term that cannot be mapped at >= threshold triggers a Clarification.
        """
        result = AlignmentResult()

        for triplet in triplets:
            aligned: Dict[str, Any] = {}
            for role in ("subject", "predicate", "object"):
                raw = triplet.get(role, "")
                if not raw:
                    aligned[role] = raw
                    continue

                mapping = self.map_term(raw)
                if mapping.mapped:
                    aligned[role] = mapping.canonical_term
                    aligned[f"{role}_confidence"] = mapping.confidence
                else:
                    aligned[role] = raw  # Keep original — flagged for clarification
                    aligned[f"{role}_confidence"] = mapping.confidence
                    result.unmapped_terms.append(raw)
                    result.all_mapped = False

            result.aligned_triplets.append(aligned)

        if not result.all_mapped:
            unique_unmapped = sorted(set(result.unmapped_terms))
            result.needs_clarification = True
            result.clarification_message = (
                f"The following terms could not be confidently matched to the "
                f"knowledge base schema (confidence < {self.threshold}): "
                f"{', '.join(repr(t) for t in unique_unmapped)}. "
                f"Could you rephrase or clarify what you mean by "
                f"{'these terms' if len(unique_unmapped) > 1 else repr(unique_unmapped[0])}?"
            )
            logger.warning(
                "OntologyManager: Clarification needed for: %s", unique_unmapped
            )

        return result
