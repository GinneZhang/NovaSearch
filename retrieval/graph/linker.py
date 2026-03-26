"""
Vector-Based Entity Linker for AsterScope Knowledge Graph.

Maps extracted entity mentions to canonical Neo4j node IDs using
embedding similarity search against the Entity node index.
"""

import os
import logging
from typing import List, Dict, Optional, Tuple, Any

logger = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
except ImportError:
    SentenceTransformer = None
    np = None

try:
    from neo4j import GraphDatabase
except ImportError:
    GraphDatabase = None


class EntityLinker:
    """
    Precision Entity Linker that performs vector similarity search
    against Neo4j Entity nodes to map extracted mentions to canonical
    graph node IDs with high confidence.
    
    Probabilistic linking: if confidence is between similarity_threshold
    and CLARIFICATION_THRESHOLD, the result is marked as 'ambiguous' and
    the agent should trigger a proactive clarification chunk.
    """
    
    CLARIFICATION_THRESHOLD = 0.8  # Below this -> trigger clarification
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", similarity_threshold: float = 0.7):
        self.similarity_threshold = similarity_threshold
        self.model = None
        self.neo4j_driver = None
        self._entity_cache: Dict[str, List[Tuple[str, str, List[float]]]] = {}
        
        # Initialize embedding model
        if SentenceTransformer:
            try:
                self.model = SentenceTransformer(model_name)
                logger.info(f"EntityLinker: Loaded embedding model '{model_name}'")
            except Exception as e:
                logger.warning(f"EntityLinker: Failed to load model: {e}")
        
        # Initialize Neo4j connection
        uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        user = os.getenv("NEO4J_USER", "neo4j")
        password = os.getenv("NEO4J_PASSWORD", "")
        if GraphDatabase and password:
            try:
                self.neo4j_driver = GraphDatabase.driver(uri, auth=(user, password))
                logger.info("EntityLinker: Connected to Neo4j")
            except Exception as e:
                logger.warning(f"EntityLinker: Failed to connect to Neo4j: {e}")
    
    def _load_entity_index(self) -> List[Tuple[str, str, List[float]]]:
        """
        Load all Entity nodes from Neo4j and compute their embeddings.
        Returns list of (name, type, embedding) tuples.
        Cached after first call.
        """
        cache_key = "all_entities"
        if cache_key in self._entity_cache:
            return self._entity_cache[cache_key]
        
        if not self.neo4j_driver or not self.model:
            return []
        
        entities = []
        try:
            with self.neo4j_driver.session() as session:
                result = session.run("MATCH (e:Entity) RETURN e.name AS name, e.type AS type")
                for record in result:
                    name = record["name"]
                    etype = record.get("type", "UNKNOWN")
                    if name:
                        embedding = self.model.encode(name).tolist()
                        entities.append((name, etype, embedding))
            
            self._entity_cache[cache_key] = entities
            logger.info(f"EntityLinker: Indexed {len(entities)} entities from Neo4j")
        except Exception as e:
            logger.error(f"EntityLinker: Failed to load entity index: {e}")
        
        return entities
    
    def link(self, mention: str) -> Optional[Dict[str, Any]]:
        """
        Link a text mention to a canonical Neo4j Entity node.
        
        Returns:
            Dict with keys:
                'name': canonical entity name
                'type': entity type
                'confidence': float similarity score
                'ambiguous': bool - True if confidence is between
                             similarity_threshold and CLARIFICATION_THRESHOLD,
                             signaling the agent should ask the user to disambiguate.
            Returns None if no match above similarity_threshold.
        """
        if not self.model:
            return None
        
        entity_index = self._load_entity_index()
        if not entity_index:
            return None
        
        mention_embedding = self.model.encode(mention)
        
        best_match = None
        best_score = -1.0
        
        for name, etype, entity_emb in entity_index:
            # Cosine similarity
            entity_vec = np.array(entity_emb)
            similarity = float(
                np.dot(mention_embedding, entity_vec) /
                (np.linalg.norm(mention_embedding) * np.linalg.norm(entity_vec) + 1e-8)
            )
            
            if similarity > best_score:
                best_score = similarity
                best_match = {
                    "name": name,
                    "type": etype,
                    "confidence": round(float(similarity), 4),
                    "ambiguous": False
                }
        
        if best_match and best_score >= self.similarity_threshold:
            # Mark as ambiguous if below clarification threshold
            if best_score < self.CLARIFICATION_THRESHOLD:
                best_match["ambiguous"] = True
                logger.info(
                    f"EntityLinker: '{mention}' -> '{best_match['name']}' "
                    f"(conf={best_score:.3f}, AMBIGUOUS — clarification recommended)"
                )
            else:
                logger.info(f"EntityLinker: '{mention}' -> '{best_match['name']}' (conf={best_score:.3f})")
            return best_match
        
        logger.debug(f"EntityLinker: No confident match for '{mention}' (best={best_score:.3f})")
        return None
    
    def link_batch(self, mentions: List[str]) -> Dict[str, Optional[Dict[str, Any]]]:
        """Link multiple mentions at once."""
        return {mention: self.link(mention) for mention in mentions}
    
    def get_ambiguous_entities(self, results: Dict[str, Optional[Dict[str, Any]]]) -> List[str]:
        """Filter linked results to find entities that need user clarification."""
        return [
            mention for mention, result in results.items()
            if result and result.get("ambiguous", False)
        ]
    
    def invalidate_cache(self):
        """Clear the entity index cache (e.g., after ingestion)."""
        self._entity_cache.clear()
        logger.info("EntityLinker: Cache invalidated")

