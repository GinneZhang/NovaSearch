"""
Structured Query Graph Parser for NovaSearch.
Converts natural language queries into semantic triplets for guided retrieval.
Validates extracted triplets against the live Neo4j schema.
"""

import os
import logging
import json
from typing import List, Dict, Any, Optional, Set
import openai

logger = logging.getLogger(__name__)

try:
    from neo4j import GraphDatabase
except ImportError:
    GraphDatabase = None


class QueryGraphParser:
    """
    Extracts semantic entities and relationships (triplets) from user queries
    to build a structured query graph. Validates triplets against the live
    Neo4j schema and aligns unknown terms to the nearest valid term.
    """

    def __init__(self, model: str = "gpt-3.5-turbo"):
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.client = openai.OpenAI(api_key=self.api_key) if self.api_key else None
        self.model = model
        
        # Schema cache for validation
        self._schema_labels: Set[str] = set()
        self._schema_rels: Set[str] = set()
        self._schema_properties: Set[str] = set()
        self._schema_loaded = False
    
    def _load_schema(self):
        """Load live Neo4j schema for triplet validation."""
        if self._schema_loaded:
            return
        
        if not GraphDatabase:
            self._schema_loaded = True
            return
        
        try:
            uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
            user = os.getenv("NEO4J_USER", "neo4j")
            password = os.getenv("NEO4J_PASSWORD", "")
            driver = GraphDatabase.driver(uri, auth=(user, password))
            
            with driver.session() as session:
                # Get node labels
                labels_result = session.run("CALL db.labels()")
                self._schema_labels = {r["label"] for r in labels_result}
                
                # Get relationship types
                rels_result = session.run("CALL db.relationshipTypes()")
                self._schema_rels = {r["relationshipType"] for r in rels_result}
                
                # Get property keys
                props_result = session.run("CALL db.propertyKeys()")
                self._schema_properties = {r["propertyKey"] for r in props_result}
            
            driver.close()
            self._schema_loaded = True
            logger.info(f"QueryGraphParser: Loaded schema — {len(self._schema_labels)} labels, "
                        f"{len(self._schema_rels)} rels, {len(self._schema_properties)} properties")
        except Exception as e:
            logger.warning(f"QueryGraphParser: Schema loading failed: {e}")
            self._schema_loaded = True
    
    def _align_to_schema(self, term: str, candidates: Set[str]) -> str:
        """
        Align an unknown term to the nearest schema-valid term using
        case-insensitive substring matching.
        """
        if not candidates:
            return term
        
        term_lower = term.lower().replace("_", " ").replace("-", " ")
        
        # Exact match (case-insensitive)
        for c in candidates:
            if c.lower() == term_lower:
                return c
        
        # Substring containment match
        for c in candidates:
            if term_lower in c.lower() or c.lower() in term_lower:
                return c
        
        # No match — return original
        return term
    
    def _validate_triplets(self, triplets: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Validate and align extracted triplets against the live Neo4j schema.
        Unknown subjects/objects are mapped to nearest labels; unknown
        relations are mapped to nearest relationship types.
        """
        self._load_schema()
        
        if not self._schema_labels and not self._schema_rels:
            return triplets  # No schema to validate against
        
        validated = []
        for triplet in triplets:
            subject = triplet.get("subject", "")
            relation = triplet.get("relation", "")
            obj = triplet.get("object", "")
            
            # Align relation to known relationship types
            aligned_rel = self._align_to_schema(relation, self._schema_rels)
            
            # Align subject/object to known labels or properties
            all_terms = self._schema_labels | self._schema_properties
            aligned_subject = self._align_to_schema(subject, all_terms)
            aligned_obj = self._align_to_schema(obj, all_terms)
            
            validated.append({
                "subject": aligned_subject,
                "relation": aligned_rel,
                "object": aligned_obj,
                "original_subject": subject if subject != aligned_subject else None,
                "original_relation": relation if relation != aligned_rel else None,
                "original_object": obj if obj != aligned_obj else None
            })
            
            # Log alignments
            if subject != aligned_subject or relation != aligned_rel or obj != aligned_obj:
                logger.info(f"QueryGraphParser: Aligned triplet — "
                            f"({subject}→{aligned_subject}, {relation}→{aligned_rel}, {obj}→{aligned_obj})")
        
        return validated

    def parse(self, query: str) -> List[Dict[str, str]]:
        """
        Parses a query into a list of triplets: [{"subject": "...", "relation": "...", "object": "..."}]
        Validates against live Neo4j schema and aligns unknown terms.
        """
        if not self.client:
            logger.warning("No OpenAI client for QueryGraphParser. Skipping.")
            return []

        prompt = f"""
        Extract the core semantic relationships from the following user query as a list of triplets.
        Format: Subject -> Relation -> Object
        
        Examples:
        Query: "Who is the CEO of Acme Corp?"
        Triplets: [ {{"subject": "Acme Corp", "relation": "HAS_CEO", "object": "person"}} ]
        
        Query: "Find documents about the new trade policy"
        Triplets: [ {{"subject": "documents", "relation": "ABOUT", "object": "trade policy"}} ]
        
        Output valid JSON only.
        
        User Query: {query}
        Triplets:
        """

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                response_format={ "type": "json_object" }
            )
            content = response.choices[0].message.content
            data = json.loads(content)
            triplets = data.get("triplets", [])
            
            # Validate and align triplets against schema
            validated_triplets = self._validate_triplets(triplets)
            
            logger.info(f"Extracted and validated Query Graph: {validated_triplets}")
            return validated_triplets
        except Exception as e:
            logger.error(f"Query parsing failed: {e}")
            return []
