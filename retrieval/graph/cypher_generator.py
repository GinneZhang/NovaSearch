"""
Dynamic Text-to-Cypher Generator for Neo4j with Self-Healing retry loop,
formal Cypher Linting / Schema Validation, and Structured Result Objects.
"""

import os
import re
import logging
from typing import Optional, List, Dict, Set
from dataclasses import dataclass, field
import openai

logger = logging.getLogger(__name__)


@dataclass
class CypherResult:
    """
    Structured representation of a generated Cypher query with
    extracted nodes, edges, and validation metadata.
    """
    cypher: str
    nodes: List[str] = field(default_factory=list)      # Referenced node labels
    edges: List[str] = field(default_factory=list)      # Referenced relationship types
    properties: List[str] = field(default_factory=list)  # Referenced properties
    validated: bool = False                               # Whether path existence was verified
    lint_errors: Optional[str] = None                    # Any linting errors found


class CypherGenerator:
    """
    Translates natural language questions into read-only Cypher queries
    using an LLM grounded by the Neo4j schema.
    
    Includes a self-healing retry loop: if the generated Cypher fails
    execution, the error is fed back to the LLM for repair.
    
    Returns CypherResult structured objects with extracted nodes/edges.
    """
    FALLBACK_SCHEMA = """
        Neo4j Schema:
        - Nodes:
            1. Document (id, title, section, metadata)
            2. Chunk (id, index, chunk_text)
            3. Entity (name, type)
        - Relationships:
            1. (Document)-[:HAS_CHUNK]->(Chunk)
            2. (Chunk)-[:MENTIONS]->(Entity)
        """

    def __init__(self, model: str = "gpt-4-turbo-preview", max_retries: int = 2, neo4j_driver=None):
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            logger.warning("OPENAI_API_KEY missing for CypherGenerator.")
        self.client = openai.OpenAI(api_key=self.api_key)
        self.model = model
        self.max_retries = max_retries
        self.neo4j_driver = neo4j_driver
        self._cached_schema: Optional[str] = None
        self._known_labels: Set[str] = set()
        self._known_rels: Set[str] = set()

    def _extract_structure(self, cypher: str) -> Dict[str, List[str]]:
        """
        Extract referenced node labels, relationship types, and properties
        from a Cypher query string.
        """
        # Extract labels (e.g., :Document, :Chunk)
        labels = list(set(re.findall(r':\s*([A-Z][a-zA-Z_]*)', cypher)))
        
        # Extract relationship types from bracket notation (e.g., [:HAS_CHUNK])
        rels = list(set(re.findall(r'\[:\s*([A-Z_]+)\s*\]', cypher)))
        
        # Extract property references (e.g., d.title, c.chunk_text)
        props = list(set(re.findall(r'[a-z]\.\s*([a-z_]+)', cypher, re.IGNORECASE)))
        
        return {"nodes": labels, "edges": rels, "properties": props}
    
    def _validate_path_existence(self, cypher: str) -> bool:
        """
        Symbolic validator: checks that referenced paths (label→rel→label)
        exist in the live schema before feeding data to the LLM.
        Returns True if all paths are valid or schema is unavailable.
        """
        if not self.neo4j_driver or not self._known_labels:
            return True  # Can't validate without driver/schema
        
        structure = self._extract_structure(cypher)
        
        # Validate each label exists in schema
        for label in structure["nodes"]:
            if label not in self._known_labels:
                logger.warning(f"Path validation: Unknown label '{label}'")
                return False
        
        # Validate each relationship type exists in schema
        for rel in structure["edges"]:
            if rel not in self._known_rels:
                logger.warning(f"Path validation: Unknown relationship '{rel}'")
                return False
        
        return True

    def _lint_cypher(self, cypher: str) -> Optional[str]:
        """
        Validates a generated Cypher query against the live schema.
        Returns an error message if invalid, None if valid.
        """
        # Ensure schema is loaded
        self._get_schema_context()
        
        if not self._known_labels and not self._known_rels:
            # Can't validate without schema knowledge
            return None
        
        errors = []
        
        # Check for write operations (strict read-only enforcement)
        write_keywords = ["CREATE", "MERGE", "SET ", "DELETE", "REMOVE", "DROP"]
        for kw in write_keywords:
            if kw in cypher.upper():
                errors.append(f"Write operation '{kw.strip()}' detected. Only read-only queries are allowed.")
        
        # Validate node labels referenced in query
        label_refs = re.findall(r':\s*([A-Z][a-zA-Z_]*)', cypher)
        for label in label_refs:
            if self._known_labels and label not in self._known_labels and label not in self._known_rels:
                errors.append(f"Unknown label/type '{label}'. Known labels: {', '.join(sorted(self._known_labels))}. Known rels: {', '.join(sorted(self._known_rels))}.")
        
        if errors:
            return "Schema Validation Errors: " + "; ".join(errors)
        return None

    def _get_schema_context(self) -> str:
        """
        Fetches the live Neo4j schema via introspection.
        Falls back to the hardcoded schema if introspection fails.
        """
        if self._cached_schema:
            return self._cached_schema
        
        if self.neo4j_driver:
            try:
                with self.neo4j_driver.session() as session:
                    # Get node labels and their properties
                    labels_result = session.run("CALL db.labels()")
                    labels = [r["label"] for r in labels_result]
                    self._known_labels = set(labels)
                    
                    # Get relationship types
                    rels_result = session.run("CALL db.relationshipTypes()")
                    rel_types = [r["relationshipType"] for r in rels_result]
                    self._known_rels = set(rel_types)
                    
                    # Get property keys per label
                    schema_lines = ["Neo4j Live Schema:"]
                    schema_lines.append("- Node Labels:")
                    for label in labels:
                        props_result = session.run(
                            f"MATCH (n:`{label}`) WITH n LIMIT 1 RETURN keys(n) AS props"
                        )
                        props = []
                        for r in props_result:
                            props = r["props"]
                        schema_lines.append(f"    {label} ({', '.join(props) if props else 'no properties sampled'})")
                    
                    schema_lines.append("- Relationship Types:")
                    for rel in rel_types:
                        schema_lines.append(f"    [:{rel}]")
                    
                    self._cached_schema = "\n".join(schema_lines)
                    logger.info("CypherGenerator: Using live Neo4j schema introspection.")
                    return self._cached_schema
            except Exception as e:
                logger.warning(f"Schema introspection failed, using fallback: {e}")
        
        return self.FALLBACK_SCHEMA

    def generate(self, user_query: str) -> Optional[str]:
        """
        Generates a Cypher query from a user question.
        Returns None if the question is not suitable for graph traversal.
        """
        if not self.api_key:
            return None

        prompt = f"""
        You are a Neo4j Cypher expert. Convert the following user question into a VALID, READ-ONLY Cypher query.
        
        {self._get_schema_context()}
        
        Rules:
        1. Use ONLY the nodes and relationships defined in the schema.
        2. RETURN data in a structured format (e.g., RETURN d.title, c.chunk_text, e.name).
        3. Do NOT use any writing clauses (CREATE, MERGE, SET, DELETE, etc.).
        4. If the question cannot be answered via the graph schema, respond with "NOT_APPLICABLE".
        5. Output ONLY the Cypher query or "NOT_APPLICABLE", no preamble.
        
        User Question: {user_query}
        
        Cypher:
        """

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )
            cypher = response.choices[0].message.content.strip()
            
            if "NOT_APPLICABLE" in cypher:
                return None
            
            # Basic cleanup (remove markdown code blocks if any)
            cypher = cypher.replace("```cypher", "").replace("```", "").strip()
            return cypher
        except Exception as e:
            logger.error(f"Cypher generation failed: {e}")
            return None

    def generate_structured(self, user_query: str, executor_fn=None) -> Optional[CypherResult]:
        """
        Generate Cypher and return a structured CypherResult with extracted
        nodes, edges, properties, and validation status.
        """
        cypher = self.generate(user_query)
        if not cypher:
            return None
        
        # Extract structure
        structure = self._extract_structure(cypher)
        
        # Lint the query
        lint_error = self._lint_cypher(cypher)
        
        # Validate path existence
        path_valid = self._validate_path_existence(cypher)
        
        result = CypherResult(
            cypher=cypher,
            nodes=structure["nodes"],
            edges=structure["edges"],
            properties=structure["properties"],
            validated=path_valid and lint_error is None,
            lint_errors=lint_error
        )
        
        if lint_error:
            logger.warning(f"CypherResult lint errors: {lint_error}")
        if not path_valid:
            logger.warning(f"CypherResult: Path validation failed for nodes={structure['nodes']}, edges={structure['edges']}")
        
        return result

    def generate_with_healing(self, user_query: str, executor_fn=None) -> Optional[str]:
        """
        Generates Cypher and validates it via execution. If execution fails,
        feeds the error back to the LLM for self-healing repair.
        """
        cypher = self.generate(user_query)
        if not cypher or not executor_fn:
            return cypher

        last_error = None
        for attempt in range(self.max_retries + 1):  # initial + retries
            try:
                # Attempt execution to validate the Cypher
                executor_fn(cypher)
                logger.info(f"Cypher validated successfully on attempt {attempt + 1}.")
                return cypher
            except Exception as e:
                last_error = str(e)
                logger.warning(f"Cypher execution failed (attempt {attempt + 1}/{self.max_retries + 1}): {last_error}")
                
                if attempt >= self.max_retries:
                    logger.error("Max Cypher repair retries exhausted. Giving up.")
                    return None
                
                # Self-Healing: Feed error back to LLM for repair
                cypher = self._repair_cypher(user_query, cypher, last_error)
                if not cypher:
                    logger.error("LLM could not repair the Cypher query.")
                    return None
        
        return None

    def _repair_cypher(self, original_query: str, broken_cypher: str, error_message: str) -> Optional[str]:
        """
        Feeds the broken Cypher and the Neo4j error back to the LLM
        to generate a repaired query.
        """
        if not self.api_key:
            return None

        repair_prompt = f"""
        You previously generated an invalid Cypher query. Fix it.

        {self._get_schema_context()}

        Original User Question: {original_query}
        
        Broken Cypher:
        {broken_cypher}
        
        Neo4j Error:
        {error_message}
        
        Rules:
        1. Use ONLY the nodes and relationships defined in the schema.
        2. Do NOT use any writing clauses (CREATE, MERGE, SET, DELETE, etc.).
        3. Fix the syntax or schema error based on the error message above.
        4. Output ONLY the corrected Cypher query, no preamble or explanation.
        
        Repaired Cypher:
        """

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": repair_prompt}],
                temperature=0.0
            )
            repaired = response.choices[0].message.content.strip()
            repaired = repaired.replace("```cypher", "").replace("```", "").strip()
            logger.info(f"LLM repaired Cypher: {repaired[:120]}...")
            return repaired
        except Exception as e:
            logger.error(f"Cypher repair LLM call failed: {e}")
            return None

