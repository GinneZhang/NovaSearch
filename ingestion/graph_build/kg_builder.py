"""
Knowledge Graph Builder for NovaSearch.

This module is responsible for taking document chunks and ingesting them into Neo4j.
It connects via the official neo4j driver and constructs the following schema:
(:Document {id, title, section}) -[:HAS_CHUNK {sequence_index}]-> (:Chunk {chunk_text, tokens})
"""

import os
import logging
from typing import List, Dict, Any

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from neo4j import GraphDatabase
except ImportError:
    raise ImportError("Please install neo4j python driver: pip install neo4j")

try:
    import spacy
except ImportError:
    pass

logger = logging.getLogger(__name__)

class KGBuilder:
    """
    Handles the ingestion of document chunks into the Neo4j Knowledge Graph.
    Ensures idempotency using MERGE operations.
    """

    def __init__(self, uri: str = None, user: str = None, password: str = None):
        """
        Initialize the Neo4j driver connection.
        If credentials aren't provided, it attempts to load from environment variables.
        """
        self.uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.user = user or os.getenv("NEO4J_USER", "neo4j")
        self.password = password or os.getenv("NEO4J_PASSWORD", "password")
        
        try:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            # Verify connectivity
            self.driver.verify_connectivity()
            logger.info("Successfully connected to Neo4j Knowledge Graph.")
        except Exception as e:
            logger.error("Failed to connect to Neo4j: %s", str(e))
            self.driver = None
            
        try:
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("Successfully loaded spaCy en_core_web_sm for NER.")
        except Exception as e:
            logger.warning(f"Failed to load spaCy model for NER (Neo4j). Entities will be skipped: {e}")
            self.nlp = None

    def close(self):
        """Closes the driver connection."""
        if self.driver:
            self.driver.close()

    def build_graph(self, chunks: List[Dict[str, Any]]) -> None:
        """
        Creates or updates the (:Document)-[:HAS_CHUNK]->(:Chunk) relationships
        for a batch of chunks.

        Args:
            chunks: A list of chunk dictionaries from SemanticChunker.
        """
        if not chunks:
            logger.warning("No chunks provided to build_graph. Skipping.")
            return

        if not self.driver:
            logger.error("Neo4j driver not initialized. Cannot build graph.")
            return

        with self.driver.session() as session:
            # First, process text with spaCy to extract entities
            processed_chunks = []
            for chunk in chunks:
                chunk_copy = dict(chunk)
                entities = []
                if self.nlp and chunk_copy.get("chunk_text"):
                    doc = self.nlp(chunk_copy["chunk_text"])
                    # Valid Neo4j property types must be primitives (no nested quotes)
                    entities = [{"name": ent.text.strip(), "type": ent.label_} for ent in doc.ents if ent.label_ in {"PERSON", "ORG", "GPE", "LOC", "PRODUCT", "EVENT"}]
                    
                    # Deduplicate and Normalize entities for the same chunk
                    seen = set()
                    unique_entities = []
                    
                    # Basic Synonym Map for Entity Resolution
                    synonym_map = {
                        "aws": "Amazon Web Services",
                        "amazon web services": "Amazon Web Services",
                        "gcp": "Google Cloud Platform",
                        "google cloud": "Google Cloud Platform"
                    }
                    
                    for e in entities:
                        # Normalize name
                        raw_name = e["name"].strip()
                        normalized_name = raw_name.lower()
                        
                        # Merge Synonyms
                        if normalized_name in synonym_map:
                            e["name"] = synonym_map[normalized_name]
                            normalized_name = e["name"].lower()
                        
                        if normalized_name not in seen and len(normalized_name) > 1:
                            seen.add(normalized_name)
                            unique_entities.append(e)
                    entities = unique_entities
                    
                chunk_copy["entities"] = entities
                processed_chunks.append(chunk_copy)
                
            # We process chunks in a single transaction for efficiency and atomicity.
            session.execute_write(self._merge_chunks_tx, processed_chunks)
            logger.info(f"Successfully ingested {len(processed_chunks)} chunks with entities into Knowledge Graph.")

    @staticmethod
    def _merge_chunks_tx(tx, chunks: List[Dict[str, Any]]):
        """
        Transaction function to execute the parameterized Cypher query.
        """
        cypher_query = """
        UNWIND $chunks AS chunk_data
        
        // Extract metadata
        WITH chunk_data, chunk_data.chunk_metadata AS meta
        
        // 1. Idempotent MERGE for Document Node
        MERGE (d:Document {id: meta.doc_id})
        ON CREATE SET 
            d.title = meta.title,
            d.section = meta.section,
            d.created_at = datetime()
        ON MATCH SET 
            d.title = meta.title,
            d.section = meta.section,
            d.updated_at = datetime()
            
        // 2. Idempotent MERGE for Chunk Node
        // We use doc_id + sequence_index as a unique identifier for the chunk
        MERGE (c:Chunk {doc_id: meta.doc_id, index: meta.sequence_index})
        ON CREATE SET 
            c.chunk_text = chunk_data.chunk_text,
            c.tokens = chunk_data.token_count,
            c.created_at = datetime()
        ON MATCH SET
            c.chunk_text = chunk_data.chunk_text,
            c.tokens = chunk_data.token_count,
            c.updated_at = datetime()
            
        // 3. Create or MERGE Relationship
        MERGE (d)-[r:HAS_CHUNK {sequence_index: meta.sequence_index}]->(c)
        
        // 4. Extract Entities
        WITH c, chunk_data.entities AS entities
        UNWIND (CASE WHEN entities IS NULL THEN [] ELSE entities END) AS ent
        
        // 5. MERGE Entity Nodes and Relationships
        MERGE (e:Entity {name: ent.name})
        ON CREATE SET e.type = ent.type
        MERGE (c)-[:MENTIONS]->(e)
        """
        
        # We need to make sure sequence_index is assigned
        # Assuming the pipeline calling KGBuilder will append sequence_index,
        # but if not, we should generate it here based on the list order.
        processed_chunks = []
        for idx, chunk in enumerate(chunks):
            chunk_copy = chunk.copy()
            meta = chunk_copy.get("chunk_metadata", {}).copy()
            
            # Ensure sequence_index exists in metadata
            if "sequence_index" not in meta:
                meta["sequence_index"] = idx
                
            chunk_copy["chunk_metadata"] = meta
            
            processed_chunks.append({
                "chunk_text": chunk_copy.get("chunk_text", ""),
                "token_count": chunk_copy.get("token_count", 0),
                "entities": chunk_copy.get("entities", []),
                "chunk_metadata": {
                    "doc_id": meta.get("doc_id", "Unknown"),
                    "title": meta.get("title", ""),
                    "section": meta.get("section", ""),
                    "sequence_index": meta.get("sequence_index", idx)
                }
            })

        tx.run(cypher_query, chunks=processed_chunks)
