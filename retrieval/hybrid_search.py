"""
Hybrid Search Coordinator for NovaSearch.

This module orchestrates:
1. Dense Retrieval (via vector_search.py)
2. Sparse Retrieval (via keyword_search.py)
3. Graph Retrieval (Neo4j Contextual Expansion)
4. Reranking (via rrf_fusion.py)
"""

import os
import logging
from typing import List, Dict, Any

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import spacy
import psycopg2
from neo4j import GraphDatabase

from retrieval.dense.vector_search import PGVectorDenseRetriever
from retrieval.dense.faiss_search import FAISSDenseRetriever
from retrieval.sparse.keyword_search import PostgresFTSSparseRetriever
from retrieval.sparse.elastic_search import ElasticSparseRetriever
from retrieval.reranker.rrf_fusion import reciprocal_rank_fusion
from retrieval.reranker.cross_encoder import CrossEncoderReranker
from retrieval.reranker.colbert_reranker import ColBERTReranker

logger = logging.getLogger(__name__)

class HybridSearchCoordinator:
    """
    Orchestrates dense, sparse, and graph search strategies, fusing the results
    to provide highly relevant, context-grounded chunks.
    """

    def __init__(
        self,
        embedding_model_name: str = "all-MiniLM-L6-v2",
        pg_dsn: str | None = None,
        neo4j_uri: str | None = None,
        neo4j_user: str | None = None,
        neo4j_password: str | None = None
    ):
        """
        Initializes the HybridSearchCoordinator with connections and retrievers.
        """
        # 1. Setup PostgreSQL (PGVector & FTS)
        self.pg_dsn = pg_dsn or os.getenv("DATABASE_URL", 
            f"dbname={os.getenv('POSTGRES_DB', 'novasearch')} "
            f"user={os.getenv('POSTGRES_USER', 'postgres')} "
            f"password={os.getenv('POSTGRES_PASSWORD', 'postgres_secure_password')} "
            f"host={os.getenv('POSTGRES_HOST', 'localhost')} "
            f"port={os.getenv('POSTGRES_PORT', '5432')}"
        )
        try:
            self.pg_conn = psycopg2.connect(self.pg_dsn)
            logger.info("Connected to PostgreSQL for Dense/Sparse search.")
        except Exception as e:
            logger.error("Failed to connect to PostgreSQL: %s", str(e))
            self.pg_conn = None

        # 2. Setup Retrievers & Reranker
        dense_backend = os.getenv("DENSE_BACKEND", os.getenv("VECTOR_STORE_TYPE", "pgvector")).lower()
        if dense_backend == "faiss":
            logger.info("Initializing FAISS local vector store...")
            self.dense_retriever = FAISSDenseRetriever(embedding_model_name, index_dir="faiss_index")
        else:
            logger.info("Initializing PGVector remote vector store...")
            self.dense_retriever = PGVectorDenseRetriever(self.pg_conn, embedding_model_name)
            
        sparse_backend = os.getenv("SPARSE_BACKEND", os.getenv("SPARSE_STORE_TYPE", "postgres")).lower()
        if sparse_backend in ["elastic", "elasticsearch"]:
            logger.info("Initializing Elasticsearch sparse store...")
            self.sparse_retriever = ElasticSparseRetriever()
        else:
            logger.info("Initializing Postgres FTS sparse store...")
            self.sparse_retriever = PostgresFTSSparseRetriever(self.pg_conn)
            
        reranker_type = os.getenv("RERANKER_TYPE", "crossencoder").lower()
        if reranker_type == "colbert":
            logger.info("Initializing ColBERT Reranker...")
            self.cross_encoder = ColBERTReranker()
        else:
            logger.info("Initializing CrossEncoder Reranker...")
            self.cross_encoder = CrossEncoderReranker()

        # 3. Setup Neo4j (Graph Retrieval)
        self.neo4j_uri = neo4j_uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.neo4j_user = neo4j_user or os.getenv("NEO4J_USER", "neo4j")
        self.neo4j_password = neo4j_password or os.getenv("NEO4J_PASSWORD", "neo4j_secure_password")
        
        try:
            self.neo4j_driver = GraphDatabase.driver(
                self.neo4j_uri, auth=(self.neo4j_user, self.neo4j_password)
            )
            self.neo4j_driver.verify_connectivity()
            logger.info("Connected to Neo4j Knowledge Graph.")
        except Exception as e:
            logger.error("Failed to connect to Neo4j: %s", str(e))
            self.neo4j_driver = None
            
        # 4. Setup NER for Graph Expansion
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("Spacy model not found. Downloading en_core_web_sm...")
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], check=True)
            self.nlp = spacy.load("en_core_web_sm")

    def __del__(self):
        """Cleanup connections on destruction."""
        if self.pg_conn and not self.pg_conn.closed:
            self.pg_conn.close()
        if hasattr(self, 'neo4j_driver') and self.neo4j_driver:
            self.neo4j_driver.close()

    def _extract_entities(self, query: str) -> List[str]:
        """Extracts Named Entities (ORG, PERSON, DATE, etc.) using spaCy."""
        if not hasattr(self, 'nlp'):
            return []
        doc = self.nlp(query)
        # Filter for meaningful entities
        entities = [ent.text for ent in doc.ents if ent.label_ in ["ORG", "PERSON", "GPE", "PRODUCT", "LAW"]]
        return list(set(entities))

    def _graph_expansion(self, base_hits: List[Dict[str, Any]], query: str = "") -> List[Dict[str, Any]]:
        """
        Takes the top hits from Dense/Sparse, queries Neo4j for their parent
        document and adjacent chunks (1-hop), and injects this graph context.
        Also attempts to resolve Entities (NER) if mentioned in the query.
        """
        if not hasattr(self, 'neo4j_driver') or not self.neo4j_driver or not base_hits:
            return base_hits
            
        enriched_hits = []
        
        # Extract Entities for potential Graph Pathing
        entities = self._extract_entities(query)
        if entities:
            logger.info("NER detected entities for Graph Pathing: %s", entities)
            # The cypher handles entity matching efficiently directly parameterized
        
        with self.neo4j_driver.session() as session:
            for hit in base_hits:
                doc_id = hit["doc_id"]
                idx = hit["chunk_index"]
                
                cypher = """
                MATCH (d:Document {id: $doc_id})-[:HAS_CHUNK]->(c:Chunk {index: $idx})
                OPTIONAL MATCH (d)-[:HAS_CHUNK]->(prev:Chunk {index: $idx - 1})
                OPTIONAL MATCH (d)-[:HAS_CHUNK]->(next:Chunk {index: $idx + 1})
                
                // 2-Hop Entity Traversal: Find entities mentioned in this chunk, 
                // and then find OTHER chunks (in other docs) that mention the exact same entities.
                OPTIONAL MATCH (c)-[:MENTIONS]->(e:Entity)<-[:MENTIONS]-(cross_c:Chunk)<-[:HAS_CHUNK]-(cross_d:Document)
                WHERE e.name IN $entities AND cross_c.id <> c.id AND cross_d.id <> d.id
                
                RETURN 
                    d.title AS doc_title,
                    d.section AS doc_section,
                    c.chunk_text AS exact_hit_text,
                    prev.chunk_text AS prev_context,
                    next.chunk_text AS next_context,
                    collect(DISTINCT e.name) AS shared_entities,
                    collect(DISTINCT cross_d.title + ": " + cross_c.chunk_text)[0..3] AS cross_document_texts
                """
                
                try:
                    result = session.run(cypher, doc_id=doc_id, idx=idx, entities=entities).single()
                    if result:
                        graph_context = {
                            "doc_title": result["doc_title"],
                            "doc_section": result["doc_section"],
                            "prev_context": result["prev_context"],
                            "next_context": result["next_context"],
                            "shared_entities": result["shared_entities"],
                            "cross_document_texts": result["cross_document_texts"]
                        }
                        enriched_hit = {**hit, "graph_context": graph_context}
                        enriched_hits.append(enriched_hit)
                    else:
                        enriched_hits.append({**hit, "graph_context": None})
                except Exception as e:
                    logger.error("Graph Expansion error for Doc %s, Idx %s: %s", doc_id, idx, str(e))
                    enriched_hits.append({**hit, "graph_context": None})
                    
        return enriched_hits

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Main orchestration method for Tri-Engine Fusion (Now with Cross-Encoder).
        """
        logger.info("Initiating Hybrid Search for query: '%s'", query)
        
        # 1. Base Retrieval (Cast a wide net)
        fetch_k = max(20, top_k * 4) 
        dense_hits = self.dense_retriever.search(query, top_k=fetch_k)
        sparse_hits = self.sparse_retriever.search(query, top_k=fetch_k)
        
        # 2. First Stage Fusion (RRF)
        fused_hits = reciprocal_rank_fusion(dense_hits, sparse_hits)
        
        # 3. Second Stage Reranking (Cross-Encoder)
        reranked_hits = self.cross_encoder.rerank(query, fused_hits, top_k=top_k)
        
        # 4. Knowledge Graph Expansion (Now with NER capability)
        final_grounded_results = self._graph_expansion(reranked_hits, query=query)
        
        logger.info("Hybrid Search Complete. Yielding %s results.", len(final_grounded_results))
        return final_grounded_results
