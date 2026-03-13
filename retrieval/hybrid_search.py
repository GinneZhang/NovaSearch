"""
Hybrid Search Coordinator for NovaSearch.

This module orchestrates:
1. Dense Retrieval (via vector_search.py)
2. Sparse Retrieval (via keyword_search.py)
3. Graph Retrieval (Neo4j Contextual Expansion)
4. Reranking (via rrf_fusion.py)
"""

import os
import uuid
import json
import logging
import re
from typing import List, Dict, Any, Optional

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
from retrieval.reranker.monot5_reranker import MonoT5Reranker
from retrieval.graph.cypher_generator import CypherGenerator

logger = logging.getLogger(__name__)

MULTI_HOP_CUES = (
    "wife", "husband", "spouse", "father", "mother", "daughter", "son",
    "founder", "author", "director", "capital", "born", "birth", "located",
    "president", "leader", "member"
)

FOLLOW_UP_RELATION_TERMS = (
    "wife", "husband", "spouse", "born", "birthplace", "capital",
    "author", "director", "president", "leader", "member"
)

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
        elif reranker_type == "monot5":
            logger.info("Initializing MonoT5 Reranker...")
            self.cross_encoder = MonoT5Reranker()
            # Graceful fallback check
            if not getattr(self.cross_encoder, "model", None):
                logger.warning("MonoT5 initialization failed. Falling back to CrossEncoder.")
                self.cross_encoder = CrossEncoderReranker()
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
            logger.warning("Spacy model not found. Continuing without spaCy NER.")
            self.nlp = None
            
        # 5. Setup Dynamic Cypher Generator
        self.cypher_gen = CypherGenerator()

    def __del__(self):
        """Cleanup connections on destruction."""
        if self.pg_conn and not self.pg_conn.closed:
            self.pg_conn.close()
        if hasattr(self, 'neo4j_driver') and self.neo4j_driver:
            self.neo4j_driver.close()

    def _extract_entities(self, query: str) -> List[str]:
        """Extracts Named Entities (ORG, PERSON, DATE, etc.) using spaCy."""
        if not hasattr(self, 'nlp') or not self.nlp:
            return []
        doc = self.nlp(query)
        # Filter for meaningful entities
        entities = [ent.text for ent in doc.ents if ent.label_ in ["ORG", "PERSON", "GPE", "PRODUCT", "LAW"]]
        return list(set(entities))

    @staticmethod
    def _normalize_terms(text: str) -> List[str]:
        return [token for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9_-]+", (text or "").lower()) if len(token) > 2]

    def _title_overlap(self, query: str, title: str) -> int:
        query_terms = set(self._normalize_terms(query))
        title_terms = set(self._normalize_terms(title))
        return len(query_terms & title_terms)

    def _is_multi_hop_query(self, query: str) -> bool:
        lowered = query.lower()
        return any(cue in lowered for cue in MULTI_HOP_CUES) and " of " in lowered

    def _build_query_variants(self, query: str) -> List[str]:
        variants: List[str] = [query.strip()]
        entities = self._extract_entities(query)

        if entities:
            variants.extend(entity for entity in entities if entity and entity not in variants)
            combined_entities = " ".join(entities[:2]).strip()
            if combined_entities and combined_entities not in variants:
                variants.append(combined_entities)

        lowered = query.lower()
        cleaned = re.sub(r"\b(who|what|when|where|which|is|was|were|the|a|an)\b", " ", lowered, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        if cleaned and cleaned not in variants:
            variants.append(cleaned)

        return variants[:4]

    def _extract_bridge_entities(self, query: str, hits: List[Dict[str, Any]]) -> List[str]:
        query_terms = set(self._normalize_terms(query))
        bridges: List[str] = []

        for hit in hits[:5]:
            candidates: List[str] = []
            title = hit.get("title") or ""
            if title:
                candidates.append(title)

            if hasattr(self, "nlp") and self.nlp:
                text = f"{title}\n{hit.get('chunk_text', '')}"
                try:
                    doc = self.nlp(text)
                    candidates.extend(
                        ent.text.strip() for ent in doc.ents
                        if ent.label_ in {"PERSON", "ORG", "GPE", "LOC", "PRODUCT", "EVENT"}
                    )
                except Exception:
                    pass
            else:
                text = hit.get("chunk_text", "")
                candidates.extend(re.findall(r"(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})", text))

            for candidate in candidates:
                candidate = candidate.strip(" .,:;()[]{}\"'")
                if not candidate:
                    continue
                candidate_terms = set(self._normalize_terms(candidate))
                if not candidate_terms:
                    continue
                if candidate_terms.issubset(query_terms):
                    continue
                if len(candidate_terms) == 1 and next(iter(candidate_terms)) in {"who", "what", "when", "where"}:
                    continue
                if candidate not in bridges:
                    bridges.append(candidate)

        return bridges[:6]

    def _follow_up_queries(self, query: str, hits: List[Dict[str, Any]]) -> List[str]:
        if not self._is_multi_hop_query(query):
            return []

        lowered = query.lower()
        relation_terms = [term for term in FOLLOW_UP_RELATION_TERMS if term in lowered]
        if not relation_terms:
            relation_terms = ["related"]

        bridges = self._extract_bridge_entities(query, hits)
        follow_ups: List[str] = []

        for bridge in bridges:
            for relation in relation_terms[:2]:
                if relation == "related":
                    candidate = bridge
                else:
                    candidate = f"{relation} {bridge}"
                if candidate.lower() != query.lower() and candidate not in follow_ups:
                    follow_ups.append(candidate)

        return follow_ups[:6]

    def _collect_candidates(self, query_variants: List[str], fetch_k: int) -> List[Dict[str, Any]]:
        dense_hits: List[Dict[str, Any]] = []
        sparse_hits: List[Dict[str, Any]] = []

        per_variant_k = max(8, fetch_k // max(1, len(query_variants)))
        for variant in query_variants:
            for hit in self.dense_retriever.search(variant, top_k=per_variant_k):
                hit["query_variant"] = variant
                dense_hits.append(hit)
            for hit in self.sparse_retriever.search(variant, top_k=per_variant_k):
                hit["query_variant"] = variant
                sparse_hits.append(hit)

        fused_hits = reciprocal_rank_fusion(dense_hits, sparse_hits)

        for hit in fused_hits:
            title = hit.get("title") or ""
            if title:
                hit["rrf_score"] = hit.get("rrf_score", 0.0) + (0.04 * self._title_overlap(query_variants[0], title))

        fused_hits.sort(key=lambda x: x.get("rrf_score", 0.0), reverse=True)
        return fused_hits[:fetch_k]

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

    def _deep_graph_search(self, query: str, query_graph: Optional[List[Dict[str, str]]] = None) -> List[Dict[str, Any]]:
        """
        Attempts to generate and execute a dynamic Cypher query for multi-hop 
        relationship reasoning. Uses self-healing: if Cypher fails, the error
        is fed back to the LLM for repair.
        """
        if not self.neo4j_driver:
            return []
            
        # Enhance prompt with query graph if available
        context_query = query
        if query_graph:
            graph_json = json.dumps(query_graph)
            context_query = f"{query}\n[Semantic Context]: {graph_json}"
        
        # Define the executor function for self-healing validation
        _captured_records = []
        
        def _execute_cypher(cypher_str: str):
            """Validates Cypher by executing it. Raises on failure."""
            with self.neo4j_driver.session() as session:
                result = session.run(cypher_str)
                _captured_records.clear()
                for record in result:
                    _captured_records.append(record.data())
            
        # Use self-healing generation
        cypher = self.cypher_gen.generate_with_healing(context_query, executor_fn=_execute_cypher)
        if not cypher:
            return []
            
        logger.info(f"Validated Dynamic Cypher: {cypher}")
        graph_hits = []
        
        # Use the already-captured records from the healing validation
        if _captured_records:
            for data in _captured_records:
                text_content = " | ".join([str(v) for v in data.values()])
                graph_hits.append({
                    "id": f"graph_{uuid.uuid4().hex[:8]}",
                    "doc_id": "Knowledge Graph",
                    "chunk_index": -1,
                    "chunk_text": f"[Symbolic Reasoning]: {text_content}",
                    "score": 1.0,
                    "source": "dynamic_cypher",
                    "graph_context": {"doc_title": "Neo4j Symbolic Path", "doc_section": "Multi-hop reasoning"}
                })
        else:
            # Fallback: re-execute valid Cypher if records weren't captured
            try:
                with self.neo4j_driver.session() as session:
                    records = session.run(cypher)
                    for record in records:
                        data = record.data()
                        text_content = " | ".join([str(v) for v in data.values()])
                        graph_hits.append({
                            "id": f"graph_{uuid.uuid4().hex[:8]}",
                            "doc_id": "Knowledge Graph",
                            "chunk_index": -1,
                            "chunk_text": f"[Symbolic Reasoning]: {text_content}",
                            "score": 1.0,
                            "source": "dynamic_cypher",
                            "graph_context": {"doc_title": "Neo4j Symbolic Path", "doc_section": "Multi-hop reasoning"}
                        })
            except Exception as e:
                logger.error(f"Dynamic Cypher re-execution failed: {e}")
                return []
        
        if graph_hits:
            logger.info(f"Dynamic Cypher returned {len(graph_hits)} symbolic paths.")
        return graph_hits

    def search(self, query: str, top_k: int = 5, query_graph: Optional[List[Dict[str, str]]] = None) -> List[Dict[str, Any]]:
        """
        Main orchestration method for Tri-Engine Fusion (Now with Cross-Encoder).
        """
        logger.info("Initiating Hybrid Search for query: '%s'", query)
        
        # 1. Base Retrieval with query variants
        fetch_k = max(40, top_k * 8)
        base_variants = self._build_query_variants(query)
        fused_hits = self._collect_candidates(base_variants, fetch_k=fetch_k)

        # 2. Second-hop query generation for relational questions
        follow_ups = self._follow_up_queries(query, fused_hits[:10])
        if follow_ups:
            logger.info("Generated %d follow-up queries for multi-hop retrieval: %s", len(follow_ups), follow_ups)
            follow_up_hits = self._collect_candidates(follow_ups, fetch_k=fetch_k)
            merged_pool = fused_hits + follow_up_hits
        else:
            merged_pool = fused_hits

        deduped_pool: Dict[tuple, Dict[str, Any]] = {}
        for hit in merged_pool:
            key = (hit.get("doc_id"), hit.get("chunk_index"))
            if key not in deduped_pool or hit.get("rrf_score", hit.get("score", 0.0)) > deduped_pool[key].get("rrf_score", deduped_pool[key].get("score", 0.0)):
                deduped_pool[key] = hit

        candidate_pool = sorted(
            deduped_pool.values(),
            key=lambda x: x.get("rrf_score", x.get("score", 0.0)),
            reverse=True
        )[:fetch_k]

        # 3. Second Stage Reranking (Cross-Encoder)
        reranked_hits = self.cross_encoder.rerank(query, candidate_pool, top_k=fetch_k)
        
        # 4. Knowledge Graph Expansion (Now with NER capability)
        grounded_candidates = self._graph_expansion(reranked_hits[: max(top_k * 3, 10)], query=query)
        
        # 5. Deep Symbolic Reasoning (Dynamic Cypher)
        deep_hits = self._deep_graph_search(query, query_graph=query_graph)
        if deep_hits:
            logger.info(f"Injecting {len(deep_hits)} symbolic paths from Neo4j.")
            # Prepend deep hits as they are often very precise for relationship queries
            grounded_candidates = deep_hits + grounded_candidates

        final_grounded_results = grounded_candidates[:top_k]
        
        logger.info("Hybrid Search Complete. Yielding %s results.", len(final_grounded_results))
        return final_grounded_results
