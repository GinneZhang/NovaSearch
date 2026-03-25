"""
Main FastAPI entry point for NovaSearch.
"""

import os
import uuid
import logging
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form
from typing import Optional
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import json
import math

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import psycopg2
from psycopg2.extras import Json

from core.db_init import initialize_databases
from api.schemas import QueryRequest, QueryResponse, DocumentUploadRequest, SourceChunk
from agent.copilot_agent import EnterpriseCopilotAgent
from core.auth import get_api_key
from ingestion.chunking.semantic_chunker import SemanticChunker
from ingestion.chunking.sliding_window import SlidingWindowChunker
from ingestion.graph_build.kg_builder import KGBuilder
from ingestion.parsers.multimodal_parser import MultimodalParser
from retrieval.dense.vision_search import PGVectorVisionRetriever
from PIL import Image
import io

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def _safe_numeric(value, fallback: float = 0.0) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return fallback
    if math.isnan(numeric) or math.isinf(numeric):
        return fallback
    return numeric

# Lifespan context manager for startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Running lifespan startup...")
    # Initialize DBs on startup
    initialize_databases()
    yield

app = FastAPI(
    title="NovaSearch API",
    description="Enterprise Copilot & Intelligent Retrieval Engine",
    version="1.0.0",
    lifespan=lifespan
)

# Secure CORS config
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"], # Specific origins for production-readiness
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Initialize Copilot Agent (Singleton instance for the app lifecycle)
try:
    copilot_agent = EnterpriseCopilotAgent()
    
    chunk_strategy = os.getenv("CHUNKING_STRATEGY", "semantic").lower()
    if chunk_strategy == "sliding_window":
        chunker = SlidingWindowChunker()
        logger.info("Using SlidingWindowChunker")
    else:
        chunker = SemanticChunker()
        logger.info("Using SemanticChunker")
        
    kg_builder = KGBuilder()
    multimodal_parser = MultimodalParser()
    enable_vision_retriever = os.getenv("ENABLE_VISION_RETRIEVER", "true").lower() in {"1", "true", "yes"}
    vision_retriever = PGVectorVisionRetriever() if enable_vision_retriever else None
except Exception as e:
    logger.error("Failed to initialize tools: %s", str(e))
    copilot_agent = None
    chunker = None
    kg_builder = None
    multimodal_parser = None
    vision_retriever = None


@app.get("/health")
async def health_check():
    """Healthcheck endpoint."""
    return {"status": "ok", "service": "NovaSearch API"}


@app.post("/ask", dependencies=[Depends(get_api_key)])
def ask_copilot(request: QueryRequest):
    """
    Main endpoint for the Enterprise Copilot.
    Takes a natural language query, performs Tri-Engine Retrieval,
    and returns a grounded, hallucination-free response via a StreamingResponse.
    """
    global copilot_agent
    if not copilot_agent:
        try:
            copilot_agent = EnterpriseCopilotAgent()
        except Exception as e:
            logger.error(f"Error details: {str(e)}")
            raise HTTPException(status_code=503, detail="Copilot Agent is not initialized due to missing configurations.")

    session_id = request.session_id or str(uuid.uuid4())
    logger.info("Handling /ask request for query: '%s', session: '%s'", request.query, session_id)

    def serialize_sources(raw_sources):
        sources_schema = []
        for s in raw_sources or []:
            sources_schema.append({
                "doc_id": s.get("doc_id", "unknown"),
                "title": s.get("title") or s.get("graph_context", {}).get("doc_title"),
                "chunk_text": s.get("chunk_text", ""),
                "score": _safe_numeric(
                    s.get("final_rank_score"),
                    _safe_numeric(s.get("cross_encoder_score"), _safe_numeric(s.get("score", 0.0)))
                ),
                "graph_context": s.get("graph_context")
            })
        return sources_schema

    def generate_stream():
        yield json.dumps({
            "type": "thought", 
            "content": "Analyzing query and establishing Tri-Engine retrieval context...", 
            "session_id": session_id
        }) + "\n"
        
        try:
            for chunk in copilot_agent.generate_response( 
                query=request.query, 
                session_id=session_id, 
                top_k=request.top_k
            ):
                if chunk.get("type") == "answer_metadata":
                    yield json.dumps({
                        "type": "answer_metadata",
                        "sources": serialize_sources(chunk.get("sources", [])),
                        "retrieval_contexts": serialize_sources(chunk.get("retrieval_contexts", [])),
                        "generation_contexts": serialize_sources(chunk.get("generation_contexts", [])),
                        "debug_metrics": chunk.get("debug_metrics"),
                        "session_id": chunk.get("session_id"),
                        "consistency_score": chunk.get("consistency_score"),
                        "hallucination_warning": chunk.get("hallucination_warning")
                    }) + "\n"
                else:
                    yield json.dumps(chunk) + "\n"
        except Exception as e:
            logger.error("Error processing /ask request: %s", str(e))
            yield json.dumps({
                "type": "error", 
                "content": f"Internal Server Error: {str(e)}", 
                "session_id": session_id
            }) + "\n"

    return StreamingResponse(generate_stream(), media_type="application/x-ndjson")


@app.post("/ask_vision", dependencies=[Depends(get_api_key)])
async def ask_vision(
    file: UploadFile = File(...),
    top_k: int = Form(5)
):
    """
    Multimodal endpoints that takes an image upload and searches the common CLIP vector space.
    """
    global vision_retriever
    if not vision_retriever:
        try:
            vision_retriever = PGVectorVisionRetriever()
        except Exception as e:
            logger.error(f"Failed to initialize Vision Retriever: {e}")
            raise HTTPException(status_code=503, detail="Vision services unavailable.")
            
    try:
        if not file.content_type or "image" not in file.content_type.lower():
            raise HTTPException(status_code=400, detail="Uploaded file must be an image.")
            
        file_bytes = await file.read()
        pic = Image.open(io.BytesIO(file_bytes))
        
        logger.info(f"Processing image query...")
        query_embedding = vision_retriever.embed(pic)
        
        # Search the multimodal vector space
        hits = vision_retriever.search(query_embedding, top_k=top_k)
        
        return {
            "status": "success",
            "query_type": "image",
            "results": hits
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Vision search failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Vision search failed: {str(e)}")


def _insert_chunks_to_postgres(doc_id: str, title: str, section: str, chunks: list):
    """Helper to insert generated chunks and dense vectors into Postgres."""
    pg_dsn = os.getenv("DATABASE_URL", 
        f"dbname={os.getenv('POSTGRES_DB', 'novasearch')} "
        f"user={os.getenv('POSTGRES_USER', 'postgres')} "
        f"password={os.getenv('POSTGRES_PASSWORD', 'postgres_secure_password')} "
        f"host={os.getenv('POSTGRES_HOST', 'localhost')} "
        f"port={os.getenv('POSTGRES_PORT', '5432')}"
    )
    conn = psycopg2.connect(pg_dsn)
    try:
        with conn.cursor() as cur:
            # 1. Insert Document metadata
            metadata = Json({"title": title, "section": section})
            cur.execute("""
                INSERT INTO documents (id, title, metadata) 
                VALUES (%s, %s, %s) ON CONFLICT (id) DO NOTHING;
            """, (doc_id, title, metadata))
            
            # 2. Insert Chunks and their embeddings
            for chunk in chunks:
                meta = chunk["chunk_metadata"]
                seq_idx = meta.get("sequence_index", 0)
                chunk_id = f"{doc_id}_{seq_idx}"
                text = chunk["chunk_text"]
                
                # Fetch embedding from semantic chunker's loaded model
                emb = chunker.encode_text(text).tolist()
                emb_str = "[" + ",".join([str(x) for x in emb]) + "]"
                
                cur.execute("""
                    INSERT INTO chunks (id, doc_id, index, chunk_text, embedding) 
                    VALUES (%s, %s, %s, %s, %s::vector)
                    ON CONFLICT (id) DO UPDATE SET 
                        chunk_text = EXCLUDED.chunk_text,
                        embedding = EXCLUDED.embedding;
                """, (chunk_id, doc_id, seq_idx, text, emb_str))
        conn.commit()
    finally:
        conn.close()


@app.post("/ingest", dependencies=[Depends(get_api_key)])
def ingest_document(
    text_input: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
    title: Optional[str] = Form("Untitled Document"),
    section: Optional[str] = Form("General")
):
    """
    Ingests a new document into the system.
    Runs text through SemanticChunker, saves to PGVector, and builds Knowledge Graph.
    Uses standard `def` (instead of `async def`) to safely offload heavy synchronous 
    CPU bounds (SentenceTransformers) and blocking DB I/O to FastAPI's threadpool.
    """
    global chunker, kg_builder, multimodal_parser, vision_retriever
    if not chunker or not kg_builder or not multimodal_parser:
        try:
            chunk_strategy = os.getenv("CHUNKING_STRATEGY", "semantic").lower()
            if not chunker:
                if chunk_strategy == "sliding_window":
                    chunker = SlidingWindowChunker()
                else:
                    chunker = SemanticChunker()
            if not kg_builder:
                kg_builder = KGBuilder()
            if not multimodal_parser:
                multimodal_parser = MultimodalParser()
            if not vision_retriever:
                vision_retriever = PGVectorVisionRetriever()
        except Exception as e:
            logger.error(f"Error details: {str(e)}")
            raise HTTPException(status_code=503, detail="Ingestion services unavailable.")
        
    doc_id = str(uuid.uuid4())
    metadata = {
        "doc_id": doc_id,
        "title": title or "Untitled Document",
        "section": section or "General"
    }
    
    
    try:
        document_text = ""
        # Extract Text from Uploaded File if provided
        file_bytes = None
        if file:
            logger.info("Parsing uploaded file: %s (MIME: %s)", title, file.content_type)
            file_bytes = file.file.read()
            parsed_text = multimodal_parser.parse(file_bytes, file.content_type)
            if parsed_text:
                document_text += parsed_text
                
        if text_input:
            if document_text:
                document_text += "\n\n" + text_input
            else:
                document_text += text_input
                
        document_text = document_text.strip()
        if not document_text:
            raise ValueError("No document text or file provided.")
                
        # 1. Chunk Document (Heavy CPU)
        logger.info("Chunking document: %s", title)
        try:
            chunks = chunker.chunk_document(document_text, metadata)
        except Exception as chunk_err:
            raise RuntimeError(f"Chunking stage failed: {chunk_err}") from chunk_err
        
        # Ensure sequence_index is present for graph and db
        for idx, c in enumerate(chunks):
            c["chunk_metadata"]["sequence_index"] = idx

        # Dual-Write Consistency Block
        try:
            # 2. Persist to Dense Backend
            dense_backend = os.getenv("DENSE_BACKEND", "pgvector").lower()
            if dense_backend == "faiss":
                logger.info("Persisting %d chunks to FAISS...", len(chunks))
                from retrieval.dense.faiss_search import FAISSDenseRetriever
                emb_model_name = chunker.embedding_model_name if hasattr(chunker, "embedding_model_name") else "all-MiniLM-L6-v2"
                faiss_retriever = FAISSDenseRetriever(emb_model_name)
                try:
                    faiss_retriever.add_documents(doc_id, chunks)
                except Exception as faiss_err:
                    raise RuntimeError(f"FAISS persistence failed: {faiss_err}") from faiss_err
            else:
                logger.info("Persisting %d chunks to PostgreSQL...", len(chunks))
                try:
                    _insert_chunks_to_postgres(doc_id, title or "Untitled Document", section or "General", chunks)
                except Exception as pg_err:
                    raise RuntimeError(f"PostgreSQL persistence failed: {pg_err}") from pg_err

            # Persist to Sparse Backend
            sparse_backend = os.getenv("SPARSE_BACKEND", "postgres").lower()
            if sparse_backend in ["elastic", "elasticsearch"]:
                logger.info("Persisting %d chunks to Elasticsearch...", len(chunks))
                from retrieval.sparse.elastic_search import ElasticSparseRetriever
                elastic_retriever = ElasticSparseRetriever()
                try:
                    elastic_retriever.add_documents(doc_id, chunks)
                except Exception as sparse_err:
                    raise RuntimeError(f"Elasticsearch persistence failed: {sparse_err}") from sparse_err

            # 3. Persist to Neo4j (Knowledge Graph Construction)
            benchmark_mode = os.getenv("NOVASEARCH_BENCHMARK_MODE", "false").lower() in {"1", "true", "yes"}
            enable_graph_ingestion = os.getenv(
                "ENABLE_GRAPH_INGESTION",
                "false" if benchmark_mode else "true"
            ).lower() in {"1", "true", "yes"}
            if enable_graph_ingestion:
                logger.info("Building Knowledge Graph for document...")
                try:
                    kg_builder.build_graph(chunks)
                except Exception as graph_err:
                    raise RuntimeError(f"Knowledge graph ingestion failed: {graph_err}") from graph_err
            
            # 4. Multimodal Vision Embeddings (CLIP)
            enable_text_vision_indexing = os.getenv("ENABLE_TEXT_VISION_INDEXING", "true").lower() in {"1", "true", "yes"}
            if vision_retriever:
                logger.info("Generating CLIP multimodal embeddings...")
                # If an image was uploaded alongside the ingest, embed the raw image once to represent the doc
                if file_bytes and file.content_type and "image" in file.content_type.lower():
                    try:
                        pic = Image.open(io.BytesIO(file_bytes))
                        v_emb = vision_retriever.embed(pic)
                        vision_retriever.insert_vision_chunk(doc_id, -1, "[Raw Image Document]", v_emb)
                    except Exception as ve:
                        logger.error(f"Failed to embed raw image: {ve}")
                
                # Also embed the isolated text chunks into the vision space
                if enable_text_vision_indexing:
                    for idx, c in enumerate(chunks):
                        text_context = c.get("chunk_text", "")
                        if text_context:
                            try:
                                v_emb = vision_retriever.embed(text_context)
                                vision_retriever.insert_vision_chunk(doc_id, idx, text_context, v_emb)
                            except Exception as vision_err:
                                raise RuntimeError(f"Vision persistence failed: {vision_err}") from vision_err
            
        except Exception as db_err:
            # Explicit rollback or CRITICAL alert
            logger.critical("CRITICAL DATA INCONSISTENCY: Dual-Write failure for Doc %s. Potential orphan records across PG/Neo4j/ES: %s", doc_id, str(db_err))
            raise HTTPException(status_code=500, detail=f"Database consistency failure: {str(db_err)}")

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Ingestion failed: %s", str(e))
        raise HTTPException(status_code=500, detail=f"Ingestion pipeline failed: {str(e)}")

    return {
        "status": "success",
        "message": f"Document '{title}' ingested successfully.",
        "details": f"Processed {len(chunks)} contextual chunks across PGVector Document schema and Neo4j."
    }
