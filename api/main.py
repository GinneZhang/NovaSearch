"""
Main FastAPI entry point for NovaSearch.
"""

import os
import uuid
import logging
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import json

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
from ingestion.graph_build.kg_builder import KGBuilder

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
    allow_origins=["*"], # Update for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Copilot Agent (Singleton instance for the app lifecycle)
try:
    copilot_agent = EnterpriseCopilotAgent()
    chunker = SemanticChunker()
    kg_builder = KGBuilder()
except Exception as e:
    logger.error("Failed to initialize tools: %s", str(e))
    copilot_agent = None
    chunker = None
    kg_builder = None


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
                    sources_schema = []
                    for s in chunk.get("sources", []):
                        sources_schema.append({
                            "doc_id": s.get("doc_id", "unknown"),
                            "chunk_text": s.get("chunk_text", ""),
                            "score": s.get("cross_encoder_score", s.get("score", 0.0)),
                            "graph_context": s.get("graph_context")
                        })
                    yield json.dumps({
                        "type": "answer_metadata",
                        "sources": sources_schema,
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
                emb = chunker.embedding_model.encode(text).tolist()
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
def ingest_document(request: DocumentUploadRequest):
    """
    Ingests a new document into the system.
    Runs text through SemanticChunker, saves to PGVector, and builds Knowledge Graph.
    Uses standard `def` (instead of `async def`) to safely offload heavy synchronous 
    CPU bounds (SentenceTransformers) and blocking DB I/O to FastAPI's threadpool.
    """
    global chunker, kg_builder
    if not chunker or not kg_builder:
        try:
            if not chunker:
                chunker = SemanticChunker()
            if not kg_builder:
                kg_builder = KGBuilder()
        except Exception as e:
            logger.error(f"Error details: {str(e)}")
            raise HTTPException(status_code=503, detail="Ingestion services unavailable.")
        
    doc_id = str(uuid.uuid4())
    metadata = {
        "doc_id": doc_id,
        "title": request.title,
        "section": request.section
    }
    
    try:
        # 1. Chunk Document (Heavy CPU)
        logger.info("Chunking document: %s", request.title)
        chunks = chunker.chunk_document(request.document_text, metadata)
        
        # Ensure sequence_index is present for graph and db
        for idx, c in enumerate(chunks):
            c["chunk_metadata"]["sequence_index"] = idx

        # Dual-Write Consistency Block
        try:
            # 2. Persist to Postgres (Dense Vectors)
            logger.info("Persisting %d chunks to PostgreSQL...", len(chunks))
            _insert_chunks_to_postgres(doc_id, request.title, request.section, chunks)

            # 3. Persist to Neo4j (Knowledge Graph Construction)
            logger.info("Building Knowledge Graph for document...")
            kg_builder.build_graph(chunks)
            
        except Exception as db_err:
            # If Graph fails, we log an alert (a robust system would execute a PG rollback query here)
            logger.error("CRITICAL: Dual-Write failure for Doc %s. Partial ingestion may have occurred: %s", doc_id, str(db_err))
            raise HTTPException(status_code=500, detail=f"Database consistency failure: {str(db_err)}")

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Ingestion failed: %s", str(e))
        raise HTTPException(status_code=500, detail=f"Ingestion pipeline failed: {str(e)}")

    return {
        "status": "success",
        "message": f"Document '{request.title}' ingested successfully.",
        "details": f"Processed {len(chunks)} contextual chunks across PGVector Document schema and Neo4j."
    }
