"""
Database Initialization Bootstrapper for AsterScope.
"""

import os
import logging

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import psycopg2
from neo4j import GraphDatabase

logger = logging.getLogger(__name__)

def init_postgres():
    """Initializes PostgreSQL with pgvector and required tables/indexes."""
    pg_dsn = os.getenv("DATABASE_URL", 
        f"dbname={os.getenv('POSTGRES_DB', 'asterscope')} "
        f"user={os.getenv('POSTGRES_USER', 'postgres')} "
        f"password={os.getenv('POSTGRES_PASSWORD', 'postgres_secure_password')} "
        f"host={os.getenv('POSTGRES_HOST', 'localhost')} "
        f"port={os.getenv('POSTGRES_PORT', '5432')}"
    )
    
    conn = None
    try:
        conn = psycopg2.connect(pg_dsn)
        conn.autocommit = True
        with conn.cursor() as cur:
            # 1. Enable pgvector
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            
            # 2. Create Documents table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id VARCHAR(255) PRIMARY KEY,
                    title TEXT NOT NULL,
                    metadata JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            
            # 3. Create Chunks table
            # Assuming embedding dim is 384 for all-MiniLM-L6-v2
            cur.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    id VARCHAR(255) PRIMARY KEY,
                    doc_id VARCHAR(255) REFERENCES documents(id) ON DELETE CASCADE,
                    index INTEGER NOT NULL,
                    chunk_text TEXT NOT NULL,
                    embedding vector(384),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            
            # 4. Create Index (HNSW)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS chunks_embedding_idx 
                ON chunks USING hnsw (embedding vector_cosine_ops);
            """)
            logger.info("PostgreSQL initialization successful.")
            
    except Exception as e:
        logger.error("PostgreSQL initialization failed: %s", str(e))
    finally:
        if conn:
            conn.close()

def init_neo4j():
    """Initializes Neo4j uniqueness constraints."""
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "neo4j_secure_password")
    
    driver = None
    try:
        driver = GraphDatabase.driver(uri, auth=(user, password))
        with driver.session() as session:
            # 1. Document Constraint
            session.run("CREATE CONSTRAINT document_id_unique IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE;")
            # 2. Chunk Constraint
            session.run("CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE;")
            logger.info("Neo4j initialization successful.")
    except Exception as e:
        logger.error("Neo4j initialization failed: %s", str(e))
    finally:
        if driver:
            driver.close()

def initialize_databases():
    """Bootstraps all required local databases."""
    logger.info("Bootstrapping AsterScope Databases...")
    init_postgres()
    init_neo4j()
    logger.info("Database bootstrapping complete.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    initialize_databases()
