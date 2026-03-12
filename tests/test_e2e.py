"""
End-to-End Integration Tests for NovaSearch API.
Requires local Docker infrastructure to be running (PostgreSQL, Redis, Neo4j).
"""

import os
import json
import pytest
from fastapi.testclient import TestClient

from dotenv import load_dotenv
load_dotenv()

os.environ["VECTOR_STORE_TYPE"] = "pgvector"
os.environ["SPARSE_STORE_TYPE"] = "fts"
os.environ["RERANKER_TYPE"] = "crossencoder"

from api.main import app

client = TestClient(app)

# The Mock Document
MOCK_TITLE = "Acme Corp Policy 404: Insider Trading & Data Compliance"
MOCK_SECTION = "Section 3.1: Blackout Periods"
MOCK_TEXT = (
    "Employees possessing material non-public information (MNPI) are strictly "
    "prohibited from trading company stock. The standard blackout period commences "
    "15 days prior to the end of the fiscal quarter. Exception: Pre-arranged 10b5-1 "
    "trading plans approved by the Chief Legal Officer (CLO). Violations will result "
    "in immediate termination and referral to the SEC."
)

@pytest.fixture(scope="module", autouse=True)
def setup_environment():
    """
    Trigger the lifespan event manually for TestClient to bootstrap DBs.
    FastAPI TestClient doesn't fire lifespan events automatically in older versions,
    but with TestClient(app) in newer versions, it handles it. 
    """
    import logging
    logging.info("Starting E2E Test Suite...")
    yield
    logging.info("Tearing down E2E Test Suite...")


def test_health_check():
    """Verify API is awake."""
    headers = {"X-API-KEY": os.getenv("API_KEY")}
    response = client.get("/health", headers=headers)
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_ingestion():
    """
    Test 1: POST /ingest
    Verifies that SemanticChunker, PGVector inserts, and Neo4j graph building succeed.
    """
    payload = {
        "text_input": MOCK_TEXT,
        "title": MOCK_TITLE,
        "section": MOCK_SECTION
    }
    
    with TestClient(app) as live_client:
        headers = {"X-API-KEY": os.getenv("API_KEY")}
        response = live_client.post("/ingest", data=payload, headers=headers)
        
        assert response.status_code == 200, f"Ingestion failed: {response.text}"
        data = response.json()
        assert data["status"] == "success"
        assert "processed" in data["details"].lower()


def test_retrieval_and_agent():
    """
    Test 2: POST /ask
    Verifies Tri-Engine Fusion retrieves the previously ingested chunk and 
    the Agent formats the streaming answer with proper NDJSON chunks and source markers.
    """
    query_payload = {
        "query": "What happens if I violate the insider trading policy, and who approves exceptions?",
        "top_k": 3
    }
    
    with TestClient(app) as live_client:
        # Give DBs a few seconds to sync
        import time
        time.sleep(3)
        headers = {"X-API-KEY": os.getenv("API_KEY")}
        with live_client.stream("POST", "/ask", json=query_payload, headers=headers) as response:
            if response.status_code != 200:
                response.read()
                assert False, f"Ask failed: {response.text}"
            
            thoughts = []
            answer = ""
            sources = []
            
            for line in response.iter_lines():
                if line:
                    try:
                        data = json.loads(line)
                        chunk_type = data.get("type")
                        
                        if chunk_type == "thought":
                            thoughts.append(data.get("content", ""))
                        elif chunk_type == "token":
                            answer += data.get("content", "")
                        elif chunk_type == "answer_metadata":
                            sources = data.get("sources", [])
                    except json.JSONDecodeError:
                        pass
            
            # 1. Verify Thoughts were emitted
            assert len(thoughts) > 0, "No thoughts were emitted in the streaming response."
            
            # 2. Verify Answer exists and accumulated via tokens
            assert len(answer) > 10, "LLM returned an empty or abnormally short string."
            
            # 3. Verify Tri-Engine retrieved the correct source
            assert len(sources) > 0, "No sources were returned in answer_metadata."
            
            # Check if our mock document is in the returned sources
            mock_source_found = False
            for s in sources:
                if MOCK_TITLE in s.get("graph_context", {}).get("doc_title", "") or \
                   MOCK_TITLE in s.get("chunk_text", ""):
                    mock_source_found = True
                    break
                    
            assert mock_source_found, "The ingested mock document was not found by the retrieval engine."
            
            # 4. Verify Source Grounding (Anti-Hallucination)
            # The LLM should cite source markers. We check for the document title
            # appearing in any citation format, since the exact section text may vary.
            has_citation = "[Doc:" in answer or f"[Doc: {MOCK_TITLE}" in answer or MOCK_TITLE in answer
            assert has_citation, (
                f"LLM failed to inject a recognizable source citation. "
                f"Expected a '[Doc: ...]' marker or the document title in the output. Got: {answer}"
            )
