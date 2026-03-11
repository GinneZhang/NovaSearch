"""
End-to-End Integration Tests for NovaSearch API.
Requires local Docker infrastructure to be running (PostgreSQL, Redis, Neo4j).
"""

import os
import pytest
from fastapi.testclient import TestClient

# Must set before importing main to ensure safe DB connections
os.environ["DATABASE_URL"] = "dbname=novasearch user=postgres password=postgres_secure_password host=localhost port=5432"
os.environ["NEO4J_URI"] = "bolt://localhost:7687"

# We bypass the actual OpenAI call using a mock if we don't have a real key, 
# but for true E2E, we assume a valid OPENAI_API_KEY is present in the local .env

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
    We just log here for verbosity.
    """
    import logging
    logging.info("Starting E2E Test Suite...")
    yield
    logging.info("Tearing down E2E Test Suite...")


def test_health_check():
    """Verify API is awake."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_ingestion():
    """
    Test 1: POST /ingest
    Verifies that SemanticChunker, PGVector inserts, and Neo4j graph building succeed.
    """
    payload = {
        "document_text": MOCK_TEXT,
        "title": MOCK_TITLE,
        "section": MOCK_SECTION
    }
    
    # We use TestClient with the lifespan context manager implicitly via client request
    with TestClient(app) as live_client:
        response = live_client.post("/ingest", json=payload)
        
        assert response.status_code == 200, f"Ingestion failed: {response.text}"
        data = response.json()
        assert data["status"] == "success"
        assert "processed" in data["details"].lower()


def test_retrieval_and_agent():
    """
    Test 2: POST /ask
    Verifies Tri-Engine Fusion retrieves the previously ingested chunk and 
    the Agent formats the answer with the [Source Marker].
    """
    query_payload = {
        "query": "What happens if I violate the insider trading policy, and who approves exceptions?",
        "top_k": 3
    }
    
    with TestClient(app) as live_client:
        # Give DBs a fraction of a second to sync (though PG/Neo4j are immediate)
        import time
        time.sleep(1)
        
        import json
        
        with live_client.stream("POST", "/ask", json=query_payload) as response:
            assert response.status_code == 200, f"Ask failed: {response.text}"
            
            answer = ""
            sources = []
            
            for line in response.iter_lines():
                if line:
                    try:
                        data = json.loads(line)
                        if data.get("type") == "token":
                            answer += data.get("content", "")
                        elif data.get("type") == "answer_metadata":
                            sources = data.get("sources", [])
                    except json.JSONDecodeError:
                        pass
                        
            # 1. Verify Answer exists
            assert len(answer) > 10, "LLM returned an empty or abnormally short answer."
            
            # 2. Verify Tri-Engine retrieved the correct source
            assert len(sources) > 0, "No sources were retrieved by HybridSearchCoordinator."
            
            # Check if our mock document is in the returned sources
            mock_source_found = False
            for s in sources:
                if MOCK_TITLE in s.get("graph_context", {}).get("doc_title", "") or \
                   MOCK_TITLE in s.get("chunk_text", ""):
                    mock_source_found = True
                    break
                    
            assert mock_source_found, "The ingested mock document was not found by the retrieval engine."
            
            # 3. Verify Source Grounding (Anti-Hallucination)
            expected_marker = f"[Doc: {MOCK_TITLE}, Section: {MOCK_SECTION}]"
            assert expected_marker in answer, f"LLM failed to inject the required source marker. Expected '{expected_marker}' somewhere in the output. Got: {answer}"
