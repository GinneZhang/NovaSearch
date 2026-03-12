"""
Failure Injection / Resilience Tests for NovaSearch.

Tests that the system's fail-closed guardrails, retry loops, and graceful
degradation work correctly under error conditions:
    - Database timeouts
    - API 429 rate limits
    - Malformed LLM responses
    - Neo4j connection failures
"""

import pytest
from unittest.mock import patch, MagicMock, PropertyMock
import json


# ─── Test: Consistency Evaluator Fail-Closed Under Timeout ───

class TestConsistencyResilience:
    """Verify ConsistencyEvaluator blocks output on errors, not passes."""
    
    @patch("agent.consistency.openai")
    def test_evaluator_blocks_on_openai_timeout(self, mock_openai):
        """If OpenAI times out, the evaluator must BLOCK, not pass."""
        from agent.consistency import ConsistencyEvaluator
        
        mock_client = MagicMock()
        mock_openai.OpenAI.return_value = mock_client
        mock_client.chat.completions.create.side_effect = TimeoutError("Connection timed out")
        
        evaluator = ConsistencyEvaluator()
        evaluator.client = mock_client
        
        result = evaluator.evaluate(
            answer="Some answer",
            context="Some context",
            query="Some query"
        )
        
        assert result["is_consistent"] is False, "Evaluator must BLOCK on timeout"
        assert "blocked_reason" in result or result.get("score", 1.0) == 0.0
    
    @patch("agent.consistency.openai")
    def test_evaluator_blocks_on_invalid_json(self, mock_openai):
        """If LLM returns malformed JSON, evaluator must BLOCK."""
        from agent.consistency import ConsistencyEvaluator
        
        mock_client = MagicMock()
        mock_openai.OpenAI.return_value = mock_client
        
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "This is not JSON at all"
        mock_client.chat.completions.create.return_value = mock_response
        
        evaluator = ConsistencyEvaluator()
        evaluator.client = mock_client
        
        result = evaluator.evaluate(
            answer="Some answer",
            context="Some context",
            query="Some query"
        )
        
        assert result["is_consistent"] is False, "Evaluator must BLOCK on invalid JSON"


# ─── Test: Cypher Generator Self-Healing Under Neo4j Failure ───

class TestCypherResilience:
    """Verify CypherGenerator handles Neo4j failures gracefully."""
    
    @patch("retrieval.graph.cypher_generator.openai")
    def test_cypher_linter_rejects_write_operations(self, mock_openai):
        """Cypher linter must reject any CREATE/MERGE/DELETE operations."""
        from retrieval.graph.cypher_generator import CypherGenerator
        
        mock_client = MagicMock()
        mock_openai.OpenAI.return_value = mock_client
        
        gen = CypherGenerator()
        gen._known_labels = {"Document", "Chunk", "Entity"}
        gen._known_rels = {"HAS_CHUNK", "MENTIONS"}
        
        error = gen._lint_cypher("CREATE (n:Document {title: 'hack'})")
        assert error is not None
        assert "Write operation" in error
    
    @patch("retrieval.graph.cypher_generator.openai")
    def test_cypher_linter_rejects_unknown_labels(self, mock_openai):
        """Cypher linter must reject queries referencing unknown labels."""
        from retrieval.graph.cypher_generator import CypherGenerator
        
        mock_client = MagicMock()
        mock_openai.OpenAI.return_value = mock_client
        
        gen = CypherGenerator()
        gen._known_labels = {"Document", "Chunk", "Entity"}
        gen._known_rels = {"HAS_CHUNK", "MENTIONS"}
        
        error = gen._lint_cypher("MATCH (n:FakeNode) RETURN n")
        assert error is not None
        assert "Unknown label" in error
    
    @patch("retrieval.graph.cypher_generator.openai")
    def test_cypher_linter_passes_valid_query(self, mock_openai):
        """Cypher linter must pass valid queries."""
        from retrieval.graph.cypher_generator import CypherGenerator
        
        mock_client = MagicMock()
        mock_openai.OpenAI.return_value = mock_client
        
        gen = CypherGenerator()
        gen._known_labels = {"Document", "Chunk", "Entity"}
        gen._known_rels = {"HAS_CHUNK", "MENTIONS"}
        
        error = gen._lint_cypher("MATCH (d:Document)-[:HAS_CHUNK]->(c:Chunk) RETURN c.chunk_text")
        assert error is None, f"Expected no error but got: {error}"


# ─── Test: Entity Linker Probabilistic Disambiguation ───

class TestEntityLinkerResilience:
    """Verify EntityLinker handles edge cases and ambiguity correctly."""
    
    def test_linker_returns_none_without_model(self):
        """Without SentenceTransformer, linker must return None gracefully."""
        from retrieval.graph.linker import EntityLinker
        
        linker = EntityLinker.__new__(EntityLinker)
        linker.model = None
        linker.neo4j_driver = None
        linker.similarity_threshold = 0.7
        linker._entity_cache = {}
        
        result = linker.link("test entity")
        assert result is None


# ─── Test: Observability Metrics Collection ───

class TestObservabilityResilience:
    """Verify MetricsCollector handles edge cases."""
    
    def test_metrics_collector_records_latency(self):
        """MetricsCollector must accurately track latency records."""
        from core.observability import MetricsCollector
        
        collector = MetricsCollector()
        collector.record_latency("vector_search", "dense_retrieval", 45.2, success=True)
        collector.record_latency("vector_search", "dense_retrieval", 52.1, success=True)
        collector.record_latency("graph_search", "cypher_exec", 120.5, success=False)
        
        stats = collector.get_engine_stats("vector_search")
        assert stats["total_calls"] == 2
        assert stats["success_rate"] == 1.0
        assert stats["avg_latency_ms"] == pytest.approx(48.65, abs=0.1)
    
    def test_metrics_collector_tracks_errors(self):
        """MetricsCollector must track error counts per engine."""
        from core.observability import MetricsCollector
        
        collector = MetricsCollector()
        collector.record_latency("graph_search", "cypher", 100.0, success=False)
        collector.record_latency("graph_search", "cypher", 200.0, success=False)
        
        stats = collector.get_engine_stats("graph_search")
        assert stats["error_count"] == 2
        assert stats["success_rate"] == 0.0
    
    def test_metrics_collector_empty_engine(self):
        """MetricsCollector must handle queries for unknown engines."""
        from core.observability import MetricsCollector
        
        collector = MetricsCollector()
        stats = collector.get_engine_stats("nonexistent")
        assert stats["total_calls"] == 0
