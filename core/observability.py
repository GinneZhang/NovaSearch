"""
Observability Hooks for NovaSearch.

Provides structured performance metrics tracking for retrieval engines,
LLM calls, and consistency evaluation. Logs metrics in a format compatible
with Prometheus/OpenTelemetry-style collectors.
"""

import time
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class LatencyRecord:
    """Single latency measurement."""
    engine: str
    operation: str
    latency_ms: float
    timestamp: float = field(default_factory=time.time)
    success: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


class MetricsCollector:
    """
    Centralized metrics collector for NovaSearch pipeline observability.
    
    Tracks:
        - Retrieval latency per engine (Vector, Graph, Sparse, Vision)
        - LLM call latency and token counts
        - Consistency evaluator scores
        - Error rates per component
    """
    
    def __init__(self):
        self._latencies: Dict[str, List[LatencyRecord]] = defaultdict(list)
        self._error_counts: Dict[str, int] = defaultdict(int)
        self._llm_scores: List[Dict[str, float]] = []
        self._request_count: int = 0
    
    def record_latency(self, engine: str, operation: str, latency_ms: float,
                       success: bool = True, metadata: Optional[Dict] = None):
        """Record a latency measurement for an engine operation."""
        record = LatencyRecord(
            engine=engine,
            operation=operation,
            latency_ms=latency_ms,
            success=success,
            metadata=metadata or {}
        )
        self._latencies[engine].append(record)
        
        if not success:
            self._error_counts[engine] += 1
        
        logger.info(
            f"[METRIC] engine={engine} op={operation} latency={latency_ms:.1f}ms "
            f"success={success}"
        )
    
    def record_consistency_score(self, score: float, model: str, query_preview: str = ""):
        """Record a consistency evaluator score."""
        self._llm_scores.append({
            "score": score,
            "model": model,
            "query": query_preview[:100],
            "timestamp": time.time()
        })
        logger.info(f"[METRIC] consistency_score={score:.2f} model={model}")
    
    def record_request(self):
        """Increment the request counter."""
        self._request_count += 1
    
    def record_error(self, engine: str):
        """Increment error count for an engine."""
        self._error_counts[engine] += 1
    
    def get_engine_stats(self, engine: str) -> Dict[str, Any]:
        """Get aggregate statistics for a specific engine."""
        records = self._latencies.get(engine, [])
        if not records:
            return {"engine": engine, "total_calls": 0}
        
        latencies = [r.latency_ms for r in records]
        successes = sum(1 for r in records if r.success)
        
        return {
            "engine": engine,
            "total_calls": len(records),
            "success_rate": round(successes / len(records), 4),
            "avg_latency_ms": round(sum(latencies) / len(latencies), 2),
            "min_latency_ms": round(min(latencies), 2),
            "max_latency_ms": round(max(latencies), 2),
            "p95_latency_ms": round(sorted(latencies)[int(len(latencies) * 0.95)], 2) if len(latencies) > 1 else round(latencies[0], 2),
            "error_count": self._error_counts.get(engine, 0)
        }
    
    def get_all_stats(self) -> Dict[str, Any]:
        """Get aggregate statistics for all engines."""
        engines = list(self._latencies.keys())
        return {
            "total_requests": self._request_count,
            "engines": {e: self.get_engine_stats(e) for e in engines},
            "consistency_scores": {
                "count": len(self._llm_scores),
                "avg": round(
                    sum(s["score"] for s in self._llm_scores) / len(self._llm_scores), 3
                ) if self._llm_scores else 0.0
            },
            "total_errors": sum(self._error_counts.values())
        }
    
    def reset(self):
        """Reset all metrics."""
        self._latencies.clear()
        self._error_counts.clear()
        self._llm_scores.clear()
        self._request_count = 0


class LatencyTimer:
    """
    Context manager for measuring operation latency and recording it
    to the MetricsCollector.
    
    Usage:
        with LatencyTimer(collector, "vector_search", "dense_retrieval"):
            results = vector_db.search(query)
    """
    
    def __init__(self, collector: MetricsCollector, engine: str, operation: str):
        self.collector = collector
        self.engine = engine
        self.operation = operation
        self._start: float = 0.0
        self._success = True
    
    def __enter__(self):
        self._start = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed_ms = (time.time() - self._start) * 1000
        self._success = exc_type is None
        self.collector.record_latency(
            engine=self.engine,
            operation=self.operation,
            latency_ms=elapsed_ms,
            success=self._success
        )
        return False  # Don't suppress exceptions


# Global singleton
_metrics = MetricsCollector()


def get_metrics() -> MetricsCollector:
    """Get the global MetricsCollector singleton."""
    return _metrics
