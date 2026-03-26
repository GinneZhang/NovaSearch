"""
Focused tests for Ragas mapping and benchmark preflight preservation.
"""

import asyncio
import json

from tests.benchmark_rag import verify_runtime_preflight
from tests.ragas_support import (
    DEFAULT_RAGAS_METRICS,
    build_ragas_sample_row,
    normalize_context_mode,
    parse_metric_selection,
    select_eval_contexts,
)


def test_ragas_mapping_uses_generation_contexts_by_default():
    row = {
        "query": "Who approves blackout exceptions?",
        "answer": "Chief Legal Officer",
        "generation_contexts": ["Generation context"],
        "retrieval_contexts": ["Retrieval context"],
        "expected_answers": ["Chief Legal Officer"],
        "reference_contexts": ["Reference context"],
        "debug_metrics": {"chain_mode_selected": "light"},
    }

    sample = build_ragas_sample_row(row)

    assert sample["question"] == row["query"]
    assert sample["contexts"] == ["Generation context"]
    assert sample["ground_truth"] == "Chief Legal Officer"
    assert sample["reference_contexts"] == ["Reference context"]
    assert sample["debug_metrics"]["chain_mode_selected"] == "light"


def test_ragas_mapping_supports_retrieval_context_mode():
    row = {
        "query": "Who approves blackout exceptions?",
        "answer": "Chief Legal Officer",
        "generation_contexts": ["Generation context"],
        "retrieval_contexts": ["Retrieval context"],
        "expected_answers": ["Chief Legal Officer"],
        "reference_contexts": ["Reference context"],
        "debug_metrics": {},
    }

    assert normalize_context_mode("retrieval") == "retrieval"
    assert select_eval_contexts(row, "retrieval") == ["Retrieval context"]
    sample = build_ragas_sample_row(row, context_mode="retrieval")
    assert sample["contexts"] == ["Retrieval context"]
    assert sample["context_mode"] == "retrieval"


def test_parse_metric_selection_defaults_to_common_core_metrics():
    assert parse_metric_selection(None) == DEFAULT_RAGAS_METRICS
    assert parse_metric_selection("") == DEFAULT_RAGAS_METRICS
    assert parse_metric_selection("faithfulness,answer_relevancy") == (
        "faithfulness",
        "answer_relevancy",
    )


def test_runtime_preflight_still_requires_chain_debug_schema():
    class FakeResponse:
        status_code = 200

        def raise_for_status(self):
            return None

        @property
        def text(self):
            return "\n".join(
                [
                    json.dumps({"type": "token", "content": "ok"}),
                    json.dumps(
                        {
                            "type": "answer_metadata",
                            "debug_metrics": {
                                "debug_schema_version": "chain-aware-v1",
                                "chain_aware_runtime_enabled": True,
                                "candidate_chains": 2,
                                "selected_chains": 1,
                                "chain_lengths": [2],
                                "path_score_distribution": [1.1],
                            },
                        }
                    ),
                ]
            )

    class FakeClient:
        async def post(self, *args, **kwargs):
            return FakeResponse()

    result = asyncio.run(
        verify_runtime_preflight(
            client=FakeClient(),
            base_url="http://127.0.0.1:8000",
            headers={},
            canary_query="Who approves blackout exceptions?",
        )
    )

    assert result["schema_version"] == "chain-aware-v1"
    assert result["runtime_enabled"] is True
    assert result["candidate_chains"] == 2
