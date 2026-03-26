"""
Regression tests for Ragas benchmark summary reporting and debug traces.
"""

from tests.benchmark_rag import build_summary_payload
from tests.benchmark_support import BenchmarkBundle
from tests.ragas_support import RagasEvaluationResult


def test_build_summary_payload_reports_ragas_and_preserves_debug_metrics():
    bundle = BenchmarkBundle(
        name="hotpotqa",
        display_name="HotpotQA",
        split="validation",
        sample_size=1,
        retrieval_metric_label="unused",
        answer_metric_label="unused",
        cases=[],
        unique_pages=[],
        total_raw_tokens=1000.0,
        notes=[],
    )

    query_results = [
        {
            "answer": "Chief Legal Officer",
            "retrieval_contexts": ["Exception Workflow\nChief Legal Officer approves blackout exceptions."],
            "generation_contexts": [
                "Exception Workflow\nChief Legal Officer approves blackout exceptions.",
                "Policy Summary\nBlackout period begins 15 days before quarter end.",
            ],
            "supporting_contexts": [
                "Exception Workflow\nChief Legal Officer approves blackout exceptions.",
            ],
            "reference_contexts": [
                "Exception Workflow\nChief Legal Officer approves blackout exceptions.",
            ],
            "retrieval_token_estimate": 12.0,
            "generation_token_estimate": 20.0,
            "expected_titles": ["Exception Workflow"],
            "expected_answers": ["Chief Legal Officer"],
            "query": "Who approves blackout exceptions?",
            "planner_confirmed_generation_count": 1,
            "supporting_selected_generation_count": 1,
            "generation_planner_chunk_ratio": 0.5,
            "planner_rescued_generation_missed": False,
            "bridge_targeting_hits": 1,
            "answer_bearing_bridge_hits": 1,
            "bridge_entity_family_chunk_miss": 0,
            "alias_bridge_candidate_count": 1,
            "bridge_targeted_query_count": 1,
            "duplicates_removed": 2,
            "bridge_coverage_score": 0.6,
            "support_coverage_score": 0.9,
            "source_type_mix": {"dense": 1, "lexical": 1},
            "evidence_role_mix": {"answer": 1, "bridge": 1},
            "debug_metrics": {
                "debug_schema_version": "chain-aware-v1",
                "chain_aware_runtime_enabled": True,
                "second_hop_triggered": True,
                "first_hop_candidates": 12,
                "merged_candidate_count": 18,
                "final_context_count": 3,
                "bridge_query_count": 2,
                "pre_pack_count": 14,
                "post_pack_count": 5,
                "candidate_chains": 4,
                "selected_chains": 2,
                "second_hop_added_count": 6,
                "chain_support_coverage_score": 0.7,
                "chain_bridge_coverage_score": 0.55,
                "bridge_chunks_kept": 1,
                "bridge_chunks_dropped": 2,
                "graph_symbolic_candidates": 3,
                "graph_symbolic_kept": 1,
                "graph_symbolic_dropped": 2,
                "corroborated_graph_kept": 1,
                "final_context_chain_mix": {"chain_1": 2, "no_chain": 1},
                "generation_compacted_count": 1,
                "blocked_seed_count": 0,
                "supporting_anchor_filtered_count": 1,
            },
        }
    ]

    ragas_result = RagasEvaluationResult(
        sample_rows=[
            {
                **query_results[0],
                "ragas_contexts": query_results[0]["generation_contexts"],
                "ragas_ground_truth": "Chief Legal Officer",
                "ragas_reference_contexts": query_results[0]["reference_contexts"],
                "ragas_scores": {
                    "context_precision": 0.8,
                    "context_recall": 0.9,
                    "context_entities_recall": 0.7,
                    "faithfulness": 0.95,
                    "answer_relevancy": 0.88,
                    "noise_sensitivity": 0.2,
                },
                "ragas_metric_modes": {
                    "context_precision": "llm_reference",
                    "context_recall": "llm_reference",
                    "context_entities_recall": "llm_reference",
                    "faithfulness": "llm",
                    "answer_relevancy": "llm_embeddings",
                    "noise_sensitivity": "llm_reference",
                },
                "ragas_metric_skips": {},
            }
        ],
        metrics_summary={
            "context_precision": 0.8,
            "context_recall": 0.9,
            "context_entities_recall": 0.7,
            "faithfulness": 0.95,
            "answer_relevancy": 0.88,
            "noise_sensitivity": 0.2,
        },
        metrics_applied={
            "context_precision": "llm_reference",
            "context_recall": "llm_reference",
            "context_entities_recall": "llm_reference",
            "faithfulness": "llm",
            "answer_relevancy": "llm_embeddings",
            "noise_sensitivity": "llm_reference",
        },
        metrics_skipped={},
    )

    summary = build_summary_payload(
        benchmark=bundle,
        query_results=query_results,
        ragas_result=ragas_result,
        total_context_tokens=20.0,
        ingest_duration=1.2,
        query_duration=2.4,
        context_mode="generation",
        ragas_model="gpt-4.1-mini",
        ragas_embedding_model="text-embedding-3-small",
    )

    assert summary["ragas_context_mode"] == "generation"
    assert summary["ragas_metrics"]["context_precision"] == 0.8
    assert summary["ragas_metrics"]["context_entities_recall"] == 0.7
    assert summary["ragas_metric_modes"]["answer_relevancy"] == "llm_embeddings"
    assert summary["avg_pre_pack_count"] == 14.0
    assert summary["avg_post_pack_count"] == 5.0
    assert summary["avg_candidate_chains"] == 4.0
    assert summary["avg_selected_chains"] == 2.0
    assert summary["chain_debug_observed_cases"] == 1
    assert summary["chain_debug_nonzero_cases"] == 1
    assert summary["chain_runtime_enabled_cases"] == 1
    assert summary["debug_schema_versions"] == ["chain-aware-v1"]
    assert summary["avg_chain_support_coverage_score"] == 0.7
    assert summary["avg_chain_bridge_coverage_score"] == 0.55
    assert summary["source_type_mix_totals"] == {"dense": 1, "lexical": 1}
    assert summary["evidence_role_mix_totals"] == {"answer": 1, "bridge": 1}
    assert summary["final_context_chain_mix_totals"] == {"chain_1": 2, "no_chain": 1}
    assert summary["sampled_ragas_cases"][0]["ragas_scores"]["faithfulness"] == 0.95


def test_build_summary_payload_scores_scored_answer_when_present():
    bundle = BenchmarkBundle(
        name="hotpotqa",
        display_name="HotpotQA",
        split="validation",
        sample_size=1,
        retrieval_metric_label="unused",
        answer_metric_label="unused",
        cases=[],
        unique_pages=[],
        total_raw_tokens=1000.0,
        notes=[],
    )

    query_results = [
        {
            "answer": "Gesellschaft mit beschrankter Haftung. Evidence sentence with citation.",
            "scored_answer": "Gesellschaft mit beschrankter Haftung",
            "retrieval_contexts": ["VIVA Media\nVIVA Media GmbH (until 2004 VIVA Media AG)."],
            "generation_contexts": ["VIVA Media\nVIVA Media GmbH (until 2004 VIVA Media AG)."],
            "supporting_contexts": ["VIVA Media\nVIVA Media GmbH (until 2004 VIVA Media AG)."],
            "reference_contexts": ["VIVA Media\nVIVA Media GmbH (until 2004 VIVA Media AG)."],
            "retrieval_token_estimate": 10.0,
            "generation_token_estimate": 10.0,
            "expected_titles": ["VIVA Media"],
            "expected_answers": ["Gesellschaft mit beschrankter Haftung"],
            "query": "What does the new acronym stand for?",
            "planner_confirmed_generation_count": 0,
            "supporting_selected_generation_count": 0,
            "generation_planner_chunk_ratio": 0.0,
            "planner_rescued_generation_missed": False,
            "bridge_targeting_hits": 0,
            "answer_bearing_bridge_hits": 0,
            "bridge_entity_family_chunk_miss": 0,
            "alias_bridge_candidate_count": 0,
            "bridge_targeted_query_count": 0,
            "duplicates_removed": 0,
            "bridge_coverage_score": 0.0,
            "support_coverage_score": 1.0,
            "source_type_mix": {"dense": 1},
            "evidence_role_mix": {"direct": 1},
            "debug_metrics": {},
        }
    ]

    empty_ragas = RagasEvaluationResult(
        sample_rows=[],
        metrics_summary={},
        metrics_applied={},
        metrics_skipped={},
    )

    summary = build_summary_payload(
        benchmark=bundle,
        query_results=query_results,
        ragas_result=empty_ragas,
        total_context_tokens=10.0,
        ingest_duration=1.0,
        query_duration=1.0,
        context_mode="generation",
        ragas_model=None,
        ragas_embedding_model=None,
    )

    assert summary["answer_em"] == 1.0
    assert summary["answer_f1"] == 1.0
