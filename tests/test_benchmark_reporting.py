"""
Regression tests for benchmark summary reporting and evidence-quality traces.
"""

from tests.benchmark_rag import build_summary_payload
from tests.benchmark_support import BenchmarkBundle


def test_build_summary_payload_keeps_new_evidence_quality_metrics():
    bundle = BenchmarkBundle(
        name="hotpotqa",
        display_name="HotpotQA",
        split="validation",
        sample_size=1,
        retrieval_metric_label="Reference page hit within top-5 retrieved contexts",
        answer_metric_label="Answer EM / token-F1",
        cases=[],
        unique_pages=[],
        total_raw_tokens=1000.0,
        notes=[],
    )

    query_results = [
        {
            "mrr": 1.0,
            "hr5": 1,
            "was_evaled": True,
            "faith_score": 0.8,
            "legacy_cp_score": 0.75,
            "generation_cp_score": 0.5,
            "supporting_cp_score": 1.0,
            "generation_audit": {
                "counts": {
                    "direct-answer": 1,
                    "bridge-useful": 0,
                    "background": 1,
                    "noise": 1,
                }
            },
            "supporting_audit": {
                "counts": {
                    "direct-answer": 1,
                    "bridge-useful": 1,
                    "background": 0,
                    "noise": 0,
                }
            },
            "answer": "Chief Legal Officer",
            "answer_em": 1.0,
            "answer_f1": 1.0,
            "retrieval_contexts": ["Exception Workflow\nChief Legal Officer approves blackout exceptions."],
            "generation_contexts": [
                "Exception Workflow\nChief Legal Officer approves blackout exceptions.",
                "Policy Summary\nBlackout period begins 15 days before quarter end.",
                "Company Overview\nAcme was founded in 1998.",
            ],
            "supporting_contexts": [
                "Exception Workflow\nChief Legal Officer approves blackout exceptions.",
                "Policy Summary\nBlackout period begins 15 days before quarter end.",
            ],
            "expected_titles": ["Exception Workflow"],
            "expected_answers": ["Chief Legal Officer"],
            "query": "Who approves blackout exceptions?",
            "planner_confirmed_generation_count": 1,
            "supporting_selected_generation_count": 1,
            "generation_planner_chunk_ratio": 1 / 3,
            "planner_rescued_generation_missed": False,
            "generation_sources_raw": ["dense", "lexical"],
            "supporting_sources_raw": ["dense"],
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
            "eval_invalid_reason": None,
            "debug_metrics": {
                "second_hop_triggered": True,
                "first_hop_candidates": 12,
                "merged_candidate_count": 18,
                "final_context_count": 3,
                "bridge_query_count": 2,
                "pre_pack_count": 14,
                "post_pack_count": 5,
                "second_hop_added_count": 6,
                "graph_symbolic_candidates": 3,
                "graph_symbolic_kept": 1,
                "graph_symbolic_dropped": 2,
                "corroborated_graph_kept": 1,
                "generation_compacted_count": 1,
                "blocked_seed_count": 0,
                "supporting_anchor_filtered_count": 1,
                "seed_mismatch_count": 0,
                "final_supporting_unanchored_count": 0,
            },
        }
    ]

    summary = build_summary_payload(
        benchmark=bundle,
        query_results=query_results,
        total_retrieved_tokens=250.0,
        ingest_duration=1.2,
        query_duration=2.4,
        generation_eval_enabled=True,
        evaluator_model="gpt-4.1-mini",
        eval_every_n=1,
    )

    assert summary["avg_pre_pack_count"] == 14.0
    assert summary["avg_post_pack_count"] == 5.0
    assert summary["graph_symbolic_candidates"] == 3
    assert summary["graph_symbolic_kept"] == 1
    assert summary["graph_symbolic_dropped"] == 2
    assert summary["corroborated_graph_kept"] == 1
    assert summary["generation_compacted_total"] == 1
    assert summary["support_selector_worse_cases"] == 0
    assert summary["generation_assembly_worse_cases"] == 1
    assert summary["source_type_mix_totals"] == {"dense": 1, "lexical": 1}
    assert summary["evidence_role_mix_totals"] == {"answer": 1, "bridge": 1}
    assert summary["generation_noise_ratio"] == (1 / 3)
