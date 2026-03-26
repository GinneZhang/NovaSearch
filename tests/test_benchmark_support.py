from typing import Any, Dict, List

from tests.benchmark_rag import build_summary_payload
from tests.benchmark_support import (
    BenchmarkBundle,
    BenchmarkCase,
    BenchmarkPage,
    default_split_for_benchmark,
    default_trace_path,
    exact_match_score,
    load_benchmark_bundle,
    normalize_answer,
    normalize_benchmark_name,
    token_f1_score,
)
from tests.ragas_support import RagasEvaluationResult


class FakeDataset:
    def __init__(self, rows: List[Dict[str, Any]]):
        self.rows = list(rows)

    def shuffle(self, seed: int):
        return FakeDataset(list(self.rows))

    def select(self, indices):
        if isinstance(indices, range):
            return FakeDataset(self.rows[indices.start:indices.stop])
        return FakeDataset([self.rows[i] for i in indices])

    def __iter__(self):
        return iter(self.rows)

    def __getitem__(self, idx: int):
        return self.rows[idx]

    def __len__(self):
        return len(self.rows)


def _fake_load_dataset(*args, **kwargs):
    name = args[0]
    config = args[1] if len(args) > 1 else None
    split = kwargs.get("split")

    if name == "hotpot_qa":
        return FakeDataset(
            [
                {
                    "question": "Who narrated Blackadder's Christmas Carol?",
                    "answer": "Hugh Laurie",
                    "supporting_facts": {"title": ["Blackadder's Christmas Carol"]},
                    "context": {
                        "title": ["Blackadder's Christmas Carol", "A Christmas Carol"],
                        "sentences": [
                            ["Blackadder's Christmas Carol is narrated by Hugh Laurie."],
                            ["A Christmas Carol was written by Charles Dickens."],
                        ],
                    },
                }
            ]
        )

    if name == "squad" and split == "validation":
        return FakeDataset(
            [
                {
                    "id": "s1",
                    "title": "Minority interest",
                    "context": "Non controlling interest is also called minority interest.",
                    "question": "what is non controlling interest on balance sheet",
                    "answers": {"text": ["minority interest"]},
                },
                {
                    "id": "s2",
                    "title": "J. R. R. Tolkien",
                    "context": "The Hobbit was written by J. R. R. Tolkien.",
                    "question": "who wrote the hobbit",
                    "answers": {"text": ["J. R. R. Tolkien"]},
                },
            ]
        )

    if name == "cmriat/musique" and split == "validation":
        return FakeDataset(
            [
                {
                    "id": "m1",
                    "question": "Who wrote The Hobbit and who was his spouse?",
                    "answer": "Edith Tolkien",
                    "answer_aliases": ["Edith Tolkien"],
                    "paragraphs": [
                        {
                            "title": "The Hobbit",
                            "paragraph_text": "The Hobbit is a novel written by J. R. R. Tolkien.",
                        },
                        {
                            "title": "J. R. R. Tolkien",
                            "paragraph_text": "Edith Tolkien was the spouse of J. R. R. Tolkien.",
                        },
                    ],
                }
            ]
        )

    if name == "din0s/asqa" and split == "dev":
        return FakeDataset(
            [
                {
                    "sample_id": "a1",
                    "ambiguous_question": "Where is Fort Presque Isle located?",
                    "annotations": [
                        {
                            "long_answer": "Fort Presque Isle was located near present-day Erie, Pennsylvania.",
                            "knowledge": [
                                {
                                    "wikipage": "Fort Presque Isle",
                                    "content": "Fort Presque Isle was built near present-day Erie, Pennsylvania.",
                                }
                            ],
                        }
                    ],
                }
            ]
        )

    raise AssertionError(f"Unexpected dataset request: args={args}, kwargs={kwargs}")


def test_normalize_answer_scores():
    assert normalize_answer("The Moon!") == "moon"
    assert exact_match_score("The Moon", ["moon"]) == 1.0
    assert token_f1_score("December 1972", ["14 December 1972 UTC"]) > 0.5


def test_benchmark_name_defaults():
    assert normalize_benchmark_name("hotpot") == "hotpotqa"
    assert normalize_benchmark_name("squad2") == "squad_v2"
    assert default_split_for_benchmark("hotpotqa") == "validation"
    assert default_split_for_benchmark("squad") == "validation"
    assert default_split_for_benchmark("squad_v2") == "validation"
    assert default_split_for_benchmark("musique") == "validation"
    assert default_split_for_benchmark("asqa") == "dev"
    assert default_trace_path("hotpotqa").endswith("docs/kpi_trace.json")
    assert default_trace_path("squad").endswith("docs/kpi_trace_squad.json")


def test_load_hotpot_bundle_from_adapter():
    bundle = load_benchmark_bundle(
        "hotpotqa",
        sample_size=1,
        split="validation",
        load_dataset_fn=_fake_load_dataset,
    )
    assert bundle.display_name == "HotpotQA"
    assert len(bundle.cases) == 1
    assert bundle.cases[0].expected_answers == ["Hugh Laurie"]
    assert "Blackadder's Christmas Carol" in bundle.cases[0].expected_titles
    assert len(bundle.unique_pages) == 2


def test_load_squad_bundle_from_adapter():
    bundle = load_benchmark_bundle(
        "squad",
        sample_size=2,
        split="validation",
        load_dataset_fn=_fake_load_dataset,
    )
    assert bundle.display_name == "SQuAD"
    assert len(bundle.cases) == 2
    assert bundle.answer_metric_label == "Ragas answer metrics"
    assert bundle.cases[0].expected_answers
    assert bundle.cases[0].expected_titles
    assert len(bundle.unique_pages) == 2


def test_load_musique_bundle_from_adapter():
    bundle = load_benchmark_bundle(
        "musique",
        sample_size=1,
        split="validation",
        load_dataset_fn=_fake_load_dataset,
    )
    assert bundle.display_name == "MuSiQue"
    assert len(bundle.cases) == 1
    assert bundle.cases[0].expected_answers == ["Edith Tolkien"]
    assert len(bundle.cases[0].pages) == 2


def test_load_asqa_bundle_from_adapter():
    bundle = load_benchmark_bundle(
        "asqa",
        sample_size=1,
        split="dev",
        load_dataset_fn=_fake_load_dataset,
    )
    assert bundle.display_name == "ASQA"
    assert len(bundle.cases) == 1
    assert "Pennsylvania" in bundle.cases[0].expected_answers[0]
    assert bundle.cases[0].pages[0].title == "Fort Presque Isle"


def test_build_summary_payload_uses_ragas_summary_schema():
    bundle = BenchmarkBundle(
        name="squad",
        display_name="SQuAD",
        split="validation",
        sample_size=1,
        retrieval_metric_label="Ragas context metrics only",
        answer_metric_label="Ragas answer metrics",
        cases=[
            BenchmarkCase(
                idx=0,
                query="who wrote the hobbit",
                expected_answers=["J. R. R. Tolkien"],
                expected_titles=["J. R. R. Tolkien"],
                pages=[
                    BenchmarkPage(
                        ref_id="doc1",
                        title="J. R. R. Tolkien",
                        text="J. R. R. Tolkien\nThe Hobbit was written by J. R. R. Tolkien.",
                        section="SQuAD Benchmark doc1",
                    )
                ],
            )
        ],
        unique_pages=[
            BenchmarkPage(
                ref_id="doc1",
                title="J. R. R. Tolkien",
                text="J. R. R. Tolkien\nThe Hobbit was written by J. R. R. Tolkien.",
                section="SQuAD Benchmark doc1",
            )
        ],
        total_raw_tokens=20.0,
        notes=["Ragas-only benchmark summary."],
    )
    results = [
        {
            "answer": "J. R. R. Tolkien",
            "retrieval_contexts": ["J. R. R. Tolkien\nThe Hobbit was written by J. R. R. Tolkien."],
            "generation_contexts": ["J. R. R. Tolkien\nThe Hobbit was written by J. R. R. Tolkien."],
            "supporting_contexts": ["J. R. R. Tolkien\nThe Hobbit was written by J. R. R. Tolkien."],
            "reference_contexts": ["J. R. R. Tolkien\nThe Hobbit was written by J. R. R. Tolkien."],
            "retrieval_token_estimate": 10.0,
            "generation_token_estimate": 10.0,
            "expected_titles": ["J. R. R. Tolkien"],
            "expected_answers": ["J. R. R. Tolkien"],
            "query": "who wrote the hobbit",
            "debug_metrics": {"final_context_count": 1},
            "planner_confirmed_generation_count": 0,
            "supporting_selected_generation_count": 1,
            "generation_planner_chunk_ratio": 0.0,
            "planner_rescued_generation_missed": False,
            "generation_sources_raw": [],
            "supporting_sources_raw": [],
            "bridge_targeting_hits": 0,
            "answer_bearing_bridge_hits": 0,
            "bridge_entity_family_chunk_miss": 0,
            "alias_bridge_candidate_count": 0,
            "bridge_targeted_query_count": 0,
            "duplicates_removed": 0,
            "bridge_coverage_score": 0.0,
            "support_coverage_score": 0.0,
            "source_type_mix": {"dense": 1},
            "evidence_role_mix": {"answer": 1},
        }
    ]
    ragas_result = RagasEvaluationResult(
        sample_rows=[
            {
                **results[0],
                "ragas_contexts": results[0]["generation_contexts"],
                "ragas_ground_truth": "J. R. R. Tolkien",
                "ragas_reference_contexts": results[0]["reference_contexts"],
                "ragas_scores": {
                    "context_precision": 1.0,
                    "context_recall": 1.0,
                    "context_entities_recall": 1.0,
                    "faithfulness": 0.9,
                    "answer_relevancy": 0.8,
                    "noise_sensitivity": 0.1,
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
            "context_precision": 1.0,
            "context_recall": 1.0,
            "context_entities_recall": 1.0,
            "faithfulness": 0.9,
            "answer_relevancy": 0.8,
            "noise_sensitivity": 0.1,
        },
        metrics_applied={"context_precision": "llm_reference"},
        metrics_skipped={},
    )
    summary = build_summary_payload(
        benchmark=bundle,
        query_results=results,
        ragas_result=ragas_result,
        total_context_tokens=10.0,
        ingest_duration=1.0,
        query_duration=2.0,
        context_mode="generation",
        ragas_model="gpt-4.1-mini",
        ragas_embedding_model="text-embedding-3-small",
    )
    assert summary["benchmark_name"] == "squad"
    assert summary["ragas_metrics"]["context_precision"] == 1.0
    assert summary["ragas_context_mode"] == "generation"
    assert summary["answer_em"] == 1.0
    assert summary["answer_f1"] == 1.0
