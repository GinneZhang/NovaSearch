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

    if name == "Hyukkyu/beir-nq" and config == "queries" and split == "train":
        return FakeDataset(
            [
                {"_id": "q1", "text": "what is non controlling interest on balance sheet"},
                {"_id": "q2", "text": "who wrote the hobbit"},
            ]
        )

    if name == "BeIR/nq-qrels" and split == "test":
        return FakeDataset(
            [
                {"query-id": "q1", "corpus-id": "doc0", "score": 1},
                {"query-id": "q2", "corpus-id": "doc1", "score": 1},
            ]
        )

    if name == "Hyukkyu/beir-nq" and config == "corpus" and split == "train":
        return FakeDataset(
            [
                {"_id": "doc0", "title": "Minority interest", "text": "Non controlling interest is also called minority interest."},
                {"_id": "doc1", "title": "J. R. R. Tolkien", "text": "The Hobbit was written by J. R. R. Tolkien."},
                {"_id": "doc2", "title": "Distractor", "text": "This is unrelated."},
            ]
        )

    raise AssertionError(f"Unexpected dataset request: args={args}, kwargs={kwargs}")


def test_normalize_answer_scores():
    assert normalize_answer("The Moon!") == "moon"
    assert exact_match_score("The Moon", ["moon"]) == 1.0
    assert token_f1_score("December 1972", ["14 December 1972 UTC"]) > 0.5


def test_benchmark_name_defaults():
    assert normalize_benchmark_name("hotpot") == "hotpotqa"
    assert normalize_benchmark_name("natural_questions") == "nq"
    assert default_split_for_benchmark("hotpotqa") == "validation"
    assert default_split_for_benchmark("nq") == "test"
    assert default_trace_path("hotpotqa").endswith("docs/kpi_trace.json")
    assert default_trace_path("nq").endswith("docs/kpi_trace_nq.json")


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


def test_load_nq_bundle_from_adapter():
    bundle = load_benchmark_bundle(
        "nq",
        sample_size=2,
        split="test",
        load_dataset_fn=_fake_load_dataset,
    )
    assert bundle.display_name == "Natural Questions (BeIR/NQ)"
    assert len(bundle.cases) == 2
    assert bundle.answer_metric_label is None
    assert bundle.cases[0].expected_answers == []
    assert bundle.cases[0].expected_titles
    assert {page.ref_id for page in bundle.unique_pages} == {"doc0", "doc1"}


def test_build_summary_payload_handles_answerless_benchmark():
    bundle = BenchmarkBundle(
        name="nq",
        display_name="Natural Questions (BeIR/NQ)",
        split="test",
        sample_size=1,
        retrieval_metric_label="Relevant Passage Title Hit/MRR",
        answer_metric_label=None,
        cases=[
            BenchmarkCase(
                idx=0,
                query="who wrote the hobbit",
                expected_answers=[],
                expected_titles=["J. R. R. Tolkien"],
                pages=[
                    BenchmarkPage(
                        ref_id="doc1",
                        title="J. R. R. Tolkien",
                        text="J. R. R. Tolkien\nThe Hobbit was written by J. R. R. Tolkien.",
                        section="NQ Benchmark doc1",
                    )
                ],
            )
        ],
        unique_pages=[
            BenchmarkPage(
                ref_id="doc1",
                title="J. R. R. Tolkien",
                text="J. R. R. Tolkien\nThe Hobbit was written by J. R. R. Tolkien.",
                section="NQ Benchmark doc1",
            )
        ],
        total_raw_tokens=20.0,
        notes=["No gold answers in this package."],
    )
    results = [
        {
            "mrr": 1.0,
            "hr5": 1.0,
            "retrieved_tokens": 10.0,
            "faith_score": 0.9,
            "legacy_cp_score": 0.8,
            "generation_cp_score": 0.7,
            "supporting_cp_score": 0.6,
            "generation_audit": {"counts": {"direct-answer": 1, "bridge-useful": 0, "background": 0, "noise": 0}},
            "supporting_audit": {"counts": {"direct-answer": 1, "bridge-useful": 0, "background": 0, "noise": 0}},
            "was_evaled": True,
            "eval_invalid_reason": None,
            "answer": "J. R. R. Tolkien",
            "answer_em": None,
            "answer_f1": None,
            "retrieval_contexts": ["J. R. R. Tolkien\nThe Hobbit was written by J. R. R. Tolkien."],
            "generation_contexts": ["J. R. R. Tolkien\nThe Hobbit was written by J. R. R. Tolkien."],
            "supporting_contexts": ["J. R. R. Tolkien\nThe Hobbit was written by J. R. R. Tolkien."],
            "expected_titles": ["J. R. R. Tolkien"],
            "expected_answers": [],
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
    summary = build_summary_payload(
        bundle,
        results,
        total_retrieved_tokens=10.0,
        ingest_duration=1.0,
        query_duration=2.0,
        generation_eval_enabled=True,
        evaluator_model="gpt-4.1-mini",
        eval_every_n=5,
    )
    assert summary["benchmark_name"] == "nq"
    assert summary["answer_em"] is None
    assert summary["answer_metric_label"] is None
    assert summary["hit_rate_5"] == 1.0
