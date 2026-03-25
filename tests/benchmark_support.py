"""
Shared benchmark adapters and metric helpers for NovaSearch evaluation.
"""

from __future__ import annotations

import os
import random
import re
import string
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from datasets import load_dataset


LoadDatasetFn = Callable[..., Any]


@dataclass
class BenchmarkPage:
    ref_id: str
    title: str
    text: str
    section: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkCase:
    idx: int
    query: str
    expected_answers: List[str]
    expected_titles: List[str]
    pages: List[BenchmarkPage]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkBundle:
    name: str
    display_name: str
    split: str
    sample_size: int
    retrieval_metric_label: str
    answer_metric_label: Optional[str]
    cases: List[BenchmarkCase]
    unique_pages: List[BenchmarkPage]
    total_raw_tokens: float
    notes: List[str] = field(default_factory=list)


def normalize_benchmark_name(name: Optional[str]) -> str:
    normalized = (name or "hotpotqa").strip().lower().replace("-", "_")
    aliases = {
        "hotpot": "hotpotqa",
        "hotpot_qa": "hotpotqa",
        "nq": "nq",
        "natural_questions": "nq",
        "natural_questions_open": "nq_open",
        "nq_open": "nq_open",
    }
    return aliases.get(normalized, normalized)


def default_trace_path(benchmark_name: str) -> str:
    normalized = normalize_benchmark_name(benchmark_name)
    if normalized == "hotpotqa":
        return "docs/kpi_trace.json"
    return f"docs/kpi_trace_{normalized}.json"


def default_split_for_benchmark(benchmark_name: str) -> str:
    normalized = normalize_benchmark_name(benchmark_name)
    if normalized == "hotpotqa":
        return "validation"
    if normalized == "nq":
        return "test"
    if normalized == "nq_open":
        return "validation"
    raise ValueError(f"Unsupported benchmark '{benchmark_name}'")


def remove_articles(text: str) -> str:
    return re.sub(r"\b(a|an|the)\b", " ", text)


def white_space_fix(text: str) -> str:
    return " ".join(text.split())


def remove_punc(text: str) -> str:
    exclude = set(string.punctuation)
    return "".join(ch for ch in text if ch not in exclude)


def lower(text: str) -> str:
    return text.lower()


def normalize_answer(text: str) -> str:
    return white_space_fix(remove_articles(remove_punc(lower(text or ""))))


def exact_match_score(prediction: str, gold_answers: Sequence[str]) -> float:
    normalized_prediction = normalize_answer(prediction)
    if not gold_answers:
        return 0.0
    return 1.0 if any(normalized_prediction == normalize_answer(answer) for answer in gold_answers) else 0.0


def token_f1_score(prediction: str, gold_answers: Sequence[str]) -> float:
    if not gold_answers:
        return 0.0
    prediction_tokens = normalize_answer(prediction).split()
    if not prediction_tokens:
        return 0.0

    best_f1 = 0.0
    for answer in gold_answers:
        gold_tokens = normalize_answer(answer).split()
        if not gold_tokens:
            continue
        common = {}
        for token in prediction_tokens:
            if token in gold_tokens:
                common[token] = min(prediction_tokens.count(token), gold_tokens.count(token))
        num_same = sum(common.values())
        if num_same == 0:
            continue
        precision = num_same / len(prediction_tokens)
        recall = num_same / len(gold_tokens)
        best_f1 = max(best_f1, (2 * precision * recall) / max(precision + recall, 1e-8))
    return best_f1


def calc_mrr_at_k(retrieved_contexts: List[str], expected_titles: List[str], k: int = 5) -> float:
    expected = [title.lower() for title in expected_titles if title]
    if not expected:
        return 0.0
    for i, ctx in enumerate(retrieved_contexts[:k]):
        lowered = (ctx or "").lower()
        if any(title in lowered for title in expected):
            return 1.0 / (i + 1)
    return 0.0


def calc_hit_rate_at_k(retrieved_contexts: List[str], expected_titles: List[str], k: int = 5) -> int:
    expected = [title.lower() for title in expected_titles if title]
    if not expected:
        return 0
    for ctx in retrieved_contexts[:k]:
        lowered = (ctx or "").lower()
        if any(title in lowered for title in expected):
            return 1
    return 0


def load_benchmark_bundle(
    benchmark_name: str,
    sample_size: int,
    split: Optional[str] = None,
    seed: int = 42,
    cache_dir: Optional[str] = None,
    load_dataset_fn: LoadDatasetFn = load_dataset,
) -> BenchmarkBundle:
    normalized = normalize_benchmark_name(benchmark_name)
    resolved_split = split or default_split_for_benchmark(normalized)
    resolved_cache_dir = cache_dir or os.getenv("BENCHMARK_DATASET_CACHE_DIR") or os.getenv("HF_DATASETS_CACHE") or "/tmp/hf_cache/datasets"
    if normalized == "hotpotqa":
        return _load_hotpotqa_bundle(sample_size, resolved_split, seed, resolved_cache_dir, load_dataset_fn)
    if normalized == "nq":
        return _load_beir_nq_bundle(sample_size, resolved_split, seed, resolved_cache_dir, load_dataset_fn)
    raise ValueError(
        f"Unsupported benchmark '{benchmark_name}'. Supported benchmarks: hotpotqa, nq"
    )


def _load_hotpotqa_bundle(
    sample_size: int,
    split: str,
    seed: int,
    cache_dir: str,
    load_dataset_fn: LoadDatasetFn,
) -> BenchmarkBundle:
    ds = load_dataset_fn("hotpot_qa", "distractor", split=split, streaming=False, cache_dir=cache_dir)
    subset = ds.shuffle(seed=seed).select(range(sample_size))

    cases: List[BenchmarkCase] = []
    page_corpus: Dict[str, BenchmarkPage] = {}
    total_raw_tokens = 0.0

    for idx, item in enumerate(subset):
        titles = item["context"]["title"]
        sentences = item["context"]["sentences"]
        pages: List[BenchmarkPage] = []
        doc_text_parts: List[str] = []

        for page_idx, title in enumerate(titles):
            page_text = " ".join(sentences[page_idx]).strip()
            full_page_text = f"{title}\n{page_text}".strip()
            ref_id = f"hotpot::{title}"
            page = BenchmarkPage(
                ref_id=ref_id,
                title=title,
                text=full_page_text,
                section=f"HotpotQA Benchmark {idx}",
                metadata={"dataset": "hotpotqa", "page_index": page_idx},
            )
            pages.append(page)
            doc_text_parts.append(full_page_text)
            page_corpus.setdefault(ref_id, page)

        doc_text = "\n\n".join(doc_text_parts).strip()
        total_raw_tokens += len(doc_text.split()) * 1.3
        cases.append(
            BenchmarkCase(
                idx=idx,
                query=item["question"],
                expected_answers=[item["answer"]],
                expected_titles=list(item["supporting_facts"]["title"]),
                pages=pages,
                metadata={"dataset": "hotpotqa"},
            )
        )

    return BenchmarkBundle(
        name="hotpotqa",
        display_name="HotpotQA",
        split=split,
        sample_size=len(cases),
        retrieval_metric_label="Supporting Title Hit/MRR",
        answer_metric_label="Answer EM/F1",
        cases=cases,
        unique_pages=list(page_corpus.values()),
        total_raw_tokens=total_raw_tokens,
        notes=[
            "HotpotQA retains supporting-title retrieval metrics because the dataset exposes page-level supervision.",
        ],
    )


def _load_beir_nq_bundle(
    sample_size: int,
    split: str,
    seed: int,
    cache_dir: str,
    load_dataset_fn: LoadDatasetFn,
) -> BenchmarkBundle:
    if split != "test":
        raise ValueError("NQ benchmark currently supports split='test' only.")

    queries_ds = load_dataset_fn("Hyukkyu/beir-nq", "queries", split="train", streaming=False, cache_dir=cache_dir)
    qrels_ds = load_dataset_fn("BeIR/nq-qrels", split="test", streaming=False, cache_dir=cache_dir)
    corpus_ds = load_dataset_fn("Hyukkyu/beir-nq", "corpus", split="train", streaming=False, cache_dir=cache_dir)

    qrels_by_query: Dict[str, List[str]] = {}
    for row in qrels_ds:
        query_id = str(row.get("query-id") or "")
        corpus_id = str(row.get("corpus-id") or "")
        if not query_id or not corpus_id:
            continue
        qrels_by_query.setdefault(query_id, []).append(corpus_id)

    query_rows: List[Tuple[str, str]] = []
    for row in queries_ds:
        query_id = str(row.get("_id") or "")
        query_text = str(row.get("text") or "").strip()
        if query_id and query_text and query_id in qrels_by_query:
            query_rows.append((query_id, query_text))

    random.Random(seed).shuffle(query_rows)
    selected_queries = query_rows[:sample_size]
    needed_doc_ids = {
        doc_id
        for query_id, _ in selected_queries
        for doc_id in qrels_by_query.get(query_id, [])
    }

    page_by_id: Dict[str, BenchmarkPage] = {}
    for row in corpus_ds:
        doc_id = str(row.get("_id") or "")
        if doc_id not in needed_doc_ids:
            continue
        title = str(row.get("title") or doc_id).strip()
        text = str(row.get("text") or "").strip()
        full_text = f"{title}\n{text}".strip()
        page_by_id[doc_id] = BenchmarkPage(
            ref_id=doc_id,
            title=title,
            text=full_text,
            section=f"NQ Benchmark {doc_id}",
            metadata={"dataset": "nq", "corpus_id": doc_id},
        )
        if len(page_by_id) >= len(needed_doc_ids):
            break

    cases: List[BenchmarkCase] = []
    total_raw_tokens = 0.0
    used_page_ids = set()
    for idx, (query_id, query_text) in enumerate(selected_queries):
        pages = [
            page_by_id[doc_id]
            for doc_id in qrels_by_query.get(query_id, [])
            if doc_id in page_by_id
        ]
        if not pages:
            continue
        for page in pages:
            if page.ref_id not in used_page_ids:
                total_raw_tokens += len(page.text.split()) * 1.3
                used_page_ids.add(page.ref_id)
        cases.append(
            BenchmarkCase(
                idx=idx,
                query=query_text,
                expected_answers=[],
                expected_titles=[page.title for page in pages],
                pages=pages,
                metadata={
                    "dataset": "nq",
                    "query_id": query_id,
                    "relevant_doc_ids": [page.ref_id for page in pages],
                },
            )
        )

    return BenchmarkBundle(
        name="nq",
        display_name="Natural Questions (BeIR/NQ)",
        split=split,
        sample_size=len(cases),
        retrieval_metric_label="Relevant Passage Title Hit/MRR",
        answer_metric_label=None,
        cases=cases,
        unique_pages=list(page_by_id.values()),
        total_raw_tokens=total_raw_tokens,
        notes=[
            "This NQ track uses BeIR/NQ-style corpus, queries, and qrels rather than answer strings.",
            "Answer EM/F1 is not reported because this retrieval benchmark package does not provide canonical gold answers.",
        ],
    )


def resolve_sample_size(benchmark_name: str) -> int:
    normalized = normalize_benchmark_name(benchmark_name)
    generic = os.getenv("BENCHMARK_SAMPLE_SIZE")
    if generic:
        return int(generic)
    if normalized == "hotpotqa":
        return int(os.getenv("HOTPOT_SAMPLE_SIZE", "1000"))
    return 100


def resolve_eval_every_n(benchmark_name: str) -> int:
    generic = os.getenv("BENCHMARK_EVAL_EVERY_N")
    if generic:
        return max(1, int(generic))
    if normalize_benchmark_name(benchmark_name) == "hotpotqa":
        return max(1, int(os.getenv("HOTPOT_EVAL_EVERY_N", "5")))
    return 5


def resolve_generation_eval_enabled(benchmark_name: str) -> bool:
    generic = os.getenv("BENCHMARK_EVAL_GENERATION")
    if generic is not None:
        return generic.lower() in {"1", "true", "yes"}
    if normalize_benchmark_name(benchmark_name) == "hotpotqa":
        return os.getenv("HOTPOT_EVAL_GENERATION", "true").lower() in {"1", "true", "yes"}
    return True
