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
try:
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None


LoadDatasetFn = Callable[..., Any]


def _to_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        converted = tolist()
        if isinstance(converted, list):
            return converted
        if isinstance(converted, tuple):
            return list(converted)
        return [converted]
    return [value]


def _load_remote_parquet_rows(url: str, sample_size: int, seed: int) -> List[Dict[str, Any]]:
    if pd is None:
        raise RuntimeError("pandas is required for parquet-backed benchmark adapters.")
    frame = pd.read_parquet(url)
    if len(frame) > sample_size:
        frame = frame.sample(n=sample_size, random_state=seed)
    return frame.to_dict(orient="records")


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
        "musique_ans": "musique",
        "mu_sique": "musique",
        "squad2": "squad_v2",
        "squad_2": "squad_v2",
        "squad_2_0": "squad_v2",
        "squad2_0": "squad_v2",
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
    if normalized == "squad":
        return "validation"
    if normalized == "squad_v2":
        return "validation"
    if normalized == "musique":
        return "validation"
    if normalized == "asqa":
        return "dev"
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
    if normalized == "squad":
        return _load_squad_bundle(sample_size, resolved_split, seed, resolved_cache_dir, load_dataset_fn)
    if normalized == "squad_v2":
        return _load_squad_v2_bundle(sample_size, resolved_split, seed, resolved_cache_dir, load_dataset_fn)
    if normalized == "musique":
        return _load_musique_bundle(sample_size, resolved_split, seed, resolved_cache_dir, load_dataset_fn)
    if normalized == "asqa":
        return _load_asqa_bundle(sample_size, resolved_split, seed, resolved_cache_dir, load_dataset_fn)
    raise ValueError(
        f"Unsupported benchmark '{benchmark_name}'. Supported benchmarks: hotpotqa, musique, asqa, squad, squad_v2"
    )


def _load_hotpotqa_bundle(
    sample_size: int,
    split: str,
    seed: int,
    cache_dir: str,
    load_dataset_fn: LoadDatasetFn,
) -> BenchmarkBundle:
    if load_dataset_fn is load_dataset:
        parquet_split = "validation" if split == "validation" else split
        subset = _load_remote_parquet_rows(
            f"https://huggingface.co/datasets/hotpotqa/hotpot_qa/resolve/main/distractor/{parquet_split}-00000-of-00001.parquet",
            sample_size=sample_size,
            seed=seed,
        )
    else:
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


def _build_squad_bundle(
    dataset_name: str,
    display_name: str,
    sample_size: int,
    split: str,
    seed: int,
    cache_dir: str,
    load_dataset_fn: LoadDatasetFn,
) -> BenchmarkBundle:
    ds = load_dataset_fn(dataset_name, split=split, streaming=False, cache_dir=cache_dir)
    subset = ds.shuffle(seed=seed).select(range(sample_size))

    cases: List[BenchmarkCase] = []
    page_corpus: Dict[Tuple[str, str], BenchmarkPage] = {}
    total_raw_tokens = 0.0

    for idx, item in enumerate(subset):
        title = str(item.get("title") or f"{display_name} Article").strip() or f"{display_name} Article"
        context = str(item.get("context") or "").strip()
        if not context:
            continue
        page_key = (title, context)
        if page_key not in page_corpus:
            ref_id = f"{dataset_name}::{len(page_corpus)}"
            page_corpus[page_key] = BenchmarkPage(
                ref_id=ref_id,
                title=title,
                text=f"{title}\n{context}".strip(),
                section=f"{display_name} Benchmark {idx}",
                metadata={"dataset": dataset_name},
            )
            total_raw_tokens += len(context.split()) * 1.3
        page = page_corpus[page_key]
        answer_list = [answer.strip() for answer in (item.get("answers") or {}).get("text", []) if str(answer).strip()]
        cases.append(
            BenchmarkCase(
                idx=idx,
                query=str(item.get("question") or "").strip(),
                expected_answers=answer_list,
                expected_titles=[title],
                pages=[page],
                metadata={
                    "dataset": dataset_name,
                    "id": item.get("id"),
                    "is_impossible": bool(item.get("is_impossible", False)),
                },
            )
        )

    notes = [
        f"{display_name} uses paragraph-level reference contexts from the official Hugging Face dataset.",
        "Ragas context metrics use the benchmark reference contexts; answer-dependent metrics are skipped only for answerless rows.",
    ]

    return BenchmarkBundle(
        name=dataset_name,
        display_name=display_name,
        split=split,
        sample_size=len(cases),
        retrieval_metric_label="Ragas context metrics only",
        answer_metric_label="Ragas answer metrics",
        cases=cases,
        unique_pages=list(page_corpus.values()),
        total_raw_tokens=total_raw_tokens,
        notes=notes,
    )


def _load_squad_bundle(
    sample_size: int,
    split: str,
    seed: int,
    cache_dir: str,
    load_dataset_fn: LoadDatasetFn,
) -> BenchmarkBundle:
    return _build_squad_bundle(
        dataset_name="squad",
        display_name="SQuAD",
        sample_size=sample_size,
        split=split,
        seed=seed,
        cache_dir=cache_dir,
        load_dataset_fn=load_dataset_fn,
    )


def _load_squad_v2_bundle(
    sample_size: int,
    split: str,
    seed: int,
    cache_dir: str,
    load_dataset_fn: LoadDatasetFn,
) -> BenchmarkBundle:
    return _build_squad_bundle(
        dataset_name="squad_v2",
        display_name="SQuAD 2.0",
        sample_size=sample_size,
        split=split,
        seed=seed,
        cache_dir=cache_dir,
        load_dataset_fn=load_dataset_fn,
    )


def _load_musique_bundle(
    sample_size: int,
    split: str,
    seed: int,
    cache_dir: str,
    load_dataset_fn: LoadDatasetFn,
) -> BenchmarkBundle:
    if load_dataset_fn is load_dataset:
        subset = _load_remote_parquet_rows(
            "https://huggingface.co/api/datasets/cmriat/musique/parquet/default/validation/0.parquet",
            sample_size=sample_size,
            seed=seed,
        )
    else:
        ds = load_dataset_fn("cmriat/musique", split=split, streaming=False, cache_dir=cache_dir)
        subset = ds.shuffle(seed=seed).select(range(sample_size))

    cases: List[BenchmarkCase] = []
    page_corpus: Dict[Tuple[str, str], BenchmarkPage] = {}
    total_raw_tokens = 0.0

    for idx, item in enumerate(subset):
        metadata = item.get("metadata") or {}
        decomposition = _to_list(metadata.get("question_decomposition"))
        if not decomposition:
            for paragraph in _to_list(item.get("paragraphs")):
                if not isinstance(paragraph, dict):
                    continue
                decomposition.append(
                    {
                        "support_paragraph": {
                            "title": paragraph.get("title") or paragraph.get("paragraph_title"),
                            "paragraph_text": paragraph.get("paragraph_text")
                            or paragraph.get("context")
                            or paragraph.get("paragraph"),
                        }
                    }
                )
        pages: List[BenchmarkPage] = []
        expected_titles: List[str] = []
        for step in decomposition:
            if not isinstance(step, dict):
                continue
            support = step.get("support_paragraph") or {}
            title = str(support.get("title") or "").strip()
            text = str(
                support.get("paragraph_text")
                or ""
            ).strip()
            if not title or not text:
                continue
            page_key = (title, text)
            if page_key not in page_corpus:
                ref_id = f"musique::{len(page_corpus)}"
                page_corpus[page_key] = BenchmarkPage(
                    ref_id=ref_id,
                    title=title,
                    text=f"{title}\n{text}".strip(),
                    section=f"MuSiQue Benchmark {idx}",
                    metadata={"dataset": "musique"},
                )
                total_raw_tokens += len(text.split()) * 1.3
            pages.append(page_corpus[page_key])
            expected_titles.append(title)

        answers = []
        for field_name in ("answer_aliases", "golden_answers"):
            for answer in _to_list(item.get(field_name)):
                normalized = str(answer).strip()
                if normalized and normalized not in answers:
                    answers.append(normalized)
        primary_answer = str(item.get("answer") or "").strip()
        if primary_answer and primary_answer not in answers:
            answers.insert(0, primary_answer)

        if not pages or not answers:
            continue

        cases.append(
            BenchmarkCase(
                idx=idx,
                query=str(item.get("question") or "").strip(),
                expected_answers=answers,
                expected_titles=expected_titles,
                pages=pages,
                metadata={"dataset": "musique", "id": item.get("id")},
            )
        )

    return BenchmarkBundle(
        name="musique",
        display_name="MuSiQue",
        split=split,
        sample_size=len(cases),
        retrieval_metric_label="Ragas context metrics + evidence diagnostics",
        answer_metric_label="Answer EM/F1 + Ragas",
        cases=cases,
        unique_pages=list(page_corpus.values()),
        total_raw_tokens=total_raw_tokens,
        notes=[
            "MuSiQue uses support paragraphs packaged with each question instance.",
            "Answer EM/F1 is computed against the provided answer aliases; Ragas uses the benchmark support paragraphs as reference contexts.",
        ],
    )


def _load_asqa_bundle(
    sample_size: int,
    split: str,
    seed: int,
    cache_dir: str,
    load_dataset_fn: LoadDatasetFn,
) -> BenchmarkBundle:
    if load_dataset_fn is load_dataset:
        subset = _load_remote_parquet_rows(
            "https://huggingface.co/api/datasets/din0s/asqa/parquet/default/dev/0.parquet",
            sample_size=sample_size,
            seed=seed,
        )
    else:
        ds = load_dataset_fn("din0s/asqa", split=split, streaming=False, cache_dir=cache_dir)
        subset = ds.shuffle(seed=seed).select(range(sample_size))

    cases: List[BenchmarkCase] = []
    page_corpus: Dict[Tuple[str, str], BenchmarkPage] = {}
    total_raw_tokens = 0.0

    for idx, item in enumerate(subset):
        annotations = _to_list(item.get("annotations"))
        expected_answers: List[str] = []
        pages: List[BenchmarkPage] = []
        expected_titles: List[str] = []

        for annotation in annotations:
            if not isinstance(annotation, dict):
                continue
            long_answer = str(annotation.get("long_answer") or "").strip()
            if long_answer and long_answer not in expected_answers:
                expected_answers.append(long_answer)
            for knowledge in _to_list(annotation.get("knowledge")):
                if not isinstance(knowledge, dict):
                    continue
                title = str(knowledge.get("wikipage") or knowledge.get("title") or "").strip()
                text = str(knowledge.get("content") or knowledge.get("text") or "").strip()
                if not title or not text:
                    continue
                page_key = (title, text)
                if page_key not in page_corpus:
                    ref_id = f"asqa::{len(page_corpus)}"
                    page_corpus[page_key] = BenchmarkPage(
                        ref_id=ref_id,
                        title=title,
                        text=f"{title}\n{text}".strip(),
                        section=f"ASQA Benchmark {idx}",
                        metadata={"dataset": "asqa"},
                    )
                    total_raw_tokens += len(text.split()) * 1.3
                pages.append(page_corpus[page_key])
                expected_titles.append(title)

        if not pages or not expected_answers:
            continue

        cases.append(
            BenchmarkCase(
                idx=idx,
                query=str(item.get("ambiguous_question") or item.get("question") or "").strip(),
                expected_answers=expected_answers,
                expected_titles=expected_titles,
                pages=pages,
                metadata={"dataset": "asqa", "id": item.get("sample_id") or item.get("id")},
            )
        )

    return BenchmarkBundle(
        name="asqa",
        display_name="ASQA",
        split=split,
        sample_size=len(cases),
        retrieval_metric_label="Ragas context metrics + evidence diagnostics",
        answer_metric_label="Long-form answer EM/F1 (diagnostic) + Ragas",
        cases=cases,
        unique_pages=list(page_corpus.values()),
        total_raw_tokens=total_raw_tokens,
        notes=[
            "ASQA is long-form grounded QA; Answer Relevancy and Faithfulness remain the primary generation metrics.",
            "EM/F1 is retained only as a diagnostic because ASQA is not a classic extractive leaderboard.",
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
