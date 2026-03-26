"""
Ragas integration helpers for AsterScope benchmarks.
"""

from __future__ import annotations

import asyncio
import os
import sys
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from tqdm import tqdm


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_VENDOR_DIR = _PROJECT_ROOT / ".vendor"
if _VENDOR_DIR.exists():
    vendor_path = str(_VENDOR_DIR)
    if vendor_path not in sys.path:
        sys.path.insert(0, vendor_path)


logger = logging.getLogger(__name__)


VALID_CONTEXT_MODES = {"generation", "retrieval"}
DEFAULT_RAGAS_METRICS = (
    "context_precision",
    "context_recall",
    "answer_relevancy",
    "faithfulness",
)
OPTIONAL_RAGAS_METRICS = {
    "context_entities_recall",
    "noise_sensitivity",
}
SUPPORTED_RAGAS_METRICS = set(DEFAULT_RAGAS_METRICS) | OPTIONAL_RAGAS_METRICS


def normalize_context_mode(mode: Optional[str]) -> str:
    normalized = (mode or "generation").strip().lower()
    return normalized if normalized in VALID_CONTEXT_MODES else "generation"


def parse_metric_selection(raw: Optional[str]) -> tuple[str, ...]:
    if not raw or not raw.strip():
        return DEFAULT_RAGAS_METRICS
    selected = []
    for part in raw.split(","):
        metric = part.strip().lower()
        if metric in SUPPORTED_RAGAS_METRICS and metric not in selected:
            selected.append(metric)
    return tuple(selected) if selected else DEFAULT_RAGAS_METRICS


def select_eval_contexts(row: Dict[str, Any], context_mode: str = "generation") -> List[str]:
    mode = normalize_context_mode(context_mode)
    if mode == "retrieval":
        return list(row.get("retrieval_contexts") or [])
    return list(row.get("generation_contexts") or [])


def build_ragas_sample_row(row: Dict[str, Any], context_mode: str = "generation") -> Dict[str, Any]:
    selected_contexts = select_eval_contexts(row, context_mode)
    expected_answers = list(row.get("expected_answers") or [])
    reference_contexts = list(row.get("reference_contexts") or [])
    return {
        "question": row.get("query", ""),
        "answer": row.get("answer", ""),
        "contexts": selected_contexts,
        "ground_truth": expected_answers[0] if expected_answers else None,
        "reference_contexts": reference_contexts,
        "context_mode": normalize_context_mode(context_mode),
        "debug_metrics": dict(row.get("debug_metrics") or {}),
        "source_row": row,
    }


@dataclass
class RagasEvaluationResult:
    sample_rows: List[Dict[str, Any]]
    metrics_summary: Dict[str, Optional[float]]
    metrics_applied: Dict[str, str]
    metrics_skipped: Dict[str, str]


class RagasRunner:
    """
    Runs Ragas metrics against AsterScope benchmark rows while keeping
    AsterScope debug metadata untouched as side-channel diagnostics.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        embedding_model: Optional[str] = None,
        max_concurrency: Optional[int] = None,
    ):
        self.model = model or os.getenv("BENCHMARK_RAGAS_MODEL", "gpt-4.1-mini")
        self.embedding_model = embedding_model or os.getenv(
            "BENCHMARK_RAGAS_EMBEDDING_MODEL",
            "text-embedding-3-small",
        )
        self.max_concurrency = max_concurrency or int(os.getenv("BENCHMARK_RAGAS_CONCURRENCY", "4"))
        self.selected_metrics = parse_metric_selection(os.getenv("BENCHMARK_RAGAS_METRICS"))

        self._imports_loaded = False
        self._ragas_import_error: Optional[Exception] = None
        self._llm = None
        self._embeddings = None
        self._SingleTurnSample = None
        self._metric_classes: Dict[str, Any] = {}

    def is_available(self) -> bool:
        try:
            self._ensure_runtime()
            return True
        except Exception:
            return False

    def _ensure_runtime(self) -> None:
        if self._imports_loaded:
            if self._ragas_import_error:
                raise RuntimeError(
                    "Ragas runtime unavailable. Install benchmark dependencies with "
                    "`pip install -r requirements.txt` after adding `ragas`."
                ) from self._ragas_import_error
            return

        verbose = os.getenv("BENCHMARK_RAGAS_VERBOSE", "false").lower() in {"1", "true", "yes"}
        try:
            if verbose:
                print("Ragas runtime init: importing dependencies", flush=True)
            from langchain_openai import OpenAIEmbeddings as LangchainOpenAIEmbeddings
            from openai import AsyncOpenAI
            from ragas.dataset_schema import SingleTurnSample
            from ragas.embeddings import LangchainEmbeddingsWrapper
            from ragas.llms import llm_factory
            from ragas.metrics import (
                ContextEntityRecall,
                Faithfulness,
                LLMContextPrecisionWithReference,
                LLMContextRecall,
                NoiseSensitivity,
                NonLLMContextPrecisionWithReference,
                NonLLMContextRecall,
                ResponseRelevancy,
            )
        except Exception as exc:  # pragma: no cover - exercised when dependency missing
            self._ragas_import_error = exc
            self._imports_loaded = True
            raise RuntimeError(
                "Ragas is required for benchmark evaluation. "
                "Install it from requirements and retry."
            ) from exc

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            self._ragas_import_error = RuntimeError("OPENAI_API_KEY is required for Ragas evaluation.")
            self._imports_loaded = True
            raise self._ragas_import_error

        if verbose:
            print(f"Ragas runtime init: creating AsyncOpenAI client ({self.model})", flush=True)
        client = AsyncOpenAI(api_key=api_key)
        if verbose:
            print(f"Ragas runtime init: creating llm_factory + embeddings ({self.embedding_model})", flush=True)
        self._llm = llm_factory(self.model, client=client)
        # ResponseRelevancy currently expects a query-style embeddings interface
        # (`embed_query` / `embed_documents`), so we wrap the LangChain embeddings
        # implementation rather than using the newer provider directly.
        self._embeddings = LangchainEmbeddingsWrapper(
            LangchainOpenAIEmbeddings(
                api_key=api_key,
                model=self.embedding_model,
            )
        )
        self._SingleTurnSample = SingleTurnSample
        self._metric_classes = {
            "context_precision_with_reference": LLMContextPrecisionWithReference,
            "context_precision_with_reference_contexts": NonLLMContextPrecisionWithReference,
            "context_recall_with_reference": LLMContextRecall,
            "context_recall_with_reference_contexts": NonLLMContextRecall,
            "context_entities_recall": ContextEntityRecall,
            "faithfulness": Faithfulness,
            "answer_relevancy": ResponseRelevancy,
            "noise_sensitivity": NoiseSensitivity,
        }
        if verbose:
            print("Ragas runtime init: ready", flush=True)
        self._imports_loaded = True

    def _resolve_metric_objects(self, row: Dict[str, Any]) -> tuple[Dict[str, Any], Dict[str, str], Dict[str, str]]:
        self._ensure_runtime()
        applied: Dict[str, Any] = {}
        applied_modes: Dict[str, str] = {}
        skipped: Dict[str, str] = {}

        has_contexts = bool(row.get("contexts"))
        has_answer = bool(str(row.get("answer") or "").strip())
        has_ground_truth = bool(str(row.get("ground_truth") or "").strip())
        has_reference_contexts = bool(row.get("reference_contexts"))

        if "context_precision" in self.selected_metrics and has_contexts and has_ground_truth:
            applied["context_precision"] = self._metric_classes["context_precision_with_reference"](llm=self._llm)
            applied_modes["context_precision"] = "llm_reference"
        elif "context_precision" in self.selected_metrics and has_contexts and has_reference_contexts:
            applied["context_precision"] = self._metric_classes["context_precision_with_reference_contexts"]()
            applied_modes["context_precision"] = "nonllm_reference_contexts"
        elif "context_precision" in self.selected_metrics:
            skipped["context_precision"] = "missing_ground_truth_or_reference_contexts"

        if "context_recall" in self.selected_metrics and has_contexts and has_ground_truth:
            applied["context_recall"] = self._metric_classes["context_recall_with_reference"](llm=self._llm)
            applied_modes["context_recall"] = "llm_reference"
        elif "context_recall" in self.selected_metrics and has_contexts and has_reference_contexts:
            applied["context_recall"] = self._metric_classes["context_recall_with_reference_contexts"]()
            applied_modes["context_recall"] = "nonllm_reference_contexts"
        elif "context_recall" in self.selected_metrics:
            skipped["context_recall"] = "missing_ground_truth_or_reference_contexts"

        if "context_entities_recall" in self.selected_metrics and has_contexts and has_ground_truth:
            applied["context_entities_recall"] = self._metric_classes["context_entities_recall"](llm=self._llm)
            applied_modes["context_entities_recall"] = "llm_reference"
        elif "context_entities_recall" in self.selected_metrics and has_contexts and has_reference_contexts:
            skipped["context_entities_recall"] = "missing_ground_truth_reference_answer"
        elif "context_entities_recall" in self.selected_metrics:
            skipped["context_entities_recall"] = "missing_ground_truth_or_reference_contexts"

        if "faithfulness" in self.selected_metrics and has_contexts and has_answer:
            applied["faithfulness"] = self._metric_classes["faithfulness"](llm=self._llm)
            applied_modes["faithfulness"] = "llm"
        elif "faithfulness" in self.selected_metrics:
            skipped["faithfulness"] = "missing_answer_or_contexts"

        if "answer_relevancy" in self.selected_metrics and has_answer:
            applied["answer_relevancy"] = self._metric_classes["answer_relevancy"](
                llm=self._llm,
                embeddings=self._embeddings,
            )
            applied_modes["answer_relevancy"] = "llm_embeddings"
        elif "answer_relevancy" in self.selected_metrics:
            skipped["answer_relevancy"] = "missing_answer"

        if "noise_sensitivity" in self.selected_metrics and has_contexts and has_answer and has_ground_truth:
            applied["noise_sensitivity"] = self._metric_classes["noise_sensitivity"](llm=self._llm)
            applied_modes["noise_sensitivity"] = "llm_reference"
        elif "noise_sensitivity" in self.selected_metrics:
            skipped["noise_sensitivity"] = "missing_ground_truth_or_disabled"

        return applied, applied_modes, skipped

    async def _evaluate_single(self, row: Dict[str, Any]) -> Dict[str, Any]:
        metric_objects, metric_modes, skipped = self._resolve_metric_objects(row)
        sample = self._SingleTurnSample(
            user_input=row["question"],
            response=row["answer"],
            retrieved_contexts=row["contexts"],
            reference=row.get("ground_truth"),
            reference_contexts=row.get("reference_contexts") or None,
        )

        ragas_scores: Dict[str, Optional[float]] = {}
        verbose = os.getenv("BENCHMARK_RAGAS_VERBOSE", "false").lower() in {"1", "true", "yes"}
        for metric_name, metric in metric_objects.items():
            if verbose:
                print(f"Ragas metric start: {metric_name} :: {row['question'][:120]}", flush=True)
            ragas_scores[metric_name] = float(await metric.single_turn_ascore(sample))
            if verbose:
                print(f"Ragas metric done: {metric_name} = {ragas_scores[metric_name]:.4f}", flush=True)
        for metric_name in skipped:
            ragas_scores.setdefault(metric_name, None)

        return {
            **row["source_row"],
            "ragas_context_mode": row["context_mode"],
            "ragas_scores": ragas_scores,
            "ragas_metric_modes": metric_modes,
            "ragas_metric_skips": skipped,
            "ragas_contexts": list(row["contexts"]),
            "ragas_ground_truth": row.get("ground_truth"),
            "ragas_reference_contexts": list(row.get("reference_contexts") or []),
        }

    async def evaluate_rows(
        self,
        query_results: List[Dict[str, Any]],
        context_mode: str = "generation",
    ) -> RagasEvaluationResult:
        rows = [build_ragas_sample_row(row, context_mode=context_mode) for row in query_results]
        semaphore = asyncio.Semaphore(max(1, self.max_concurrency))
        scored_rows: List[Optional[Dict[str, Any]]] = [None] * len(rows)
        progress = tqdm(total=len(rows), desc="Ragas Eval", leave=True)

        async def _run(index: int, row: Dict[str, Any]) -> None:
            async with semaphore:
                try:
                    scored_rows[index] = await self._evaluate_single(row)
                finally:
                    progress.update(1)

        try:
            await asyncio.gather(*[_run(index, row) for index, row in enumerate(rows)])
        finally:
            progress.close()
        final_rows = [row for row in scored_rows if row is not None]

        metrics_summary: Dict[str, Optional[float]] = {}
        metrics_applied: Dict[str, str] = {}
        metrics_skipped: Dict[str, str] = {}
        for metric_name in self.selected_metrics:
            values = [
                float(row["ragas_scores"][metric_name])
                for row in final_rows
                if row.get("ragas_scores", {}).get(metric_name) is not None
            ]
            metrics_summary[metric_name] = (sum(values) / len(values)) if values else None
            mode_values = {
                str((row.get("ragas_metric_modes") or {}).get(metric_name))
                for row in final_rows
                if (row.get("ragas_metric_modes") or {}).get(metric_name)
            }
            if mode_values:
                metrics_applied[metric_name] = ",".join(sorted(mode_values))
            else:
                skip_values = {
                    str((row.get("ragas_metric_skips") or {}).get(metric_name))
                    for row in final_rows
                    if (row.get("ragas_metric_skips") or {}).get(metric_name)
                }
                if skip_values:
                    metrics_skipped[metric_name] = ",".join(sorted(skip_values))

        return RagasEvaluationResult(
            sample_rows=final_rows,
            metrics_summary=metrics_summary,
            metrics_applied=metrics_applied,
            metrics_skipped=metrics_skipped,
        )
