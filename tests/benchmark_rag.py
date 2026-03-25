"""
Comprehensive benchmark runner for NovaSearch.

Supports:
- HotpotQA (page-supervised, multi-hop)
- Natural Questions via BeIR/NQ (corpus + queries + qrels)
"""

import asyncio
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import httpx
from dotenv import load_dotenv
from tqdm import tqdm

try:
    from tests.benchmark_support import (
        BenchmarkBundle,
        BenchmarkCase,
        calc_hit_rate_at_k,
        calc_mrr_at_k,
        default_split_for_benchmark,
        default_trace_path,
        exact_match_score,
        load_benchmark_bundle,
        normalize_benchmark_name,
        resolve_eval_every_n,
        resolve_generation_eval_enabled,
        resolve_sample_size,
        token_f1_score,
    )
except ImportError:
    from benchmark_support import (
        BenchmarkBundle,
        BenchmarkCase,
        calc_hit_rate_at_k,
        calc_mrr_at_k,
        default_split_for_benchmark,
        default_trace_path,
        exact_match_score,
        load_benchmark_bundle,
        normalize_benchmark_name,
        resolve_eval_every_n,
        resolve_generation_eval_enabled,
        resolve_sample_size,
        token_f1_score,
    )

load_dotenv()
logger = logging.getLogger(__name__)

# Reduce httpx/openai noise
logging.getLogger("httpx").setLevel(logging.WARNING)

try:
    import openai
except ImportError:
    openai = None

try:
    import psycopg2
except ImportError:
    psycopg2 = None

try:
    from neo4j import GraphDatabase
except ImportError:
    GraphDatabase = None


class RAGEvaluator:
    """Uses LLM-as-Judge to evaluate grounded-answer metrics."""

    def __init__(self, model: str | None = None):
        self.model = model or os.getenv("BENCHMARK_EVAL_MODEL", "gpt-4.1-mini")
        self.client = None
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key and openai:
            self.client = openai.OpenAI(api_key=api_key)

    async def _await_llm_json(self, sys_prompt: str, user_prompt: str) -> Dict[str, Any]:
        if not self.client:
            raise RuntimeError("Generation evaluation requested without a configured OpenAI client.")
        loop = asyncio.get_running_loop()

        def _call():
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.0,
                    max_tokens=512,
                )
                return json.loads(resp.choices[0].message.content)
            except Exception as exc:
                raise RuntimeError(f"Judge model call failed: {exc}") from exc

        return await loop.run_in_executor(None, _call)

    async def faithfulness(self, answer: str, context: str) -> float:
        sys = (
            "You evaluate factual consistency. Given context and an answer, "
            "score 0.0-1.0 how well the answer is supported by the context. "
            "Return JSON {'score': float}."
        )
        payload = await self._await_llm_json(sys, f"Context:\n{context}\n\nAnswer:\n{answer}")
        return float(payload.get("score", 0.0))

    async def context_precision(self, query: str, contexts: List[str], benchmark_name: str) -> float:
        if benchmark_name == "hotpotqa":
            sys = (
                "You evaluate retrieval context precision for multi-hop question answering.\n"
                "Score from 0.0 to 1.0 based on what fraction of the provided chunks are actually relevant.\n"
                "IMPORTANT: A chunk counts as relevant if it either:\n"
                "1. directly supports the final answer, OR\n"
                "2. provides a necessary bridge entity, role, alias, date, title, or relationship needed for an intermediate hop.\n"
                "Do NOT penalize a chunk just because it is intermediate rather than final-answer evidence.\n"
                "Return JSON {'score': float}."
            )
        else:
            sys = (
                "You evaluate retrieval context precision for grounded open-domain question answering.\n"
                "Score from 0.0 to 1.0 based on what fraction of the provided chunks are actually useful for answering the query.\n"
                "A chunk counts as relevant if it either directly supports the answer or materially narrows the right entity/page needed for the answer.\n"
                "Return JSON {'score': float}."
            )
        formatted_chunks = "\n".join(f"Chunk {idx + 1}:\n{chunk}" for idx, chunk in enumerate(contexts))
        payload = await self._await_llm_json(sys, f"Query:\n{query}\n\nContexts:\n{formatted_chunks}")
        return float(payload.get("score", 0.0))

    async def context_precision_audit(
        self,
        query: str,
        contexts: List[str],
        benchmark_name: str,
    ) -> Dict[str, Any]:
        if not contexts:
            return {"precision": 0.0, "counts": {}, "chunks": []}

        if benchmark_name == "hotpotqa":
            sys = (
                "You audit retrieval chunks for a multi-hop QA system.\n"
                "Classify EACH chunk into exactly one label:\n"
                "- direct-answer: directly needed to answer the query\n"
                "- bridge-useful: needed as an intermediate hop or bridge entity/fact\n"
                "- background: topically related but not necessary to answer the query\n"
                "- noise: irrelevant, misleading, or off-target\n"
                "Return JSON with this shape:\n"
                "{\"chunks\": [{\"index\": 1, \"label\": \"direct-answer|bridge-useful|background|noise\", \"reason\": \"short reason\"}]}\n"
                "Do not skip chunks."
            )
        else:
            sys = (
                "You audit retrieval chunks for a grounded open-domain QA system.\n"
                "Classify EACH chunk into exactly one label:\n"
                "- direct-answer: directly needed to answer the query\n"
                "- bridge-useful: useful for locking onto the right entity/page even if not the final answer chunk\n"
                "- background: topically related but not necessary to answer the query\n"
                "- noise: irrelevant, misleading, or off-target\n"
                "Return JSON with this shape:\n"
                "{\"chunks\": [{\"index\": 1, \"label\": \"direct-answer|bridge-useful|background|noise\", \"reason\": \"short reason\"}]}\n"
                "Do not skip chunks."
            )
        formatted_chunks = "\n".join(f"Chunk {idx + 1}:\n{chunk}" for idx, chunk in enumerate(contexts))
        payload = await self._await_llm_json(sys, f"Query:\n{query}\n\nContexts:\n{formatted_chunks}")

        raw_chunks = payload.get("chunks", []) if isinstance(payload, dict) else []
        audited_chunks = []
        counts = {
            "direct-answer": 0,
            "bridge-useful": 0,
            "background": 0,
            "noise": 0,
        }
        for idx, chunk in enumerate(contexts, start=1):
            matched = None
            for candidate in raw_chunks:
                if int(candidate.get("index", -1)) == idx:
                    matched = candidate
                    break
            label = (matched or {}).get("label", "noise")
            if label not in counts:
                label = "noise"
            counts[label] += 1
            audited_chunks.append(
                {
                    "index": idx,
                    "label": label,
                    "reason": (matched or {}).get("reason", ""),
                    "chunk": chunk,
                }
            )

        precision = (counts["direct-answer"] + counts["bridge-useful"]) / max(1, len(contexts))
        return {
            "precision": precision,
            "counts": counts,
            "chunks": audited_chunks,
        }


def _compose_retrieval_context(source: Dict[str, Any]) -> str:
    graph_context = source.get("graph_context") or {}
    title = source.get("title") or graph_context.get("doc_title") or ""
    chunk_text = source.get("chunk_text", "") or ""
    graph_title = graph_context.get("doc_title") or ""
    parts = [part for part in [title, graph_title, chunk_text] if part]
    return "\n".join(parts)


def _source_chunk_key(source: Dict[str, Any]) -> Tuple[str, str]:
    doc_id = str(source.get("doc_id") or "")
    chunk_index = str(source.get("chunk_index") if source.get("chunk_index") is not None else "")
    return doc_id, chunk_index


def _is_planner_confirmed_source(source: Dict[str, Any], debug_metrics: Dict[str, Any]) -> bool:
    planner_queries = {
        str(query).strip().lower()
        for query in (debug_metrics or {}).get("bridge_queries", [])
        if str(query).strip()
    }
    if not planner_queries:
        return False
    retrieval_queries = {
        str(query).strip().lower()
        for query in (source.get("retrieval_queries") or [])
        if str(query).strip()
    }
    return bool(retrieval_queries & planner_queries)


def _mean(values: List[float]) -> Optional[float]:
    return (sum(values) / len(values)) if values else None


def _aggregate_audit_counts(audits: List[Dict[str, Any]]) -> Dict[str, int]:
    totals = {
        "direct-answer": 0,
        "bridge-useful": 0,
        "background": 0,
        "noise": 0,
    }
    for audit in audits:
        counts = (audit or {}).get("counts") or {}
        for label in totals:
            totals[label] += int(counts.get(label, 0) or 0)
    return totals


def _context_titles(contexts: List[str]) -> List[str]:
    titles = []
    for context in contexts:
        if not context:
            continue
        titles.append(context.split("\n", 1)[0].strip())
    return titles


def _duplicate_title_ratio(contexts: List[str]) -> float:
    titles = [title.lower() for title in _context_titles(contexts) if title]
    duplicate_count = max(0, len(titles) - len(set(titles)))
    return duplicate_count / max(1, len(titles))


def _benchmark_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.lower() in {"1", "true", "yes"}


def reset_benchmark_stores():
    """Best-effort cleanup so repeated benchmark runs don't stack stale state."""
    graph_enabled = (
        os.getenv("ENABLE_GRAPH_INGESTION", "false").lower() in {"1", "true", "yes"}
        or os.getenv("ENABLE_GRAPH_RETRIEVAL", "false").lower() in {"1", "true", "yes"}
    )
    pg_dsn = os.getenv(
        "DATABASE_URL",
        f"dbname={os.getenv('POSTGRES_DB', 'novasearch')} "
        f"user={os.getenv('POSTGRES_USER', 'postgres')} "
        f"password={os.getenv('POSTGRES_PASSWORD', 'postgres_secure_password')} "
        f"host={os.getenv('POSTGRES_HOST', 'localhost')} "
        f"port={os.getenv('POSTGRES_PORT', '5432')}"
    )

    if psycopg2:
        conn = None
        try:
            conn = psycopg2.connect(pg_dsn)
            conn.autocommit = True
            with conn.cursor() as cur:
                cur.execute("TRUNCATE TABLE vision_embeddings RESTART IDENTITY CASCADE;")
                cur.execute("TRUNCATE TABLE chunks RESTART IDENTITY CASCADE;")
                cur.execute("TRUNCATE TABLE documents RESTART IDENTITY CASCADE;")
            print("Reset PostgreSQL benchmark tables.")
        except Exception as exc:
            print(f"Warning: PostgreSQL reset skipped: {exc}")
        finally:
            if conn:
                conn.close()

    if graph_enabled and GraphDatabase:
        driver = None
        try:
            driver = GraphDatabase.driver(
                os.getenv("NEO4J_URI", "bolt://localhost:7687"),
                auth=(
                    os.getenv("NEO4J_USER", "neo4j"),
                    os.getenv("NEO4J_PASSWORD", "neo4j_secure_password"),
                ),
            )
            with driver.session() as session:
                session.run("MATCH (n) DETACH DELETE n")
            print("Reset Neo4j benchmark graph.")
        except Exception as exc:
            print(f"Warning: Neo4j reset skipped: {exc}")
        finally:
            if driver:
                driver.close()


async def ingest_page_document(
    client,
    base_url,
    headers,
    title: str,
    doc_text: str,
    section: str,
    idx: int,
    semaphore,
    pbar,
    error_list,
):
    async with semaphore:
        try:
            resp = await client.post(
                f"{base_url}/ingest",
                data={"text_input": doc_text, "title": title, "section": section},
                headers=headers,
            )
            if resp.status_code != 200:
                error_list.append(f"Title '{title}' Failed: Code {resp.status_code}")
        except Exception as e:
            error_list.append(f"Title '{title}' Error: {str(e)}")
        finally:
            pbar.update(1)


async def evaluate_query(
    client,
    base_url,
    headers,
    case: BenchmarkCase,
    evaluator: RAGEvaluator,
    semaphore,
    pbar,
    results,
    benchmark_name: str,
    generation_eval_enabled: bool,
    eval_every_n: int,
):
    async with semaphore:
        query = case.query
        expected_titles = case.expected_titles
        expected_answers = case.expected_answers
        retrieval_contexts: List[str] = []
        generation_contexts: List[str] = []
        supporting_contexts: List[str] = []

        mrr = 0.0
        hr5 = 0.0
        retrieved_tokens = 0.0
        faith_score = 0.0
        legacy_cp_score = 0.0
        generation_cp_score = 0.0
        supporting_cp_score = 0.0
        answer_em = None
        answer_f1 = None
        answer = ""
        debug_metrics: Dict[str, Any] = {}
        generation_audit = None
        supporting_audit = None
        eval_attempted = False
        eval_invalid_reason = None
        retrieval_sources_raw: List[Dict[str, Any]] = []
        generation_sources_raw: List[Dict[str, Any]] = []
        supporting_sources_raw: List[Dict[str, Any]] = []

        try:
            resp = await client.post(
                f"{base_url}/ask",
                json={"query": query, "top_k": 5},
                headers=headers,
            )

            if resp.status_code == 200:
                for line in resp.text.strip().split("\n"):
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except Exception:
                        continue
                    if data.get("type") == "token":
                        answer += data.get("content", "")
                    elif data.get("type") == "answer_metadata":
                        retrieval_sources = data.get("retrieval_contexts") or data.get("sources", [])
                        generation_sources = data.get("generation_contexts") or data.get("sources", [])
                        supporting_sources = data.get("sources", [])
                        debug_metrics = data.get("debug_metrics") or {}
                        retrieval_sources_raw = retrieval_sources
                        generation_sources_raw = generation_sources
                        supporting_sources_raw = supporting_sources
                        retrieval_contexts = [_compose_retrieval_context(s) for s in retrieval_sources]
                        generation_contexts = [_compose_retrieval_context(s) for s in generation_sources]
                        supporting_contexts = [_compose_retrieval_context(s) for s in supporting_sources]

            mrr = calc_mrr_at_k(retrieval_contexts, expected_titles, k=5)
            hr5 = calc_hit_rate_at_k(retrieval_contexts, expected_titles, k=5)
            retrieved_tokens = sum(len(c.split()) * 1.3 for c in generation_contexts)

            if expected_answers:
                normalized_answer = (answer or "").strip()
                answer_em = exact_match_score(normalized_answer, expected_answers)
                answer_f1 = token_f1_score(normalized_answer, expected_answers)

            eval_contexts = supporting_contexts or generation_contexts
            should_eval_generation = generation_eval_enabled and (case.idx % eval_every_n == 0)
            normalized_answer = (answer or "").strip()
            if normalized_answer == "Benchmark retrieval completed.":
                eval_invalid_reason = "placeholder_benchmark_answer"
            elif should_eval_generation and eval_contexts and answer:
                eval_attempted = True
                context_str = "\n".join(eval_contexts)
                try:
                    faith_score = await evaluator.faithfulness(answer, context_str)
                    legacy_cp_score = await evaluator.context_precision(query, eval_contexts, benchmark_name)
                    if generation_contexts:
                        generation_audit = await evaluator.context_precision_audit(
                            query,
                            generation_contexts,
                            benchmark_name,
                        )
                        generation_cp_score = float(generation_audit.get("precision", 0.0))
                    if supporting_contexts:
                        supporting_audit = await evaluator.context_precision_audit(
                            query,
                            supporting_contexts,
                            benchmark_name,
                        )
                        supporting_cp_score = float(supporting_audit.get("precision", 0.0))
                except Exception as eval_exc:
                    logger.warning("Generation evaluation skipped for sample %s: %s", case.idx, eval_exc)
                    faith_score = 0.0
                    legacy_cp_score = 0.0
                    generation_cp_score = 0.0
                    supporting_cp_score = 0.0
                    generation_audit = None
                    supporting_audit = None
                    eval_attempted = False
        except Exception:
            retrieval_contexts = []
            generation_contexts = []
            supporting_contexts = []
        finally:
            generation_key_set = {
                _source_chunk_key(source)
                for source in generation_sources_raw
                if isinstance(source, dict)
            }
            supporting_key_set = {
                _source_chunk_key(source)
                for source in supporting_sources_raw
                if isinstance(source, dict)
            }
            planner_confirmed_generation_count = sum(
                1
                for source in generation_sources_raw
                if isinstance(source, dict) and _is_planner_confirmed_source(source, debug_metrics)
            )
            supporting_selected_generation_count = len(generation_key_set & supporting_key_set)
            generation_planner_chunk_ratio = (
                planner_confirmed_generation_count / max(1, len(generation_sources_raw))
            ) if generation_sources_raw else 0.0
            planner_rescued_generation_missed = any(
                isinstance(source, dict)
                and _is_planner_confirmed_source(source, debug_metrics)
                and _source_chunk_key(source) not in generation_key_set
                for source in supporting_sources_raw
            )
            bridge_targeting_hits = int(debug_metrics.get("bridge_targeting_hits", 0) or 0)
            answer_bearing_bridge_hits = int(debug_metrics.get("answer_bearing_bridge_hits", 0) or 0)
            bridge_entity_family_chunk_miss = int(debug_metrics.get("bridge_entity_family_chunk_miss", 0) or 0)
            alias_bridge_candidate_count = int(debug_metrics.get("alias_bridge_candidate_count", 0) or 0)
            bridge_targeted_query_count = int(debug_metrics.get("bridge_targeted_query_count", 0) or 0)
            duplicates_removed = int(debug_metrics.get("duplicates_removed", 0) or 0)
            bridge_coverage_score = float(debug_metrics.get("bridge_coverage_score", 0.0) or 0.0)
            support_coverage_score = float(debug_metrics.get("support_coverage_score", 0.0) or 0.0)
            source_type_mix = debug_metrics.get("source_type_mix") or {}
            evidence_role_mix = debug_metrics.get("evidence_role_mix") or {}
            results.append(
                {
                    "mrr": mrr,
                    "hr5": hr5,
                    "retrieved_tokens": retrieved_tokens,
                    "faith_score": faith_score,
                    "legacy_cp_score": legacy_cp_score,
                    "generation_cp_score": generation_cp_score,
                    "supporting_cp_score": supporting_cp_score,
                    "generation_audit": generation_audit,
                    "supporting_audit": supporting_audit,
                    "was_evaled": eval_attempted,
                    "eval_invalid_reason": eval_invalid_reason,
                    "answer": answer,
                    "answer_em": answer_em,
                    "answer_f1": answer_f1,
                    "retrieval_contexts": retrieval_contexts,
                    "generation_contexts": generation_contexts,
                    "supporting_contexts": supporting_contexts,
                    "expected_titles": expected_titles,
                    "expected_answers": expected_answers,
                    "query": query,
                    "debug_metrics": debug_metrics,
                    "planner_confirmed_generation_count": planner_confirmed_generation_count,
                    "supporting_selected_generation_count": supporting_selected_generation_count,
                    "generation_planner_chunk_ratio": generation_planner_chunk_ratio,
                    "planner_rescued_generation_missed": planner_rescued_generation_missed,
                    "generation_sources_raw": generation_sources_raw,
                    "supporting_sources_raw": supporting_sources_raw,
                    "bridge_targeting_hits": bridge_targeting_hits,
                    "answer_bearing_bridge_hits": answer_bearing_bridge_hits,
                    "bridge_entity_family_chunk_miss": bridge_entity_family_chunk_miss,
                    "alias_bridge_candidate_count": alias_bridge_candidate_count,
                    "bridge_targeted_query_count": bridge_targeted_query_count,
                    "duplicates_removed": duplicates_removed,
                    "bridge_coverage_score": bridge_coverage_score,
                    "support_coverage_score": support_coverage_score,
                    "source_type_mix": source_type_mix,
                    "evidence_role_mix": evidence_role_mix,
                }
            )
            pbar.update(1)


def build_summary_payload(
    benchmark: BenchmarkBundle,
    query_results: List[Dict[str, Any]],
    total_retrieved_tokens: float,
    ingest_duration: float,
    query_duration: float,
    generation_eval_enabled: bool,
    evaluator_model: Optional[str],
    eval_every_n: int,
) -> Dict[str, Any]:
    n = len(query_results)
    raw_mrr = sum(r["mrr"] for r in query_results)
    raw_hr5 = sum(r["hr5"] for r in query_results)
    debug_rows = [r.get("debug_metrics") or {} for r in query_results]
    second_hop_trigger_rate = sum(1 for row in debug_rows if row.get("second_hop_triggered")) / max(1, n)
    avg_first_hop_candidates = sum((row.get("first_hop_candidates") or 0) for row in debug_rows) / max(1, n)
    avg_merged_candidates = sum((row.get("merged_candidate_count") or 0) for row in debug_rows) / max(1, n)
    avg_final_context_count = sum((row.get("final_context_count") or 0) for row in debug_rows) / max(1, n)
    avg_bridge_query_count = sum((row.get("bridge_query_count") or 0) for row in debug_rows) / max(1, n)
    avg_pre_pack_count = sum((row.get("pre_pack_count") or 0) for row in debug_rows) / max(1, n)
    avg_post_pack_count = sum((row.get("post_pack_count") or 0) for row in debug_rows) / max(1, n)
    total_second_hop_added = sum((row.get("second_hop_added_count") or 0) for row in debug_rows)
    total_merged_candidates = sum((row.get("merged_candidate_count") or 0) for row in debug_rows)
    second_hop_new_evidence_ratio = total_second_hop_added / max(1, total_merged_candidates)
    avg_duplicates_removed = _mean([float(r.get("duplicates_removed", 0)) for r in query_results]) or 0.0
    avg_bridge_coverage_score = _mean([float(r.get("bridge_coverage_score", 0.0)) for r in query_results]) or 0.0
    avg_support_coverage_score = _mean([float(r.get("support_coverage_score", 0.0)) for r in query_results]) or 0.0
    total_graph_symbolic_candidates = sum(int(row.get("graph_symbolic_candidates", 0) or 0) for row in debug_rows)
    total_graph_symbolic_kept = sum(int(row.get("graph_symbolic_kept", 0) or 0) for row in debug_rows)
    total_graph_symbolic_dropped = sum(int(row.get("graph_symbolic_dropped", 0) or 0) for row in debug_rows)
    total_corroborated_graph_kept = sum(int(row.get("corroborated_graph_kept", 0) or 0) for row in debug_rows)
    generation_compacted_total = sum(int(row.get("generation_compacted_count", 0) or 0) for row in debug_rows)
    source_type_mix_totals: Dict[str, int] = {}
    evidence_role_mix_totals: Dict[str, int] = {}
    for row in query_results:
        for key, value in (row.get("source_type_mix") or {}).items():
            source_type_mix_totals[key] = source_type_mix_totals.get(key, 0) + int(value or 0)
        for key, value in (row.get("evidence_role_mix") or {}).items():
            evidence_role_mix_totals[key] = evidence_role_mix_totals.get(key, 0) + int(value or 0)

    evaled = [r for r in query_results if r["was_evaled"]]
    eval_n = len(evaled)
    invalid_eval_rows = [r for r in query_results if r.get("eval_invalid_reason")]
    invalid_eval_reasons: Dict[str, int] = {}
    for row in invalid_eval_rows:
        reason = row.get("eval_invalid_reason") or "unknown"
        invalid_eval_reasons[reason] = invalid_eval_reasons.get(reason, 0) + 1

    answer_rows = [r for r in query_results if r.get("answer_em") is not None]
    answer_em = _mean([float(r["answer_em"]) for r in answer_rows]) if answer_rows else None
    answer_f1 = _mean([float(r["answer_f1"]) for r in answer_rows]) if answer_rows else None
    faithfulness_metric = _mean([r["faith_score"] for r in evaled]) if generation_eval_enabled else None
    legacy_context_precision_metric = _mean(
        [r["legacy_cp_score"] for r in evaled if r["legacy_cp_score"] is not None]
    ) if generation_eval_enabled else None
    generation_context_precision_metric = _mean(
        [r["generation_cp_score"] for r in evaled if r.get("generation_audit")]
    ) if generation_eval_enabled else None
    supporting_context_precision_metric = _mean(
        [r["supporting_cp_score"] for r in evaled if r.get("supporting_audit")]
    ) if generation_eval_enabled else None

    generation_audits = [r["generation_audit"] for r in evaled if r.get("generation_audit")]
    supporting_audits = [r["supporting_audit"] for r in evaled if r.get("supporting_audit")]
    generation_label_counts = _aggregate_audit_counts(generation_audits)
    supporting_label_counts = _aggregate_audit_counts(supporting_audits)
    avg_generation_context_count = _mean([len(r["generation_contexts"]) for r in query_results]) or 0.0
    avg_generation_duplicate_title_ratio = _mean(
        [_duplicate_title_ratio(r["generation_contexts"]) for r in query_results if r["generation_contexts"]]
    ) or 0.0
    avg_planner_confirmed_generation_count = _mean(
        [float(r.get("planner_confirmed_generation_count", 0)) for r in query_results]
    ) or 0.0
    avg_supporting_selected_generation_count = _mean(
        [float(r.get("supporting_selected_generation_count", 0)) for r in query_results]
    ) or 0.0
    generation_planner_chunk_ratio_metric = _mean(
        [float(r.get("generation_planner_chunk_ratio", 0.0)) for r in query_results]
    ) or 0.0
    blocked_seed_count = sum(
        int((r.get("debug_metrics") or {}).get("blocked_seed_count", 0) or 0)
        for r in query_results
    )
    supporting_anchor_filtered_count = sum(
        int((r.get("debug_metrics") or {}).get("supporting_anchor_filtered_count", 0) or 0)
        for r in query_results
    )
    planner_rescued_generation_missed_cases = sum(
        1 for r in query_results if r.get("planner_rescued_generation_missed")
    )
    seed_error_degradation_cases = sum(
        1
        for r in evaled
        if int((r.get("debug_metrics") or {}).get("seed_mismatch_count", 0) or 0) > 0
        and r.get("supporting_audit")
        and r.get("generation_audit")
        and (r["supporting_cp_score"] - r["generation_cp_score"]) >= 0.2
    )
    total_generation_labeled = sum(generation_label_counts.values())
    generation_background_ratio = generation_label_counts["background"] / max(1, total_generation_labeled)
    generation_noise_ratio = generation_label_counts["noise"] / max(1, total_generation_labeled)
    support_selector_worse_cases = sum(
        1
        for r in evaled
        if r.get("generation_audit")
        and r.get("supporting_audit")
        and (r["generation_cp_score"] - r["supporting_cp_score"]) >= 0.2
    )
    generation_assembly_worse_cases = sum(
        1
        for r in evaled
        if r.get("generation_audit")
        and r.get("supporting_audit")
        and (r["supporting_cp_score"] - r["generation_cp_score"]) >= 0.2
    )
    both_contexts_noisy_cases = sum(
        1
        for r in evaled
        if r.get("generation_audit")
        and r.get("supporting_audit")
        and r["generation_cp_score"] <= 0.5
        and r["supporting_cp_score"] <= 0.5
    )
    supporting_wrong_page_family_cases = sum(
        1
        for r in evaled
        if int((r.get("debug_metrics") or {}).get("final_supporting_unanchored_count", 0) or 0) > 0
        and (r.get("supporting_cp_score") or 0.0) <= 0.5
    )
    supporting_entity_right_generation_failed_cases = sum(
        1
        for r in evaled
        if (r.get("supporting_cp_score") or 0.0) >= 0.8
        and (r.get("generation_cp_score") or 0.0) <= ((r.get("supporting_cp_score") or 0.0) - 0.2)
    )
    planner_bridge_targeting_problem_cases = sum(
        1
        for r in evaled
        if (r.get("supporting_cp_score") or 0.0) <= 0.5
        and (r.get("generation_cp_score") or 0.0) <= 0.5
        and int((r.get("debug_metrics") or {}).get("final_supporting_unanchored_count", 0) or 0) == 0
    )
    bridge_targeting_hit_cases = sum(1 for r in query_results if int(r.get("bridge_targeting_hits", 0) or 0) > 0)
    alias_identity_resolution_improved_cases = sum(
        1
        for r in query_results
        if int(r.get("alias_bridge_candidate_count", 0) or 0) > 0
        and int(r.get("answer_bearing_bridge_hits", 0) or 0) > 0
    )
    page_family_correct_chunk_wrong_cases = sum(
        1
        for r in query_results
        if int(r.get("bridge_targeting_hits", 0) or 0) > 0
        and int(r.get("answer_bearing_bridge_hits", 0) or 0) == 0
    )
    planner_bridge_improved_generation_lag_cases = sum(
        1
        for r in evaled
        if int(r.get("bridge_targeting_hits", 0) or 0) > 0
        and (r.get("supporting_cp_score") or 0.0) >= 0.7
        and (r.get("generation_cp_score") or 0.0) <= ((r.get("supporting_cp_score") or 0.0) - 0.2)
    )

    ratio = benchmark.total_raw_tokens / max(1, total_retrieved_tokens)
    savings = (1 - (total_retrieved_tokens / max(1, benchmark.total_raw_tokens))) * 100

    return {
        "benchmark_name": benchmark.name,
        "benchmark_display_name": benchmark.display_name,
        "benchmark_split": benchmark.split,
        "benchmark_notes": benchmark.notes,
        "sample_size": n,
        "unique_pages_ingested": len(benchmark.unique_pages),
        "retrieval_metric_label": benchmark.retrieval_metric_label,
        "answer_metric_label": benchmark.answer_metric_label,
        "hit_rate_5": raw_hr5 / max(1, n),
        "mrr_5": raw_mrr / max(1, n),
        "answer_em": answer_em,
        "answer_f1": answer_f1,
        "faithfulness": faithfulness_metric,
        "context_precision": legacy_context_precision_metric,
        "legacy_context_precision": legacy_context_precision_metric,
        "generation_context_precision": generation_context_precision_metric,
        "supporting_context_precision": supporting_context_precision_metric,
        "cost_savings_pct": savings,
        "noise_reduction_ratio": ratio,
        "second_hop_trigger_rate": second_hop_trigger_rate,
        "avg_first_hop_candidates": avg_first_hop_candidates,
        "avg_merged_candidates": avg_merged_candidates,
        "avg_pre_pack_count": avg_pre_pack_count,
        "avg_post_pack_count": avg_post_pack_count,
        "avg_final_context_count": avg_final_context_count,
        "avg_bridge_query_count": avg_bridge_query_count,
        "second_hop_new_evidence_ratio": second_hop_new_evidence_ratio,
        "avg_duplicates_removed": avg_duplicates_removed,
        "avg_bridge_coverage_score": avg_bridge_coverage_score,
        "avg_support_coverage_score": avg_support_coverage_score,
        "graph_symbolic_candidates": total_graph_symbolic_candidates,
        "graph_symbolic_kept": total_graph_symbolic_kept,
        "graph_symbolic_dropped": total_graph_symbolic_dropped,
        "corroborated_graph_kept": total_corroborated_graph_kept,
        "source_type_mix_totals": source_type_mix_totals,
        "evidence_role_mix_totals": evidence_role_mix_totals,
        "generation_eval_case_count": eval_n,
        "invalid_generation_eval_cases": invalid_eval_reasons,
        "generation_eval_every_n": eval_every_n,
        "support_selector_worse_cases": support_selector_worse_cases,
        "generation_assembly_worse_cases": generation_assembly_worse_cases,
        "both_contexts_noisy_cases": both_contexts_noisy_cases,
        "avg_generation_context_count": avg_generation_context_count,
        "avg_generation_duplicate_title_ratio": avg_generation_duplicate_title_ratio,
        "generation_background_ratio": generation_background_ratio,
        "generation_noise_ratio": generation_noise_ratio,
        "avg_planner_confirmed_generation_count": avg_planner_confirmed_generation_count,
        "avg_supporting_selected_generation_count": avg_supporting_selected_generation_count,
        "generation_planner_chunk_ratio": generation_planner_chunk_ratio_metric,
        "generation_compacted_total": generation_compacted_total,
        "blocked_seed_count": blocked_seed_count,
        "supporting_anchor_filtered_count": supporting_anchor_filtered_count,
        "planner_rescued_generation_missed_cases": planner_rescued_generation_missed_cases,
        "seed_error_degradation_cases": seed_error_degradation_cases,
        "supporting_wrong_page_family_cases": supporting_wrong_page_family_cases,
        "supporting_entity_right_generation_failed_cases": supporting_entity_right_generation_failed_cases,
        "planner_bridge_targeting_problem_cases": planner_bridge_targeting_problem_cases,
        "bridge_targeting_hit_cases": bridge_targeting_hit_cases,
        "alias_identity_resolution_improved_cases": alias_identity_resolution_improved_cases,
        "page_family_correct_chunk_wrong_cases": page_family_correct_chunk_wrong_cases,
        "planner_bridge_improved_generation_lag_cases": planner_bridge_improved_generation_lag_cases,
        "generation_label_counts": generation_label_counts,
        "supporting_label_counts": supporting_label_counts,
        "raw_token_total": benchmark.total_raw_tokens,
        "retrieved_token_total": total_retrieved_tokens,
        "ingestion_duration_seconds": ingest_duration,
        "query_duration_seconds": query_duration,
        "generation_eval_enabled": generation_eval_enabled,
        "generation_eval_model": evaluator_model if generation_eval_enabled else None,
        "benchmark_mode": os.getenv("NOVASEARCH_BENCHMARK_MODE", "false").lower() in {"1", "true", "yes"},
        "graph_ingestion_enabled": os.getenv("ENABLE_GRAPH_INGESTION", "false").lower() in {"1", "true", "yes"},
        "graph_retrieval_enabled": os.getenv("ENABLE_GRAPH_RETRIEVAL", "false").lower() in {"1", "true", "yes"},
        "vision_retriever_enabled": os.getenv("ENABLE_VISION_RETRIEVER", "true").lower() in {"1", "true", "yes"},
        "text_vision_indexing_enabled": os.getenv("ENABLE_TEXT_VISION_INDEXING", "true").lower() in {"1", "true", "yes"},
        "sampled_generation_cases": [
            {
                "query": r["query"],
                "answer": r["answer"],
                "expected_titles": r["expected_titles"],
                "expected_answers": r["expected_answers"],
                "retrieval_contexts": r["retrieval_contexts"],
                "generation_contexts": r["generation_contexts"],
                "supporting_contexts": r["supporting_contexts"],
                "answer_em": r["answer_em"],
                "answer_f1": r["answer_f1"],
                "faith_score": r["faith_score"],
                "legacy_cp_score": r["legacy_cp_score"],
                "generation_cp_score": r["generation_cp_score"],
                "supporting_cp_score": r["supporting_cp_score"],
                "planner_confirmed_generation_count": r["planner_confirmed_generation_count"],
                "supporting_selected_generation_count": r["supporting_selected_generation_count"],
                "generation_planner_chunk_ratio": r["generation_planner_chunk_ratio"],
                "planner_rescued_generation_missed": r["planner_rescued_generation_missed"],
                "bridge_targeting_hits": r["bridge_targeting_hits"],
                "answer_bearing_bridge_hits": r["answer_bearing_bridge_hits"],
                "bridge_entity_family_chunk_miss": r["bridge_entity_family_chunk_miss"],
                "alias_bridge_candidate_count": r["alias_bridge_candidate_count"],
                "bridge_targeted_query_count": r["bridge_targeted_query_count"],
                "duplicates_removed": r["duplicates_removed"],
                "bridge_coverage_score": r["bridge_coverage_score"],
                "support_coverage_score": r["support_coverage_score"],
                "source_type_mix": r["source_type_mix"],
                "evidence_role_mix": r["evidence_role_mix"],
                "generation_audit": r["generation_audit"],
                "supporting_audit": r["supporting_audit"],
                "eval_invalid_reason": r["eval_invalid_reason"],
                "debug_metrics": r["debug_metrics"],
            }
            for r in query_results
            if r["was_evaled"]
        ],
    }


def print_summary(summary: Dict[str, Any]):
    print("\n" + "=" * 60)
    print("KPI DIMENSION 1: RETRIEVAL QUALITY")
    print("=" * 60)
    print(f"Dataset: {summary['benchmark_display_name']} ({summary['benchmark_split']} split)")
    print(f"Sample Size: {summary['sample_size']} Questions / {summary['unique_pages_ingested']} Ingested Reference Pages")
    print(f"Retrieval Target: {summary['retrieval_metric_label']}")
    print(f"| Metric | Value |")
    print(f"|---|---|")
    print(f"| Hit Rate @ 5 | {summary['hit_rate_5']:.2f} |")
    print(f"| MRR @ 5 | {summary['mrr_5']:.2f} |")
    print(f"| Second-hop Trigger Rate | {summary['second_hop_trigger_rate']:.2f} |")
    print(f"| Avg First-hop Candidates | {summary['avg_first_hop_candidates']:.1f} |")
    print(f"| Avg Merged Candidates | {summary['avg_merged_candidates']:.1f} |")
    print(f"| Avg Pre-pack Candidates | {summary['avg_pre_pack_count']:.1f} |")
    print(f"| Avg Post-pack Candidates | {summary['avg_post_pack_count']:.1f} |")
    print(f"| Avg Duplicates Removed | {summary['avg_duplicates_removed']:.2f} |")
    print(f"| Avg Bridge Coverage Score | {summary['avg_bridge_coverage_score']:.3f} |")
    print(f"| Avg Support Coverage Score | {summary['avg_support_coverage_score']:.3f} |")
    print(f"| Graph/Symbolic Candidates | {summary['graph_symbolic_candidates']} |")
    print(f"| Graph/Symbolic Kept | {summary['graph_symbolic_kept']} |")
    print(f"| Graph/Symbolic Dropped | {summary['graph_symbolic_dropped']} |")
    print(f"| Corroborated Graph Kept | {summary['corroborated_graph_kept']} |")
    print(f"| Final Source-Type Mix | {summary['source_type_mix_totals']} |")
    print(f"| Final Evidence-Role Mix | {summary['evidence_role_mix_totals']} |")

    print("\n" + "=" * 60)
    print("KPI DIMENSION 2: ANSWER QUALITY")
    print("=" * 60)
    if summary["answer_metric_label"]:
        print(f"Answer Metric Family: {summary['answer_metric_label']}")
        print(f"Answer EM: {summary['answer_em']:.3f}" if summary["answer_em"] is not None else "Answer EM: not available.")
        print(f"Answer F1: {summary['answer_f1']:.3f}" if summary["answer_f1"] is not None else "Answer F1: not available.")
    else:
        print("Answer EM/F1: not available for this benchmark package.")
    if summary["generation_eval_enabled"] and summary["faithfulness"] is not None:
        print(f"Faithfulness Score: {summary['faithfulness']:.3f}")
        print(f"* Generation evaluation cases: {summary['generation_eval_case_count']} / {summary['sample_size']} (every {summary['generation_eval_every_n']}th query)")
    else:
        print("Faithfulness Score: not evaluated in this run.")

    print("\n" + "=" * 60)
    print("KPI DIMENSION 3: FINAL EVIDENCE QUALITY")
    print("=" * 60)
    if summary["generation_eval_enabled"] and summary["legacy_context_precision"] is not None:
        print(f"Legacy Context Precision:      {summary['legacy_context_precision']:.3f}")
        print(f"Generation Context Precision:  {summary['generation_context_precision']:.3f}")
        print(f"Supporting Context Precision:  {summary['supporting_context_precision']:.3f}")
        print(f"* Support selector worse cases: {summary['support_selector_worse_cases']}")
        print(f"* Generation assembly worse cases: {summary['generation_assembly_worse_cases']}")
        print(f"* Both contexts noisy cases: {summary['both_contexts_noisy_cases']}")
        print(f"* Avg generation context chunks: {summary['avg_generation_context_count']:.2f}")
        print(f"* Avg generation duplicate-title ratio: {summary['avg_generation_duplicate_title_ratio']:.3f}")
        print(f"* Generation background ratio: {summary['generation_background_ratio']:.3f}")
        print(f"* Generation noise ratio: {summary['generation_noise_ratio']:.3f}")
        print(f"* Generation compacted chunks removed: {summary['generation_compacted_total']}")
    else:
        print("Context precision metrics: not evaluated in this run.")

    print("\n" + "=" * 60)
    print("KPI DIMENSION 4: DATA & COST EFFICIENCY")
    print("=" * 60)
    print(f"Dehydration/Noise Reduction Ratio: {summary['noise_reduction_ratio']:.1f}x")
    print(f"Total Raw Tokens (Benchmark Corpus): {summary['raw_token_total']:,.0f}")
    print(f"Retrieved Top-K Tokens sent to LLM: {summary['retrieved_token_total']:,.0f}")
    print(f"Token Cost Savings Percentage: {summary['cost_savings_pct']:.2f}%")
    print(f"Avg Final Context Evidence Count: {summary['avg_final_context_count']:.1f}")
    print(f"Avg Bridge Query Count: {summary['avg_bridge_query_count']:.1f}")
    print(f"Second-hop New Evidence Ratio: {summary['second_hop_new_evidence_ratio']:.3f}")

    if summary.get("benchmark_notes"):
        print("\nNotes:")
        for note in summary["benchmark_notes"]:
            print(f"- {note}")

    print("\nCompleted benchmark run.")


async def main_async():
    base_url = os.getenv("NOVASEARCH_URL", "http://127.0.0.1:8000")
    api_key = os.getenv("API_KEY", "")
    headers = {"X-API-KEY": api_key}

    benchmark_name = normalize_benchmark_name(os.getenv("BENCHMARK_NAME", "hotpotqa"))
    benchmark_split = os.getenv("BENCHMARK_SPLIT", default_split_for_benchmark(benchmark_name))
    sample_size = resolve_sample_size(benchmark_name)
    generation_eval_enabled = resolve_generation_eval_enabled(benchmark_name)
    eval_every_n = resolve_eval_every_n(benchmark_name)
    trace_path = os.getenv("KPI_TRACE_PATH", default_trace_path(benchmark_name))
    reset_stores = _benchmark_bool(
        "BENCHMARK_RESET_STORES",
        _benchmark_bool("HOTPOT_RESET_STORES", True) if benchmark_name == "hotpotqa" else True,
    )

    print(f"Waiting for NovaSearch API at {base_url}...")
    for i in range(45):
        try:
            with httpx.Client() as check_client:
                r = check_client.get(f"{base_url}/health", timeout=2.0)
                if r.status_code == 200:
                    print("API is ONLINE.")
                    break
        except Exception:
            pass
        if i % 5 == 0:
            print(f"Still waiting for server... ({i})")
        time.sleep(2)
    else:
        print("CRITICAL: Server failed to start. Aborting.")
        return

    print("\n" + "=" * 60)
    print(f"KPI DIMENSION: BENCHMARK RUN ({benchmark_name})")
    print("=" * 60)
    print(f"Loading benchmark bundle: benchmark={benchmark_name}, split={benchmark_split}, sample_size={sample_size}")
    benchmark = load_benchmark_bundle(benchmark_name, sample_size=sample_size, split=benchmark_split)
    print(
        f"Loaded {benchmark.sample_size} cases and {len(benchmark.unique_pages)} unique pages "
        f"for {benchmark.display_name}."
    )

    if reset_stores:
        reset_benchmark_stores()

    print(f"\nStarting ASYNC ingestion of {len(benchmark.unique_pages)} reference pages...")
    error_list = []
    ingest_concurrency = int(os.getenv("BENCHMARK_INGEST_CONCURRENCY", os.getenv("HOTPOT_INGEST_CONCURRENCY", "3")))
    ingest_semaphore = asyncio.Semaphore(ingest_concurrency)
    ingest_start = time.perf_counter()

    async with httpx.AsyncClient(timeout=120) as client:
        with tqdm(total=len(benchmark.unique_pages), desc="Ingesting Pages") as pbar:
            tasks = [
                ingest_page_document(
                    client,
                    base_url,
                    headers,
                    page.title,
                    page.text,
                    page.section,
                    idx,
                    ingest_semaphore,
                    pbar,
                    error_list,
                )
                for idx, page in enumerate(benchmark.unique_pages)
            ]
            await asyncio.gather(*tasks)
    ingest_duration = time.perf_counter() - ingest_start

    if error_list:
        print(f"\nCompleted ingestion with {len(error_list)} errors (e.g., {error_list[0]}).")
    else:
        print("\nCompleted ingestion successfully.")

    graph_enabled = (
        os.getenv("ENABLE_GRAPH_INGESTION", "false").lower() in {"1", "true", "yes"}
        or os.getenv("ENABLE_GRAPH_RETRIEVAL", "false").lower() in {"1", "true", "yes"}
    )
    if graph_enabled:
        print("\nWaiting 5 seconds for background graph indexing to settle...")
        await asyncio.sleep(5)

    print(f"\nStarting ASYNC query evaluation phase ({benchmark.sample_size} queries)...")
    evaluator = RAGEvaluator()
    query_concurrency = int(os.getenv("BENCHMARK_QUERY_CONCURRENCY", os.getenv("HOTPOT_QUERY_CONCURRENCY", "5")))
    query_semaphore = asyncio.Semaphore(query_concurrency)
    query_results: List[Dict[str, Any]] = []
    query_start = time.perf_counter()
    if generation_eval_enabled and not evaluator.client:
        print("Warning: generation evaluation requested, but no OpenAI API key is configured. Skipping generation metrics.")
        generation_eval_enabled = False

    async with httpx.AsyncClient(timeout=120) as client:
        with tqdm(total=benchmark.sample_size, desc="Eval Queries") as pbar:
            tasks = [
                evaluate_query(
                    client,
                    base_url,
                    headers,
                    case,
                    evaluator,
                    query_semaphore,
                    pbar,
                    query_results,
                    benchmark.name,
                    generation_eval_enabled,
                    eval_every_n,
                )
                for case in benchmark.cases
            ]
            await asyncio.gather(*tasks)
    query_duration = time.perf_counter() - query_start

    total_retrieved_tokens = sum(r["retrieved_tokens"] for r in query_results)
    summary = build_summary_payload(
        benchmark,
        query_results,
        total_retrieved_tokens=total_retrieved_tokens,
        ingest_duration=ingest_duration,
        query_duration=query_duration,
        generation_eval_enabled=generation_eval_enabled,
        evaluator_model=evaluator.model,
        eval_every_n=eval_every_n,
    )
    print_summary(summary)

    with open(trace_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Trace written to {trace_path}")


if __name__ == "__main__":
    asyncio.run(main_async())
