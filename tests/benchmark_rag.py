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
        default_split_for_benchmark,
        default_trace_path,
        exact_match_score,
        load_benchmark_bundle,
        normalize_benchmark_name,
        resolve_sample_size,
        token_f1_score,
    )
except ImportError:
    from benchmark_support import (
        BenchmarkBundle,
        BenchmarkCase,
        default_split_for_benchmark,
        default_trace_path,
        exact_match_score,
        load_benchmark_bundle,
        normalize_benchmark_name,
        resolve_sample_size,
        token_f1_score,
    )

try:
    from tests.ragas_support import (
        RagasEvaluationResult,
        RagasRunner,
        normalize_context_mode,
    )
except ImportError:
    from ragas_support import (
        RagasEvaluationResult,
        RagasRunner,
        normalize_context_mode,
    )

load_dotenv()
logger = logging.getLogger(__name__)

# Reduce httpx/openai noise
logging.getLogger("httpx").setLevel(logging.WARNING)

try:
    import psycopg2
except ImportError:
    psycopg2 = None

try:
    from neo4j import GraphDatabase
except ImportError:
    GraphDatabase = None


def _estimate_context_tokens(contexts: List[str]) -> float:
    return sum(len((context or "").split()) * 1.3 for context in contexts)


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
    semaphore,
    pbar,
    results,
):
    async with semaphore:
        query = case.query
        retrieval_contexts: List[str] = []
        generation_contexts: List[str] = []
        supporting_contexts: List[str] = []
        answer = ""
        scored_answer = ""
        debug_metrics: Dict[str, Any] = {}
        retrieval_sources_raw: List[Dict[str, Any]] = []
        generation_sources_raw: List[Dict[str, Any]] = []
        supporting_sources_raw: List[Dict[str, Any]] = []
        reference_contexts: List[str] = [page.text for page in case.pages]

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
                        scored_answer = str(data.get("benchmark_answer") or "").strip()
                        debug_metrics = data.get("debug_metrics") or {}
                        retrieval_sources_raw = retrieval_sources
                        generation_sources_raw = generation_sources
                        supporting_sources_raw = supporting_sources
                        retrieval_contexts = [_compose_retrieval_context(s) for s in retrieval_sources]
                        generation_contexts = [_compose_retrieval_context(s) for s in generation_sources]
                        supporting_contexts = [_compose_retrieval_context(s) for s in supporting_sources]
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
                    "answer": answer,
                    "scored_answer": scored_answer or answer,
                    "retrieval_contexts": retrieval_contexts,
                    "generation_contexts": generation_contexts,
                    "supporting_contexts": supporting_contexts,
                    "reference_contexts": reference_contexts,
                    "retrieval_token_estimate": _estimate_context_tokens(retrieval_contexts),
                    "generation_token_estimate": _estimate_context_tokens(generation_contexts),
                    "expected_titles": case.expected_titles,
                    "expected_answers": case.expected_answers,
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


async def verify_runtime_preflight(
    client: httpx.AsyncClient,
    base_url: str,
    headers: Dict[str, str],
    canary_query: str,
) -> Dict[str, Any]:
    resp = await client.post(
        f"{base_url}/ask",
        json={"query": canary_query, "top_k": 5},
        headers=headers,
    )
    resp.raise_for_status()

    answer_metadata: Dict[str, Any] = {}
    for line in resp.text.strip().split("\n"):
        if not line:
            continue
        try:
            data = json.loads(line)
        except Exception:
            continue
        if data.get("type") == "answer_metadata":
            answer_metadata = data
            break

    if not answer_metadata:
        raise RuntimeError("Runtime preflight failed: /ask did not emit answer_metadata.")

    debug_metrics = answer_metadata.get("debug_metrics") or {}
    schema_version = debug_metrics.get("debug_schema_version")
    runtime_enabled = debug_metrics.get("chain_aware_runtime_enabled")

    required_chain_fields = (
        "candidate_chains",
        "selected_chains",
        "chain_lengths",
        "path_score_distribution",
    )
    missing_chain_fields = [
        field for field in required_chain_fields
        if field not in debug_metrics or debug_metrics.get(field) is None
    ]

    if not schema_version:
        raise RuntimeError(
            "Runtime preflight failed: debug_schema_version missing. "
            "Likely stale API service instance."
        )
    if runtime_enabled is not True:
        raise RuntimeError(
            "Runtime preflight failed: chain_aware_runtime_enabled is not true. "
            "Likely stale API service instance or misconfigured runtime."
        )
    if missing_chain_fields:
        raise RuntimeError(
            "Runtime preflight failed: missing chain debug fields: "
            + ", ".join(missing_chain_fields)
        )

    return {
        "endpoint": base_url,
        "schema_version": schema_version,
        "runtime_enabled": runtime_enabled,
        "candidate_chains": debug_metrics.get("candidate_chains"),
        "selected_chains": debug_metrics.get("selected_chains"),
        "chain_lengths": debug_metrics.get("chain_lengths"),
        "path_score_distribution": debug_metrics.get("path_score_distribution"),
    }


def build_summary_payload(
    benchmark: BenchmarkBundle,
    query_results: List[Dict[str, Any]],
    ragas_result: RagasEvaluationResult,
    total_context_tokens: float,
    ingest_duration: float,
    query_duration: float,
    context_mode: str,
    ragas_model: Optional[str],
    ragas_embedding_model: Optional[str],
    runtime_preflight: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    n = len(query_results)
    debug_rows = [r.get("debug_metrics") or {} for r in query_results]
    second_hop_trigger_rate = sum(1 for row in debug_rows if row.get("second_hop_triggered")) / max(1, n)
    avg_first_hop_candidates = sum((row.get("first_hop_candidates") or 0) for row in debug_rows) / max(1, n)
    avg_merged_candidates = sum((row.get("merged_candidate_count") or 0) for row in debug_rows) / max(1, n)
    avg_final_context_count = sum((row.get("final_context_count") or 0) for row in debug_rows) / max(1, n)
    avg_bridge_query_count = sum((row.get("bridge_query_count") or 0) for row in debug_rows) / max(1, n)
    avg_pre_pack_count = sum((row.get("pre_pack_count") or 0) for row in debug_rows) / max(1, n)
    avg_post_pack_count = sum((row.get("post_pack_count") or 0) for row in debug_rows) / max(1, n)
    avg_candidate_chains = sum((row.get("candidate_chains") or 0) for row in debug_rows) / max(1, n)
    avg_selected_chains = sum((row.get("selected_chains") or 0) for row in debug_rows) / max(1, n)
    chain_debug_observed_cases = sum(
        1
        for row in debug_rows
        if any(
            key in row
            for key in (
                "candidate_chains",
                "selected_chains",
                "chain_lengths",
                "path_score_distribution",
                "final_context_chain_mix",
            )
        )
    )
    chain_debug_nonzero_cases = sum(
        1
        for row in debug_rows
        if (row.get("candidate_chains") or 0) > 0
        or (row.get("selected_chains") or 0) > 0
        or bool(row.get("final_context_chain_mix"))
    )
    chain_runtime_enabled_cases = sum(
        1
        for row in debug_rows
        if row.get("chain_aware_runtime_enabled") is True
    )
    debug_schema_versions = sorted({
        str(row.get("debug_schema_version"))
        for row in debug_rows
        if row.get("debug_schema_version")
    })
    total_second_hop_added = sum((row.get("second_hop_added_count") or 0) for row in debug_rows)
    total_merged_candidates = sum((row.get("merged_candidate_count") or 0) for row in debug_rows)
    second_hop_new_evidence_ratio = total_second_hop_added / max(1, total_merged_candidates)
    avg_duplicates_removed = _mean([float(r.get("duplicates_removed", 0)) for r in query_results]) or 0.0
    avg_bridge_coverage_score = _mean([float(r.get("bridge_coverage_score", 0.0)) for r in query_results]) or 0.0
    avg_support_coverage_score = _mean([float(r.get("support_coverage_score", 0.0)) for r in query_results]) or 0.0
    avg_chain_support_coverage_score = sum((row.get("chain_support_coverage_score") or 0.0) for row in debug_rows) / max(1, n)
    avg_chain_bridge_coverage_score = sum((row.get("chain_bridge_coverage_score") or 0.0) for row in debug_rows) / max(1, n)
    avg_bridge_chunks_kept = sum((row.get("bridge_chunks_kept") or 0) for row in debug_rows) / max(1, n)
    avg_bridge_chunks_dropped = sum((row.get("bridge_chunks_dropped") or 0) for row in debug_rows) / max(1, n)
    total_graph_symbolic_candidates = sum(int(row.get("graph_symbolic_candidates", 0) or 0) for row in debug_rows)
    total_graph_symbolic_kept = sum(int(row.get("graph_symbolic_kept", 0) or 0) for row in debug_rows)
    total_graph_symbolic_dropped = sum(int(row.get("graph_symbolic_dropped", 0) or 0) for row in debug_rows)
    total_corroborated_graph_kept = sum(int(row.get("corroborated_graph_kept", 0) or 0) for row in debug_rows)
    generation_compacted_total = sum(int(row.get("generation_compacted_count", 0) or 0) for row in debug_rows)
    source_type_mix_totals: Dict[str, int] = {}
    evidence_role_mix_totals: Dict[str, int] = {}
    final_context_chain_mix_totals: Dict[str, int] = {}
    for row in query_results:
        for key, value in (row.get("source_type_mix") or {}).items():
            source_type_mix_totals[key] = source_type_mix_totals.get(key, 0) + int(value or 0)
        for key, value in (row.get("evidence_role_mix") or {}).items():
            evidence_role_mix_totals[key] = evidence_role_mix_totals.get(key, 0) + int(value or 0)
        for key, value in ((row.get("debug_metrics") or {}).get("final_context_chain_mix") or {}).items():
            final_context_chain_mix_totals[key] = final_context_chain_mix_totals.get(key, 0) + int(value or 0)

    avg_generation_context_count = _mean([len(r["generation_contexts"]) for r in query_results]) or 0.0
    avg_retrieval_context_count = _mean([len(r["retrieval_contexts"]) for r in query_results]) or 0.0
    avg_generation_duplicate_title_ratio = _mean(
        [_duplicate_title_ratio(r["generation_contexts"]) for r in query_results if r["generation_contexts"]]
    ) or 0.0
    answer_em = _mean([
        exact_match_score(r.get("scored_answer", "") or r.get("answer", ""), r.get("expected_answers") or [])
        for r in query_results
    ]) or 0.0
    answer_f1 = _mean([
        token_f1_score(r.get("scored_answer", "") or r.get("answer", ""), r.get("expected_answers") or [])
        for r in query_results
    ]) or 0.0
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

    ratio = benchmark.total_raw_tokens / max(1, total_context_tokens)
    savings = (1 - (total_context_tokens / max(1, benchmark.total_raw_tokens))) * 100

    return {
        "benchmark_name": benchmark.name,
        "benchmark_display_name": benchmark.display_name,
        "benchmark_split": benchmark.split,
        "benchmark_notes": benchmark.notes,
        "runtime_endpoint": (runtime_preflight or {}).get("endpoint"),
        "runtime_preflight": runtime_preflight or {},
        "sample_size": n,
        "unique_pages_ingested": len(benchmark.unique_pages),
        "ragas_context_mode": normalize_context_mode(context_mode),
        "ragas_metrics": ragas_result.metrics_summary,
        "ragas_metric_modes": ragas_result.metrics_applied,
        "ragas_metric_skips": ragas_result.metrics_skipped,
        "ragas_eval_sample_count": len(ragas_result.sample_rows),
        "ragas_model": ragas_model,
        "ragas_embedding_model": ragas_embedding_model,
        "cost_savings_pct": savings,
        "noise_reduction_ratio": ratio,
        "second_hop_trigger_rate": second_hop_trigger_rate,
        "avg_first_hop_candidates": avg_first_hop_candidates,
        "avg_merged_candidates": avg_merged_candidates,
        "avg_pre_pack_count": avg_pre_pack_count,
        "avg_post_pack_count": avg_post_pack_count,
        "avg_candidate_chains": avg_candidate_chains,
        "avg_selected_chains": avg_selected_chains,
        "chain_debug_observed_cases": chain_debug_observed_cases,
        "chain_debug_nonzero_cases": chain_debug_nonzero_cases,
        "chain_runtime_enabled_cases": chain_runtime_enabled_cases,
        "debug_schema_versions": debug_schema_versions,
        "avg_final_context_count": avg_final_context_count,
        "avg_bridge_query_count": avg_bridge_query_count,
        "second_hop_new_evidence_ratio": second_hop_new_evidence_ratio,
        "avg_duplicates_removed": avg_duplicates_removed,
        "avg_bridge_coverage_score": avg_bridge_coverage_score,
        "avg_support_coverage_score": avg_support_coverage_score,
        "avg_chain_support_coverage_score": avg_chain_support_coverage_score,
        "avg_chain_bridge_coverage_score": avg_chain_bridge_coverage_score,
        "avg_bridge_chunks_kept": avg_bridge_chunks_kept,
        "avg_bridge_chunks_dropped": avg_bridge_chunks_dropped,
        "graph_symbolic_candidates": total_graph_symbolic_candidates,
        "graph_symbolic_kept": total_graph_symbolic_kept,
        "graph_symbolic_dropped": total_graph_symbolic_dropped,
        "corroborated_graph_kept": total_corroborated_graph_kept,
        "source_type_mix_totals": source_type_mix_totals,
        "evidence_role_mix_totals": evidence_role_mix_totals,
        "final_context_chain_mix_totals": final_context_chain_mix_totals,
        "avg_generation_context_count": avg_generation_context_count,
        "avg_retrieval_context_count": avg_retrieval_context_count,
        "avg_generation_duplicate_title_ratio": avg_generation_duplicate_title_ratio,
        "answer_em": answer_em,
        "answer_f1": answer_f1,
        "avg_planner_confirmed_generation_count": avg_planner_confirmed_generation_count,
        "avg_supporting_selected_generation_count": avg_supporting_selected_generation_count,
        "generation_planner_chunk_ratio": generation_planner_chunk_ratio_metric,
        "generation_compacted_total": generation_compacted_total,
        "blocked_seed_count": blocked_seed_count,
        "supporting_anchor_filtered_count": supporting_anchor_filtered_count,
        "planner_rescued_generation_missed_cases": planner_rescued_generation_missed_cases,
        "bridge_targeting_hit_cases": bridge_targeting_hit_cases,
        "alias_identity_resolution_improved_cases": alias_identity_resolution_improved_cases,
        "page_family_correct_chunk_wrong_cases": page_family_correct_chunk_wrong_cases,
        "raw_token_total": benchmark.total_raw_tokens,
        "retrieved_token_total": total_context_tokens,
        "ingestion_duration_seconds": ingest_duration,
        "query_duration_seconds": query_duration,
        "benchmark_mode": os.getenv("NOVASEARCH_BENCHMARK_MODE", "false").lower() in {"1", "true", "yes"},
        "graph_ingestion_enabled": os.getenv("ENABLE_GRAPH_INGESTION", "false").lower() in {"1", "true", "yes"},
        "graph_retrieval_enabled": os.getenv("ENABLE_GRAPH_RETRIEVAL", "false").lower() in {"1", "true", "yes"},
        "vision_retriever_enabled": os.getenv("ENABLE_VISION_RETRIEVER", "true").lower() in {"1", "true", "yes"},
        "text_vision_indexing_enabled": os.getenv("ENABLE_TEXT_VISION_INDEXING", "true").lower() in {"1", "true", "yes"},
        "sampled_ragas_cases": [
            {
                "query": row["query"],
                "answer": row["answer"],
                "contexts": row.get("ragas_contexts", row.get("generation_contexts", [])),
                "ground_truth": row.get("ragas_ground_truth"),
                "reference_contexts": row.get("ragas_reference_contexts", row.get("reference_contexts", [])),
                "ragas_scores": row.get("ragas_scores", {}),
                "ragas_metric_modes": row.get("ragas_metric_modes", {}),
                "ragas_metric_skips": row.get("ragas_metric_skips", {}),
                "debug_metrics": row.get("debug_metrics", {}),
            }
            for row in ragas_result.sample_rows
        ],
    }


def print_summary(summary: Dict[str, Any]):
    print("\n" + "=" * 60)
    print("KPI DIMENSION 1: RAGAS METRICS")
    print("=" * 60)
    print(f"Dataset: {summary['benchmark_display_name']} ({summary['benchmark_split']} split)")
    print(f"Sample Size: {summary['sample_size']} Questions / {summary['unique_pages_ingested']} Ingested Reference Pages")
    if summary.get("runtime_endpoint"):
        print(f"Runtime Endpoint: {summary['runtime_endpoint']}")
    if summary.get("runtime_preflight"):
        print(
            "Runtime Preflight: "
            f"schema={summary['runtime_preflight'].get('schema_version')}, "
            f"chain_enabled={summary['runtime_preflight'].get('runtime_enabled')}, "
            f"candidate_chains={summary['runtime_preflight'].get('candidate_chains')}, "
            f"selected_chains={summary['runtime_preflight'].get('selected_chains')}"
        )
    print(f"Ragas Context Mode: {summary['ragas_context_mode']}")
    print(f"Ragas Model: {summary['ragas_model']}")
    print(f"Ragas Embedding Model: {summary['ragas_embedding_model']}")
    print(f"Ragas Metric Modes: {summary['ragas_metric_modes']}")
    if summary.get("ragas_metric_skips"):
        print(f"Ragas Metric Skips: {summary['ragas_metric_skips']}")
    print(f"Answer EM: {summary['answer_em']:.3f}")
    print(f"Answer F1: {summary['answer_f1']:.3f}")
    print(f"| Metric | Value |")
    print(f"|---|---|")
    for metric_name, metric_value in summary["ragas_metrics"].items():
        rendered = "not available" if metric_value is None else f"{metric_value:.3f}"
        print(f"| {metric_name} | {rendered} |")

    print("\n" + "=" * 60)
    print("KPI DIMENSION 2: NOVASEARCH DEBUG SUMMARY")
    print("=" * 60)
    print(f"| Second-hop Trigger Rate | {summary['second_hop_trigger_rate']:.2f} |")
    print(f"| Avg First-hop Candidates | {summary['avg_first_hop_candidates']:.1f} |")
    print(f"| Avg Merged Candidates | {summary['avg_merged_candidates']:.1f} |")
    print(f"| Avg Pre-pack Candidates | {summary['avg_pre_pack_count']:.1f} |")
    print(f"| Avg Post-pack Candidates | {summary['avg_post_pack_count']:.1f} |")
    print(f"| Avg Candidate Chains | {summary['avg_candidate_chains']:.1f} |")
    print(f"| Avg Selected Chains | {summary['avg_selected_chains']:.1f} |")
    print(f"| Chain Debug Observed Cases | {summary['chain_debug_observed_cases']} / {summary['sample_size']} |")
    print(f"| Chain Debug Nonzero Cases | {summary['chain_debug_nonzero_cases']} / {summary['sample_size']} |")
    print(f"| Chain Runtime Enabled Cases | {summary['chain_runtime_enabled_cases']} / {summary['sample_size']} |")
    print(f"| Debug Schema Versions | {summary['debug_schema_versions']} |")
    print(f"| Avg Duplicates Removed | {summary['avg_duplicates_removed']:.2f} |")
    print(f"| Avg Bridge Coverage Score | {summary['avg_bridge_coverage_score']:.3f} |")
    print(f"| Avg Support Coverage Score | {summary['avg_support_coverage_score']:.3f} |")
    print(f"| Avg Chain Support Coverage Score | {summary['avg_chain_support_coverage_score']:.3f} |")
    print(f"| Avg Chain Bridge Coverage Score | {summary['avg_chain_bridge_coverage_score']:.3f} |")
    print(f"| Avg Bridge Chunks Kept | {summary['avg_bridge_chunks_kept']:.2f} |")
    print(f"| Avg Bridge Chunks Dropped | {summary['avg_bridge_chunks_dropped']:.2f} |")
    print(f"| Graph/Symbolic Candidates | {summary['graph_symbolic_candidates']} |")
    print(f"| Graph/Symbolic Kept | {summary['graph_symbolic_kept']} |")
    print(f"| Graph/Symbolic Dropped | {summary['graph_symbolic_dropped']} |")
    print(f"| Corroborated Graph Kept | {summary['corroborated_graph_kept']} |")
    print(f"| Final Source-Type Mix | {summary['source_type_mix_totals']} |")
    print(f"| Final Evidence-Role Mix | {summary['evidence_role_mix_totals']} |")
    print(f"| Final Context Chain Mix | {summary['final_context_chain_mix_totals']} |")

    print("\n" + "=" * 60)
    print("KPI DIMENSION 3: DATA & COST EFFICIENCY")
    print("=" * 60)
    print(f"Dehydration/Noise Reduction Ratio: {summary['noise_reduction_ratio']:.1f}x")
    print(f"Total Raw Tokens (Benchmark Corpus): {summary['raw_token_total']:,.0f}")
    print(f"Retrieved Top-K Tokens sent to LLM: {summary['retrieved_token_total']:,.0f}")
    print(f"Token Cost Savings Percentage: {summary['cost_savings_pct']:.2f}%")
    print(f"Avg Final Context Evidence Count: {summary['avg_final_context_count']:.1f}")
    print(f"Avg Generation Context Count: {summary['avg_generation_context_count']:.1f}")
    print(f"Avg Retrieval Context Count: {summary['avg_retrieval_context_count']:.1f}")
    print(f"Avg Bridge Query Count: {summary['avg_bridge_query_count']:.1f}")
    print(f"Second-hop New Evidence Ratio: {summary['second_hop_new_evidence_ratio']:.3f}")

    if summary.get("benchmark_notes"):
        print("\nNotes:")
        for note in summary["benchmark_notes"]:
            print(f"- {note}")
    if summary.get("chain_runtime_enabled_cases", 0) > 0 and summary.get("chain_debug_nonzero_cases", 0) == 0:
        print("- Warning: chain-aware runtime was enabled, but no nonzero chain metrics were observed.")
        print("- This usually indicates a stale API service instance or a debug-propagation break in the /ask path.")

    print("\nCompleted benchmark run.")


async def main_async():
    base_url = os.getenv("NOVASEARCH_URL", "http://127.0.0.1:8000")
    api_key = os.getenv("API_KEY", "")
    headers = {"X-API-KEY": api_key}
    inprocess_mode = _benchmark_bool("BENCHMARK_INPROCESS", False)

    benchmark_name = normalize_benchmark_name(os.getenv("BENCHMARK_NAME", "hotpotqa"))
    benchmark_split = os.getenv("BENCHMARK_SPLIT", default_split_for_benchmark(benchmark_name))
    sample_size = resolve_sample_size(benchmark_name)
    ragas_context_mode = normalize_context_mode(os.getenv("BENCHMARK_CONTEXT_MODE", "generation"))
    trace_path = os.getenv("KPI_TRACE_PATH", default_trace_path(benchmark_name))
    skip_ragas = _benchmark_bool("BENCHMARK_SKIP_RAGAS", False)
    reset_stores = _benchmark_bool(
        "BENCHMARK_RESET_STORES",
        _benchmark_bool("HOTPOT_RESET_STORES", True) if benchmark_name == "hotpotqa" else True,
    )

    async def _run_benchmark_flow(client: httpx.AsyncClient, resolved_base_url: str):
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

        with tqdm(total=len(benchmark.unique_pages), desc="Ingesting Pages") as pbar:
            tasks = [
                ingest_page_document(
                    client,
                    resolved_base_url,
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

        canary_query = benchmark.cases[0].query if benchmark.cases else "What is this document about?"
        print("\nRunning runtime preflight /ask verification...")
        runtime_preflight = await verify_runtime_preflight(
            client=client,
            base_url=resolved_base_url,
            headers=headers,
            canary_query=canary_query,
        )
        print(
            "Runtime preflight passed: "
            f"endpoint={runtime_preflight['endpoint']}, "
            f"schema={runtime_preflight['schema_version']}, "
            f"chain_enabled={runtime_preflight['runtime_enabled']}, "
            f"candidate_chains={runtime_preflight['candidate_chains']}, "
            f"selected_chains={runtime_preflight['selected_chains']}"
        )

        print(f"\nStarting ASYNC query evaluation phase ({benchmark.sample_size} queries)...")
        query_concurrency = int(os.getenv("BENCHMARK_QUERY_CONCURRENCY", os.getenv("HOTPOT_QUERY_CONCURRENCY", "5")))
        query_semaphore = asyncio.Semaphore(query_concurrency)
        query_results: List[Dict[str, Any]] = []
        query_start = time.perf_counter()

        with tqdm(total=benchmark.sample_size, desc="Eval Queries") as pbar:
            tasks = [
                evaluate_query(
                    client,
                    resolved_base_url,
                    headers,
                    case,
                    query_semaphore,
                    pbar,
                    query_results,
                )
                for case in benchmark.cases
            ]
            await asyncio.gather(*tasks)
        query_duration = time.perf_counter() - query_start

        ragas_runner = RagasRunner()
        if skip_ragas:
            print("\nSkipping Ragas evaluation for this run (BENCHMARK_SKIP_RAGAS=true).")
            ragas_result = RagasEvaluationResult(
                sample_rows=list(query_results),
                metrics_summary={metric_name: None for metric_name in ragas_runner.selected_metrics},
                metrics_applied={},
                metrics_skipped={metric_name: "skipped_for_report_mode" for metric_name in ragas_runner.selected_metrics},
            )
        else:
            print("\nRunning Ragas evaluation...")
            ragas_result = await ragas_runner.evaluate_rows(query_results, context_mode=ragas_context_mode)
        total_context_tokens = sum(
            float(
                row["generation_token_estimate"]
                if ragas_context_mode == "generation" else row["retrieval_token_estimate"]
            )
            for row in query_results
        )
        summary = build_summary_payload(
            benchmark,
            query_results,
            ragas_result=ragas_result,
            total_context_tokens=total_context_tokens,
            ingest_duration=ingest_duration,
            query_duration=query_duration,
            context_mode=ragas_context_mode,
            ragas_model=ragas_runner.model,
            ragas_embedding_model=ragas_runner.embedding_model,
            runtime_preflight=runtime_preflight,
        )
        print_summary(summary)

        with open(trace_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Trace written to {trace_path}")

    if inprocess_mode:
        print("Running benchmark in-process via ASGITransport using current workspace code.")
        from api.main import app

        base_url = "http://testserver"
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url=base_url, timeout=120) as client:
                health = await client.get("/health")
                if health.status_code != 200:
                    print("CRITICAL: In-process app failed health check. Aborting.")
                    return
                print("In-process API is ONLINE.")
                await _run_benchmark_flow(client, base_url)
        return

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

    async with httpx.AsyncClient(timeout=120) as client:
        await _run_benchmark_flow(client, base_url)


if __name__ == "__main__":
    asyncio.run(main_async())
