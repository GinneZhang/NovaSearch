"""
Enterprise Copilot Agent Logic.

This module acts as the "brain" of AsterScope. It orchestrates the retrieval
of context via the HybridSearchCoordinator and constructs source-grounded LLM
prompts for the OpenAI API to enforce strict adherence to enterprise facts.
"""

import os
import re
import json
import uuid
import logging
import math
from collections import Counter, defaultdict
from typing import List, Dict, Any, Optional, Tuple

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import openai

# Assuming hybrid_search is in the project's Python path
from retrieval.hybrid_search import HybridSearchCoordinator
from core.memory import RedisMemoryManager
from agent.planner import TaskDecomposer
from agent.query_parser import QueryGraphParser

logger = logging.getLogger(__name__)


def _normalize_terms(text: str) -> List[str]:
    return [token for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9_-]+", (text or "").lower()) if len(token) > 2]


def _safe_numeric(value: Any, fallback: float = 0.0) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return fallback
    if math.isnan(numeric) or math.isinf(numeric):
        return fallback
    return numeric


_GENERIC_BRIDGE_TOKENS = {
    "doc", "section", "general", "content", "source", "marker",
    "document", "unknown", "hotpotqa", "benchmark"
}

_QUERY_FOCUS_STOPWORDS = {
    "what", "which", "who", "whom", "when", "where", "why", "how",
    "is", "was", "were", "are", "be", "been", "being",
    "the", "a", "an", "of", "in", "on", "at", "to", "for", "from",
    "by", "with", "about", "into", "after", "before", "during",
    "does", "did", "do", "that", "this", "these", "those",
    "its", "their", "his", "her", "than", "then"
}

class EnterpriseCopilotAgent:
    """
    The main reasoning loop that connects the user's query, the Tri-Engine
    Retrieval system, and the LLM for grounded response generation.
    """

    def __init__(self, model_provider: str = "openai", model: Optional[str] = None):
        """Initialize the LLM client, retrieval system, and Redis memory."""
        self.model_provider = model_provider.lower()
        
        # Setup OpenAI (needed as default fallback and Consistency Evaluator)
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            logger.warning("OPENAI_API_KEY missing from environment.")
        self.openai_client = openai.OpenAI(api_key=self.openai_api_key) if self.openai_api_key else None
        
        if self.model_provider == "anthropic":
            self.model = model or "claude-3-haiku-20240307"
            self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
            if not self.anthropic_api_key:
                logger.warning("ANTHROPIC_API_KEY missing from environment.")
            if self.anthropic_api_key:
                import anthropic
                self.client = anthropic.Anthropic(api_key=self.anthropic_api_key)
            else:
                self.client = None
        else:
            self.model = model or os.getenv("OPENAI_CHAT_MODEL", "gpt-4.1")
            self.client = self.openai_client
        
        # Instantiate Tri-Engine Retrieval Coordinator
        self.retriever = HybridSearchCoordinator()
        
        # Instantiate Semantic Memory
        self.memory = RedisMemoryManager()
        
        # Instantiate LangChain Task Planner
        self.planner = TaskDecomposer(model_name=os.getenv("PLANNER_MODEL", "gpt-4.1-mini"))
        
        # Instantiate Query Graph Parser
        self.query_parser = QueryGraphParser(model=os.getenv("QUERY_GRAPH_MODEL", "gpt-4.1-mini"))

    def _format_context(self, hits: List[Dict[str, Any]]) -> str:
        """
        Assembles retrieved chunks and graph metadata into a highly structured
        string format for the LLM prompt.
        """
        if not hits:
            return "No relevant context found."

        context_blocks = []
        for i, hit in enumerate(hits):
            # Extract basic info
            source = hit.get("source", "unknown")
            score = hit.get("score", 0.0)
            text = hit.get("chunk_text", "").strip()
            evidence_role = str(hit.get("evidence_role") or hit.get("role") or "support").strip().upper()
            source_type = self._infer_hit_source_type(hit).upper()
            merged_indices = hit.get("merged_chunk_indices") or []
            primary_chain_id = hit.get("primary_chain_id")
            primary_chain_rank = hit.get("primary_chain_rank")
            best_chain_score = _safe_numeric(hit.get("best_chain_score"), 0.0)
            
            # Extract Graph Context
            graph_info = hit.get("graph_context")
            if graph_info:
                title = graph_info.get("doc_title", "Unknown Title")
                section = graph_info.get("doc_section", "Unknown Section")
            else:
                title = hit.get("title", "Unknown Document")
                section = "General"

            # Format the block with strict tracking markers
            block = f"--- [Document {i+1}] ---\n"
            block += f"[Source Marker]: [Doc: {title}, Section: {section}]\n"
            block += f"[Retrieval Type]: {source.upper()} (Score: {score:.3f})\n"
            block += f"[Evidence Role]: {evidence_role}\n"
            block += f"[Source Type]: {source_type}\n"
            if primary_chain_id:
                block += f"[Evidence Chain]: {primary_chain_id}"
                if primary_chain_rank:
                    block += f" (rank {primary_chain_rank}"
                    if best_chain_score:
                        block += f", score {best_chain_score:.3f}"
                    block += ")"
                block += "\n"
            if merged_indices:
                block += f"[Merged Chunk Indices]: {', '.join(str(idx) for idx in merged_indices)}\n"
            if hit.get("is_corroborated") is not None:
                block += f"[Corroborated]: {'YES' if hit.get('is_corroborated') else 'NO'}\n"
            block += f"[Content]: {text}\n"
            context_blocks.append(block)

        return "\n".join(context_blocks)

    def _infer_hit_source_type(self, hit: Dict[str, Any]) -> str:
        source_type = str(hit.get("source_type") or "").strip().lower()
        if source_type:
            return source_type
        retriever = getattr(self, "retriever", None)
        infer_fn = getattr(retriever, "_infer_source_type", None)
        if callable(infer_fn):
            try:
                inferred = str(infer_fn(hit) or "").strip().lower()
                if inferred:
                    return inferred
            except Exception:
                pass

        source = str(hit.get("source") or "").strip().lower()
        if source == "dynamic_cypher":
            return "symbolic"
        if source in {"doc_expansion", "graph_expansion"}:
            return "graph_expansion"
        if source == "raw_lexical":
            return "lexical"
        if source:
            return source
        if hit.get("graph_context"):
            return "graph_enriched"
        return "unknown"

    @staticmethod
    def _is_graph_like_source_type(source_type: str) -> bool:
        return source_type in {"graph_expansion", "graph_enriched", "symbolic"}

    def _merge_context_text_pair(self, left_text: str, right_text: str) -> str:
        left = (left_text or "").strip()
        right = (right_text or "").strip()
        if not left:
            return right
        if not right:
            return left
        if right in left:
            return left
        if left in right:
            return right

        max_overlap = min(len(left), len(right), 240)
        overlap = 0
        for size in range(max_overlap, 24, -1):
            if left[-size:] == right[:size]:
                overlap = size
                break
        if overlap:
            return f"{left}{right[overlap:]}".strip()
        return f"{left}\n{right}".strip()

    def _compact_generation_context_hits(self, hits: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int]:
        if not hits:
            return [], 0
        if os.getenv("ENABLE_CONTEXT_FAMILY_COMPACTION", "true").lower() not in {"1", "true", "yes"}:
            return hits, 0

        family_order: List[Tuple[str, str]] = []
        family_hits: Dict[Tuple[str, str], List[Tuple[int, Dict[str, Any]]]] = {}
        for position, hit in enumerate(hits):
            graph_info = hit.get("graph_context") or {}
            title = (graph_info.get("doc_title") or hit.get("title") or "").strip()
            doc_id = str(hit.get("doc_id") or "")
            family_key = (doc_id, title.lower() or doc_id or f"hit_{position}")
            if family_key not in family_hits:
                family_order.append(family_key)
            family_hits.setdefault(family_key, []).append((position, hit))

        role_priority = {"answer": 4, "direct": 4, "bridge": 3, "graph": 2, "symbolic": 2, "background": 1}
        compacted: List[Tuple[int, Dict[str, Any]]] = []
        compacted_away = 0

        for family_key in family_order:
            ordered_hits = family_hits[family_key]
            ordered_hits.sort(
                key=lambda item: (
                    item[1].get("chunk_index") if isinstance(item[1].get("chunk_index"), int) else 10**9,
                    item[0],
                )
            )
            family_compacted: List[Tuple[int, Dict[str, Any]]] = []
            for position, hit in ordered_hits:
                if not family_compacted:
                    merged_hit = dict(hit)
                    merged_hit["merged_chunk_indices"] = [hit.get("chunk_index")]
                    merged_hit["merged_source_types"] = [self._infer_hit_source_type(hit)]
                    family_compacted.append((position, merged_hit))
                    continue

                _, previous_hit = family_compacted[-1]
                previous_index = previous_hit.get("merged_chunk_indices", [previous_hit.get("chunk_index")])[-1]
                current_index = hit.get("chunk_index")
                same_doc = previous_hit.get("doc_id") == hit.get("doc_id")
                is_adjacent = (
                    same_doc
                    and isinstance(previous_index, int)
                    and isinstance(current_index, int)
                    and abs(current_index - previous_index) <= 1
                )
                same_family_duplicate = self._is_near_duplicate_chunk(
                    previous_hit.get("chunk_text", ""),
                    hit.get("chunk_text", ""),
                )
                if not is_adjacent and not same_family_duplicate:
                    merged_hit = dict(hit)
                    merged_hit["merged_chunk_indices"] = [hit.get("chunk_index")]
                    merged_hit["merged_source_types"] = [self._infer_hit_source_type(hit)]
                    family_compacted.append((position, merged_hit))
                    continue

                previous_hit["chunk_text"] = self._merge_context_text_pair(
                    previous_hit.get("chunk_text", ""),
                    hit.get("chunk_text", ""),
                )
                previous_hit["merged_chunk_indices"] = [
                    *previous_hit.get("merged_chunk_indices", [previous_hit.get("chunk_index")]),
                    hit.get("chunk_index"),
                ]
                merged_source_types = previous_hit.get("merged_source_types", [self._infer_hit_source_type(previous_hit)])
                current_source_type = self._infer_hit_source_type(hit)
                if current_source_type not in merged_source_types:
                    merged_source_types.append(current_source_type)
                previous_hit["merged_source_types"] = merged_source_types
                previous_hit["score"] = max(
                    _safe_numeric(previous_hit.get("score", 0.0)),
                    _safe_numeric(hit.get("score", 0.0)),
                )
                previous_hit["final_rank_score"] = max(
                    _safe_numeric(previous_hit.get("final_rank_score", 0.0)),
                    _safe_numeric(hit.get("final_rank_score", 0.0)),
                )
                previous_role = str(previous_hit.get("evidence_role") or previous_hit.get("role") or "background").lower()
                current_role = str(hit.get("evidence_role") or hit.get("role") or "background").lower()
                if role_priority.get(current_role, 0) > role_priority.get(previous_role, 0):
                    previous_hit["evidence_role"] = current_role
                    previous_hit["role"] = current_role
                previous_hit["is_corroborated"] = bool(previous_hit.get("is_corroborated")) or bool(hit.get("is_corroborated"))
                compacted_away += 1

            compacted.extend(family_compacted)

        compacted.sort(key=lambda item: item[0])
        return [hit for _, hit in compacted], compacted_away

    def _build_system_prompt(self, benchmark_grounded: bool = False) -> str:
        """
        Constructs the rigorous anti-hallucination system prompt.
        """
        if benchmark_grounded:
            return """You are AsterScope Copilot in benchmark answer mode.
Answer the user's question using ONLY the provided `<CONTEXT>` blocks.

STRICT RULES:
1. If the answer is not directly supported by the context, reply exactly: "I don't have enough information in the provided context to answer that."
2. Keep the answer brief and direct. Prefer the final answer in the first sentence.
3. Use only the minimum evidence needed. Do not mention irrelevant entities or background facts.
4. Every factual sentence must include an inline citation in the format [Doc: Title, Section: Section Name].
5. Do not use outside knowledge, speculation, or unstated assumptions.
"""
        return """You are AsterScope Copilot, a highly precise, compliance-focused enterprise AI assistant.
Your primary directive is to answer the user's query ONLY using the provided `<CONTEXT>` blocks.

STRICT GENERATION RULES (Must be followed exactly):
1. FACTUAL GROUNDING: You must base every single factual claim in your response entirely on the `<CONTEXT>`.
2. NO HALLUCINATION: If the answer cannot be confidently derived from the `<CONTEXT>`, you must explicitly state: "I don't have enough information in the provided context to answer that." Do not attempt to guess or use external pre-training knowledge.
3. GRAPH-FIRST CONSTRAINT: You must ALWAYS prioritize relationship data from the Graph Context (Shared Entities, Cross-Document Texts) over dense text chunks if contradictions exist. The Knowledge Graph is the absolute source of truth.
4. INLINE CITATIONS: Every claim or fact you state MUST include an inline citation matching the `[Source Marker]` provided with the chunk. Example format: "The onboarding process requires 3 signatures [Doc: HR Manual, Section: 1.2]."
5. STRUCTURED THOUGHT PROCESS: Before answering a complex query, you MUST briefly output a one-sentence plan of how you will synthesize the documents, and you MUST explicitly cite the graph relationships you are relying on (e.g., [Graph Thought: Found Entity A -> RELATES_TO -> Entity B]).
6. TONE: Professional, concise, and definitive.

If the user greets you or asks about your capabilities, you may respond naturally, but still reinforce your reliance on grounded data.
"""

    def _extract_cited_titles(self, answer: str) -> List[str]:
        titles = []
        seen = set()
        for raw_title in re.findall(r"\[Doc:\s*([^,\]]+)", answer or ""):
            title = raw_title.strip()
            lowered = title.lower()
            if title and lowered not in seen:
                titles.append(title)
                seen.add(lowered)
        return titles

    def _select_supporting_hits_with_debug(
        self,
        query: str,
        answer: str,
        hits: List[Dict[str, Any]],
        limit: int,
        follow_up_queries: Optional[List[str]] = None
    ) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
        if not hits or limit <= 0:
            return [], {
                "anchor_filtered_count": 0,
                "subject_anchored_count": 0,
                "follow_up_confirmed_count": 0,
                "unanchored_selected_count": 0,
            }

        cited_titles = [title.lower() for title in self._extract_cited_titles(answer)]
        query_terms = set(_normalize_terms(query))
        query_subjects = self._extract_query_subject_candidates(query)
        lowered_query_subjects = [subject.lower() for subject in query_subjects if subject]
        query_subject_terms = set()
        for subject in query_subjects:
            query_subject_terms.update(_normalize_terms(subject))
        abstention_markers = [
            "don't have enough information",
            "do not have enough information",
            "insufficient information",
        ]
        is_abstention = any(marker in (answer or "").lower() for marker in abstention_markers)
        answer_terms = set() if is_abstention else set(_normalize_terms(answer))
        answer_subjects = []
        if not is_abstention:
            query_subject_set = {subject.lower() for subject in query_subjects}
            for candidate in self._extract_query_subject_candidates(answer):
                lowered = candidate.lower()
                if lowered not in query_subject_set:
                    answer_subjects.append(candidate)
        lowered_answer_subjects = [subject.lower() for subject in answer_subjects if subject]
        focus_terms = {
            token for token in query_terms
            if token not in _QUERY_FOCUS_STOPWORDS and token not in query_subject_terms
        }
        scored_rows = []

        for position, hit in enumerate(hits):
            graph_info = hit.get("graph_context") or {}
            title = (graph_info.get("doc_title") or hit.get("title") or "").strip()
            body_text = hit.get("chunk_text", "") or ""
            combined_text = f"{title}\n{body_text}".strip().lower()
            title_terms = set(_normalize_terms(title))
            text_terms = set(_normalize_terms(body_text))
            combined_terms = title_terms | text_terms
            focus_overlap = len(combined_terms & focus_terms)
            query_subject_overlap = len(combined_terms & query_subject_terms)
            subject_phrase_anchor = any(
                subject and subject in combined_text
                for subject in lowered_query_subjects
            )
            answer_phrase_anchor = any(
                subject and subject in combined_text
                for subject in lowered_answer_subjects
            )
            follow_up_confirmed = self._is_follow_up_confirmed_hit(hit, follow_up_queries)
            cited_title_hit = bool(title and title.lower() in cited_titles)

            score = _safe_numeric(
                hit.get("final_rank_score"),
                _safe_numeric(
                    hit.get("cross_encoder_score"),
                    _safe_numeric(hit.get("rrf_score"), _safe_numeric(hit.get("score", 0.0)))
                )
            )
            entity_anchor = subject_phrase_anchor or answer_phrase_anchor or follow_up_confirmed

            if cited_title_hit:
                score += 3.4 if entity_anchor else 1.6
            if subject_phrase_anchor:
                score += 1.35
            if answer_phrase_anchor:
                score += 0.72
            if follow_up_confirmed:
                score += 0.42 if (focus_overlap > 0 or cited_title_hit or answer_phrase_anchor) else 0.12
            if title_terms:
                score += 0.35 * len(query_terms & title_terms)
                score += 0.25 * len(answer_terms & title_terms)
            if text_terms and answer_terms:
                score += min(1.0, 0.08 * len(answer_terms & text_terms))
            if is_abstention and text_terms:
                score += min(0.8, 0.05 * len(query_terms & text_terms))
            if focus_overlap:
                score += 0.16 * focus_overlap
            if query_subject_overlap:
                score += 0.18 * query_subject_overlap
            if query_subjects and not subject_phrase_anchor and not follow_up_confirmed:
                score -= 0.56
            if cited_title_hit and not entity_anchor:
                score -= 1.05
            if not entity_anchor and focus_overlap == 0 and cited_title_hit is False:
                score -= 0.72

            lowered_body = body_text.strip().lower()
            lowered_title = title.strip().lower()
            definition_like = bool(
                lowered_body and (
                    lowered_body.startswith(f"{lowered_title} is")
                    or lowered_body.startswith(f"{lowered_title} was")
                    or lowered_body.startswith("is a ")
                    or lowered_body.startswith("was a ")
                    or lowered_body.startswith("is an ")
                    or lowered_body.startswith("was an ")
                    or lowered_body.startswith("may refer to")
                )
            )
            if definition_like and not entity_anchor and focus_overlap == 0:
                score -= 0.28

            scored_rows.append({
                "hit": hit,
                "score": score,
                "position": position,
                "title": title,
                "title_key": title.lower(),
                "key": (title.lower(), hit.get("doc_id"), hit.get("chunk_index")),
                "cited_title_hit": cited_title_hit,
                "subject_phrase_anchor": subject_phrase_anchor,
                "answer_phrase_anchor": answer_phrase_anchor,
                "follow_up_confirmed": follow_up_confirmed,
                "entity_anchor": entity_anchor,
                "focus_overlap": focus_overlap,
            })

        scored_rows.sort(
            key=lambda row: (
                row["score"],
                row["entity_anchor"],
                row["follow_up_confirmed"],
                -row["position"],
            ),
            reverse=True
        )

        selected = []
        seen_keys = set()
        seen_titles = set()
        anchor_filtered_count = 0
        has_anchor_candidate = any(row["entity_anchor"] for row in scored_rows)
        best_anchor_score = max(
            (row["score"] for row in scored_rows if row["entity_anchor"]),
            default=scored_rows[0]["score"] if scored_rows else 0.0
        )

        def _try_select(row: Dict[str, Any]) -> bool:
            nonlocal anchor_filtered_count
            if row["key"] in seen_keys:
                return False
            if row["title"] and row["title_key"] in seen_titles:
                return False
            if (
                has_anchor_candidate
                and not row["entity_anchor"]
                and row["score"] < (best_anchor_score - 0.35)
            ):
                anchor_filtered_count += 1
                return False
            if (
                query_subjects
                and not row["entity_anchor"]
                and row["focus_overlap"] == 0
            ):
                anchor_filtered_count += 1
                return False
            if (
                selected
                and row["subject_phrase_anchor"]
                and not row["cited_title_hit"]
                and not row["answer_phrase_anchor"]
                and row["focus_overlap"] == 0
            ):
                anchor_filtered_count += 1
                return False
            selected.append(row)
            seen_keys.add(row["key"])
            if row["title"]:
                seen_titles.add(row["title_key"])
            return True

        for row in scored_rows:
            if len(selected) >= limit:
                break
            if row["subject_phrase_anchor"]:
                _try_select(row)

        for row in scored_rows:
            if len(selected) >= limit:
                break
            _try_select(row)

        if not selected and scored_rows:
            selected = [scored_rows[0]]

        debug = {
            "anchor_filtered_count": anchor_filtered_count,
            "subject_anchored_count": sum(1 for row in selected if row["subject_phrase_anchor"]),
            "follow_up_confirmed_count": sum(1 for row in selected if row["follow_up_confirmed"]),
            "unanchored_selected_count": sum(1 for row in selected if not row["entity_anchor"]),
        }
        return [row["hit"] for row in selected[:limit]], debug

    def _select_supporting_hits(
        self,
        query: str,
        answer: str,
        hits: List[Dict[str, Any]],
        limit: int,
        follow_up_queries: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        selected, _ = self._select_supporting_hits_with_debug(
            query,
            answer,
            hits,
            limit,
            follow_up_queries=follow_up_queries
        )
        return selected

    def _protect_supporting_seed_hits(
        self,
        query: str,
        hits: List[Dict[str, Any]],
        follow_up_queries: Optional[List[str]] = None,
        limit: Optional[int] = None
    ) -> Tuple[List[Dict[str, Any]], int]:
        if not hits:
            return [], 0

        query_terms = set(_normalize_terms(query))
        subject_candidates = self._extract_query_subject_candidates(query)
        lowered_subject_candidates = [subject.lower() for subject in subject_candidates if subject]
        subject_terms = set()
        for subject in subject_candidates:
            subject_terms.update(_normalize_terms(subject))
        focus_terms = {
            token for token in query_terms
            if token not in _QUERY_FOCUS_STOPWORDS and token not in subject_terms
        }

        scored_rows = []
        for position, hit in enumerate(hits):
            scores = self._score_dual_head_hit(query, hit)
            graph_info = hit.get("graph_context") or {}
            title = (graph_info.get("doc_title") or hit.get("title") or "").strip()
            body_text = hit.get("chunk_text", "") or ""
            if "\n" in body_text:
                body_text = body_text.split("\n", 1)[1]
            combined_text = f"{title}\n{body_text}".strip().lower()
            title_terms = set(_normalize_terms(title))
            body_terms = set(_normalize_terms(body_text))
            focus_overlap = len((title_terms | body_terms) & focus_terms)
            query_overlap = len((title_terms | body_terms) & query_terms)
            subject_overlap = len((title_terms | body_terms) & subject_terms)
            subject_phrase_anchor = any(
                subject and subject in combined_text
                for subject in lowered_subject_candidates
            )
            follow_up_confirmed = self._is_follow_up_confirmed_hit(hit, follow_up_queries)
            lowered_body = body_text.strip().lower()
            lowered_title = title.strip().lower()
            definition_like = bool(
                lowered_body and (
                    lowered_body.startswith(f"{lowered_title} is")
                    or lowered_body.startswith(f"{lowered_title} was")
                    or lowered_body.startswith("is a ")
                    or lowered_body.startswith("was a ")
                    or lowered_body.startswith("is an ")
                    or lowered_body.startswith("was an ")
                    or lowered_body.startswith("may refer to")
                )
            )

            quality_score = (
                max(scores["answer_score"], scores["bridge_score"])
                + (0.28 * min(scores["answer_score"], scores["bridge_score"]))
                + (0.20 * focus_overlap)
                + (0.26 * subject_overlap)
                + (0.42 if subject_phrase_anchor else 0.0)
                + (0.08 * query_overlap)
                + (0.28 if follow_up_confirmed else 0.0)
                - (0.18 if definition_like and focus_overlap == 0 and not follow_up_confirmed else 0.0)
                - (0.22 if focus_overlap == 0 and scores["bridge_score"] < 0.5 and not follow_up_confirmed else 0.0)
                - (0.26 if subject_terms and subject_overlap == 0 and not follow_up_confirmed and scores["bridge_score"] < 0.7 else 0.0)
                - (0.02 * position)
            )
            scored_rows.append({
                "hit": hit,
                "quality_score": quality_score,
                "focus_overlap": focus_overlap,
                "subject_overlap": subject_overlap,
                "subject_phrase_anchor": subject_phrase_anchor,
                "follow_up_confirmed": follow_up_confirmed,
                "answer_score": scores["answer_score"],
                "bridge_score": scores["bridge_score"],
            })

        scored_rows.sort(
            key=lambda row: (
                row["quality_score"],
                row["answer_score"] + row["bridge_score"],
                row["focus_overlap"],
            ),
            reverse=True
        )

        blocked = 0
        kept_rows = []
        best_quality = scored_rows[0]["quality_score"] if scored_rows else 0.0
        keep_limit = max(1, limit or len(hits))
        anchored_seed_kept = False

        for row in scored_rows:
            weak_seed = (
                row["focus_overlap"] == 0
                and not row["subject_phrase_anchor"]
                and not row["follow_up_confirmed"]
                and row["answer_score"] < 0.95
                and row["bridge_score"] < 0.55
            )
            anchorless_seed = (
                bool(subject_candidates)
                and not row["subject_phrase_anchor"]
                and not row["follow_up_confirmed"]
            )
            significantly_worse = row["quality_score"] < (best_quality - 0.45)
            if weak_seed and significantly_worse:
                blocked += 1
                continue
            if (
                anchored_seed_kept
                and not row["subject_phrase_anchor"]
                and not row["follow_up_confirmed"]
                and row["focus_overlap"] == 0
            ):
                blocked += 1
                continue
            if anchorless_seed and anchored_seed_kept:
                blocked += 1
                continue
            if anchorless_seed and len(kept_rows) >= 1 and not any(
                kept["subject_phrase_anchor"] or kept["follow_up_confirmed"]
                for kept in kept_rows
            ):
                blocked += 1
                continue
            kept_rows.append(row)
            if row["subject_phrase_anchor"] or row["follow_up_confirmed"]:
                anchored_seed_kept = True
            if len(kept_rows) >= keep_limit:
                break

        if not kept_rows:
            kept_rows = scored_rows[:1]
            blocked = max(0, len(scored_rows) - len(kept_rows))

        return [row["hit"] for row in kept_rows], blocked

    def _extract_benchmark_bridge_queries(self, query: str, hits: List[Dict[str, Any]]) -> List[str]:
        """
        Backward-compatible wrapper around generic bridge follow-up planning.
        The implementation intentionally avoids dataset-specific relation rules.
        """
        return self._deterministic_bridge_follow_up_queries(query, hits, query_graph=None)

    def _extract_bridge_entity_candidates(self, query: str, hits: List[Dict[str, Any]]) -> List[str]:
        query_terms = set(_normalize_terms(query))
        query_subject_candidates = self._extract_query_subject_candidates(query)
        lowered_query_subjects = [subject.lower() for subject in query_subject_candidates if subject]
        query_subject_terms = set()
        for subject in query_subject_candidates:
            query_subject_terms.update(_normalize_terms(subject))
        scored_candidates: Dict[str, Dict[str, Any]] = {}
        seen = set()

        def _score_hit(hit: Dict[str, Any], subject_anchor: bool) -> float:
            score = _safe_numeric(
                hit.get("final_rank_score"),
                _safe_numeric(
                    hit.get("cross_encoder_score"),
                    _safe_numeric(hit.get("rrf_score"), _safe_numeric(hit.get("score", 0.0)))
                )
            )
            title = ((hit.get("graph_context") or {}).get("doc_title") or hit.get("title") or "").strip()
            title_terms = set(_normalize_terms(title))
            if title_terms and title_terms.isdisjoint(query_terms):
                score += 0.5
            if query_subject_terms:
                score += 0.45 if subject_anchor else -0.35
            return score

        def _append(candidate: str, score: float):
            candidate = candidate.strip(" .,:;()[]{}\"'")
            lowered = candidate.lower()
            if not candidate:
                return
            candidate_terms = set(_normalize_terms(candidate))
            if not candidate_terms or candidate_terms.issubset(query_terms):
                return
            if len(candidate.split()) > 8:
                return
            if candidate_terms & _GENERIC_BRIDGE_TOKENS:
                return
            generic_overlap = candidate_terms & _GENERIC_BRIDGE_TOKENS
            if generic_overlap and len(generic_overlap) >= max(1, len(candidate_terms) // 2):
                return
            if len(candidate_terms) == 1 and next(iter(candidate_terms), "") in _GENERIC_BRIDGE_TOKENS:
                return
            existing = scored_candidates.get(lowered)
            if existing is not None and score <= existing["score"]:
                return
            if lowered not in seen:
                seen.add(lowered)
            scored_candidates[lowered] = {"candidate": candidate, "score": score}

        for position, hit in enumerate(hits[:20]):
            title = ((hit.get("graph_context") or {}).get("doc_title") or hit.get("title") or "").strip()
            text = hit.get("chunk_text", "") or ""
            combined_text = f"{title}\n{text}".lower()
            combined_terms = set(_normalize_terms(title)) | set(_normalize_terms(text))
            subject_phrase_anchor = any(
                subject and subject in combined_text
                for subject in lowered_query_subjects
            )
            subject_term_anchor = bool(combined_terms & query_subject_terms)
            subject_anchor = (not query_subject_terms) or subject_phrase_anchor or subject_term_anchor
            hit_score = _score_hit(hit, subject_anchor) - (0.03 * position)
            title = ((hit.get("graph_context") or {}).get("doc_title") or hit.get("title") or "").strip()
            if title:
                title_terms = set(_normalize_terms(title))
                if query_subject_terms and not subject_anchor and title_terms.isdisjoint(query_terms):
                    title = ""
            if title:
                _append(title, hit_score + 0.8)

            for alias_pattern in (
                r"(?:also known as|aka|a\.k\.a\.|known professionally as|better known as|stage name|birth name|born as)\s+([A-Z][A-Za-z0-9'`().-]+(?:\s+[A-Z][A-Za-z0-9'`().-]+){0,5})",
            ):
                if query_subject_terms and not subject_anchor and position > 3:
                    continue
                for match in re.findall(alias_pattern, text, flags=re.IGNORECASE):
                    alias_boost = 1.05 if subject_anchor else 0.28
                    _append(match, hit_score + alias_boost)

            for quoted in re.findall(r"['\"]([^'\"]+)['\"]", text):
                if query_subject_terms and not subject_anchor:
                    continue
                _append(quoted, hit_score + 0.35)

            for match in re.findall(r"(?:[A-Z][A-Za-z'`-]+(?:\s+[A-Z][A-Za-z'`-]+){0,4})", text):
                if query_subject_terms and not subject_anchor:
                    continue
                _append(match, hit_score)

        ranked_candidates = sorted(
            scored_candidates.values(),
            key=lambda item: item["score"],
            reverse=True
        )
        return [item["candidate"] for item in ranked_candidates[:8]]

    def _derive_follow_up_query_hints(
        self,
        query: str,
        query_graph: Optional[List[Dict[str, Any]]] = None
    ) -> List[str]:
        lowered_query = query.lower()
        reduced_query = lowered_query
        for subject in self._extract_query_subject_candidates(query):
            reduced_query = re.sub(re.escape(subject.lower()), " ", reduced_query, flags=re.IGNORECASE)

        hints: List[str] = []

        def _append(hint: str):
            normalized = " ".join(_normalize_terms(hint))
            if normalized and normalized not in hints:
                hints.append(normalized)

        focus_terms = [
            token for token in _normalize_terms(reduced_query)
            if token not in _QUERY_FOCUS_STOPWORDS
        ]
        if focus_terms:
            _append(" ".join(focus_terms[:6]))
            if len(focus_terms) > 2:
                _append(" ".join(focus_terms[-3:]))

        for triplet in query_graph or []:
            relation = str(triplet.get("relation", "")).strip().replace("_", " ").lower()
            obj = str(triplet.get("object", "")).strip().replace("_", " ").lower()
            _append(relation)
            _append(obj)

        return hints[:4]

    def _extract_planner_bridge_candidates(self, query: str, hits: List[Dict[str, Any]]) -> List[str]:
        query_terms = set(_normalize_terms(query))
        query_subject_candidates = self._extract_query_subject_candidates(query)
        lowered_query_subjects = [subject.lower() for subject in query_subject_candidates if subject]
        query_subject_terms = set()
        for subject in query_subject_candidates:
            query_subject_terms.update(_normalize_terms(subject))
        candidates: Dict[str, Dict[str, Any]] = {}

        def _append(candidate: str, score: float):
            cleaned = (candidate or "").strip(" .,:;()[]{}\"'")
            lowered = cleaned.lower()
            if not cleaned or len(cleaned.split()) > 8:
                return
            terms = set(_normalize_terms(cleaned))
            if not terms or len(terms) == 1 and next(iter(terms), "") in _GENERIC_BRIDGE_TOKENS:
                return
            generic_overlap = terms & _GENERIC_BRIDGE_TOKENS
            if generic_overlap and len(generic_overlap) >= max(1, len(terms) // 2):
                return
            if terms.issubset(query_terms) and len(terms) <= 2:
                return
            existing = candidates.get(lowered)
            if existing is not None:
                existing["score"] = max(existing["score"], score)
                existing["mentions"] += 1
                return
            candidates[lowered] = {"candidate": cleaned, "score": score, "mentions": 1}

        for position, hit in enumerate(hits[:10]):
            base_score = _safe_numeric(
                hit.get("final_rank_score"),
                _safe_numeric(
                    hit.get("cross_encoder_score"),
                    _safe_numeric(hit.get("rrf_score"), _safe_numeric(hit.get("score", 0.0)))
                )
            ) - (0.03 * position)
            title = ((hit.get("graph_context") or {}).get("doc_title") or hit.get("title") or "").strip()
            text = hit.get("chunk_text", "") or ""
            combined_text = f"{title}\n{text}".lower()
            title_terms = set(_normalize_terms(title))
            text_terms = set(_normalize_terms(text))
            combined_terms = title_terms | text_terms
            subject_phrase_anchor = any(
                subject and subject in combined_text
                for subject in lowered_query_subjects
            )
            subject_term_anchor = bool(combined_terms & query_subject_terms)
            subject_anchor = (not query_subject_terms) or subject_phrase_anchor or subject_term_anchor
            if query_subject_terms:
                base_score += 0.42 if subject_anchor else -0.32

            if title:
                if query_subject_terms and not subject_anchor and title_terms.isdisjoint(query_terms):
                    title = ""
                if not title:
                    continue
                novelty_bonus = 0.35 if title_terms and title_terms.isdisjoint(query_terms) else 0.0
                _append(title, base_score + 0.55 + novelty_bonus)

            for quoted in re.findall(r"['\"]([^'\"]+)['\"]", text):
                if query_subject_terms and not subject_anchor:
                    continue
                _append(quoted, base_score + 0.25)

            for span in re.findall(r"(?:[A-Z][A-Za-z'`-]+(?:\s+[A-Z][A-Za-z'`()-]+){0,4})", text):
                if query_subject_terms and not subject_anchor:
                    continue
                _append(span, base_score + 0.12)

        ranked = sorted(
            candidates.values(),
            key=lambda item: (item["score"] + (0.12 * min(item["mentions"], 3)), item["mentions"]),
            reverse=True
        )
        return [item["candidate"] for item in ranked[:8]]

    def _derive_planner_focus_hints(
        self,
        query: str,
        query_graph: Optional[List[Dict[str, Any]]] = None
    ) -> List[str]:
        hints: List[str] = []
        seen = set()
        query_terms = [
            token for token in _normalize_terms(query)
            if token not in _QUERY_FOCUS_STOPWORDS
        ]

        def _append(tokens: List[str]):
            normalized = " ".join(token for token in tokens if token)
            if not normalized or normalized in seen:
                return
            seen.add(normalized)
            hints.append(normalized)

        if query_terms:
            _append(query_terms[:6])
            if len(query_terms) > 3:
                _append(query_terms[-3:])

        for triplet in query_graph or []:
            relation = [token for token in _normalize_terms(str(triplet.get("relation", ""))) if token]
            obj = [token for token in _normalize_terms(str(triplet.get("object", ""))) if token]
            _append(relation[:4])
            _append(obj[:4])

        return hints[:4]

    def _build_subject_anchor_follow_up_queries(self, query: str, focus_hints: List[str]) -> List[str]:
        subjects = self._extract_query_subject_candidates(query)
        if not subjects:
            return []

        ordered_focus_terms: List[str] = []
        seen_focus = set()
        for hint in focus_hints:
            for token in _normalize_terms(hint):
                if token in _QUERY_FOCUS_STOPWORDS or token in seen_focus:
                    continue
                seen_focus.add(token)
                ordered_focus_terms.append(token)

        anchored_queries: List[str] = []
        seen = set()
        for subject in subjects[:2]:
            subject_terms = set(_normalize_terms(subject))
            suffix_terms = [
                token for token in ordered_focus_terms
                if token not in subject_terms
            ][:3]
            candidate = subject if not suffix_terms else f"{subject} {' '.join(suffix_terms)}"
            lowered = candidate.lower()
            if lowered in seen or lowered == query.lower().strip():
                continue
            anchored_queries.append(candidate)
            seen.add(lowered)

        return anchored_queries[:2]

    def _merge_priority_queries(self, priority_queries: List[str], fallback_queries: List[str], limit: int = 3) -> List[str]:
        merged: List[str] = []
        seen = set()
        for candidate in [*(priority_queries or []), *(fallback_queries or [])]:
            normalized = (candidate or "").strip()
            lowered = normalized.lower()
            if not normalized or lowered in seen:
                continue
            merged.append(normalized)
            seen.add(lowered)
            if len(merged) >= max(1, limit):
                break
        return merged[: max(1, limit)]

    @staticmethod
    def _resolve_feature_flag(name: str, default: bool) -> bool:
        raw_value = os.getenv(name)
        if raw_value is None:
            return default
        return raw_value.lower() in {"1", "true", "yes"}

    def _should_use_early_second_hop(
        self,
        search_query: str,
        sub_queries: List[str],
        query_graph: Optional[List[Dict[str, Any]]],
        initial_hits: List[Dict[str, Any]],
        initial_chain_mode: str,
    ) -> Tuple[bool, str]:
        chain_mode = (initial_chain_mode or "light").lower()
        if chain_mode == "bypass":
            return False, "bypass_chain_mode"

        bridge_hits = 0
        bridge_signal_sum = 0.0
        for hit in initial_hits[:8]:
            bridge_signal = max(
                _safe_numeric(hit.get("chain_bridge_signal"), 0.0),
                _safe_numeric(hit.get("bridge_score"), 0.0),
            )
            is_bridge_member = str(hit.get("primary_chain_member_role") or "").lower() == "bridge"
            if bridge_signal >= 0.18 or is_bridge_member or int(hit.get("best_chain_length") or 1) > 1:
                bridge_hits += 1
                bridge_signal_sum += bridge_signal

        avg_bridge_signal = bridge_signal_sum / max(1, bridge_hits)
        has_planner_decomposition = len(sub_queries) > 1 or bool(query_graph)
        has_query_multi_hop_signal = self.retriever._is_multi_hop_query(search_query)

        if has_planner_decomposition:
            return True, "planner_or_query_graph"
        if chain_mode == "full":
            return True, "full_chain_mode"
        if has_query_multi_hop_signal and bridge_hits >= 1:
            return True, "query_and_bridge_signal"
        if bridge_hits >= 2 and avg_bridge_signal >= 0.16:
            return True, "multiple_bridge_hits"
        return False, "insufficient_bridge_pressure"

    def _extract_alias_bridge_candidates(self, query: str, hits: List[Dict[str, Any]]) -> List[str]:
        query_terms = set(_normalize_terms(query))
        query_subject_candidates = self._extract_query_subject_candidates(query)
        lowered_query_subjects = [subject.lower() for subject in query_subject_candidates if subject]
        query_subject_terms = set()
        for subject in query_subject_candidates:
            query_subject_terms.update(_normalize_terms(subject))
        candidates: Dict[str, Dict[str, Any]] = {}

        alias_patterns = [
            r"(?:also known as|aka|a\.k\.a\.|known professionally as|better known as|stage name|birth name|born as)\s+([A-Z][A-Za-z0-9'`().-]+(?:\s+[A-Z][A-Za-z0-9'`().-]+){0,5})",
        ]

        def _append(candidate: str, score: float):
            cleaned = (candidate or "").strip(" .,:;()[]{}\"'")
            lowered = cleaned.lower()
            if not cleaned or len(cleaned.split()) > 8:
                return
            terms = set(_normalize_terms(cleaned))
            if not terms or terms.issubset(query_terms):
                return
            generic_overlap = terms & _GENERIC_BRIDGE_TOKENS
            if generic_overlap and len(generic_overlap) >= max(1, len(terms) // 2):
                return
            if len(terms) == 1 and next(iter(terms), "") in _GENERIC_BRIDGE_TOKENS:
                return
            existing = candidates.get(lowered)
            if existing is not None:
                existing["score"] = max(existing["score"], score)
                existing["mentions"] += 1
                return
            candidates[lowered] = {"candidate": cleaned, "score": score, "mentions": 1}

        for position, hit in enumerate(hits[:10]):
            base_score = _safe_numeric(
                hit.get("final_rank_score"),
                _safe_numeric(
                    hit.get("cross_encoder_score"),
                    _safe_numeric(hit.get("rrf_score"), _safe_numeric(hit.get("score", 0.0)))
                )
            ) - (0.03 * position)
            title = ((hit.get("graph_context") or {}).get("doc_title") or hit.get("title") or "").strip()
            text = (hit.get("chunk_text") or "").replace("\n", " ")
            combined_text = f"{title}\n{text}".lower()
            combined_terms = set(_normalize_terms(title)) | set(_normalize_terms(text))
            subject_phrase_anchor = any(
                subject and subject in combined_text
                for subject in lowered_query_subjects
            )
            subject_term_anchor = bool(combined_terms & query_subject_terms)
            subject_anchor = (not query_subject_terms) or subject_phrase_anchor or subject_term_anchor
            if query_subject_terms and not subject_anchor and position > 2:
                continue
            for pattern in alias_patterns:
                for match in re.findall(pattern, text, flags=re.IGNORECASE):
                    alias_boost = 0.95 if subject_anchor else 0.22
                    _append(match, base_score + alias_boost)

        ranked = sorted(
            candidates.values(),
            key=lambda item: (item["score"] + (0.08 * min(item["mentions"], 3)), item["mentions"]),
            reverse=True
        )
        return [item["candidate"] for item in ranked[:6]]

    def _refine_bridge_queries_for_targeting(
        self,
        query: str,
        queries: List[str],
        entity_candidates: List[str],
        focus_hints: List[str],
        alias_candidates: Optional[List[str]] = None
    ) -> List[str]:
        if not queries:
            return []

        def _sanitize_candidate(raw_query: str) -> str:
            text = (raw_query or "").replace("\n", " ").strip()
            if not text:
                return ""
            text = re.sub(
                r"\b(?:title|snippet|evidence|bridge candidates|relation hints)\s*:\s*",
                " ",
                text,
                flags=re.IGNORECASE,
            )
            text = re.sub(r"^\s*(?:\d+[\).\s-]*)+", "", text)
            text = re.sub(r"^(?:search(?: for)?|find|look\s*up|retrieve|query)\s+", "", text, flags=re.IGNORECASE)
            parts = re.findall(r"[A-Za-z0-9][A-Za-z0-9'`().-]*", text)
            if not parts:
                return ""
            return " ".join(parts[:12]).strip()

        focus_terms = set()
        ordered_focus_terms: List[str] = []
        for hint in focus_hints:
            for token in _normalize_terms(hint):
                if token in focus_terms:
                    continue
                focus_terms.add(token)
                ordered_focus_terms.append(token)

        entity_sets: List[Tuple[str, set]] = []
        for entity in entity_candidates:
            tokens = set(_normalize_terms(entity))
            if tokens:
                entity_sets.append((entity, tokens))
        alias_set = {candidate.lower() for candidate in (alias_candidates or [])}
        query_subject_terms = set()
        for subject in self._extract_query_subject_candidates(query):
            query_subject_terms.update(_normalize_terms(subject))
        query_lower = query.lower().strip()

        refined_rows = []
        seen = set()
        for position, raw_query in enumerate(queries):
            normalized = _sanitize_candidate(raw_query)
            if not normalized:
                continue

            candidate_text = normalized
            candidate_terms = set(_normalize_terms(candidate_text))
            if not candidate_terms:
                continue
            if len(candidate_terms & _GENERIC_BRIDGE_TOKENS) >= max(1, len(candidate_terms) // 2):
                continue

            best_entity = ""
            best_entity_terms = set()
            best_overlap = 0
            for entity, entity_terms in entity_sets:
                overlap = len(candidate_terms & entity_terms)
                if entity.lower() in candidate_text.lower():
                    overlap += 1
                if overlap > best_overlap:
                    best_entity = entity
                    best_entity_terms = entity_terms
                    best_overlap = overlap

            if entity_sets and not best_entity:
                continue

            candidate_focus = []
            if best_entity:
                for token in _normalize_terms(candidate_text):
                    if token in focus_terms and token not in best_entity_terms and token not in candidate_focus:
                        candidate_focus.append(token)
                if not candidate_focus:
                    for token in ordered_focus_terms:
                        if token in best_entity_terms or token in candidate_focus:
                            continue
                        candidate_focus.append(token)
                        if len(candidate_focus) >= 3:
                            break
                candidate_text = best_entity
                if candidate_focus:
                    candidate_text = f"{candidate_text} {' '.join(candidate_focus[:3])}".strip()

            candidate_terms = set(_normalize_terms(candidate_text))
            entity_overlap = max((len(candidate_terms & entity_terms) for _, entity_terms in entity_sets), default=0)
            if entity_sets and entity_overlap == 0:
                continue
            focus_overlap = len(candidate_terms & focus_terms)
            lowered = candidate_text.lower()
            if lowered == query_lower or lowered in seen:
                continue
            seen.add(lowered)

            alias_bonus = 0.4 if any(alias in lowered for alias in alias_set) else 0.0
            mismatch_penalty = 0.0
            if (
                best_entity
                and query_subject_terms
                and best_entity_terms
                and best_entity_terms.isdisjoint(query_subject_terms)
                and best_entity.lower() not in alias_set
            ):
                mismatch_penalty = 0.55
            length_penalty = 0.05 * max(0, len(candidate_terms) - 6)
            score = (
                1.55 * entity_overlap
                + 0.55 * focus_overlap
                + alias_bonus
                - (0.03 * position)
                - mismatch_penalty
                - length_penalty
            )
            refined_rows.append((score, candidate_text))

        if not refined_rows:
            fallback = []
            fallback_seen = set()
            for entity, entity_terms in entity_sets[:4]:
                suffix_terms = [
                    token for token in ordered_focus_terms
                    if token not in entity_terms
                ][:2]
                candidate_text = entity if not suffix_terms else f"{entity} {' '.join(suffix_terms)}"
                lowered = candidate_text.lower()
                if lowered in fallback_seen or lowered == query_lower:
                    continue
                fallback.append(candidate_text)
                fallback_seen.add(lowered)
            return fallback[:4]
        refined_rows.sort(key=lambda item: item[0], reverse=True)
        return [query for _, query in refined_rows[:4]]

    def _count_entity_targeted_queries(self, queries: List[str], entities: List[str]) -> int:
        if not queries or not entities:
            return 0
        entity_sets = [set(_normalize_terms(entity)) for entity in entities if entity]
        entity_sets = [tokens for tokens in entity_sets if tokens]
        if not entity_sets:
            return 0
        count = 0
        for query in queries:
            query_terms = set(_normalize_terms(query))
            if any(query_terms & entity_terms for entity_terms in entity_sets):
                count += 1
        return count

    def _summarize_bridge_targeting_hits(
        self,
        query: str,
        hits: List[Dict[str, Any]],
        bridge_queries: List[str],
        bridge_entities: List[str],
    ) -> Dict[str, int]:
        if not hits or not bridge_queries:
            return {
                "bridge_targeting_hits": 0,
                "answer_bearing_bridge_hits": 0,
                "bridge_entity_family_chunk_miss": 0,
            }

        focus_terms = set()
        for hint in self._derive_planner_focus_hints(query, query_graph=None):
            focus_terms.update(_normalize_terms(hint))

        bridge_query_set = {q.lower().strip() for q in bridge_queries if isinstance(q, str) and q.strip()}
        entity_sets = [
            set(_normalize_terms(entity))
            for entity in bridge_entities
            if isinstance(entity, str) and entity.strip()
        ]
        entity_sets = [tokens for tokens in entity_sets if tokens]
        bridge_anchor_terms = set()
        for bridge_query in bridge_queries:
            for token in _normalize_terms(bridge_query):
                if token in _QUERY_FOCUS_STOPWORDS:
                    continue
                bridge_anchor_terms.add(token)

        targeting_hits = 0
        answer_bearing_hits = 0
        for hit in hits:
            retrieval_queries = [
                rq.strip().lower()
                for rq in (hit.get("retrieval_queries") or [])
                if isinstance(rq, str) and rq.strip()
            ]
            if not any(rq in bridge_query_set for rq in retrieval_queries):
                continue
            title = ((hit.get("graph_context") or {}).get("doc_title") or hit.get("title") or "").strip()
            text = hit.get("chunk_text") or ""
            if "\n" in text:
                text = text.split("\n", 1)[1]
            title_terms = set(_normalize_terms(title))
            text_terms = set(_normalize_terms(text))
            combined_terms = title_terms | text_terms
            if entity_sets:
                entity_anchor = any(combined_terms & entity_terms for entity_terms in entity_sets)
            else:
                entity_anchor = len(combined_terms & bridge_anchor_terms) >= 2
            if not entity_anchor:
                continue
            targeting_hits += 1
            if combined_terms & focus_terms:
                answer_bearing_hits += 1

        return {
            "bridge_targeting_hits": targeting_hits,
            "answer_bearing_bridge_hits": answer_bearing_hits,
            "bridge_entity_family_chunk_miss": 1 if targeting_hits > 0 and answer_bearing_hits == 0 else 0,
        }

    def _prune_bridge_queries_by_retrieval_signal(
        self,
        search_query: str,
        retrieval_top_k: int,
        sub_queries: List[str],
        bridge_queries: List[str],
        bridge_entities: List[str],
        query_graph: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[List[str], Dict[str, Any]]:
        if len(bridge_queries) <= 2:
            return bridge_queries[:], {
                "bridge_query_pruned_from": len(bridge_queries),
                "bridge_query_pruned_to": len(bridge_queries),
                "bridge_query_branch_scores": {},
            }

        branch_rows: List[Tuple[float, str]] = []
        branch_scores: Dict[str, Dict[str, float]] = {}
        branch_top_k = max(8, min(retrieval_top_k, 12))
        for candidate_query in bridge_queries[:4]:
            candidate_pool = self.retriever.collect_candidate_pool(
                search_query,
                top_k=branch_top_k,
                additional_queries=sub_queries[1:] + [candidate_query],
                include_follow_ups=False,
            )
            branch_hits = self.retriever.finalize_candidates(
                search_query,
                candidate_pool,
                top_k=branch_top_k,
                query_graph=query_graph,
            )
            targeting_summary = self._summarize_bridge_targeting_hits(
                search_query,
                branch_hits,
                [candidate_query],
                bridge_entities,
            )
            answer_bearing = float(targeting_summary["answer_bearing_bridge_hits"])
            targeted = float(targeting_summary["bridge_targeting_hits"])
            miss_penalty = float(targeting_summary["bridge_entity_family_chunk_miss"])
            top_score = max(
                (
                    _safe_numeric(
                        hit.get("final_rank_score"),
                        _safe_numeric(hit.get("cross_encoder_score"), _safe_numeric(hit.get("score"), 0.0)),
                    )
                    for hit in branch_hits[:4]
                ),
                default=0.0,
            )
            score = (2.0 * answer_bearing) + (0.8 * targeted) + (0.25 * top_score) - (0.75 * miss_penalty)
            branch_rows.append((score, candidate_query))
            branch_scores[candidate_query] = {
                "score": round(score, 4),
                "answer_bearing_bridge_hits": answer_bearing,
                "bridge_targeting_hits": targeted,
                "bridge_entity_family_chunk_miss": miss_penalty,
                "top_rank_score": round(top_score, 4),
            }

        branch_rows.sort(key=lambda item: item[0], reverse=True)
        kept_queries = [query for _, query in branch_rows[:2]]
        # Preserve any untouched tail query only if pruning would collapse to a single option.
        if len(kept_queries) < 2:
            for query in bridge_queries:
                if query not in kept_queries:
                    kept_queries.append(query)
                if len(kept_queries) >= 2:
                    break

        return kept_queries[:2], {
            "bridge_query_pruned_from": len(bridge_queries),
            "bridge_query_pruned_to": len(kept_queries[:2]),
            "bridge_query_branch_scores": branch_scores,
        }

    def _build_evidence_conditioned_follow_up_queries(
        self,
        query: str,
        hits: List[Dict[str, Any]],
        relation_hints: List[str],
        entity_candidates: List[str],
    ) -> List[str]:
        if not hits:
            return []

        ranked_hits = sorted(
            hits,
            key=lambda hit: (
                _safe_numeric(
                    hit.get("final_rank_score"),
                    _safe_numeric(hit.get("cross_encoder_score"), _safe_numeric(hit.get("score"), 0.0)),
                )
                + (0.35 * _safe_numeric(hit.get("chain_support_signal"), 0.0))
                + (0.25 * _safe_numeric(hit.get("answer_score"), 0.0))
            ),
            reverse=True,
        )
        seen = set()
        conditioned_queries: List[str] = []
        query_lower = query.lower().strip()

        def _append(candidate: str) -> None:
            normalized = (candidate or "").strip()
            lowered = normalized.lower()
            if not normalized or lowered == query_lower or lowered in seen:
                return
            seen.add(lowered)
            conditioned_queries.append(normalized)

        for hit in ranked_hits[:3]:
            title = (((hit.get("graph_context") or {}).get("doc_title")) or hit.get("title") or "").strip()
            if not title:
                continue
            title_terms = set(_normalize_terms(title))
            _append(title)
            for hint in relation_hints[:2]:
                hint_terms = [token for token in _normalize_terms(hint) if token not in title_terms][:3]
                if not hint_terms:
                    continue
                _append(f"{title} {' '.join(hint_terms)}")

        for entity in entity_candidates[:3]:
            _append(entity)
            for hint in relation_hints[:1]:
                _append(f"{entity} {hint}")

        return conditioned_queries[:3]

    def _generic_bridge_follow_up_queries(
        self,
        query: str,
        hits: List[Dict[str, Any]],
        query_graph: Optional[List[Dict[str, Any]]] = None
    ) -> List[str]:
        entities = self._extract_planner_bridge_candidates(query, hits)
        focus_hints = self._derive_planner_focus_hints(query, query_graph)
        queries: List[str] = []
        seen = set()

        for entity in entities[:4]:
            entity_clean = entity.strip()
            entity_lower = entity_clean.lower()
            if entity_lower and entity_lower != query.lower() and entity_lower not in seen:
                queries.append(entity_clean)
                seen.add(entity_lower)
            for hint in focus_hints[:2]:
                candidate = f"{entity_clean} {hint}".strip()
                lowered = candidate.lower()
                if not candidate or lowered == query.lower() or lowered in seen:
                    continue
                queries.append(candidate)
                seen.add(lowered)

        return queries[:3]

    def _deterministic_bridge_follow_up_queries(
        self,
        query: str,
        hits: List[Dict[str, Any]],
        query_graph: Optional[List[Dict[str, Any]]] = None
    ) -> List[str]:
        entities = self._extract_bridge_entity_candidates(query, hits)
        hints = self._derive_follow_up_query_hints(query, query_graph)
        queries: List[str] = []
        seen = set()

        for entity in entities[:4]:
            base_entity = entity.strip()
            entity_lower = base_entity.lower()
            if entity_lower not in seen:
                queries.append(base_entity)
                seen.add(entity_lower)
            for hint in hints[:3]:
                candidate = f"{base_entity} {hint}".strip()
                lowered = candidate.lower()
                if lowered == query.lower() or lowered in seen:
                    continue
                queries.append(candidate)
                seen.add(lowered)

        return queries[:3]

    def _extract_query_subject_candidates(self, query: str) -> List[str]:
        subjects: List[str] = []
        seen = set()

        def _append(candidate: str):
            candidate = (candidate or "").strip(" ?.,:;()[]{}\"'")
            lowered = candidate.lower()
            if not candidate or lowered in seen:
                return
            if len(candidate.split()) > 8:
                return
            subjects.append(candidate)
            seen.add(lowered)

        for quoted in re.findall(r"['\"]([^'\"]+)['\"]", query or ""):
            _append(quoted)

        for pattern in (
            r"\bof ([A-Z][A-Za-z'`-]+(?:\s+[A-Z][A-Za-z'`-]+){0,5})\??$",
            r"\bin the film ([A-Z][A-Za-z'`-]+(?:\s+[A-Z][A-Za-z'`()-]+){0,5})\??$",
            r"\bnarrator of ([A-Z][A-Za-z'`-]+(?:\s+[A-Z][A-Za-z'`()-]+){0,5})\??$",
        ):
            match = re.search(pattern, query or "")
            if match:
                _append(match.group(1))

        for match in re.findall(r"(?:[A-Z][A-Za-z'`-]+(?:\s+[A-Z][A-Za-z'`-]+){1,5})", query or ""):
            _append(match)

        return subjects[:6]

    def _augment_benchmark_sub_queries(self, query: str, sub_queries: List[str]) -> List[str]:
        augmented = list(sub_queries)

        def _append(candidate: str):
            candidate = candidate.strip()
            if candidate and candidate not in augmented:
                augmented.append(candidate)

        for subject in self._extract_query_subject_candidates(query)[:3]:
            _append(subject)
            _append(f"\"{subject}\"")

        return augmented[:6]

    def _score_dual_head_hit(self, search_query: str, hit: Dict[str, Any]) -> Dict[str, float]:
        query_terms = set(_normalize_terms(search_query))
        title = (((hit.get("graph_context") or {}).get("doc_title")) or hit.get("title") or "").strip()
        title_terms = set(_normalize_terms(title))
        text_terms = set(_normalize_terms(hit.get("chunk_text") or ""))
        base_score = _safe_numeric(
            hit.get("final_rank_score"),
            _safe_numeric(hit.get("cross_encoder_score"), _safe_numeric(hit.get("score", 0.0)))
        )

        retrieval_queries = [
            rq.strip() for rq in (hit.get("retrieval_queries") or [])
            if isinstance(rq, str) and rq.strip()
        ]
        extra_queries = [rq for rq in retrieval_queries if rq.lower() != search_query.lower()]
        bridge_terms = set()
        for rq in extra_queries:
            bridge_terms.update(_normalize_terms(rq))

        answer_score = (
            base_score
            + 0.18 * len(title_terms & query_terms)
            + 0.06 * len(text_terms & query_terms)
        )
        bridge_score = (
            (0.45 if extra_queries else 0.0)
            + 0.22 * len(title_terms & bridge_terms)
            + 0.08 * len(text_terms & bridge_terms)
        )
        if extra_queries and title_terms and title_terms.isdisjoint(query_terms) and (title_terms & bridge_terms):
            bridge_score += 0.55
        if extra_queries and title_terms and (title_terms & query_terms):
            answer_score += 0.12

        joint_score = (0.65 * answer_score) + (0.55 * bridge_score)
        return {
            "answer_score": answer_score,
            "bridge_score": bridge_score,
            "joint_score": joint_score,
        }

    def _select_dual_head_hits(
        self,
        search_query: str,
        hits: List[Dict[str, Any]],
        limit: int,
        max_chunks_per_title: int
    ) -> List[Dict[str, Any]]:
        if not hits or limit <= 0:
            return []

        scored_hits = []
        for position, hit in enumerate(hits):
            scores = self._score_dual_head_hit(search_query, hit)
            hit["answer_score"] = scores["answer_score"]
            hit["bridge_score"] = scores["bridge_score"]
            hit["joint_score"] = scores["joint_score"]
            scored_hits.append((scores["joint_score"], -position, hit))

        scored_hits.sort(reverse=True)

        selected = []
        seen_keys = set()
        title_counts: Dict[str, int] = {}
        for _, _, hit in scored_hits:
            title = (((hit.get("graph_context") or {}).get("doc_title")) or hit.get("title") or "").strip()
            title_key = title.lower() or f"{hit.get('doc_id')}_{hit.get('chunk_index')}"
            key = (hit.get("doc_id"), hit.get("chunk_index"))
            if key in seen_keys:
                continue
            if title and title_counts.get(title_key, 0) >= max(1, max_chunks_per_title):
                continue
            selected.append(hit)
            seen_keys.add(key)
            title_counts[title_key] = title_counts.get(title_key, 0) + 1
            if len(selected) >= limit:
                break

        return selected[:limit]

    def _is_near_duplicate_chunk(self, left_text: str, right_text: str) -> bool:
        left_tokens = _normalize_terms(left_text)[:32]
        right_tokens = _normalize_terms(right_text)[:32]
        if not left_tokens or not right_tokens:
            return False
        if left_tokens[:20] == right_tokens[:20]:
            return True
        left_set = set(left_tokens)
        right_set = set(right_tokens)
        overlap = len(left_set & right_set)
        union = len(left_set | right_set)
        return bool(union) and (overlap / union) >= 0.82

    def _is_follow_up_confirmed_hit(
        self,
        hit: Dict[str, Any],
        follow_up_queries: Optional[List[str]] = None
    ) -> bool:
        if not follow_up_queries:
            return False
        follow_up_set = {query.strip().lower() for query in follow_up_queries if isinstance(query, str) and query.strip()}
        if not follow_up_set:
            return False
        retrieval_queries = [
            query.strip().lower()
            for query in (hit.get("retrieval_queries") or [])
            if isinstance(query, str) and query.strip()
        ]
        return any(query in follow_up_set for query in retrieval_queries)

    def _build_answer_messages(
        self,
        query: str,
        context_str: str,
        chat_history: List[Dict[str, str]],
        benchmark_grounded: bool = False
    ) -> List[Dict[str, str]]:
        messages = [{
            "role": "system",
            "content": self._build_system_prompt(
                benchmark_grounded=benchmark_grounded
            )
        }]

        for msg in chat_history:
            messages.append({"role": msg["role"], "content": msg["content"]})

        final_user_content = (
            f"Please answer the following User Query precisely, using ONLY the provided Source Context.\n\n"
            f"<Source Context Block>\n{context_str}\n</Source Context Block>\n\n"
            f"User Query: {query}"
        )
        if benchmark_grounded:
            final_user_content += (
                "\n\nReturn a short final answer first, followed by at most one brief evidence sentence. "
                "Cite only the documents you actually used."
        )
        messages.append({"role": "user", "content": final_user_content})
        return messages

    def _extract_brief_answer_text(self, answer: str) -> str:
        text = (answer or "").strip()
        if not text:
            return ""
        if self._is_refusal_answer(text):
            return "I don't have enough information in the provided context to answer that."

        text = re.sub(r"\[Doc:\s*[^\]]+\]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        text = re.sub(r"^(?:answer|short answer|final answer)\s*:\s*", "", text, flags=re.IGNORECASE)
        first_sentence = re.split(r"(?<=[.!?])\s+", text, maxsplit=1)[0].strip()
        if first_sentence:
            text = first_sentence
        return text.strip(" \"'")

    @staticmethod
    def _is_refusal_answer(answer: str) -> bool:
        lowered = (answer or "").lower()
        refusal_markers = (
            "don't have enough information",
            "do not have enough information",
            "insufficient information",
            "not enough information",
            "cannot be determined from the provided context",
        )
        return any(marker in lowered for marker in refusal_markers)

    def _build_benchmark_answer_projection_messages(
        self,
        query: str,
        context_str: str,
        draft_answer: str,
    ) -> List[Dict[str, str]]:
        return [
            {
                "role": "system",
                "content": (
                    "You are a benchmark answer projector.\n"
                    "Given a user question, grounded source context, and a draft grounded answer, "
                    "extract the shortest final answer string supported by the context.\n"
                    "Rules:\n"
                    "1. Return only the minimal answer phrase or sentence fragment.\n"
                    "2. No citations, no explanation, no preamble.\n"
                    "3. If the context does not support a clear answer, return exactly: "
                    "\"I don't have enough information in the provided context to answer that.\""
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question: {query}\n\n"
                    f"<Context>\n{context_str}\n</Context>\n\n"
                    f"Draft Answer:\n{draft_answer}\n\n"
                    "Return the shortest supported final answer."
                ),
            },
        ]

    def _project_benchmark_answer(
        self,
        query: str,
        context_str: str,
        draft_answer: str,
    ) -> str:
        concise = self._extract_brief_answer_text(draft_answer)
        if not self.client:
            return concise
        try:
            messages = self._build_benchmark_answer_projection_messages(query, context_str, draft_answer)
            projected = self._generate_buffered(messages)
            cleaned = self._extract_brief_answer_text(projected)
            return cleaned or concise
        except Exception:
            return concise

    def _build_benchmark_reader_messages(
        self,
        query: str,
        context_str: str,
        draft_answer: str,
    ) -> List[Dict[str, str]]:
        return [
            {
                "role": "system",
                "content": (
                    "You are an exact-answer reader for grounded question answering.\n"
                    "Use only the provided context.\n"
                    "Think about the evidence chain privately, but output only the final short answer.\n"
                    "Rules:\n"
                    "1. Return the shortest exact answer span or phrase supported by the context.\n"
                    "2. Prefer proper names, dates, locations, titles, counts, or noun phrases exactly as supported.\n"
                    "3. Do not explain. Do not cite. Do not restate the question.\n"
                    "4. If the context does not support a clear answer, return exactly: "
                    "\"I don't have enough information in the provided context to answer that.\""
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question: {query}\n\n"
                    f"<Context>\n{context_str}\n</Context>\n\n"
                    f"Draft grounded answer:\n{draft_answer}\n\n"
                    "Return only the final short answer."
                ),
            },
        ]

    def _generate_benchmark_short_answer(
        self,
        query: str,
        context_str: str,
        draft_answer: str,
    ) -> str:
        return self._project_benchmark_answer(query, context_str, draft_answer)

    def _update_generation_debug_metrics(
        self,
        debug_metrics: Dict[str, Any],
        generation_hits: List[Dict[str, Any]],
        supporting_hits: Optional[List[Dict[str, Any]]] = None,
        follow_up_queries: Optional[List[str]] = None,
        compacted_count: int = 0,
    ) -> None:
        debug_metrics["final_context_count"] = len(generation_hits)
        debug_metrics["final_context_titles"] = [
            ((hit.get("graph_context") or {}).get("doc_title") or hit.get("title") or "Unknown")
            for hit in generation_hits
        ]
        final_titles = [title.lower() for title in debug_metrics["final_context_titles"] if title]
        duplicate_titles = max(0, len(final_titles) - len(set(final_titles)))
        debug_metrics["generation_duplicate_title_ratio"] = duplicate_titles / max(1, len(final_titles))
        generation_keys = {
            (hit.get("doc_id"), hit.get("chunk_index"))
            for hit in generation_hits
        }
        supporting_keys = {
            (hit.get("doc_id"), hit.get("chunk_index"))
            for hit in (supporting_hits or [])
        }
        debug_metrics["supporting_inherited_generation_count"] = len(generation_keys & supporting_keys)
        planner_confirmed_generation_count = sum(
            1
            for hit in generation_hits
            if self._is_follow_up_confirmed_hit(hit, follow_up_queries)
        )
        debug_metrics["planner_confirmed_generation_count"] = planner_confirmed_generation_count
        debug_metrics["generation_planner_chunk_ratio"] = (
            planner_confirmed_generation_count / max(1, len(generation_hits))
        )
        debug_metrics["generation_compacted_count"] = compacted_count
        debug_metrics["generation_source_type_mix"] = dict(
            Counter(self._infer_hit_source_type(hit) for hit in generation_hits)
        )
        debug_metrics["generation_evidence_role_mix"] = dict(
            Counter(str(hit.get("evidence_role") or hit.get("role") or "background").lower() for hit in generation_hits)
        )
        generation_chain_ids = {
            str(hit.get("primary_chain_id"))
            for hit in generation_hits
            if hit.get("primary_chain_id")
        }
        supporting_chain_ids = {
            str(hit.get("primary_chain_id"))
            for hit in (supporting_hits or [])
            if hit.get("primary_chain_id")
        }
        debug_metrics["generation_chain_mix"] = dict(
            Counter(
                (
                    f"chain_{min(max(int(hit.get('primary_chain_rank') or 1), 1), 3)}"
                    if hit.get("primary_chain_id") else "no_chain"
                )
                for hit in generation_hits
            )
        )
        debug_metrics["generation_chain_count"] = len(generation_chain_ids)
        debug_metrics["supporting_inherited_chain_count"] = len(generation_chain_ids & supporting_chain_ids)
        debug_metrics["generation_uncorroborated_graph_count"] = sum(
            1
            for hit in generation_hits
            if self._is_graph_like_source_type(self._infer_hit_source_type(hit))
            and not bool(hit.get("is_corroborated"))
        )
        generation_selection_debug = getattr(self, "_last_generation_selection_debug", {}) or {}
        debug_metrics["generation_chain_mode_selected"] = generation_selection_debug.get("chain_mode_selected")
        debug_metrics["generation_chain_activation_reason"] = generation_selection_debug.get("chain_activation_reason")
        debug_metrics["generation_bridge_budget_used"] = generation_selection_debug.get("bridge_budget_used", 0)
        debug_metrics["generation_weak_bridge_candidates_dropped"] = generation_selection_debug.get("weak_bridge_candidates_dropped", 0)
        debug_metrics["generation_final_context_bridge_fraction"] = generation_selection_debug.get("final_context_bridge_fraction", 0.0)
        debug_metrics["generation_final_context_direct_support_fraction"] = generation_selection_debug.get("final_context_direct_support_fraction", 0.0)
        debug_metrics["generation_chain_vs_standalone_mix"] = generation_selection_debug.get("chain_vs_standalone_mix", {})

    def _assemble_generation_context_hits(
        self,
        search_query: str,
        hits: List[Dict[str, Any]],
        limit: int,
        max_chunks_per_title: int,
        background_limit: int,
        supporting_hits: Optional[List[Dict[str, Any]]] = None,
        follow_up_queries: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        if not hits or limit <= 0:
            self._last_generation_selection_debug = {}
            return []

        query_terms = set(_normalize_terms(search_query))
        subject_terms = set()
        for subject in self._extract_query_subject_candidates(search_query):
            subject_terms.update(_normalize_terms(subject))
        focus_terms = {
            token for token in query_terms
            if token not in _QUERY_FOCUS_STOPWORDS and token not in subject_terms
        }
        supporting_keys = {
            (support_hit.get("doc_id"), support_hit.get("chunk_index"))
            for support_hit in (supporting_hits or [])
        }
        supporting_chain_ids = {
            str(support_hit.get("primary_chain_id"))
            for support_hit in (supporting_hits or [])
            if support_hit.get("primary_chain_id")
        }
        supporting_title_keys = {
            ((((support_hit.get("graph_context") or {}).get("doc_title")) or support_hit.get("title") or "").strip().lower())
            for support_hit in (supporting_hits or [])
            if (((support_hit.get("graph_context") or {}).get("doc_title")) or support_hit.get("title") or "").strip()
        }
        title_family_sources: Dict[str, set] = {}
        title_family_focus_hits: Counter = Counter()
        title_family_role_hits: Counter = Counter()
        for hit in hits:
            graph_info = hit.get("graph_context") or {}
            title = (graph_info.get("doc_title") or hit.get("title") or "").strip()
            title_key = title.lower() or f"{hit.get('doc_id')}_{hit.get('chunk_index')}"
            source_type = self._infer_hit_source_type(hit)
            title_family_sources.setdefault(title_key, set()).add(source_type)
            hit_query_terms = set(_normalize_terms(title)) | set(_normalize_terms(hit.get("chunk_text", "") or ""))
            if hit_query_terms & focus_terms:
                title_family_focus_hits[title_key] += 1
            role = str(hit.get("evidence_role") or hit.get("role") or "").lower()
            if role in {"answer", "direct", "bridge", "graph", "symbolic"}:
                title_family_role_hits[title_key] += 1
        prepared_rows = []
        observed_chain_modes = Counter(
            str(hit.get("chain_mode_selected") or "").lower()
            for hit in hits
            if str(hit.get("chain_mode_selected") or "").strip()
        )
        generation_chain_mode = "full"
        if observed_chain_modes:
            generation_chain_mode = observed_chain_modes.most_common(1)[0][0]
        generation_chain_reason = next(
            (
                str(hit.get("chain_activation_reason"))
                for hit in hits
                if str(hit.get("chain_activation_reason") or "").strip()
            ),
            "mixed_evidence_profile",
        )
        packing_settings = {
            "bypass": {"bridge_cap": 0, "background_cap": 0, "weak_bridge_threshold": 0.72, "chain_bundle_cap": 0},
            "light": {"bridge_cap": 1, "background_cap": 0, "weak_bridge_threshold": 0.64, "chain_bundle_cap": 1},
            "full": {"bridge_cap": 2, "background_cap": background_limit, "weak_bridge_threshold": 0.55, "chain_bundle_cap": 2},
        }.get(generation_chain_mode, {"bridge_cap": 2, "background_cap": background_limit, "weak_bridge_threshold": 0.55, "chain_bundle_cap": 2})
        for position, hit in enumerate(hits):
            scores = {
                "answer_score": _safe_numeric(hit.get("answer_score"), None),
                "bridge_score": _safe_numeric(hit.get("bridge_score"), None),
                "joint_score": _safe_numeric(hit.get("joint_score"), None),
            }
            if scores["answer_score"] is None or scores["bridge_score"] is None or scores["joint_score"] is None:
                scores = self._score_dual_head_hit(search_query, hit)
                hit["answer_score"] = scores["answer_score"]
                hit["bridge_score"] = scores["bridge_score"]
                hit["joint_score"] = scores["joint_score"]

            graph_info = hit.get("graph_context") or {}
            title = (graph_info.get("doc_title") or hit.get("title") or "").strip()
            title_key = title.lower() or f"{hit.get('doc_id')}_{hit.get('chunk_index')}"
            body_text = hit.get("chunk_text", "") or ""
            if "\n" in body_text:
                body_text = body_text.split("\n", 1)[1]

            title_terms = set(_normalize_terms(title))
            body_terms = set(_normalize_terms(body_text))
            retrieval_queries = [
                rq.strip() for rq in (hit.get("retrieval_queries") or [])
                if isinstance(rq, str) and rq.strip()
            ]
            bridge_queries = [rq for rq in retrieval_queries if rq.lower() != search_query.lower()]
            bridge_terms = set()
            for bridge_query in bridge_queries:
                bridge_terms.update(_normalize_terms(bridge_query))
            focus_title_overlap = len(title_terms & focus_terms)
            focus_body_overlap = len(body_terms & focus_terms)
            focus_overlap = focus_title_overlap + focus_body_overlap

            answer_signal = (
                scores["answer_score"]
                + (0.14 * len(title_terms & query_terms))
                + (0.05 * len(body_terms & query_terms))
                + (0.08 if title_terms & query_terms else 0.0)
                + (0.16 * focus_title_overlap)
                + (0.08 * focus_body_overlap)
            )
            bridge_signal = (
                scores["bridge_score"]
                + (0.14 * len(title_terms & bridge_terms))
                + (0.05 * len(body_terms & bridge_terms))
                + (0.10 if bridge_queries else 0.0)
            )

            lowered_body = body_text.strip().lower()
            lowered_title = title.strip().lower()
            definition_like = False
            if lowered_body:
                definition_like = lowered_body.startswith(f"{lowered_title} is") or lowered_body.startswith(f"{lowered_title} was")
                definition_like = definition_like or lowered_body.startswith("is a ") or lowered_body.startswith("was a ")
                definition_like = definition_like or lowered_body.startswith("is an ") or lowered_body.startswith("was an ")
                definition_like = definition_like or lowered_body.startswith("may refer to")

            filler_penalty = 0.0
            if focus_terms and focus_overlap == 0 and not (title_terms & bridge_terms or body_terms & bridge_terms):
                filler_penalty += 0.28
            if definition_like and focus_overlap == 0 and role != "bridge":
                filler_penalty += 0.22
            if role == "background" and focus_overlap == 0:
                filler_penalty += 0.18

            is_supporting_selected = (hit.get("doc_id"), hit.get("chunk_index")) in supporting_keys
            supporting_title_match = title_key in supporting_title_keys if title else False
            is_planner_confirmed = self._is_follow_up_confirmed_hit(hit, follow_up_queries)
            source_type = self._infer_hit_source_type(hit)
            is_graph_like = self._is_graph_like_source_type(source_type)
            primary_chain_id = str(hit.get("primary_chain_id") or "").strip()
            primary_chain_rank = int(hit.get("primary_chain_rank") or 0) if hit.get("primary_chain_rank") else 0
            best_chain_score = _safe_numeric(hit.get("best_chain_score"), 0.0)
            best_chain_length = int(hit.get("best_chain_length") or 1)
            primary_chain_complete = bool(hit.get("primary_chain_complete"))
            chain_selected = bool(hit.get("chain_selected"))
            primary_chain_member_role = str(hit.get("primary_chain_member_role") or "").lower()
            chain_support_signal = _safe_numeric(hit.get("chain_support_signal"), 0.0)
            chain_bridge_signal = _safe_numeric(hit.get("chain_bridge_signal"), 0.0)
            chain_bridge_candidate = bool(
                primary_chain_id
                and (
                    primary_chain_member_role == "bridge"
                    or (chain_bridge_signal >= 0.28 and best_chain_length > 1)
                )
            )
            chain_support_candidate = bool(
                primary_chain_id
                and (
                    primary_chain_member_role == "support"
                    or chain_support_signal >= 0.24
                )
            )
            if chain_bridge_candidate:
                bridge_signal = max(
                    bridge_signal,
                    min(1.2, (0.56 * chain_bridge_signal) + (0.12 if primary_chain_complete else 0.0)),
                )
            if chain_support_candidate:
                answer_signal = max(
                    answer_signal,
                    min(1.4, (0.52 * chain_support_signal) + (0.10 if primary_chain_complete else 0.0)),
                )
            weak_match = not (
                (title_terms & query_terms)
                or (body_terms & query_terms)
                or (title_terms & bridge_terms)
                or (body_terms & bridge_terms)
            )
            role = "background"
            if primary_chain_member_role == "bridge" and chain_bridge_candidate and answer_signal < 1.18:
                role = "bridge"
            elif chain_bridge_candidate and bridge_signal >= max(0.48, answer_signal - 0.12):
                role = "bridge"
            elif answer_signal >= max(bridge_signal + 0.18, 0.72):
                role = "answer"
            elif bridge_signal >= 0.55:
                role = "bridge"
            elif answer_signal >= 0.78:
                role = "answer"
            elif chain_support_candidate and answer_signal >= 0.62:
                role = "answer"

            family_sources = title_family_sources.get(title_key, set())
            corroboration_count = max(0, len(family_sources) - 1)
            if title_family_focus_hits.get(title_key, 0) > 1:
                corroboration_count += 1
            if title_family_role_hits.get(title_key, 0) > 1:
                corroboration_count += 1
            if is_supporting_selected or supporting_title_match:
                corroboration_count += 1
            if is_planner_confirmed:
                corroboration_count += 1
            if answer_signal >= 1.0:
                corroboration_count += 1
            is_corroborated = (
                corroboration_count > 0
                or (not is_graph_like and focus_overlap > 0)
            )
            chain_bonus = 0.0
            if primary_chain_id:
                chain_bonus += min(0.30, 0.06 * best_chain_length)
                chain_bonus += min(0.18, 0.08 * max(0.0, best_chain_score - scores["joint_score"]))
                if chain_selected:
                    chain_bonus += 0.10
                if primary_chain_complete and role in {"answer", "bridge"}:
                    chain_bonus += 0.16
                if supporting_chain_ids and primary_chain_id in supporting_chain_ids:
                    chain_bonus += 0.28

            utility_score = (
                max(answer_signal, bridge_signal)
                + (0.28 * min(answer_signal, bridge_signal))
                - (0.28 if weak_match else 0.0)
                - filler_penalty
                + chain_bonus
                - (0.015 * position)
            )
            if is_supporting_selected:
                utility_score += 3.8
            elif supporting_title_match:
                utility_score += 0.65
            elif supporting_title_keys:
                utility_score -= 0.26
                if role == "background":
                    utility_score -= 0.24
            if is_planner_confirmed:
                utility_score += 0.48 if role in {"answer", "bridge"} else 0.16
            if source_type == "lexical" and focus_overlap > 0:
                utility_score += 0.18
            if is_graph_like and is_corroborated:
                utility_score += 0.16
            if is_graph_like and not is_corroborated:
                utility_score -= 0.72
            if source_type == "symbolic" and not is_corroborated:
                utility_score -= 0.30
            if is_graph_like and role == "background":
                utility_score -= 0.24

            prepared_rows.append({
                "hit": hit,
                "title": title,
                "title_key": title_key,
                "body_text": body_text,
                "role": role,
                "answer_signal": answer_signal,
                "bridge_signal": bridge_signal,
                "utility_score": utility_score,
                "position": position,
                "key": (hit.get("doc_id"), hit.get("chunk_index")),
                "is_supporting_selected": is_supporting_selected,
                "supporting_title_match": supporting_title_match,
                "is_planner_confirmed": is_planner_confirmed,
                "source_type": source_type,
                "is_graph_like": is_graph_like,
                "corroboration_count": corroboration_count,
                "is_corroborated": is_corroborated,
                "primary_chain_id": primary_chain_id or None,
                "primary_chain_rank": primary_chain_rank,
                "best_chain_score": best_chain_score,
                "best_chain_length": best_chain_length,
                "primary_chain_complete": primary_chain_complete,
                "chain_selected": chain_selected,
                "primary_chain_member_role": primary_chain_member_role or None,
                "chain_bridge_candidate": chain_bridge_candidate,
                "chain_support_candidate": chain_support_candidate,
                "chain_bonus": chain_bonus,
            })

        prepared_rows.sort(
            key=lambda row: (
                row["utility_score"],
                row["answer_signal"] + row["bridge_signal"],
                -row["position"],
            ),
            reverse=True
        )

        selected_rows = []
        seen_keys = set()
        title_counts: Dict[str, int] = {}
        selected_texts_by_title: Dict[str, List[str]] = {}
        role_counts = {"answer": 0, "bridge": 0, "background": 0}
        source_counts = Counter()
        selected_family_answer = set()
        selected_chain_roles: Dict[str, set] = defaultdict(set)
        non_support_title_keys = set()
        support_family_active = bool(supporting_title_keys)
        bridge_budget_used = 0
        weak_bridge_candidates_dropped = 0
        source_type_caps = {
            "symbolic": 1,
            "graph_expansion": 1,
            "graph_enriched": 1,
        }

        def _try_select(row: Dict[str, Any], allow_background: bool) -> bool:
            nonlocal bridge_budget_used
            if row["key"] in seen_keys:
                return False
            if row["role"] == "background" and not allow_background and not row["is_supporting_selected"]:
                return False
            if row["is_graph_like"] and not row["is_corroborated"] and not row["is_supporting_selected"]:
                return False
            if row["title"] and title_counts.get(row["title_key"], 0) >= max(1, max_chunks_per_title):
                return False
            source_cap = source_type_caps.get(row["source_type"])
            if (
                source_cap is not None
                and source_counts.get(row["source_type"], 0) >= source_cap
                and not row["is_supporting_selected"]
                and not row["is_planner_confirmed"]
            ):
                return False
            existing_texts = selected_texts_by_title.get(row["title_key"], [])
            if any(self._is_near_duplicate_chunk(row["body_text"], existing) for existing in existing_texts):
                return False
            in_support_family = row["is_supporting_selected"] or row["supporting_title_match"]
            if support_family_active and not in_support_family:
                if row["role"] == "background" and not row["is_planner_confirmed"]:
                    return False
                if row["utility_score"] < 1.02 and not row["is_planner_confirmed"]:
                    return False
                if row["title"] and row["title_key"] not in non_support_title_keys:
                    if len(non_support_title_keys) >= 1 and not row["is_planner_confirmed"]:
                        return False
            if row["role"] == "background":
                if role_counts["background"] >= max(0, int(packing_settings["background_cap"])):
                    return False
                if role_counts["answer"] >= 1 and role_counts["bridge"] >= 1:
                    return False
            if row["role"] == "bridge" and not row["is_supporting_selected"]:
                if row["answer_signal"] < 0.92:
                    chain_id = row.get("primary_chain_id")
                    family_answer_selected = row["title_key"] in selected_family_answer
                    chain_answer_selected = bool(
                        chain_id and ("answer" in selected_chain_roles.get(chain_id, set()) or "support" in selected_chain_roles.get(chain_id, set()))
                    )
                    if not family_answer_selected and not chain_answer_selected:
                        if not row.get("primary_chain_complete") or not row.get("is_corroborated"):
                            return False
                    if bridge_budget_used >= int(packing_settings["bridge_cap"]):
                        return False
                    if row["bridge_signal"] < float(packing_settings["weak_bridge_threshold"]):
                        return False

            selected_rows.append(row)
            seen_keys.add(row["key"])
            if row["title"]:
                title_counts[row["title_key"]] = title_counts.get(row["title_key"], 0) + 1
                selected_texts_by_title.setdefault(row["title_key"], []).append(row["body_text"])
                if support_family_active and not in_support_family:
                    non_support_title_keys.add(row["title_key"])
            role_counts[row["role"]] = role_counts.get(row["role"], 0) + 1
            source_counts[row["source_type"]] += 1
            if row["role"] == "answer":
                selected_family_answer.add(row["title_key"])
            if row.get("primary_chain_id"):
                selected_chain_roles[row["primary_chain_id"]].add(row["role"])
                if row.get("chain_support_candidate"):
                    selected_chain_roles[row["primary_chain_id"]].add("support")
            if row["role"] == "bridge" and not row["is_supporting_selected"] and row["answer_signal"] < 0.92:
                bridge_budget_used += 1
            return True

        family_rows: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        family_scores: Dict[str, float] = {}
        for row in prepared_rows:
            family_rows[row["title_key"]].append(row)
        for title_key, rows_in_family in family_rows.items():
            best_answer = max(
                (
                    row["utility_score"]
                    for row in rows_in_family
                    if row["role"] == "answer" or row["is_supporting_selected"] or row.get("chain_support_candidate")
                ),
                default=max(row["utility_score"] for row in rows_in_family) - 0.20,
            )
            best_any = max(row["utility_score"] for row in rows_in_family)
            family_scores[title_key] = (
                (0.72 * best_answer)
                + (0.28 * best_any)
                + (0.20 if supporting_title_keys and title_key in supporting_title_keys else 0.0)
                + (0.08 * min(2, title_family_focus_hits.get(title_key, 0)))
                + (0.06 * min(2, title_family_role_hits.get(title_key, 0)))
                + (0.06 * len(title_family_sources.get(title_key, set())))
            )

        answer_rows = [row for row in prepared_rows if row["role"] == "answer"]
        bridge_rows = [row for row in prepared_rows if row["role"] == "bridge"]
        background_rows = [row for row in prepared_rows if row["role"] == "background"]
        locked_support_rows = [row for row in prepared_rows if row["is_supporting_selected"]]
        planner_rows = [
            row for row in prepared_rows
            if row["is_planner_confirmed"] and row["role"] in {"answer", "bridge"}
        ]
        planner_rows.sort(
            key=lambda row: (
                1 if row["role"] == "answer" or row.get("chain_support_candidate") else 0,
                row["utility_score"],
            ),
            reverse=True,
        )
        chain_seed_rows = []
        seen_chain_ids = set()
        for row in prepared_rows:
            chain_id = row.get("primary_chain_id")
            if not chain_id or chain_id in seen_chain_ids:
                continue
            if (
                row["role"] != "answer"
                and not row.get("chain_support_candidate")
                and not row["is_supporting_selected"]
                and not row["is_planner_confirmed"]
            ):
                continue
            chain_seed_rows.append(row)
            seen_chain_ids.add(chain_id)

        family_quota = max(2, min(limit, max(2, (limit // 2) + 1)))
        ranked_families = sorted(family_scores.items(), key=lambda item: item[1], reverse=True)
        for title_key, _ in ranked_families[:family_quota]:
            rows_in_family = sorted(
                family_rows[title_key],
                key=lambda row: (
                    row["role"] == "answer" or row["is_supporting_selected"] or row.get("chain_support_candidate"),
                    row["is_corroborated"],
                    row["utility_score"],
                ),
                reverse=True,
            )
            family_anchor_taken = False
            for row in rows_in_family:
                if row["role"] not in {"answer", "bridge"} and not row["is_supporting_selected"] and not row.get("chain_support_candidate"):
                    continue
                if row["role"] == "bridge" and not (row["is_corroborated"] or row["is_supporting_selected"]):
                    continue
                if _try_select(row, allow_background=False):
                    family_anchor_taken = True
                    break
            if not family_anchor_taken:
                continue
            for row in rows_in_family:
                if len(selected_rows) >= limit:
                    break
                if row["role"] != "bridge":
                    continue
                if _try_select(row, allow_background=False):
                    break

        for row in locked_support_rows:
            if len(selected_rows) >= limit:
                break
            _try_select(row, allow_background=False)

        for row in answer_rows:
            if len(selected_rows) >= limit:
                break
            _try_select(row, allow_background=False)

        for row in planner_rows:
            if len(selected_rows) >= limit:
                break
            _try_select(row, allow_background=False)

        for row in chain_seed_rows:
            if len(selected_rows) >= limit:
                break
            _try_select(row, allow_background=False)

        ordered_chain_ids = []
        for row in prepared_rows:
            chain_id = row.get("primary_chain_id")
            if chain_id and chain_id not in ordered_chain_ids:
                ordered_chain_ids.append(chain_id)

        chain_bundle_cap = int(packing_settings.get("chain_bundle_cap", 1))
        for chain_id in ordered_chain_ids:
            if len(selected_rows) >= limit or chain_bundle_cap <= 0:
                break
            chain_rows = [row for row in prepared_rows if row.get("primary_chain_id") == chain_id]
            if not chain_rows:
                continue
            bundle_limit = 1
            if chain_bundle_cap > 1 and any(row.get("primary_chain_complete") for row in chain_rows):
                bundle_limit = chain_bundle_cap
            selected_in_bundle = 0

            def _take_bundle_role(role_name: str) -> bool:
                nonlocal selected_in_bundle
                for row in chain_rows:
                    if role_name == "answer":
                        matches_role = row["role"] == "answer" or row.get("chain_support_candidate")
                    else:
                        matches_role = row["role"] == "bridge" or row.get("chain_bridge_candidate")
                    if not matches_role:
                        continue
                    if _try_select(row, allow_background=False):
                        selected_in_bundle += 1
                        return True
                return False

            _take_bundle_role("answer")
            if selected_in_bundle < bundle_limit:
                _take_bundle_role("bridge")
            if selected_in_bundle < bundle_limit:
                for row in chain_rows:
                    if _try_select(row, allow_background=False):
                        selected_in_bundle += 1
                    if selected_in_bundle >= bundle_limit:
                        break
        if answer_rows:
            _try_select(answer_rows[0], allow_background=False)
        if len(selected_rows) < limit and bridge_rows:
            _try_select(bridge_rows[0], allow_background=False)

        for row in prepared_rows:
            if len(selected_rows) >= limit:
                break
            if (
                row["role"] == "bridge"
                and not row["is_supporting_selected"]
                and row["answer_signal"] < 0.92
                and row["bridge_signal"] < float(packing_settings["weak_bridge_threshold"])
            ):
                weak_bridge_candidates_dropped += 1
            _try_select(row, allow_background=False)

        if len(selected_rows) < limit:
            needs_more_context = role_counts["answer"] == 0 or role_counts["bridge"] == 0 or len(selected_rows) < 2
            if needs_more_context:
                for row in background_rows:
                    if len(selected_rows) >= limit:
                        break
                    if support_family_active and not (row["is_supporting_selected"] or row["supporting_title_match"] or row["is_planner_confirmed"]):
                        continue
                    if row["utility_score"] < 0.92:
                        continue
                    _try_select(row, allow_background=True)

        if len(selected_rows) < limit:
            for row in prepared_rows:
                if len(selected_rows) >= limit:
                    break
                if row["role"] == "background":
                    continue
                if row["utility_score"] < 0.98:
                    continue
                _try_select(row, allow_background=False)

        final_hits = []
        for row in selected_rows[:limit]:
            annotated_hit = dict(row["hit"])
            annotated_hit["evidence_role"] = row["role"]
            annotated_hit["role"] = row["role"]
            annotated_hit["source_type"] = row["source_type"]
            annotated_hit["corroboration_count"] = row["corroboration_count"]
            annotated_hit["is_corroborated"] = row["is_corroborated"]
            final_hits.append(annotated_hit)

        compacted_hits, compacted_count = self._compact_generation_context_hits(final_hits)
        chain_bundle_rows_kept = sum(
            1
            for row in selected_rows
            if row.get("primary_chain_id") and row.get("primary_chain_complete")
        )
        self._last_generation_compacted_count = compacted_count
        self._last_generation_selection_debug = {
            "chain_mode_selected": generation_chain_mode,
            "chain_activation_reason": generation_chain_reason,
            "bridge_budget_used": bridge_budget_used,
            "chain_bundle_rows_kept": chain_bundle_rows_kept,
            "weak_bridge_candidates_dropped": weak_bridge_candidates_dropped,
            "final_context_bridge_fraction": round(
                sum(1 for row in selected_rows if row["role"] == "bridge") / max(1, len(selected_rows)),
                4,
            ),
            "final_context_direct_support_fraction": round(
                sum(1 for row in selected_rows if row["role"] == "answer") / max(1, len(selected_rows)),
                4,
            ),
            "chain_vs_standalone_mix": {
                "chain_backed": sum(1 for row in selected_rows if row.get("primary_chain_id")),
                "standalone": sum(1 for row in selected_rows if not row.get("primary_chain_id")),
            },
        }
        return compacted_hits[:limit]

    def _select_benchmark_generation_hits(
        self,
        search_query: str,
        hits: List[Dict[str, Any]],
        limit: int,
        max_chunks_per_title: int
    ) -> List[Dict[str, Any]]:
        if not hits or limit <= 0:
            return []

        query_lower = search_query.lower()
        title_buckets: Dict[str, List[Dict[str, Any]]] = {}
        ordered_bucket_keys: List[str] = []
        query_terms = set(_normalize_terms(search_query))
        for hit in hits:
            graph_info = hit.get("graph_context") or {}
            title = (graph_info.get("doc_title") or hit.get("title") or "").strip()
            bucket_key = title.lower() or f"{hit.get('doc_id')}_{hit.get('chunk_index')}"
            if bucket_key not in title_buckets:
                ordered_bucket_keys.append(bucket_key)
            title_buckets.setdefault(bucket_key, []).append(hit)

        bucket_representatives: List[Dict[str, Any]] = []
        for bucket_key in ordered_bucket_keys:
            bucket_hits = title_buckets[bucket_key]
            selected_bucket_hits = bucket_hits[:max(1, max_chunks_per_title)]

            specific_queries: List[str] = []
            seen_queries = set()
            for hit in bucket_hits:
                for retrieval_query in hit.get("retrieval_queries", []) or []:
                    if not isinstance(retrieval_query, str):
                        continue
                    normalized = retrieval_query.strip()
                    lowered = normalized.lower()
                    if not normalized or lowered == query_lower or lowered in seen_queries:
                        continue
                    specific_queries.append(normalized)
                    seen_queries.add(lowered)

            if specific_queries and getattr(self.retriever, "cross_encoder", None):
                ranking_query = max(specific_queries, key=len)
                rerank_pool = [hit.copy() for hit in bucket_hits]
                reranked_bucket = self.retriever.cross_encoder.rerank(
                    ranking_query,
                    rerank_pool,
                    top_k=min(len(rerank_pool), max(2, max_chunks_per_title))
                )
                reranked_keys = {
                    (hit.get("doc_id"), hit.get("chunk_index")): hit
                    for hit in reranked_bucket
                }
                selected_bucket_hits = []
                for original_hit in bucket_hits:
                    key = (original_hit.get("doc_id"), original_hit.get("chunk_index"))
                    if key in reranked_keys:
                        selected_bucket_hits.append(original_hit)
                selected_bucket_hits.sort(
                    key=lambda hit: _safe_numeric(
                        reranked_keys[(hit.get("doc_id"), hit.get("chunk_index"))].get("cross_encoder_score"),
                        _safe_numeric(hit.get("final_rank_score"), _safe_numeric(hit.get("score", 0.0)))
                    ),
                    reverse=True
                )
                selected_bucket_hits = selected_bucket_hits[:max(1, max_chunks_per_title)]

            for hit in selected_bucket_hits:
                title_terms = set(_normalize_terms(hit.get("title") or ""))
                text_terms = set(_normalize_terms(hit.get("chunk_text") or ""))
                retrieval_terms = set()
                for retrieval_query in hit.get("retrieval_queries", []) or []:
                    retrieval_terms.update(_normalize_terms(retrieval_query))
                body_text = hit.get("chunk_text", "") or ""
                if "\n" in body_text:
                    body_text = body_text.split("\n", 1)[1]
                lead_chunk_bonus = 0.0
                title = (hit.get("title") or "").strip().lower()
                if title and body_text.strip().lower().startswith(title):
                    lead_chunk_bonus = 0.45
                title_retrieval_overlap = len(title_terms & retrieval_terms)
                text_retrieval_overlap = len(text_terms & retrieval_terms)
                weak_match_penalty = 0.0
                if not (title_terms & query_terms) and not title_retrieval_overlap and not text_retrieval_overlap:
                    weak_match_penalty = 0.35
                hit["generation_selection_score"] = (
                    _safe_numeric(hit.get("final_rank_score"), _safe_numeric(hit.get("score", 0.0)))
                    + 0.12 * len(title_terms & query_terms)
                    + 0.05 * len(text_terms & query_terms)
                    + 0.18 * title_retrieval_overlap
                    + 0.08 * text_retrieval_overlap
                    + (0.2 if retrieval_terms else 0.0)
                    + lead_chunk_bonus
                    - weak_match_penalty
                )
                bucket_representatives.append(hit)

        bucket_representatives.sort(
            key=lambda hit: _safe_numeric(
                hit.get("generation_selection_score"),
                _safe_numeric(hit.get("final_rank_score"), _safe_numeric(hit.get("score", 0.0)))
            ),
            reverse=True
        )
        return bucket_representatives[:limit]

    def _select_benchmark_top_hits(
        self,
        search_query: str,
        hits: List[Dict[str, Any]],
        limit: int,
        max_chunks_per_title: int
    ) -> List[Dict[str, Any]]:
        if not hits or limit <= 0:
            return []

        query_terms = set(_normalize_terms(search_query))
        scored_hits = []
        for position, hit in enumerate(hits):
            graph_info = hit.get("graph_context") or {}
            title = (graph_info.get("doc_title") or hit.get("title") or "").strip()
            title_terms = set(_normalize_terms(title))
            text_terms = set(_normalize_terms(hit.get("chunk_text") or ""))
            retrieval_terms = set()
            for retrieval_query in hit.get("retrieval_queries", []) or []:
                retrieval_terms.update(_normalize_terms(retrieval_query))

            body_text = hit.get("chunk_text", "") or ""
            if "\n" in body_text:
                body_text = body_text.split("\n", 1)[1]
            lead_chunk_bonus = 0.3 if title and body_text.strip().lower().startswith(title.lower()) else 0.0
            specificity_bonus = 0.18 * len(title_terms & retrieval_terms) + 0.07 * len(text_terms & retrieval_terms)
            query_bonus = 0.12 * len(title_terms & query_terms) + 0.04 * len(text_terms & query_terms)
            diversity_score = _safe_numeric(
                hit.get("final_rank_score"),
                _safe_numeric(hit.get("cross_encoder_score"), _safe_numeric(hit.get("score", 0.0)))
            ) + lead_chunk_bonus + specificity_bonus + query_bonus

            scored_hits.append((diversity_score, -position, hit))

        scored_hits.sort(reverse=True)

        selected = []
        seen_keys = set()
        title_counts: Dict[str, int] = {}

        for _, _, hit in scored_hits:
            graph_info = hit.get("graph_context") or {}
            title = (graph_info.get("doc_title") or hit.get("title") or "").strip()
            title_key = title.lower() or f"{hit.get('doc_id')}_{hit.get('chunk_index')}"
            key = (hit.get("doc_id"), hit.get("chunk_index"))
            if key in seen_keys:
                continue
            if title and title_counts.get(title_key, 0) >= max(1, max_chunks_per_title):
                continue
            selected.append(hit)
            seen_keys.add(key)
            title_counts[title_key] = title_counts.get(title_key, 0) + 1
            if len(selected) >= limit:
                break

        if len(selected) < limit:
            for _, _, hit in scored_hits:
                key = (hit.get("doc_id"), hit.get("chunk_index"))
                if key in seen_keys:
                    continue
                selected.append(hit)
                seen_keys.add(key)
                if len(selected) >= limit:
                    break

        return selected[:limit]

    def _plan_benchmark_follow_up_queries(
        self,
        query: str,
        hits: List[Dict[str, Any]],
        query_graph: Optional[List[Dict[str, Any]]] = None,
        precomputed_entities: Optional[List[str]] = None,
        precomputed_aliases: Optional[List[str]] = None,
    ) -> List[str]:
        if not hits:
            return []

        entity_candidates = precomputed_entities or self._extract_planner_bridge_candidates(query, hits)
        alias_candidates = precomputed_aliases or self._extract_alias_bridge_candidates(query, hits)
        merged_entities: List[str] = []
        seen_entities = set()
        for candidate in [*entity_candidates, *alias_candidates]:
            normalized = (candidate or "").strip()
            lowered = normalized.lower()
            if normalized and lowered not in seen_entities:
                merged_entities.append(normalized)
                seen_entities.add(lowered)
        relation_hints = self._derive_planner_focus_hints(query, query_graph)
        fallback_queries = self._generic_bridge_follow_up_queries(query, hits, query_graph)
        conditioned_queries = self._build_evidence_conditioned_follow_up_queries(
            query,
            hits,
            relation_hints,
            merged_entities,
        )

        if not self.openai_client:
            seeded_queries = self._merge_priority_queries(conditioned_queries, fallback_queries, limit=3)
            refined_queries = self._refine_bridge_queries_for_targeting(
                query,
                seeded_queries,
                merged_entities,
                relation_hints,
                alias_candidates=alias_candidates,
            ) or seeded_queries[:3]
            return refined_queries[:3]

        evidence_lines = []
        for hit in hits[:8]:
            graph_info = hit.get("graph_context") or {}
            title = graph_info.get("doc_title") or hit.get("title") or "Unknown"
            snippet = (hit.get("chunk_text") or "").replace("\n", " ").strip()[:240]
            evidence_lines.append(f"- Title: {title} | Snippet: {snippet}")

        prompt = (
            "You are a retrieval planner for multi-hop question answering.\n"
            "Given a user question, bridge-entity candidates extracted from first-hop evidence, and the first-hop evidence itself, propose up to 4 follow-up search queries.\n"
            "Rules:\n"
            "1. Use only entities explicitly present in the evidence snippets or bridge-entity candidate list.\n"
            "2. Produce standalone search queries, not answers.\n"
            "3. Prefer bridge-entity queries that unlock the missing second hop.\n"
            "4. Preserve the user's target focus in the follow-up query, but rewrite it around the bridge entity.\n"
            "5. If the evidence reveals an alias, role holder, page title, organization, place, or work that can unlock the answer, prefer that entity over repeating the original question.\n"
            "6. Favor precise entity-targeted follow-up queries over broad paraphrases.\n"
            "7. Return an empty list only if no bridge entity is present in the evidence.\n"
            "Return valid JSON with exactly this shape: {\"queries\": [\"...\"]}"
        )
        user_prompt = (
            f"Question:\n{query}\n\n"
            f"Bridge candidates:\n{merged_entities}\n\n"
            f"Relation hints:\n{relation_hints}\n\n"
            f"First-hop evidence:\n" + "\n".join(evidence_lines)
        )

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
                max_tokens=160,
            )
            payload = response.choices[0].message.content or "{}"
            parsed = json.loads(payload)
            queries = []
            seen = set()
            for candidate in parsed.get("queries", []):
                if not isinstance(candidate, str):
                    continue
                normalized = candidate.strip()
                if not normalized:
                    continue
                lowered = normalized.lower()
                if lowered in seen or lowered == query.lower():
                    continue
                queries.append(normalized)
                seen.add(lowered)
            if queries:
                for fallback_query in conditioned_queries + fallback_queries:
                    lowered = fallback_query.lower()
                    if lowered in seen or lowered == query.lower():
                        continue
                    queries.append(fallback_query)
                    seen.add(lowered)
            else:
                queries = (conditioned_queries + fallback_queries)[:]

            refined = self._refine_bridge_queries_for_targeting(
                query,
                queries,
                merged_entities,
                relation_hints,
                alias_candidates=alias_candidates,
            )
            selected_queries = refined or (conditioned_queries + fallback_queries)[:3]
            return selected_queries[:3]
        except Exception as e:
            logger.warning("Benchmark follow-up planning failed: %s", str(e))
            seeded_queries = self._merge_priority_queries(conditioned_queries, fallback_queries, limit=3)
            refined = self._refine_bridge_queries_for_targeting(
                query,
                seeded_queries,
                merged_entities,
                relation_hints,
                alias_candidates=alias_candidates,
            )
            selected_queries = refined or seeded_queries[:3]
            return selected_queries[:3]

    def _plan_benchmark_sub_queries(self, query: str) -> List[str]:
        if not self.openai_client:
            return []

        prompt = (
            "You are a retrieval planner for multi-hop question answering.\n"
            "Rewrite the user question into up to 3 standalone search queries that improve first-hop recall.\n"
            "Rules:\n"
            "1. Do not answer the question.\n"
            "2. Use only entities or literal phrases present in the original question.\n"
            "3. Prefer bridge-discovery queries such as identity, title, role, narrator, location, or alias lookups.\n"
            "4. If the original question is already the best search query, you may return an empty list.\n"
            "Return valid JSON with exactly this shape: {\"queries\": [\"...\"]}"
        )

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": query},
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
                max_tokens=160,
            )
            payload = response.choices[0].message.content or "{}"
            parsed = json.loads(payload)
            queries = []
            seen = set()
            for candidate in parsed.get("queries", []):
                if not isinstance(candidate, str):
                    continue
                normalized = candidate.strip()
                if not normalized:
                    continue
                lowered = normalized.lower()
                if lowered in seen or lowered == query.lower():
                    continue
                queries.append(normalized)
                seen.add(lowered)
            return queries[:3]
        except Exception as e:
            logger.warning("Benchmark sub-query planning failed: %s", str(e))
            return []

    def _check_sufficiency(self, query: str, context: str) -> Dict[str, Any]:
        """
        Evaluates if the provided context is sufficient to answer the query.
        If not, returns a refined search query.
        """
        prompt = f"""
        Evaluation Task: Is the provided <CONTEXT> sufficient to fully answer the <USER_QUERY>?
        
        <USER_QUERY>: {query}
        
        <CONTEXT>:
        {context}
        
        Instructions:
        1. If it IS sufficient, respond with "SUFFICIENT".
        2. If it is NOT sufficient or missing key facts, respond with a NEW, highly specific search query to find the missing details.
        3. Respond with ONLY "SUFFICIENT" or the refined query. No preamble.
        """
        try:
            if not self.openai_client:
                return {"sufficient": bool(context and context != "No relevant context found."), "refinement": None}
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=64
            )
            result = response.choices[0].message.content.strip()
            if result.upper() == "SUFFICIENT":
                return {"sufficient": True, "refinement": None}
            result = re.sub(
                r"^(?:NOT\s+SUFFICIENT|INSUFFICIENT)\s*(?::|\n)?\s*(?:NEW\s+QUERY\s*:)?\s*",
                "",
                result,
                flags=re.IGNORECASE
            ).strip()
            return {"sufficient": False, "refinement": result}
        except Exception as e:
            logger.warning(f"Sufficiency check failed: {e}")
            return {"sufficient": True, "refinement": None}

    def _classify_intent(self, query: str, chat_history: List[Dict[str, str]]) -> str:
        """
        Lightweight classification:
        Returns 'SEARCH' if the user needs factual/enterprise retrieval.
        Returns 'CLARIFICATION' if the user is just saying 'thanks', 'hi', or asking about the agent.
        """
        prompt = (
            "You are an intent classifier for an enterprise search system.\n"
            "Given the user's latest query and the conversation history, classify the intent into one of two categories:\n"
            "1. 'SEARCH': The user is asking a factual question, requesting data, or referring to policy/documents.\n"
            "2. 'CLARIFICATION': The user is merely saying hello, thank you, or asking 'who are you?'.\n\n"
            "Respond with ONLY the word SEARCH or CLARIFICATION."
        )
        
        messages = [{"role": "system", "content": prompt}]
        for msg in chat_history[-2:]:
            messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": query})

        try:
            if not self.client:
                return "SEARCH"
            if self.model_provider == "anthropic":
                response = self.client.messages.create(
                    model="claude-3-haiku-20240307",
                    system=prompt,
                    messages=[{"role": m["role"], "content": m["content"]} for m in messages[1:]],
                    max_tokens=10,
                    temperature=0.0
                )
                intent = response.content[0].text.strip().upper()
            else:
                response = self.client.chat.completions.create(
                     model="gpt-3.5-turbo",
                     messages=messages, # type: ignore
                     temperature=0.0,
                     max_tokens=10
                 )
                intent = response.choices[0].message.content.strip().upper() # type: ignore
                
            if "CLARIFICATION" in intent:
                return "CLARIFICATION"
            return "SEARCH"
        except Exception as e:
             logger.warning("Intent classification failed, defaulting to SEARCH: %s", str(e))
             return "SEARCH"

    def _rewrite_query(self, query: str, chat_history: List[Dict[str, str]]) -> str:
        """
        Rewrites conversational queries into standalone queries using history.
        """
        if not chat_history:
            return query
            
        prompt = (
            "You are an expert query reformulator. Your job is to make a user's target query "
            "completely standalone by resolving any pronouns or references using the provided conversation history.\n"
            "If the target query is already standalone, return it exactly as is.\n"
            "DO NOT answer the query. ONLY output the rewritten standalone query string."
        )
        
        messages = [{"role": "system", "content": prompt}]
        history_str = "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in chat_history])
        user_prompt = f"Conversation History:\n{history_str}\n\nTarget Query to Rewrite: {query}\n\nRewritten Standalone Query:"
        messages.append({"role": "user", "content": user_prompt})
        
        try:
            if not self.client:
                return query
            logger.info("Executing Semantic Query Rewriter...")
            if self.model_provider == "anthropic":
                response = self.client.messages.create(
                     model="claude-3-haiku-20240307",
                     system=prompt,
                     messages=[{"role": "user", "content": user_prompt}],
                     max_tokens=256,
                     temperature=0.0
                )
                rewritten_query = response.content[0].text.strip()
            else:
                response = self.client.chat.completions.create(
                     model="gpt-3.5-turbo",
                     messages=messages, # type: ignore
                     temperature=0.0,
                     max_tokens=256
                )
                rewritten_query = response.choices[0].message.content.strip() # type: ignore
            
            logger.info("Original: '%s' | Rewritten: '%s'", query, rewritten_query)
            return rewritten_query
        except Exception as e:
            logger.warning("Query rewriting failed, using original: %s", str(e))
            return query

    def generate_response(self, query: str, session_id: Optional[str] = None, top_k: int = 5):
        """
        Main reasoning loop: Retrieve context, inject session history, build prompt, stream text natively.
        Yields dictionaries with 'type' indicating 'thought', 'token', 'error', or 'answer_metadata'.
        """
        logger.info("Agent processing query: '%s'", query)
        benchmark_mode = os.getenv("ASTERSCOPE_BENCHMARK_MODE", os.getenv("NOVASEARCH_BENCHMARK_MODE", "false")).lower() in {"1", "true", "yes"}
        benchmark_generate_answer = os.getenv(
            "ASTERSCOPE_BENCHMARK_GENERATE_ANSWER",
            os.getenv("NOVASEARCH_BENCHMARK_GENERATE_ANSWER", "false")
        ).lower() in {"1", "true", "yes"}
        benchmark_context_limit = int(os.getenv("ASTERSCOPE_BENCHMARK_CONTEXT_LIMIT", os.getenv("NOVASEARCH_BENCHMARK_CONTEXT_LIMIT", "4")))
        benchmark_source_limit = int(os.getenv("ASTERSCOPE_BENCHMARK_SOURCE_LIMIT", os.getenv("NOVASEARCH_BENCHMARK_SOURCE_LIMIT", "2")))
        benchmark_max_chunks_per_title = int(
            os.getenv("ASTERSCOPE_BENCHMARK_MAX_CHUNKS_PER_TITLE", os.getenv("NOVASEARCH_BENCHMARK_MAX_CHUNKS_PER_TITLE", "2"))
        )
        benchmark_generation_max_chunks_per_title = int(
            os.getenv(
                "ASTERSCOPE_BENCHMARK_GENERATION_MAX_CHUNKS_PER_TITLE",
                os.getenv("NOVASEARCH_BENCHMARK_GENERATION_MAX_CHUNKS_PER_TITLE", "1"),
            )
        )
        benchmark_generation_background_limit = int(
            os.getenv(
                "ASTERSCOPE_BENCHMARK_GENERATION_BACKGROUND_LIMIT",
                os.getenv("NOVASEARCH_BENCHMARK_GENERATION_BACKGROUND_LIMIT", "1"),
            )
        )
        enable_support_context_inheritance = self._resolve_feature_flag(
            "ENABLE_SUPPORT_CONTEXT_INHERITANCE", False
        )
        enable_early_second_hop = self._resolve_feature_flag(
            "ENABLE_EARLY_SECOND_HOP", True
        )
        enable_bridge_planner = self._resolve_feature_flag(
            "ENABLE_BRIDGE_PLANNER", bool(self.openai_client)
        )
        enable_dual_head_scoring = self._resolve_feature_flag(
            "ENABLE_DUAL_HEAD_SCORING", False
        )
        enable_retrieval_debug = self._resolve_feature_flag(
            "ENABLE_RETRIEVAL_DEBUG", True
        )
        emit_debug_metadata = benchmark_mode or enable_retrieval_debug
        debug_metrics: Dict[str, Any] = {
            "debug_schema_version": "chain-aware-v1",
            "chain_aware_runtime_enabled": os.getenv("ENABLE_CHAIN_AWARE_RETRIEVAL", "true").lower() in {"1", "true", "yes"},
            "enable_support_context_inheritance": enable_support_context_inheritance,
            "enable_early_second_hop": enable_early_second_hop,
            "enable_bridge_planner": enable_bridge_planner,
            "enable_dual_head_scoring": enable_dual_head_scoring,
            "enable_retrieval_debug": enable_retrieval_debug,
            "first_hop_candidates": 0,
            "merged_candidate_count": 0,
            "second_hop_triggered": False,
            "second_hop_added_count": 0,
            "bridge_query_count": 0,
            "bridge_entity_candidate_count": 0,
            "alias_bridge_candidate_count": 0,
            "bridge_targeted_query_count": 0,
            "bridge_targeting_hits": 0,
            "answer_bearing_bridge_hits": 0,
            "bridge_entity_family_chunk_miss": 0,
            "final_context_count": 0,
            "planner_evidence_titles": [],
            "bridge_queries": [],
            "final_context_titles": [],
            "generation_duplicate_title_ratio": 0.0,
            "generation_background_limit": benchmark_generation_background_limit,
            "generation_title_limit": benchmark_generation_max_chunks_per_title,
            "supporting_seed_titles": [],
            "final_supporting_titles": [],
            "blocked_seed_count": 0,
            "seed_mismatch_count": 0,
            "supporting_anchor_filtered_count": 0,
            "final_supporting_subject_anchored_count": 0,
            "final_supporting_follow_up_confirmed_count": 0,
            "final_supporting_unanchored_count": 0,
            "planner_confirmed_generation_count": 0,
            "supporting_inherited_generation_count": 0,
            "generation_planner_chunk_ratio": 0.0,
            "generation_compacted_count": 0,
            "generation_source_type_mix": {},
            "generation_evidence_role_mix": {},
            "generation_chain_mix": {},
            "generation_chain_count": 0,
            "supporting_inherited_chain_count": 0,
            "generation_uncorroborated_graph_count": 0,
        }
        
        # 1. Ensure Session ID & Fetch History
        sid = session_id or str(uuid.uuid4())
        chat_history = self.memory.get_history(sid, max_turns=5)
        metadata_hits: List[Dict[str, Any]] = []
        retrieval_hits_for_metadata: List[Dict[str, Any]] = []
        generation_hits_for_metadata: List[Dict[str, Any]] = []
        support_pool_hits: List[Dict[str, Any]] = []
        
        # 2. Phase 1: Intent Recognition & Routing
        intent = "SEARCH" if benchmark_mode else self._classify_intent(query, chat_history)
        logger.info("Intent Classified as: %s", intent)
        
        if intent == "CLARIFICATION":
            yield {"type": "thought", "content": "Recognized intent as clarification. Bypassing search."}
            graph_expanded_hits = []
            context_str = "No enterprise context needed for clarification."
            search_query = query
        else:
            yield {"type": "thought", "content": "Recognized intent as enterprise search."}
            search_query = query if benchmark_mode else self._rewrite_query(query, chat_history)
            if search_query != query:
                yield {"type": "thought", "content": f"Query rewritten to: '{search_query}'"}
                
            # Extract Structured Query Graph (Semantic Triplets)
            if benchmark_mode and not benchmark_generate_answer:
                query_graph = None
                sub_queries = [search_query]
            else:
                yield {"type": "thought", "content": "Parsing structured semantic query graph..."}
                query_graph = self.query_parser.parse(search_query)
                if query_graph:
                    yield {"type": "thought", "content": f"Extracted semantic triplets: {query_graph}"}
                
                # Use LangChain to Decompose Complex Queries
                planner_result = self.planner.decompose(search_query)
                if isinstance(planner_result, dict) and planner_result.get("type") == "clarification":
                    clarification_query = (planner_result.get("content") or "").strip()
                    if benchmark_mode and benchmark_generate_answer and clarification_query:
                        yield {"type": "thought", "content": f"Planner surfaced a bridge query for retrieval: '{clarification_query}'"}
                        sub_queries = [search_query, clarification_query]
                    else:
                        yield {"type": "thought", "content": "Planner requested clarification, but continuing with the original query for retrieval coverage."}
                        sub_queries = [search_query]
                else:
                    sub_queries = planner_result

            if len(sub_queries) > 1:
                yield {"type": "thought", "content": f"Task decomposed into {len(sub_queries)} sub-queries: {sub_queries}"}

            retrieval_top_k = max(top_k * 4, 16)
            first_hop_pool = self.retriever.collect_candidate_pool(
                search_query,
                top_k=retrieval_top_k,
                additional_queries=sub_queries[1:],
                include_follow_ups=False
            )
            initial_hits = self.retriever.finalize_candidates(
                search_query,
                first_hop_pool,
                top_k=retrieval_top_k,
                query_graph=query_graph
            )
            debug_metrics["first_hop_candidates"] = len(first_hop_pool)
            debug_metrics["planner_evidence_titles"] = [
                ((hit.get("graph_context") or {}).get("doc_title") or hit.get("title") or "Unknown")
                for hit in initial_hits[:8]
            ]
            initial_chain_mode = str(
                ((self.retriever.last_search_debug or {}).get("chain_mode_selected") or "light")
            ).lower()
            debug_metrics["initial_chain_mode_selected"] = initial_chain_mode
            debug_metrics["initial_chain_activation_reason"] = (
                (self.retriever.last_search_debug or {}).get("chain_activation_reason")
            )

            use_early_second_hop = False
            second_hop_activation_reason = "disabled_by_flag"
            if enable_early_second_hop:
                use_early_second_hop, second_hop_activation_reason = self._should_use_early_second_hop(
                    search_query,
                    sub_queries,
                    query_graph,
                    initial_hits,
                    initial_chain_mode,
                )
            debug_metrics["second_hop_activation_reason"] = second_hop_activation_reason

            if use_early_second_hop:
                bridge_evidence = initial_hits[: min(len(initial_hits), max(8, top_k * 2))]
                bridge_entity_candidates = self._extract_planner_bridge_candidates(search_query, bridge_evidence)
                alias_bridge_candidates = self._extract_alias_bridge_candidates(search_query, bridge_evidence)
                merged_bridge_entities = []
                seen_bridge_entities = set()
                for candidate in [*bridge_entity_candidates, *alias_bridge_candidates]:
                    normalized = (candidate or "").strip()
                    lowered = normalized.lower()
                    if normalized and lowered not in seen_bridge_entities:
                        merged_bridge_entities.append(normalized)
                        seen_bridge_entities.add(lowered)
                debug_metrics["bridge_entity_candidate_count"] = len(bridge_entity_candidates)
                debug_metrics["alias_bridge_candidate_count"] = len(alias_bridge_candidates)
                bridge_queries = (
                    self._plan_benchmark_follow_up_queries(
                        search_query,
                        bridge_evidence,
                        query_graph=query_graph,
                        precomputed_entities=bridge_entity_candidates,
                        precomputed_aliases=alias_bridge_candidates,
                    )
                    if enable_bridge_planner
                    else self._deterministic_bridge_follow_up_queries(
                        search_query,
                        bridge_evidence,
                        query_graph=query_graph
                    )
                )
                debug_metrics["bridge_query_count"] = len(bridge_queries)
                debug_metrics["bridge_targeted_query_count"] = self._count_entity_targeted_queries(
                    bridge_queries,
                    merged_bridge_entities,
                )
                debug_metrics["second_hop_triggered"] = bool(bridge_queries)
                debug_metrics["bridge_queries"] = bridge_queries[:]
                if bridge_queries:
                    yield {"type": "thought", "content": f"Bridge queries planned from first-hop evidence: {bridge_queries}"}
                candidate_pool = self.retriever.collect_candidate_pool(
                    search_query,
                    top_k=retrieval_top_k,
                    additional_queries=sub_queries[1:] + bridge_queries,
                    include_follow_ups=False,
                )
                first_hop_keys = {
                    (hit.get("doc_id"), hit.get("chunk_index"))
                    for hit in first_hop_pool
                }
                candidate_keys = {
                    (hit.get("doc_id"), hit.get("chunk_index"))
                    for hit in candidate_pool
                }
                debug_metrics["merged_candidate_count"] = len(candidate_pool)
                debug_metrics["second_hop_added_count"] = len(candidate_keys - first_hop_keys)
                all_hits = self.retriever.finalize_candidates(
                    search_query,
                    candidate_pool,
                    top_k=retrieval_top_k,
                    query_graph=query_graph
                )
                targeting_summary = self._summarize_bridge_targeting_hits(
                    search_query,
                    all_hits,
                    bridge_queries,
                    merged_bridge_entities,
                )
                debug_metrics["bridge_targeting_hits"] = targeting_summary["bridge_targeting_hits"]
                debug_metrics["answer_bearing_bridge_hits"] = targeting_summary["answer_bearing_bridge_hits"]
                debug_metrics["bridge_entity_family_chunk_miss"] = targeting_summary["bridge_entity_family_chunk_miss"]
            else:
                all_hits = initial_hits[:]
                debug_metrics["merged_candidate_count"] = len(first_hop_pool)
            debug_metrics.update(self.retriever.last_search_debug or {})
            seen_chunk_ids = {
                f"{hit.get('doc_id')}_{hit.get('chunk_index')}"
                for hit in all_hits
            }
            
            # Sort final pool by RRF score descending
            all_hits.sort(
                key=lambda x: _safe_numeric(
                    x.get("final_rank_score"),
                    _safe_numeric(x.get("cross_encoder_score"), _safe_numeric(x.get("score", 0.0)))
                ),
                reverse=True
            )
            if benchmark_mode and benchmark_generate_answer and enable_dual_head_scoring:
                retrieval_hits_for_metadata = self._select_dual_head_hits(
                    search_query,
                    all_hits,
                    limit=top_k,
                    max_chunks_per_title=benchmark_max_chunks_per_title
                )
            elif benchmark_mode and benchmark_generate_answer:
                retrieval_hits_for_metadata = self._select_benchmark_top_hits(
                    search_query,
                    all_hits,
                    limit=top_k,
                    max_chunks_per_title=benchmark_max_chunks_per_title
                )
            else:
                retrieval_hits_for_metadata = all_hits[:top_k]
            graph_expanded_hits = all_hits[:top_k * 2] # Keep a larger context pool since we have sub-queries
            metadata_hits = graph_expanded_hits[:]
            support_pool_hits = metadata_hits[:]
            if benchmark_mode and benchmark_generate_answer:
                compact_hits = self._assemble_generation_context_hits(
                    search_query,
                    all_hits,
                    limit=max(1, benchmark_context_limit),
                    max_chunks_per_title=benchmark_generation_max_chunks_per_title,
                    background_limit=benchmark_generation_background_limit,
                    follow_up_queries=debug_metrics["bridge_queries"]
                )
                graph_expanded_hits = compact_hits
                compact_titles = [
                    ((hit.get("graph_context") or {}).get("doc_title") or hit.get("title") or "Unknown")
                    for hit in graph_expanded_hits
                ]
                yield {"type": "thought", "content": f"Benchmark answer context titles: {compact_titles}"}
            generation_hits_for_metadata = graph_expanded_hits[:]
            debug_metrics["final_context_count"] = len(graph_expanded_hits)
            debug_metrics["final_context_titles"] = [
                ((hit.get("graph_context") or {}).get("doc_title") or hit.get("title") or "Unknown")
                for hit in graph_expanded_hits
            ]
            final_titles = [title.lower() for title in debug_metrics["final_context_titles"] if title]
            duplicate_titles = max(0, len(final_titles) - len(set(final_titles)))
            debug_metrics["generation_duplicate_title_ratio"] = duplicate_titles / max(1, len(final_titles))
            
            if not graph_expanded_hits:
                yield {"type": "thought", "content": "No relevant context found in enterprise knowledge."}
                yield {"type": "token", "content": "I could not find any relevant enterprise knowledge to safely answer your query."}
                yield {
                    "type": "answer_metadata",
                    "sources": [],
                    "session_id": sid,
                    "consistency_score": 1.0,
                    "hallucination_warning": False,
                    "debug_metrics": debug_metrics if emit_debug_metadata else None,
                }
                return
                
            yield {"type": "thought", "content": f"Successfully retrieved {len(graph_expanded_hits)} grounded chunks from Enterprise Knowledge Graph."}
            context_str = self._format_context(graph_expanded_hits)

            if benchmark_mode and not benchmark_generate_answer:
                yield {"type": "token", "content": "Benchmark retrieval completed."}
                yield {
                    "type": "answer_metadata",
                    "sources": graph_expanded_hits,
                    "session_id": sid,
                    "consistency_score": 1.0,
                    "hallucination_warning": False,
                    "debug_metrics": debug_metrics if emit_debug_metadata else None,
                }
                return
            
            # --- ReAct Iterative Loop ---
            max_iterations = 2 # Start small for responsiveness
            for iteration in range(max_iterations):
                yield {"type": "thought", "content": f"Evaluating context sufficiency (Iteration {iteration + 1})..."}
                check = self._check_sufficiency(search_query, context_str)
                
                if check["sufficient"]:
                    yield {"type": "thought", "content": "Context is sufficient for a grounded answer."}
                    break
                
                refinement = check["refinement"]
                yield {"type": "thought", "content": f"Information missing. Formulating follow-up query: '{refinement}'"}
                
                # Perform follow-up search
                new_hits = self.retriever.search(refinement, top_k=top_k)
                if not new_hits:
                    yield {"type": "thought", "content": "Follow-up search yielded no new results. Proceeding with best available info."}
                    break
                    
                # Merge unique hits
                newly_added = 0
                for hit in new_hits:
                    unique_id = f"{hit.get('doc_id')}_{hit.get('chunk_index')}"
                    if unique_id not in seen_chunk_ids:
                        graph_expanded_hits.append(hit)
                        seen_chunk_ids.add(unique_id)
                        newly_added += 1
                
                if newly_added == 0:
                    yield {"type": "thought", "content": "No novel information found in follow-up. Terminating loop."}
                    break
                
                yield {"type": "thought", "content": f"Injected {newly_added} new grounded chunks into context pool."}
                context_str = self._format_context(graph_expanded_hits)
            else:
                # Clarification Exhaustion: Loop finished without break (max_iterations reached)
                yield {"type": "thought", "content": "Max iterations reached. Information is still insufficient. Triggering clarification exhaustion..."}
                yield {
                    "type": "clarification", 
                    "content": f"I was able to find some information, but I'm still missing specific details to fully answer: '{search_query}'. Could you please provide more context or clarify your requirements?"
                }
                return
            # --- End ReAct Loop ---
            if not support_pool_hits:
                support_pool_hits = metadata_hits[:] or graph_expanded_hits[:]
        
        if benchmark_mode and benchmark_generate_answer and enable_support_context_inheritance:
            draft_messages = self._build_answer_messages(
                query,
                context_str,
                chat_history,
                benchmark_grounded=True
            )
            draft_answer = self._generate_buffered(draft_messages)
            raw_supporting_seed_hits, supporting_seed_debug = self._select_supporting_hits_with_debug(
                query,
                draft_answer,
                support_pool_hits or graph_expanded_hits,
                limit=benchmark_source_limit,
                follow_up_queries=debug_metrics["bridge_queries"]
            )
            supporting_seed_hits, blocked_seed_count = self._protect_supporting_seed_hits(
                query,
                raw_supporting_seed_hits,
                follow_up_queries=debug_metrics["bridge_queries"],
                limit=benchmark_source_limit
            )
            debug_metrics["supporting_seed_titles"] = [
                ((hit.get("graph_context") or {}).get("doc_title") or hit.get("title") or "Unknown")
                for hit in supporting_seed_hits
            ]
            debug_metrics["blocked_seed_count"] = blocked_seed_count
            debug_metrics["supporting_anchor_filtered_count"] += int(
                supporting_seed_debug.get("anchor_filtered_count", 0) or 0
            )
            inherited_hits = self._assemble_generation_context_hits(
                search_query,
                support_pool_hits or graph_expanded_hits,
                limit=max(1, benchmark_context_limit),
                max_chunks_per_title=benchmark_generation_max_chunks_per_title,
                background_limit=benchmark_generation_background_limit,
                supporting_hits=supporting_seed_hits,
                follow_up_queries=debug_metrics["bridge_queries"]
            )
            if inherited_hits:
                graph_expanded_hits = inherited_hits
                generation_hits_for_metadata = graph_expanded_hits[:]
                context_str = self._format_context(graph_expanded_hits)
                self._update_generation_debug_metrics(
                    debug_metrics,
                    graph_expanded_hits,
                    supporting_hits=supporting_seed_hits,
                    follow_up_queries=debug_metrics["bridge_queries"],
                    compacted_count=int(getattr(self, "_last_generation_compacted_count", 0) or 0),
                )

        # 6. Build LLM Messages payload
        messages = self._build_answer_messages(
            query,
            context_str,
            chat_history,
            benchmark_grounded=benchmark_mode and benchmark_generate_answer
        )
        
        # --- TWO-STAGE PRE-FLIGHT HALLUCINATION INTERCEPTOR ---
        # For high-stakes queries, buffer the full response, validate with
        # ConsistencyEvaluator, and only release if score > 0.8.
        high_stakes = os.getenv("PREFLIGHT_MODE", "auto").lower()
        use_preflight = (high_stakes == "always") or (
            high_stakes == "auto" and intent == "SEARCH"
            and any(kw in query.lower() for kw in [
                "compliance", "legal", "regulation", "policy", "violation",
                "penalty", "risk", "audit", "sec", "gdpr", "hipaa"
            ])
        )
        
        try:
            from agent.consistency import ConsistencyEvaluator
            evaluator = ConsistencyEvaluator(self.openai_client)
            supporting_hits_override: Optional[List[Dict[str, Any]]] = None
            projected_benchmark_answer = ""

            if not self.client:
                fallback_answer = (
                    "Retrieval completed, but no LLM API key is configured for answer synthesis. "
                    "Returning grounded sources only."
                )
                yield {"type": "token", "content": fallback_answer}
                self.memory.add_message(sid, "user", query)
                self.memory.add_message(sid, "assistant", fallback_answer)
                yield {
                    "type": "answer_metadata",
                    "sources": graph_expanded_hits,
                    "session_id": sid,
                    "consistency_score": 1.0,
                    "hallucination_warning": False,
                    "debug_metrics": debug_metrics if emit_debug_metadata else None,
                }
                return
            
            if benchmark_mode and benchmark_generate_answer and enable_support_context_inheritance:
                final_answer = self._generate_buffered(messages)
                supporting_hits_override, final_support_debug = self._select_supporting_hits_with_debug(
                    query,
                    final_answer,
                    support_pool_hits or metadata_hits or graph_expanded_hits,
                    limit=benchmark_source_limit,
                    follow_up_queries=debug_metrics["bridge_queries"]
                )
                supporting_hits_override, blocked_final_seed_count = self._protect_supporting_seed_hits(
                    query,
                    supporting_hits_override,
                    follow_up_queries=debug_metrics["bridge_queries"],
                    limit=benchmark_source_limit
                )
                debug_metrics["blocked_seed_count"] += blocked_final_seed_count
                debug_metrics["supporting_anchor_filtered_count"] += int(
                    final_support_debug.get("anchor_filtered_count", 0) or 0
                )
                debug_metrics["final_supporting_titles"] = [
                    ((hit.get("graph_context") or {}).get("doc_title") or hit.get("title") or "Unknown")
                    for hit in supporting_hits_override
                ]
                debug_metrics["final_supporting_subject_anchored_count"] = int(
                    final_support_debug.get("subject_anchored_count", 0) or 0
                )
                debug_metrics["final_supporting_follow_up_confirmed_count"] = int(
                    final_support_debug.get("follow_up_confirmed_count", 0) or 0
                )
                debug_metrics["final_supporting_unanchored_count"] = int(
                    final_support_debug.get("unanchored_selected_count", 0) or 0
                )
                seed_title_set = {title.lower() for title in debug_metrics["supporting_seed_titles"] if title}
                final_support_title_set = {title.lower() for title in debug_metrics["final_supporting_titles"] if title}
                debug_metrics["seed_mismatch_count"] = len(seed_title_set ^ final_support_title_set)

                refined_generation_hits = self._assemble_generation_context_hits(
                    search_query,
                    support_pool_hits or graph_expanded_hits,
                    limit=max(1, benchmark_context_limit),
                    max_chunks_per_title=benchmark_generation_max_chunks_per_title,
                    background_limit=benchmark_generation_background_limit,
                    supporting_hits=supporting_hits_override,
                    follow_up_queries=debug_metrics["bridge_queries"]
                )
                current_generation_keys = {
                    (hit.get("doc_id"), hit.get("chunk_index"))
                    for hit in generation_hits_for_metadata
                }
                refined_generation_keys = {
                    (hit.get("doc_id"), hit.get("chunk_index"))
                    for hit in refined_generation_hits
                }
                if refined_generation_hits and refined_generation_keys != current_generation_keys:
                    graph_expanded_hits = refined_generation_hits
                    generation_hits_for_metadata = refined_generation_hits[:]
                    context_str = self._format_context(graph_expanded_hits)
                    self._update_generation_debug_metrics(
                        debug_metrics,
                        graph_expanded_hits,
                        supporting_hits=supporting_hits_override,
                        follow_up_queries=debug_metrics["bridge_queries"],
                        compacted_count=int(getattr(self, "_last_generation_compacted_count", 0) or 0),
                    )
                    messages = self._build_answer_messages(
                        query,
                        context_str,
                        chat_history,
                        benchmark_grounded=True
                    )
                    final_answer = self._generate_buffered(messages)
                    supporting_hits_override, rerank_support_debug = self._select_supporting_hits_with_debug(
                        query,
                        final_answer,
                        support_pool_hits or metadata_hits or graph_expanded_hits,
                        limit=benchmark_source_limit,
                        follow_up_queries=debug_metrics["bridge_queries"]
                    )
                    supporting_hits_override, blocked_rerank_seed_count = self._protect_supporting_seed_hits(
                        query,
                        supporting_hits_override,
                        follow_up_queries=debug_metrics["bridge_queries"],
                        limit=benchmark_source_limit
                    )
                    debug_metrics["blocked_seed_count"] += blocked_rerank_seed_count
                    debug_metrics["supporting_anchor_filtered_count"] += int(
                        rerank_support_debug.get("anchor_filtered_count", 0) or 0
                    )
                    debug_metrics["final_supporting_titles"] = [
                        ((hit.get("graph_context") or {}).get("doc_title") or hit.get("title") or "Unknown")
                        for hit in supporting_hits_override
                    ]
                    debug_metrics["final_supporting_subject_anchored_count"] = int(
                        rerank_support_debug.get("subject_anchored_count", 0) or 0
                    )
                    debug_metrics["final_supporting_follow_up_confirmed_count"] = int(
                        rerank_support_debug.get("follow_up_confirmed_count", 0) or 0
                    )
                    debug_metrics["final_supporting_unanchored_count"] = int(
                        rerank_support_debug.get("unanchored_selected_count", 0) or 0
                    )
                    final_support_title_set = {title.lower() for title in debug_metrics["final_supporting_titles"] if title}
                    debug_metrics["seed_mismatch_count"] = len(seed_title_set ^ final_support_title_set)

                eval_result = evaluator.evaluate(final_answer, context_str)
                for token in final_answer.split(" "):
                    yield {"type": "token", "content": token + " "}
            elif use_preflight:
                yield {"type": "thought", "content": "High-stakes query detected. Activating pre-flight validation (buffered mode)..."}
                
                # Stage 1: Generate full response into buffer (no streaming)
                buffered_answer = self._generate_buffered(messages)
                
                # Stage 2: Pre-flight consistency check
                eval_result = evaluator.evaluate(buffered_answer, context_str)
                preflight_score = eval_result.get("consistency_score", 0.0)
                
                if preflight_score > 0.8:
                    yield {"type": "thought", "content": f"Pre-flight validation passed (score: {preflight_score:.2f}). Releasing response."}
                    # Release buffered tokens
                    for token in buffered_answer.split(" "):
                        yield {"type": "token", "content": token + " "}
                    final_answer = buffered_answer
                else:
                    yield {"type": "thought", "content": f"Pre-flight validation FAILED (score: {preflight_score:.2f}). Blocking response."}
                    safety_msg = (
                        "⚠️ Safety Block: The generated response did not pass the pre-flight "
                        "consistency check against the source documents. The answer may contain "
                        "ungrounded claims. Please refine your query or contact the compliance team "
                        "for manual review."
                    )
                    yield {"type": "safety_block", "content": safety_msg}
                    supporting_hits = supporting_hits_override or self._select_supporting_hits(
                        query,
                        buffered_answer,
                        support_pool_hits or metadata_hits or graph_expanded_hits,
                        limit=benchmark_source_limit if benchmark_mode and benchmark_generate_answer else top_k,
                        follow_up_queries=debug_metrics["bridge_queries"] if benchmark_mode else None
                    )
                    yield {
                        "type": "answer_metadata",
                        "sources": supporting_hits or graph_expanded_hits,
                        "session_id": sid,
                        "consistency_score": preflight_score,
                        "hallucination_warning": True,
                        "preflight_blocked": True
                        ,
                        "retrieval_contexts": retrieval_hits_for_metadata if benchmark_mode and benchmark_generate_answer else None,
                        "generation_contexts": generation_hits_for_metadata if benchmark_mode and benchmark_generate_answer else None,
                        "debug_metrics": debug_metrics if emit_debug_metadata else None
                    }
                    return
            else:
                # Standard streaming path (non-high-stakes)
                final_answer = ""
                if self.model_provider == "anthropic":
                    logger.info("Executing Anthropic generation (Model: %s) with stream=True...", self.model)
                    system_prompt = self._build_system_prompt()
                    anthropic_messages = [{"role": m["role"], "content": m["content"]} for m in messages[1:]]
                    
                    with self.client.messages.stream(
                        model=self.model,
                        system=system_prompt,
                        messages=anthropic_messages,
                        max_tokens=1024,
                        temperature=0.0
                    ) as stream:
                        for text in stream.text_stream:
                            final_answer += text
                            yield {"type": "token", "content": text}
                else:
                    logger.info("Executing OpenAI generation (Model: %s) with stream=True...", self.model)
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=messages, # type: ignore
                        temperature=0.0,
                        max_tokens=1024,
                        stream=True
                    )
                    
                    for chunk in response:
                        if not getattr(chunk, "choices", None) or len(chunk.choices) == 0:
                             continue
                        delta = chunk.choices[0].delta
                        if getattr(delta, "content", None):
                            token = delta.content
                            final_answer += token
                            yield {"type": "token", "content": token}
                
                # Post-stream consistency evaluation
                eval_result = evaluator.evaluate(final_answer, context_str)
            
            if benchmark_mode and benchmark_generate_answer:
                projected_benchmark_answer = self._generate_benchmark_short_answer(
                    query,
                    context_str,
                    final_answer,
                )

            # Save new turn to Semantic Memory
            self.memory.add_message(sid, "user", query)
            self.memory.add_message(sid, "assistant", final_answer)
            supporting_hits = supporting_hits_override or self._select_supporting_hits(
                query,
                final_answer,
                support_pool_hits or metadata_hits or graph_expanded_hits,
                limit=benchmark_source_limit if benchmark_mode and benchmark_generate_answer else top_k,
                follow_up_queries=debug_metrics["bridge_queries"] if benchmark_mode else None
            )
            debug_metrics["final_supporting_titles"] = [
                ((hit.get("graph_context") or {}).get("doc_title") or hit.get("title") or "Unknown")
                for hit in supporting_hits
            ]
            
            yield {
                "type": "answer_metadata",
                "sources": supporting_hits or graph_expanded_hits,
                "session_id": sid,
                "consistency_score": eval_result["consistency_score"],
                "hallucination_warning": eval_result["hallucination_warning"],
                "benchmark_answer": projected_benchmark_answer if benchmark_mode and benchmark_generate_answer else None,
                "retrieval_contexts": retrieval_hits_for_metadata if benchmark_mode and benchmark_generate_answer else None,
                "generation_contexts": generation_hits_for_metadata if benchmark_mode and benchmark_generate_answer else None,
                "debug_metrics": debug_metrics if emit_debug_metadata else None
            }
        except Exception as e:
            logger.error("LLM Generation failed: %s", str(e))
            yield {"type": "error", "content": f"Error during response generation: {str(e)}"}
    
    def _generate_buffered(self, messages: List[Dict[str, str]]) -> str:
        """
        Generate a complete LLM response into a buffer (no streaming).
        Used by the pre-flight hallucination interceptor.
        """
        try:
            if self.model_provider == "anthropic":
                system_prompt = messages[0]["content"]
                anthropic_messages = [{"role": m["role"], "content": m["content"]} for m in messages[1:]]
                response = self.client.messages.create(
                    model=self.model,
                    system=system_prompt,
                    messages=anthropic_messages,
                    max_tokens=1024,
                    temperature=0.0
                )
                return response.content[0].text.strip()
            else:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages, # type: ignore
                    temperature=0.0,
                    max_tokens=1024,
                    stream=False
                )
                return response.choices[0].message.content.strip() # type: ignore
        except Exception as e:
            logger.error("Buffered generation failed: %s", str(e))
            return ""
