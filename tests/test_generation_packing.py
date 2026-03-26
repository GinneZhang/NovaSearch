"""
Unit tests for generation context packing and family compaction.
"""

from types import SimpleNamespace

from agent.copilot_agent import EnterpriseCopilotAgent


def _make_agent_stub() -> EnterpriseCopilotAgent:
    agent = EnterpriseCopilotAgent.__new__(EnterpriseCopilotAgent)
    agent.retriever = SimpleNamespace(
        _infer_source_type=lambda hit: hit.get("source_type") or hit.get("source") or "dense",
        _is_multi_hop_query=lambda query: " of " in (query or "").lower(),
    )
    agent.client = None
    agent.openai_client = None
    return agent


def test_extract_brief_answer_text_prefers_short_supported_answer():
    agent = _make_agent_stub()
    answer = (
        "Gesellschaft mit beschrankter Haftung. "
        "VIVA Media GmbH is the renamed entity [Doc: VIVA Media, Section: HotpotQA Benchmark 0]."
    )

    brief = agent._extract_brief_answer_text(answer)

    assert brief == "Gesellschaft mit beschrankter Haftung."


def test_generate_benchmark_short_answer_prefers_non_refusal_projection_without_client():
    agent = _make_agent_stub()

    answer = agent._generate_benchmark_short_answer(
        "What does GmbH stand for?",
        "GmbH means Gesellschaft mit beschrankter Haftung.",
        "Gesellschaft mit beschrankter Haftung. It is a German limited liability company form.",
    )

    assert answer == "Gesellschaft mit beschrankter Haftung."


def test_generate_benchmark_short_answer_prefers_exact_reader_output():
    agent = _make_agent_stub()
    agent.client = object()
    calls = []

    def _fake_generate_buffered(messages):
        calls.append(messages[0]["content"])
        if "exact-answer reader" in messages[0]["content"]:
            return "Gesellschaft mit beschrankter Haftung"
        return "VIVA Media GmbH stands for Gesellschaft mit beschrankter Haftung."

    agent._generate_buffered = _fake_generate_buffered

    answer = agent._generate_benchmark_short_answer(
        "What does GmbH stand for?",
        "GmbH means Gesellschaft mit beschrankter Haftung.",
        "The renamed company used GmbH.",
    )

    assert answer == "Gesellschaft mit beschrankter Haftung"
    assert any("exact-answer reader" in call for call in calls)


def test_should_use_early_second_hop_for_full_chain_mode():
    agent = _make_agent_stub()

    use_second_hop, reason = agent._should_use_early_second_hop(
        "Who is the wife of the author of The Hobbit?",
        ["Who is the wife of the author of The Hobbit?"],
        None,
        [
            {
                "chain_bridge_signal": 0.22,
                "bridge_score": 0.19,
                "primary_chain_member_role": "bridge",
                "best_chain_length": 2,
            }
        ],
        "full",
    )

    assert use_second_hop is True
    assert reason == "full_chain_mode"


def test_should_skip_early_second_hop_for_bypass_mode():
    agent = _make_agent_stub()

    use_second_hop, reason = agent._should_use_early_second_hop(
        "When was Toronto founded?",
        ["When was Toronto founded?"],
        None,
        [],
        "bypass",
    )

    assert use_second_hop is False
    assert reason == "bypass_chain_mode"


def test_prune_bridge_queries_by_retrieval_signal_keeps_best_two_queries():
    agent = _make_agent_stub()

    def _collect_candidate_pool(query, top_k, additional_queries, include_follow_ups=False):
        follow_up = additional_queries[-1]
        return [{"doc_id": follow_up, "chunk_index": 0, "score": 0.9, "retrieval_queries": [follow_up]}]

    def _finalize_candidates(query, candidate_pool, top_k, query_graph=None):
        return candidate_pool

    def _summarize(query, hits, bridge_queries, bridge_entities):
        follow_up = bridge_queries[0]
        if follow_up == "best query":
            return {"bridge_targeting_hits": 2, "answer_bearing_bridge_hits": 2, "bridge_entity_family_chunk_miss": 0}
        if follow_up == "good query":
            return {"bridge_targeting_hits": 2, "answer_bearing_bridge_hits": 1, "bridge_entity_family_chunk_miss": 0}
        return {"bridge_targeting_hits": 1, "answer_bearing_bridge_hits": 0, "bridge_entity_family_chunk_miss": 1}

    agent.retriever.collect_candidate_pool = _collect_candidate_pool
    agent.retriever.finalize_candidates = _finalize_candidates
    agent._summarize_bridge_targeting_hits = _summarize

    kept_queries, debug = agent._prune_bridge_queries_by_retrieval_signal(
        search_query="Who wrote The Hobbit and who was his spouse?",
        retrieval_top_k=10,
        sub_queries=["Who wrote The Hobbit and who was his spouse?"],
        bridge_queries=["weak query", "best query", "good query"],
        bridge_entities=["J. R. R. Tolkien"],
        query_graph=None,
    )

    assert kept_queries == ["best query", "good query"]
    assert debug["bridge_query_pruned_from"] == 3
    assert debug["bridge_query_pruned_to"] == 2


def test_build_evidence_conditioned_follow_up_queries_prioritizes_titles_and_hints():
    agent = _make_agent_stub()
    hits = [
        {
            "title": "J. R. R. Tolkien",
            "graph_context": {"doc_title": "J. R. R. Tolkien"},
            "final_rank_score": 1.1,
            "chain_support_signal": 0.8,
            "answer_score": 0.9,
        },
        {
            "title": "The Hobbit",
            "graph_context": {"doc_title": "The Hobbit"},
            "final_rank_score": 0.9,
            "chain_support_signal": 0.5,
            "answer_score": 0.7,
        },
    ]

    queries = agent._build_evidence_conditioned_follow_up_queries(
        "Who wrote The Hobbit and who was his spouse?",
        hits,
        ["spouse", "author spouse"],
        ["Edith Tolkien"],
    )

    assert "J. R. R. Tolkien" in queries
    assert any(query.startswith("J. R. R. Tolkien ") for query in queries)
    assert any("spouse" in query.lower() for query in queries)


def test_compact_generation_context_merges_adjacent_family_chunks():
    agent = _make_agent_stub()
    hits = [
        {
            "doc_id": "policy",
            "chunk_index": 4,
            "title": "Acme Trading Policy",
            "chunk_text": "Blackout period begins 15 days before quarter end. ",
            "source_type": "dense",
            "evidence_role": "answer",
            "is_corroborated": True,
        },
        {
            "doc_id": "policy",
            "chunk_index": 5,
            "title": "Acme Trading Policy",
            "chunk_text": "Exceptions require written approval from the Chief Legal Officer.",
            "source_type": "dense",
            "evidence_role": "answer",
            "is_corroborated": True,
        },
        {
            "doc_id": "workflow",
            "chunk_index": 0,
            "title": "Exception Workflow",
            "chunk_text": "Escalation requests are logged in the exception register.",
            "source_type": "lexical",
            "evidence_role": "bridge",
            "is_corroborated": True,
        },
    ]

    compacted, compacted_count = agent._compact_generation_context_hits(hits)

    assert compacted_count == 1
    assert len(compacted) == 2
    assert compacted[0]["merged_chunk_indices"] == [4, 5]
    assert "Chief Legal Officer" in compacted[0]["chunk_text"]


def test_generation_assembly_blocks_uncorroborated_symbolic_filler():
    agent = _make_agent_stub()
    supporting_hits = [
        {
            "doc_id": "approvals",
            "chunk_index": 0,
            "title": "Exception Workflow",
            "chunk_text": "Chief Legal Officer approves blackout exceptions.",
            "source_type": "dense",
            "evidence_role": "answer",
            "answer_score": 1.4,
            "bridge_score": 0.2,
            "joint_score": 1.1,
        }
    ]
    hits = [
        supporting_hits[0],
        {
            "doc_id": "policy",
            "chunk_index": 1,
            "title": "Acme Trading Policy",
            "chunk_text": "Blackout period begins 15 days before quarter end.",
            "source_type": "lexical",
            "evidence_role": "answer",
            "answer_score": 1.2,
            "bridge_score": 0.1,
            "joint_score": 0.98,
        },
        {
            "doc_id": "kg",
            "chunk_index": -1,
            "title": "Neo4j Symbolic Path",
            "chunk_text": "[Symbolic Reasoning]: Policy -> APPROVED_BY -> Officer",
            "source_type": "symbolic",
            "source": "dynamic_cypher",
            "evidence_role": "graph",
            "answer_score": 0.1,
            "bridge_score": 0.18,
            "joint_score": 0.3,
        },
    ]

    selected = agent._assemble_generation_context_hits(
        "Who approves blackout exceptions in Acme policy?",
        hits,
        limit=3,
        max_chunks_per_title=2,
        background_limit=1,
        supporting_hits=supporting_hits,
        follow_up_queries=[],
    )


def test_format_ragflow_knowledge_blocks_includes_title_and_content():
    agent = _make_agent_stub()
    blocks = agent._format_ragflow_knowledge_blocks(
        [
            {
                "title": "VIVA Media",
                "chunk_text": "VIVA Media AG was renamed to VIVA Media GmbH in 2004.",
                "source_type": "dense",
            }
        ]
    )

    assert "ID: 1" in blocks
    assert "Title: VIVA Media" in blocks
    assert "Content:" in blocks
    assert "renamed to VIVA Media GmbH" in blocks


def test_run_ragflow_recursive_retrieval_accumulates_new_hits():
    agent = _make_agent_stub()

    agent._check_ragflow_sufficiency = lambda question, knowledge_blocks: {
        "sufficient": False,
        "missing_information": "What does GmbH stand for?",
    }
    agent._generate_ragflow_next_step_queries = lambda question, current_query, missing_information, knowledge_blocks, limit=2: [
        "Gesellschaft mit beschrankter Haftung"
    ]

    def _search(query, top_k=5, query_graph=None):
        return [
            {
                "doc_id": "gmbh",
                "chunk_index": 0,
                "title": "Gesellschaft mit beschrankter Haftung",
                "chunk_text": "GmbH stands for Gesellschaft mit beschrankter Haftung.",
            }
        ]

    def _finalize(query, candidate_pool, top_k, query_graph=None):
        return candidate_pool

    agent.retriever.search = _search
    agent.retriever.finalize_candidates = _finalize

    debug = {}
    hits = agent._run_ragflow_recursive_retrieval(
        "What does the acronym stand for?",
        [
            {
                "doc_id": "viva",
                "chunk_index": 0,
                "title": "VIVA Media",
                "chunk_text": "VIVA Media AG was renamed to VIVA Media GmbH in 2004.",
            }
        ],
        query_graph=None,
        top_k=5,
        debug_metrics=debug,
        depth=1,
    )

    assert len(hits) == 2
    assert debug["ragflow_recursive_added_hits"] == 1
    assert debug["ragflow_recursive_final_count"] == 2

    selected_titles = {hit.get("title") for hit in selected}
    assert "Exception Workflow" in selected_titles
    assert "Acme Trading Policy" in selected_titles
    assert "Neo4j Symbolic Path" not in selected_titles


def test_format_context_uses_citation_first_knowledge_blocks():
    agent = _make_agent_stub()
    formatted = agent._format_context([
        {
            "doc_id": "policy",
            "chunk_index": 0,
            "title": "Trading Policy",
            "chunk_text": "The blackout period starts 15 days before quarter end.",
            "source_type": "dense",
            "evidence_role": "direct",
            "ragflow_family_role": "anchor",
            "final_rank_score": 1.2,
        }
    ])

    assert "[Knowledge 1]" in formatted
    assert "Title: Trading Policy" in formatted
    assert "FamilyRole: ANCHOR" in formatted
    assert "Content:" in formatted


def test_generation_assembly_prefers_support_family_before_orphan_bridge():
    agent = _make_agent_stub()
    supporting_hits = [
        {
            "doc_id": "policy",
            "chunk_index": 0,
            "title": "Acme Trading Policy",
            "chunk_text": "The blackout period starts 15 days before quarter end.",
            "source_type": "dense",
            "evidence_role": "answer",
            "answer_score": 1.35,
            "bridge_score": 0.12,
            "joint_score": 1.05,
        }
    ]
    hits = [
        supporting_hits[0],
        {
            "doc_id": "policy",
            "chunk_index": 1,
            "title": "Acme Trading Policy",
            "chunk_text": "Exceptions require approval from the Chief Legal Officer.",
            "source_type": "lexical",
            "evidence_role": "bridge",
            "answer_score": 0.94,
            "bridge_score": 0.72,
            "joint_score": 1.02,
            "retrieval_queries": ["Who approves blackout exceptions?", "Chief Legal Officer approval"],
            "primary_chain_id": "chain_1",
            "primary_chain_complete": True,
            "chain_bridge_signal": 0.64,
            "chain_support_signal": 0.71,
            "primary_chain_member_role": "bridge",
        },
        {
            "doc_id": "bridge_only",
            "chunk_index": 0,
            "title": "Escalation Register",
            "chunk_text": "Chief Legal Officer escalation approval workflow.",
            "source_type": "dense",
            "evidence_role": "bridge",
            "answer_score": 0.66,
            "bridge_score": 1.05,
            "joint_score": 1.08,
            "retrieval_queries": ["Who approves blackout exceptions?", "Chief Legal Officer approval"],
            "primary_chain_id": "chain_2",
            "primary_chain_complete": False,
            "chain_bridge_signal": 0.82,
            "chain_support_signal": 0.18,
            "primary_chain_member_role": "bridge",
        },
    ]

    selected = agent._assemble_generation_context_hits(
        "Who approves blackout exceptions and when does the blackout period start?",
        hits,
        limit=2,
        max_chunks_per_title=2,
        background_limit=0,
        supporting_hits=supporting_hits,
        follow_up_queries=["Chief Legal Officer approval"],
    )

    selected_titles = {hit.get("title") for hit in selected}
    assert "Acme Trading Policy" in selected_titles
    assert "Escalation Register" not in selected_titles


def test_generation_assembly_keeps_corroborated_graph_hit():
    agent = _make_agent_stub()
    supporting_hits = [
        {
            "doc_id": "workflow",
            "chunk_index": 0,
            "title": "Exception Workflow",
            "chunk_text": "Exception approvals require Chief Legal Officer sign-off.",
            "source_type": "dense",
            "evidence_role": "answer",
            "answer_score": 1.3,
            "bridge_score": 0.2,
            "joint_score": 1.0,
        }
    ]
    hits = [
        supporting_hits[0],
        {
            "doc_id": "workflow",
            "chunk_index": 1,
            "title": "Exception Workflow",
            "chunk_text": "Escalation path confirms CLO approval for blackout exceptions.",
            "source_type": "graph_enriched",
            "evidence_role": "graph",
            "answer_score": 0.95,
            "bridge_score": 0.72,
            "joint_score": 0.92,
            "graph_context": {"doc_title": "Exception Workflow", "doc_section": "Approvals"},
        },
    ]

    selected = agent._assemble_generation_context_hits(
        "Who approves blackout exceptions in Acme policy?",
        hits,
        limit=2,
        max_chunks_per_title=2,
        background_limit=0,
        supporting_hits=supporting_hits,
        follow_up_queries=[],
    )

    assert len(selected) == 1
    assert selected[0]["merged_chunk_indices"] == [0, 1]
    assert "graph_enriched" in selected[0]["merged_source_types"]
    assert "Escalation path confirms CLO approval" in selected[0]["chunk_text"]


def test_generation_assembly_prefers_chain_seeded_bridge_hits():
    agent = _make_agent_stub()
    hits = [
        {
            "doc_id": "policy",
            "chunk_index": 0,
            "title": "Trading Policy",
            "chunk_text": "The blackout period starts 15 days before quarter end.",
            "source_type": "dense",
            "evidence_role": "answer",
            "answer_score": 1.0,
            "bridge_score": 0.1,
            "joint_score": 0.9,
            "primary_chain_id": "chain_1",
            "primary_chain_rank": 1,
            "best_chain_score": 2.1,
            "best_chain_length": 2,
            "primary_chain_complete": True,
            "chain_selected": True,
        },
        {
            "doc_id": "workflow",
            "chunk_index": 0,
            "title": "Exception Workflow",
            "chunk_text": "The Chief Legal Officer approves blackout exceptions.",
            "source_type": "dense",
            "evidence_role": "bridge",
            "answer_score": 1.05,
            "bridge_score": 0.8,
            "joint_score": 1.02,
            "retrieval_queries": [
                "Who approves the exception to the blackout period?",
                "Chief Legal Officer blackout exception",
            ],
            "primary_chain_id": "chain_1",
            "primary_chain_rank": 1,
            "best_chain_score": 2.1,
            "best_chain_length": 2,
            "primary_chain_complete": True,
            "chain_selected": True,
        },
        {
            "doc_id": "background",
            "chunk_index": 0,
            "title": "Company Overview",
            "chunk_text": "Acme was founded in 1998 and operates globally.",
            "source_type": "sparse",
            "evidence_role": "background",
            "answer_score": 0.4,
            "bridge_score": 0.05,
            "joint_score": 0.35,
        },
    ]

    selected = agent._assemble_generation_context_hits(
        "Who approves the exception to the blackout period?",
        hits,
        limit=2,
        max_chunks_per_title=1,
        background_limit=0,
        supporting_hits=[],
        follow_up_queries=["Chief Legal Officer blackout exception"],
    )

    selected_titles = {hit.get("title") for hit in selected}
    assert "Trading Policy" in selected_titles
    assert "Exception Workflow" in selected_titles
    assert "Company Overview" not in selected_titles


def test_generation_assembly_preserves_chain_bridge_role_without_explicit_follow_up_query():
    agent = _make_agent_stub()
    hits = [
        {
            "doc_id": "suburb",
            "chunk_index": 0,
            "title": "Para Hills West, South Australia",
            "chunk_text": "Para Hills West lies in the City of Salisbury.",
            "source_type": "dense",
            "evidence_role": "answer",
            "answer_score": 1.05,
            "bridge_score": 0.18,
            "joint_score": 0.96,
            "primary_chain_id": "chain_1",
            "primary_chain_rank": 1,
            "best_chain_score": 1.9,
            "best_chain_length": 2,
            "primary_chain_complete": True,
            "chain_selected": True,
            "primary_chain_member_role": "support",
            "chain_support_signal": 0.82,
            "chain_bridge_signal": 0.66,
        },
        {
            "doc_id": "city",
            "chunk_index": 0,
            "title": "City of Salisbury",
            "chunk_text": "The City of Salisbury has an estimated population of 148,500.",
            "source_type": "dense",
            "evidence_role": "bridge",
            "answer_score": 0.72,
            "bridge_score": 0.22,
            "joint_score": 0.71,
            "primary_chain_id": "chain_1",
            "primary_chain_rank": 1,
            "best_chain_score": 1.9,
            "best_chain_length": 2,
            "primary_chain_complete": True,
            "chain_selected": True,
            "primary_chain_member_role": "bridge",
            "chain_support_signal": 0.82,
            "chain_bridge_signal": 0.66,
        },
    ]

    selected = agent._assemble_generation_context_hits(
        "Para Hills West, South Australia lies within a city with what estimated population?",
        hits,
        limit=2,
        max_chunks_per_title=1,
        background_limit=0,
        supporting_hits=[],
        follow_up_queries=[],
    )

    selected_titles = {hit.get("title") for hit in selected}
    assert "Para Hills West, South Australia" in selected_titles
    assert "City of Salisbury" in selected_titles


def test_generation_assembly_preserves_complete_chain_bundle():
    agent = _make_agent_stub()
    hits = [
        {
            "doc_id": "hobbit",
            "chunk_index": 0,
            "title": "The Hobbit",
            "chunk_text": "The Hobbit is a novel written by J. R. R. Tolkien.",
            "source_type": "dense",
            "evidence_role": "answer",
            "answer_score": 1.12,
            "bridge_score": 0.20,
            "joint_score": 0.98,
            "primary_chain_id": "chain_1",
            "primary_chain_rank": 1,
            "best_chain_score": 2.10,
            "best_chain_length": 2,
            "primary_chain_complete": True,
            "chain_selected": True,
            "primary_chain_member_role": "support",
            "chain_support_signal": 0.86,
            "chain_bridge_signal": 0.58,
        },
        {
            "doc_id": "tolkien",
            "chunk_index": 0,
            "title": "J. R. R. Tolkien",
            "chunk_text": "Edith Tolkien was the spouse of J. R. R. Tolkien.",
            "source_type": "dense",
            "evidence_role": "bridge",
            "answer_score": 0.96,
            "bridge_score": 0.84,
            "joint_score": 0.95,
            "primary_chain_id": "chain_1",
            "primary_chain_rank": 1,
            "best_chain_score": 2.10,
            "best_chain_length": 2,
            "primary_chain_complete": True,
            "chain_selected": True,
            "primary_chain_member_role": "bridge",
            "chain_support_signal": 0.86,
            "chain_bridge_signal": 0.58,
        },
    ]

    selected = agent._assemble_generation_context_hits(
        "Who wrote The Hobbit and who was his spouse?",
        hits,
        limit=2,
        max_chunks_per_title=1,
        background_limit=0,
        supporting_hits=[],
        follow_up_queries=[],
    )

    selected_titles = {hit.get("title") for hit in selected}
    assert selected_titles == {"The Hobbit", "J. R. R. Tolkien"}
    assert agent._last_generation_selection_debug["chain_bundle_rows_kept"] >= 2


def test_generation_assembly_respects_bypass_mode_bridge_budget():
    agent = _make_agent_stub()
    hits = [
        {
            "doc_id": "policy",
            "chunk_index": 0,
            "title": "Trading Policy",
            "chunk_text": "The blackout period starts 15 days before quarter end and exceptions are approved by the Chief Legal Officer.",
            "source_type": "dense",
            "evidence_role": "answer",
            "answer_score": 1.4,
            "bridge_score": 0.25,
            "joint_score": 1.2,
            "chain_mode_selected": "bypass",
            "chain_activation_reason": "strong_direct_evidence",
        },
        {
            "doc_id": "workflow",
            "chunk_index": 0,
            "title": "Exception Workflow",
            "chunk_text": "Chief Legal Officer exception routing.",
            "source_type": "dense",
            "evidence_role": "bridge",
            "answer_score": 0.55,
            "bridge_score": 0.45,
            "joint_score": 0.5,
            "retrieval_queries": ["Chief Legal Officer exception routing"],
            "chain_mode_selected": "bypass",
            "chain_activation_reason": "strong_direct_evidence",
        },
    ]

    selected = agent._assemble_generation_context_hits(
        "Who approves exceptions to the blackout period?",
        hits,
        limit=2,
        max_chunks_per_title=1,
        background_limit=1,
        supporting_hits=[],
        follow_up_queries=[],
    )

    assert [hit.get("title") for hit in selected] == ["Trading Policy"]
    assert agent._last_generation_selection_debug["chain_mode_selected"] == "bypass"
    assert agent._last_generation_selection_debug["bridge_budget_used"] == 0


def test_generation_debug_metrics_include_bridge_budget_and_fractions():
    agent = _make_agent_stub()
    debug_metrics = {}
    generation_hits = [
        {
            "doc_id": "policy",
            "chunk_index": 0,
            "title": "Trading Policy",
            "chunk_text": "The blackout period starts 15 days before quarter end.",
            "source_type": "dense",
            "evidence_role": "answer",
        }
    ]
    agent._last_generation_selection_debug = {
        "chain_mode_selected": "light",
        "chain_activation_reason": "moderate_chain_benefit",
        "bridge_budget_used": 1,
        "weak_bridge_candidates_dropped": 2,
        "final_context_bridge_fraction": 0.25,
        "final_context_direct_support_fraction": 0.75,
        "chain_vs_standalone_mix": {"chain_backed": 1, "standalone": 0},
    }

    agent._update_generation_debug_metrics(
        debug_metrics=debug_metrics,
        generation_hits=generation_hits,
        supporting_hits=[],
        follow_up_queries=[],
        compacted_count=0,
    )

    assert debug_metrics["generation_chain_mode_selected"] == "light"
    assert debug_metrics["generation_bridge_budget_used"] == 1
    assert debug_metrics["generation_final_context_bridge_fraction"] == 0.25
    assert debug_metrics["generation_chain_vs_standalone_mix"]["chain_backed"] == 1
