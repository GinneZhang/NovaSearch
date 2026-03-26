"""
Unit tests for role-aware evidence selection and symbolic gating.
"""

from types import SimpleNamespace

from retrieval.hybrid_search import HybridSearchCoordinator


def _make_coordinator_stub() -> HybridSearchCoordinator:
    coordinator = HybridSearchCoordinator.__new__(HybridSearchCoordinator)
    coordinator.last_search_debug = {}
    coordinator.enable_graph_retrieval = True
    coordinator.neo4j_driver = object()
    coordinator.cypher_gen = object()
    return coordinator


def test_role_aware_selection_preserves_direct_and_bridge():
    coordinator = _make_coordinator_stub()
    query = "What is the blackout period and who approves the exception?"
    candidates = [
        {
            "doc_id": "d1",
            "chunk_index": 0,
            "title": "Acme Trading Policy",
            "chunk_text": "Blackout period begins 15 days before fiscal quarter end.",
            "final_rank_score": 1.6,
            "sources": ["dense", "sparse"],
        },
        {
            "doc_id": "d2",
            "chunk_index": 1,
            "title": "Exception Workflow",
            "chunk_text": "Exceptions are approved by the Chief Legal Officer.",
            "final_rank_score": 1.3,
            "retrieval_queries": ["Chief Legal Officer exception"],
            "sources": ["dense"],
        },
        {
            "doc_id": "kg",
            "chunk_index": -1,
            "title": "Symbolic Path",
            "chunk_text": "[Symbolic Reasoning]: Policy -> APPROVED_BY -> CLO",
            "final_rank_score": 0.9,
            "source": "dynamic_cypher",
        },
        {
            "doc_id": "d3",
            "chunk_index": 0,
            "title": "Company Overview",
            "chunk_text": "Acme was founded in 1998 and has offices worldwide.",
            "final_rank_score": 0.7,
            "sources": ["sparse"],
        },
    ]

    selected, debug = coordinator._select_role_aware_candidates(query=query, candidates=candidates, top_k=3)
    selected_titles = {hit.get("title") for hit in selected}

    assert "Acme Trading Policy" in selected_titles
    assert "Exception Workflow" in selected_titles
    assert len(selected) == 3
    assert debug["evidence_role_mix"].get("symbolic", 0) <= 1
    assert debug["support_coverage_score"] > 0


def test_role_aware_selection_prefers_family_anchor_before_orphan_bridge():
    coordinator = _make_coordinator_stub()
    query = "Who approves blackout exceptions and when does the blackout period start?"
    candidates = [
        {
            "doc_id": "policy",
            "chunk_index": 0,
            "title": "Acme Trading Policy",
            "chunk_text": "The blackout period starts 15 days before quarter end.",
            "final_rank_score": 1.15,
            "sources": ["dense", "lexical"],
        },
        {
            "doc_id": "policy",
            "chunk_index": 1,
            "title": "Acme Trading Policy",
            "chunk_text": "Exceptions require approval from the Chief Legal Officer.",
            "final_rank_score": 1.11,
            "retrieval_queries": [query, "Chief Legal Officer exception approval"],
            "sources": ["dense"],
        },
        {
            "doc_id": "bridge_only",
            "chunk_index": 0,
            "title": "Escalation Register",
            "chunk_text": "Chief Legal Officer escalation approval workflow.",
            "final_rank_score": 1.45,
            "retrieval_queries": [query, "Chief Legal Officer escalation approval"],
            "sources": ["dense"],
        },
    ]

    selected, debug = coordinator._select_role_aware_candidates(query=query, candidates=candidates, top_k=2)
    selected_titles = {hit.get("title") for hit in selected}

    assert "Acme Trading Policy" in selected_titles
    assert debug["selection_drop_reasons"].get("orphan_bridge", 0) >= 1


def test_symbolic_gate_depends_on_query_structure_and_focus_coverage():
    coordinator = _make_coordinator_stub()
    query = "Who approves blackout exceptions in Acme policy?"
    query_graph = [{"subject": "Acme policy", "relation": "APPROVED_BY", "object": "role"}]

    low_focus_hits = [
        {"title": "Acme History", "chunk_text": "Founded in 1998 in Toronto."},
        {"title": "Office Locations", "chunk_text": "Offices across North America."},
    ]
    high_focus_hits = [
        {"title": "Acme Trading Policy", "chunk_text": "Exception approvals require CLO sign-off."},
        {"title": "Blackout Rules", "chunk_text": "Blackout exception workflow includes CLO approval."},
    ]

    assert coordinator._should_use_symbolic_search(query, query_graph, low_focus_hits) is True
    assert coordinator._should_use_symbolic_search(query, query_graph, high_focus_hits) is False


def test_finalize_candidates_emits_role_aware_debug_metrics():
    coordinator = _make_coordinator_stub()
    coordinator.cross_encoder = SimpleNamespace(rerank=lambda _q, hits, top_k=10: hits[:top_k])
    coordinator._expand_candidate_documents = lambda _q, hits, fetch_k=40: hits[:fetch_k]
    coordinator._graph_expansion = lambda hits, query="": hits
    coordinator._deep_graph_search = lambda query, query_graph=None: [{
        "doc_id": "kg",
        "chunk_index": -1,
        "title": "Symbolic Path",
        "chunk_text": "[Symbolic Reasoning]: X -> Y",
        "source": "dynamic_cypher",
        "score": 1.0,
    }]
    coordinator._should_use_symbolic_search = lambda query, query_graph, reranked_hits: True

    pool = [
        {
            "doc_id": "d1",
            "chunk_index": 0,
            "title": "Policy",
            "chunk_text": "Blackout period applies 15 days before quarter end.",
            "score": 1.2,
            "sources": ["dense", "sparse"],
        },
        {
            "doc_id": "d2",
            "chunk_index": 0,
            "title": "Approvals",
            "chunk_text": "Chief Legal Officer approves exceptions.",
            "score": 1.1,
            "retrieval_queries": ["Chief Legal Officer approves exceptions"],
            "sources": ["dense"],
        },
    ]

    selected = coordinator.finalize_candidates(
        query="Who approves blackout exceptions?",
        candidate_pool=pool,
        top_k=2,
        query_graph=[{"subject": "exception", "relation": "APPROVED_BY", "object": "role"}],
    )

    assert len(selected) == 2
    assert coordinator.last_search_debug["symbolic_triggered"] is True
    assert coordinator.last_search_debug["dynamic_cypher_added"] == 1
    assert coordinator.last_search_debug["pre_pack_count"] >= 2
    assert coordinator.last_search_debug["post_pack_count"] == 2
    assert coordinator.last_search_debug["candidate_chains"] >= 1
    assert coordinator.last_search_debug["selected_chains"] >= 1
    assert "evidence_role_mix" in coordinator.last_search_debug
    assert "source_type_mix" in coordinator.last_search_debug


def test_uncorroborated_symbolic_candidates_are_dropped():
    coordinator = _make_coordinator_stub()
    query = "Who approves blackout exceptions?"
    candidates = [
        {
            "doc_id": "d1",
            "chunk_index": 0,
            "title": "Exception Workflow",
            "chunk_text": "Chief Legal Officer approves blackout exceptions.",
            "final_rank_score": 1.3,
            "sources": ["dense"],
        },
        {
            "doc_id": "d2",
            "chunk_index": 0,
            "title": "Acme Trading Policy",
            "chunk_text": "Blackout period begins 15 days before quarter end.",
            "final_rank_score": 1.1,
            "sources": ["lexical"],
        },
        {
            "doc_id": "kg",
            "chunk_index": -1,
            "title": "Neo4j Symbolic Path",
            "chunk_text": "[Symbolic Reasoning]: Policy -> APPROVED_BY -> Officer",
            "final_rank_score": 1.4,
            "source": "dynamic_cypher",
        },
    ]

    selected, debug = coordinator._select_role_aware_candidates(query=query, candidates=candidates, top_k=3)

    assert all(hit.get("title") != "Neo4j Symbolic Path" for hit in selected)
    assert debug["selection_drop_reasons"].get("uncorroborated_graph", 0) >= 1


def test_corroborated_graph_candidates_can_survive_final_selection():
    coordinator = _make_coordinator_stub()
    query = "Who approves blackout exceptions?"
    candidates = [
        {
            "doc_id": "d1",
            "chunk_index": 0,
            "title": "Exception Workflow",
            "chunk_text": "Chief Legal Officer approves blackout exceptions.",
            "final_rank_score": 1.3,
            "sources": ["dense"],
        },
        {
            "doc_id": "d1",
            "chunk_index": 1,
            "title": "Exception Workflow",
            "chunk_text": "Escalation path confirms CLO approval for blackout exceptions.",
            "final_rank_score": 1.05,
            "source": "graph_expansion",
            "graph_context": {"doc_title": "Exception Workflow", "shared_entities": ["Chief Legal Officer"]},
        },
        {
            "doc_id": "d2",
            "chunk_index": 0,
            "title": "Policy Summary",
            "chunk_text": "Acme policy defines the blackout window.",
            "final_rank_score": 0.9,
            "sources": ["sparse"],
        },
    ]

    selected, debug = coordinator._select_role_aware_candidates(query=query, candidates=candidates, top_k=3)

    assert any(hit.get("source_type") == "graph_expansion" for hit in selected)
    assert debug["corroborated_graph_kept"] >= 1


def test_chain_builder_marks_follow_up_reasoning_paths():
    coordinator = _make_coordinator_stub()
    query = "Who approves the exception to the blackout period?"
    candidates = [
        {
            "doc_id": "policy",
            "chunk_index": 0,
            "title": "Trading Policy",
            "chunk_text": "The blackout period starts 15 days before quarter end.",
            "final_rank_score": 1.2,
            "retrieval_queries": [query],
            "sources": ["dense"],
        },
        {
            "doc_id": "workflow",
            "chunk_index": 1,
            "title": "Exception Workflow",
            "chunk_text": "The Chief Legal Officer approves blackout exceptions.",
            "final_rank_score": 1.15,
            "retrieval_queries": [query, "Chief Legal Officer blackout exception"],
            "sources": ["dense"],
        },
    ]

    annotated, debug = coordinator._build_candidate_chains(query, candidates)

    workflow_hit = next(hit for hit in annotated if hit["doc_id"] == "workflow")
    assert debug["candidate_chains"] >= 1
    assert debug["selected_chains"] >= 1
    assert workflow_hit["primary_chain_id"]
    assert workflow_hit["best_chain_length"] >= 1


def test_role_aware_selection_respects_chain_bridge_role_without_follow_up_overlap():
    coordinator = _make_coordinator_stub()
    query = "What city contains Para Hills West, South Australia and what is its population?"
    candidates = [
        {
            "doc_id": "suburb",
            "chunk_index": 0,
            "title": "Para Hills West, South Australia",
            "chunk_text": "Para Hills West is a suburb in the City of Salisbury.",
            "final_rank_score": 1.22,
            "primary_chain_id": "chain_1",
            "primary_chain_rank": 1,
            "best_chain_score": 1.95,
            "best_chain_length": 2,
            "primary_chain_complete": True,
            "chain_selected": True,
            "primary_chain_member_role": "support",
            "chain_support_signal": 0.82,
            "chain_bridge_signal": 0.66,
            "sources": ["dense"],
        },
        {
            "doc_id": "city",
            "chunk_index": 0,
            "title": "City of Salisbury",
            "chunk_text": "The City of Salisbury has an estimated population of 148,500.",
            "final_rank_score": 1.05,
            "primary_chain_id": "chain_1",
            "primary_chain_rank": 1,
            "best_chain_score": 1.95,
            "best_chain_length": 2,
            "primary_chain_complete": True,
            "chain_selected": True,
            "primary_chain_member_role": "bridge",
            "chain_support_signal": 0.82,
            "chain_bridge_signal": 0.66,
            "sources": ["dense"],
        },
    ]

    selected, debug = coordinator._select_role_aware_candidates(query=query, candidates=candidates, top_k=2, chain_mode="full")

    selected_titles = {hit.get("title") for hit in selected}
    assert "Para Hills West, South Australia" in selected_titles
    assert "City of Salisbury" in selected_titles
    assert debug["bridge_chunks_kept"] >= 1


def test_role_aware_selection_preserves_complete_chain_bundle():
    coordinator = _make_coordinator_stub()
    query = "Who wrote The Hobbit and who was his spouse?"
    candidates = [
        {
            "doc_id": "hobbit",
            "chunk_index": 0,
            "title": "The Hobbit",
            "chunk_text": "The Hobbit is a novel written by J. R. R. Tolkien.",
            "final_rank_score": 1.22,
            "primary_chain_id": "chain_1",
            "primary_chain_rank": 1,
            "best_chain_score": 2.14,
            "best_chain_length": 2,
            "primary_chain_complete": True,
            "chain_selected": True,
            "primary_chain_member_role": "support",
            "chain_support_signal": 0.88,
            "chain_bridge_signal": 0.62,
            "sources": ["dense"],
        },
        {
            "doc_id": "tolkien",
            "chunk_index": 0,
            "title": "J. R. R. Tolkien",
            "chunk_text": "Edith Tolkien was the spouse of J. R. R. Tolkien.",
            "final_rank_score": 1.04,
            "primary_chain_id": "chain_1",
            "primary_chain_rank": 1,
            "best_chain_score": 2.14,
            "best_chain_length": 2,
            "primary_chain_complete": True,
            "chain_selected": True,
            "primary_chain_member_role": "bridge",
            "chain_support_signal": 0.88,
            "chain_bridge_signal": 0.62,
            "sources": ["dense"],
        },
        {
            "doc_id": "background",
            "chunk_index": 0,
            "title": "Fantasy Literature",
            "chunk_text": "Fantasy literature became globally popular in the twentieth century.",
            "final_rank_score": 0.72,
            "sources": ["sparse"],
        },
    ]

    selected, debug = coordinator._select_role_aware_candidates(query=query, candidates=candidates, top_k=2, chain_mode="full")

    selected_titles = {hit.get("title") for hit in selected}
    assert selected_titles == {"The Hobbit", "J. R. R. Tolkien"}
    assert debug["chain_bundle_rows_kept"] >= 2


def test_decide_chain_mode_bypasses_when_direct_evidence_is_strong():
    coordinator = _make_coordinator_stub()
    reranked_hits = [
        {
            "doc_id": "d1",
            "chunk_index": 0,
            "title": "Blackout Policy",
            "chunk_text": "The blackout period starts 15 days before quarter end and exceptions are approved by the Chief Legal Officer.",
            "final_rank_score": 1.6,
        },
        {
            "doc_id": "d2",
            "chunk_index": 0,
            "title": "Approval Policy",
            "chunk_text": "The Chief Legal Officer approves exceptions to the blackout period.",
            "final_rank_score": 1.45,
        },
        {
            "doc_id": "d3",
            "chunk_index": 0,
            "title": "Policy FAQ",
            "chunk_text": "Blackout exceptions require CLO sign-off.",
            "final_rank_score": 1.35,
        },
    ]

    decision = coordinator._decide_chain_mode(
        query="Who approves exceptions to the blackout period?",
        reranked_hits=reranked_hits,
        query_graph=[],
        top_k=5,
    )

    assert decision["mode"] == "bypass"
    assert decision["reason"] in {"strong_direct_evidence", "simple_query_strong_direct_evidence"}


def test_decide_chain_mode_uses_full_when_bridge_pressure_is_high():
    coordinator = _make_coordinator_stub()
    reranked_hits = [
        {
            "doc_id": "policy",
            "chunk_index": 0,
            "title": "Trading Policy",
            "chunk_text": "The blackout period starts before quarter end.",
            "final_rank_score": 1.1,
            "retrieval_queries": ["Who approves the exception to the blackout period?"],
        },
        {
            "doc_id": "workflow",
            "chunk_index": 0,
            "title": "Exception Workflow",
            "chunk_text": "The Chief Legal Officer approves blackout exceptions.",
            "final_rank_score": 1.0,
            "retrieval_queries": [
                "Who approves the exception to the blackout period?",
                "Chief Legal Officer blackout exception",
                "blackout exception approver",
            ],
        },
    ]

    decision = coordinator._decide_chain_mode(
        query="Who approves the exception to the blackout period?",
        reranked_hits=reranked_hits,
        query_graph=[{"subject": "exception", "relation": "APPROVED_BY", "object": "officer"}],
        top_k=5,
    )

    assert decision["mode"] == "full"
    assert decision["reason"] == "bridge_or_multi_step_pressure"


def test_role_aware_selection_enforces_bridge_budget_in_bypass_mode():
    coordinator = _make_coordinator_stub()
    query = "What is the blackout period and who approves the exception?"
    candidates = [
        {
            "doc_id": "d1",
            "chunk_index": 0,
            "title": "Policy",
            "chunk_text": "The blackout period starts 15 days before quarter end and exceptions are approved by the Chief Legal Officer.",
            "final_rank_score": 1.8,
            "sources": ["dense"],
        },
        {
            "doc_id": "d2",
            "chunk_index": 0,
            "title": "Exception Workflow",
            "chunk_text": "Chief Legal Officer exception path.",
            "final_rank_score": 1.1,
            "retrieval_queries": ["Chief Legal Officer exception"],
            "sources": ["dense"],
        },
    ]

    selected, debug = coordinator._select_role_aware_candidates(
        query=query,
        candidates=candidates,
        top_k=2,
        chain_mode="bypass",
        chain_activation_reason="strong_direct_evidence",
    )

    assert {hit["title"] for hit in selected} == {"Policy"}
    assert debug["bridge_budget_used"] == 0
    assert debug["selection_drop_reasons"].get("bridge_budget", 0) >= 1
    assert debug["chain_mode_selected"] == "bypass"


def test_finalize_candidates_emits_chain_mode_and_pruning_debug():
    coordinator = _make_coordinator_stub()
    coordinator.cross_encoder = SimpleNamespace(rerank=lambda _q, hits, top_k=10: hits[:top_k])
    coordinator._expand_candidate_documents = lambda _q, hits, fetch_k=40: hits[:fetch_k]
    coordinator._graph_expansion = lambda hits, query="": hits
    coordinator._deep_graph_search = lambda query, query_graph=None: []
    coordinator._should_use_symbolic_search = lambda query, query_graph, reranked_hits: False

    pool = [
        {
            "doc_id": "d1",
            "chunk_index": 0,
            "title": "Policy",
            "chunk_text": "The blackout period starts 15 days before quarter end and exceptions are approved by the Chief Legal Officer.",
            "score": 1.4,
            "sources": ["dense", "sparse"],
        },
        {
            "doc_id": "d2",
            "chunk_index": 0,
            "title": "Workflow",
            "chunk_text": "Chief Legal Officer approval flow.",
            "score": 1.0,
            "retrieval_queries": ["Chief Legal Officer approval flow"],
            "sources": ["dense"],
        },
    ]

    coordinator.finalize_candidates(
        query="Who approves exceptions to the blackout period?",
        candidate_pool=pool,
        top_k=2,
        query_graph=[],
    )

    assert coordinator.last_search_debug["chain_mode_selected"] in {"bypass", "light", "full"}
    assert "chain_activation_reason" in coordinator.last_search_debug
    assert "second_hop_candidates_added" in coordinator.last_search_debug
    assert "bridge_budget_used" in coordinator.last_search_debug
    assert "chain_score_components" in coordinator.last_search_debug
