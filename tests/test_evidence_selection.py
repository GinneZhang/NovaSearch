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
