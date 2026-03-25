"""
Unit tests for generation context packing and family compaction.
"""

from types import SimpleNamespace

from agent.copilot_agent import EnterpriseCopilotAgent


def _make_agent_stub() -> EnterpriseCopilotAgent:
    agent = EnterpriseCopilotAgent.__new__(EnterpriseCopilotAgent)
    agent.retriever = SimpleNamespace(
        _infer_source_type=lambda hit: hit.get("source_type") or hit.get("source") or "dense"
    )
    return agent


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

    selected_titles = {hit.get("title") for hit in selected}
    assert "Exception Workflow" in selected_titles
    assert "Acme Trading Policy" in selected_titles
    assert "Neo4j Symbolic Path" not in selected_titles


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
