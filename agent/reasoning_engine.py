"""
Symbolic Reasoning Engine for NovaSearch.

Provides a deterministic verification layer that validates LLM-generated
answers against the symbolic facts retrieved from the Knowledge Graph.
If the generated text contradicts any graph path (A -> REL -> B), the
ConsistencyEvaluator triggers a hard block.

This moves the system from "LLM-first trust" to "Logic-first proof."
"""

import os
import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

try:
    import openai
except ImportError:
    openai = None


# --------------------------------------------------------------------------- #
#  Data Structures
# --------------------------------------------------------------------------- #

@dataclass
class GraphFact:
    """A single symbolic fact extracted from the Knowledge Graph."""
    subject: str
    predicate: str
    obj: str  # 'object' is a Python builtin
    source_cypher: Optional[str] = None

    def __str__(self) -> str:
        return f"({self.subject})-[:{self.predicate}]->({self.obj})"


@dataclass
class ProofResult:
    """Outcome of symbolic verification against graph facts."""
    verified: bool = False
    score: float = 0.0
    contradictions: List[str] = field(default_factory=list)
    supported_facts: List[str] = field(default_factory=list)
    total_facts_checked: int = 0
    blocked: bool = False
    block_reason: Optional[str] = None


# --------------------------------------------------------------------------- #
#  Symbolic Validator
# --------------------------------------------------------------------------- #

class SymbolicValidator:
    """
    Validates LLM-generated answers against retrieved graph paths.

    Two-layer verification:
    1. **Structural Check** — extracts factual claims from the answer and
       checks whether they are present in the graph facts via exact /
       substring matching.
    2. **LLM Proof** — uses GPT-4 Turbo to formally judge whether the
       generated text contradicts any graph fact. If *any* contradiction
       is found, the score is set to 0.0 and a hard block is triggered.

    Usage
    -----
    >>> validator = SymbolicValidator()
    >>> facts = [GraphFact("Alice", "WORKS_AT", "Acme")]
    >>> result = validator.verify("Alice works at Globex.", facts)
    >>> assert result.blocked is True  # contradicts the KG fact
    """

    PROOF_MODEL = "gpt-4-turbo"

    def __init__(self, openai_client=None, model: str = None):
        self.client = openai_client
        if self.client is None and openai is not None:
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                self.client = openai.OpenAI(api_key=api_key)
        self.model = model or self.PROOF_MODEL

    # ------------------------------------------------------------------ #
    #  Structural pre-check (fast, deterministic)
    # ------------------------------------------------------------------ #

    @staticmethod
    def _normalize(text: str) -> str:
        """Lowercase and collapse whitespace for comparison."""
        return re.sub(r"\s+", " ", text.lower().strip())

    def _structural_check(
        self, answer: str, facts: List[GraphFact]
    ) -> Tuple[List[str], List[str]]:
        """
        Fast structural scan — checks whether key entities from graph
        facts appear in the answer text.

        Returns (supported, potential_contradictions).
        """
        norm_answer = self._normalize(answer)
        supported: List[str] = []
        potential_contradictions: List[str] = []

        for fact in facts:
            subj_present = self._normalize(fact.subject) in norm_answer
            obj_present = self._normalize(fact.obj) in norm_answer
            pred_words = set(self._normalize(fact.predicate).replace("_", " ").split())

            if subj_present and obj_present:
                # Both endpoints mentioned → likely supported
                supported.append(str(fact))
            elif subj_present and not obj_present:
                # Subject mentioned but object missing — could be contradictions
                # e.g., answer says "Alice works at Globex" but fact says Acme
                # We flag it for the LLM proof layer
                potential_contradictions.append(str(fact))

        return supported, potential_contradictions

    # ------------------------------------------------------------------ #
    #  LLM Proof Layer (deep, non-deterministic but high-fidelity)
    # ------------------------------------------------------------------ #

    def _llm_proof(
        self, answer: str, facts: List[GraphFact]
    ) -> ProofResult:
        """
        Uses GPT-4 Turbo to formally judge whether the answer contradicts
        any of the provided symbolic facts.
        """
        if not self.client:
            logger.warning(
                "SymbolicValidator: No OpenAI client available. "
                "Falling back to structural-only verification."
            )
            return self._fallback_result(answer, facts)

        facts_block = "\n".join(
            f"  FACT {i+1}: {fact}" for i, fact in enumerate(facts)
        )

        system_prompt = (
            "You are a deterministic fact-checker for an enterprise knowledge system.\n"
            "You are given a list of SYMBOLIC FACTS from a trusted Knowledge Graph and a "
            "GENERATED ANSWER from an LLM.\n\n"
            "Your task:\n"
            "1. For EACH fact, determine if the Generated Answer CONTRADICTS it.\n"
            "   A contradiction means the answer states something that is logically "
            "incompatible with the fact.\n"
            "2. If ANY fact is contradicted, set 'has_contradiction' to true.\n"
            "3. List each contradiction explicitly.\n\n"
            "Respond in JSON with exactly these keys:\n"
            "- 'has_contradiction': boolean\n"
            "- 'contradictions': list of strings (each describing a contradiction)\n"
            "- 'score': float 0.0 (total contradiction) to 1.0 (fully consistent)\n"
        )

        user_prompt = (
            f"SYMBOLIC FACTS (from Knowledge Graph):\n{facts_block}\n\n"
            f"GENERATED ANSWER:\n{answer}\n\n"
            "Analyze and respond in JSON."
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content.strip()

            import json
            data = json.loads(raw)

            has_contradiction = data.get("has_contradiction", False)
            contradictions = data.get("contradictions", [])
            score = float(data.get("score", 1.0))

            result = ProofResult(
                verified=not has_contradiction,
                score=score,
                contradictions=contradictions,
                total_facts_checked=len(facts),
            )

            if has_contradiction or score < 1.0:
                result.blocked = True
                result.block_reason = (
                    f"Symbolic proof failed: {len(contradictions)} contradiction(s) "
                    f"detected against Knowledge Graph facts. Score: {score:.2f}"
                )
                logger.warning("SymbolicValidator: HARD BLOCK — %s", result.block_reason)

            return result

        except Exception as exc:
            logger.error("SymbolicValidator: LLM proof call failed: %s", exc)
            # FAIL-CLOSED: If we can't verify, block
            return ProofResult(
                verified=False,
                score=0.0,
                total_facts_checked=len(facts),
                blocked=True,
                block_reason=f"Symbolic proof engine error: {exc}",
            )

    def _fallback_result(
        self, answer: str, facts: List[GraphFact]
    ) -> ProofResult:
        """Structural-only fallback when no LLM client is available."""
        supported, contradictions = self._structural_check(answer, facts)
        total = len(facts)
        supported_ratio = len(supported) / max(total, 1)

        return ProofResult(
            verified=len(contradictions) == 0,
            score=round(supported_ratio, 2),
            supported_facts=supported,
            contradictions=contradictions,
            total_facts_checked=total,
            blocked=len(contradictions) > 0,
            block_reason=(
                f"Structural check found {len(contradictions)} potential "
                f"contradiction(s): {'; '.join(contradictions)}"
                if contradictions
                else None
            ),
        )

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #

    def verify(
        self, answer: str, graph_facts: List[GraphFact]
    ) -> ProofResult:
        """
        Full two-layer verification pipeline.

        1. Structural pre-check (fast).
        2. LLM proof (deep) — invoked for all sets with graph facts.

        Returns a ``ProofResult`` with *blocked=True* if any contradiction
        is detected.
        """
        if not graph_facts:
            logger.info("SymbolicValidator: No graph facts to check against.")
            return ProofResult(verified=True, score=1.0)

        if not answer or not answer.strip():
            return ProofResult(
                verified=False,
                score=0.0,
                blocked=True,
                block_reason="Empty answer cannot be verified.",
            )

        # Layer 1 — structural
        supported, potential_issues = self._structural_check(answer, graph_facts)
        logger.info(
            "SymbolicValidator structural: %d supported, %d potential issues "
            "out of %d facts.",
            len(supported),
            len(potential_issues),
            len(graph_facts),
        )

        # Layer 2 — LLM proof (always run when facts exist)
        result = self._llm_proof(answer, graph_facts)
        result.supported_facts = supported
        return result

    def extract_facts_from_cypher_results(
        self, cypher_results: List[Dict[str, Any]]
    ) -> List[GraphFact]:
        """
        Utility: convert raw Neo4j Cypher result rows into GraphFact objects.

        Expects each row to have keys like 'subject', 'predicate', 'object'
        or 'source', 'rel_type', 'target'.
        """
        facts: List[GraphFact] = []
        for row in cypher_results:
            subj = row.get("subject") or row.get("source") or row.get("n.name", "")
            pred = row.get("predicate") or row.get("rel_type") or row.get("type(r)", "RELATED_TO")
            obj = row.get("object") or row.get("target") or row.get("m.name", "")

            if subj and obj:
                facts.append(GraphFact(subject=str(subj), predicate=str(pred), obj=str(obj)))

        logger.info("Extracted %d graph facts from Cypher results.", len(facts))
        return facts
