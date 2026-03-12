"""
Quantitative RAG Evaluation Framework for NovaSearch (RAGAS-inspired).

Measures:
    - Faithfulness: Is the answer derived only from retrieved context?
    - Answer Relevancy: Does the answer address the user query?
    - Context Precision: Are the retrieved chunks actually useful?
"""

import os
import json
import logging
import time
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

try:
    import openai
except ImportError:
    openai = None


class RAGEvaluator:
    """
    Lightweight evaluation harness for RAG pipeline quality.
    Uses LLM-as-Judge for automated scoring.
    """
    
    def __init__(self, model: str = "gpt-3.5-turbo"):
        self.model = model
        self.client = None
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key and openai:
            self.client = openai.OpenAI(api_key=api_key)
    
    def _llm_score(self, system_prompt: str, user_prompt: str) -> float:
        """Call LLM to get a score between 0.0 and 1.0."""
        if not self.client:
            logger.warning("No OpenAI client for evaluation. Returning 0.5.")
            return 0.5
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
                max_tokens=32
            )
            result = json.loads(response.choices[0].message.content.strip())
            return float(result.get("score", 0.5))
        except Exception as e:
            logger.error(f"LLM scoring failed: {e}")
            return 0.5
    
    def faithfulness(self, answer: str, context: str) -> float:
        """
        Faithfulness: Is the answer strictly derived from the context?
        Score: 0.0 (hallucinated) to 1.0 (fully grounded).
        """
        system = (
            "You evaluate factual consistency. Given context and an answer, "
            "score how well the answer is supported by the context. "
            "Return JSON: {\"score\": 0.0-1.0}"
        )
        user = f"Context:\n{context}\n\nAnswer:\n{answer}"
        return self._llm_score(system, user)
    
    def answer_relevancy(self, query: str, answer: str) -> float:
        """
        Answer Relevancy: Does the answer address the user's query?
        Score: 0.0 (irrelevant) to 1.0 (perfectly relevant).
        """
        system = (
            "You evaluate answer relevancy. Given a question and an answer, "
            "score how well the answer addresses the question. "
            "Return JSON: {\"score\": 0.0-1.0}"
        )
        user = f"Question:\n{query}\n\nAnswer:\n{answer}"
        return self._llm_score(system, user)
    
    def context_precision(self, query: str, contexts: List[str]) -> float:
        """
        Context Precision: Are the retrieved chunks actually useful?
        Score: 0.0 (all irrelevant) to 1.0 (all relevant).
        """
        system = (
            "You evaluate retrieval quality. Given a question and retrieved chunks, "
            "score how many of the chunks are actually relevant to answering the question. "
            "Return JSON: {\"score\": 0.0-1.0}"
        )
        context_block = "\n---\n".join(contexts[:5])  # Limit to 5 chunks
        user = f"Question:\n{query}\n\nRetrieved Chunks:\n{context_block}"
        return self._llm_score(system, user)
    
    def evaluate(self, query: str, answer: str, contexts: List[str]) -> Dict[str, float]:
        """
        Run all three metrics on a single QA pair.
        """
        context_str = "\n\n".join(contexts)
        
        results = {
            "faithfulness": self.faithfulness(answer, context_str),
            "answer_relevancy": self.answer_relevancy(query, answer),
            "context_precision": self.context_precision(query, contexts),
        }
        
        # Aggregate
        results["overall"] = round(sum(results.values()) / 3, 4)
        return results


# ---------- CLI Runner ----------

BENCHMARK_CASES = [
    {
        "query": "What happens if I violate the insider trading policy?",
        "expected_keywords": ["termination", "SEC", "referral"]
    },
    {
        "query": "Who approves exceptions to the blackout period?",
        "expected_keywords": ["CLO", "Chief Legal Officer", "10b5-1"]
    },
    {
        "query": "What is the standard blackout period duration?",
        "expected_keywords": ["15 days", "fiscal quarter"]
    }
]


def run_benchmark():
    """Run the benchmark suite against the live NovaSearch API."""
    import httpx
    from dotenv import load_dotenv
    load_dotenv()
    
    base_url = os.getenv("NOVASEARCH_URL", "http://localhost:8000")
    api_key = os.getenv("API_KEY", "")
    headers = {"X-API-KEY": api_key}
    
    evaluator = RAGEvaluator()
    all_results = []
    
    print("\n" + "=" * 60)
    print("NovaSearch RAG Benchmark")
    print("=" * 60)
    
    for i, case in enumerate(BENCHMARK_CASES):
        print(f"\n--- Case {i + 1}: {case['query'][:50]}...")
        
        start = time.time()
        try:
            with httpx.Client(timeout=60) as client:
                resp = client.post(
                    f"{base_url}/ask",
                    json={"query": case["query"], "top_k": 5},
                    headers=headers
                )
                
                answer = ""
                contexts = []
                for line in resp.text.strip().split("\n"):
                    if line:
                        data = json.loads(line)
                        if data.get("type") == "token":
                            answer += data.get("content", "")
                        elif data.get("type") == "answer_metadata":
                            contexts = [s.get("chunk_text", "") for s in data.get("sources", [])]
        except Exception as e:
            print(f"  API Error: {e}")
            continue
        
        latency = time.time() - start
        
        if not answer:
            print(f"  No answer received (latency={latency:.2f}s)")
            continue
        
        metrics = evaluator.evaluate(case["query"], answer, contexts)
        metrics["latency_s"] = round(latency, 2)
        
        # Keyword check
        found = sum(1 for kw in case["expected_keywords"] if kw.lower() in answer.lower())
        metrics["keyword_recall"] = round(found / len(case["expected_keywords"]), 2)
        
        all_results.append(metrics)
        
        print(f"  Faithfulness:       {metrics['faithfulness']:.2f}")
        print(f"  Answer Relevancy:   {metrics['answer_relevancy']:.2f}")
        print(f"  Context Precision:  {metrics['context_precision']:.2f}")
        print(f"  Keyword Recall:     {metrics['keyword_recall']:.2f}")
        print(f"  Latency:            {metrics['latency_s']:.2f}s")
    
    if all_results:
        avg = {
            k: round(sum(r[k] for r in all_results) / len(all_results), 3)
            for k in all_results[0]
        }
        print(f"\n{'=' * 60}")
        print(f"AGGREGATE ({len(all_results)} cases):")
        for k, v in avg.items():
            print(f"  {k}: {v}")
        print(f"{'=' * 60}")


if __name__ == "__main__":
    run_benchmark()
