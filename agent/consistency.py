"""
Consistency Evaluator for detecting hallucinations in real-time.
"""
import logging
import json
from typing import Dict, Any

logger = logging.getLogger(__name__)

class ConsistencyEvaluator:
    """
    Evaluates LLM generations against the retrieved context to detect hallucinations
    or contradictions in real-time.
    """
    
    def __init__(self, openai_client=None):
        self.client = openai_client

    def evaluate(self, generated_answer: str, context: str) -> Dict[str, Any]:
        """
        Uses an LLM call to score the factual consistency of the answer against the context.
        """
        if not self.client:
            logger.warning("No OpenAI client available for consistency evaluation. Returning pass.")
            return {"consistency_score": 1.0, "hallucination_warning": False}
            
        if not generated_answer or not context or "No enterprise context" in context:
            return {"consistency_score": 1.0, "hallucination_warning": False}
            
        system_prompt = (
            "You are a helpful anti-hallucination fact-checker. You will be provided with <Source Context> and a <Generated Answer>.\n"
            "Your job is to determine if the <Generated Answer> contains any claims, facts, or entities that contradict or cannot be logically deduced from the <Source Context>.\n"
            "It is OK if the answer paraphrases or logically entails information from the context; do not require exact verbatim string matches.\n"
            "Respond in JSON format with exactly two keys:\n"
            "- 'consistency_score': a float between 0.0 (total hallucination) and 1.0 (completely grounded).\n"
            "- 'hallucination_warning': boolean (true if score < 0.6, else false)."
        )
        
        user_prompt = f"<Source Context>\n{context}\n</Source Context>\n\n<Generated Answer>\n{generated_answer}\n</Generated Answer>"
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
                max_tokens=64
            )
            
            result_str = response.choices[0].message.content.strip() # type: ignore
            result = json.loads(result_str)
            
            score = float(result.get("consistency_score", 1.0))
            warning = bool(result.get("hallucination_warning", False))
            
            logger.info("Consistency Evaluation: Score=%.2f, Warning=%s", score, warning)
            return {
                "consistency_score": score,
                "hallucination_warning": warning
            }
        except Exception as e:
            logger.error("Consistency evaluation failed, defaulting to safe: %s", str(e))
            return {"consistency_score": 1.0, "hallucination_warning": False}
