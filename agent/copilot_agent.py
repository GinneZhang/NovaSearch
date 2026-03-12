"""
Enterprise Copilot Agent Logic.

This module acts as the "brain" of NovaSearch. It orchestrates the retrieval
of context via the HybridSearchCoordinator and constructs source-grounded LLM
prompts for the OpenAI API to enforce strict adherence to enterprise facts.
"""

import os
import uuid
import logging
from typing import List, Dict, Any, Optional

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
        self.openai_client = openai.OpenAI(api_key=self.openai_api_key)
        
        if self.model_provider == "anthropic":
            self.model = model or "claude-3-haiku-20240307"
            self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
            if not self.anthropic_api_key:
                logger.warning("ANTHROPIC_API_KEY missing from environment.")
            import anthropic
            self.client = anthropic.Anthropic(api_key=self.anthropic_api_key)
        else:
            self.model = model or "gpt-4-turbo-preview"
            self.client = self.openai_client
        
        # Instantiate Tri-Engine Retrieval Coordinator
        self.retriever = HybridSearchCoordinator()
        
        # Instantiate Semantic Memory
        self.memory = RedisMemoryManager()
        
        # Instantiate LangChain Task Planner
        self.planner = TaskDecomposer(model_name="gpt-3.5-turbo")
        
        # Instantiate Query Graph Parser
        self.query_parser = QueryGraphParser()

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
            
            # Extract Graph Context
            graph_info = hit.get("graph_context")
            if graph_info:
                title = graph_info.get("doc_title", "Unknown Title")
                section = graph_info.get("doc_section", "Unknown Section")
            else:
                title = "Unknown Document"
                section = "General"

            # Format the block with strict tracking markers
            block = f"--- [Document {i+1}] ---\n"
            block += f"[Source Marker]: [Doc: {title}, Section: {section}]\n"
            block += f"[Retrieval Type]: {source.upper()} (Score: {score:.3f})\n"
            block += f"[Content]: {text}\n"
            context_blocks.append(block)

        return "\n".join(context_blocks)

    def _build_system_prompt(self) -> str:
        """
        Constructs the rigorous anti-hallucination system prompt.
        """
        return """You are NovaSearch Copilot, a highly precise, compliance-focused enterprise AI assistant.
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
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=64
            )
            result = response.choices[0].message.content.strip()
            if result.upper() == "SUFFICIENT":
                return {"sufficient": True, "refinement": None}
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
        
        # 1. Ensure Session ID & Fetch History
        sid = session_id or str(uuid.uuid4())
        chat_history = self.memory.get_history(sid, max_turns=5)
        
        # 2. Phase 1: Intent Recognition & Routing
        intent = self._classify_intent(query, chat_history)
        logger.info("Intent Classified as: %s", intent)
        
        if intent == "CLARIFICATION":
            yield {"type": "thought", "content": "Recognized intent as clarification. Bypassing search."}
            graph_expanded_hits = []
            context_str = "No enterprise context needed for clarification."
            search_query = query
        else:
            yield {"type": "thought", "content": "Recognized intent as enterprise search."}
            search_query = self._rewrite_query(query, chat_history)
            if search_query != query:
                yield {"type": "thought", "content": f"Query rewritten to: '{search_query}'"}
                
            # Extract Structured Query Graph (Semantic Triplets)
            yield {"type": "thought", "content": "Parsing structured semantic query graph..."}
            query_graph = self.query_parser.parse(search_query)
            if query_graph:
                yield {"type": "thought", "content": f"Extracted semantic triplets: {query_graph}"}
            
            # Use LangChain to Decompose Complex Queries
            planner_result = self.planner.decompose(search_query)
            if isinstance(planner_result, dict) and planner_result.get("type") == "clarification":
                yield {"type": "thought", "content": "Query is completely ambiguous. Requesting clarification..."}
                yield {"type": "clarification", "content": planner_result.get("content", "Can you please clarify your request? I need more details.")}
                return
                
            sub_queries = planner_result
            if len(sub_queries) > 1:
                yield {"type": "thought", "content": f"Task decomposed into {len(sub_queries)} sub-queries: {sub_queries}"}
            
            # Aggregate Hybrid Results from Sub-Queries
            all_hits = []
            seen_chunk_ids = set()
            
            for sq in sub_queries:
                sq_hits = self.retriever.search(sq, top_k=top_k, query_graph=query_graph)
                for hit in sq_hits:
                    unique_id = f"{hit.get('doc_id')}_{hit.get('chunk_index')}"
                    if unique_id not in seen_chunk_ids:
                        all_hits.append(hit)
                        seen_chunk_ids.add(unique_id)
            
            # Sort final pool by RRF score descending
            all_hits.sort(key=lambda x: x.get("cross_encoder_score", x.get("score", 0.0)), reverse=True)
            graph_expanded_hits = all_hits[:top_k * 2] # Keep a larger context pool since we have sub-queries
            
            if not graph_expanded_hits:
                yield {"type": "thought", "content": "No relevant context found in enterprise knowledge."}
                yield {"type": "token", "content": "I could not find any relevant enterprise knowledge to safely answer your query."}
                yield {
                    "type": "answer_metadata",
                    "sources": [],
                    "session_id": sid,
                    "consistency_score": 1.0,
                    "hallucination_warning": False
                }
                return
                
            yield {"type": "thought", "content": f"Successfully retrieved {len(graph_expanded_hits)} grounded chunks from Enterprise Knowledge Graph."}
            context_str = self._format_context(graph_expanded_hits)
            
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
        
        # 6. Build LLM Messages payload
        messages = [{"role": "system", "content": self._build_system_prompt()}]
        
        # Inject Memory BEFORE the final query context so LLM knows 'past' context
        for msg in chat_history:
            messages.append({"role": msg["role"], "content": msg["content"]})
            
        # Inject the final user prompt wrapped with retrieval context
        final_user_content = (
            f"Please answer the following User Query precisely, using ONLY the provided Source Context.\n\n"
            f"<Source Context Block>\n{context_str}\n</Source Context Block>\n\n"
            f"User Query: {query}"
        )
        messages.append({"role": "user", "content": final_user_content})
        
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
            
            if use_preflight:
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
                    yield {
                        "type": "answer_metadata",
                        "sources": graph_expanded_hits,
                        "session_id": sid,
                        "consistency_score": preflight_score,
                        "hallucination_warning": True,
                        "preflight_blocked": True
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
            
            # Save new turn to Semantic Memory
            self.memory.add_message(sid, "user", query)
            self.memory.add_message(sid, "assistant", final_answer)
            
            yield {
                "type": "answer_metadata",
                "sources": graph_expanded_hits,
                "session_id": sid,
                "consistency_score": eval_result["consistency_score"],
                "hallucination_warning": eval_result["hallucination_warning"]
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
