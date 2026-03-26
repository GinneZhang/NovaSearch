"""
Formal Agent State Machine for AsterScope.

Provides durable conversational state tracking, a DependencyGraph for
Plan-and-Execute multi-hop flows, and stateful clarification persistence.
"""

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum

logger = logging.getLogger(__name__)


class AgentPhase(str, Enum):
    """Phases the agent can be in during a conversation turn."""
    UNDERSTANDING = "understanding"
    PLANNING = "planning"
    RETRIEVING = "retrieving"
    EVALUATING = "evaluating"
    GENERATING = "generating"
    CLARIFYING = "clarifying"
    SUSPENDED = "suspended"    # Waiting for external input (user, API, etc.)
    REPLANNING = "replanning"  # Critic rejected the plan; re-planning
    COMPLETE = "complete"


@dataclass
class SubTask:
    """A single sub-task in a Plan-and-Execute flow."""
    id: str
    query: str
    depends_on: List[str] = field(default_factory=list)
    status: str = "pending"  # pending | running | completed | failed
    result: Optional[str] = None
    
    def is_ready(self, completed_ids: set) -> bool:
        """Check if all dependencies are satisfied."""
        return all(dep in completed_ids for dep in self.depends_on)


class DependencyGraph:
    """
    Tracks sub-task ordering, completions, and result chaining
    for multi-hop Plan-and-Execute flows.
    """
    
    def __init__(self):
        self.tasks: Dict[str, SubTask] = {}
        self.execution_order: List[str] = []
    
    def add_task(self, task_id: str, query: str, depends_on: Optional[List[str]] = None):
        """Add a sub-task to the graph."""
        self.tasks[task_id] = SubTask(
            id=task_id,
            query=query,
            depends_on=depends_on or []
        )
        self.execution_order.append(task_id)
    
    def get_next_ready(self) -> Optional[SubTask]:
        """Returns the next task whose dependencies are all completed."""
        completed = {tid for tid, t in self.tasks.items() if t.status == "completed"}
        for tid in self.execution_order:
            task = self.tasks[tid]
            if task.status == "pending" and task.is_ready(completed):
                return task
        return None
    
    def mark_completed(self, task_id: str, result: str):
        """Mark a sub-task as completed with its result."""
        if task_id in self.tasks:
            self.tasks[task_id].status = "completed"
            self.tasks[task_id].result = result
    
    def mark_failed(self, task_id: str):
        """Mark a sub-task as failed."""
        if task_id in self.tasks:
            self.tasks[task_id].status = "failed"
    
    def all_completed(self) -> bool:
        """Check if all tasks are completed or failed."""
        return all(t.status in ("completed", "failed") for t in self.tasks.values())
    
    def get_intermediate_results(self) -> Dict[str, str]:
        """Get all intermediate results for context chaining."""
        return {
            tid: t.result
            for tid, t in self.tasks.items()
            if t.status == "completed" and t.result
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize for Redis persistence."""
        return {
            "tasks": {
                tid: {
                    "query": t.query,
                    "depends_on": t.depends_on,
                    "status": t.status,
                    "result": t.result
                }
                for tid, t in self.tasks.items()
            },
            "execution_order": self.execution_order
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DependencyGraph":
        """Deserialize from Redis."""
        graph = cls()
        graph.execution_order = data.get("execution_order", [])
        for tid, tdata in data.get("tasks", {}).items():
            graph.tasks[tid] = SubTask(
                id=tid,
                query=tdata["query"],
                depends_on=tdata.get("depends_on", []),
                status=tdata.get("status", "pending"),
                result=tdata.get("result")
            )
        return graph


@dataclass
class ConversationState:
    """
    Durable state for a single conversation session.
    Persists across API calls via Redis serialization.
    """
    session_id: str
    phase: AgentPhase = AgentPhase.UNDERSTANDING
    original_query: str = ""
    
    # Clarification State
    pending_clarification: Optional[str] = None
    clarification_attempts: int = 0
    max_clarification_attempts: int = 3
    
    # Multi-hop Plan-and-Execute State
    dependency_graph: Optional[DependencyGraph] = None
    
    # Iteration tracking
    react_iteration: int = 0
    max_react_iterations: int = 3
    
    # Accumulated context
    accumulated_context: List[str] = field(default_factory=list)
    
    # Timestamps
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    
    def needs_clarification(self) -> bool:
        """Check if a clarification is pending."""
        return self.pending_clarification is not None
    
    def set_clarification(self, question: str):
        """Set a pending clarification question."""
        self.pending_clarification = question
        self.clarification_attempts += 1
        self.phase = AgentPhase.CLARIFYING
        self.updated_at = time.time()
    
    def resolve_clarification(self, user_response: str):
        """Resolve a pending clarification with the user's response."""
        self.accumulated_context.append(
            f"[Clarification] Q: {self.pending_clarification} A: {user_response}"
        )
        self.pending_clarification = None
        self.phase = AgentPhase.PLANNING
        self.updated_at = time.time()
    
    def can_clarify(self) -> bool:
        """Check if more clarification attempts are allowed."""
        return self.clarification_attempts < self.max_clarification_attempts
    
    def init_plan(self, sub_queries: List[Dict[str, Any]]):
        """Initialize a Plan-and-Execute dependency graph."""
        self.dependency_graph = DependencyGraph()
        for i, sq in enumerate(sub_queries):
            task_id = sq.get("id", f"subtask_{i}")
            depends = sq.get("depends_on", [])
            self.dependency_graph.add_task(task_id, sq["query"], depends)
        self.phase = AgentPhase.RETRIEVING
        self.updated_at = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize for Redis persistence."""
        return {
            "session_id": self.session_id,
            "phase": self.phase.value,
            "original_query": self.original_query,
            "pending_clarification": self.pending_clarification,
            "clarification_attempts": self.clarification_attempts,
            "dependency_graph": self.dependency_graph.to_dict() if self.dependency_graph else None,
            "react_iteration": self.react_iteration,
            "accumulated_context": self.accumulated_context,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConversationState":
        """Deserialize from Redis."""
        state = cls(session_id=data["session_id"])
        state.phase = AgentPhase(data.get("phase", "understanding"))
        state.original_query = data.get("original_query", "")
        state.pending_clarification = data.get("pending_clarification")
        state.clarification_attempts = data.get("clarification_attempts", 0)
        state.react_iteration = data.get("react_iteration", 0)
        state.accumulated_context = data.get("accumulated_context", [])
        state.created_at = data.get("created_at", time.time())
        state.updated_at = data.get("updated_at", time.time())
        
        dg_data = data.get("dependency_graph")
        if dg_data:
            state.dependency_graph = DependencyGraph.from_dict(dg_data)
        
        return state


class StateManager:
    """
    Manages ConversationState persistence via Redis.
    Provides load/save operations for durable multi-turn state tracking.
    """
    
    STATE_PREFIX = "nova:state:"
    STATE_TTL = 3600  # 1 hour TTL
    
    def __init__(self, redis_client=None):
        self.redis = redis_client
    
    def load(self, session_id: str) -> ConversationState:
        """Load state from Redis, or create a new one."""
        if self.redis:
            try:
                raw = self.redis.get(f"{self.STATE_PREFIX}{session_id}")
                if raw:
                    data = json.loads(raw)
                    logger.debug(f"Loaded conversation state for {session_id}")
                    return ConversationState.from_dict(data)
            except Exception as e:
                logger.warning(f"Failed to load state from Redis: {e}")
        
        return ConversationState(session_id=session_id)
    
    def save(self, state: ConversationState):
        """Persist state to Redis."""
        if self.redis:
            try:
                state.updated_at = time.time()
                raw = json.dumps(state.to_dict())
                self.redis.setex(
                    f"{self.STATE_PREFIX}{state.session_id}",
                    self.STATE_TTL,
                    raw
                )
                logger.debug(f"Saved conversation state for {state.session_id}")
            except Exception as e:
                logger.warning(f"Failed to save state to Redis: {e}")
    
    def clear(self, session_id: str):
        """Clear state for a session."""
        if self.redis:
            try:
                self.redis.delete(f"{self.STATE_PREFIX}{session_id}")
            except Exception:
                pass


class PlannerCritic:
    """
    Iterative Planner-Critic loop for complex reasoning tasks.
    
    Flow: Plan -> Execute Step -> Evaluate Observation -> Re-plan if needed.
    Persists evaluation history for auditing.
    """
    
    def __init__(self, max_replan_cycles: int = 2):
        self.max_replan_cycles = max_replan_cycles
        self.evaluation_history: List[Dict[str, Any]] = []
    
    def evaluate_observation(self, plan_step: str, observation: str, original_goal: str) -> Dict[str, Any]:
        """
        Critic evaluation: does the observation satisfy the plan step?
        Returns {"sufficient": bool, "reason": str, "suggested_replan": Optional[str]}
        """
        evaluation = {
            "plan_step": plan_step,
            "observation_preview": observation[:200],
            "sufficient": False,
            "reason": "",
            "suggested_replan": None
        }
        
        # Heuristic checks
        if not observation or len(observation.strip()) < 20:
            evaluation["reason"] = "Observation is empty or too short to be meaningful."
            evaluation["suggested_replan"] = f"Reformulate query for: {plan_step}"
        elif "no relevant" in observation.lower() or "not found" in observation.lower():
            evaluation["reason"] = "Retrieval returned no relevant results."
            evaluation["suggested_replan"] = f"Try broader or alternative terms for: {plan_step}"
        elif "error" in observation.lower():
            evaluation["reason"] = "Observation contains error indicators."
            evaluation["suggested_replan"] = f"Retry with fallback strategy for: {plan_step}"
        else:
            evaluation["sufficient"] = True
            evaluation["reason"] = "Observation appears to contain relevant information."
        
        self.evaluation_history.append(evaluation)
        return evaluation
    
    def should_replan(self, state: ConversationState) -> bool:
        """
        Determine if re-planning is warranted based on evaluation history.
        """
        if not self.evaluation_history:
            return False
        
        recent_failures = sum(
            1 for e in self.evaluation_history[-3:]
            if not e.get("sufficient", True)
        )
        
        replan_count = sum(
            1 for e in self.evaluation_history
            if e.get("suggested_replan")
        )
        
        return recent_failures > 0 and replan_count <= self.max_replan_cycles
    
    def get_replan_suggestions(self) -> List[str]:
        """Get all pending replan suggestions."""
        return [
            e["suggested_replan"]
            for e in self.evaluation_history
            if e.get("suggested_replan") and not e.get("sufficient")
        ]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_replan_cycles": self.max_replan_cycles,
            "evaluation_history": self.evaluation_history
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PlannerCritic":
        pc = cls(max_replan_cycles=data.get("max_replan_cycles", 2))
        pc.evaluation_history = data.get("evaluation_history", [])
        return pc
