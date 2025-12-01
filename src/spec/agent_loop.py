# src/spec/agent_loop.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Protocol, TYPE_CHECKING

from observation.trace_schema import PlanTrace

if TYPE_CHECKING:
    # Only for type checkers; avoids runtime circular imports.
    from .llm import (
        PlannerModel,
        PlanCodeModel,
        CriticModel,
        ErrorModel,
        ScribeModel,
    )


# ======================================================================
# Core agent loop domain types
# ======================================================================

@dataclass
class AgentGoal:
    """
    High-level objective for a single episode.

    Fields
    ------
    id:
        Stable identifier for logging and curriculum tracking.
    text:
        Natural-language description of the goal (what the agent should do).
    phase:
        Curriculum phase / context identifier (e.g. "P1.steam_age").
    source:
        Where this goal came from:
          - "curriculum"
          - "manual"
          - "fallback"
          - etc.
    """
    id: str
    text: str
    phase: str
    source: str = "curriculum"


@dataclass
class Task:
    """
    One sub-task inside an episode's TaskPlan.

    Typically derived from the planner's Goal → tasks phase.
    """
    id: str
    description: str
    status: str = "pending"  # "pending" | "in_progress" | "done" | "failed"


@dataclass
class TaskPlan:
    """
    Structured representation of a goal's decomposition into tasks.

    This is the planner-facing plan type used by AgentLoop.
    """
    goal_id: str
    tasks: List[Task] = field(default_factory=list)


# ======================================================================
# Virtue / evaluation plumbing
# ======================================================================

class VirtueEngine(Protocol):
    """
    Interface for the virtue lattice scorer (M4).

    Implementations should be pure(ish) functions of PlanTrace + context_id.
    """

    def score_trace(self, trace: PlanTrace, context_id: str) -> Dict[str, float]:
        """
        Compute per-virtue scores for a finished (or candidate) trace.

        Return:
            Mapping from virtue name → score (usually in [0, 1]).
        """
        ...


@dataclass
class PlanEvaluation:
    """
    Canonical evaluation payload attached to a plan at different stages:

      - pre-execution (critic)
      - post-execution (error model / outcome)

    Fields align with Q1 critic/error responses so that both can be
    funneled into the same monitoring / learning pipeline.
    """
    plan_id: str
    attempt_index: int
    virtue_scores: Dict[str, float] = field(default_factory=dict)
    critic_feedback: Dict[str, Any] = field(default_factory=dict)
    failure_type: Optional[str] = None
    severity: Optional[str] = None
    fix_suggestions: Optional[List[str]] = None


@dataclass
class RetryPolicy:
    """
    Retry configuration for the agent loop.

    Q1.1-compatible shape, even if current hierarchical loop does not
    fully use it yet.
    """
    max_retries: int = 0
    retry_budget: int = 0
    retryable_failure_types: List[str] = field(default_factory=list)
    abort_on_severity: List[str] = field(default_factory=list)

