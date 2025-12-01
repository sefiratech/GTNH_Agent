#src/learning/schema.py

"""
Experience & skill-learning schema for M10.

This module defines the core data structures used by the learning layer:
- ExperienceEpisode: a single plan/episode as seen by M10 (legacy-compatible)
- Experience: Q1.4-first-class experience object for replay / similarity
- SkillPerformanceStats: aggregated metrics for a single skill
- SkillCandidate: a proposed new or refined skill

The design assumes:
- TechState comes from `semantics.schema.TechState`
- PlanTrace comes from `observation.trace_schema.PlanTrace`
- Virtue scores are produced by M4 (virtues.*)
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict, is_dataclass
from typing import Any, Dict, List, Optional, Literal, Callable, TypeVar
from datetime import datetime

# External schemas from existing modules
from semantics.schema import TechState          # M3 tech progression representation
from observation.trace_schema import PlanTrace  # M7 execution trace representation
from spec.agent_loop import AgentGoal, TaskPlan
from spec.skills import SkillInvocation


# ---------------------------------------------------------------------------
# Type aliases & small helpers
# ---------------------------------------------------------------------------

VirtueScores = Dict[str, float]
EpisodeId = str
CandidateId = str
SkillName = str

SkillCandidateStatus = Literal[
    "proposed",    # just synthesized by M10
    "evaluating",  # being A/B tested or under review
    "accepted",    # promoted into main registry
    "rejected",    # discarded / archived
]

T = TypeVar("T")


def _maybe_to_dict(obj: Any) -> Any:
    """
    Best-effort conversion of arbitrary objects into JSON-friendly structures.

    Priority:
    1. If object has `to_dict()`, call it.
    2. If it's a dataclass, use `asdict()`.
    3. If it has __dict__, return that.
    4. Otherwise, return as-is (caller must ensure JSON-serializable).
    """
    if obj is None:
        return None

    to_dict = getattr(obj, "to_dict", None)
    if callable(to_dict):
        return to_dict()

    if is_dataclass(obj):
        return asdict(obj)

    if hasattr(obj, "__dict__"):
        return dict(obj.__dict__)

    return obj


def _datetime_to_iso(dt: Optional[datetime]) -> Optional[str]:
    if dt is None:
        return None
    return dt.isoformat()


def _datetime_from_iso(value: Optional[str]) -> Optional[datetime]:
    if value is None:
        return None
    return datetime.fromisoformat(value)


# ---------------------------------------------------------------------------
# Core episode schema (legacy / storage-oriented)
# ---------------------------------------------------------------------------

@dataclass
class EpisodeMetadata:
    """
    Auxiliary metadata for an episode.

    Mirrors what M9 / runtime already know:
    - timestamps
    - environment / profile IDs
    - tags (e.g., curriculum slice, skill pack, scenario)
    """
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    env_profile_id: Optional[str] = None       # link back to EnvProfile / env.yaml
    curriculum_id: Optional[str] = None        # which curriculum unit (M11) this came from
    skill_pack_id: Optional[str] = None        # which skill pack was active (e.g. "steam_age")
    tags: List[str] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "start_time": _datetime_to_iso(self.start_time),
            "end_time": _datetime_to_iso(self.end_time),
            "env_profile_id": self.env_profile_id,
            "curriculum_id": self.curriculum_id,
            "skill_pack_id": self.skill_pack_id,
            "tags": list(self.tags),
            "extra": dict(self.extra),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EpisodeMetadata":
        return cls(
            start_time=_datetime_from_iso(data.get("start_time")),
            end_time=_datetime_from_iso(data.get("end_time")),
            env_profile_id=data.get("env_profile_id"),
            curriculum_id=data.get("curriculum_id"),
            skill_pack_id=data.get("skill_pack_id"),
            tags=list(data.get("tags") or []),
            extra=dict(data.get("extra") or {}),
        )


@dataclass
class ExperienceEpisode:
    """
    A single learning episode derived from one plan execution in M8.

    Original (pre-Q1) fields:
        - id
        - goal
        - tech_state
        - trace
        - virtue_scores
        - success
        - metadata

    Q1 extensions:
        - plan: structured planner output
        - pre_eval: pre-execution evaluation (Critic / virtues, etc.)
        - post_eval: post-execution evaluation (ErrorModel, etc.)
        - final_outcome: high-level outcome summary for the episode
        - failure_type / severity: rolled-up labels for easy filtering

    NOTE:
    - This struct is storage-oriented and legacy-friendly.
      Q1.4 introduces a higher-level `Experience` type below.
    """

    # Original required fields (kept as-is for compatibility)
    id: EpisodeId
    goal: str
    tech_state: TechState
    trace: PlanTrace
    virtue_scores: VirtueScores
    success: bool

    # Original metadata with default
    metadata: EpisodeMetadata = field(default_factory=EpisodeMetadata)

    # Q1 fields (all optional / defaulted)
    plan: Dict[str, Any] = field(default_factory=dict)
    pre_eval: Dict[str, Any] = field(default_factory=dict)
    post_eval: Dict[str, Any] = field(default_factory=dict)
    final_outcome: Dict[str, Any] = field(default_factory=dict)
    failure_type: Optional[str] = None
    severity: Optional[str] = None

    # ------------------------------------------------------------------
    # Serialization helpers
    # ------------------------------------------------------------------

    def to_dict(
        self,
        *,
        tech_state_to_dict: Optional[Callable[[TechState], Dict[str, Any]]] = None,
        plan_trace_to_dict: Optional[Callable[[PlanTrace], Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Convert this episode into a JSON-serializable dict.

        Callers can override how TechState and PlanTrace are serialized by
        passing `tech_state_to_dict` and `plan_trace_to_dict`.
        """
        if tech_state_to_dict is None:
            tech_state_to_dict = _maybe_to_dict  # type: ignore[assignment]
        if plan_trace_to_dict is None:
            plan_trace_to_dict = _maybe_to_dict  # type: ignore[assignment]

        return {
            "id": self.id,
            "goal": self.goal,
            "tech_state": tech_state_to_dict(self.tech_state),
            "trace": plan_trace_to_dict(self.trace),
            "virtue_scores": dict(self.virtue_scores),
            "success": self.success,
            "metadata": self.metadata.to_dict(),
            # Q1+ fields:
            "plan": _maybe_to_dict(self.plan),
            "pre_eval": _maybe_to_dict(self.pre_eval),
            "post_eval": _maybe_to_dict(self.post_eval),
            "final_outcome": _maybe_to_dict(self.final_outcome),
            "failure_type": self.failure_type,
            "severity": self.severity,
        }

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
        *,
        tech_state_from_dict: Callable[[Dict[str, Any]], TechState],
        plan_trace_from_dict: Callable[[Dict[str, Any]], PlanTrace],
    ) -> "ExperienceEpisode":
        """
        Reconstruct an ExperienceEpisode from a dict.

        Handles both:
        - old episodes (no plan/pre_eval/post_eval/final_outcome/failure fields)
        - new Q1-style episodes with full data.
        """
        tech_state_raw = data.get("tech_state")
        trace_raw = data.get("trace")

        return cls(
            id=data["id"],
            goal=data["goal"],
            tech_state=tech_state_from_dict(tech_state_raw) if isinstance(tech_state_raw, dict) else tech_state_raw,
            trace=plan_trace_from_dict(trace_raw) if isinstance(trace_raw, dict) else trace_raw,
            virtue_scores=dict(data.get("virtue_scores") or {}),
            success=bool(data["success"]),
            metadata=EpisodeMetadata.from_dict(data.get("metadata") or {}),
            plan=dict(data.get("plan") or {}),
            pre_eval=dict(data.get("pre_eval") or {}),
            post_eval=dict(data.get("post_eval") or {}),
            final_outcome=dict(data.get("final_outcome") or {}),
            failure_type=data.get("failure_type"),
            severity=data.get("severity"),
        )


# ---------------------------------------------------------------------------
# Q1.4: First-class Experience object (replay / similarity)
# ---------------------------------------------------------------------------

@dataclass
class ExperiencePlan:
    """
    Plan payload for an Experience:

    - task_plan: high-level TaskPlan from the hierarchical planner
    - skill_invocations: flattened SkillInvocation list used for execution
    """
    task_plan: TaskPlan
    skill_invocations: List[SkillInvocation]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_plan": _maybe_to_dict(self.task_plan),
            "skill_invocations": [
                _maybe_to_dict(inv) for inv in self.skill_invocations
            ],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperiencePlan":
        task_plan_raw = data.get("task_plan") or {}
        inv_raw = data.get("skill_invocations") or []

        # Best-effort reconstruction; caller can pass fully-typed objects
        # and let _maybe_to_dict handle serialization.
        task_plan: TaskPlan
        if isinstance(task_plan_raw, TaskPlan):
            task_plan = task_plan_raw
        else:
            task_plan = TaskPlan(
                goal_id=str(task_plan_raw.get("goal_id", "")),
                tasks=[],  # Q1.4 can extend this with richer reconstruction
            )

        invocations: List[SkillInvocation] = []
        for item in inv_raw:
            if isinstance(item, SkillInvocation):
                invocations.append(item)
            elif isinstance(item, dict):
                # Protocol-friendly; concrete SkillInvocation implementations
                # can add richer reconstruction later.
                invocations.append(  # type: ignore[arg-type]
                    SkillInvocation(  # type: ignore[call-arg]
                        task_id=str(item.get("task_id", "")),
                        skill_name=str(item.get("skill_name", item.get("skill", ""))),
                        parameters=dict(item.get("parameters") or item.get("params") or {}),
                        expected_outcome=str(item.get("expected_outcome", "")),
                    )
                )
        return cls(task_plan=task_plan, skill_invocations=invocations)


@dataclass
class Experience:
    """
    Q1.4-first-class experience object used by the replay buffer.

    Fields:
        problem_signature:
            Compact, semi-stable identifier for "what problem this was",
            e.g. hashes of (goal, tech_state, environment slice).

        goal:
            AgentGoal selected for this episode.

        plan:
            ExperiencePlan: TaskPlan + SkillInvocations used for execution.

        attempts:
            List of attempt/evaluation snapshots, including retries/failures.
            For Q1.4 this is kept as a generic list[dict]; higher layers
            can define a richer Attempt schema later.

        final_outcome:
            Structured outcome summary; typically contains:
                - success flag
                - error codes / failure_type
                - aggregate stats

        virtue_scores:
            Final virtue score vector for the episode.

        lessons:
            Human/LLM-oriented summary of "what we learned" from this episode.
    """
    problem_signature: Dict[str, Any]
    goal: AgentGoal
    plan: ExperiencePlan
    attempts: List[Dict[str, Any]]
    final_outcome: Dict[str, Any]
    virtue_scores: VirtueScores
    lessons: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "problem_signature": dict(self.problem_signature),
            "goal": _maybe_to_dict(self.goal),
            "plan": self.plan.to_dict(),
            "attempts": [dict(a) for a in self.attempts],
            "final_outcome": _maybe_to_dict(self.final_outcome),
            "virtue_scores": dict(self.virtue_scores),
            "lessons": self.lessons,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Experience":
        goal_raw = data.get("goal") or {}
        if isinstance(goal_raw, AgentGoal):
            goal = goal_raw
        else:
            goal = AgentGoal(
                id=str(goal_raw.get("id", "")),
                text=str(goal_raw.get("text", "")),
                phase=str(goal_raw.get("phase", "")),
                source=str(goal_raw.get("source", "")),
            )

        plan_raw = data.get("plan") or {}
        plan = ExperiencePlan.from_dict(plan_raw)

        return cls(
            problem_signature=dict(data.get("problem_signature") or {}),
            goal=goal,
            plan=plan,
            attempts=[dict(a) for a in (data.get("attempts") or [])],
            final_outcome=dict(data.get("final_outcome") or {}),
            virtue_scores=dict(data.get("virtue_scores") or {}),
            lessons=str(data.get("lessons", "")),
        )


# ---------------------------------------------------------------------------
# Aggregated performance metrics
# ---------------------------------------------------------------------------

@dataclass
class SkillPerformanceStats:
    """
    Aggregated statistics for a single skill across many episodes.

    This struct is computed by the evaluator and is used both for:
    - baseline skills (current registry entries)
    - candidate skills (after experimental rollout)
    """
    skill_name: SkillName
    uses: int
    success_rate: float
    avg_time: float
    avg_resource_cost: float
    avg_virtue_scores: VirtueScores = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "skill_name": self.skill_name,
            "uses": int(self.uses),
            "success_rate": float(self.success_rate),
            "avg_time": float(self.avg_time),
            "avg_resource_cost": float(self.avg_resource_cost),
            "avg_virtue_scores": dict(self.avg_virtue_scores),
        }

    @classmethod
    def zero(cls, skill_name: SkillName) -> "SkillPerformanceStats":
        """Convenience constructor for 'no data yet' stats."""
        return cls(
            skill_name=skill_name,
            uses=0,
            success_rate=0.0,
            avg_time=0.0,
            avg_resource_cost=0.0,
            avg_virtue_scores={},
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SkillPerformanceStats":
        return cls(
            skill_name=data["skill_name"],
            uses=int(data.get("uses", 0)),
            success_rate=float(data.get("success_rate", 0.0)),
            avg_time=float(data.get("avg_time", 0.0)),
            avg_resource_cost=float(data.get("avg_resource_cost", 0.0)),
            avg_virtue_scores=dict(data.get("avg_virtue_scores") or {}),
        )


# ---------------------------------------------------------------------------
# Skill candidates
# ---------------------------------------------------------------------------

@dataclass
class SkillCandidate:
    """
    A proposed new or refined skill produced by M10.

    This struct is the bridge between:
    - LLM-based synthesis (spec_yaml + impl_code + rationale)
    - evaluation (metrics_before / metrics_after)
    - lifecycle management (status)
    """
    id: CandidateId
    base_skill_name: Optional[SkillName]          # None if brand new, else refined skill
    spec_yaml: str                                # full YAML for SkillSpec
    impl_code: str                                # Python implementation stub
    rationale: str                                # explanation from synthesizer
    status: SkillCandidateStatus                  # lifecycle state

    metrics_before: Optional[SkillPerformanceStats] = None
    metrics_after: Optional[SkillPerformanceStats] = None

    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    tags: List[str] = field(default_factory=list)  # arbitrary labels (e.g., "steam_age", "LV")
    extra: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Lifecycle helpers
    # ------------------------------------------------------------------

    def mark_status(self, new_status: SkillCandidateStatus) -> None:
        """Update candidate status and bump updated_at."""
        self.status = new_status
        self.updated_at = datetime.utcnow()

    def touch(self) -> None:
        """Bump updated_at without changing status."""
        self.updated_at = datetime.utcnow()

    # ------------------------------------------------------------------
    # Serialization helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "base_skill_name": self.base_skill_name,
            "spec_yaml": self.spec_yaml,
            "impl_code": self.impl_code,
            "rationale": self.rationale,
            "status": self.status,
            "metrics_before": (
                self.metrics_before.to_dict() if self.metrics_before else None
            ),
            "metrics_after": (
                self.metrics_after.to_dict() if self.metrics_after else None
            ),
            "created_at": _datetime_to_iso(self.created_at),
            "updated_at": _datetime_to_iso(self.updated_at),
            "tags": list(self.tags),
            "extra": dict(self.extra),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SkillCandidate":
        metrics_before_raw = data.get("metrics_before")
        metrics_after_raw = data.get("metrics_after")

        return cls(
            id=data["id"],
            base_skill_name=data.get("base_skill_name"),
            spec_yaml=data.get("spec_yaml", ""),
            impl_code=data.get("impl_code", ""),
            rationale=data.get("rationale", ""),
            status=data.get("status", "proposed"),
            metrics_before=(
                SkillPerformanceStats.from_dict(metrics_before_raw)
                if metrics_before_raw
                else None
            ),
            metrics_after=(
                SkillPerformanceStats.from_dict(metrics_after_raw)
                if metrics_after_raw
                else None
            ),
            created_at=_datetime_from_iso(data.get("created_at")),
            updated_at=_datetime_from_iso(data.get("updated_at")),
            tags=list(data.get("tags") or []),
            extra=dict(data.get("extra") or {}),
        )


# ---------------------------------------------------------------------------
# Public exports
# ---------------------------------------------------------------------------

__all__ = [
    "VirtueScores",
    "EpisodeId",
    "CandidateId",
    "SkillName",
    "SkillCandidateStatus",
    "EpisodeMetadata",
    "ExperienceEpisode",
    "ExperiencePlan",
    "Experience",
    "SkillPerformanceStats",
    "SkillCandidate",
]

