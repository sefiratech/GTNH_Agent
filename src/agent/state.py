#"src/agent/state.py"

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional

from spec.agent_loop import AgentGoal, TaskPlan, Task
from spec.skills import SkillInvocation
from observation.trace_schema import PlanTrace


class AgentLoopPhase(Enum):
    """
    High-level phases for the M8 agent loop state machine.

    The loop should move through these in order during a typical episode:

        GoalSelection -> TaskPlanning -> SkillResolution -> Execution -> Review

    Implementations are free to:
    - skip phases (e.g. reuse an existing TaskPlan)
    - loop within phases (e.g. multiple Execution passes)
    as long as they keep this as the canonical set of phase labels.
    """

    GOAL_SELECTION = auto()
    TASK_PLANNING = auto()
    SKILL_RESOLUTION = auto()
    EXECUTION = auto()
    REVIEW = auto()


@dataclass
class AgentLoopState:
    """
    Mutable state snapshot for a single agent episode.

    This structure is intentionally simple and explicit so that:
    - AgentLoop.run_episode can be written as a clear state machine.
    - Tests and tools can introspect "where the agent is" and why.
    - Future features (e.g. partial recovery, interactive debugging)
      can key off phase + fields without reverse-engineering traces.

    Fields
    ------
    episode_id:
        Optional external identifier for the episode (e.g. log index).

    phase:
        Current phase of the loop (see AgentLoopPhase).

    goal:
        The active AgentGoal for this episode, if one has been selected.

    task_plan:
        The TaskPlan corresponding to the current goal, if any.

    current_task_index:
        Index into task_plan.tasks for the task being resolved/executed.
        -1 means "no task selected yet".

    current_task:
        Convenience reference to the current Task, if any. This SHOULD
        be kept in sync with task_plan.tasks[current_task_index].

    skill_invocations:
        Flat list of SkillInvocation objects produced by SkillResolution
        for the current TaskPlan (ordered). Implementations may choose to
        regenerate this per-task instead of storing all at once.

    current_skill_index:
        Index into skill_invocations for the skill being executed.
        -1 means "no skill selected yet".

    trace:
        PlanTrace for this episode, including:
        - planner plan(s)
        - execution TraceStep list
        - tech_state
        - virtue_scores

    meta:
        Arbitrary metadata dict for additional bookkeeping:
        - "retries_used"
        - "planner_calls"
        - "notes"
        etc.
    """

    episode_id: Optional[int] = None
    phase: AgentLoopPhase = AgentLoopPhase.GOAL_SELECTION

    goal: Optional[AgentGoal] = None
    task_plan: Optional[TaskPlan] = None
    current_task_index: int = -1
    current_task: Optional[Task] = None

    skill_invocations: List[SkillInvocation] = field(default_factory=list)
    current_skill_index: int = -1

    trace: Optional[PlanTrace] = None

    meta: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def has_goal(self) -> bool:
        """Return True if a goal has been selected for this episode."""
        return self.goal is not None

    def has_task_plan(self) -> bool:
        """Return True if a TaskPlan has been produced for the current goal."""
        return self.task_plan is not None and bool(self.task_plan.tasks)

    def has_pending_tasks(self) -> bool:
        """
        Return True if there is at least one task remaining that is not
        in a terminal state ("done" / "failed").
        """
        if not self.task_plan:
            return False
        return any(t.status not in ("done", "failed") for t in self.task_plan.tasks)

    def has_pending_skills(self) -> bool:
        """Return True if there are remaining SkillInvocation entries to execute."""
        return self.current_skill_index + 1 < len(self.skill_invocations)

    def advance_task_index(self) -> None:
        """
        Move to the next task in task_plan.tasks.

        If no more tasks exist, current_task is set to None and
        current_task_index is set to -1.
        """
        if not self.task_plan or not self.task_plan.tasks:
            self.current_task_index = -1
            self.current_task = None
            return

        self.current_task_index += 1
        if 0 <= self.current_task_index < len(self.task_plan.tasks):
            self.current_task = self.task_plan.tasks[self.current_task_index]
        else:
            self.current_task_index = -1
            self.current_task = None

    def reset_skills_for_task(self, invocations: List[SkillInvocation]) -> None:
        """
        Replace the current skill_invocations list with a new sequence for
        the active task, and reset the skill index.
        """
        self.skill_invocations = list(invocations)
        self.current_skill_index = -1

    def advance_skill_index(self) -> None:
        """
        Move to the next skill invocation.

        If there are no more skills, current_skill_index will end up
        >= len(skill_invocations) and has_pending_skills() will be False.
        """
        self.current_skill_index += 1

    def is_in_terminal_phase(self) -> bool:
        """
        Return True if the loop is in its final REVIEW phase.
        """
        return self.phase == AgentLoopPhase.REVIEW


# ----------------------------------------------------------------------
# Factory helpers
# ----------------------------------------------------------------------

def initial_state(episode_id: Optional[int] = None) -> AgentLoopState:
    """
    Construct a fresh AgentLoopState for a new episode.

    The loop should start in GOAL_SELECTION phase with no goal, plan,
    or trace yet attached.
    """
    return AgentLoopState(
        episode_id=episode_id,
        phase=AgentLoopPhase.GOAL_SELECTION,
    )

