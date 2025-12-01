# src/integration/episode_logging.py

import logging
import uuid
from typing import Dict, Any, Iterable, List, Optional

from observation.trace_schema import PlanTrace
from spec.agent_loop import AgentGoal, Task, TaskPlan
from spec.skills import SkillInvocation
from learning.schema import Experience, ExperiencePlan

LOGGER_NAME = "gtnh_agent.phase1"


def generate_correlation_id() -> str:
    """
    Generate a short correlation ID for a planning episode.

    Using UUID4 hex shortened to 8 chars is enough for logs and debugging.
    """
    return uuid.uuid4().hex[:8]


def get_base_logger() -> logging.Logger:
    """
    Get the base logger for Phase 1 integration.

    If no handlers are configured yet, attach a simple STDERR handler.
    The application can override this later with a richer logging config.
    """
    logger = logging.getLogger(LOGGER_NAME)

    if not logger.handlers:
        # Default: INFO level with simple formatting
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


class EpisodeLogger:
    """
    Structured logger for a single planning episode.

    Every log line includes the episode correlation ID, so you can trace
    context build -> planner call -> feature extraction -> scoring -> failures.
    """

    def __init__(self, episode_id: str, logger: Optional[logging.Logger] = None) -> None:
        self.episode_id = episode_id
        self._logger = logger or get_base_logger()

    # --- Internal helpers ----------------------------------------------------

    def _prefix(self, msg: str) -> str:
        """Prefix messages with the correlation ID."""
        return f"[episode={self.episode_id}] {msg}"

    # --- Context build -------------------------------------------------------

    def log_context_built(
        self,
        tech_state_name: str,
        num_skills: int,
        num_packs: int,
    ) -> None:
        """
        Log summary of the integration context for this episode.
        """
        self._logger.info(
            self._prefix(
                "context built: tech_state=%s, skills=%d, packs=%d",
            ),
            tech_state_name,
            num_skills,
            num_packs,
        )

    # --- Planner call --------------------------------------------------------

    def log_planner_request(
        self,
        goal: str,
        virtue_context_id: str,
        world_summary_hash: str,
        num_visible_skills: int,
    ) -> None:
        """
        Log high-level input info for the planner call.
        """
        self._logger.info(
            self._prefix(
                "planner request: goal=%r, virtue_context_id=%s, "
                "world_hash=%s, visible_skills=%d",
            ),
            goal,
            virtue_context_id,
            world_summary_hash,
            num_visible_skills,
        )

    def log_planner_response(
        self,
        num_plans: int,
    ) -> None:
        """
        Log how many candidate plans the planner returned.
        """
        self._logger.info(
            self._prefix("planner response: plans=%d"),
            num_plans,
        )

    # --- Feature extraction (virtue feature layer) ---------------------------

    def log_plan_features(
        self,
        plan_id: str,
        skill_names: Iterable[str],
        missing_skills: Optional[Iterable[str]] = None,
    ) -> None:
        """
        Log which skills appear in a plan summary and any unknown/missing ones.
        """
        skills_list = list(skill_names)
        self._logger.debug(
            self._prefix("plan features: plan_id=%s, skills=%s"),
            plan_id,
            skills_list,
        )

        if missing_skills:
            missing_list = list(missing_skills)
            if missing_list:
                self._logger.warning(
                    self._prefix("plan features: plan_id=%s, missing_skills=%s"),
                    plan_id,
                    missing_list,
                )

    # --- Virtue scoring ------------------------------------------------------

    def log_virtue_scores(
        self,
        plan_id: str,
        scores: Dict[str, Any],
    ) -> None:
        """
        Log per-plan virtue scores at debug level.
        """
        self._logger.debug(
            self._prefix("virtue scores: plan_id=%s, scores=%s"),
            plan_id,
            scores,
        )

    def log_selected_plan(
        self,
        plan_id: str,
        scores: Dict[str, Any],
    ) -> None:
        """
        Log which plan was selected as best and summarize its scores.
        """
        self._logger.info(
            self._prefix("selected plan: plan_id=%s, scores=%s"),
            plan_id,
            scores,
        )

    # --- Failures / anomalies -----------------------------------------------

    def log_no_plans(self, goal: str) -> None:
        """
        Planner returned no candidate plans.
        """
        self._logger.error(
            self._prefix("failure: planner returned no plans for goal=%r"),
            goal,
        )

    def log_exception(
        self,
        phase: str,
        exc: BaseException,
    ) -> None:
        """
        Log an exception that occurred during a specific phase (e.g. 'summarize_plan').
        """
        self._logger.exception(
            self._prefix("exception in phase=%s: %s"),
            phase,
            exc,
        )


# ---------------------------------------------------------------------------
# Episode → Experience builder (Q1.4)
# ---------------------------------------------------------------------------

def _snapshot_tech_state(tech_state: Any) -> Dict[str, Any]:
    """
    Best-effort snapshot of TechState for use in problem_signature.

    Intentionally mirrors the defensive pattern used elsewhere:
    - prefer .to_serializable()
    - then dict copy
    - then raw wrapper
    """
    if tech_state is None:
        return {}
    if hasattr(tech_state, "to_serializable"):
        try:
            return tech_state.to_serializable()  # type: ignore[return-value]
        except Exception:
            return {"raw": repr(tech_state)}
    if isinstance(tech_state, dict):
        return dict(tech_state)
    return {"raw": repr(tech_state)}


def _infer_success_from_trace(trace: PlanTrace) -> bool:
    """
    Heuristic success flag from PlanTrace.

    If any step has result.success == False, treat as failure.
    If we can't introspect, default to True so we don't spam "failed" episodes.
    """
    try:
        for step in trace.steps:
            result = getattr(step, "result", None)
            if result is not None and getattr(result, "success", True) is False:
                return False
        return True
    except Exception:
        return True


def build_experience_from_episode(
    episode_trace: PlanTrace,
    plan: Dict[str, Any],
    goal: AgentGoal,
    virtue_scores: Dict[str, float],
) -> Experience:
    """
    Construct a Q1.4 Experience object from a single episode.

    Inputs:
      - episode_trace: full PlanTrace from M8 (with steps + tech_state)
      - plan: flattened execution plan dict (goal_id, tasks, steps)
      - goal: AgentGoal selected at the start of the episode
      - virtue_scores: final virtue score vector for the episode

    This function does NOT:
      - know about retry loops in detail
      - embed raw EpisodeResult; higher layers can extend attempts[] later.
    """
    tech_snapshot = _snapshot_tech_state(episode_trace.tech_state)

    # Problem signature: enough to cluster "similar problems" without being insane.
    problem_signature: Dict[str, Any] = {
        "goal_id": goal.id,
        "goal_text": goal.text,
        "phase": goal.phase,
        "tech_state": tech_snapshot,
    }

    # Rebuild TaskPlan from plan["tasks"]
    tasks_raw = plan.get("tasks") or []
    tasks: List[Task] = []
    for idx, t in enumerate(tasks_raw):
        if not isinstance(t, dict):
            continue
        tasks.append(
            Task(
                id=str(t.get("id", f"{goal.id}_task_{idx}")),
                description=str(t.get("description", goal.text)),
                status=str(t.get("status", "pending")),
            )
        )

    task_plan = TaskPlan(
        goal_id=str(plan.get("goal_id", goal.id)),
        tasks=tasks,
    )

    # Rebuild SkillInvocations from plan["steps"]
    steps_raw = plan.get("steps") or []
    skill_invocations: List[SkillInvocation] = []
    for step in steps_raw:
        if not isinstance(step, dict):
            continue

        skill_name = step.get("skill") or step.get("skill_name")
        if not skill_name:
            continue

        params = step.get("params") or step.get("parameters") or {}
        expected_outcome = step.get(
            "expected_outcome",
            f"Execute skill '{skill_name}' for goal '{goal.text}'.",
        )

        skill_invocations.append(
            SkillInvocation(
                task_id=str(step.get("task_id", "")),
                skill_name=str(skill_name),
                parameters=dict(params),
                expected_outcome=str(expected_outcome),
            )
        )

    experience_plan = ExperiencePlan(
        task_plan=task_plan,
        skill_invocations=skill_invocations,
    )

    # Attempts: leave as an empty list for now.
    attempts: List[Dict[str, Any]] = []

    success_flag = _infer_success_from_trace(episode_trace)

    final_outcome: Dict[str, Any] = {
        "success": success_flag,
        "step_count": len(episode_trace.steps),
    }

    lessons = (
        f"Ran goal '{goal.text}' in phase '{goal.phase}' with "
        f"{len(tasks)} tasks and {len(skill_invocations)} skill invocations. "
        f"Success={success_flag}."
    )

    return Experience(
        problem_signature=problem_signature,
        goal=goal,
        plan=experience_plan,
        attempts=attempts,
        final_outcome=final_outcome,
        virtue_scores=dict(virtue_scores or {}),
        lessons=lessons,
    )

