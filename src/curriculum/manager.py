# Path: src/curriculum/manager.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Any, Dict

from semantics.schema import TechState
from spec.agent_loop import AgentGoal
from learning.manager import SkillLearningManager, SkillView
from curriculum.policy import SkillPolicy, SkillUsageMode
from curriculum.strategies import CurriculumStrategy
from curriculum.engine import CurriculumEngine


@dataclass
class CurriculumManager:
    """
    Glue between:
      - SkillLearningManager (skill stats, experience memory)
      - CurriculumEngine / CurriculumStrategy (how to choose goals)
      - Planner/dispatcher (which skills are allowed)

    Responsibilities:
      1. Select next goal (via engine/strategy)
      2. Expose SkillView to planners
      3. Receive episode feedback from AgentLoop
      4. Adjust SkillPolicy when appropriate

    Pass B integration:
      - AgentLoop calls: curriculum.next_goal(tech_state=..., experience_summary=...)
      - CurriculumEngine is the primary goal authority.
      - Strategy remains optional and can be used for legacy / higher-level logic.

    Pass C integration:
      - `get_skill_view_for_goal` decides whether to allow candidate skills
        based on goal.source / goal.phase and delegates to SkillLearningManager.
    """

    # NOTE: field order is chosen to match how AgentController will likely
    # construct this: CurriculumManager(learning_manager, curriculum_engine, policy)
    learning_manager: SkillLearningManager
    engine: CurriculumEngine
    policy: SkillPolicy
    strategy: Optional[CurriculumStrategy] = None

    # ------------------------------------------------------------------
    # Goal selection
    # ------------------------------------------------------------------

    def next_goal(
        self,
        *,
        tech_state: Optional[TechState] = None,
        experience_summary: Optional[Dict[str, Any]] = None,
    ) -> Optional[AgentGoal]:
        """
        Select goal using the curriculum engine as the primary authority.

        In Pass B:
          - AgentLoop provides `tech_state` and an optional `experience_summary`.
          - CurriculumEngine decides the goal.
          - Manager updates SkillPolicy based on the chosen goal.

        If no engine is available, falls back to `strategy.select_next_goal()`.
        """
        goal: Optional[AgentGoal] = None

        if self.engine is not None and tech_state is not None:
            # Minimal world stub for CurriculumEngine.
            # Engine only expects `world.context` to be a mapping.
            class _WorldStub:
                def __init__(self) -> None:
                    self.context: Dict[str, Any] = {}

            world_stub = _WorldStub()
            goal = self.engine.next_goal(
                tech_state=tech_state,
                world=world_stub,
                experience_summary=experience_summary,
            )
        elif self.strategy is not None:
            # Legacy path: ignore tech_state/experience_summary, let strategy decide.
            goal = self.strategy.select_next_goal()
        else:
            return None

        if goal is None:
            return None

        # Keep policy usage_mode roughly in sync with the selected goal.
        self._update_skill_policy_for_goal(goal)
        return goal

    # Alias for AgentLoop compatibility
    select_goal = next_goal

    # ------------------------------------------------------------------
    # Skill usage policy
    # ------------------------------------------------------------------

    def _decide_usage_mode(self, goal: AgentGoal) -> SkillUsageMode:
        """
        Pass C heuristic:

        - Exploratory / early curriculum:
            * goal.source == "curriculum" AND phase.startswith("P1")
            * OR goal.source == "curriculum_explore"
          → ALLOW_CANDIDATES

        - Everything else:
          → STABLE_ONLY
        """
        phase = str(getattr(goal, "phase", "") or "").strip()
        source = str(getattr(goal, "source", "") or "").strip().lower()
        phase_upper = phase.upper()

        if (
            (source == "curriculum" and phase_upper.startswith("P1"))
            or source == "curriculum_explore"
        ):
            return SkillUsageMode.ALLOW_CANDIDATES

        return SkillUsageMode.STABLE_ONLY

    def _update_skill_policy_for_goal(self, goal: AgentGoal) -> None:
        """
        Update the SkillPolicy's usage_mode based on the goal.
        """
        self.policy.usage_mode = self._decide_usage_mode(goal)

    # ------------------------------------------------------------------
    # Skill view exposure
    # ------------------------------------------------------------------

    def get_skill_view_for_goal(self, goal: AgentGoal) -> SkillView:
        """
        Planner calls this to know which skills are allowed.

        Pass C behavior:
          - Decide usage_mode (stable-only vs allow candidates) from the goal.
          - Update SkillPolicy to reflect that decision.
          - Ask SkillLearningManager to build a SkillView with or without
            candidate skills accordingly.
        """
        usage_mode = self._decide_usage_mode(goal)
        self.policy.usage_mode = usage_mode

        include_candidates = usage_mode == SkillUsageMode.ALLOW_CANDIDATES

        return self.learning_manager.build_skill_view(
            include_candidates=include_candidates
        )

    # ------------------------------------------------------------------
    # AgentLoop feedback hooks
    # ------------------------------------------------------------------

    def on_episode_complete(
        self,
        *,
        goal: AgentGoal,
        episode_result: Any,
        task_plan: Any = None,
    ) -> None:
        """
        Primary feedback hook for AgentLoop.

        In v1, we simply pass the episode outcome to the strategy (if any).
        Future: adaptive policy, goal difficulty adjustments, curriculum reshaping.
        """
        if self.strategy is None:
            return

        self.strategy.on_episode_complete(
            goal=goal,
            task_plan=task_plan,
            episode_result=episode_result,
        )

    # Backwards-compatible alias for older call sites.
    def update_from_episode(
        self,
        *,
        goal: AgentGoal,
        task_plan: Any,
        episode_result: Any,
    ) -> None:
        """
        Legacy compatibility wrapper; delegates to on_episode_complete().
        """
        self.on_episode_complete(
            goal=goal,
            task_plan=task_plan,
            episode_result=episode_result,
        )

