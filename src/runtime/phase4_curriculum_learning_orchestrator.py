# path: src/runtime/phase4_curriculum_learning_orchestrator.py

from __future__ import annotations  # allow forward-annotated types

from dataclasses import dataclass  # lightweight config holder
from typing import Any, Dict, List, Optional  # basic typing primitives

from semantics.schema import TechState  # M3: structured tech progression
from spec.types import WorldState       # M7: normalized world snapshot

from curriculum.engine import CurriculumEngine, ActiveCurriculumView  # M11: curriculum logic


@dataclass
class LearningScheduleConfig:
    """
    Configuration for curriculum-driven learning scheduling.
    """
    min_episodes_per_skill: int = 5          # how many episodes before we try to train a skill
    max_skills_per_tick: int = 3             # cap skills per integration step
    always_include_preferred: bool = True    # whether to include preferred skills after must_have
    context_prefix: str = "curriculum"       # prefix for learning context IDs


class CurriculumLearningOrchestrator:
    """
    Glue between M11 (curriculum) and M10 (skill learning).

    Responsibilities:
      - Query M11 for current phase + skill_focus.
      - Inspect replay buffer (M10) for experience counts.
      - Decide which skills to train.
      - Trigger M10 learning cycles.
      - Return a structured summary for logging / monitoring.

    Note:
      `learning_manager` and `replay_store` are duck-typed:
        - replay_store must implement:
              count_episodes(skill_name: str, context_id: str) -> int
        - learning_manager must implement:
              run_learning_cycle_for_goal(...kwargs...) -> dict
    """

    def __init__(
        self,
        *,
        curriculum_engine: CurriculumEngine,
        learning_manager: Any,
        replay_store: Any,
        config: Optional[LearningScheduleConfig] = None,
    ) -> None:
        """Initialize orchestrator with its dependencies."""
        self._curriculum_engine = curriculum_engine          # M11 engine instance
        self._learning_manager = learning_manager            # M10 manager instance (duck-typed)
        self._replay_store = replay_store                    # M10 replay buffer instance (duck-typed)
        self._config = config or LearningScheduleConfig()    # use provided config or defaults

    # ------------------------------------------------------------------
    # Core entrypoint
    # ------------------------------------------------------------------

    def run_after_episode(
        self,
        *,
        tech_state: TechState,
        world: WorldState,
        episode_meta: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Main integration entrypoint called after an episode completes.

        Implements the M10 ← M11 data flow:

          1. M8 has already stored the episode into the replay buffer.
          2. Query curriculum for current phase + skill-focus.
          3. Derive learning targets from skill_focus.
          4. Check replay counts per skill.
          5. Trigger learning cycles for eligible skills.
          6. Return a summary record for logging / monitoring.
        """
        # --------------------------------------------------------------
        # 2. Ask M11 for current curriculum view
        # --------------------------------------------------------------
        curr_view: ActiveCurriculumView = self._curriculum_engine.view(
            tech_state,
            world,
        )
        phase_view = curr_view.phase_view                        # convenience alias for phase-level info

        phase_id = phase_view.phase.id                           # current phase id
        skill_focus = phase_view.skill_focus                     # dict: { "must_have": [...], "preferred": [...] }

        # --------------------------------------------------------------
        # 3. Derive learning targets from skill_focus
        # --------------------------------------------------------------
        targets = self._select_learning_targets(skill_focus)     # ordered candidate skill names

        # Enforce per-tick cap to avoid over-scheduling
        targets = targets[: self._config.max_skills_per_tick]

        # Build curriculum-aware context identifier used by M10
        context_id = self._build_context_id(
            curriculum_id=curr_view.curriculum_id,
            phase_id=phase_id,
        )

        # --------------------------------------------------------------
        # 4. Check replay buffer for experience counts
        # --------------------------------------------------------------
        ready_skills: List[str] = []
        for skill_name in targets:
            count = self._replay_store.count_episodes(           # duck-typed call into replay_store
                skill_name=skill_name,
                context_id=context_id,
            )
            if count >= self._config.min_episodes_per_skill:
                ready_skills.append(skill_name)

        # --------------------------------------------------------------
        # 5. Run learning cycles for eligible skills
        # --------------------------------------------------------------
        learning_results: List[Dict[str, Any]] = []
        for skill_name in ready_skills:
            # simple heuristic: use skill name as a fuzzy natural-language goal
            goal_substring = skill_name.replace("_", " ")

            # delegate to M10 learning manager (duck-typed)
            result = self._learning_manager.run_learning_cycle_for_goal(
                goal_substring=goal_substring,
                target_skill_name=skill_name,
                context_id=context_id,
                tech_tier=tech_state.active,
                success_only=True,
                min_episodes=self._config.min_episodes_per_skill,
            )

            learning_results.append(
                {
                    "skill_name": skill_name,
                    "context_id": context_id,
                    "tech_tier": tech_state.active,
                    "result": result,
                }
            )

        # --------------------------------------------------------------
        # 6. Produce summary payload for logging / debug
        # --------------------------------------------------------------
        summary: Dict[str, Any] = {
            "curriculum_id": curr_view.curriculum_id,
            "phase_id": phase_id,
            "phase_name": phase_view.phase.name,
            "episode_meta": episode_meta,
            "targets_considered": targets,
            "skills_trained": [r["skill_name"] for r in learning_results],
            "learning_results": learning_results,
            "unlocked_projects": [p.id for p in curr_view.unlocked_projects],
        }

        return summary

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _select_learning_targets(
        self,
        skill_focus: Dict[str, List[str]],
    ) -> List[str]:
        """
        Turn skill_focus into an ordered list of skills to potentially train.

        Priority:
          1. must_have
          2. preferred (optional, controlled by config)
        """
        must_have = list(skill_focus.get("must_have", []))            # copy to avoid mutating input
        preferred = list(skill_focus.get("preferred", []))            # copy to avoid mutating input

        if not self._config.always_include_preferred:                 # config can disable preferred skills
            return must_have

        seen = set(must_have)                                         # track which skills we've already added
        ordered: List[str] = []                                       # final ordered list

        ordered.extend(must_have)                                     # must_have always come first
        for name in preferred:                                        # then preferred (deduped)
            if name not in seen:
                ordered.append(name)
                seen.add(name)

        return ordered

    def _build_context_id(
        self,
        *,
        curriculum_id: str,
        phase_id: str,
    ) -> str:
        """
        Build a context_id string of the form:

            "{prefix}:{curriculum_id}:{phase_id}"
        """
        prefix = self._config.context_prefix                          # base prefix from config
        return f"{prefix}:{curriculum_id}:{phase_id}"                 # formatted context_id string

