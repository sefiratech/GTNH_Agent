# path: src/runtime/curriculum_learning_coordinator.py

from __future__ import annotations  # allow forward-annotated types

from dataclasses import dataclass  # lightweight config holder
from typing import Any, Dict, List, Optional  # basic typing primitives

from semantics.schema import TechState  # M3: structured tech progression
from spec.types import WorldState       # M7: normalized world snapshot

from curriculum.engine import CurriculumEngine, ActiveCurriculumView  # M11: curriculum logic
from learning.manager import SkillLearningManager                     # M10: learning core
from learning.replay import EpisodeReplayStore                        # M10: replay buffer API


@dataclass
class CoordinatorConfig:
    """
    Configuration for curriculum-driven skill learning.
    """
    min_episodes_per_skill: int = 5          # threshold for "enough experience"
    max_skills_per_tick: int = 3             # cap scheduled skills per call
    context_prefix: str = "curriculum"       # prefix for context_id used by learning
    include_preferred_skills: bool = True    # whether to include preferred skills after must_have


class CurriculumLearningCoordinator:
    """
    Coordinator that implements the M10 ↔ M11 data flow:

      1. (External) M8 stores episode into replay buffer (not handled here).
      2. Ask M11 for current curriculum view.
      3. Derive learning targets from skill_focus.
      4. Check M10 replay buffer for experience counts.
      5. Run M10 learning cycles for eligible skills.
      6. Return a structured summary for logging / monitoring.
    """

    def __init__(
        self,
        *,
        curriculum_engine: CurriculumEngine,
        learning_manager: SkillLearningManager,
        replay_store: EpisodeReplayStore,
        config: Optional[CoordinatorConfig] = None,
    ) -> None:
        self._curriculum_engine = curriculum_engine             # M11 engine instance
        self._learning_manager = learning_manager               # M10 manager instance
        self._replay_store = replay_store                       # M10 replay buffer instance
        self._config = config or CoordinatorConfig()            # use provided config or defaults

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_episode(
        self,
        *,
        tech_state: TechState,
        world_state: WorldState,
        episode_trace: Dict[str, Any],
        episode_meta: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Main entrypoint called after an episode is finished and stored.

        Implements steps 2–6 of the data flow description.
        """
        # --------------------------------------------------------------
        # 2. Coordinator asks M11:
        #    - curr_view = curriculum_engine.view(tech_state, world_state)
        #    - phase_id, skill_focus, etc.
        # --------------------------------------------------------------
        curr_view: ActiveCurriculumView = self._curriculum_engine.view(
            tech_state,
            world_state,
        )
        phase_view = curr_view.phase_view                        # convenience alias for phase-level info

        phase_id = phase_view.phase.id                           # current phase id
        skill_focus = phase_view.skill_focus                     # dict: { "must_have": [...], "preferred": [...] }

        # --------------------------------------------------------------
        # 3. Coordinator derives learning targets:
        #    - from skill_focus.must_have / skill_focus.preferred
        # --------------------------------------------------------------
        targets = self._derive_learning_targets(skill_focus)     # ordered list of candidate skill names

        # apply per-tick cap to avoid scheduling too many skills at once
        targets = targets[: self._config.max_skills_per_tick]

        # build curriculum-aware context identifier used by M10
        context_id = self._build_context_id(
            curriculum_id=curr_view.curriculum_id,
            phase_id=phase_id,
        )

        # --------------------------------------------------------------
        # 4. Coordinator checks M10 buffer:
        #    - experience_count = replay_store.count_episodes(skill_name, context_id)
        #    - if >= min_episodes, schedule a learning cycle
        # --------------------------------------------------------------
        ready_skills: List[str] = []
        for skill_name in targets:
            count = self._replay_store.count_episodes(
                skill_name=skill_name,
                context_id=context_id,
            )
            if count >= self._config.min_episodes_per_skill:
                ready_skills.append(skill_name)

        # --------------------------------------------------------------
        # 5. M10 runs learning cycle per chosen skill:
        #    - learning_manager.run_learning_cycle_for_goal(...)
        # --------------------------------------------------------------
        learning_results: List[Dict[str, Any]] = []
        for skill_name in ready_skills:
            # derive a loose goal substring from the skill name
            goal_substring = skill_name.replace("_", " ")

            # call into M10’s learning manager
            result = self._learning_manager.run_learning_cycle_for_goal(
                goal_substring=goal_substring,
                target_skill_name=skill_name,
                context_id=context_id,
                tech_tier=tech_state.active,
                success_only=True,
                min_episodes=self._config.min_episodes_per_skill,
            )

            # normalize to a logging-friendly record
            learning_results.append(
                {
                    "skill_name": skill_name,
                    "context_id": context_id,
                    "tech_tier": tech_state.active,
                    "result": result,
                }
            )

        # --------------------------------------------------------------
        # 6. Coordinator logs results:
        #    - caller can log / store this summary for dashboards / dev tools
        # --------------------------------------------------------------
        summary: Dict[str, Any] = {
            "curriculum_id": curr_view.curriculum_id,
            "phase_id": phase_id,
            "phase_name": phase_view.phase.name,
            "episode_meta": episode_meta,
            "episode_trace_summary": episode_trace.get("summary"),  # optional short episode summary
            "targets_considered": targets,
            "skills_ready": ready_skills,
            "learning_results": learning_results,
            "unlocked_projects": [p.id for p in curr_view.unlocked_projects],
        }

        return summary

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _derive_learning_targets(
        self,
        skill_focus: Dict[str, List[str]],
    ) -> List[str]:
        """
        Derive an ordered list of skills to *try* to train based on skill_focus.

        Priority:
          1. must_have (in listed order)
          2. preferred (if config.include_preferred_skills is True)
        """
        must_have = list(skill_focus.get("must_have", []))
        preferred = list(skill_focus.get("preferred", []))

        if not self._config.include_preferred_skills:
            return must_have

        seen = set(must_have)
        ordered: List[str] = list(must_have)

        for name in preferred:
            if name not in seen:
                ordered.append(name)
                seen.add(name)

        return ordered

    def _build_context_id(self, *, curriculum_id: str, phase_id: str) -> str:
        """
        Build a context_id of the form:

          "{prefix}:{curriculum_id}:{phase_id}"
        """
        return f"{self._config.context_prefix}:{curriculum_id}:{phase_id}"

