# path: src/runtime/curriculum_learning_triggers.py

from __future__ import annotations  # allow forward references in type hints

from dataclasses import dataclass, field  # for simple config / state holders
from typing import Any, Dict, List, Set  # basic typing primitives

from semantics.schema import TechState  # M3: structured tech progression state
from spec.types import WorldState       # M7: normalized world snapshot

from curriculum.engine import CurriculumEngine  # M11: curriculum engine
from runtime.curriculum_learning_coordinator import (  # coordinator glues M10 ↔ M11
    CurriculumLearningCoordinator,
)
from curriculum.engine import ActiveCurriculumView  # for type hints only


@dataclass
class TriggerConfig:
    """
    Configuration for when to trigger curriculum-driven learning.

    This directly encodes the "Event / Trigger Model" for M10 ← M11.
    """
    episodes_per_phase: int = 5     # periodic trigger: every N episodes in the same phase
    enable_phase_completion: bool = True   # trigger burst on phase completion
    enable_project_unlock: bool = True     # trigger burst when a long-horizon project unlocks


@dataclass
class TriggerState:
    """
    Internal state for trigger decisions across episodes.
    """
    episodes_in_phase: Dict[str, int] = field(default_factory=dict)   # phase_id -> episode count
    phase_complete: Dict[str, bool] = field(default_factory=dict)     # phase_id -> last known is_complete flag
    seen_projects: Set[str] = field(default_factory=set)              # set of project ids already seen as unlocked


class CurriculumLearningTriggerManager:
    """
    Wraps CurriculumLearningCoordinator with an event / trigger model.

    Responsibilities:
      - Track how many episodes have run in each phase.
      - Detect phase completion transitions.
      - Detect new long-horizon project unlocks.
      - Decide when to call the underlying coordinator to actually run M10 learning.
    """

    def __init__(
        self,
        *,
        curriculum_engine: CurriculumEngine,
        coordinator: CurriculumLearningCoordinator,
        config: TriggerConfig | None = None,
    ) -> None:
        self._curriculum_engine = curriculum_engine               # M11 engine for phase / project view
        self._coordinator = coordinator                           # orchestration for M10 learning cycles
        self._config = config or TriggerConfig()                  # trigger configuration
        self._state = TriggerState()                              # mutable trigger state

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def handle_episode(
        self,
        *,
        tech_state: TechState,
        world_state: WorldState,
        episode_trace: Dict[str, Any],
        episode_meta: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Main entrypoint called after an episode ends.

        The AgentLoop (M8) should:
          1. Store the episode into the replay buffer (M10).
          2. Call this method with tech_state, world_state, and episode metadata.

        This method:
          - Queries M11 for current phase / projects.
          - Updates trigger state (episodes_in_phase, completion flags, project set).
          - Decides whether learning should be triggered.
          - If yes, calls the coordinator to perform learning and returns a rich summary.
          - If no, returns a "no-op" summary for logging.
        """
        # Get the current curriculum view from M11
        curr_view: ActiveCurriculumView = self._curriculum_engine.view(
            tech_state,
            world_state,
        )
        phase_view = curr_view.phase_view                         # convenience alias
        phase_id = phase_view.phase.id                            # current phase identifier
        is_complete = phase_view.is_complete                      # boolean completion flag
        unlocked_projects = [p.id for p in curr_view.unlocked_projects]  # list of unlocked project ids

        # Update counters for episodes-per-phase trigger
        episode_count = self._state.episodes_in_phase.get(phase_id, 0) + 1
        self._state.episodes_in_phase[phase_id] = episode_count

        # Determine whether each trigger type should fire
        triggers: List[str] = []                                  # collect which triggers fired this tick

        # 1) Per N episodes in the current phase
        if episode_count % self._config.episodes_per_phase == 0:
            triggers.append("periodic_phase_episode")

        # 2) On phase completion (rising edge only)
        if self._config.enable_phase_completion:
            was_complete = self._state.phase_complete.get(phase_id, False)
            if is_complete and not was_complete:
                triggers.append("phase_completion")
            self._state.phase_complete[phase_id] = is_complete

        # 3) On project unlock (newly unlocked projects)
        new_projects = set(unlocked_projects) - self._state.seen_projects
        if self._config.enable_project_unlock and new_projects:
            triggers.append("project_unlock")
            self._state.seen_projects.update(new_projects)

        # If nothing fired, return a no-op summary for monitoring / logs
        if not triggers:
            return {
                "triggered": False,
                "triggers": [],
                "curriculum_id": curr_view.curriculum_id,
                "phase_id": phase_id,
                "phase_name": phase_view.phase.name,
                "phase_is_complete": is_complete,
                "episode_meta": episode_meta,
                "unlocked_projects": unlocked_projects,
                "episodes_in_phase": episode_count,
                "coordinator_summary": None,
            }

        # At least one trigger fired: delegate to the coordinator to actually run learning
        coordinator_summary = self._coordinator.process_episode(
            tech_state=tech_state,
            world_state=world_state,
            episode_trace=episode_trace,
            episode_meta=episode_meta,
        )

        # Wrap everything into a unified summary
        return {
            "triggered": True,
            "triggers": triggers,
            "curriculum_id": curr_view.curriculum_id,
            "phase_id": phase_id,
            "phase_name": phase_view.phase.name,
            "phase_is_complete": is_complete,
            "episode_meta": episode_meta,
            "unlocked_projects": unlocked_projects,
            "episodes_in_phase": episode_count,
            "new_projects": list(new_projects),
            "coordinator_summary": coordinator_summary,
        }

