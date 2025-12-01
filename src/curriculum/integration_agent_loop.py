# path: src/curriculum/integration_agent_loop.py

from __future__ import annotations

from typing import Optional, Tuple

from semantics.schema import TechState
from spec.types import WorldState

from .engine import CurriculumEngine, ActiveCurriculumView


def select_goal_from_curriculum(
    engine: CurriculumEngine,
    tech_state: TechState,
    world: WorldState,
    current_goal: Optional[str] = None,
) -> Tuple[str, ActiveCurriculumView]:
    """
    Resolve a planning goal using the curriculum engine.

    This is the recommended entrypoint for M8 (AgentLoop):

      1. Call curriculum_engine.view(tech_state, world)
      2. If the agent already has a non-empty current_goal, keep it.
      3. Otherwise, choose the first curriculum goal as default.
      4. Return (goal, curr_view).

    Parameters
    ----------
    engine:
        CurriculumEngine initialized with the active CurriculumConfig.
    tech_state:
        Current TechState (M3) inferred from semantics.
    world:
        Current WorldState (M6/M7 normalized view).
    current_goal:
        Optional already-set goal string. If non-empty, it is preserved.

    Returns
    -------
    (goal, curr_view):
        goal: str
            Goal string to pass into the planner.
        curr_view: ActiveCurriculumView
            Structured curriculum view for downstream use (M4, M10, UI).
    """
    curr_view = engine.view(tech_state, world)
    phase_view = curr_view.phase_view

    # Preserve an explicit user-defined or previously-set goal
    if isinstance(current_goal, str) and current_goal.strip():
        goal = current_goal.strip()
    else:
        # Fallback: take the first active goal from the phase, or a generic one
        active_goals = phase_view.active_goals or []
        goal = active_goals[0] if active_goals else "advance GTNH tech progression"

    return goal, curr_view

