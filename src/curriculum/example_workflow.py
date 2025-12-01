# path: src/curriculum/example_workflow.py

from __future__ import annotations

from typing import Any, Callable, Dict, List, Tuple

from semantics.schema import TechState          # M3: tech inference
from spec.types import WorldState               # M6/M7: normalized world view

from .engine import CurriculumEngine, ActiveCurriculumView


def _select_goal_from_curriculum(
    curriculum_engine: CurriculumEngine,
    tech_state: TechState,
    world_state: WorldState,
    current_goal: str | None = None,
) -> Tuple[str, ActiveCurriculumView]:
    """
    Step 1–4 (curriculum side):

      1. M8 observes → infers TechState           (done outside this helper)
      2. M11 resolves phase                       (curriculum_engine.view)
      3. Outputs goals, virtue overrides, skills
      4. M8 uses them to select a planning goal

    If `current_goal` is non-empty, it is preserved.
    Otherwise the first active curriculum goal is chosen, or a generic fallback.
    """
    curr_view = curriculum_engine.view(tech_state, world_state)
    phase_view = curr_view.phase_view

    if isinstance(current_goal, str) and current_goal.strip():
        goal = current_goal.strip()
    else:
        active_goals = phase_view.active_goals or []
        goal = active_goals[0] if active_goals else "advance GTNH tech progression"

    return goal, curr_view


def _select_learning_targets_from_skill_focus(
    skill_focus: Dict[str, List[str]],
    default_targets: List[str] | None = None,
) -> List[str]:
    """
    Derive an ordered list of skills to train from curriculum skill_focus.

    Priority:
      1. must_have (listed order)
      2. preferred (listed order)
      3. remaining defaults (if provided)
    """
    must_have = list(skill_focus.get("must_have", []))
    preferred = list(skill_focus.get("preferred", []))

    seen = set(must_have) | set(preferred)
    remaining: List[str] = []

    if default_targets is not None:
        for name in default_targets:
            if name not in seen:
                remaining.append(name)

    return must_have + preferred + remaining


def run_curriculum_example_step(
    *,
    curriculum_engine: CurriculumEngine,
    tech_state: TechState,
    world_state: WorldState,
    planner: Callable[[str, WorldState, Dict[str, Any]], Dict[str, Any]],
    episode_buffer: List[Dict[str, Any]],
    skill_learning_manager: Any | None = None,
    current_goal: str | None = None,
    default_learning_targets: List[str] | None = None,
    min_episodes_per_skill: int = 5,
) -> Dict[str, Any]:
    """
    Example end-to-end step showing how M8, M10, and M11 interact.

    This function is intentionally high-level and side-effectful:
      - Chooses a goal using curriculum (M11)
      - Calls a planner (M8-style) with that goal
      - Appends the resulting episode to an episode buffer (M8/M10)
      - Optionally uses skill_focus to schedule learning (M10)

    It corresponds roughly to the doc's “Example Workflow”:

      1. M8 observes → infers TechState.          (tech_state, world_state are inputs)
      2. M11 resolves phase via CurriculumEngine.view(...)
      3. Outputs:
           - goals
           - virtue overrides
           - skill-focus
      4. M8 uses these to:
           - bias planner
           - select goal
      5. Agent acts → generates episode (planner)
      6. Episode stored (episode_buffer)
      7. Curriculum may schedule learning cycle if phase rules say so.

    Parameters
    ----------
    curriculum_engine:
        Initialized CurriculumEngine for the active curriculum.
    tech_state:
        Current TechState (from M3).
    world_state:
        Current WorldState (from observation stack M6/M7).
    planner:
        Callable implementing the planning+acting step:

            episode = planner(goal: str, world_state: WorldState, hints: Dict[str, Any])

        It should return a dict-like episode object (logs, actions, outcome).
    episode_buffer:
        Mutable list that will be appended with the generated episode.
        In a real system this would be your replay buffer / trace store.
    skill_learning_manager:
        Optional learning manager (M10). If provided, this function will
        trigger curriculum-driven learning cycles.
        Expected to expose something like:

            run_learning_cycle_for_goal(
                goal_substring: str,
                target_skill_name: str,
                context_id: str,
                tech_tier: str,
                success_only: bool,
                min_episodes: int,
            ) -> Dict[str, Any]
    current_goal:
        Existing goal string for the agent. If non-empty, it is preserved.
    default_learning_targets:
        Optional list of fallback skill names for learning.
    min_episodes_per_skill:
        Minimum number of episodes to require before triggering
        a learning cycle per skill.

    Returns
    -------
    Dict[str, Any]
        A summary dict with:
          - "goal"
          - "curriculum_view"
          - "episode"
          - "learning_results" (if any)
    """
    # ------------------------------------------------------------------
    # 1–4: Curriculum → goal & hints
    # ------------------------------------------------------------------
    goal, curr_view = _select_goal_from_curriculum(
        curriculum_engine=curriculum_engine,
        tech_state=tech_state,
        world_state=world_state,
        current_goal=current_goal,
    )
    phase_view = curr_view.phase_view

    # Build hints for planner: virtue overrides + skill-focus etc.
    planner_hints: Dict[str, Any] = {
        "curriculum_id": curr_view.curriculum_id,
        "phase_id": phase_view.phase.id,
        "phase_name": phase_view.phase.name,
        "virtue_overrides": phase_view.virtue_overrides,
        "skill_focus": phase_view.skill_focus,
        "unlocked_projects": [
            {"id": p.id, "name": p.name} for p in curr_view.unlocked_projects
        ],
    }

    # ------------------------------------------------------------------
    # 5: Agent acts → generates episode
    # ------------------------------------------------------------------
    episode = planner(goal, world_state, planner_hints)

    # ------------------------------------------------------------------
    # 6: Episode stored
    # ------------------------------------------------------------------
    episode_buffer.append(episode)

    # ------------------------------------------------------------------
    # 7: Curriculum-driven learning (optional)
    # ------------------------------------------------------------------
    learning_results: List[Dict[str, Any]] = []

    if skill_learning_manager is not None:
        # Turn skill_focus into a learning target list
        targets = _select_learning_targets_from_skill_focus(
            skill_focus=phase_view.skill_focus,
            default_targets=default_learning_targets,
        )

        context_id = f"curriculum:{curr_view.curriculum_id}:{phase_view.phase.id}"
        tech_tier = getattr(tech_state, "active", "unknown")

        for skill_name in targets:
            # simple heuristic: use skill_name as goal substring
            goal_substring = skill_name.replace("_", " ")
            result = skill_learning_manager.run_learning_cycle_for_goal(
                goal_substring=goal_substring,
                target_skill_name=skill_name,
                context_id=context_id,
                tech_tier=tech_tier,
                success_only=True,
                min_episodes=min_episodes_per_skill,
            )
            learning_results.append(
                {
                    "skill_name": skill_name,
                    "result": result,
                }
            )

    return {
        "goal": goal,
        "curriculum_view": curr_view,
        "episode": episode,
        "learning_results": learning_results,
    }

