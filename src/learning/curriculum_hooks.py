# path: src/learning/curriculum_hooks.py

from __future__ import annotations

from typing import Iterable, List, Dict, Any, Optional

from semantics.schema import TechState
from spec.types import WorldState

from curriculum.engine import ActiveCurriculumView
from .manager import SkillLearningManager


def select_learning_targets_from_skill_focus(
    skill_focus: Dict[str, List[str]],
    default_targets: Optional[Iterable[str]] = None,
) -> List[str]:
    """
    Compute an ordered list of skills to prioritize for learning,
    based on curriculum skill_focus.

    Priority order:
      1. must_have (in order listed)
      2. preferred (in order listed)
      3. any remaining defaults, if provided

    Parameters
    ----------
    skill_focus:
        Dict with keys:
          - "must_have": List[str]
          - "preferred": List[str]
    default_targets:
        Optional iterable of fallback skill names.

    Returns
    -------
    List[str]
        Ordered list of skill names to focus learning on.
    """
    must_have = list(skill_focus.get("must_have", []))
    preferred = list(skill_focus.get("preferred", []))
    remaining: List[str] = []

    seen = set(must_have) | set(preferred)

    if default_targets is not None:
        for name in default_targets:
            if name not in seen:
                remaining.append(name)

    return must_have + preferred + remaining


def schedule_learning_from_curriculum(
    manager: SkillLearningManager,
    curriculum_view: ActiveCurriculumView,
    tech_state: TechState,
    world: WorldState,
    *,
    context_prefix: str = "",
    min_episodes_per_skill: int = 5,
) -> List[Dict[str, Any]]:
    """
    High-level helper to schedule learning cycles based on the active curriculum.

    This does NOT auto-deploy new skills. It simply:
      - reads phase skill_focus
      - chooses learning targets
      - runs learning cycles via SkillLearningManager
      - returns a list of result dicts for monitoring / UI

    Parameters
    ----------
    manager:
        The SkillLearningManager instance (M10).
    curriculum_view:
        ActiveCurriculumView from CurriculumEngine.view(...).
    tech_state:
        Current TechState (M3).
    world:
        Current WorldState (M6/M7).
    context_prefix:
        Optional prefix for context_id (e.g. "curriculum:default_speedrun").
    min_episodes_per_skill:
        Minimum number of successful episodes to require before attempting
        a learning cycle for that skill.

    Returns
    -------
    List[Dict[str, Any]]
        One entry per attempted learning cycle, each containing:
          - "skill_name"
          - "result" (SkillLearningManager result)
    """
    phase_view = curriculum_view.phase_view
    skill_focus = phase_view.skill_focus

    # Use phase id as context; allow prefix override for nicer separation
    base_context = phase_view.phase.id
    if context_prefix:
        context_id = f"{context_prefix}:{base_context}"
    else:
        context_id = base_context

    # Choose skills to focus on for learning
    targets = select_learning_targets_from_skill_focus(skill_focus)

    results: List[Dict[str, Any]] = []

    for skill_name in targets:
        # Heuristic: derive a loose goal substring from the skill name
        # You can make this more structured later.
        goal_substring = skill_name.replace("_", " ")

        result = manager.run_learning_cycle_for_goal(
            goal_substring=goal_substring,
            target_skill_name=skill_name,
            context_id=context_id,
            tech_tier=tech_state.active,
            success_only=True,
            min_episodes=min_episodes_per_skill,
        )

        results.append(
            {
                "skill_name": skill_name,
                "result": result,
            }
        )

    return results
