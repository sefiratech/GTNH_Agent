# src/planning/dispatcher.py

from __future__ import annotations

from typing import Dict, Any

from planning.adapter import PlanningAdapter
from skills.registry import SkillRegistry


class PlanningDispatcher:
    """
    Lightweight façade that filters skills according to skill_view
    and forwards planning calls to the actual adapter.
    """

    def __init__(self, skills: SkillRegistry, adapter: PlanningAdapter | None = None):
        self.skills = skills
        self.adapter = adapter or PlanningAdapter()

    def _filter_skills(self, skill_view) -> Dict[str, Any]:
        # Union of active + candidates
        allowed = set(skill_view.active_skills) | set(skill_view.candidate_skills)
        all_meta = self.skills.describe_all()

        return {name: meta for name, meta in all_meta.items() if name in allowed}

    def plan_goal(self, goal, world_summary):
        skill_view = world_summary.get("skill_view")
        if skill_view:
            filtered = self._filter_skills(skill_view)
            world_summary = dict(world_summary)
            world_summary["skills"] = filtered

        return self.adapter.plan_goal(goal, world_summary)

    def plan_task(self, task, world_summary):
        skill_view = world_summary.get("skill_view")
        if skill_view:
            filtered = self._filter_skills(skill_view)
            world_summary = dict(world_summary)
            world_summary["skills"] = filtered

        return self.adapter.plan_task(task, world_summary)

