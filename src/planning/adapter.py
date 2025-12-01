# src/planning/adapter.py

from __future__ import annotations

"""
This is the minimal façade between dispatcher and LLM planner.

You can expand this later with different planner backends.
"""

class PlanningAdapter:
    def plan_goal(self, goal, world_summary):
        # Placeholder: call your LLM here.
        # Should return TaskPlan according to spec.llm.
        raise NotImplementedError("plan_goal not implemented")

    def plan_task(self, task, world_summary):
        # Placeholder for skill-resolution planning.
        raise NotImplementedError("plan_task not implemented")

