# PlannerModel wrapper logic
#"src/llm_stack/planner.py"

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List

from spec.agent_loop import AgentGoal, TaskPlan, Task
from spec.skills import SkillInvocation
from spec.types import Observation

from .plan_code import PlanCodeModelImpl
from .json_utils import load_json_or_none
from .log_files import log_llm_call

logger = logging.getLogger(__name__)


@dataclass
class _GoalPlanLLMResponse:
    """
    Internal helper representation of the LLM's goal → tasks response.

    Expected JSON shape from the model:

        {
          "tasks": [
            {"id": "t1", "description": "Do X"},
            {"id": "t2", "description": "Do Y"},
            ...
          ],
          "notes": "optional explanation"
        }
    """
    tasks: List[Dict[str, Any]]
    notes: str
    raw_text: str


class HierarchicalPlanner:
    """
    Hierarchical planner façade over PlanCodeModelImpl.

    Responsibilities
    ----------------
    - Goal → TaskPlan:
        - Given an AgentGoal + world_summary, produce a TaskPlan with
          a list of Task objects (status starts as "pending").
    - Task → SkillInvocation list:
        - For a single Task, call the underlying PlanCodeModel to get
          a simple step plan, and convert those steps into SkillInvocation
          records bound to that Task.

    This keeps M8's state machine clean:

        GoalSelection → TaskPlanning → SkillResolution → Execution → Review

    while reusing the existing PlanCodeModelImpl for the low-level
    step planning.
    """

    def __init__(self, plan_model: PlanCodeModelImpl) -> None:
        """
        Parameters
        ----------
        plan_model:
            Concrete PlanCodeModelImpl used for Task → Skill planning.
            We also reuse its JSON extraction behavior and logging utilities.
        """
        self._plan_model = plan_model

    # ----------------------------------------------------------------------
    # Public API: Goal → TaskPlan
    # ----------------------------------------------------------------------

    def plan_goal(
        self,
        goal: AgentGoal,
        world_summary: Dict[str, Any],
    ) -> TaskPlan:
        """
        Break a top-level AgentGoal into a TaskPlan.

        Inputs
        ------
        goal:
            AgentGoal with id/text/phase/source.
        world_summary:
            Dict containing at least:
                - "observation": Observation
              May also include:
                - "constraints": Dict[str, Any]
                - "skills": Dict[str, Dict[str, Any]] (not strictly required here)

        Returns
        -------
        TaskPlan:
            - goal_id: goal.id
            - tasks: list[Task] with status="pending"
        """
        observation = world_summary.get("observation")
        constraints = world_summary.get("constraints", {})

        if not isinstance(observation, Observation):
            raise TypeError(
                "world_summary['observation'] must be an Observation; "
                f"got {type(observation)!r}"
            )

        prompt = self._build_goal_prompt(goal=goal, observation=observation, constraints=constraints)
        logger.debug("HierarchicalPlanner goal prompt: %s", prompt)

        raw = self._plan_model._backend.generate(  # type: ignore[attr-defined]
            prompt,
            max_tokens=self._plan_model._preset.max_tokens,          # type: ignore[attr-defined]
            temperature=self._plan_model._preset.temperature,        # type: ignore[attr-defined]
            stop=self._plan_model._preset.stop,                      # type: ignore[attr-defined]
            system_prompt=self._plan_model._preset.system_prompt,    # type: ignore[attr-defined]
        )
        logger.debug("HierarchicalPlanner goal raw output: %s", raw)

        log_llm_call(
            role="planner",
            operation="plan_goal",
            prompt=prompt,
            raw_response=raw,
            extra={
                "goal_id": goal.id,
                "goal_text": goal.text,
                "phase": goal.phase,
                "source": goal.source,
            },
        )

        # Reuse PlanCodeModelImpl's JSON extraction behavior
        candidate = self._plan_model._extract_json_object(raw)  # type: ignore[attr-defined]
        data, err = load_json_or_none(candidate, context="HierarchicalPlanner.plan_goal")

        if data is None:
            logger.error("HierarchicalPlanner goal JSON decode error: %s", err)
            # Fallback: no tasks, just a degenerate plan.
            return TaskPlan(
                goal_id=goal.id,
                tasks=[],
            )

        resp = _GoalPlanLLMResponse(
            tasks=list(data.get("tasks", []) or []),
            notes=str(data.get("notes", "") or ""),
            raw_text=raw,
        )

        tasks: List[Task] = []
        for idx, t in enumerate(resp.tasks):
            if not isinstance(t, dict):
                continue
            tid = str(t.get("id") or f"{goal.id}_task_{idx}")
            desc = str(t.get("description") or "").strip()
            if not desc:
                # Skip useless empty tasks
                continue
            tasks.append(
                Task(
                    id=tid,
                    description=desc,
                    status="pending",
                )
            )

        return TaskPlan(goal_id=goal.id, tasks=tasks)

    def _build_goal_prompt(
        self,
        goal: AgentGoal,
        observation: Observation,
        constraints: Dict[str, Any],
    ) -> str:
        """
        Build a hierarchical planning prompt for Goal → Tasks.
        """
        obs_dict = {
            "json_payload": observation.json_payload,
            "text_summary": observation.text_summary,
        }

        obs_json = json.dumps(obs_dict, indent=2)
        constraints_json = json.dumps(constraints or {}, indent=2)

        prompt = f"""
You are the high-level planner for a Minecraft GregTech New Horizons agent.

Your job is to break a single top-level goal into a small list of clear tasks.

RULES:
- Return ONLY a JSON object, with no commentary before or after it.
- The FIRST non-whitespace character MUST be '{{'.
- The LAST non-whitespace character MUST be '}}'.
- Prefer 2–6 tasks unless the goal is extremely trivial.
- Each task should be self-contained and testable, not vague fluff.

Current world observation:
{obs_json}

Top-level goal (phase: {goal.phase}, source: {goal.source}):
{goal.text}

Constraints (if any):
{constraints_json}

Return JSON in exactly this shape:
{{
  "tasks": [
    {{"id": "short_task_id", "description": "what to do, in game terms"}},
    ...
  ],
  "notes": "optional explanation of ordering, dependencies, or assumptions"
}}
"""
        return prompt.strip()

    # ----------------------------------------------------------------------
    # Public API: Task → list[SkillInvocation]
    # ----------------------------------------------------------------------

    def plan_task(
        self,
        task: Task,
        world_summary: Dict[str, Any],
    ) -> List[SkillInvocation]:
        """
        Break a single Task into a sequence of SkillInvocation objects.

        Inputs
        ------
        task:
            Task with id/description/status.
        world_summary:
            Dict containing:
                - "observation": Observation
                - "skills": Dict[str, Dict[str, Any]] (skill metadata)
              and optionally:
                - "constraints": Dict[str, Any]

        Returns
        -------
        list[SkillInvocation]:
            Ordered list of skill calls the agent should attempt to
            satisfy this task, bound via task_id.
        """
        observation = world_summary.get("observation")
        skills = world_summary.get("skills", {})
        constraints = world_summary.get("constraints", {})

        if not isinstance(observation, Observation):
            raise TypeError(
                "world_summary['observation'] must be an Observation; "
                f"got {type(observation)!r}"
            )

        # Use the existing low-level planner to get a flat step plan.
        plan_dict = self._plan_model.call_planner(
            goal=task.description,
            world_summary={
                "observation": observation,
                "skills": skills,
                "constraints": constraints,
            },
        )

        steps = plan_dict.get("steps", []) or []
        invocations: List[SkillInvocation] = []

        for step in steps:
            if not isinstance(step, dict):
                continue
            skill_name = step.get("skill")
            if not skill_name:
                continue
            params = step.get("params", {}) or {}

            expected_outcome = step.get(
                "expected_outcome",
                f"Execute skill '{skill_name}' to advance task '{task.description}'.",
            )

            invocations.append(
                SkillInvocation(
                    task_id=task.id,
                    skill_name=str(skill_name),
                    parameters=dict(params),
                    expected_outcome=str(expected_outcome),
                )
            )

        return invocations

