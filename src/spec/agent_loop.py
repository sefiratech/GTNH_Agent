# AgentLoop interface / contract
# src/spec/agent_loop.py

from __future__ import annotations

from typing import Any, Mapping, Protocol

from .types import WorldState
from .llm import PlannerModel, CriticModel
from .skills import SkillRegistry
from .bot_core import BotCore


class AgentLoop(Protocol):
    """High-level control loop for the GTNH Agent.

    This ties together:
    - BotCore (M6) for acting in the world
    - SkillRegistry (M5) for available behaviors
    - PlannerModel / CriticModel (M2) for plan generation & evaluation
    """

    # Structural dependencies (not required at runtime, but part of the design)
    bot: BotCore
    skills: SkillRegistry
    planner: PlannerModel
    critic: CriticModel

    def step(self) -> None:
        """
        Perform one full agent iteration:
        - observe world via BotCore.get_world_state()
        - encode world into an Observation (M7)
        - compute or reuse a plan using PlannerModel (M2)
        - select and execute skills (M5) via BotCore.execute_action()
        - evaluate / record experience (M10)
        """
        ...

    def set_goal(self, goal: str, context: Mapping[str, Any]) -> None:
        """Set or update the current top-level goal (e.g. 'establish LV steam power')."""
        ...

    def get_status(self) -> Mapping[str, Any]:
        """
        Provide a snapshot of:
        - current goal
        - current plan
        - last action/result
        - any error state
        - any high-level metrics (ticks alive, deaths, etc.)
        """
        ...

