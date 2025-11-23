# spec package
"""
Core interface specifications for the GTNH Agent.

This package defines stable contracts for:
- WorldState / Observation / Action types
- BotCore interface (the "body")
- Skill and SkillRegistry interfaces
- Planner / Code / Critic LLM interfaces
- AgentLoop interface (control logic)
- Experience recording / skill learning hooks

Everything else in the project is expected to import these abstractions,
not redefine them.
"""

from .types import WorldState, Observation, Action, ActionResult
from .bot_core import BotCore
from .skills import Skill, SkillRegistry
from .llm import PlannerModel, CodeModel, CriticModel
from .agent_loop import AgentLoop
from .experience import ExperienceRecorder, SkillLearner

__all__ = [
    "WorldState",
    "Observation",
    "Action",
    "ActionResult",
    "BotCore",
    "Skill",
    "SkillRegistry",
    "PlannerModel",
    "CodeModel",
    "CriticModel",
    "AgentLoop",
    "ExperienceRecorder",
    "SkillLearner",
]

