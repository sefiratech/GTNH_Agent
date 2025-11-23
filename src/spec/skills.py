# SkillImplBase and skill-related interfaces
# src/spec/skills.py

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Protocol

from .types import WorldState, Action


class Skill(Protocol):
    """A reusable macro/behavior unit."""

    @property
    def name(self) -> str:
        """Unique name used in plans and registry."""
        ...

    def describe(self) -> Dict[str, Any]:
        """
        Return metadata for planners and UIs, for example:
        {
            "description": "Chop down a nearby tree and collect logs.",
            "params": {
                "min_logs": {"type": "int", "default": 8},
            },
            "preconditions": ["has_axe"],
            "effects": ["inventory.logs >= min_logs"],
        }
        """
        ...

    def suggest_actions(
        self,
        world: WorldState,
        params: Mapping[str, Any],
    ) -> List[Action]:
        """
        Given current world state and skill parameters,
        propose a sequence of Actions for BotCore to execute.

        This is where high-level intentions turn into concrete steps like:
        - move to position
        - break block
        - pick up item
        """
        ...


class SkillRegistry(Protocol):
    """Central registry of available skills."""

    def list_skills(self) -> List[str]:
        """Return all skill names."""
        ...

    def get_skill(self, name: str) -> Skill:
        """Fetch a skill by name; should raise if the skill is unknown."""
        ...

    def describe_all(self) -> Dict[str, Dict[str, Any]]:
        """
        Return metadata for all skills for use by planners and LLM prompts.

        Example shape:
        {
          "chop_tree": {...},
          "craft_planks": {...},
        }
        """
        ...

