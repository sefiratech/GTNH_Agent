# PlannerModel, CodeModel, CriticModel interfaces
# src/spec/llm.py

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Protocol

from .types import Observation


class PlannerModel(Protocol):
    """Generates high-level plans from observations and goals."""

    def plan(
        self,
        observation: Observation,
        goal: str,
        skill_descriptions: Mapping[str, Dict[str, Any]],
        constraints: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """
        Return a structured plan, e.g.:

        {
          "steps": [
            {"skill": "chop_tree", "params": {...}},
            {"skill": "craft_planks", "params": {...}},
          ]
        }

        Concrete implementations (M2) can wrap this into a richer Plan
        dataclass if desired, but the contract here is JSON-like.
        """
        ...


class CodeModel(Protocol):
    """Generates or refines skill implementations or scripts."""

    def propose_skill_implementation(
        self,
        skill_spec: Mapping[str, Any],
        examples: List[Mapping[str, Any]],
    ) -> str:
        """
        Given a skill spec and example traces, return new/updated code as a string.

        M10 will own persistence / evaluation / rollout of this code.
        """
        ...


class CriticModel(Protocol):
    """Evaluates and reflects on plans or skill performance."""

    def evaluate_plan(
        self,
        observation: Observation,
        plan: Mapping[str, Any],
        virtue_scores: Mapping[str, float],
    ) -> Dict[str, Any]:
        """
        Return critique & suggested adjustments, e.g.:

        {
          "score": 0.82,
          "issues": ["too risky near lava", "ignores low food"],
          "suggested_changes": [...]
        }
        """
        ...

