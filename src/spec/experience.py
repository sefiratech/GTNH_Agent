# src/spec/experience.py

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Protocol

from .types import WorldState, Action, ActionResult


class ExperienceRecorder(Protocol):
    """Collects data about episodes for future learning."""

    def record_step(
        self,
        world_before: WorldState,
        action: Action,
        result: ActionResult,
        world_after: WorldState,
        meta: Mapping[str, Any],
    ) -> None:
        """Save a single transition."""
        ...

    def flush(self) -> None:
        """Force writing any buffered experience to disk or durable storage."""
        ...


class SkillLearner(Protocol):
    """Learns new skills or refines existing ones from recorded experience."""

    def propose_skill_updates(self) -> List[Dict[str, Any]]:
        """
        Analyze recorded experience and suggest:
        - new skills
        - updated skill implementations
        - deprecations

        Returned items are JSON-like dicts; a later module (M10)
        decides how to turn them into actual code / registry updates.
        """
        ...

