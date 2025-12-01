# src/observation/schema.py
"""
Schema types for M7: observation encodings.

These are JSON-like payloads consumed by:
- PlannerModel (M2)
- CriticModel (M2)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from spec.types import Observation  # used by make_planner_observation


@dataclass
class PlannerEncoding:
    """
    Structured JSON-style encoding for the planner.

    This will typically be attached to Observation.json_payload.
    """

    tech_state: Dict[str, Any]
    agent: Dict[str, Any]
    inventory_summary: Dict[str, Any]
    machines_summary: Dict[str, Any]
    nearby_entities: List[Dict[str, Any]]
    env_summary: Dict[str, Any]
    craftable_summary: Dict[str, Any]
    context_id: str
    text_summary: str


@dataclass
class CriticEncoding:
    """
    Structured JSON-style encoding for the critic.

    Built from a PlanTrace and used to drive CriticModel evaluation / scoring.
    """

    tech_state: Dict[str, Any]
    context_id: str
    plan: Dict[str, Any]
    steps: List[Dict[str, Any]]
    planner_observation: Dict[str, Any]
    virtue_scores: Optional[Dict[str, Any]]
    text_summary: str


__all__ = [
    "PlannerEncoding",
    "CriticEncoding",
]

