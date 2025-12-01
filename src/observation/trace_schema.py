# TraceStep, PlanTrace definitions
# src/observation/trace_schema.py
"""
Trace schema for critic encoding (M7).

Defines the internal types representing an executed plan:
- TraceStep: one action + world_before/world_after + meta
- PlanTrace: full execution trace for a plan
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from spec.types import WorldState, Action, ActionResult
from semantics.schema import TechState


@dataclass
class TraceStep:
    """
    One step in the execution trace.

    world_before/world_after should be reasonably small WorldState views,
    not raw snapshots.
    """

    world_before: WorldState
    action: Action
    result: ActionResult
    world_after: WorldState
    meta: Dict[str, Any]


@dataclass
class PlanTrace:
    """
    Execution trace for a plan.

    planner_payload is usually the PlannerEncoding dict that was given to the
    planner when this plan was generated.
    """

    plan: Dict[str, Any]
    steps: List[TraceStep]
    tech_state: TechState
    planner_payload: Dict[str, Any]
    context_id: str
    virtue_scores: Dict[str, Any]


__all__ = [
    "TraceStep",
    "PlanTrace",
]

