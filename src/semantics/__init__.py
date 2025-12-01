# semantics package
# src/semantics/__init__.py

"""
Top-level semantics façade for GTNH_Agent.

This module provides a small, stable surface for consumers:

- get_tech_state(world)  -> TechState
- suggest_next_targets(tech_state, graph) -> List[TechTarget]
- SemanticsDB            -> items/blocks/recipes lookup
- TechGraph              -> tech progression DAG
- WorldModel             -> lightweight predictive world model
"""

from __future__ import annotations

from typing import List

from spec.types import WorldState

from .schema import TechState, TechTarget, BlockInfo, ItemInfo  # type hints / exports
from .tech_state import (
    TechGraph,
    infer_tech_state_from_world,
    suggest_next_targets,
)
from .loader import SemanticsDB
from world.world_model import WorldModel


def get_tech_state(world: WorldState) -> TechState:
    """
    Convenience wrapper used by higher-level modules (M3, M8, M11).

    Given a WorldState, infer the current TechState using the global TechGraph.
    """
    graph = TechGraph()
    return infer_tech_state_from_world(world=world, graph=graph)


__all__ = [
    "TechState",
    "TechTarget",
    "BlockInfo",
    "ItemInfo",
    "TechGraph",
    "SemanticsDB",
    "WorldModel",
    "get_tech_state",
    "suggest_next_targets",
]

