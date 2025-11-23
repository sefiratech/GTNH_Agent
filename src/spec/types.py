# core shared types: WorldState, Action, ActionResult, Observation
# src/spec/types.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional


@dataclass
class WorldState:
    """Semantic snapshot of the world relevant to decision-making.

    This is not raw packets. It's a structured view that BotCore (M6) / world
    semantics (M3) produce for the AgentLoop and skills.
    """
    tick: int                                # current server tick number
    position: Mapping[str, float]           # {"x": ..., "y": ..., "z": ...}
    dimension: str                          # e.g. "overworld", "nether"
    inventory: List[Dict[str, Any]]         # list of item stacks
    nearby_entities: List[Dict[str, Any]]   # mobs, players, dropped items, etc.
    blocks_of_interest: List[Dict[str, Any]]  # ores, machines, machines, etc.
    tech_state: Dict[str, Any]              # inferred tech progression info
    context: Dict[str, Any]                 # misc extra metadata (phase, flags, etc.)


@dataclass
class Observation:
    """Compact encoding of WorldState for LLMs/tools.

    Produced by the observation encoder (M7).
    """
    json_payload: Dict[str, Any]            # LLM-ready JSON-like dict
    text_summary: str                       # optional textual summary for prompts


@dataclass
class Action:
    """Abstract action that AgentLoop can send to BotCore."""
    type: str                               # e.g. "move", "break_block", "use_item"
    params: Dict[str, Any]                  # parameters for the action


@dataclass
class ActionResult:
    """Result of executing an Action."""
    success: bool                           # did it work?
    error: Optional[str]                    # error message if not
    details: Dict[str, Any]                 # optional extra info (e.g. new position)
