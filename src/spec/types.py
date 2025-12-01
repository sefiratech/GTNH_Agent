# core shared types: WorldState, Action, ActionResult, Observation
# src/spec/types.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, TypedDict


# ---------------------------------------------------------------------------
# Normalized shapes used inside WorldState
# ---------------------------------------------------------------------------

class InventoryStack(TypedDict, total=False):
    """
    Normalized inventory entry.

    Runtime representation is just a dict, but the *intended* shape is:

      {
        "item_id": "gregtech:gt.metaitem.01",   # canonical item id
        "variant": "plate.copper",              # semantic variant key or None
        "count": 32                             # stack size (>= 0)
      }

    Observation encoder (M7) is responsible for normalizing raw BotCore
    inventory snapshots into this shape before passing into WorldState.
    """
    item_id: str
    variant: Optional[str]
    count: int


class MachineEntry(TypedDict, total=False):
    """
    Normalized machine entry inside world.context["machines"].

    Runtime representation is also just a dict; expected shape:

      {
        "id": "gregtech:steam_macerator",       # canonical machine id
        "type": "gregtech:steam_macerator",     # legacy/free-form type; may equal id
        "tier": "steam",                        # semantic tier label if known
        "extra": {...}                          # arbitrary extra fields (coords, dim, etc.)
      }

    Again, M7 is responsible for producing this from BotCore snapshots.
    """
    id: Optional[str]
    type: Optional[str]
    tier: Optional[str]
    extra: Dict[str, Any]


# ---------------------------------------------------------------------------
# Core world / agent types
# ---------------------------------------------------------------------------

@dataclass
class WorldState:
    """Semantic snapshot of the world relevant to decision-making.

    This is not raw packets. It's a structured view that BotCore (M6) /
    observation encoder (M7) produce for the AgentLoop and semantics (M3).

    Fields:

      - tick:
          Current server tick number.

      - position:
          Player/bot position as a simple dict:
            {"x": float, "y": float, "z": float}

      - dimension:
          Dimension id, e.g. "overworld", "nether", "end", or GTNH-specific.

      - inventory:
          List of normalized inventory stacks (see InventoryStack).
          Semantics modules assume the dict keys "item_id", "variant", "count"
          are present, even though this is typed as a list of dict-like values.

      - nearby_entities:
          Free-form entity descriptors (mobs, players, dropped items, etc.).
          Shape is intentionally loose and can evolve with BotCore.

      - blocks_of_interest:
          Highlighted blocks (ores, machines, chests, etc.) near the agent.
          Again, shape is intentionally loose; M3 may later use this.

      - tech_state:
          Inferred tech progression info. M3's tech_state module may overwrite
          this with a structured dict or serialized TechState.

      - context:
          Misc extra metadata and derived state.

          By convention, MUST at least expose:

            context["machines"]: list[MachineEntry]

          where each machine entry follows the MachineEntry TypedDict shape.
    """
    tick: int
    position: Mapping[str, float]
    dimension: str
    inventory: List[InventoryStack]
    nearby_entities: List[Dict[str, Any]]
    blocks_of_interest: List[Dict[str, Any]]
    tech_state: Dict[str, Any]
    context: Dict[str, Any]


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

