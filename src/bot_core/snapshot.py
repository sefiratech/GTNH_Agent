# RawWorldSnapshot + conversions to WorldState
# src/bot_core/snapshot.py
"""
Snapshot structures for bot_core_1_7_10.

This module defines the raw world snapshot types used internally by M6 and
provides the adapter that converts a RawWorldSnapshot into the canonical
WorldState type defined in `spec.types`.

Design goals:
- Keep Raw* structures close to the data we ingest from the network / IPC.
- Keep WorldState stable and M1-owned; this module only adapts into it.
- Do not embed GTNH semantics or tech inference here. That is M3’s job.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, MutableMapping, Tuple


# ---------------------------------------------------------------------------
# Raw world types (internal to bot_core)
# ---------------------------------------------------------------------------


@dataclass
class RawChunk:
    """
    Raw representation of a chunk as seen by the bot_core.

    This is intentionally opaque with respect to block storage layout.
    Higher-level semantics and block categorization are handled by M3.
    """

    x: int
    z: int
    blocks: Any  # implementation-defined representation (e.g. 16x256x16 array)
    biome_data: Any | None = None


@dataclass
class RawEntity:
    """
    Raw entity data as captured from packets or IPC messages.

    `data` should contain the original payload or a thin normalized view
    (e.g. type-specific fields, NBT-ish data, health, motion).
    """

    entity_id: int
    kind: str  # "player", "mob", "item", etc.
    x: float
    y: float
    z: float
    data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RawWorldSnapshot:
    """
    Full raw snapshot of the world as tracked by M6.

    This is the canonical "raw" view for bot_core. All other subsystems
    (observation encoder, semantics, planner) should work with WorldState,
    not this type.
    """

    tick: int
    dimension: str

    player_pos: Dict[str, float]  # {"x": float, "y": float, "z": float}
    player_yaw: float
    player_pitch: float
    on_ground: bool

    chunks: Dict[Tuple[int, int], RawChunk]
    entities: List[RawEntity]
    inventory: List[Dict[str, Any]]

    # Misc runtime context, used to carry extra signals such as:
    # - current env profile metadata
    # - high-level tags (near machines, hazards, etc.)
    # - bridge info from M0/M1
    context: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# WorldState adapter (RawWorldSnapshot -> spec.types.WorldState)
# ---------------------------------------------------------------------------


def _entity_to_summary(entity: RawEntity) -> Dict[str, Any]:
    """
    Convert a RawEntity to the generic "nearby_entities" dict format
    expected by WorldState.

    The exact structure is intentionally simple and stable; richer per-entity
    semantics belong in M3 / semantics.
    """
    return {
        "id": entity.entity_id,
        "kind": entity.kind,
        "x": entity.x,
        "y": entity.y,
        "z": entity.z,
        "data": dict(entity.data),
    }


def _compute_blocks_of_interest(snapshot: RawWorldSnapshot) -> List[Dict[str, Any]]:
    """
    Extract a shallow 'blocks_of_interest' list from the snapshot.

    M6 does not know GTNH tech semantics, so this is intentionally minimal.
    You can later extend this to surface:
      - blocks directly under / around the player
      - known "front-of-player" interaction targets
      - cached points of interest maintained by world_tracker

    For now, this returns an empty list; semantics modules can derive
    more specific views directly from RawWorldSnapshot + chunk data if needed.
    """
    # Placeholder: no opinion about which blocks are "interesting" at M6.
    # This keeps M6 decoupled from GTNH semantics (M3).
    return []


def snapshot_to_world_state(snapshot: RawWorldSnapshot) -> "WorldState":
    """
    Adapt a RawWorldSnapshot into the canonical WorldState.

    This is the ONLY place bot_core should construct a WorldState. All other
    code should either:
      - operate on RawWorldSnapshot, or
      - consume WorldState produced here.

    Expected WorldState fields (as defined in spec.types):
      - tick: int
      - position: Dict[str, float]
      - dimension: str
      - inventory: List[Dict[str, Any]]
      - nearby_entities: List[Dict[str, Any]]
      - blocks_of_interest: List[Dict[str, Any]]
      - tech_state: Mapping[str, Any] or similar (filled by M3)
      - context: Dict[str, Any]
    """
    # Import here to avoid circular imports at module import time.
    from spec.types import WorldState  # type: ignore

    # Shallow copy to avoid surprising mutations from outside.
    position = {
        "x": float(snapshot.player_pos.get("x", 0.0)),
        "y": float(snapshot.player_pos.get("y", 0.0)),
        "z": float(snapshot.player_pos.get("z", 0.0)),
    }

    nearby_entities: List[Dict[str, Any]] = [
        _entity_to_summary(e) for e in snapshot.entities
    ]

    blocks_of_interest = _compute_blocks_of_interest(snapshot)

    # tech_state is intentionally empty at this layer.
    tech_state: Dict[str, Any] = {}

    world_state = WorldState(
        tick=int(snapshot.tick),
        position=position,
        dimension=str(snapshot.dimension),
        inventory=[dict(stack) for stack in snapshot.inventory],
        nearby_entities=nearby_entities,
        blocks_of_interest=blocks_of_interest,
        tech_state=tech_state,
        context=dict(snapshot.context),
    )

    return world_state


__all__ = [
    "RawChunk",
    "RawEntity",
    "RawWorldSnapshot",
    "snapshot_to_world_state",
]

