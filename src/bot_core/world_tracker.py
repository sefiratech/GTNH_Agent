# track chunks/entities and build snapshots from packets
# src/bot_core/world_tracker.py
"""
World tracker for bot_core_1_7_10.

Consumes normalized packet / IPC events from a PacketClient and maintains
a raw, incrementally updated view of the world. This module is the ONLY
owner of RawWorldSnapshot assembly.

Rules:
- Never construct WorldState here (that lives in bot_core.snapshot).
- Never embed GTNH semantics or tech-state inference (M3 owns that).
- Keep storage minimal and "raw"; avoid bloated derived structures.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Tuple

from .snapshot import RawWorldSnapshot, RawChunk, RawEntity
from .net import PacketClient, PacketHandler


@dataclass
class _PlayerState:
    """Minimal tracked state for the local player."""

    pos: Dict[str, float] = field(
        default_factory=lambda: {"x": 0.0, "y": 64.0, "z": 0.0}
    )
    yaw: float = 0.0
    pitch: float = 0.0
    on_ground: bool = True
    dimension: str = "overworld"


class WorldTracker:
    """
    Maintains an incrementally updated RawWorldSnapshot.

    PacketClient implementations must normalize wire data into logically
    named event types:

        - "time_update"        → tick/time updates
        - "position_update"    → player position/rotation
        - "chunk_data"         → chunk or region data
        - "spawn_entity"       → entity created
        - "destroy_entities"   → entities destroyed
        - "set_slot"           → single inventory slot changed
        - "window_items"       → full inventory snapshot
        - (optionally) "dimension_change" → dimension switch

    This tracker does NOT:
      - Interpret block types or GTNH tech semantics
      - Compute tech_state
      - Talk to LLMs

    It only stores enough to build a coherent RawWorldSnapshot.
    """

    def __init__(self, client: PacketClient) -> None:
        self._client = client

        # Basic scalar world state
        self._tick: int = 0
        self._player: _PlayerState = _PlayerState()

        # Chunks keyed by (chunk_x, chunk_z)
        self._chunks: Dict[Tuple[int, int], RawChunk] = {}

        # Entities keyed by numeric ID
        self._entities: Dict[int, RawEntity] = {}

        # Flat inventory representation (list of slot dicts)
        self._inventory: List[Dict[str, Any]] = []

        # Context map for misc metadata (env profile, hazards, etc.)
        self._context: Dict[str, Any] = {}

        # Wire client → handlers
        self._register_handlers()

    # ------------------------------------------------------------------
    # Packet wiring
    # ------------------------------------------------------------------

    def _register_handlers(self) -> None:
        """
        Register handlers for the normalized event types we care about.

        PacketClient is responsible for translating real wire protocol /
        IPC messages into these logical events.
        """
        self._client.on_packet("time_update", self._handle_time_update)
        self._client.on_packet("position_update", self._handle_position_update)
        self._client.on_packet("chunk_data", self._handle_chunk_data)
        self._client.on_packet("spawn_entity", self._handle_spawn_entity)
        self._client.on_packet("destroy_entities", self._handle_destroy_entities)
        self._client.on_packet("set_slot", self._handle_set_slot)
        self._client.on_packet("window_items", self._handle_window_items)
        # Optional, but useful if IPC exposes it:
        self._client.on_packet("dimension_change", self._handle_dimension_change)

    # ------------------------------------------------------------------
    # Packet handlers
    # ------------------------------------------------------------------

    def _handle_time_update(self, pkt: Mapping[str, Any]) -> None:
        """
        Update world tick/time from server.

        Expected fields (normalized by client / IPC layer):
            - "tick" or "time": int
        """
        value = pkt.get("tick", pkt.get("time", self._tick))
        try:
            self._tick = int(value)
        except (TypeError, ValueError):
            # Leave tick unchanged on garbage input.
            return

    def _handle_position_update(self, pkt: Mapping[str, Any]) -> None:
        """
        Update player position/rotation.

        Expected fields:
            - "x", "y", "z": float
            - "yaw", "pitch": float
            - "on_ground": bool (optional)
        """
        x = pkt.get("x", self._player.pos["x"])
        y = pkt.get("y", self._player.pos["y"])
        z = pkt.get("z", self._player.pos["z"])

        try:
            self._player.pos = {
                "x": float(x),
                "y": float(y),
                "z": float(z),
            }
        except (TypeError, ValueError):
            # Ignore malformed position updates.
            pass

        yaw = pkt.get("yaw")
        pitch = pkt.get("pitch")
        if yaw is not None:
            try:
                self._player.yaw = float(yaw)
            except (TypeError, ValueError):
                pass
        if pitch is not None:
            try:
                self._player.pitch = float(pitch)
            except (TypeError, ValueError):
                pass

        if "on_ground" in pkt:
            self._player.on_ground = bool(pkt["on_ground"])

    def _handle_chunk_data(self, pkt: Mapping[str, Any]) -> None:
        """
        Store or update chunk data.

        Expected fields:
            - "chunk_x", "chunk_z": ints
            - "blocks": opaque block storage
            - "biomes": optional biome data
        """
        try:
            cx = int(pkt["chunk_x"])
            cz = int(pkt["chunk_z"])
        except Exception:
            return

        blocks = pkt.get("blocks")
        biomes = pkt.get("biomes")

        if blocks is None:
            # No block data, nothing to store.
            return

        self._chunks[(cx, cz)] = RawChunk(
            x=cx,
            z=cz,
            blocks=blocks,
            biome_data=biomes,
        )

    def _handle_spawn_entity(self, pkt: Mapping[str, Any]) -> None:
        """
        Track newly spawned entities.

        Expected fields:
            - "entity_id": int
            - "kind" or "type": str
            - "x", "y", "z": float
            - plus any additional data (kept in 'data')
        """
        try:
            entity_id = int(pkt["entity_id"])
        except Exception:
            return

        kind = pkt.get("kind", pkt.get("type", "unknown"))

        try:
            x = float(pkt.get("x", 0.0))
            y = float(pkt.get("y", 0.0))
            z = float(pkt.get("z", 0.0))
        except (TypeError, ValueError):
            x = y = z = 0.0

        # Keep original payload as data for now.
        data = dict(pkt)

        self._entities[entity_id] = RawEntity(
            entity_id=entity_id,
            kind=str(kind),
            x=x,
            y=y,
            z=z,
            data=data,
        )

    def _handle_destroy_entities(self, pkt: Mapping[str, Any]) -> None:
        """
        Remove entities that the server reports as destroyed.

        Expected fields:
            - "entity_ids": iterable of ints
        """
        ids = pkt.get("entity_ids")
        if not ids:
            return

        for raw_id in ids:
            try:
                eid = int(raw_id)
            except Exception:
                continue
            self._entities.pop(eid, None)

    def _handle_set_slot(self, pkt: Mapping[str, Any]) -> None:
        """
        Update a single inventory slot.

        Expected fields (normalized):
            - "slot": int
            - "item": mapping (stack data) or None
        """
        slot = pkt.get("slot")
        item = pkt.get("item")

        try:
            idx = int(slot)
        except Exception:
            return

        # Ensure list is large enough.
        while len(self._inventory) <= idx:
            self._inventory.append({})

        # Represent empty slot as {} for simplicity.
        self._inventory[idx] = dict(item) if isinstance(item, Mapping) else {}

    def _handle_window_items(self, pkt: Mapping[str, Any]) -> None:
        """
        Replace the entire inventory representation.

        Expected fields:
            - "items": list[Mapping[str, Any]] (one per slot)
        """
        items = pkt.get("items")
        if not isinstance(items, list):
            return

        new_inv: List[Dict[str, Any]] = []
        for entry in items:
            if isinstance(entry, Mapping):
                new_inv.append(dict(entry))
            else:
                new_inv.append({})
        self._inventory = new_inv

    def _handle_dimension_change(self, pkt: Mapping[str, Any]) -> None:
        """
        Handle dimension change events.

        Expected fields:
            - "dimension": str or int
        """
        dim = pkt.get("dimension")
        if dim is None:
            return
        self._player.dimension = str(dim)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def tick(self) -> int:
        """Current world tick as tracked by the server."""
        return self._tick

    @property
    def dimension(self) -> str:
        """Current dimension ID/name."""
        return self._player.dimension

    def set_context(self, key: str, value: Any) -> None:
        """
        Attach arbitrary metadata to the tracker context.

        Useful for storing:
          - env profile labels
          - world seed (if known)
          - runtime tags (e.g. "phase": "m6_online")
        """
        self._context[key] = value

    def update_context(self, data: Mapping[str, Any]) -> None:
        """Bulk update context with a mapping."""
        for k, v in data.items():
            self._context[k] = v

    def build_snapshot(self) -> RawWorldSnapshot:
        """
        Build a RawWorldSnapshot from the current tracked state.

        This is the ONLY way to get a snapshot out of the tracker and the
        ONLY place in M6 that assembles a RawWorldSnapshot.
        """
        return RawWorldSnapshot(
            tick=self._tick,
            dimension=self._player.dimension,
            player_pos=dict(self._player.pos),
            player_yaw=self._player.yaw,
            player_pitch=self._player.pitch,
            on_ground=self._player.on_ground,
            chunks=dict(self._chunks),
            entities=list(self._entities.values()),
            inventory=[dict(stack) for stack in self._inventory],
            context=dict(self._context),
        )


__all__ = ["WorldTracker"]

