# tests/test_world_tracker.py
"""
Unit tests for WorldTracker.

Covers:
- time_update
- position_update
- chunk_data
- spawn_entity / destroy_entities
- set_slot / window_items
"""

from __future__ import annotations

from typing import Any, Dict

from bot_core.world_tracker import WorldTracker
from bot_core.snapshot import RawWorldSnapshot
from bot_core.testing.fakes import FakePacketClient


def test_time_update_updates_tick() -> None:
    client = FakePacketClient()
    tracker = WorldTracker(client)

    client.emit("time_update", {"tick": 42})
    snap = tracker.build_snapshot()

    assert isinstance(snap, RawWorldSnapshot)
    assert snap.tick == 42


def test_position_update_updates_player_state() -> None:
    client = FakePacketClient()
    tracker = WorldTracker(client)

    client.emit(
        "position_update",
        {
            "x": 10.5,
            "y": 65.0,
            "z": -3.25,
            "yaw": 90.0,
            "pitch": 10.0,
            "on_ground": False,
        },
    )

    snap = tracker.build_snapshot()
    assert snap.player_pos["x"] == 10.5
    assert snap.player_pos["y"] == 65.0
    assert snap.player_pos["z"] == -3.25
    assert snap.player_yaw == 90.0
    assert snap.player_pitch == 10.0
    assert snap.on_ground is False


def test_chunk_data_stores_chunk() -> None:
    client = FakePacketClient()
    tracker = WorldTracker(client)

    client.emit(
        "chunk_data",
        {
            "chunk_x": 1,
            "chunk_z": -2,
            "blocks": "opaque_blocks_representation",
            "biomes": "biome_data_stub",
        },
    )

    snap = tracker.build_snapshot()
    assert (1, -2) in snap.chunks
    chunk = snap.chunks[(1, -2)]
    assert chunk.blocks == "opaque_blocks_representation"
    assert chunk.biome_data == "biome_data_stub"


def test_spawn_and_destroy_entities() -> None:
    client = FakePacketClient()
    tracker = WorldTracker(client)

    client.emit(
        "spawn_entity",
        {
            "entity_id": 123,
            "kind": "mob",
            "x": 1.0,
            "y": 64.0,
            "z": 2.0,
            "extra": "data",
        },
    )

    snap = tracker.build_snapshot()
    assert any(e.entity_id == 123 for e in snap.entities)

    client.emit("destroy_entities", {"entity_ids": [123]})
    snap2 = tracker.build_snapshot()
    assert all(e.entity_id != 123 for e in snap2.entities)


def test_inventory_set_slot_and_window_items() -> None:
    client = FakePacketClient()
    tracker = WorldTracker(client)

    # full inventory snapshot
    client.emit(
        "window_items",
        {
            "items": [
                {"id": "minecraft:stone", "count": 1},
                {"id": "minecraft:dirt", "count": 2},
            ]
        },
    )

    snap = tracker.build_snapshot()
    assert len(snap.inventory) == 2
    assert snap.inventory[0]["id"] == "minecraft:stone"

    # single slot update
    client.emit(
        "set_slot",
        {
            "slot": 1,
            "item": {"id": "minecraft:planks", "count": 4},
        },
    )

    snap2 = tracker.build_snapshot()
    assert snap2.inventory[1]["id"] == "minecraft:planks"
    assert snap2.inventory[1]["count"] == 4

