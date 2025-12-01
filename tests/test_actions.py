# tests/test_actions.py
"""
Unit tests for ActionExecutor.

Focus:
- move_to calls pathfinder and emits move_step packets.
- break_block emits block_dig start/stop.
- place_block emits block_place.
"""

from __future__ import annotations

from typing import Any, Dict

from spec.types import Action  # M1 type
from bot_core.snapshot import RawWorldSnapshot
from bot_core.actions import ActionExecutor, ActionExecutorConfig
from bot_core.testing.fakes import FakePacketClient
from bot_core.nav import NavGrid, BlockSolidFn


def make_flat_snapshot() -> RawWorldSnapshot:
    return RawWorldSnapshot(
        tick=0,
        dimension="overworld",
        player_pos={"x": 0.0, "y": 64.0, "z": 0.0},
        player_yaw=0.0,
        player_pitch=0.0,
        on_ground=True,
        chunks={},
        entities=[],
        inventory=[],
        context={},
    )


def floor_solid(x: int, y: int, z: int, snapshot: RawWorldSnapshot) -> bool:
    return y == 63


def test_move_to_emits_move_step_packets() -> None:
    client = FakePacketClient()
    executor = ActionExecutor(
        client,
        is_solid_block=floor_solid,
        config=ActionExecutorConfig(max_nav_steps=256, default_move_radius=0.5),
    )

    snapshot = make_flat_snapshot()

    action = Action(
        type="move_to",
        params={"x": 3, "y": 64, "z": 0, "radius": 0.5},
    )

    result = executor.execute(action, snapshot)

    assert result.success
    assert result.error is None
    assert result.details["steps"] > 0

    move_packets = [
        p for p in client.sent_packets if p.packet_type == "move_step"
    ]
    assert len(move_packets) == result.details["steps"]
    # Last step should be near the target
    last = move_packets[-1].data
    assert last["x"] == 3
    assert last["z"] == 0


def test_move_to_nav_failure_returns_error() -> None:
    client = FakePacketClient()
    # everything non-solid -> no path
    executor = ActionExecutor(
        client,
        is_solid_block=lambda x, y, z, s: False,  # type: ignore[arg-type]
        config=ActionExecutorConfig(max_nav_steps=64),
    )

    snapshot = make_flat_snapshot()

    action = Action(
        type="move_to",
        params={"x": 5, "y": 64, "z": 5},
    )

    result = executor.execute(action, snapshot)

    assert not result.success
    assert result.error == "nav_failure"
    assert "reason" in result.details


def test_break_block_sends_block_dig_packets() -> None:
    client = FakePacketClient()
    executor = ActionExecutor(client)

    snapshot = make_flat_snapshot()
    action = Action(
        type="break_block",
        params={"x": 1, "y": 64, "z": 2},
    )

    result = executor.execute(action, snapshot)

    assert result.success
    types = [p.packet_type for p in client.sent_packets]
    assert types.count("block_dig") == 2

    payloads = [p.data for p in client.sent_packets]
    statuses = {p["status"] for p in payloads}
    assert {"start", "stop"} <= statuses


def test_place_block_sends_block_place_packet() -> None:
    client = FakePacketClient()
    executor = ActionExecutor(client)

    snapshot = make_flat_snapshot()
    action = Action(
        type="place_block",
        params={"x": 2, "y": 64, "z": 2},
    )

    result = executor.execute(action, snapshot)

    assert result.success
    types = [p.packet_type for p in client.sent_packets]
    assert "block_place" in types

