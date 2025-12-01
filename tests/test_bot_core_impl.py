# tests/test_bot_core_impl.py
"""
Integration tests for BotCoreImpl using FakePacketClient.

Covers:
- Wiring between BotCoreImpl, WorldTracker, ActionExecutor.
- observe() and get_world_state() path.
- execute_action() sending packets via FakePacketClient.
"""

from __future__ import annotations

from typing import Any

import types

from spec.types import Action, WorldState  # M1 types
from bot_core.core import BotCoreImpl
from bot_core.testing.fakes import FakePacketClient
from bot_core.snapshot import RawWorldSnapshot
from bot_core.actions import ActionExecutorConfig


class DummyEnv:
    """Minimal stand-in for EnvProfile for tests."""

    def __init__(self) -> None:
        self.name = "test_env"
        self.bot_mode = "forge_mod"


def _patch_load_environment(monkeypatch: Any) -> None:
    import env.loader as loader

    monkeypatch.setattr(loader, "load_environment", lambda: DummyEnv())


def test_botcore_observe_and_world_state(monkeypatch: Any) -> None:
    _patch_load_environment(monkeypatch)

    client = FakePacketClient()
    bot = BotCoreImpl(client=client)

    bot.connect()
    client.emit(
        "position_update",
        {"x": 1.0, "y": 64.0, "z": 2.0, "yaw": 45.0, "pitch": 0.0},
    )
    client.emit("time_update", {"tick": 99})

    bot.tick()  # no-op for FakePacketClient but keeps the API consistent

    raw = bot.observe()
    assert isinstance(raw, RawWorldSnapshot)
    assert raw.tick == 99
    assert raw.player_pos["x"] == 1.0

    world = bot.get_world_state()
    assert isinstance(world, WorldState)
    assert world.tick == 99
    assert world.position["x"] == 1.0


def test_botcore_execute_action_routes_to_executor(monkeypatch: Any) -> None:
    _patch_load_environment(monkeypatch)

    client = FakePacketClient()
    bot = BotCoreImpl(client=client, action_config=ActionExecutorConfig())

    bot.connect()

    # Set initial position in tracker via FakePacketClient
    client.emit(
        "position_update",
        {"x": 0.0, "y": 64.0, "z": 0.0},
    )

    action = Action(
        type="break_block",
        params={"x": 1, "y": 64, "z": 2},
    )

    result = bot.execute_action(action)

    assert result.success
    types = [p.packet_type for p in client.sent_packets]
    assert "block_dig" in types

