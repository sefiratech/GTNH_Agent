#!/usr/bin/env python3
"""
tools/smoke_botcore.py

Minimal harness to sanity-check BotCoreImpl wiring.

Default mode:
    - Uses FakePacketClient (no real network)
    - Emits a few synthetic packets
    - Calls:
        - observe()
        - get_world_state()
        - execute_action(move_to)
        - execute_action(break_block)
    - Prints WorldState, ActionResult, and sent packets

Later:
    - Use --mode real to use the real PacketClient from env.yaml
      (once the Forge IPC side exists).
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Ensure src/ is on sys.path when running as a script
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# ---------------------------------------------------------------------------
# Imports from the project
# ---------------------------------------------------------------------------

from spec.types import Action, WorldState, ActionResult  # type: ignore[import]
from bot_core import BotCoreImpl  # type: ignore[import]
from bot_core.snapshot import RawWorldSnapshot  # type: ignore[import]
from bot_core.testing.fakes import FakePacketClient  # type: ignore[import]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_dict(obj: Any) -> Any:
    """Best-effort conversion of dataclasses to plain dicts for printing."""
    if is_dataclass(obj):
        return asdict(obj)
    return obj


def _print_header(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def run_fake_mode() -> None:
    """
    Run a pure in-memory smoke test with FakePacketClient.

    This does NOT require a running Minecraft or Forge mod.
    """
    _print_header("Fake mode: initializing BotCoreImpl with FakePacketClient")

    client = FakePacketClient()
    bot = BotCoreImpl(client=client)

    # Connect (FakePacketClient.connect is a no-op except for state flag)
    bot.connect()

    # Emit some basic world packets into the WorldTracker via FakePacketClient
    client.emit("time_update", {"tick": 42})
    client.emit(
        "position_update",
        {
            "x": 1.5,
            "y": 64.0,
            "z": -2.0,
            "yaw": 90.0,
            "pitch": 10.0,
            "on_ground": True,
        },
    )

    # One tick() call to match the usual flow; FakePacketClient.tick is a no-op.
    bot.tick()

    # ------------------------------------------------------------------
    # Observe raw + semantic world state
    # ------------------------------------------------------------------
    _print_header("RawWorldSnapshot (observe)")
    raw: RawWorldSnapshot = bot.observe()
    print(_to_dict(raw))

    _print_header("WorldState (get_world_state)")
    world: WorldState = bot.get_world_state()
    print(_to_dict(world))

    # ------------------------------------------------------------------
    # Execute a move_to action
    # ------------------------------------------------------------------
    _print_header("execute_action: move_to {x=3,y=64,z=-2}")

    move_action = Action(
        type="move_to",
        params={"x": 3, "y": 64, "z": -2, "radius": 0.5},
    )

    move_result: ActionResult = bot.execute_action(move_action)
    print("ActionResult:", _to_dict(move_result))

    print("\nSent packets (after move_to):")
    for p in client.sent_packets:
        print(f"  - {p.packet_type}: {p.data}")

    # Clear sent packets so we can see only the break_block effect next.
    client.sent_packets.clear()

    # ------------------------------------------------------------------
    # Execute a break_block action
    # ------------------------------------------------------------------
    _print_header("execute_action: break_block {x=1,y=64,z=-2}")

    break_action = Action(
        type="break_block",
        params={"x": 1, "y": 64, "z": -2},
    )

    break_result: ActionResult = bot.execute_action(break_action)
    print("ActionResult:", _to_dict(break_result))

    print("\nSent packets (after break_block):")
    for p in client.sent_packets:
        print(f"  - {p.packet_type}: {p.data}")

    _print_header("Fake mode completed")


def run_real_mode() -> None:
    """
    Placeholder for future real IPC smoke test.

    This expects:
        - env.yaml to be configured with bot_mode: forge_mod or similar
        - create_packet_client_for_env() to construct a working transport
        - A running Forge mod IPC server

    For now this just wires BotCoreImpl with its default client and
    attempts a minimal call sequence.
    """
    _print_header("Real mode: initializing BotCoreImpl with real PacketClient")

    bot = BotCoreImpl()  # uses real client from env.yaml

    try:
        bot.connect()
    except Exception as exc:  # BotCoreError or lower-level error
        print("connect() failed:", repr(exc))
        return

    try:
        # Give the transport a few ticks to populate basic state.
        for _ in range(3):
            bot.tick()

        _print_header("WorldState (get_world_state)")
        world = bot.get_world_state()
        print(_to_dict(world))

        # Try a small move near the current position.
        pos = world.position  # type: ignore[attr-defined]
        move_action = Action(
            type="move_to",
            params={
                "x": int(pos.get("x", 0)) + 1,
                "y": int(pos.get("y", 64)),
                "z": int(pos.get("z", 0)),
                "radius": 0.5,
            },
        )

        _print_header("execute_action (real): move_to one block over")
        result = bot.execute_action(move_action)
        print("ActionResult:", _to_dict(result))

    finally:
        try:
            bot.disconnect()
        except Exception as exc:
            print("disconnect() failed:", repr(exc))

    _print_header("Real mode completed")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Smoke test harness for bot_core BotCoreImpl",
    )
    parser.add_argument(
        "--mode",
        choices=["fake", "real"],
        default="fake",
        help="Run in 'fake' (no network) or 'real' (env.yaml PacketClient) mode",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR)",
    )

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    if args.mode == "fake":
        run_fake_mode()
    else:
        run_real_mode()


if __name__ == "__main__":
    main()

