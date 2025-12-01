# src/bot_core/runtime.py
"""
Runtime wiring for BotCore (M6).

For now this module provides a lightweight, in-process FakeBotCore that
implements the BotCore contract without talking to a real Minecraft
server or IPC layer.

This exists primarily so that:

  - M6 can be considered "functionally complete" for observation.
  - tests/test_m6_observe_contract.py can construct a BotCore via
    `get_bot_core(...)` and exercise the RawWorldSnapshot contract.

Later, when the Forge/IPC pipeline is ready, `get_bot_core` can be
updated to return a real networked BotCore implementation based on the
env profile (e.g. server address, world path, etc.).
"""

from __future__ import annotations

from typing import Optional, Dict, Any, List, Tuple

from spec.bot_core import BotCore
from spec.types import WorldState, Action, ActionResult
from bot_core.snapshot import RawWorldSnapshot, RawChunk, RawEntity, snapshot_to_world_state


class FakeBotCore(BotCore):
    """
    Fake bot core that produces a RawWorldSnapshot and a derived WorldState.

    This is intentionally minimal and in-process:

      - `observe()` returns a RawWorldSnapshot instance whose shape matches
        tests/test_m6_observe_contract.py expectations.
      - `get_world_state()` adapts that snapshot via snapshot_to_world_state().
      - `execute_action()` understands a single primitive "move_to" action and
        updates the internal position, always reporting success.

    It is good enough to:
      - validate the BotCore → RawWorldSnapshot → WorldState contract (M6/M7)
      - drive offline tests for M6/M7/M8 without any external server.
    """

    def __init__(self) -> None:
        self._tick: int = 0
        # Simple player position; y ~ "ground level" for tests.
        self._pos: Dict[str, float] = {"x": 0.0, "y": 64.0, "z": 0.0}
        self._dimension: str = "overworld"

    # ------------------------------------------------------------------
    # Observation API
    # ------------------------------------------------------------------

    def observe(self) -> RawWorldSnapshot:
        """
        Return a RawWorldSnapshot instance.

        Snapshot structure matches tests/test_m6_observe_contract.py:

          - player_pos: dict with keys x, y, z
          - entities: list/tuple (empty is allowed)
          - inventory: list/tuple (empty is allowed)

        We keep entities and inventory empty for now, which the tests
        explicitly allow.
        """
        self._tick += 1

        snapshot = RawWorldSnapshot(
            tick=self._tick,
            dimension=self._dimension,
            player_pos=dict(self._pos),
            player_yaw=0.0,
            player_pitch=0.0,
            on_ground=True,
            chunks={},            # type: Dict[Tuple[int, int], RawChunk]
            entities=[],          # type: List[RawEntity]
            inventory=[],         # type: List[Dict[str, Any]]
            context={"profile": "fake_bot_core"},
        )
        return snapshot

    def get_world_state(self) -> WorldState:
        """
        Adapt the most recent RawWorldSnapshot into a WorldState.

        This is what higher-level modules (M7/M8) should normally consume.
        """
        snapshot = self.observe()
        return snapshot_to_world_state(snapshot)

    # ------------------------------------------------------------------
    # Action API
    # ------------------------------------------------------------------

    def execute_action(self, action: Action) -> ActionResult:
        """
        Execute a single high-level Action and return an ActionResult.

        Supported primitive in this fake:
          - type == "move_to" with params {"x", "z"} and optional "y".

        Any other action type is treated as a no-op that still succeeds.
        """
        if action.type == "move_to":
            params = action.params or {}
            # Update internal position; keep y if not specified.
            self._pos["x"] = float(params["x"])
            self._pos["y"] = float(params.get("y", self._pos["y"]))
            self._pos["z"] = float(params["z"])

        # In this fake, everything "works".
        return ActionResult(success=True, error=None, details={})


def get_bot_core(env_profile_name: Optional[str] = None) -> BotCore:
    """
    Factory for obtaining a BotCore instance for the given environment.

    Parameters
    ----------
    env_profile_name:
        Name of the EnvProfile (e.g. "dev_local", "offline_fake").
        Ignored for now; kept for future compatibility when we add
        real IPC-backed BotCore implementations.

    Returns
    -------
    BotCore
        For now, always returns a `FakeBotCore` instance.
    """
    # Future: choose between FakeBotCore, IPCBotCore, etc. based on
    # env_profile_name and env.yaml config.
    return FakeBotCore()

