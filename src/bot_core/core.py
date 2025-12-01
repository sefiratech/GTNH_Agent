# src/bot_core/core.py
"""
Concrete BotCore implementation for Minecraft 1.7.10 (GTNH).

This module wires together:
- PacketClient / IPC transport (M6 network layer)
- WorldTracker (incremental raw world state)
- ActionExecutor (high-level action execution)
- Snapshot adapter (RawWorldSnapshot -> WorldState)
- ActionTracer (logging / metrics for actions)

Public surface (for higher layers like M8):
    class BotCoreImpl(BotCore):
        connect() -> None
        disconnect() -> None
        tick() -> None
        observe() -> RawWorldSnapshot
        get_world_state() -> WorldState
        execute_action(Action) -> ActionResult

Design constraints:
- No packet or protocol details leak to callers.
- No navigation internals or exceptions leak to callers.
- All action-related failures return ActionResult with explicit error codes.
- Non-action failures (e.g., connection problems) raise domain errors.
- No semantics, virtues, or tech-state logic here.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Optional

from env.loader import load_environment  # Phase 0: EnvProfile loader
from spec.types import Action, ActionResult, WorldState  # M1 shared types

from spec.bot_core import BotCore  # BotCore interface definition (M1)
from .net import PacketClient, create_packet_client_for_env
from .world_tracker import WorldTracker
from .snapshot import RawWorldSnapshot, snapshot_to_world_state
from .actions import ActionExecutor, ActionExecutorConfig
from .collision import default_is_solid_block
from .tracing import ActionTracer


log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Domain errors
# ---------------------------------------------------------------------------


@dataclass
class BotCoreError(RuntimeError):
    """
    Domain-level error raised by BotCoreImpl for non-action failures.

    Examples:
        - failed to connect or disconnect cleanly
        - tick loop I/O failures
        - configuration errors

    Action execution errors should generally NOT raise this; instead,
    ActionExecutor must return an ActionResult with error set.
    """

    code: str
    details: dict[str, Any]

    def __str__(self) -> str:
        return f"BotCoreError(code={self.code!r}, details={self.details!r})"


# ---------------------------------------------------------------------------
# Implementation
# ---------------------------------------------------------------------------


class BotCoreImpl(BotCore):
    """
    Concrete BotCore for MC 1.7.10 GTNH.

    Orchestrates:
        - PacketClient / IPC (transport)
        - WorldTracker (RawWorldSnapshot)
        - ActionExecutor (high-level actions)
        - ActionTracer (metrics)

    Consumers (M8, etc.) see:
        - connect / disconnect / tick
        - observe -> RawWorldSnapshot
        - get_world_state -> WorldState
        - execute_action(Action) -> ActionResult
    """

    def __init__(
        self,
        client: Optional[PacketClient] = None,
        *,
        action_config: Optional[ActionExecutorConfig] = None,
        # Optional hook for supplying custom collision semantics.
        is_solid_block=None,
        tracer: Optional[ActionTracer] = None,
    ) -> None:
        """
        Build a BotCoreImpl.

        If `client` is None, it is constructed from the Phase 0 environment
        via create_packet_client_for_env(), which uses env.yaml / EnvProfile.
        """
        self._env = load_environment()

        # Transport layer
        self._client: PacketClient = client or create_packet_client_for_env()

        # World tracking
        self._tracker = WorldTracker(self._client)

        # Collision function: injected or default profile.
        solid_fn = is_solid_block if is_solid_block is not None else default_is_solid_block

        # Action execution
        self._executor = ActionExecutor(
            self._client,
            is_solid_block=solid_fn,
            config=action_config,
        )

        # Tracing / metrics
        self._tracer: ActionTracer = tracer or ActionTracer()

        self._connected: bool = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """
        Establish connection to the Minecraft world.

        Raises:
            BotCoreError if the underlying client fails to connect.
        """
        if self._connected:
            return

        try:
            self._client.connect()
        except Exception as exc:
            raise BotCoreError(
                code="connect_failed",
                details={"exception": repr(exc)},
            ) from exc

        self._connected = True

        # Seed tracker context with basic env profile metadata (optional).
        self._tracker.set_context(
            "env_profile_name",
            getattr(self._env, "name", "default"),
        )
        self._tracker.set_context("bot_mode", getattr(self._env, "bot_mode", "unknown"))

    def disconnect(self) -> None:
        """
        Disconnect from the Minecraft world.

        Raises:
            BotCoreError if the underlying client fails to disconnect.
        """
        if not self._connected:
            return

        try:
            self._client.disconnect()
        except Exception as exc:
            self._connected = False
            raise BotCoreError(
                code="disconnect_failed",
                details={"exception": repr(exc)},
            ) from exc

        self._connected = False

    def tick(self) -> None:
        """
        Pump the transport layer and feed events into the WorldTracker.

        This must be called regularly (e.g., once per iteration of the main
        agent loop). It does NOT itself execute any actions.
        """
        if not self._connected:
            return

        try:
            self._client.tick()
        except Exception as exc:
            raise BotCoreError(
                code="tick_failed",
                details={"exception": repr(exc)},
            ) from exc

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def observe(self) -> RawWorldSnapshot:
        """
        Return a raw snapshot of the current world state.

        This is intended for internal consumers (e.g., debugging, advanced
        semantics modules). Most of the time higher layers should use
        get_world_state() instead.
        """
        return self._tracker.build_snapshot()

    def get_world_state(self) -> WorldState:
        """
        Return a semantic WorldState for agent consumption.
        """
        raw = self._tracker.build_snapshot()
        return snapshot_to_world_state(raw)

    # ------------------------------------------------------------------
    # Action execution
    # ------------------------------------------------------------------

    def execute_action(self, action: Action) -> ActionResult:
        """
        Execute a single high-level Action.

        All navigation / pathfinding / I/O details are internal. Nav and
        transport exceptions are caught and surfaced as ActionResult errors.
        """
        raw = self._tracker.build_snapshot()

        start = perf_counter()
        try:
            result = self._executor.execute(action, raw)
        except Exception as exc:
            duration = perf_counter() - start
            log.exception("BotCoreImpl.execute_action unexpected exception")
            # Even if executor exploded, we still try to trace the failure.
            fallback = ActionResult(
                success=False,
                error="botcore_execute_exception",
                details={
                    "exception": repr(exc),
                    "action_type": getattr(action, "type", None),
                },
            )
            try:
                self._tracer.record(
                    action=action,
                    snapshot=raw,
                    result=fallback,
                    duration_s=duration,
                )
            except Exception:
                # Tracing must never break the caller.
                log.exception("Action tracing failed after executor exception")
            return fallback

        duration = perf_counter() - start

        # Ensure we always have a proper ActionResult object.
        if not isinstance(result, ActionResult):
            result = ActionResult(
                success=False,
                error="invalid_executor_result",
                details={"result_repr": repr(result)},
            )

        # Record trace; errors here must be non-fatal.
        try:
            self._tracer.record(
                action=action,
                snapshot=raw,
                result=result,
                duration_s=duration,
            )
        except Exception:
            log.exception("Action tracing failed")

        return result

    # ------------------------------------------------------------------
    # Tracing access (optional helpers)
    # ------------------------------------------------------------------

    def get_action_traces(self) -> list[Any]:
        """
        Return a snapshot of recorded action traces.

        Primarily for debugging / monitoring tooling. The exact record type
        is ActionTraceRecord, but we keep the signature loose to avoid
        coupling callers to this module.
        """
        return self._tracer.get_records()

