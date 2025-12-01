# AgentController linking control commands to AgentLoopV1
#src/monitoring/controller.py
"""
Control surface for M9 – monitoring_and_tools.

AgentController wraps an AgentLoop-like object and exposes safe external
control via ControlCommand messages on the EventBus.

Supported commands (ControlCommandType):
- PAUSE          -> freeze stepping
- RESUME         -> unfreeze stepping
- SINGLE_STEP    -> run exactly one AgentLoop step, then pause again
- CANCEL_PLAN    -> drop the current plan
- SET_GOAL       -> update the agent's high-level goal
- DUMP_STATE     -> emit a debug snapshot as a SNAPSHOT event
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Protocol

from .bus import EventBus
from .events import (
    ControlCommand,
    ControlCommandType,
    EventType,
)
from .logger import log_event


# ============================================================
# Agent interface expected by the controller
# ============================================================

class AgentLoopControl(Protocol):
    """
    Minimal protocol describing what the controller expects from M8's AgentLoop.

    Any concrete agent loop that wants to be controlled by AgentController
    must implement these methods.
    """

    def step(self) -> None:
        """Execute one iteration of the agent loop."""

    def cancel_current_plan(self) -> None:
        """Cancel the currently active plan, if any."""

    def set_goal(self, goal: str) -> None:
        """Set a new high-level goal for the agent."""

    def debug_state(self) -> Dict[str, Any]:
        """
        Return a JSON-serializable snapshot of internal state.

        Recommended contents:
        - current phase
        - current/last plan
        - latest tech_state snapshot
        - env_profile_name / context_id
        - any other useful metadata
        """
        ...


# ============================================================
# Agent Controller
# ============================================================

class AgentController:
    """
    Control surface for the AgentLoop.

    - Listens for ControlCommand instances on the EventBus.
    - Provides pause/resume/single-step semantics for stepping.
    - Forwards CANCEL_PLAN and SET_GOAL into the agent.
    - Emits state snapshots as SNAPSHOT events when requested.

    The main runtime should call `maybe_step_agent()` instead of calling
    `agent.step()` directly, so pause/single-step flags are respected.
    """

    def __init__(self, agent: AgentLoopControl, bus: EventBus) -> None:
        self._agent = agent
        self._bus = bus

        # Internal control flags
        self._paused: bool = False
        self._single_step: bool = False

        # Subscribe to control commands
        self._bus.subscribe_commands(self._handle_command)

    # --------------------------------------------------------
    # Command handling
    # --------------------------------------------------------

    def _handle_command(self, cmd: ControlCommand) -> None:
        """
        Process an incoming ControlCommand from dashboards, CLIs, or scripts.
        """
        if cmd.cmd == ControlCommandType.PAUSE:
            self._paused = True
            self._single_step = False
            self._log_control("PAUSE", {"paused": True})

        elif cmd.cmd == ControlCommandType.RESUME:
            self._paused = False
            self._single_step = False
            self._log_control("RESUME", {"paused": False})

        elif cmd.cmd == ControlCommandType.SINGLE_STEP:
            # Next maybe_step_agent() call will run exactly one step
            self._single_step = True
            self._paused = False
            self._log_control("SINGLE_STEP", {"single_step": True})

        elif cmd.cmd == ControlCommandType.CANCEL_PLAN:
            self._agent.cancel_current_plan()
            self._log_control("CANCEL_PLAN", {})

        elif cmd.cmd == ControlCommandType.SET_GOAL:
            goal = cmd.args.get("goal", "")
            self._agent.set_goal(goal)
            self._log_control("SET_GOAL", {"goal": goal})

        elif cmd.cmd == ControlCommandType.DUMP_STATE:
            state = self._safe_debug_state()
            self._log_snapshot(state)

    # --------------------------------------------------------
    # Stepping API for the main loop
    # --------------------------------------------------------

    def maybe_step_agent(self) -> None:
        """
        Replacement for direct agent.step() calls.

        Behavior:
        - If paused: do nothing.
        - If single_step is set: run one step, then re-pause.
        - Otherwise: run one step as normal.
        """
        if self._paused:
            return

        # Run exactly one iteration of the agent loop
        self._agent.step()

        if self._single_step:
            # After one step, go back to PAUSE
            self._paused = True
            self._single_step = False
            self._log_control("SINGLE_STEP_COMPLETED", {"paused": True})

    # --------------------------------------------------------
    # Introspection helpers (for tests / tooling)
    # --------------------------------------------------------

    @property
    def paused(self) -> bool:
        """Return whether the controller is currently paused."""
        return self._paused

    @property
    def single_step_pending(self) -> bool:
        """Return whether the next step will be a single-step execution."""
        return self._single_step

    # --------------------------------------------------------
    # Logging helpers
    # --------------------------------------------------------

    def _log_control(self, cmd_name: str, payload: Dict[str, Any]) -> None:
        """
        Emit a CONTROL_COMMAND monitoring event describing a control action.
        """
        log_event(
            bus=self._bus,
            module="M9.controller",
            event_type=EventType.CONTROL_COMMAND,
            message=f"Control command: {cmd_name}",
            payload={"cmd": cmd_name, **payload},
            correlation_id=None,
        )

    def _log_snapshot(self, state: Dict[str, Any]) -> None:
        """
        Emit a SNAPSHOT monitoring event containing debug state.
        """
        log_event(
            bus=self._bus,
            module="M9.controller",
            event_type=EventType.SNAPSHOT,
            message="Agent state snapshot",
            payload={"state": state},
            correlation_id=None,
        )

    def _safe_debug_state(self) -> Dict[str, Any]:
        """
        Call agent.debug_state() and do a best-effort normalization so
        it is JSON-serializable for logging.
        """
        try:
            state = self._agent.debug_state()
        except Exception as exc:  # pragma: no cover - defensive fallback
            return {
                "error": "debug_state_failed",
                "details": repr(exc),
            }

        # If it's a dataclass, convert it to a dict
        if is_dataclass(state):
            return asdict(state)

        # Otherwise assume it's already JSON-like
        return state

