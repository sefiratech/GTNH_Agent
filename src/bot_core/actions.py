# src/bot_core/actions.py
"""
Action execution for bot_core_1_7_10.

This module translates high-level Actions into protocol- or IPC-level
messages via a PacketClient.

Design constraints:
- High-level, atomic-ish actions:
    - move_to
    - break_block
    - place_block
    - use_item
    - interact
- One execute_action(...) call per logical operation.
- Explicit, structured failures:
    - no path
    - unsupported action
    - IO/IPC error
    - guards: nav_too_long, move_timeout
- No mutation of world state; this layer only sends packets/messages.
- No semantics, no tech_state, no virtues.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Dict, Mapping

from spec.types import Action, ActionResult  # from M1
from .snapshot import RawWorldSnapshot
from .net import PacketClient
from .nav import (
    NavGrid,
    BlockSolidFn,
    find_path,
    path_to_actions,
    current_coord_from_snapshot,
)
from .collision import (
    default_is_solid_block,
)


log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class ActionExecutorConfig:
    """
    Configuration knobs for ActionExecutor.

    These should be environment- and modpack-agnostic. GTNH-specific
    tuning belongs in higher layers or injected strategies.
    """

    # Limit on nav search complexity (A* internal steps)
    max_nav_steps: int = 2048

    # Default arrival radius for move_to
    default_move_radius: float = 0.5

    # Hard guard: maximum number of move steps we are willing to send
    # in a single move_to action. None disables this guard.
    max_move_steps: int | None = 10_000

    # Soft guard: maximum wall-clock time (seconds) we allow for sending
    # movement steps in a single execute_action("move_to", ...).
    # None disables this guard.
    max_move_duration_s: float | None = 30.0

    # Reserved for future use when we have async dig feedback.
    # For now, break_block is a fire-and-forget start+stop pair.
    max_dig_duration_s: float | None = 15.0


class ActionExecutor:
    """
    Translate high-level Actions into PacketClient messages.

    Public contract:
      execute(action, snapshot) -> ActionResult

    Responsibilities:
    - Build NavGrid from RawWorldSnapshot.
    - Run A* pathfinding for move_to.
    - Emit movement / dig / place / use / interact packets.
    - Surface explicit, structured errors.

    Non-responsibilities:
    - WorldState construction.
    - Tech semantics.
    - Planning, skill selection, or virtue scoring.
    """

    def __init__(
        self,
        client: PacketClient,
        *,
        is_solid_block: BlockSolidFn | None = None,
        config: ActionExecutorConfig | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self._client = client

        # Use injected collision logic if provided; otherwise fall back
        # to the module-level default profile.
        self._is_solid_block: BlockSolidFn = (
            is_solid_block if is_solid_block is not None else default_is_solid_block
        )

        self._cfg = config if config is not None else ActionExecutorConfig()
        self._log = logger or log

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def execute(self, action: Action, snapshot: RawWorldSnapshot) -> ActionResult:
        """
        Execute a single high-level Action against the given snapshot.

        The snapshot is treated as read-only and must not be mutated here.
        """
        atype = getattr(action, "type", None)
        params = getattr(action, "params", {}) or {}

        self._log.debug("ActionExecutor.execute start type=%s params=%r", atype, params)

        if not isinstance(params, Mapping):
            self._log.warning(
                "ActionExecutor.execute invalid params (not mapping): %r", params
            )
            return ActionResult(
                success=False,
                error="invalid_params",
                details={"reason": "params_not_mapping", "action_type": atype},
            )

        try:
            if atype == "move_to":
                result = self._execute_move_to(params, snapshot)
            elif atype == "break_block":
                result = self._execute_break_block(params, snapshot)
            elif atype == "place_block":
                result = self._execute_place_block(params, snapshot)
            elif atype == "use_item":
                result = self._execute_use_item(params, snapshot)
            elif atype == "interact":
                result = self._execute_interact(params, snapshot)
            else:
                result = ActionResult(
                    success=False,
                    error="unsupported_action",
                    details={"action_type": atype},
                )
        except Exception as exc:
            # Catch-all for transport / unexpected errors.
            self._log.exception(
                "ActionExecutor.execute raised unexpectedly for type=%s", atype
            )
            return ActionResult(
                success=False,
                error="execution_exception",
                details={
                    "action_type": atype,
                    "exception": repr(exc),
                },
            )

        self._log.debug(
            "ActionExecutor.execute end type=%s success=%s error=%s",
            atype,
            getattr(result, "success", None),
            getattr(result, "error", None),
        )
        return result

    # ------------------------------------------------------------------
    # Action handlers
    # ------------------------------------------------------------------

    def _execute_move_to(
        self,
        params: Mapping[str, Any],
        snapshot: RawWorldSnapshot,
    ) -> ActionResult:
        """
        Move the player to a target coordinate using NavGrid + A*.

        Expected params:
            - "x", "y", "z": numeric target position (block coords)
            - "radius": optional float radius (default from config)

        Guards:
            - max_nav_steps: bounds A* search complexity
            - max_move_steps: bounds number of steps we are willing to send
            - max_move_duration_s: bounds wall-clock sending duration
        """
        try:
            tx = int(params["x"])
            ty = int(params.get("y", snapshot.player_pos.get("y", 0.0)))
            tz = int(params["z"])
        except Exception:
            return ActionResult(
                success=False,
                error="invalid_params",
                details={"reason": "missing_or_non_numeric_xyz", "params": dict(params)},
            )

        radius = float(params.get("radius", self._cfg.default_move_radius))

        grid = NavGrid(snapshot=snapshot, is_solid_block=self._is_solid_block)

        start = current_coord_from_snapshot(snapshot)
        goal = (tx, ty, tz)

        pf_result = find_path(
            grid,
            start=start,
            goal=goal,
            max_steps=self._cfg.max_nav_steps,
        )

        if not pf_result.success or not pf_result.path:
            return ActionResult(
                success=False,
                error="nav_failure",
                details={
                    "from": start,
                    "to": goal,
                    "reason": pf_result.reason,
                },
            )

        move_actions = path_to_actions(
            pf_result,
            snapshot,
            radius=radius,
        )

        steps_planned = len(move_actions)
        max_steps_guard = self._cfg.max_move_steps

        if max_steps_guard is not None and steps_planned > max_steps_guard:
            # Path is conceptually valid but too long for one atomic action.
            return ActionResult(
                success=False,
                error="nav_too_long",
                details={
                    "from": start,
                    "to": goal,
                    "planned_steps": steps_planned,
                    "max_move_steps": max_steps_guard,
                },
            )

        # Emit movement packets for each step with a runtime guard.
        steps_sent = 0
        start_time = perf_counter()
        max_duration = self._cfg.max_move_duration_s

        for step in move_actions:
            # Check runtime timeout before sending the next step.
            if max_duration is not None:
                elapsed = perf_counter() - start_time
                if elapsed > max_duration:
                    return ActionResult(
                        success=False,
                        error="move_timeout",
                        details={
                            "from": start,
                            "to": goal,
                            "steps_attempted": steps_sent,
                            "planned_steps": steps_planned,
                            "max_move_duration_s": max_duration,
                            "elapsed_s": elapsed,
                        },
                    )

            try:
                self._send_move_step(step.params)
                steps_sent += 1
            except Exception as exc:
                return ActionResult(
                    success=False,
                    error="io_error",
                    details={
                        "stage": "move_to",
                        "from": start,
                        "to": goal,
                        "steps_attempted": steps_sent,
                        "planned_steps": steps_planned,
                        "exception": repr(exc),
                    },
                )

        return ActionResult(
            success=True,
            error=None,
            details={
                "from": start,
                "to": goal,
                "steps": steps_sent,
            },
        )

    def _execute_break_block(
        self,
        params: Mapping[str, Any],
        snapshot: RawWorldSnapshot,  # unused but kept for future context
    ) -> ActionResult:
        """
        Break a block at the given coordinates.

        Expected params:
            - "x", "y", "z": block coordinates
            - "face": optional face index or direction (depends on IPC schema)

        Note:
            For now this is synchronous fire-and-forget (start+stop). The
            max_dig_duration_s guard will become relevant once we have async
            feedback from the server / IPC layer.
        """
        try:
            x = int(params["x"])
            y = int(params["y"])
            z = int(params["z"])
        except Exception:
            return ActionResult(
                success=False,
                error="invalid_params",
                details={"reason": "missing_or_non_numeric_xyz", "params": dict(params)},
            )

        face = params.get("face", "auto")

        try:
            self._client.send_packet(
                "block_dig",
                {
                    "status": "start",
                    "x": x,
                    "y": y,
                    "z": z,
                    "face": face,
                },
            )
            self._client.send_packet(
                "block_dig",
                {
                    "status": "stop",
                    "x": x,
                    "y": y,
                    "z": z,
                    "face": face,
                },
            )
        except Exception as exc:
            return ActionResult(
                success=False,
                error="io_error",
                details={
                    "stage": "break_block",
                    "x": x,
                    "y": y,
                    "z": z,
                    "exception": repr(exc),
                },
            )

        return ActionResult(
            success=True,
            error=None,
            details={"x": x, "y": y, "z": z},
        )

    def _execute_place_block(
        self,
        params: Mapping[str, Any],
        snapshot: RawWorldSnapshot,  # unused but reserved for future checks
    ) -> ActionResult:
        """
        Place a block using the currently held item at the target coordinates.

        Expected params:
            - "x", "y", "z": target block position or adjacent position
            - "face": optional face direction
        """
        try:
            x = int(params["x"])
            y = int(params["y"])
            z = int(params["z"])
        except Exception:
            return ActionResult(
                success=False,
                error="invalid_params",
                details={"reason": "missing_or_non_numeric_xyz", "params": dict(params)},
            )

        face = params.get("face", "auto")

        try:
            self._client.send_packet(
                "block_place",
                {
                    "x": x,
                    "y": y,
                    "z": z,
                    "face": face,
                },
            )
        except Exception as exc:
            return ActionResult(
                success=False,
                error="io_error",
                details={
                    "stage": "place_block",
                    "x": x,
                    "y": y,
                    "z": z,
                    "exception": repr(exc),
                },
            )

        return ActionResult(
            success=True,
            error=None,
            details={"x": x, "y": y, "z": z},
        )

    def _execute_use_item(
        self,
        params: Mapping[str, Any],
        snapshot: RawWorldSnapshot,  # reserved for later (line of sight, etc.)
    ) -> ActionResult:
        """
        Use the currently held item (e.g., right-click in air or on a block).

        Expected params (schema is flexible and IPC-defined):
            - "target": one of {"self", "block", "entity", "air"} (optional)
            - "x", "y", "z": optional coordinates (for block/entity)
            - any other keys are passed through to IPC layer.
        """
        payload: Dict[str, Any] = {"target": params.get("target", "air")}
        for key in ("x", "y", "z", "face", "entity_id"):
            if key in params:
                payload[key] = params[key]

        for k, v in params.items():
            if k not in payload:
                payload[k] = v

        try:
            self._client.send_packet("use_item", payload)
        except Exception as exc:
            return ActionResult(
                success=False,
                error="io_error",
                details={
                    "stage": "use_item",
                    "payload": payload,
                    "exception": repr(exc),
                },
            )

        return ActionResult(
            success=True,
            error=None,
            details=payload,
        )

    def _execute_interact(
        self,
        params: Mapping[str, Any],
        snapshot: RawWorldSnapshot,  # reserved (e.g., for range checks)
    ) -> ActionResult:
        """
        Interact with an entity or block.

        Expected params:
            - "kind": "entity" or "block"
            - "entity_id": int if kind == "entity"
            - "x", "y", "z": coords if kind == "block"
        """
        kind = params.get("kind")

        if kind not in ("entity", "block"):
            return ActionResult(
                success=False,
                error="invalid_params",
                details={
                    "reason": "kind_must_be_entity_or_block",
                    "params": dict(params),
                },
            )

        payload: Dict[str, Any] = {"kind": kind}

        if kind == "entity":
            try:
                payload["entity_id"] = int(params["entity_id"])
            except Exception:
                return ActionResult(
                    success=False,
                    error="invalid_params",
                    details={
                        "reason": "missing_or_non_numeric_entity_id",
                        "params": dict(params),
                    },
                )
        else:  # kind == "block"
            try:
                payload["x"] = int(params["x"])
                payload["y"] = int(params["y"])
                payload["z"] = int(params["z"])
            except Exception:
                return ActionResult(
                    success=False,
                    error="invalid_params",
                    details={
                        "reason": "missing_or_non_numeric_xyz",
                        "params": dict(params),
                    },
                )

        for k, v in params.items():
            if k not in payload:
                payload[k] = v

        try:
            self._client.send_packet("interact", payload)
        except Exception as exc:
            return ActionResult(
                success=False,
                error="io_error",
                details={
                    "stage": "interact",
                    "payload": payload,
                    "exception": repr(exc),
                },
            )

        return ActionResult(
            success=True,
            error=None,
            details=payload,
        )

    # ------------------------------------------------------------------
    # Low-level emitters
    # ------------------------------------------------------------------

    def _send_move_step(self, step_params: Mapping[str, Any]) -> None:
        """
        Emit a single movement step.

        For IPC mode, you can interpret this generically:
          - type: "move_step"
          - payload: {x, y, z, radius}
        """
        x = step_params.get("x")
        y = step_params.get("y")
        z = step_params.get("z")
        radius = step_params.get("radius")

        payload = {
            "x": x,
            "y": y,
            "z": z,
            "radius": radius,
        }

        self._client.send_packet("move_step", payload)

