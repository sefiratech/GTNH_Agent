# src/bot_core/tracing.py
"""
Tracing and metrics for bot_core_1_7_10.

This module provides a thin, structured logging layer around action
execution so that higher modules (M9 monitoring, M10 experience) can
consume consistent traces.

It does NOT:
- Call LLMs
- Infer semantics
- Make control decisions
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, List, Optional

from spec.types import Action, ActionResult
from .snapshot import RawWorldSnapshot


@dataclass
class ActionTraceRecord:
    """
    Structured record of a single action execution.

    Fields are intentionally generic and stable so that M9/M10 can build
    on top of this without depending on internal details of bot_core.
    """

    timestamp: float           # wall-clock time (time.time())
    duration_s: float          # execution duration in seconds

    action_type: Optional[str]
    params: dict[str, Any]

    success: bool
    error: Optional[str]

    # Minimal world context at decision time
    tick: Optional[int]
    dimension: Optional[str]
    position: dict[str, float]


class ActionTracer:
    """
    In-memory action tracer with optional logging.

    Responsibilities:
    - Keep a rolling buffer of recent ActionTraceRecord entries.
    - Emit a single structured log line per action (info level).

    This is deliberately minimal; M9/M10 can later:
      - consume tracer records
      - or replace this with a more complex sink.
    """

    def __init__(
        self,
        *,
        logger: Optional[logging.Logger] = None,
        max_records: int = 10_000,
    ) -> None:
        self._logger = logger or logging.getLogger("bot_core.action")
        self._records: Deque[ActionTraceRecord] = deque(maxlen=max_records)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record(
        self,
        *,
        action: Action,
        snapshot: RawWorldSnapshot,
        result: ActionResult,
        duration_s: float,
    ) -> None:
        """
        Record a trace for a completed action.

        This should be called even on failures; `success` and `error`
        capture outcome.
        """
        try:
            record = self._build_record(action, snapshot, result, duration_s)
        except Exception:
            # Tracing must never crash the caller.
            self._logger.exception("Failed to build ActionTraceRecord")
            return

        self._records.append(record)

        # Emit a structured log line. Keep it compact but informative.
        self._logger.info(
            "action_exec type=%s success=%s error=%s duration=%.4fs tick=%s "
            "dim=%s pos=(%.2f,%.2f,%.2f)",
            record.action_type,
            record.success,
            record.error,
            record.duration_s,
            record.tick,
            record.dimension,
            record.position.get("x", 0.0),
            record.position.get("y", 0.0),
            record.position.get("z", 0.0),
        )

    def get_records(self) -> List[ActionTraceRecord]:
        """
        Return a snapshot of all currently buffered records.

        Intended for debugging / M9 / M10; not performance-critical.
        """
        return list(self._records)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_record(
        self,
        action: Action,
        snapshot: RawWorldSnapshot,
        result: ActionResult,
        duration_s: float,
    ) -> ActionTraceRecord:
        # Extract minimal context
        tick = snapshot.tick
        dimension = snapshot.dimension
        pos = {
            "x": float(snapshot.player_pos.get("x", 0.0)),
            "y": float(snapshot.player_pos.get("y", 0.0)),
            "z": float(snapshot.player_pos.get("z", 0.0)),
        }

        # Params can be large; use a shallow copy to avoid surprises.
        params = {}
        raw_params = getattr(action, "params", {}) or {}
        if isinstance(raw_params, dict):
            params = dict(raw_params)

        return ActionTraceRecord(
            timestamp=time.time(),
            duration_s=duration_s,
            action_type=getattr(action, "type", None),
            params=params,
            success=bool(getattr(result, "success", False)),
            error=getattr(result, "error", None),
            tick=tick,
            dimension=dimension,
            position=pos,
        )

