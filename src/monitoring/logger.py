# JSON logger subscribing to EventBus
"""
Structured logging for M9 – monitoring_and_tools.

Provides:
- JsonFileLogger: subscribes to an EventBus and writes MonitoringEvents as JSONL.
- log_event: convenience helper for publishing MonitoringEvents via the EventBus.

Usage patterns:

    from pathlib import Path
    from monitoring.bus import EventBus
    from monitoring.logger import JsonFileLogger, log_event
    from monitoring.events import EventType

    bus = EventBus()
    logger = JsonFileLogger(Path("logs/monitoring/events.log"), bus)

    log_event(
        bus=bus,
        module="example.module",
        event_type=EventType.LOG,
        message="Something happened",
        payload={"foo": "bar"},
    )
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

from .bus import EventBus
from .events import EventType, MonitoringEvent


# ============================================================
# JSONL File Logger
# ============================================================

class JsonFileLogger:
    """
    JSON-lines logger for MonitoringEvent instances.

    - Subscribes to an EventBus and writes one JSON object per line.
    - Ensures UTF-8 encoding.
    - Ensures parent directory exists.
    """

    def __init__(self, path: Path, bus: EventBus) -> None:
        """
        Initialize the logger and subscribe to the event bus.

        Parameters
        ----------
        path:
            Path to the log file (e.g. logs/monitoring/events.log).
        bus:
            EventBus instance to subscribe to.
        """
        self._path = path
        self._ensure_parent_dir(path)
        # Open file in append mode
        self._file = path.open("a", encoding="utf-8")
        # Subscribe as a MonitoringEvent consumer
        bus.subscribe(self._on_event)

    @staticmethod
    def _ensure_parent_dir(path: Path) -> None:
        """
        Create parent directories for `path` if they don't exist.
        """
        parent = path.parent
        if not parent.exists():
            parent.mkdir(parents=True, exist_ok=True)

    def _on_event(self, event: MonitoringEvent) -> None:
        """
        Callback invoked for each MonitoringEvent published on the EventBus.
        Writes the event as a JSON object to the log file.
        """
        data = event.to_dict()
        line = json.dumps(data, ensure_ascii=False)
        try:
            self._file.write(line + "\n")
            self._file.flush()
        except Exception:
            # Logging must not crash the process.
            # If disk is full or handle is invalid, we silently drop.
            # You can add a fallback stderr logger here if desired.
            pass

    def close(self) -> None:
        """
        Close the underlying file handle.

        Should be called at graceful shutdown.
        """
        try:
            self._file.close()
        except Exception:
            pass


# ============================================================
# Convenience helper for emitting events
# ============================================================

def log_event(
    bus: EventBus,
    module: str,
    event_type: EventType,
    message: str,
    payload: Optional[Dict[str, Any]] = None,
    correlation_id: Optional[str] = None,
) -> None:
    """
    Convenience function to create and publish a MonitoringEvent.

    Intended usage in other modules:

        log_event(
            bus=self._bus,
            module="M8.agent_loop",
            event_type=EventType.PLAN_CREATED,
            message="New plan created",
            payload={"plan": plan_dict, "goal": self._goal},
            correlation_id=plan_id,
        )

    Parameters
    ----------
    bus:
        EventBus instance to publish the event to.
    module:
        String identifying the source module ("M8.agent_loop", "bot_core", "llm_stack").
    event_type:
        EventType enum member describing what kind of event this is.
    message:
        Short human-readable description.
    payload:
        Structured JSON-safe data attached to this event.
    correlation_id:
        Optional ID linking related events (per-episode, per-plan, etc.).
    """
    event = MonitoringEvent(
        ts=time.time(),
        module=module,
        event_type=event_type,
        message=message,
        payload=payload or {},
        correlation_id=correlation_id,
    )
    bus.publish(event)

