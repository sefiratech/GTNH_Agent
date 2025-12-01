# path: src/runtime/error_handling.py

"""
Error handling helpers for the GTNH agent runtime.

Implements the "Potential failure points & mitigation" suggestions:
- Wrap agent stepping in a guard that logs AGENT_STEP_EXCEPTION events.
- Keep the monitoring/logging behavior consistent with the rest of M9.

This does NOT change core semantics of AgentController; it wraps calls
from your runtime loop.
"""

from __future__ import annotations

from typing import Optional

from monitoring.bus import EventBus
from monitoring.events import EventType
from monitoring.logger import log_event
from monitoring.controller import AgentController


def safe_step_with_logging(
    controller: AgentController,
    bus: EventBus,
    episode_id: Optional[str] = None,
    context_id: Optional[str] = None,
) -> None:
    """
    Call controller.maybe_step_agent() inside a try/except block.

    If agent.step() (inside the controller) throws, we:
    - Emit a LOG event with subtype "AGENT_STEP_EXCEPTION".
    - Re-raise the exception so the runtime can decide whether to abort
      or continue.

    This keeps the monitoring layer informed without silently hiding failures.
    """
    try:
        # Normal stepping: AgentController will decide whether to run a step
        controller.maybe_step_agent()
    except Exception as exc:
        # Emit a structured monitoring event describing the failure
        log_event(
            bus=bus,
            module="runtime.safe_step",
            event_type=EventType.LOG,
            message="Agent step raised an exception",
            payload={
                "subtype": "AGENT_STEP_EXCEPTION",
                "episode_id": episode_id,
                "context_id": context_id,
                "exception_repr": repr(exc),
            },
            correlation_id=episode_id,
        )
        # Re-raise so the caller decides how fatal this is
        raise

