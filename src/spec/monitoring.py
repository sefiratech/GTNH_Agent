# src/spec/monitoring.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Any, Dict, Optional


# ---------------------------------------------------------------------------
# Event types
# ---------------------------------------------------------------------------

@dataclass
class GoalSelectedEvent:
    """Emitted when a new AgentGoal is chosen for an episode."""
    goal_id: str
    source: str          # e.g. "curriculum", "manual", "fallback"
    phase: str           # e.g. "P1_steam_age"
    episode_id: Optional[int] = None


@dataclass
class TaskPlannedEvent:
    """Emitted after TaskPlanning finishes for a goal."""
    goal_id: str
    task_count: int
    episode_id: Optional[int] = None


@dataclass
class SkillPlanGeneratedEvent:
    """Emitted after SkillResolution finishes for a single Task."""
    goal_id: str
    task_id: str
    skill_count: int
    episode_id: Optional[int] = None


# ---------------------------------------------------------------------------
# Event sink interface (M9)
# ---------------------------------------------------------------------------

class EventSink(Protocol):
    """
    Generic monitoring / event sink.

    Implementations might:
      - log to stdout / logging
      - append to a file
      - push to a metrics backend
      - collect in-memory for tests
    """

    def emit(self, event: Any) -> None:
        """
        Consume a single event object.

        Implementations are expected to handle unknown event types gracefully.
        """
        ...


def event_to_dict(event: Any) -> Dict[str, Any]:
    """
    Best-effort conversion of an event dataclass into a dict for logging.

    This helper is safe to use in logging-only sinks and tests.
    """
    if hasattr(event, "__dict__"):
        return dict(event.__dict__)
    try:
        from dataclasses import asdict
        return asdict(event)  # type: ignore[arg-type]
    except Exception:
        return {"repr": repr(event)}

