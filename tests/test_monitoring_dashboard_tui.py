#tests/test_monitoring_dashboard_tui.py
"""
Smoke tests for monitoring.dashboard_tui.TuiDashboard.

Covers:
- Layout builds cleanly
- Event updates patch internal state
- Rendering functions do not crash
"""

from __future__ import annotations

from monitoring.bus import EventBus
from monitoring.dashboard_tui import TuiDashboard
from monitoring.events import EventType, MonitoringEvent


def make_event(event_type: EventType, payload: dict) -> MonitoringEvent:
    return MonitoringEvent(
        ts=0.0,
        module="test",
        event_type=event_type,
        message="",
        payload=payload,
        correlation_id=None,
    )


def test_dashboard_handles_basic_events_and_renders():
    bus = EventBus()
    dashboard = TuiDashboard(bus)

    # Phase change
    bus.publish(make_event(EventType.AGENT_PHASE_CHANGE, {"phase": "PLANNING"}))

    # Plan created
    bus.publish(
        make_event(
            EventType.PLAN_CREATED,
            {
                "goal": "Build coke ovens",
                "plan": {"id": "plan-1", "steps": [1, 2, 3]},
                "step_count": 3,
            },
        )
    )

    # Plan step executed
    bus.publish(
        make_event(
            EventType.PLAN_STEP_EXECUTED,
            {"step_index": 1, "step_spec": {}, "trace_step": {}},
        )
    )

    # Tech state update
    bus.publish(
        make_event(
            EventType.TECH_STATE_UPDATED,
            {
                "tech_state": {
                    "tier": "LV",
                    "missing_unlocks": ["Basic Steam Turbine"],
                    "traits": {"energy": "steam"},
                }
            },
        )
    )

    # Virtue scores
    bus.publish(
        make_event(
            EventType.VIRTUE_SCORES,
            {"scores": {"prudence": 0.9, "temperance": 0.7}},
        )
    )

    # Plan failure
    bus.publish(
        make_event(
            EventType.PLAN_FAILED,
            {"reason": "Out of steam", "step_index": 2},
        )
    )

    # Critic result (optional)
    bus.publish(
        make_event(
            EventType.CRITIC_RESULT,
            {"critic_result": {"summary": "Good plan, bad resources"}},
        )
    )

    # Now try building the layout; it should not throw.
    layout = dashboard._build_layout()  # type: ignore[attr-defined]
    assert layout is not None
