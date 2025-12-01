# path: tests/test_failure_mitigation.py

"""
Unit tests for runtime.failure_mitigation helpers.

These tests verify that the helpers:
- Emit events with the correct event_type.
- Carry the expected subtype and payload fields.
"""

from __future__ import annotations

from typing import Any, Dict, List  # type hints

from monitoring.bus import EventBus          # event bus
from monitoring.events import EventType      # event types
from monitoring.events import MonitoringEvent  # event dataclass

from runtime.failure_mitigation import (
    emit_config_error,
    emit_model_path_error,
    emit_llm_failure,
    emit_action_failure,
    emit_plan_failed_due_to_actions,
    emit_monitoring_overload_warning,
)


def _capture_events(bus: EventBus) -> List[MonitoringEvent]:
    """
    Subscribe a collector to the bus and return the underlying list.
    """
    captured: List[MonitoringEvent] = []

    def subscriber(evt: MonitoringEvent) -> None:
        captured.append(evt)

    bus.subscribe(subscriber)
    return captured


def test_emit_config_error_and_model_path_error():
    bus = EventBus()
    captured = _capture_events(bus)

    emit_config_error(
        bus=bus,
        message="Failed to load env profile",
        env_profile_repr="EnvProfile(name='test')",
        error_repr="ValueError('oops')",
    )

    emit_model_path_error(
        bus=bus,
        message="Model path does not exist",
        model_id="planner-model",
        expected_path="/models/planner.bin",
        error_repr=None,
    )

    assert len(captured) == 2

    cfg_evt, model_evt = captured

    assert cfg_evt.event_type == EventType.LOG
    assert cfg_evt.payload["subtype"] == "CONFIG_ERROR"
    assert "EnvProfile(name='test')" in cfg_evt.payload["env_profile"]

    assert model_evt.event_type == EventType.LOG
    assert model_evt.payload["subtype"] == "MODEL_PATH_ERROR"
    assert model_evt.payload["model_id"] == "planner-model"
    assert model_evt.payload["expected_path"] == "/models/planner.bin"


def test_emit_llm_failure():
    bus = EventBus()
    captured = _capture_events(bus)

    emit_llm_failure(
        bus=bus,
        subtype="LLM_TIMEOUT",
        role="planner",
        model="test-model",
        episode_id="ep-123",
        context_id="ctx-1",
        error_repr="TimeoutError('boom')",
        meta={"timeout_s": 5.0},
    )

    assert len(captured) == 1
    evt = captured[0]

    assert evt.event_type == EventType.LOG
    assert evt.payload["subtype"] == "LLM_TIMEOUT"
    assert evt.payload["llm_role"] == "planner"
    assert evt.payload["model"] == "test-model"
    assert evt.payload["episode_id"] == "ep-123"
    assert evt.payload["context_id"] == "ctx-1"
    assert evt.payload["meta"]["timeout_s"] == 5.0
    assert evt.correlation_id == "ep-123"


def test_emit_action_failure_and_plan_failed_due_to_actions():
    bus = EventBus()
    captured = _capture_events(bus)

    emit_action_failure(
        bus=bus,
        episode_id="ep-abc",
        context_id="ctx-xyz",
        action_name="move_to",
        action_args={"pos": [0, 64, 0]},
        error_repr="IOError('no path')",
    )

    failing_actions: List[Dict[str, Any]] = [
        {"action": "move_to", "error": "path_blocked"},
        {"action": "move_to", "error": "path_blocked"},
    ]

    emit_plan_failed_due_to_actions(
        bus=bus,
        episode_id="ep-abc",
        context_id="ctx-xyz",
        reason="Too many failed movement attempts",
        failing_actions=failing_actions,
    )

    assert len(captured) == 2

    action_evt, plan_evt = captured

    # First event: ACTION_EXECUTED failure
    assert action_evt.event_type == EventType.ACTION_EXECUTED
    assert action_evt.payload["success"] is False
    assert action_evt.payload["action_name"] == "move_to"
    assert action_evt.payload["error"].startswith("IOError")

    # Second event: PLAN_FAILED escalation
    assert plan_evt.event_type == EventType.PLAN_FAILED
    assert plan_evt.payload["reason"] == "Too many failed movement attempts"
    assert len(plan_evt.payload["failing_actions"]) == 2
    assert plan_evt.correlation_id == "ep-abc"


def test_emit_monitoring_overload_warning():
    bus = EventBus()
    captured = _capture_events(bus)

    emit_monitoring_overload_warning(
        bus=bus,
        approx_rate_hz=123.45,
        context_id="ctx-overload",
    )

    assert len(captured) == 1
    evt = captured[0]

    assert evt.event_type == EventType.LOG
    assert evt.payload["subtype"] == "MONITORING_OVERLOAD"
    assert evt.payload["approx_rate_hz"] == 123.45
    assert evt.payload["context_id"] == "ctx-overload"

