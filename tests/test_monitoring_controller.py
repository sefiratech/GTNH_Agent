#tests/test_monitoring_controller.py
"""
Tests for monitoring.controller.AgentController.

Covers:
- pause/resume semantics
- single-step correctness
- plan cancellation
- goal setting
- DUMP_STATE wiring (basic)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from monitoring.bus import EventBus
from monitoring.controller import AgentController
from monitoring.events import (
    ControlCommand,
    ControlCommandType,
    EventType,
    MonitoringEvent,
)


@dataclass
class FakeAgent:
    """
    Simple fake AgentLoop implementing the AgentLoopControl protocol.
    """
    calls: Dict[str, Any]

    def __init__(self) -> None:
        self.calls = {
            "step": 0,
            "cancel_current_plan": 0,
            "set_goal": [],
            "debug_state": 0,
        }

    def step(self) -> None:
        self.calls["step"] += 1

    def cancel_current_plan(self) -> None:
        self.calls["cancel_current_plan"] += 1

    def set_goal(self, goal: str) -> None:
        self.calls["set_goal"].append(goal)

    def debug_state(self) -> Dict[str, Any]:
        self.calls["debug_state"] += 1
        return {
            "phase": "TEST",
            "plan": {"id": "plan-xyz", "steps": []},
            "tech_state": {"tier": "LV"},
        }


def test_controller_pause_resume_and_single_step():
    bus = EventBus()
    agent = FakeAgent()
    controller = AgentController(agent, bus)

    # Initially not paused, not single step
    assert controller.paused is False

    # PAUSE -> maybe_step_agent should do nothing
    bus.publish_command(ControlCommand(cmd=ControlCommandType.PAUSE, args={}))
    assert controller.paused is True
    controller.maybe_step_agent()
    assert agent.calls["step"] == 0

    # RESUME -> normal stepping
    bus.publish_command(ControlCommand(cmd=ControlCommandType.RESUME, args={}))
    assert controller.paused is False
    controller.maybe_step_agent()
    assert agent.calls["step"] == 1

    # SINGLE_STEP -> one step, then back to paused
    bus.publish_command(ControlCommand(cmd=ControlCommandType.SINGLE_STEP, args={}))
    assert controller.paused is False
    controller.maybe_step_agent()
    assert agent.calls["step"] == 2
    # After single step, should be paused again
    assert controller.paused is True


def test_controller_cancel_plan_and_set_goal_and_dump_state():
    bus = EventBus()
    agent = FakeAgent()
    controller = AgentController(agent, bus)

    # Capture monitoring events for inspection
    received: List[MonitoringEvent] = []

    def subscriber(evt: MonitoringEvent) -> None:
        received.append(evt)

    bus.subscribe(subscriber)

    # CANCEL_PLAN
    bus.publish_command(ControlCommand(cmd=ControlCommandType.CANCEL_PLAN, args={}))
    assert agent.calls["cancel_current_plan"] == 1

    # SET_GOAL
    bus.publish_command(
        ControlCommand(cmd=ControlCommandType.SET_GOAL, args={"goal": "Automate LV base"})
    )
    assert agent.calls["set_goal"] == ["Automate LV base"]

    # DUMP_STATE
    bus.publish_command(ControlCommand(cmd=ControlCommandType.DUMP_STATE, args={}))
    assert agent.calls["debug_state"] == 1

    # There should be at least one SNAPSHOT event from controller
    snapshot_events = [e for e in received if e.event_type == EventType.SNAPSHOT]
    assert snapshot_events, "Expected at least one SNAPSHOT event"

    # And the payload should have a 'state'
    latest = snapshot_events[-1]
    assert "state" in latest.payload
    assert latest.payload["state"]["phase"] == "TEST"
