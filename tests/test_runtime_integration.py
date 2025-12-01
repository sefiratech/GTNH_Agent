# path: tests/test_runtime_integration.py

"""
Integration tests for M8 (AgentLoop) + M9 (monitoring & tools).

Covers:
- EventBus + JsonFileLogger end-to-end.
- AgentController driving a fake AgentLoopControl implementation.
- Emitting a small set of MonitoringEvents that form a coherent "episode".
- Verifying that the JSONL log contains consistent, structured data.

This does NOT require:
- Minecraft
- Real LLMs
- Real AgentLoopV1 implementation
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from monitoring.bus import EventBus
from monitoring.controller import AgentController
from monitoring.events import (
    ControlCommand,
    ControlCommandType,
    EventType,
    MonitoringEvent,
)
from monitoring.logger import JsonFileLogger, log_event


class FakeAgentLoop:
    """
    Minimal fake implementing the AgentLoopControl protocol.

    Behaviors:
    - `step()` increments a counter.
    - `cancel_current_plan()` toggles a flag.
    - `set_goal()` stores the last goal.
    - `debug_state()` returns a JSON-safe dict.
    """

    def __init__(self) -> None:
        self.step_count: int = 0
        self.cancelled: bool = False
        self.goals: List[str] = []

    def step(self) -> None:
        self.step_count += 1

    def cancel_current_plan(self) -> None:
        self.cancelled = True

    def set_goal(self, goal: str) -> None:
        self.goals.append(goal)

    def debug_state(self) -> Dict[str, Any]:
        # Return a JSON-safe diagnostic snapshot
        return {
            "phase": "EXECUTING",
            "current_plan": {"id": "plan-ep1", "steps": [1, 2, 3]},
            "step_count": self.step_count,
            "tech_state": {"tier": "LV"},
            "episode_id": "ep-1",
        }


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """
    Helper: read a JSONL file into a list of dicts.
    """
    if not path.exists():
        return []
    items: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            items.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return items


def test_m8_m9_integration_basic_episode(tmp_path: Path) -> None:
    """
    End-to-end integration test:

    - Build EventBus and JsonFileLogger.
    - Wrap a FakeAgentLoop with AgentController.
    - Emit events that represent a simple episode:
        - phase change
        - plan created
        - one step executed
        - virtue scores
        - critic result
    - Drive controller with a few ControlCommand messages.
    - Assert the JSONL log contains a coherent episode for episode_id "ep-1".
    """
    # Arrange: event bus + logger
    bus = EventBus()
    log_path = tmp_path / "events.log"
    logger = JsonFileLogger(log_path, bus)

    # Arrange: fake agent + controller
    agent = FakeAgentLoop()
    controller = AgentController(agent=agent, bus=bus)

    # Episode identifiers
    episode_id = "ep-1"
    context_id = "test-context-1"

    # Act: emit a phase change
    log_event(
        bus=bus,
        module="M8.agent_loop",
        event_type=EventType.AGENT_PHASE_CHANGE,
        message="Agent entering PLANNING",
        payload={"phase": "PLANNING", "episode_id": episode_id, "context_id": context_id},
        correlation_id=episode_id,
    )

    # Act: emit a plan created event
    plan = {"id": "plan-ep1", "steps": [{"idx": 0}, {"idx": 1}, {"idx": 2}]}
    log_event(
        bus=bus,
        module="M8.agent_loop",
        event_type=EventType.PLAN_CREATED,
        message="New plan created",
        payload={
            "plan": plan,
            "goal": "Automate LV steam",
            "step_count": 3,
            "episode_id": episode_id,
            "context_id": context_id,
        },
        correlation_id=episode_id,
    )

    # Act: simulate one executed step
    log_event(
        bus=bus,
        module="M8.agent_loop",
        event_type=EventType.PLAN_STEP_EXECUTED,
        message="Executed step 0",
        payload={
            "episode_id": episode_id,
            "context_id": context_id,
            "step_index": 0,
            "step_spec": {"idx": 0},
            "trace_step": {"status": "ok"},
        },
        correlation_id=episode_id,
    )

    # Act: virtue scores for this episode
    log_event(
        bus=bus,
        module="M4.virtues",
        event_type=EventType.VIRTUE_SCORES,
        message="Virtue scores computed",
        payload={
            "episode_id": episode_id,
            "context_id": context_id,
            "scores": {"prudence": 0.9, "temperance": 0.7},
        },
        correlation_id=episode_id,
    )

    # Act: critic result summary
    log_event(
        bus=bus,
        module="M8.agent_loop",
        event_type=EventType.CRITIC_RESULT,
        message="Critic evaluation completed",
        payload={
            "episode_id": episode_id,
            "context_id": context_id,
            "critic_result": {"summary": "Good plan, watch resources"},
        },
        correlation_id=episode_id,
    )

    # Act: basic control interactions
    bus.publish_command(ControlCommand(cmd=ControlCommandType.SET_GOAL, args={"goal": "Automate LV steam"}))
    bus.publish_command(ControlCommand(cmd=ControlCommandType.SINGLE_STEP, args={}))
    controller.maybe_step_agent()  # should perform one step and then pause again
    bus.publish_command(ControlCommand(cmd=ControlCommandType.DUMP_STATE, args={}))

    # Cleanup / flush
    logger.close()

    # Assert: agent received commands
    assert agent.goals == ["Automate LV steam"]
    assert agent.step_count == 1

    # Assert: log file exists and has multiple events
    events = _read_jsonl(log_path)
    assert len(events) >= 5, "Expected multiple monitoring events in JSONL log"

    # Filter down to just this episode_id
    ep_events = [e for e in events if e.get("payload", {}).get("episode_id") == episode_id]
    assert ep_events, "Expected events associated with episode_id ep-1"

    # Check that we have at least one of each important type
    types = {e["event_type"] for e in ep_events}
    assert "AGENT_PHASE_CHANGE" in types
    assert "PLAN_CREATED" in types
    assert "PLAN_STEP_EXECUTED" in types
    assert "VIRTUE_SCORES" in types
    assert "CRITIC_RESULT" in types

    # Check that correlation_id is consistent across the main episode events
    corr_ids = {e.get("correlation_id") for e in ep_events}
    assert corr_ids == {episode_id}, "Expected a single coherent correlation_id for the episode"

