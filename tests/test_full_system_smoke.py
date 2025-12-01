# path: tests/test_full_system_smoke.py

"""
Full system–style smoke test for GTNH_Agent Phases 0–3.

This test does NOT talk to:
- Real Minecraft
- Real LLM backends
- Real semantics/virtues/skills

Instead, it:
- Builds an EventBus + JsonFileLogger (M9).
- Wraps a FakeAgentLoop in AgentController (M9, M8 interface).
- Uses safe_step_with_logging (runtime error handling).
- Simulates a small "episode" by emitting:
    - AGENT_PHASE_CHANGE
    - PLAN_CREATED
    - PLAN_STEP_EXECUTED
- Asserts that:
    - Logs are written.
    - Event types and correlation_id form a coherent episode.

Conceptually, this stands in for a full stack:
- Phase 0: DummyEnvProfile (not explicitly exercised here).
- Phase 1: LLM / semantics / virtues / skills represented by no-op attributes.
- Phase 2: BotCore + ObservationEncoder represented by no-op attributes.
- Phase 3: AgentLoop + monitoring & tools fully exercised on the logging side.
"""

from __future__ import annotations

import json                           # to parse JSONL file content
from pathlib import Path              # for temp log path
from typing import Any, Dict, List    # for type hints

from monitoring.bus import EventBus              # core event bus (M9)
from monitoring.logger import JsonFileLogger, log_event  # JSONL logger + helper (M9)
from monitoring.controller import AgentController        # control surface (M9)
from monitoring.events import (
    EventType,
)                                              # event type enum (M9 schema)
from runtime.error_handling import safe_step_with_logging  # runtime helper


class FakeAgentLoop:
    """
    Minimal fake AgentLoop implementing AgentLoopControl protocol.

    This stands in for the fully-wired AgentLoopV1 (M8) with Phase 0–2
    dependencies hidden behind simple attributes.

    Behavior:
    - `step()` increments a counter.
    - On the first step, emits AGENT_PHASE_CHANGE + PLAN_CREATED.
    - On subsequent steps, emits PLAN_STEP_EXECUTED events.
    - `cancel_current_plan()` toggles a flag.
    - `set_goal()` records the last goal.
    - `debug_state()` returns a JSON-safe state snapshot used by DUMP_STATE.
    """

    def __init__(self, bus: EventBus) -> None:
        # Monitoring bus used to emit events
        self._bus = bus
        # Count how many steps have been executed
        self.step_count: int = 0
        # Track whether the current plan has been cancelled
        self.cancelled: bool = False
        # Track goals that have been set
        self.goals: List[str] = []
        # Fixed IDs to mimic an episode
        self.episode_id: str = "ep-full-1"
        self.context_id: str = "test-context-full"

    # --- AgentLoopControl protocol methods -----------------------------------

    def step(self) -> None:
        """
        Simulate one iteration of the agent loop.

        On the first step:
        - Emits AGENT_PHASE_CHANGE and PLAN_CREATED.

        On later steps:
        - Emits PLAN_STEP_EXECUTED for the next step index.
        """
        # Increment internal step counter
        self.step_count += 1

        # On first step, mark planning and create a new plan
        if self.step_count == 1:
            # Phase change: entering PLANNING
            log_event(
                bus=self._bus,
                module="M8.agent_loop",
                event_type=EventType.AGENT_PHASE_CHANGE,
                message="Agent entering PLANNING (fake loop)",
                payload={
                    "phase": "PLANNING",
                    "episode_id": self.episode_id,
                    "context_id": self.context_id,
                },
                correlation_id=self.episode_id,
            )

            # Plan creation (fake plan with 3 steps)
            fake_plan = {"id": "plan-full-1", "steps": [0, 1, 2]}
            log_event(
                bus=self._bus,
                module="M8.agent_loop",
                event_type=EventType.PLAN_CREATED,
                message="Fake plan created",
                payload={
                    "plan": fake_plan,
                    "goal": "Fake goal: bootstrap full system",
                    "step_count": 3,
                    "episode_id": self.episode_id,
                    "context_id": self.context_id,
                },
                correlation_id=self.episode_id,
            )
        else:
            # For subsequent steps, emit PLAN_STEP_EXECUTED
            step_index = self.step_count - 2  # 0-based index for executed steps
            log_event(
                bus=self._bus,
                module="M8.agent_loop",
                event_type=EventType.PLAN_STEP_EXECUTED,
                message=f"Fake plan step executed (index={step_index})",
                payload={
                    "episode_id": self.episode_id,
                    "context_id": self.context_id,
                    "step_index": step_index,
                    "step_spec": {"idx": step_index},
                    "trace_step": {"status": "ok"},
                },
                correlation_id=self.episode_id,
            )

    def cancel_current_plan(self) -> None:
        """
        Mark the current plan as cancelled.
        """
        self.cancelled = True

    def set_goal(self, goal: str) -> None:
        """
        Record a new high-level goal.
        """
        self.goals.append(goal)

    def debug_state(self) -> Dict[str, Any]:
        """
        Return a JSON-safe snapshot of internal state.

        Used by AgentController when handling DUMP_STATE to emit a SNAPSHOT event.
        """
        return {
            "phase": "EXECUTING" if self.step_count > 0 else "IDLE",
            "current_plan": {
                "id": "plan-full-1",
                "steps": [0, 1, 2],
            },
            "step_count": self.step_count,
            "tech_state": {"tier": "LV"},
            "episode_id": self.episode_id,
            "context_id": self.context_id,
        }


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """
    Helper: read a JSONL file and return a list of JSON dicts.
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
            # Skip malformed lines; shouldn't normally happen under JsonFileLogger
            continue
    return items


def test_full_system_style_smoke(tmp_path: Path) -> None:
    """
    Full system–style smoke test:

    Steps:
    - Create an EventBus + JsonFileLogger.
    - Create a FakeAgentLoop bound to the bus.
    - Wrap it in AgentController.
    - Call safe_step_with_logging() a few times to simulate the main loop.
    - Verify that:
        - The log file exists and is non-empty.
        - There is a coherent episode with:
            - AGENT_PHASE_CHANGE
            - PLAN_CREATED
            - PLAN_STEP_EXECUTED
        - correlation_id == episode_id across those events.
    """
    # Arrange: create a fresh EventBus
    bus = EventBus()

    # Arrange: pick a log file under pytest's tmp_path
    log_path = tmp_path / "events_full_system.log"

    # Arrange: create a JsonFileLogger subscribed to the bus
    logger = JsonFileLogger(path=log_path, bus=bus)

    # Arrange: create a fake agent loop and wrap it with AgentController
    agent_loop = FakeAgentLoop(bus=bus)
    controller = AgentController(agent=agent_loop, bus=bus)

    # Act: simulate a few iterations of the main runtime loop
    # We don't run forever; just enough to:
    # - emit AGENT_PHASE_CHANGE + PLAN_CREATED on first step
    # - emit a couple of PLAN_STEP_EXECUTED events
    for _ in range(4):
        safe_step_with_logging(
            controller=controller,
            bus=bus,
            episode_id=agent_loop.episode_id,
            context_id=agent_loop.context_id,
        )

    # Cleanup: close logger to flush all events to disk
    logger.close()

    # Assert: log file exists and contains data
    events = _read_jsonl(log_path)
    assert events, "Expected monitoring events in JSONL log for full-system smoke test"

    # Filter events belonging to our fake episode
    ep_events = [
        e for e in events
        if e.get("payload", {}).get("episode_id") == agent_loop.episode_id
    ]
    assert ep_events, "Expected events tagged with the fake episode_id"

    # Extract event types and correlation_ids from those events
    types = {e["event_type"] for e in ep_events}
    corr_ids = {e.get("correlation_id") for e in ep_events}

    # We expect a coherent episode with:
    # - AGENT_PHASE_CHANGE
    # - PLAN_CREATED
    # - one or more PLAN_STEP_EXECUTED
    assert "AGENT_PHASE_CHANGE" in types
    assert "PLAN_CREATED" in types
    assert "PLAN_STEP_EXECUTED" in types

    # All events for this episode should share the same correlation_id
    assert corr_ids == {agent_loop.episode_id}, (
        "Expected all episode events to share a single correlation_id matching episode_id"
    )

