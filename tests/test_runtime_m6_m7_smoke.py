# tests/test_runtime_m6_m7_smoke.py
"""
Runtime smoke tests for M6 + M7 integration.

Covers:
  - DummyBotCore (M6 stub)
  - DummyPlannerModel (M2 stub)
  - DummyCriticModel (M2 stub)
  - AgentRuntime (Phase 2 integration shell)

We assert:
  - planner_tick() returns a plan dict with expected structure
  - get_latest_planner_observation() returns a valid Observation
  - evaluate_trace() returns a critic evaluation dict
"""

from typing import Dict, Any

import pytest

from agent.runtime_m6_m7 import (  # type: ignore[import]
    AgentRuntime,
    AgentRuntimeConfig,
)
from observation.pipeline import (  # Protocols only
    BotCore,
    PlannerModel,
    CriticModel,
)
from observation.testing import DummySemanticsDB, make_minimal_tech_state
from observation.trace_schema import TraceStep, PlanTrace
from semantics.schema import TechState
from spec.types import WorldState, Observation, Action, ActionResult
from bot_core.snapshot import RawWorldSnapshot  # type: ignore[import]


# ---------------------------------------------------------------------------
# Dummy implementations for smoke tests
# ---------------------------------------------------------------------------


class DummyBotCore:
    """
    Very small fake BotCore (M6 stub).

    Returns a fixed RawWorldSnapshot with minimal but valid structure.
    """

    def observe(self) -> RawWorldSnapshot:
        return RawWorldSnapshot(
            tick=123,
            dimension="overworld",
            player_pos={"x": 0.0, "y": 64.0, "z": 0.0},
            player_yaw=0.0,
            player_pitch=0.0,
            on_ground=True,
            chunks={},
            # Use dict entities so we don't depend on RawEntity's constructor
            entities=[
                {
                    "entity_id": 1,
                    "type": "item",
                    "x": 2.0,
                    "y": 64.0,
                    "z": 2.0,
                    "data": {"hostile": False},
                }
            ],
            inventory=[
                {"item_id": "minecraft:log", "variant": None, "count": 16},
            ],
            context={"machines": []},
        )


class DummyPlannerModel(PlannerModel):
    """
    Dummy planner (M2 stub).

    It just inspects the Observation payload and returns a trivial plan dict.
    """

    def plan(self, observation: Observation) -> Dict[str, Any]:
        payload = observation.json_payload
        return {
            "kind": "dummy_plan",
            "source_dim": payload.get("agent", {}).get("dimension"),
            "has_inventory_summary": "inventory_summary" in payload,
        }


class DummyCriticModel(CriticModel):
    """
    Dummy critic (M2 stub).

    It ignores almost everything and returns a simple evaluation dict.
    """

    def evaluate(self, critic_payload: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "score": 0.5,
            "reason": "dummy critic",
            "received_steps": len(critic_payload.get("steps", [])),
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_runtime() -> AgentRuntime:
    """
    Build an AgentRuntime using dummy M6/M2 components.
    """
    bot_core = DummyBotCore()
    semantics_db = DummySemanticsDB()
    tech_state: TechState = make_minimal_tech_state()

    planner_model = DummyPlannerModel()
    critic_model = DummyCriticModel()

    config = AgentRuntimeConfig(
        context_id="lv_early_factory",
        initial_tech_state=tech_state,
    )

    return AgentRuntime(
        bot_core=bot_core,
        semantics_db=semantics_db,
        planner_model=planner_model,
        critic_model=critic_model,
        config=config,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_planner_tick_returns_plan_dict():
    """
    Smoke test:
      - AgentRuntime.planner_tick() calls M6 + M7 + planner_model
      - Returned plan has expected keys and values
    """
    runtime = make_runtime()

    plan = runtime.planner_tick()

    assert isinstance(plan, dict)
    assert plan["kind"] == "dummy_plan"
    assert plan["source_dim"] == "overworld"
    assert plan["has_inventory_summary"] is True


def test_get_latest_planner_observation_shape():
    """
    Smoke test:
      - AgentRuntime.get_latest_planner_observation() returns Observation
      - Observation.json_payload has expected fields
    """
    runtime = make_runtime()

    obs = runtime.get_latest_planner_observation()

    assert isinstance(obs, Observation)
    payload = obs.json_payload

    assert isinstance(payload, dict)
    assert "tech_state" in payload
    assert "agent" in payload
    assert "inventory_summary" in payload
    assert "machines_summary" in payload
    assert "nearby_entities" in payload
    assert "env_summary" in payload
    assert "craftable_summary" in payload
    assert "context_id" in payload
    assert "text_summary" in payload
    assert isinstance(payload["text_summary"], str)
    assert len(payload["text_summary"]) > 0


def test_evaluate_trace_returns_evaluation_dict():
    """
    Smoke test:
      - Build a tiny PlanTrace
      - AgentRuntime.evaluate_trace() passes through M7 -> DummyCriticModel
    """
    runtime = make_runtime()

    # Build a minimal WorldState for the trace
    world = WorldState(
        tick=1,
        position={"x": 0, "y": 64, "z": 0},
        dimension="overworld",
        inventory=[],
        nearby_entities=[],
        blocks_of_interest=[],
        tech_state={},
        context={},
    )

    # Minimal action + result
    action = Action(type="move_to", params={"x": 1, "y": 64, "z": 0})
    result = ActionResult(success=True, error=None, details={})

    # Single TraceStep
    step = TraceStep(
        world_before=world,
        action=action,
        result=result,
        world_after=world,
        meta={"skill": "test_skill", "timestamp": 123},
    )

    # Use same TechState as runtime config for consistency
    tech = runtime.config.initial_tech_state

    # PlanTrace: pretend planner generated one step
    trace = PlanTrace(
        plan={"steps": [{"skill": "test_skill", "params": {}}]},
        steps=[step],
        tech_state=tech,
        planner_payload={"some": "payload"},
        context_id=runtime.config.context_id,
        virtue_scores={"overall": 0.5},
    )

    evaluation = runtime.evaluate_trace(trace)

    assert isinstance(evaluation, dict)
    assert evaluation["score"] == 0.5
    assert evaluation["reason"] == "dummy critic"
    assert evaluation["received_steps"] == 1

