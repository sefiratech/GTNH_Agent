# tests/test_observation_pipeline.py
"""
Sanity tests for M7 system-position helpers (observation.pipeline).

We don't care about real LLM behavior here, only that:
  - M6 -> M7 -> M2 planner path wires correctly
  - M8 -> M7 -> M2 critic path wires correctly
"""

from typing import Dict, Any

from observation.pipeline import (
    PlannerContext,
    CriticContext,
    build_planner_observation,
    planner_step,
    critic_step,
    BotCore,
    PlannerModel,
    CriticModel,
)
from observation.trace_schema import PlanTrace, TraceStep
from observation.testing import DummySemanticsDB
from semantics.schema import TechState
from spec.types import WorldState, Observation, Action, ActionResult

from bot_core.snapshot import RawWorldSnapshot


# ---------------------------------------------------------------------------
# Dummy implementations
# ---------------------------------------------------------------------------


class DummyBotCore:
    """Very small fake BotCore that returns a fixed RawWorldSnapshot."""

    def observe(self) -> RawWorldSnapshot:
        return RawWorldSnapshot(
            tick=123,
            dimension="overworld",
            player_pos={"x": 0.0, "y": 64.0, "z": 0.0},
            player_yaw=0.0,
            player_pitch=0.0,
            on_ground=True,
            chunks={},
            # Use dict entities instead of RawEntity to avoid constructor issues.
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


class DummyPlanner(PlannerModel):
    """Fake planner that just echoes a trivial plan."""

    def plan(self, observation: Observation) -> Dict[str, Any]:
        assert isinstance(observation.json_payload, dict)
        return {
            "kind": "dummy_plan",
            "source_dim": observation.json_payload.get("agent", {}).get("dimension"),
        }


class DummyCritic(CriticModel):
    """Fake critic that just adds a fixed score."""

    def evaluate(self, critic_payload: Dict[str, Any]) -> Dict[str, Any]:
        assert "plan" in critic_payload
        assert "steps" in critic_payload
        return {
            "score": 0.5,
            "reason": "dummy critic",
        }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_build_planner_observation():
    bot = DummyBotCore()
    db = DummySemanticsDB()
    tech = TechState(unlocked=["stone_age"], active="stone_age", evidence={})

    obs = build_planner_observation(
        bot_core=bot,
        semantics_db=db,
        tech_state=tech,
        context_id="lv_early_factory",
    )

    assert isinstance(obs, Observation)
    payload = obs.json_payload
    assert payload["tech_state"]["active"] == "stone_age"
    assert payload["agent"]["dimension"] == "overworld"
    assert isinstance(payload["text_summary"], str)


def test_planner_step_roundtrip():
    ctx = PlannerContext(
        bot_core=DummyBotCore(),
        semantics_db=DummySemanticsDB(),
        planner_model=DummyPlanner(),
    )
    tech = TechState(unlocked=["stone_age"], active="stone_age", evidence={})

    plan = planner_step(
        ctx=ctx,
        tech_state=tech,
        context_id="lv_early_factory",
    )

    assert plan["kind"] == "dummy_plan"
    assert plan["source_dim"] == "overworld"


def test_critic_step_roundtrip():
    critic_ctx = CriticContext(critic_model=DummyCritic())

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

    action = Action(type="move_to", params={"x": 1, "y": 64, "z": 0})
    result = ActionResult(success=True, error=None, details={})

    step = TraceStep(
        world_before=world,
        action=action,
        result=result,
        world_after=world,
        meta={"skill": "test_skill"},
    )

    tech = TechState(unlocked=["stone_age"], active="stone_age", evidence={})

    trace = PlanTrace(
        plan={"steps": [{"skill": "test_skill", "params": {}}]},
        steps=[step],
        tech_state=tech,
        planner_payload={"some": "payload"},
        context_id="lv_early_factory",
        virtue_scores={"overall": 0.5},
    )

    out = critic_step(critic_ctx, trace)

    assert out["score"] == 0.5
    assert out["reason"] == "dummy critic"

