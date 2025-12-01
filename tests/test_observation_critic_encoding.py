# tests/test_observation_critic_encoding.py
"""
Critic tests for M7 - observation_encoding.

Covers:
  - synthetic PlanTrace
  - encoding structure
  - step serialization correctness
"""

from observation.encoder import encode_for_critic
from observation.trace_schema import TraceStep, PlanTrace
from observation.testing import make_minimal_tech_state
from semantics.schema import TechState
from spec.types import WorldState, Action, ActionResult


def _make_world_state() -> WorldState:
    return WorldState(
        tick=1,
        position={"x": 0, "y": 64, "z": 0},
        dimension="overworld",
        inventory=[],
        nearby_entities=[],
        blocks_of_interest=[],
        tech_state={},
        context={},
    )


def test_encode_for_critic_basic():
    world = _make_world_state()
    tech = make_minimal_tech_state()

    action = Action(
        type="move_to",
        params={"x": 1, "y": 64, "z": 0},
    )
    result = ActionResult(
        success=True,
        error=None,
        details={},
    )

    step = TraceStep(
        world_before=world,
        action=action,
        result=result,
        world_after=world,
        meta={"skill": "test_skill", "timestamp": 123},
    )

    trace = PlanTrace(
        plan={"steps": [{"skill": "test_skill", "params": {}}]},
        steps=[step],
        tech_state=tech,
        planner_payload={"some": "payload"},
        context_id="lv_early_factory",
        virtue_scores={"overall": 0.5},
    )

    enc = encode_for_critic(trace)

    # Structural checks
    assert "tech_state" in enc
    assert "context_id" in enc
    assert "plan" in enc
    assert "steps" in enc
    assert "planner_observation" in enc
    assert "virtue_scores" in enc
    assert "text_summary" in enc

    # Field sanity
    assert enc["tech_state"]["active"] == "stone_age"
    assert enc["context_id"] == "lv_early_factory"
    assert isinstance(enc["steps"], list)
    assert len(enc["steps"]) == 1

    s0 = enc["steps"][0]
    assert s0["action"]["type"] == "move_to"
    assert s0["result"]["success"] is True
    assert s0["meta"]["skill"] == "test_skill"
    assert s0["world_before_pos"] == {"x": 0, "y": 64, "z": 0}
    assert s0["world_after_pos"] == {"x": 0, "y": 64, "z": 0}

    # Summary sanity
    assert isinstance(enc["text_summary"], str)
    assert "plan with" in enc["text_summary"]

