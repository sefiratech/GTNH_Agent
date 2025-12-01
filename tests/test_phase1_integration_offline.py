# tests/test_phase1_integration_offline.py

from integration.testing import FakeWorldState
from integration import phase1_integration as p1


def test_phase1_integration_lv_coke_oven_scenario() -> None:
    """
    Offline integration test for Phase 1.

    Scenario:
    - LV coke oven + boiler layout design
    - Fake planner returns both 'clustered' and 'scattered' plans
    - Fake virtue lattice prefers 'clustered_coke_ovens' in context 'lv_coke_ovens'

    Expectations:
    - run_phase1_planning_episode returns at least one plan
    - the chosen plan is the clustered coke-oven layout
    - virtue scores are returned as a dict
    """
    world = FakeWorldState(label="lv_test_world")

    goal = "Design optimal LV coke oven and boiler layout"
    virtue_context_id = "lv_coke_ovens"

    best_plan, scores = p1.run_phase1_planning_episode(
        world=world,
        goal=goal,
        virtue_context_id=virtue_context_id,
    )

    assert isinstance(best_plan, dict)
    assert "id" in best_plan
    assert best_plan["id"] == "clustered_coke_ovens"

    assert isinstance(scores, dict)
    assert scores.get("chosen") == "clustered_coke_ovens"
    assert "missing_skills" in scores
    assert "skill_count" in scores


def test_phase1_integration_lv_resource_scenario() -> None:
    """
    Offline integration test for early LV resource scenario.

    Scenario:
    - Goal mentions resources, triggering the resource plan set in FakePlannerBackend
    - Virtue context 'lv_resources' prefers 'focused_resource_run'
    """
    world = FakeWorldState(label="lv_resource_world")

    goal = "Plan an efficient early LV resource run (ores, logs)"
    virtue_context_id = "lv_resources"

    best_plan, scores = p1.run_phase1_planning_episode(
        world=world,
        goal=goal,
        virtue_context_id=virtue_context_id,
    )

    assert isinstance(best_plan, dict)
    assert best_plan["id"] == "focused_resource_run"

    assert isinstance(scores, dict)
    assert scores.get("chosen") == "focused_resource_run"

