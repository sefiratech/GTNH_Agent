import pytest

from integration.phase1_integration import run_phase1_planning_episode
from integration.testing.fakes import FakeWorldState, FakePlannerBackend

# Monkeypatch slot: the integration layer will call get_planner_backend()
# so we override it to return a FakePlannerBackend that produces zero plans.


class ZeroPlanBackend(FakePlannerBackend):
    def generate_plan(self, payload):
        return {"plans": []}


def test_phase1_no_plans_raises(monkeypatch):
    # Force the planner backend to be our zero-plan version
    monkeypatch.setattr(
        "integration.phase1_integration.get_planner_backend",
        lambda name: ZeroPlanBackend(),
    )

    world = FakeWorldState(label="empty_world")

    with pytest.raises(RuntimeError):
        run_phase1_planning_episode(
            world=world,
            goal="something impossible",
            virtue_context_id="lv_test",
        )

