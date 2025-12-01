# tests/learning/test_evaluator.py

from types import SimpleNamespace
from typing import Dict, Any, List

import pytest

from learning.schema import ExperienceEpisode, EpisodeMetadata, SkillPerformanceStats
from learning.evaluator import SkillEvaluator


# --- Fakes for TechState / PlanTrace ----------------------------------------


class FakeTechState:
    def __init__(self, active: str, unlocked: list[str]) -> None:
        self.active = active
        self.unlocked = unlocked


class FakeAction:
    def __init__(self, type: str, params: Dict[str, Any]) -> None:
        self.type = type
        self.params = params


class FakeResult:
    def __init__(self, success: bool) -> None:
        self.success = success
        self.error = None


class FakeStep:
    def __init__(self, skill: str, success: bool) -> None:
        self.action = FakeAction("noop", {})
        self.result = FakeResult(success)
        self.world_before = SimpleNamespace()
        self.world_after = SimpleNamespace()
        self.meta = {"skill": skill}


class FakePlanTrace:
    def __init__(self, skill: str) -> None:
        self.steps = [FakeStep(skill, True)]
        self.plan = {"steps": [{"skill": skill, "params": {}}]}


# --- Monkeypatched virtues.* -------------------------------------------------


class FakeMetrics:
    def __init__(self, t: float, c: float) -> None:
        self.estimated_time = t
        self.estimated_resource_cost = c


@pytest.fixture(autouse=True)
def patch_virtues(monkeypatch):
    # patch load_virtue_config
    monkeypatch.setattr("learning.evaluator.load_virtue_config", lambda: {})

    # patch extract_plan_metrics
    def fake_extract_plan_metrics(plan, world, tech_state, db, skill_metadata):
        return FakeMetrics(t=10.0, c=5.0)

    monkeypatch.setattr("learning.evaluator.extract_plan_metrics", fake_extract_plan_metrics)

    # patch score_plan
    def fake_score_plan(plan, context_id, metrics, config):
        return {"virtue_scores": {"safety": 0.6, "efficiency": 0.7}}

    monkeypatch.setattr("learning.evaluator.score_plan", fake_score_plan)

    yield


def make_episode(skill: str, success: bool) -> ExperienceEpisode:
    ts = FakeTechState(active="steam_age", unlocked=["steam_age"])
    trace = FakePlanTrace(skill=skill)
    return ExperienceEpisode(
        id=f"ep_{skill}_{success}",
        goal="maintain coke ovens",
        tech_state=ts,
        trace=trace,
        virtue_scores={"overall": 0.5},
        success=success,
        metadata=EpisodeMetadata(),
    )


def test_aggregate_skill_stats_basic() -> None:
    evaluator = SkillEvaluator()

    episodes = [
        make_episode("test_skill", True),
        make_episode("test_skill", False),
        make_episode("other_skill", True),
    ]

    stats = evaluator.aggregate_skill_stats(
        episodes=episodes,
        skill_name="test_skill",
        context_id="steam_age",
        skill_metadata={},
        semantics_db=None,
    )

    assert stats.skill_name == "test_skill"
    assert stats.uses == 2
    assert 0.0 < stats.success_rate <= 1.0
    assert stats.avg_time == pytest.approx(10.0)
    assert stats.avg_resource_cost == pytest.approx(5.0)
    assert "safety" in stats.avg_virtue_scores


def test_compare_stats_promote_candidate() -> None:
    evaluator = SkillEvaluator()

    baseline = SkillPerformanceStats(
        skill_name="test_skill",
        uses=20,
        success_rate=0.7,
        avg_time=10.0,
        avg_resource_cost=5.0,
        avg_virtue_scores={"safety": 0.5},
    )
    candidate = SkillPerformanceStats(
        skill_name="test_skill_v2",
        uses=20,
        success_rate=0.72,
        avg_time=7.0,
        avg_resource_cost=3.0,
        avg_virtue_scores={"safety": 0.8},
    )

    result = evaluator.compare_stats(baseline, candidate)
    assert result["recommendation"] in {"promote_candidate", "keep_baseline"}
    assert "baseline" in result and "candidate" in result

