# path: tests/test_curriculum_skill_view_policy.py
from dataclasses import dataclass
from typing import Any, Dict, Optional

from curriculum.manager import CurriculumManager
from curriculum.policy import SkillPolicy, SkillUsageMode
from curriculum.engine import CurriculumEngine
from learning.manager import SkillView
from semantics.schema import TechState
from spec.agent_loop import AgentGoal


class FakeLearningManager:
    """
    Minimal stub that records what include_candidates was
    when build_skill_view() was called.
    """

    def __init__(self) -> None:
        self.last_include_candidates: Optional[bool] = None

    def build_skill_view(self, *, include_candidates: bool) -> SkillView:
        self.last_include_candidates = include_candidates
        # Return a dummy view so callers don't explode.
        return SkillView(active_skills=["stable_skill"], candidate_skills=["candidate_skill"])


class FakeEngine(CurriculumEngine):
    """
    We don't actually use next_goal() in these tests; we just need
    a non-None engine to satisfy CurriculumManager.
    """

    def __init__(self) -> None:
        # Bypass real config; parent may expect an object with .phases etc.
        self._config = type("Cfg", (), {"phases": []})()


def _make_goal(phase: str, source: str) -> AgentGoal:
    return AgentGoal(
        id="g1",
        text="dummy goal",
        phase=phase,
        source=source,
    )


def test_skill_view_policy_p1_curriculum_allows_candidates():
    lm = FakeLearningManager()
    engine = FakeEngine()
    policy = SkillPolicy(usage_mode=SkillUsageMode.STABLE_ONLY)

    mgr = CurriculumManager(
        learning_manager=lm,
        engine=engine,
        policy=policy,
        strategy=None,
    )

    goal = _make_goal(phase="P1_base", source="curriculum")
    view = mgr.get_skill_view_for_goal(goal)

    # Manager should switch policy to ALLOW_CANDIDATES
    assert mgr.policy.usage_mode == SkillUsageMode.ALLOW_CANDIDATES
    # Learning manager was asked to include candidates
    assert lm.last_include_candidates is True
    # And the resulting view includes candidate skills
    assert "candidate_skill" in view.candidate_skills


def test_skill_view_policy_p2_curriculum_stable_only():
    lm = FakeLearningManager()
    engine = FakeEngine()
    policy = SkillPolicy(usage_mode=SkillUsageMode.ALLOW_CANDIDATES)

    mgr = CurriculumManager(
        learning_manager=lm,
        engine=engine,
        policy=policy,
        strategy=None,
    )

    goal = _make_goal(phase="P2_progression", source="curriculum")
    view = mgr.get_skill_view_for_goal(goal)

    # Manager should switch policy to STABLE_ONLY
    assert mgr.policy.usage_mode == SkillUsageMode.STABLE_ONLY
    # Learning manager was asked NOT to include candidates
    assert lm.last_include_candidates is False
    # The dummy SkillView still "has" candidates, but the key behavior
    # is the flag we passed into build_skill_view.
    assert isinstance(view, SkillView)


def test_skill_view_policy_curriculum_explore_allows_candidates():
    lm = FakeLearningManager()
    engine = FakeEngine()
    policy = SkillPolicy(usage_mode=SkillUsageMode.STABLE_ONLY)

    mgr = CurriculumManager(
        learning_manager=lm,
        engine=engine,
        policy=policy,
        strategy=None,
    )

    goal = _make_goal(phase="P3_something", source="curriculum_explore")
    mgr.get_skill_view_for_goal(goal)

    assert mgr.policy.usage_mode == SkillUsageMode.ALLOW_CANDIDATES
    assert lm.last_include_candidates is True

