# path: tests/test_curriculum_learning_orchestrator.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import pytest

from runtime.phase4_curriculum_learning_orchestrator import (
    CurriculumLearningOrchestrator,
    LearningScheduleConfig,
)


# -----------------------------
# Minimal fakes / stubs
# -----------------------------

@dataclass
class FakePhase:
    id: str
    name: str


@dataclass
class FakePhaseView:
    phase: FakePhase
    is_complete: bool
    active_goals: List[str]
    virtue_overrides: Dict[str, float]
    skill_focus: Dict[str, List[str]]


@dataclass
class FakeActiveCurriculumView:
    curriculum_id: str
    phase_view: FakePhaseView
    unlocked_projects: List[Any]


class FakeCurriculumEngine:
    def __init__(self, view: FakeActiveCurriculumView) -> None:
        self._view = view
        self.calls: List[Dict[str, Any]] = []

    def view(self, tech_state: Any, world: Any) -> FakeActiveCurriculumView:
        self.calls.append({"tech_state": tech_state, "world": world})
        return self._view


class FakeReplayStore:
    def __init__(self, counts: Dict[str, int]) -> None:
        self._counts = counts
        self.calls: List[Dict[str, Any]] = []

    def count_episodes(self, *, skill_name: str, context_id: str) -> int:
        self.calls.append({"skill_name": skill_name, "context_id": context_id})
        return self._counts.get(skill_name, 0)


class FakeLearningManager:
    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    def run_learning_cycle_for_goal(
        self,
        *,
        goal_substring: str,
        target_skill_name: str,
        context_id: str,
        tech_tier: str,
        success_only: bool,
        min_episodes: int,
    ) -> Dict[str, Any]:
        call = {
            "goal_substring": goal_substring,
            "target_skill_name": target_skill_name,
            "context_id": context_id,
            "tech_tier": tech_tier,
            "success_only": success_only,
            "min_episodes": min_episodes,
        }
        self.calls.append(call)
        return {"status": "ok", "echo": call}


# Simple tech/world stubs so we don't drag real schemas in if not needed
@dataclass
class FakeTechState:
    active: str
    unlocked: List[str]


@dataclass
class FakeWorldState:
    context: Dict[str, Any]


# -----------------------------
# Tests
# -----------------------------

def _make_orchestrator_for_order_test() -> tuple[
    CurriculumLearningOrchestrator,
    FakeLearningManager,
]:
    # Skill-focus with both must_have and preferred
    skill_focus = {
        "must_have": ["maintain_boilers", "feed_coke_ovens"],
        "preferred": ["chunk_mining", "optimize_routes"],
    }

    phase_view = FakePhaseView(
        phase=FakePhase(id="steam_early", name="Steam Early"),
        is_complete=False,
        active_goals=["Maintain stable steam power."],
        virtue_overrides={"Safety": 1.5},
        skill_focus=skill_focus,
    )
    curr_view = FakeActiveCurriculumView(
        curriculum_id="default_speedrun",
        phase_view=phase_view,
        unlocked_projects=[],
    )

    curriculum_engine = FakeCurriculumEngine(curr_view)
    replay_store = FakeReplayStore(
        counts={
            # All skills have plenty of experience
            "maintain_boilers": 10,
            "feed_coke_ovens": 10,
            "chunk_mining": 10,
            "optimize_routes": 10,
        }
    )
    learning_manager = FakeLearningManager()

    config = LearningScheduleConfig(
        min_episodes_per_skill=5,
        max_skills_per_tick=2,          # cap at 2 to test ordering
        always_include_preferred=True,
        context_prefix="curriculum",
    )

    orchestrator = CurriculumLearningOrchestrator(
        curriculum_engine=curriculum_engine,
        learning_manager=learning_manager,
        replay_store=replay_store,
        config=config,
    )
    return orchestrator, learning_manager


def test_run_after_episode_respects_order_and_max_skills() -> None:
    """
    Orchestrator should:
      - Respect skill_focus ordering (must_have first).
      - Enforce max_skills_per_tick.
      - Pass correct context_id and tech_tier to learning manager.
    """
    orchestrator, learning_manager = _make_orchestrator_for_order_test()

    tech_state = FakeTechState(active="steam_age", unlocked=[])
    world = FakeWorldState(context={})
    episode_meta = {"episode_id": "ep_001"}

    summary = orchestrator.run_after_episode(
        tech_state=tech_state,
        world=world,
        episode_meta=episode_meta,
    )

    # Only 2 skills trained due to max_skills_per_tick=2
    trained = summary["skills_trained"]
    assert trained == ["maintain_boilers", "feed_coke_ovens"]

    # Learning manager called twice, in the same order
    assert len(learning_manager.calls) == 2
    c1, c2 = learning_manager.calls

    # Check correct context_id and tech_tier passed
    assert c1["context_id"] == "curriculum:default_speedrun:steam_early"
    assert c2["context_id"] == "curriculum:default_speedrun:steam_early"
    assert c1["tech_tier"] == "steam_age"
    assert c2["tech_tier"] == "steam_age"

    # Check goal_substring heuristic
    assert c1["goal_substring"] == "maintain boilers"
    assert c2["goal_substring"] == "feed coke ovens"


def test_run_after_episode_no_learning_when_insufficient_experience() -> None:
    """
    If replay counts are below min_episodes_per_skill, no learning cycles should run.
    """
    skill_focus = {
        "must_have": ["maintain_boilers"],
        "preferred": ["feed_coke_ovens"],
    }
    phase_view = FakePhaseView(
        phase=FakePhase(id="steam_early", name="Steam Early"),
        is_complete=False,
        active_goals=["Maintain stable steam power."],
        virtue_overrides={},
        skill_focus=skill_focus,
    )
    curr_view = FakeActiveCurriculumView(
        curriculum_id="eco_factory",
        phase_view=phase_view,
        unlocked_projects=[],
    )

    curriculum_engine = FakeCurriculumEngine(curr_view)
    replay_store = FakeReplayStore(
        counts={
            # Not enough experience for either skill
            "maintain_boilers": 1,
            "feed_coke_ovens": 2,
        }
    )
    learning_manager = FakeLearningManager()

    config = LearningScheduleConfig(
        min_episodes_per_skill=5,   # require 5
        max_skills_per_tick=5,
        always_include_preferred=True,
        context_prefix="curriculum",
    )

    orchestrator = CurriculumLearningOrchestrator(
        curriculum_engine=curriculum_engine,
        learning_manager=learning_manager,
        replay_store=replay_store,
        config=config,
    )

    tech_state = FakeTechState(active="steam_age", unlocked=[])
    world = FakeWorldState(context={})
    episode_meta = {"episode_id": "ep_002"}

    summary = orchestrator.run_after_episode(
        tech_state=tech_state,
        world=world,
        episode_meta=episode_meta,
    )

    assert summary["skills_trained"] == []
    assert summary["learning_results"] == []
    assert learning_manager.calls == []


def test_select_learning_targets_respects_preferred_toggle() -> None:
    """
    Changing always_include_preferred should flip whether preferred skills are included.
    """
    skill_focus = {
        "must_have": ["core_skill"],
        "preferred": ["extra_skill_1", "extra_skill_2"],
    }
    phase_view = FakePhaseView(
        phase=FakePhase(id="phase_x", name="Phase X"),
        is_complete=False,
        active_goals=[],
        virtue_overrides={},
        skill_focus=skill_focus,
    )
    curr_view = FakeActiveCurriculumView(
        curriculum_id="test_curriculum",
        phase_view=phase_view,
        unlocked_projects=[],
    )

    curriculum_engine = FakeCurriculumEngine(curr_view)
    replay_store = FakeReplayStore(counts={})
    learning_manager = FakeLearningManager()

    # Config with preferred included
    config_yes = LearningScheduleConfig(
        min_episodes_per_skill=1,
        max_skills_per_tick=10,
        always_include_preferred=True,
    )
    orch_yes = CurriculumLearningOrchestrator(
        curriculum_engine=curriculum_engine,
        learning_manager=learning_manager,
        replay_store=replay_store,
        config=config_yes,
    )
    targets_yes = orch_yes._select_learning_targets(skill_focus)
    assert targets_yes == ["core_skill", "extra_skill_1", "extra_skill_2"]

    # Config with preferred excluded
    config_no = LearningScheduleConfig(
        min_episodes_per_skill=1,
        max_skills_per_tick=10,
        always_include_preferred=False,
    )
    orch_no = CurriculumLearningOrchestrator(
        curriculum_engine=curriculum_engine,
        learning_manager=learning_manager,
        replay_store=replay_store,
        config=config_no,
    )
    targets_no = orch_no._select_learning_targets(skill_focus)
    assert targets_no == ["core_skill"]

