# path: tests/test_curriculum_learning_properties.py

from __future__ import annotations

import random
from typing import Dict, List

from runtime.phase4_curriculum_learning_orchestrator import (
    CurriculumLearningOrchestrator,
    LearningScheduleConfig,
)

from dataclasses import dataclass


# Tiny fake engine/view just to reach _select_learning_targets nicely if needed

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
    unlocked_projects: List[object]


class FakeCurriculumEngine:
    def __init__(self, view: FakeActiveCurriculumView) -> None:
        self._view = view

    def view(self, *args, **kwargs) -> FakeActiveCurriculumView:
        return self._view


class FakeReplayStore:
    def count_episodes(self, *, skill_name: str, context_id: str) -> int:
        return 999  # always enough episodes for property tests


class FakeLearningManager:
    def __init__(self) -> None:
        self.calls: List[Dict] = []

    def run_learning_cycle_for_goal(self, **kwargs):
        self.calls.append(kwargs)
        return {"status": "ok"}


def test_select_learning_targets_no_duplicates_and_stable_behavior() -> None:
    """
    Property-ish test:

      - Randomize order of must_have / preferred.
      - Ensure no duplicates in targets.
      - Ensure must_have always come first.
      - Ensure toggling always_include_preferred removes preferred but keeps must_have.
    """
    must_have = ["skill_a", "skill_b", "skill_c"]
    preferred = ["skill_c", "skill_d", "skill_e"]  # deliberate overlap

    # Randomize internal order; logic should still behave
    random.shuffle(must_have)
    random.shuffle(preferred)

    skill_focus = {
        "must_have": must_have,
        "preferred": preferred,
    }

    phase_view = FakePhaseView(
        phase=FakePhase(id="p", name="Phase P"),
        is_complete=False,
        active_goals=[],
        virtue_overrides={},
        skill_focus=skill_focus,
    )
    curr_view = FakeActiveCurriculumView(
        curriculum_id="c",
        phase_view=phase_view,
        unlocked_projects=[],
    )

    engine = FakeCurriculumEngine(curr_view)
    replay_store = FakeReplayStore()
    learning_manager = FakeLearningManager()

    # Preferred included
    config_yes = LearningScheduleConfig(
        min_episodes_per_skill=1,
        max_skills_per_tick=10,
        always_include_preferred=True,
    )
    orch_yes = CurriculumLearningOrchestrator(
        curriculum_engine=engine,
        learning_manager=learning_manager,
        replay_store=replay_store,
        config=config_yes,
    )

    targets_yes = orch_yes._select_learning_targets(skill_focus)

    # Must_have must be prefix of targets_yes in the same relative order
    assert targets_yes[: len(must_have)] == must_have

    # No duplicates in the full list
    assert len(set(targets_yes)) == len(targets_yes)

    # Preferred-only set is (preferred - must_have)
    expected_tail = [s for s in preferred if s not in must_have]
    assert targets_yes[len(must_have) :] == expected_tail

    # Preferred excluded
    config_no = LearningScheduleConfig(
        min_episodes_per_skill=1,
        max_skills_per_tick=10,
        always_include_preferred=False,
    )
    orch_no = CurriculumLearningOrchestrator(
        curriculum_engine=engine,
        learning_manager=learning_manager,
        replay_store=replay_store,
        config=config_no,
    )

    targets_no = orch_no._select_learning_targets(skill_focus)

    # Only must_have should remain
    assert targets_no == must_have

