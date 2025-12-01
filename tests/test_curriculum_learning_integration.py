# path: tests/test_curriculum_learning_integration.py

from __future__ import annotations

from pathlib import Path
from textwrap import dedent
from typing import Any, Dict, List

from curriculum.loader import load_curriculum
from curriculum.engine import CurriculumEngine
from runtime.phase4_curriculum_learning_orchestrator import (
    CurriculumLearningOrchestrator,
    LearningScheduleConfig,
)

from dataclasses import dataclass


# Minimal stubs for learning + replay

class FakeReplayStore:
    def __init__(self, counts: Dict[tuple[str, str], int]) -> None:
        # counts is keyed by (skill_name, context_id)
        self._counts = counts

    def count_episodes(self, *, skill_name: str, context_id: str) -> int:
        return self._counts.get((skill_name, context_id), 0)


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
        return {"status": "ok"}


@dataclass
class FakeTechState:
    active: str
    unlocked: List[str]


@dataclass
class FakeWorldState:
    context: Dict[str, Any]


def _write_yaml(path: Path, text: str) -> None:
    normalized = dedent(text).lstrip("\n")
    path.write_text(normalized, encoding="utf-8")


def test_learning_triggers_only_after_enough_episodes(tmp_path: Path) -> None:
    """
    Integration-style test:

      - Uses a small real curriculum YAML.
      - Uses real CurriculumEngine.
      - Uses fake replay store seeded with varying episode counts.
      - Asserts that skills are trained independently once *each* crosses threshold.
    """
    yaml_text = """
    id: "integration_curriculum"
    name: "Integration Curriculum"
    description: "Tiny curriculum for integration testing."
    phases:
      - id: "steam_early"
        name: "Steam Early"
        tech_targets:
          required_active: "steam_age"
          required_unlocked: []
        goals:
          - id: "g1"
            description: "Make steam."
        virtue_overrides: {}
        skill_focus:
          must_have:
            - "maintain_boilers"
          preferred:
            - "feed_coke_ovens"
        completion_conditions:
          tech_unlocked: []
          machines_present: []
    long_horizon_projects: []
    """
    cfg_path = tmp_path / "integration_curriculum.yaml"
    _write_yaml(cfg_path, yaml_text)

    curriculum_config = load_curriculum(cfg_path)
    engine = CurriculumEngine(curriculum_config)

    # context id the orchestrator will build:
    # "curriculum:integration_curriculum:steam_early"
    context_id = "curriculum:integration_curriculum:steam_early"

    # Replay store with:
    #   - maintain_boilers below threshold
    #   - feed_coke_ovens above threshold
    replay_store = FakeReplayStore(
        counts={
            ("maintain_boilers", context_id): 3,  # below min=5
            ("feed_coke_ovens", context_id): 10,  # above min=5
        }
    )
    learning_manager = FakeLearningManager()

    config = LearningScheduleConfig(
        min_episodes_per_skill=5,
        max_skills_per_tick=5,
        always_include_preferred=True,
        context_prefix="curriculum",
    )

    orchestrator = CurriculumLearningOrchestrator(
        curriculum_engine=engine,
        learning_manager=learning_manager,
        replay_store=replay_store,
        config=config,
    )

    tech_state = FakeTechState(active="steam_age", unlocked=[])
    world_state = FakeWorldState(context={})

    # First run: only feed_coke_ovens has enough episodes, so only that skill should be trained
    summary1 = orchestrator.run_after_episode(
        tech_state=tech_state,
        world=world_state,
        episode_meta={"episode_id": "ep_001"},
    )

    assert summary1["skills_trained"] == ["feed_coke_ovens"]
    assert len(learning_manager.calls) == 1
    assert learning_manager.calls[0]["target_skill_name"] == "feed_coke_ovens"

    # Now bump maintain_boilers above threshold
    replay_store._counts[("maintain_boilers", context_id)] = 6

    # Second run: both skills have enough episodes, so both should be trained
    summary2 = orchestrator.run_after_episode(
        tech_state=tech_state,
        world=world_state,
        episode_meta={"episode_id": "ep_002"},
    )

    assert summary2["skills_trained"] == ["maintain_boilers", "feed_coke_ovens"]

    # We now expect 3 total learning calls:
    #   0: feed_coke_ovens (first run)
    #   1: maintain_boilers (second run)
    #   2: feed_coke_ovens (second run)
    assert len(learning_manager.calls) == 3
    assert learning_manager.calls[0]["target_skill_name"] == "feed_coke_ovens"
    assert learning_manager.calls[1]["target_skill_name"] == "maintain_boilers"
    assert learning_manager.calls[2]["target_skill_name"] == "feed_coke_ovens"

