# path: tests/test_lab_integration_skill_view.py

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import agent.loop as agent_loop_module
from agent.loop import AgentLoop, AgentLoopConfig
from curriculum.engine import CurriculumEngine
from curriculum.manager import CurriculumManager
from curriculum.policy import SkillPolicy
from curriculum.schema import (
    CurriculumConfig,
    PhaseConfig,
    PhaseGoal,
    PhaseTechTargets,
)
from learning.buffer import ExperienceBuffer
from learning.manager import SkillLearningManager, SkillView
from learning.schema import Experience
from spec.agent_loop import AgentGoal


# ---------------------------------------------------------------------------
# Fakes / stubs
# ---------------------------------------------------------------------------

class FakeRuntime:
    """
    Minimal M6/M7 stand-in for AgentLoop.

    Exposes:
      - config.default_goal / config.phase
      - get_latest_planner_observation()
      - get_world_summary()
    """

    def __init__(self) -> None:
        self.config = type(
            "Cfg",
            (),
            {
                "default_goal": "[LAB] dummy fallback goal",
                "phase": "demo_p0",
            },
        )()
        self.latest_observation = {"placeholder": True}

    def get_latest_planner_observation(self) -> dict:
        return self.latest_observation

    def get_world_summary(self) -> dict:
        return {
            "inventory": {},
            "machines": [],
        }


class DummyTechState:
    """
    Minimal TechState stub.

    CurriculumEngine only cares about:
      - .active
      - .unlocked
    """

    def __init__(self, active: str, unlocked: list[str]) -> None:
        self.active = active
        self.unlocked = unlocked


class DummySynthesizer:
    """
    Placeholder for SkillSynthesizer.

    Not used in this test; exists only to satisfy SkillLearningManager.__init__.
    """

    def __init__(self) -> None:
        pass


class DummyEvaluator:
    """
    Placeholder for SkillEvaluator.

    Not used in this test; exists only to satisfy SkillLearningManager.__init__.
    """

    def __init__(self) -> None:
        pass


class DummySkillRegistry:
    """
    Minimal stand-in for SkillRegistry.

    SkillLearningManager.build_skill_view() only requires:
      - describe_all() -> { skill_name: { "metadata": { "status": ... } } }
    """

    def __init__(self) -> None:
        self._meta: Dict[str, Dict[str, Dict[str, str]]] = {
            "stable_skill": {
                "metadata": {
                    "status": "active",
                }
            },
            "experimental_skill": {
                "metadata": {
                    "status": "candidate",
                }
            },
            "retired_skill": {
                "metadata": {
                    "status": "retired",
                }
            },
        }

    def describe_all(self) -> Dict[str, Dict[str, Dict[str, str]]]:
        return self._meta


# ---------------------------------------------------------------------------
# Integration test: real SkillLearningManager + ExperienceBuffer
# ---------------------------------------------------------------------------

def test_lab_integration_real_skill_view(tmp_path, monkeypatch) -> None:
    """
    Golden-path integration focusing on M10+M11 wiring:

    - Fake runtime (no Minecraft, no real LLMs)
    - Real AgentLoop
    - Real CurriculumEngine + CurriculumManager
    - Real ExperienceBuffer
    - Real SkillLearningManager backed by:
        * DummySkillRegistry with active/candidate/retired skills
        * DummySynthesizer / DummyEvaluator

    Asserts:
      - An episode runs and writes at least one Experience to replay buffer.
      - Curriculum goal is used (not fallback).
      - SkillLearningManager.build_skill_view() returns expected active/candidate
        skills based on registry metadata and include_candidates flag.
    """

    # --------------------------------------------------------------
    # 1. Monkeypatch semantics_get_tech_state to deterministic stub
    # --------------------------------------------------------------
    def fake_get_tech_state(obs: Any) -> DummyTechState:
        return DummyTechState(active="stone_age", unlocked=["stone_age"])

    monkeypatch.setattr(
        agent_loop_module,
        "semantics_get_tech_state",
        fake_get_tech_state,
    )

    # --------------------------------------------------------------
    # 2. Fake runtime
    # --------------------------------------------------------------
    runtime = FakeRuntime()

    # --------------------------------------------------------------
    # 3. Tiny curriculum config (single phase, single goal)
    # --------------------------------------------------------------
    phase = PhaseConfig(
        id="P1_demo",
        name="Demo Phase",
        tech_targets=PhaseTechTargets(required_active="stone_age"),
        goals=[PhaseGoal(id="demo_core", description="Stabilize base")],
    )

    curriculum_cfg = CurriculumConfig(
        id="demo_curriculum",
        name="Demo Curriculum",
        description="Tiny lab curriculum",
        phases=[phase],
    )

    engine = CurriculumEngine(curriculum_cfg)

    # --------------------------------------------------------------
    # 4. Real ExperienceBuffer + real SkillLearningManager
    # --------------------------------------------------------------
    replay_path: Path = tmp_path / "replay.jsonl"
    replay_buffer = ExperienceBuffer(path=replay_path)

    candidates_dir: Path = tmp_path / "candidates"
    candidates_dir.mkdir(parents=True, exist_ok=True)

    skills = DummySkillRegistry()
    synthesizer = DummySynthesizer()
    evaluator = DummyEvaluator()

    learning_manager = SkillLearningManager(
        buffer=replay_buffer,
        synthesizer=synthesizer,
        evaluator=evaluator,
        skills=skills,
        candidates_dir=candidates_dir,
        semantics_db=None,
    )

    policy = SkillPolicy()
    curriculum = CurriculumManager(
        learning_manager=learning_manager,
        engine=engine,
        policy=policy,
        strategy=None,
    )

    # --------------------------------------------------------------
    # 5. Real AgentLoop wired to curriculum + replay buffer
    # --------------------------------------------------------------
    loop_cfg = AgentLoopConfig(
        enable_critic=False,
        enable_retry_loop=False,
        store_experiences=True,
        max_planner_calls=1,
        max_skill_steps=4,
        fail_fast_on_invalid_plan=False,
        log_virtue_scores=False,
        log_traces=False,
    )

    loop = AgentLoop(
        runtime=runtime,
        planner=None,
        curriculum=curriculum,
        skills=None,
        monitor=None,
        replay_buffer=replay_buffer,
        config=loop_cfg,
    )

    # --------------------------------------------------------------
    # 6. Run a couple of episodes to seed the replay buffer
    # --------------------------------------------------------------
    for ep in range(2):
        result = loop.run_episode(episode_id=ep)
        plan = result.plan or {}
        # goal_text must come from curriculum, not fallback
        assert "Stabilize base" in plan.get("goal_text", "")

    # --------------------------------------------------------------
    # 7. Assert replay buffer has at least one Experience
    # --------------------------------------------------------------
    assert replay_buffer.count() >= 1
    experiences = list(replay_buffer.load_all())
    assert experiences, "Replay buffer should contain at least one Experience"

    last: Experience = experiences[-1]
    assert isinstance(last, Experience)
    assert isinstance(last.goal, AgentGoal)
    assert last.goal.source == "curriculum"

    # --------------------------------------------------------------
    # 8. Assert SkillLearningManager.build_skill_view behavior
    # --------------------------------------------------------------
    # Stable-only mode: candidates excluded
    view_stable_only: SkillView = learning_manager.build_skill_view(
        include_candidates=False
    )
    assert "stable_skill" in view_stable_only.active_skills
    assert "experimental_skill" not in view_stable_only.candidate_skills
    assert "retired_skill" not in view_stable_only.active_skills

    # Exploratory mode: candidates included
    view_with_candidates: SkillView = learning_manager.build_skill_view(
        include_candidates=True
    )
    assert "stable_skill" in view_with_candidates.active_skills
    assert "experimental_skill" in view_with_candidates.candidate_skills
    assert "retired_skill" not in view_with_candidates.active_skills

