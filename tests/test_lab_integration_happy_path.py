# path: tests/test_lab_integration_happy_path.py

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Tuple

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
from learning.manager import SkillView
from learning.schema import Experience
from spec.agent_loop import AgentGoal


# ---------------------------------------------------------------------------
# Fakes / stubs
# ---------------------------------------------------------------------------

class FakeRuntime:
    """
    Minimal M6/M7 stand-in for AgentLoop.

    Exposes just enough interface for:
      - goal selection (via config)
      - observation for semantics_get_tech_state(...)
      - world summary for planning context
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


class FakeSkillLearningManager:
    """
    Tiny M10 stub used only for SkillView generation.

    It returns a deterministic SkillView so we can assert that:
      - CurriculumManager.get_skill_view_for_goal(...) is called
      - AgentLoop.world_summary includes a skill_view without exploding
    """

    def __init__(self) -> None:
        self.calls: List[Tuple[bool, SkillView]] = []

    def build_skill_view(self, *, include_candidates: bool) -> SkillView:
        view = SkillView(
            active_skills=["stable_skill"],
            candidate_skills=["experimental_skill"] if include_candidates else [],
        )
        self.calls.append((include_candidates, view))
        return view


# ---------------------------------------------------------------------------
# Golden-path integration test
# ---------------------------------------------------------------------------

def test_lab_integration_happy_path_creates_experience(tmp_path, monkeypatch) -> None:
    """
    Golden-path integration:

    - Fake runtime (no Minecraft, no real LLMs)
    - Real AgentLoop
    - Real CurriculumEngine + CurriculumManager
    - Real ExperienceBuffer
    - Fake SkillLearningManager

    Asserts:
      1. Goal came from curriculum (not fallback).
      2. A plan dict with expected keys is produced.
      3. At least one Experience is written to the replay buffer and
         its goal metadata matches the curriculum goal.
    """

    # ------------------------------------------------------------------
    # 1. Monkeypatch semantics_get_tech_state so M3 → M11 is deterministic
    # ------------------------------------------------------------------

    def fake_get_tech_state(obs: Any) -> DummyTechState:
        # We always pretend we're in "stone_age"
        return DummyTechState(active="stone_age", unlocked=["stone_age"])

    monkeypatch.setattr(
        agent_loop_module,
        "semantics_get_tech_state",
        fake_get_tech_state,
    )

    # ------------------------------------------------------------------
    # 2. Build fake runtime (M6/M7 stand-in)
    # ------------------------------------------------------------------

    runtime = FakeRuntime()

    # ------------------------------------------------------------------
    # 3. Build minimal curriculum stack (M11)
    # ------------------------------------------------------------------

    phase = PhaseConfig(
        id="P1_demo",
        name="Demo Phase",
        tech_targets=PhaseTechTargets(required_active="stone_age"),
        goals=[PhaseGoal(id="demo_core", description="Stabilize base")],
    )

    curriculum_cfg = CurriculumConfig(
        id="demo_curriculum",
        name="Demo Curriculum",
        description="Lab integration test curriculum",
        phases=[phase],
    )

    engine = CurriculumEngine(curriculum_cfg)

    learning_manager = FakeSkillLearningManager()
    policy = SkillPolicy()

    curriculum = CurriculumManager(
        learning_manager=learning_manager,
        engine=engine,
        policy=policy,
        strategy=None,
    )

    # ------------------------------------------------------------------
    # 4. Real replay buffer (M10 ExperienceBuffer)
    # ------------------------------------------------------------------

    replay_path: Path = tmp_path / "replay.jsonl"
    replay_buffer = ExperienceBuffer(path=replay_path)

    # ------------------------------------------------------------------
    # 5. Real AgentLoop with Pass B/C config
    # ------------------------------------------------------------------

    loop_cfg = AgentLoopConfig(
        enable_critic=False,          # keep it quiet for the lab
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
        planner=None,          # fallback planning path; plan shape still checked
        curriculum=curriculum,
        skills=None,
        monitor=None,
        replay_buffer=replay_buffer,
        config=loop_cfg,
    )

    # ------------------------------------------------------------------
    # 6. Run one episode
    # ------------------------------------------------------------------

    result = loop.run_episode(episode_id=0)

    # ------------------------------------------------------------------
    # 7. Assertions: curriculum goal, plan shape, replay written
    # ------------------------------------------------------------------

    plan = result.plan or {}

    # 7.1 Plan shape exists: "goal_id", "tasks", "steps"
    assert "goal_id" in plan
    assert "tasks" in plan
    assert "steps" in plan

    # 7.2 Goal came from curriculum, not fallback text
    goal_text = plan.get("goal_text", "")
    goal_id = plan.get("goal_id", "")

    assert goal_text == "Stabilize base"
    # Engine builds AgentGoal.id as f"{phase.id}:{goal.id}"
    assert goal_id.startswith("P1_demo:")

    # 7.3 Skill view was actually queried at least once
    assert learning_manager.calls, "SkillLearningManager.build_skill_view was never called"

    # 7.4 Replay buffer has at least one Experience
    assert replay_buffer.count() >= 1

    experiences = list(replay_buffer.load_all())
    assert len(experiences) >= 1

    last: Experience = experiences[-1]
    assert isinstance(last, Experience)

    # Goal metadata on the stored Experience should match curriculum origin
    assert isinstance(last.goal, AgentGoal)
    assert last.goal.source == "curriculum"
    assert last.goal.text == "Stabilize base"

    # Success flag should be a boolean; value itself doesn't matter for lab correctness
    assert last.final_outcome.get("success") in (True, False)

