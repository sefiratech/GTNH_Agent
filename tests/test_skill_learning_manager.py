# tests/learning/test_skill_learning_manager.py

from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Any, List, Optional

from learning.buffer import ExperienceBuffer
from learning.schema import ExperienceEpisode, EpisodeMetadata
from learning.synthesizer import SkillSynthesizer
from learning.evaluator import SkillEvaluator
from learning.manager import SkillLearningManager


# --- Shared fakes ------------------------------------------------------------


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
    def __init__(self, skill: str, success: bool = True) -> None:
        self.action = FakeAction("noop", {})
        self.result = FakeResult(success)
        self.world_before = SimpleNamespace()
        self.world_after = SimpleNamespace()
        self.meta = {"skill": skill}


class FakePlanTrace:
    def __init__(self, skill: str) -> None:
        self.steps = [FakeStep(skill)]
        self.plan = {"steps": [{"skill": skill, "params": {}}]}


def tech_state_to_dict(ts: FakeTechState) -> Dict[str, Any]:
    return {"active": ts.active, "unlocked": list(ts.unlocked)}


def tech_state_from_dict(data: Dict[str, Any]) -> FakeTechState:
    return FakeTechState(active=data.get("active", ""), unlocked=data.get("unlocked", []))


def trace_to_dict(trace: FakePlanTrace) -> Dict[str, Any]:
    return {
        "plan": trace.plan,
        "steps": [
            {
                "action_type": s.action.type,
                "params": s.action.params,
                "success": s.result.success,
                "meta": s.meta,
            }
            for s in trace.steps
        ],
    }


def trace_from_dict(data: Dict[str, Any]) -> FakePlanTrace:
    steps_data = data.get("steps") or []
    skill = steps_data[0].get("meta", {}).get("skill", "unknown") if steps_data else "unknown"
    return FakePlanTrace(skill=skill)


class FakeCodeModel:
    def generate_json(self, *, prompt: Dict[str, Any]) -> Dict[str, Any]:
        goal = prompt.get("episodes", [{}])[0].get("goal", "unknown_goal")
        skill = prompt.get("target_skill_name") or "auto_skill"

        spec_yaml = f"""name: {skill}
version: 1
description: Auto skill for {goal}
"""

        impl_code = """class AutoSkill:
    def run(self, ctx):
        # TODO: implement
        return True
"""
        return {
            "spec_yaml": spec_yaml,
            "impl_code": impl_code,
            "rationale": "fake rationale",
        }


class FakeSkillRegistry:
    """
    Minimal fake M5 SkillRegistry.
    """

    def __init__(self) -> None:
        self._skills: Dict[str, Dict[str, Any]] = {
            "maintain_coke_ovens": {"name": "maintain_coke_ovens", "version": 1}
        }

    def describe_all(self) -> Dict[str, Dict[str, Any]]:
        return dict(self._skills)


def make_episode(goal: str, skill: str, success: bool) -> ExperienceEpisode:
    ts = FakeTechState(active="steam_age", unlocked=["steam_age"])
    trace = FakePlanTrace(skill=skill)
    return ExperienceEpisode(
        id=f"ep_{goal}_{success}",
        goal=goal,
        tech_state=ts,
        trace=trace,
        virtue_scores={"overall": 0.5},
        success=success,
        metadata=EpisodeMetadata(),
    )


# --- Test --------------------------------------------------------------------


def test_skill_learning_manager_end_to_end(tmp_path: Path, monkeypatch) -> None:
    # Patch virtues inside evaluator to avoid pulling in real lattice
    from learning import evaluator as evaluator_mod

    evaluator_mod.load_virtue_config = lambda: {}
    evaluator_mod.extract_plan_metrics = lambda plan, world, tech_state, db, skill_metadata: SimpleNamespace(
        estimated_time=10.0,
        estimated_resource_cost=5.0,
    )
    evaluator_mod.score_plan = lambda plan, context_id, metrics, config: {
        "virtue_scores": {"safety": 0.6}
    }

    buf_path = tmp_path / "episodes.jsonl"
    buffer = ExperienceBuffer(
        path=buf_path,
        tech_state_to_dict=tech_state_to_dict,
        tech_state_from_dict=tech_state_from_dict,
        trace_to_dict=trace_to_dict,
        trace_from_dict=trace_from_dict,
    )

    # Populate buffer with enough episodes
    for i in range(6):
        ep = make_episode(
            goal="maintain coke ovens",
            skill="maintain_coke_ovens",
            success=True,
        )
        ep.id = f"ep{i}"
        buffer.append(ep)

    code_model = FakeCodeModel()
    synthesizer = SkillSynthesizer(code_model)
    evaluator = SkillEvaluator()
    skills = FakeSkillRegistry()
    candidates_dir = tmp_path / "skills_candidates"

    manager = SkillLearningManager(
        buffer=buffer,
        synthesizer=synthesizer,
        evaluator=evaluator,
        skills=skills,
        candidates_dir=candidates_dir,
        semantics_db=None,
    )

    result = manager.run_learning_cycle_for_goal(
        goal_substring="maintain coke ovens",
        target_skill_name="maintain_coke_ovens",
        context_id="steam_age",
        tech_tier="steam_age",
        success_only=True,
        min_episodes=5,
    )

    # Basic smoke: learning cycle produced something
    assert result is not None
    candidate = result["candidate"]
    assert candidate is not None
    assert candidate.spec_yaml
    assert candidate.impl_code

    # Files written
    spec_path = candidates_dir / f"{candidate.id}.yaml"
    code_path = candidates_dir / f"{candidate.id}.py"
    meta_path = candidates_dir / f"{candidate.id}.meta.json"

    assert spec_path.exists()
    assert code_path.exists()
    assert meta_path.exists()

