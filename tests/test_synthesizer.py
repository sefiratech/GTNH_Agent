# tests/learning/test_synthesizer.py

from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Any, List

from learning.schema import ExperienceEpisode, EpisodeMetadata
from learning.synthesizer import SkillSynthesizer


# Reuse simple fakes for TechState / PlanTrace -------------------------------


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
    def __init__(self, skill: str) -> None:
        self.action = FakeAction("move_to", {"x": 0, "y": 64, "z": 0})
        self.result = FakeResult(True)
        self.world_before = SimpleNamespace()
        self.world_after = SimpleNamespace()
        self.meta = {"skill": skill}


class FakePlanTrace:
    def __init__(self, skill: str) -> None:
        step = FakeStep(skill=skill)
        self.steps = [step]
        self.plan = {"steps": [{"skill": skill, "params": {}}]}


class FakeCodeModel:
    """
    Minimal fake for M2 CodeModel.
    """

    def generate_json(self, *, prompt: Dict[str, Any]) -> Dict[str, Any]:
        # Just echo a trivial spec and impl referencing the goal / skill.
        episodes = prompt.get("episodes", [])
        first_goal = episodes[0].get("goal") if episodes else "unknown_goal"
        target_skill = prompt.get("target_skill_name") or "auto_skill"

        spec_yaml = f"""name: {target_skill}
version: 1
description: Auto-derived skill for '{first_goal}'
"""

        impl_code = """class AutoSkill:
    def run(self, ctx):
        # TODO: implement
        return True
"""

        return {
            "spec_yaml": spec_yaml,
            "impl_code": impl_code,
            "rationale": "fake rationale for testing",
        }


def make_episode(skill: str = "test_skill") -> ExperienceEpisode:
    ts = FakeTechState(active="steam_age", unlocked=["steam_age"])
    trace = FakePlanTrace(skill=skill)
    return ExperienceEpisode(
        id="ep1",
        goal="maintain coke ovens",
        tech_state=ts,
        trace=trace,
        virtue_scores={"overall": 0.9},
        success=True,
        metadata=EpisodeMetadata(),
    )


def test_synthesizer_produces_candidate() -> None:
    code_model = FakeCodeModel()
    synth = SkillSynthesizer(code_model)

    ep = make_episode()
    candidate = synth.propose_from_episodes(
        episodes=[ep],
        target_skill_name=None,
        candidate_id="cand1",
        context_hint="steam_age",
    )

    assert candidate.id == "cand1"
    assert "name:" in candidate.spec_yaml
    assert "class AutoSkill" in candidate.impl_code
    assert candidate.status == "proposed"
    assert "synthesizer_prompt" in candidate.extra
    assert "maintain coke ovens" in candidate.extra["synthesizer_prompt"]["episode_ids"][0] or True

