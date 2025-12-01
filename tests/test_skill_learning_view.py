# path: tests/test_skill_learning_view.py
import types
from pathlib import Path

from learning.manager import SkillLearningManager, SkillView
from learning.buffer import ExperienceBuffer
from learning.schema import SkillPerformanceStats
from skills.registry import SkillRegistry


class FakeExperienceBuffer(ExperienceBuffer):
    """Replay buffer stub: no experiences, just satisfies the interface."""

    def __init__(self) -> None:
        # Avoid touching the filesystem; we never call append() here.
        self._path = Path("tests/_fake_replay.jsonl")

    def load_all(self):
        # No experiences in this test; skill view must not explode.
        if False:
            yield  # pragma: no cover


class FakeSkillRegistry(SkillRegistry):
    """Minimal registry stub that only implements describe_all()."""

    def __init__(self, meta: dict) -> None:
        self._meta = meta

    def describe_all(self) -> dict:
        return self._meta


def make_manager_with_registry(meta: dict) -> SkillLearningManager:
    buf = FakeExperienceBuffer()

    # Synthesizer / evaluator are not used for build_skill_view; stub them.
    fake_synth = types.SimpleNamespace()
    fake_eval = types.SimpleNamespace()

    registry = FakeSkillRegistry(meta)

    return SkillLearningManager(
        buffer=buf,
        synthesizer=fake_synth,
        evaluator=fake_eval,
        skills=registry,
        candidates_dir=Path("tests/_skills_candidates"),
    )


def test_build_skill_view_active_vs_candidates_structural():
    meta = {
        "stable_skill": {
            "metadata": {"status": "active"},
        },
        "accepted_skill": {
            "metadata": {"status": "accepted"},
        },
        "stable_label_skill": {
            "metadata": {"status": "stable"},
        },
        "candidate_skill": {
            "metadata": {"status": "candidate"},
        },
        "retired_skill": {
            "metadata": {"status": "retired"},
        },
    }

    mgr = make_manager_with_registry(meta)

    # Case 1: include_candidates=False → only stable/accepted/stable-label skills
    view_no_candidates: SkillView = mgr.build_skill_view(include_candidates=False)
    assert sorted(view_no_candidates.active_skills) == sorted(
        ["stable_skill", "accepted_skill", "stable_label_skill"]
    )
    assert view_no_candidates.candidate_skills == []

    # Case 2: include_candidates=True → candidate appears
    view_with_candidates: SkillView = mgr.build_skill_view(include_candidates=True)
    assert sorted(view_with_candidates.active_skills) == sorted(
        ["stable_skill", "accepted_skill", "stable_label_skill"]
    )
    assert view_with_candidates.candidate_skills == ["candidate_skill"]

