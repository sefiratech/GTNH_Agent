# path: tests/test_experience_buffer.py

from __future__ import annotations

from typing import Any, Dict, List

from pathlib import Path

from learning.buffer import ExperienceBuffer
from learning.schema import Experience


class DummyExperience:
    """
    Minimal stand-in for Experience that satisfies ExperienceBuffer.append*.

    We let ExperienceBuffer.serialize via .to_dict(), then later
    we verify that Experience.from_dict(...) can reconstruct objects
    via ExperienceBuffer.load_all().
    """

    def __init__(self, payload: Dict[str, Any]) -> None:
        self._payload = payload

    def to_dict(self) -> Dict[str, Any]:
        return self._payload


def _make_payload(goal_id: str, success: bool = True) -> Dict[str, Any]:
    """
    Construct a dict matching the expected Experience serialization shape.

    This mirrors the comments in learning.buffer and episode_logging:
      {
        "problem_signature": {...},
        "goal": {...},
        "plan": {...},
        "attempts": [...],
        "final_outcome": {...},
        "virtue_scores": {...},
        "lessons": "..."
      }
    """
    return {
        "problem_signature": {
            "goal_id": goal_id,
            "goal_text": f"Goal {goal_id}",
            "phase": "demo_p0",
            "tech_state": {"active": "stone_age"},
        },
        "goal": {
            "id": goal_id,
            "text": f"Goal {goal_id}",
            "phase": "demo_p0",
            "source": "curriculum",
        },
        "plan": {
            "task_plan": {"goal_id": goal_id, "tasks": []},
            "skill_invocations": [],
        },
        "attempts": [],
        "final_outcome": {
            "success": success,
            "step_count": 0,
        },
        "virtue_scores": {},
        "lessons": f"Test experience for {goal_id}",
    }


def test_experience_buffer_append_and_count(tmp_path) -> None:
    """Appending experiences should create one non-empty line per experience."""
    path: Path = tmp_path / "replay.jsonl"
    buffer = ExperienceBuffer(path=path)

    # Initially empty
    assert buffer.count() == 0

    # Append two dummy experiences
    exp1 = DummyExperience(_make_payload("g1", success=True))
    exp2 = DummyExperience(_make_payload("g2", success=False))

    buffer.append_experience(exp1)
    buffer.append_experience(exp2)

    # Count should reflect 2 non-empty lines
    assert buffer.count() == 2

    # Underlying file should exist and be non-empty
    assert path.exists()
    contents = path.read_text(encoding="utf-8").strip().splitlines()
    assert len([ln for ln in contents if ln.strip()]) == 2


def test_experience_buffer_load_all_raw_matches_written(tmp_path) -> None:
    """load_all_raw() should return the same dicts we serialized."""
    path: Path = tmp_path / "replay_raw.jsonl"
    buffer = ExperienceBuffer(path=path)

    payloads: List[Dict[str, Any]] = [
        _make_payload("g1", success=True),
        _make_payload("g2", success=False),
    ]

    for p in payloads:
        buffer.append_experience(DummyExperience(p))

    raw = list(buffer.load_all_raw())
    assert len(raw) == len(payloads)

    # Compare a couple of key fields to make sure round-trip is sane
    for original, loaded in zip(payloads, raw):
        assert loaded["goal"]["id"] == original["goal"]["id"]
        assert loaded["final_outcome"]["success"] == original["final_outcome"]["success"]


def test_experience_buffer_load_all_returns_experience_objects(tmp_path) -> None:
    """load_all() should reconstruct typed Experience objects via Experience.from_dict()."""
    path: Path = tmp_path / "replay_typed.jsonl"
    buffer = ExperienceBuffer(path=path)

    buffer.append_experience(DummyExperience(_make_payload("g1", success=True)))
    buffer.append_experience(DummyExperience(_make_payload("g2", success=False)))

    typed = list(buffer.load_all())
    assert len(typed) == 2

    # Ensure we actually got Experience instances back, not raw dicts
    for exp in typed:
        assert isinstance(exp, Experience)
        # Spot-check a couple of attributes expected from the schema
        assert getattr(exp.goal, "id", None) in {"g1", "g2"}
        assert isinstance(exp.final_outcome, dict)
        assert "success" in exp.final_outcome

