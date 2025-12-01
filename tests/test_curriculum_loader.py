# path: tests/test_curriculum_loader.py

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest

from curriculum.loader import load_curriculum, list_curricula
from curriculum.schema import CurriculumConfig


def _write_yaml(path: Path, text: str) -> None:
    """
    Write a YAML string to disk, properly dedented so top-level keys align.

    Using dedent avoids the "id at column 1, name at column 5" nonsense
    that makes the YAML parser cry.
    """
    normalized = dedent(text).lstrip("\n")
    path.write_text(normalized, encoding="utf-8")


def test_load_curriculum_minimal_valid(tmp_path: Path) -> None:
    """
    Verify YAML loads into a valid CurriculumConfig with minimal fields.
    """
    yaml_text = """
    id: "test_curriculum"
    name: "Test Curriculum"
    description: "Minimal valid curriculum."
    phases:
      - id: "steam_early"
        name: "Steam Early"
        tech_targets:
          required_active: "steam_age"
          required_unlocked: ["steam_machines"]
        goals:
          - id: "g1"
            description: "Make steam."
        virtue_overrides: {}
        skill_focus:
          must_have: []
          preferred: []
        completion_conditions:
          tech_unlocked: []
          machines_present: []
    long_horizon_projects: []
    """
    path = tmp_path / "test_curriculum.yaml"
    _write_yaml(path, yaml_text)

    cfg = load_curriculum(path)
    assert isinstance(cfg, CurriculumConfig)
    assert cfg.id == "test_curriculum"
    assert len(cfg.phases) == 1
    phase = cfg.phases[0]
    assert phase.id == "steam_early"
    assert phase.tech_targets.required_active == "steam_age"
    assert phase.goals[0].description == "Make steam."


def test_load_curriculum_missing_required_fields(tmp_path: Path) -> None:
    """
    Validate that missing required top-level fields cause a ValueError.
    """
    yaml_text = """
    name: "No ID Here"
    description: "This curriculum is missing an id."
    phases: []
    long_horizon_projects: []
    """
    path = tmp_path / "broken_curriculum.yaml"
    _write_yaml(path, yaml_text)

    with pytest.raises(ValueError):
        load_curriculum(path)


def test_load_phase_missing_tech_targets(tmp_path: Path) -> None:
    """
    Validate that missing required phase keys cause a ValueError.
    """
    yaml_text = """
    id: "broken_curriculum"
    name: "Broken"
    description: ""
    phases:
      - id: "steam_early"
        name: "Steam Early"
        # tech_targets missing
        goals: []
        virtue_overrides: {}
        skill_focus:
          must_have: []
          preferred: []
        completion_conditions:
          tech_unlocked: []
          machines_present: []
    long_horizon_projects: []
    """
    path = tmp_path / "broken_curriculum.yaml"
    _write_yaml(path, yaml_text)

    with pytest.raises(ValueError):
        load_curriculum(path)


def test_list_curricula_reads_ids(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """
    Ensure list_curricula discovers YAMLs in CURRICULA_DIR and uses top-level id.
    """
    # Point CURRICULA_DIR at tmp_path
    monkeypatch.setattr("curriculum.loader.CURRICULA_DIR", tmp_path, raising=True)

    yaml_a = """
    id: "cur_a"
    name: "A"
    description: ""
    phases: []
    long_horizon_projects: []
    """
    yaml_b = """
    id: "cur_b"
    name: "B"
    description: ""
    phases: []
    long_horizon_projects: []
    """
    _write_yaml(tmp_path / "a.yaml", yaml_a)
    _write_yaml(tmp_path / "b.yaml", yaml_b)

    found = list_curricula()
    assert "cur_a" in found
    assert "cur_b" in found
    from pathlib import Path as _P  # avoid shadowing / type confusion
    assert all(isinstance(p, _P) for p in found.values())


def test_unknown_skill_names_dont_break_loader(tmp_path: Path) -> None:
    """
    Validate unknown skill names don't crash the loader.

    Actual validation of skill names (against a skill registry) is handled
    elsewhere. Here we just ensure arbitrary strings are accepted structurally.
    """
    yaml_text = """
    id: "skill_weirdness"
    name: "Weird Skills"
    description: "Uses arbitrary skill names."
    phases:
      - id: "phase1"
        name: "Phase 1"
        tech_targets:
          required_active: "steam_age"
          required_unlocked: []
        goals:
          - id: "g1"
            description: "Do something."
        virtue_overrides: {}
        skill_focus:
          must_have:
            - "this_skill_does_not_exist"
          preferred:
            - "another_fake_skill"
        completion_conditions:
          tech_unlocked: []
          machines_present: []
    long_horizon_projects: []
    """
    path = tmp_path / "skill_weirdness.yaml"
    _write_yaml(path, yaml_text)

    cfg = load_curriculum(path)
    phase = cfg.phases[0]
    assert "this_skill_does_not_exist" in phase.skill_focus.must_have
    assert "another_fake_skill" in phase.skill_focus.preferred

