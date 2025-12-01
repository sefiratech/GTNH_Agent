# path: tests/test_curriculum_engine_phase.py

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

from curriculum.loader import load_curriculum
from curriculum.engine import CurriculumEngine
from semantics.schema import TechState
from spec.types import WorldState


def _write_yaml(path: Path, text: str) -> None:
    """
    Dedent YAML text so that all top-level keys line up correctly.
    """
    normalized = dedent(text).lstrip("\n")
    path.write_text(normalized, encoding="utf-8")


def _world_with_machines(machines: list[dict] | None = None) -> WorldState:
    """
    Construct a minimal WorldState with machine context.

    If your WorldState signature ever changes, adjust this dummy factory.
    """
    return WorldState(
        tick=1,
        position={"x": 0, "y": 64, "z": 0},
        dimension="overworld",
        inventory=[],
        nearby_entities=[],
        blocks_of_interest=[],
        tech_state={},
        context={"machines": machines or []},
    )


def test_phase_selection_basic_progression(tmp_path: Path) -> None:
    """
    Correct phase chosen for a given tech_state.
    """
    yaml_text = """
    id: "phase_selection_test"
    name: "Phase Selection Test"
    description: ""
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
          must_have: []
          preferred: []
        completion_conditions:
          tech_unlocked: ["lv_age"]
          machines_present: []
      - id: "lv_bootstrap"
        name: "LV Bootstrap"
        tech_targets:
          required_active: "lv_age"
          required_unlocked: []
        goals:
          - id: "g2"
            description: "Set up LV grid."
        virtue_overrides: {}
        skill_focus:
          must_have: []
          preferred: []
        completion_conditions:
          tech_unlocked: []
          machines_present: []
    long_horizon_projects: []
    """
    path = tmp_path / "phases.yaml"
    _write_yaml(path, yaml_text)

    cfg = load_curriculum(path)
    engine = CurriculumEngine(cfg)

    # Early steam tech_state
    tech_steam = TechState(unlocked=[], active="steam_age", evidence={})
    world = _world_with_machines()

    view_steam = engine.view(tech_steam, world)
    assert view_steam.phase_view.phase.id == "steam_early"
    assert not view_steam.phase_view.is_complete
    assert "Make steam." in view_steam.phase_view.active_goals

    # LV tech_state
    tech_lv = TechState(unlocked=["lv_age"], active="lv_age", evidence={})
    view_lv = engine.view(tech_lv, world)
    assert view_lv.phase_view.phase.id == "lv_bootstrap"
    assert "Set up LV grid." in view_lv.phase_view.active_goals


def test_phase_completion_with_machines(tmp_path: Path) -> None:
    """
    Phase transitions when completion conditions are met:
    tech unlock + machine presence.
    """
    yaml_text = """
    id: "phase_complete_test"
    name: "Phase Completion Test"
    description: ""
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
          tech_unlocked: ["lv_age"]
          machines_present:
            - { type: "large_boiler", min_count: 1 }
    long_horizon_projects: []
    """
    path = tmp_path / "phases_complete.yaml"
    _write_yaml(path, yaml_text)

    cfg = load_curriculum(path)
    engine = CurriculumEngine(cfg)

    # Not enough tech, no machines
    tech = TechState(unlocked=["steam_machines"], active="steam_age", evidence={})
    world = _world_with_machines()
    view = engine.view(tech, world)
    assert view.phase_view.phase.id == "steam_early"
    assert view.phase_view.is_complete is False

    # Add machine, still missing lv_age
    world2 = _world_with_machines(machines=[{"type": "large_boiler"}])
    view2 = engine.view(tech, world2)
    assert view2.phase_view.is_complete is False

    # Add lv_age unlock → now complete
    tech2 = TechState(
        unlocked=["steam_machines", "lv_age"],
        active="steam_age",
        evidence={},
    )
    view3 = engine.view(tech2, world2)
    assert view3.phase_view.is_complete is True

