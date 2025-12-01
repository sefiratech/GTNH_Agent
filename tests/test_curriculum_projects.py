# path: tests/test_curriculum_projects.py

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

from curriculum.loader import load_curriculum
from curriculum.engine import CurriculumEngine
from semantics.schema import TechState
from spec.types import WorldState


def _write_yaml(path: Path, text: str) -> None:
    """
    Dedent YAML to keep top-level keys aligned.
    """
    normalized = dedent(text).lstrip("\n")
    path.write_text(normalized, encoding="utf-8")


def _world() -> WorldState:
    """
    Minimal WorldState stub for project unlock tests.
    """
    return WorldState(
        tick=1,
        position={"x": 0, "y": 64, "z": 0},
        dimension="overworld",
        inventory=[],
        nearby_entities=[],
        blocks_of_interest=[],
        tech_state={},
        context={"machines": []},
    )


def test_project_unlocks_after_required_phases(tmp_path: Path) -> None:
    """
    Ensure long-horizon projects unlock properly once all required phases
    in a stage's depends_on_phases are completed.

    With the current simple semantics:

      * A phase is "complete" if its completion_conditions are satisfied,
        regardless of whether its tech_targets match the current active tier.
      * A project unlocks if ANY of its stages has all depends_on_phases
        in the completed phase set.

    So once mv_automation's completion conditions are met AND hv_age's
    completion conditions (empty) are trivially satisfied, the project
    unlocks.
    """
    yaml_text = """
    id: "project_unlock_test"
    name: "Project Unlock Test"
    description: ""
    phases:
      - id: "mv_automation"
        name: "MV Automation"
        tech_targets:
          required_active: "mv_age"
          required_unlocked: []
        goals:
          - id: "g_mv"
            description: "Do MV things."
        virtue_overrides: {}
        skill_focus:
          must_have: []
          preferred: []
        completion_conditions:
          tech_unlocked: ["hv_age"]
          machines_present: []
      - id: "hv_age"
        name: "HV Age"
        tech_targets:
          required_active: "hv_age"
          required_unlocked: []
        goals:
          - id: "g_hv"
            description: "Do HV things."
        virtue_overrides: {}
        skill_focus:
          must_have: []
          preferred: []
        completion_conditions:
          tech_unlocked: []
          machines_present: []
    long_horizon_projects:
      - id: "stargate_project"
        name: "Stargate Construction"
        description: "Complete the Stargate and related infrastructure."
        stages:
          - id: "preparation"
            description: "Mass automation of high-tier materials."
            depends_on_phases:
              - "mv_automation"
              - "hv_age"
    """
    path = tmp_path / "project_unlock.yaml"
    _write_yaml(path, yaml_text)

    cfg = load_curriculum(path)
    engine = CurriculumEngine(cfg)

    # State 1: mv_age active, no unlocks → no project
    tech = TechState(unlocked=[], active="mv_age", evidence={})
    world = _world()
    view1 = engine.view(tech, world)
    assert view1.unlocked_projects == []

    # State 2: hv_age is unlocked; mv_automation completion conditions are satisfied.
    # hv_age completion conditions are empty → also satisfied.
    # Both mv_automation and hv_age count as "completed" → project unlocks here.
    tech_mv_done = TechState(unlocked=["hv_age"], active="mv_age", evidence={})
    view2 = engine.view(tech_mv_done, world)
    assert any(p.id == "stargate_project" for p in view2.unlocked_projects)

    # State 3: active tier is hv_age; semantics still report the same project as unlocked.
    tech_hv = TechState(unlocked=["hv_age"], active="hv_age", evidence={})
    view3 = engine.view(tech_hv, world)
    assert any(p.id == "stargate_project" for p in view3.unlocked_projects)

