# tests/test_semantics_tech_inference.py
"""
Tech inference tests for semantics.tech_state.

Goal:
  - Verify that infer_tech_state_from_world uses the tech graph +
    world evidence (machines/flags) to infer a reasonable active state.
  - Verify that suggest_next_targets surfaces lv_electric as a target
    once steam_age is reached, even if lv requirements are not yet met.
"""

from pathlib import Path

from spec.types import WorldState
from semantics.tech_state import (
    infer_tech_state_from_world,
    TechGraph,
    suggest_next_targets,
)
import semantics.tech_state as tech_state_module


def _write_minimal_tech_graph(config_dir: Path) -> None:
    """Create a minimal gtnh_tech_graph.yaml in config_dir."""
    yaml_text = """
tech_states:
  stone_age:
    description: "No machines, stone tools only."
    tags: ["early-game"]
    prerequisites: []
    unlocks: ["steam_age"]
    requirements: {}
    recommended_goals:
      - "Punch trees"
      - "Make stone tools"

  steam_age:
    description: "Basic steam machines, bronze tools."
    tags: ["steam"]
    prerequisites: ["stone_age"]
    unlocks: ["lv_electric"]
    requirements:
      flags: ["has_steam_power"]
    recommended_goals:
      - "Build a steam macerator"
      - "Automate early ore processing"

  lv_electric:
    description: "LV power, LV machines, basic circuits."
    tags: ["lv"]
    prerequisites: ["steam_age"]
    unlocks: []
    # Intentionally require a flag we don't have yet so it stays locked
    requirements:
      flags: ["has_lv_power"]
    recommended_goals:
      - "Set up LV machines"
      - "Produce LV circuits"
""".lstrip()

    (config_dir / "gtnh_tech_graph.yaml").write_text(yaml_text, encoding="utf-8")


def test_tech_inference_with_steam_machine(tmp_path, monkeypatch):
    """
    Given a world with a steam machine but no LV power, we expect:

      - stone_age and steam_age to be unlocked
      - active tech state to be 'steam_age'
      - lv_electric to *not* be unlocked yet
      - lv_electric to appear in suggested next targets
    """
    # Arrange: minimal tech graph in a temp config dir
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    _write_minimal_tech_graph(config_dir)

    # Monkeypatch the CONFIG_DIR used inside semantics.tech_state
    monkeypatch.setattr(tech_state_module, "CONFIG_DIR", config_dir, raising=True)

    # Build a dummy world with a steam machine present
    world = WorldState(
        tick=0,
        position={"x": 0.0, "y": 64.0, "z": 0.0},
        dimension="overworld",
        inventory=[],              # no items yet
        nearby_entities=[],
        blocks_of_interest=[],
        tech_state={},
        context={
            # Uses "type" → _extract_machine_ids_from_context should see this
            "machines": [
                {"type": "steam_macerator"},
            ]
        },
    )

    graph = TechGraph()

    # Act: infer tech state from this world
    state = infer_tech_state_from_world(world, graph)

    # Assert: basic structure
    assert isinstance(state.active, str)
    assert isinstance(state.unlocked, list)

    # We expect to have at least stone_age and steam_age unlocked
    assert "stone_age" in state.unlocked
    assert "steam_age" in state.unlocked

    # LV should *not* be unlocked yet because we don't have has_lv_power
    assert "lv_electric" not in state.unlocked

    # Active state should be steam_age (the highest unlocked node)
    assert state.active == "steam_age"

    # And lv_electric should appear as a suggested next target
    targets = suggest_next_targets(state, graph)
    target_ids = {t.id for t in targets}
    assert "lv_electric" in target_ids

