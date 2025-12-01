# tests/test_skill_loader.py

from pathlib import Path

from skills.loader import load_skill_spec_from_file, load_all_skill_specs
from skills.schema import (
    SkillSpec,
    SkillPreconditions,
    SkillEffects,
    SkillMetrics,
    ParamSpec,
)


def test_load_skill_spec_from_file_basic(tmp_path: Path) -> None:
    yaml_content = """
name: "chop_tree"
version: 2
status: "active"
origin: "manual"
description: "Test chop tree skill"

params:
  radius:
    type: int
    default: 8
    description: "Search radius in blocks."
  min_logs:
    type: int
    default: 4
    description: "Minimum logs to gather."

preconditions:
  tech_states_any_of: ["stone_age", "steam_age"]
  required_tools:
    - item_id: "minecraft:iron_axe"
      min_durability: 10
  dimension_allowlist: ["overworld"]
  semantic_tags_any_of: ["tree"]

effects:
  inventory_delta:
    logs:
      min_increase: 4
  tech_delta: {}
  tags: ["gathers_logs"]

tags:
  - "mining"
  - "progression"

metrics:
  success_rate: 0.9
  avg_cost: 42
"""
    path = tmp_path / "chop_tree.yaml"
    path.write_text(yaml_content, encoding="utf-8")

    spec = load_skill_spec_from_file(path)

    assert isinstance(spec, SkillSpec)
    assert spec.name == "chop_tree"
    assert spec.version == 2
    assert spec.status == "active"
    assert spec.origin == "manual"
    assert spec.description == "Test chop tree skill"

    assert "radius" in spec.params
    assert isinstance(spec.params["radius"], ParamSpec)
    assert spec.params["radius"].default == 8

    assert "stone_age" in spec.preconditions.tech_states_any_of
    assert spec.preconditions.dimension_allowlist == ["overworld"]
    assert spec.preconditions.semantic_tags_any_of == ["tree"]

    assert spec.effects.inventory_delta["logs"]["min_increase"] == 4
    assert "gathers_logs" in spec.effects.tags

    assert "mining" in spec.tags
    assert spec.metrics.success_rate == 0.9
    assert spec.metrics.avg_cost == 42


def test_load_all_skill_specs_duplicate_name_raises(tmp_path: Path) -> None:
    # Two files with the same 'name' should cause a ValueError
    yaml_a = """
name: "duplicate_skill"
version: 1
status: "active"
origin: "manual"
"""
    yaml_b = """
name: "duplicate_skill"
version: 2
status: "candidate"
origin: "manual"
"""

    path_a = tmp_path / "a.yaml"
    path_b = tmp_path / "b.yaml"
    path_a.write_text(yaml_a, encoding="utf-8")
    path_b.write_text(yaml_b, encoding="utf-8")

    # load_all_skill_specs should raise
    try:
        load_all_skill_specs(skills_dir=tmp_path)
    except ValueError as e:
        assert "Duplicate skill spec name 'duplicate_skill'" in str(e)
    else:
        assert False, "Expected ValueError for duplicate skill spec names"

