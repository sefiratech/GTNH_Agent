# tests/test_semantics_craftability.py
"""
Craftability tests for semantics.crafting.

Goal:
  - Given a minimal recipe set and a world with the right inventory + machine,
    craftable_items should surface the expected recipe as craftable.
"""

from pathlib import Path

from spec.types import WorldState
from semantics.crafting import craftable_items
from semantics.loader import SemanticsDB
import semantics.loader as loader_module


def _write_minimal_semantics_configs(config_dir: Path) -> None:
    """
    Create minimal gtnh_blocks.yaml, gtnh_items.yaml, and gtnh_recipes.json
    in config_dir for craftability tests.
    """
    blocks_yaml = """
blocks:
  "gregtech:gt.blockores":
    default_category: "ore"
    variants:
      "copper_ore":
        material: "copper"

  "gregtech:gt.blockmachines":
    default_category: "gt_machine"
    variants:
      "steam_macerator":
        category: "gt_machine"
        tier: "steam"
""".lstrip()

    items_yaml = """
items:
  "gregtech:gt.metaitem.01":
    variants:
      "dust.copper":
        category: "dust"
        material: "copper"

  "minecraft:iron_ingot":
    default_category: "ingot"
    material: "iron"
""".lstrip()

    # One machine recipe: copper ore -> copper dust via steam macerator
    recipes_json = """
{
  "recipes": [
    {
      "id": "gt:steam_macerator_copper_ore",
      "type": "machine",
      "machine": "gregtech:steam_macerator",
      "tier": "steam",
      "io": {
        "inputs": [
          {
            "item": "gregtech:gt.blockores",
            "variant": "copper_ore",
            "count": 1
          }
        ],
        "outputs": [
          {
            "item": "gregtech:gt.metaitem.01",
            "variant": "dust.copper",
            "count": 2
          }
        ],
        "byproducts": []
      }
    }
  ]
}
""".lstrip()

    (config_dir / "gtnh_blocks.yaml").write_text(blocks_yaml, encoding="utf-8")
    (config_dir / "gtnh_items.yaml").write_text(items_yaml, encoding="utf-8")
    (config_dir / "gtnh_recipes.json").write_text(recipes_json, encoding="utf-8")


def test_craftable_items_with_steam_macerator_and_copper_ore(tmp_path, monkeypatch):
    """
    With a steam macerator present and enough copper ore in inventory,
    the copper-ore-to-dust recipe should be reported as craftable.
    """
    # Arrange: minimal configs in a temp config dir
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    _write_minimal_semantics_configs(config_dir)

    # Point semantics.loader.CONFIG_DIR at our temp config dir
    monkeypatch.setattr(loader_module, "CONFIG_DIR", config_dir, raising=True)

    # Build a world state that has:
    #   - 3x copper ore
    #   - a steam macerator machine
    world = WorldState(
        tick=0,
        position={"x": 0.0, "y": 64.0, "z": 0.0},
        dimension="overworld",
        inventory=[
            {
                "item_id": "gregtech:gt.blockores",
                "variant": "copper_ore",
                "count": 3,
            }
        ],
        nearby_entities=[],
        blocks_of_interest=[],
        tech_state={},
        context={
            "machines": [
                # Using "type" here; _extract_machine_ids_from_context should pick it up
                {"type": "gregtech:steam_macerator", "tier": "steam"},
            ]
        },
    )

    db = SemanticsDB()

    # Act: compute craftable options
    options = craftable_items(world, db)

    # Assert: exactly one option, and it's the copper ore recipe
    assert len(options) == 1

    opt = options[0]
    assert opt.recipe_id == "gt:steam_macerator_copper_ore"
    assert opt.machine_id == "gregtech:steam_macerator"
    assert opt.tier == "steam"

    # Primary output should be copper dust
    primary = opt.primary_output
    assert primary.item_id == "gregtech:gt.metaitem.01"
    assert primary.variant == "dust.copper"
    assert primary.count == 2

    # Since we have 3 ore and only require 1, the recipe is clearly craftable
    assert opt.limiting_resource is not None

