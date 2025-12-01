# tests/test_semantics_categorization.py
"""
Unit tests for basic block/item categorization in SemanticsDB.

Goal:
  - Verify that SemanticsDB respects gtnh_blocks.yaml / gtnh_items.yaml
    structure and populates BlockInfo / ItemInfo correctly.
  - Keep tests hermetic by using a temporary config directory instead of
    the real project config/.
"""

from pathlib import Path

from semantics.loader import SemanticsDB
from semantics.schema import ItemInfo, BlockInfo
import semantics.loader as loader_module


def _write_minimal_configs(config_dir: Path) -> None:
    """Create minimal gtnh_blocks.yaml and gtnh_items.yaml in config_dir."""
    blocks_yaml = """
blocks:
  "minecraft:stone":
    default_category: "stone"

  "gregtech:gt.blockmachines":
    default_category: "gt_machine"
    variants:
      "basic_machine_lv":
        category: "gt_machine"
        tier: "lv"
      "basic_machine_mv":
        category: "gt_machine"
        tier: "mv"
""".lstrip()

    items_yaml = """
items:
  "gregtech:gt.metaitem.01":
    variants:
      "plate.copper":
        category: "plate"
        material: "copper"
      "dust.copper":
        category: "dust"
        material: "copper"

  "minecraft:iron_ingot":
    default_category: "ingot"
    material: "iron"
""".lstrip()

    recipes_json = '{"recipes": []}\n'

    (config_dir / "gtnh_blocks.yaml").write_text(blocks_yaml, encoding="utf-8")
    (config_dir / "gtnh_items.yaml").write_text(items_yaml, encoding="utf-8")
    (config_dir / "gtnh_recipes.json").write_text(recipes_json, encoding="utf-8")


def test_semanticsdb_uses_config_dir_for_item_and_block_info(tmp_path, monkeypatch):
    """
    SemanticsDB should load from semantics.loader.CONFIG_DIR and interpret
    YAML entries into ItemInfo / BlockInfo correctly.
    """
    # Arrange: point CONFIG_DIR at a temp "config" directory
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    _write_minimal_configs(config_dir)

    # Monkeypatch the CONFIG_DIR used inside semantics.loader
    monkeypatch.setattr(loader_module, "CONFIG_DIR", config_dir, raising=True)

    # Act: construct SemanticsDB with the test configs
    db = SemanticsDB()

    # ---- Item categorization ----
    item_info = db.get_item_info("gregtech:gt.metaitem.01", "plate.copper")
    assert isinstance(item_info, ItemInfo)
    assert item_info.item_id == "gregtech:gt.metaitem.01"
    assert item_info.variant == "plate.copper"
    assert item_info.category == "plate"
    assert item_info.material == "copper"
    # attributes should contain any extra keys beyond category/material
    assert isinstance(item_info.attributes, dict)

    iron_info = db.get_item_info("minecraft:iron_ingot", None)
    assert iron_info.category == "ingot"
    assert iron_info.material == "iron"

    # ---- Block categorization ----
    block_info_default = db.get_block_info("minecraft:stone", None)
    assert isinstance(block_info_default, BlockInfo)
    assert block_info_default.block_id == "minecraft:stone"
    assert block_info_default.category == "stone"

    block_lv = db.get_block_info("gregtech:gt.blockmachines", "basic_machine_lv")
    assert block_lv.category == "gt_machine"
    assert block_lv.attributes.get("tier") == "lv"

    block_mv = db.get_block_info("gregtech:gt.blockmachines", "basic_machine_mv")
    assert block_mv.category == "gt_machine"
    assert block_mv.attributes.get("tier") == "mv"

