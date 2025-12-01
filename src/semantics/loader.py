# src/semantics/loader.py
"""
Semantic config loader for GTNH blocks/items/recipes.

Responsibility:
  - Load YAML/JSON config files from CONFIG_DIR
  - Merge hand-authored and generated configs:
      * gtnh_blocks.yaml              (base)
      * gtnh_blocks.generated.yaml    (auto-ingested, optional)
      * gtnh_items.yaml               (base)
      * gtnh_items.generated.yaml     (auto-ingested, optional)
      * gtnh_recipes.json             (base, hand-authored overrides)
      * gtnh_recipes.agent.json       (preferred compact auto-ingested set)
      * gtnh_recipes.generated.json   (fallback auto-ingested set)
  - Provide a simple in-memory index for:
      * blocks
      * items
      * recipes
  - Map those into BlockInfo / ItemInfo dataclasses.

This is deliberately boring plumbing so the rest of the system can pretend
it has “world semantics” without hand-rolling CSV parsing in three places.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List

import json
import yaml

from .schema import BlockInfo, ItemInfo


# Default config directory; tests monkeypatch this to point at a temp dir.
CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"


# ---------------------------------------------------------------------------
# Low-level loaders
# ---------------------------------------------------------------------------

def _load_yaml(name: str) -> Dict[str, Any]:
    """
    Load a YAML config file from CONFIG_DIR and return it as a dict.

    Raises FileNotFoundError if the file does not exist.
    """
    path = CONFIG_DIR / name
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML config {path} must be a mapping at top level.")
    return data


def _load_yaml_optional(name: str) -> Dict[str, Any]:
    """
    Load a YAML config file if it exists, otherwise return {}.
    """
    path = CONFIG_DIR / name
    if not path.exists():
        return {}
    return _load_yaml(name)


def _load_json(name: str) -> Dict[str, Any]:
    """
    Load a JSON config file from CONFIG_DIR and return it as a dict.

    Raises FileNotFoundError if the file does not exist.
    """
    path = CONFIG_DIR / name
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"JSON config {path} must be a mapping at top level.")
    return data


def _load_json_optional(name: str) -> Dict[str, Any]:
    """
    Load a JSON config file if it exists, otherwise return {}.
    """
    path = CONFIG_DIR / name
    if not path.exists():
        return {}
    return _load_json(name)


# ---------------------------------------------------------------------------
# Merge helpers
# ---------------------------------------------------------------------------

def _merge_mapping_with_precedence(
    base: Dict[str, Any],
    generated: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Merge two top-level mappings for items/blocks.

    - generated: auto-ingested data
    - base:      hand-authored data

    For a given key (item_id/block_id):
      - if present in base → base entry wins entirely
      - else → generated entry is used
    """
    result: Dict[str, Any] = dict(generated or {})
    result.update(base or {})
    return result


def _merge_recipe_lists(
    base_recipes: List[Dict[str, Any]],
    generated_recipes: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Merge recipe lists by recipe id.

    - generated recipes are loaded first
    - base recipes override any recipe with the same id

    This lets hand-authored recipes correct or replace ingested ones.
    """
    by_id: Dict[str, Dict[str, Any]] = {}

    for r in generated_recipes or []:
        rid = r.get("id")
        if isinstance(rid, str):
            by_id[rid] = r

    for r in base_recipes or []:
        rid = r.get("id")
        if isinstance(rid, str):
            by_id[rid] = r

    return list(by_id.values())


# ---------------------------------------------------------------------------
# SemanticsDB
# ---------------------------------------------------------------------------

class SemanticsDB:
    """
    In-memory semantic database for GTNH blocks/items/recipes.

    Config expectations:

      gtnh_blocks.yaml / gtnh_blocks.generated.yaml:
        blocks:
          "mod:block":
            default_category: "something"
            # optional extra attributes at top level
            variants:
              "variant_key":
                category: "..."
                # extra attributes per variant

      gtnh_items.yaml / gtnh_items.generated.yaml:
        items:
          "mod:item":
            default_category: "..."
            material: "iron"        # optional default material
            variants:
              "plate.copper":
                category: "plate"
                material: "copper"

    Recipes are read from:
      - gtnh_recipes.agent.json       (preferred compact set)
      - gtnh_recipes.generated.json   (fallback large set)
      - gtnh_recipes.json             (hand-authored overrides)
    and merged by id, with hand-authored base recipes overriding the auto-ingested ones.
    """

    def __init__(self) -> None:
        # Blocks: merge generated + base, base wins
        blocks_base = _load_yaml_optional("gtnh_blocks.yaml")
        blocks_gen = _load_yaml_optional("gtnh_blocks.generated.yaml")
        blocks_merged = _merge_mapping_with_precedence(
            base=blocks_base.get("blocks", {}) if isinstance(blocks_base, dict) else {},
            generated=blocks_gen.get("blocks", {}) if isinstance(blocks_gen, dict) else {},
        )

        # Items: merge generated + base, base wins
        items_base = _load_yaml_optional("gtnh_items.yaml")
        items_gen = _load_yaml_optional("gtnh_items.generated.yaml")
        items_merged = _merge_mapping_with_precedence(
            base=items_base.get("items", {}) if isinstance(items_base, dict) else {},
            generated=items_gen.get("items", {}) if isinstance(items_gen, dict) else {},
        )

        # Recipes:
        # - Prefer compact agent file
        # - Fall back to full generated
        # - Always let hand-authored gtnh_recipes.json override both
        recipes_base_cfg = _load_json_optional("gtnh_recipes.json")
        recipes_agent_cfg = _load_json_optional("gtnh_recipes.agent.json")
        recipes_generated_cfg = _load_json_optional("gtnh_recipes.generated.json")

        if recipes_agent_cfg:
            generated_list = recipes_agent_cfg.get("recipes", []) if isinstance(recipes_agent_cfg, dict) else []
        else:
            generated_list = recipes_generated_cfg.get("recipes", []) if isinstance(recipes_generated_cfg, dict) else []

        base_list = recipes_base_cfg.get("recipes", []) if isinstance(recipes_base_cfg, dict) else []

        recipes_merged = _merge_recipe_lists(base_list, generated_list)

        self._block_index: Dict[str, Dict[str, Any]] = blocks_merged
        self._item_index: Dict[str, Dict[str, Any]] = items_merged
        self._recipes: List[Dict[str, Any]] = recipes_merged

    # ------------------------------------------------------------------
    # Block lookup
    # ------------------------------------------------------------------

    def get_block_info(self, block_id: str, variant: str | None) -> BlockInfo:
        """
        Return semantic info for a given block id/variant.

        If the block id is unknown, category defaults to "unknown" and
        attributes are an empty dict.
        """
        entry = self._block_index.get(block_id, {})

        # Default category can be provided either as "default_category"
        # or as a shorthand "category" at top level.
        default_category = entry.get(
            "default_category",
            entry.get("category", "unknown"),
        )

        variants = entry.get("variants", {}) or {}

        if variant and variant in variants:
            v = variants[variant]
            category = v.get("category", default_category)
            attributes = {
                k: v for k, v in v.items()
                if k != "category"
            }
        else:
            category = default_category
            attributes = {
                k: v
                for k, v in entry.items()
                if k not in ("default_category", "category", "variants")
            }

        return BlockInfo(
            block_id=block_id,
            variant=variant,
            category=category,
            attributes=attributes,
        )

    # ------------------------------------------------------------------
    # Item lookup
    # ------------------------------------------------------------------

    def get_item_info(self, item_id: str, variant: str | None) -> ItemInfo:
        """
        Return semantic info for a given item id/variant.

        If the item id is unknown, category defaults to "unknown" and
        material defaults to None.
        """
        entry = self._item_index.get(item_id, {})

        # Default category: support both "default_category" and "category"
        default_category = entry.get(
            "default_category",
            entry.get("category", "unknown"),
        )
        # Default material at top level (optional)
        material = entry.get("material")

        variants = entry.get("variants", {}) or {}

        if variant and variant in variants:
            v = variants[variant]
            category = v.get("category", default_category)
            # Variant may override material
            material = v.get("material", material)
            attributes = {
                k: v
                for k, v in v.items()
                if k not in ("category", "material")
            }
        else:
            category = default_category
            attributes = {
                k: v
                for k, v in entry.items()
                if k not in ("default_category", "category", "variants", "material")
            }

        return ItemInfo(
            item_id=item_id,
            variant=variant,
            category=category,
            material=material,
            attributes=attributes,
        )

    # ------------------------------------------------------------------
    # Recipes
    # ------------------------------------------------------------------

    @property
    def recipes(self) -> List[Dict[str, Any]]:
        """Return the raw recipes list."""
        return self._recipes


__all__ = [
    "SemanticsDB",
]

