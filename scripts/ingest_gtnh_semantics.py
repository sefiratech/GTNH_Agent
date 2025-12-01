# scripts/ingest_gtnh_semantics.py

"""
Ingest raw GTNH modpack dumps into M3 semantics configs.

Usage (from project root):

    (.venv) python3 scripts/ingest_gtnh_semantics.py \
        --config-dir config \
        --raw-dir config/raw

Expected raw files (JSON, produced by a GTNH-side mod/script):

  - {raw_dir}/registry_items.json
      {
        "items": [
          {
            "id": "minecraft:iron_ingot",
            "display_name": "Iron Ingot",
            "category_hint": "ingot",        # optional
            "material_hint": "iron",         # optional
            "tags": ["forge:ingots/iron"]    # optional
          },
          {
            "id": "gregtech:gt.metaitem.01",
            "variant": "plate.copper",       # optional
            "display_name": "Copper Plate",
            "category_hint": "plate",
            "material_hint": "copper",
            "tags": ["gt:plate", "material:copper"]
          }
        ]
      }

  - {raw_dir}/registry_blocks.json
      {
        "blocks": [
          {
            "id": "minecraft:stone",
            "display_name": "Stone",
            "category_hint": "stone",
            "tags": ["minecraft:stone"]
          },
          {
            "id": "gregtech:gt.blockmachines",
            "variant": "basic_machine_lv",
            "display_name": "Basic Machine LV",
            "category_hint": "gt_machine",
            "tier_hint": "lv",
            "tags": ["gt:machine", "tier:lv"]
          }
        ]
      }

  - {raw_dir}/recipes_raw.json
      {
        "recipes": [
          {
            "id": "gt:steam_macerator_copper_ore",
            "type": "machine",
            "machine": "gregtech:steam_macerator",
            "tier": "steam",
            "inputs": [
              { "item": "gregtech:gt.blockores", "variant": "copper_ore", "count": 1 }
            ],
            "outputs": [
              { "item": "gregtech:gt.metaitem.01", "variant": "dust.copper", "count": 2 }
            ],
            "byproducts": []
          }
        ]
      }

Output (into {config_dir}):

  - gtnh_items.generated.yaml
  - gtnh_blocks.generated.yaml
  - gtnh_recipes.generated.json

These are *merged* with hand-authored configs at runtime by SemanticsDB.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import yaml


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required raw dump: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Raw JSON {path} must be an object at top level.")
    return data


# ---------------------------------------------------------------------------
# Items
# ---------------------------------------------------------------------------

def build_items_generated(raw_items: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build gtnh_items.generated.yaml structure from registry_items.json.

    Output structure:

      items:
        "mod:item":
          default_category: "..."
          material: "iron"
          variants:
            "plate.copper":
              category: "plate"
              material: "copper"
              tags: [...]
    """
    items_section: Dict[str, Any] = {}
    items = raw_items.get("items", []) or []
    if not isinstance(items, list):
        raise ValueError("registry_items.json: 'items' must be a list.")

    for entry in items:
        if not isinstance(entry, dict):
            continue

        item_id = entry.get("id")
        if not isinstance(item_id, str):
            continue

        variant = entry.get("variant")
        cat_hint = entry.get("category_hint")
        mat_hint = entry.get("material_hint")
        tags = entry.get("tags") or []

        if not isinstance(tags, list):
            tags = [tags]

        item_cfg = items_section.setdefault(item_id, {})

        if variant:
            # Ensure variants dict exists
            variants = item_cfg.setdefault("variants", {})
            v_cfg = variants.setdefault(variant, {})

            if cat_hint:
                v_cfg.setdefault("category", cat_hint)
            if mat_hint:
                v_cfg.setdefault("material", mat_hint)
            if tags:
                v_cfg.setdefault("tags", tags)
        else:
            # Top-level (non-variant) item
            if cat_hint:
                item_cfg.setdefault("default_category", cat_hint)
            if mat_hint:
                item_cfg.setdefault("material", mat_hint)
            if tags:
                item_cfg.setdefault("tags", tags)

    return {"items": items_section}


# ---------------------------------------------------------------------------
# Blocks
# ---------------------------------------------------------------------------

def build_blocks_generated(raw_blocks: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build gtnh_blocks.generated.yaml structure from registry_blocks.json.

    Output structure:

      blocks:
        "mod:block":
          default_category: "stone"
          variants:
            "basic_machine_lv":
              category: "gt_machine"
              tier: "lv"
              tags: [...]
    """
    blocks_section: Dict[str, Any] = {}
    blocks = raw_blocks.get("blocks", []) or []
    if not isinstance(blocks, list):
        raise ValueError("registry_blocks.json: 'blocks' must be a list.")

    for entry in blocks:
        if not isinstance(entry, dict):
            continue

        block_id = entry.get("id")
        if not isinstance(block_id, str):
            continue

        variant = entry.get("variant")
        cat_hint = entry.get("category_hint")
        tier_hint = entry.get("tier_hint")
        tags = entry.get("tags") or []

        if not isinstance(tags, list):
            tags = [tags]

        block_cfg = blocks_section.setdefault(block_id, {})

        if variant:
            variants = block_cfg.setdefault("variants", {})
            v_cfg = variants.setdefault(variant, {})

            if cat_hint:
                v_cfg.setdefault("category", cat_hint)
            if tier_hint:
                v_cfg.setdefault("tier", tier_hint)
            if tags:
                v_cfg.setdefault("tags", tags)
        else:
            if cat_hint:
                block_cfg.setdefault("default_category", cat_hint)
            if tags:
                block_cfg.setdefault("tags", tags)

    return {"blocks": blocks_section}


# ---------------------------------------------------------------------------
# Recipes
# ---------------------------------------------------------------------------

def build_recipes_generated(raw_recipes: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build gtnh_recipes.generated.json structure from recipes_raw.json.

    Input recipe entry (example):

      {
        "id": "gt:steam_macerator_copper_ore",
        "type": "machine",
        "machine": "gregtech:steam_macerator",
        "tier": "steam",
        "inputs": [...],
        "outputs": [...],
        "byproducts": [...]
      }

    Output recipe entry:

      {
        "id": "gt:steam_macerator_copper_ore",
        "type": "machine",
        "machine": "gregtech:steam_macerator",
        "tier": "steam",
        "io": {
          "inputs": [...],
          "outputs": [...],
          "byproducts": [...]
        }
      }
    """
    recipes_out: List[Dict[str, Any]] = []
    recipes_in = raw_recipes.get("recipes", []) or []
    if not isinstance(recipes_in, list):
        raise ValueError("recipes_raw.json: 'recipes' must be a list.")

    for r in recipes_in:
        if not isinstance(r, dict):
            continue

        rid = r.get("id")
        if not isinstance(rid, str):
            continue

        r_type = r.get("type", "crafting")
        machine = r.get("machine")
        tier = r.get("tier")

        inputs = r.get("inputs", []) or []
        outputs = r.get("outputs", []) or []
        byproducts = r.get("byproducts", []) or []

        recipes_out.append(
            {
                "id": rid,
                "type": r_type,
                "machine": machine,
                "tier": tier,
                "io": {
                    "inputs": inputs,
                    "outputs": outputs,
                    "byproducts": byproducts,
                },
            }
        )

    return {"recipes": recipes_out}


# ---------------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------------

def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Ingest raw GTNH dumps into generated semantics configs."
    )
    parser.add_argument(
        "--config-dir",
        type=str,
        default="config",
        help="Path to GTNH_Agent config directory (default: config/).",
    )
    parser.add_argument(
        "--raw-dir",
        type=str,
        default=None,
        help="Path to directory containing raw modpack dumps "
             "(default: <config-dir>/raw).",
    )

    args = parser.parse_args(argv)

    config_dir = Path(args.config_dir).resolve()
    raw_dir = Path(args.raw_dir).resolve() if args.raw_dir else (config_dir / "raw")

    if not config_dir.exists():
        raise SystemExit(f"Config directory does not exist: {config_dir}")
    if not raw_dir.exists():
        raise SystemExit(f"Raw dump directory does not exist: {raw_dir}")

    # Load raw JSONs
    items_raw_path = raw_dir / "registry_items.json"
    blocks_raw_path = raw_dir / "registry_blocks.json"
    recipes_raw_path = raw_dir / "recipes_raw.json"

    raw_items = _load_json(items_raw_path)
    raw_blocks = _load_json(blocks_raw_path)
    raw_recipes = _load_json(recipes_raw_path)

    # Build generated structures
    items_generated = build_items_generated(raw_items)
    blocks_generated = build_blocks_generated(raw_blocks)
    recipes_generated = build_recipes_generated(raw_recipes)

    # Write outputs
    items_out_path = config_dir / "gtnh_items.generated.yaml"
    blocks_out_path = config_dir / "gtnh_blocks.generated.yaml"
    recipes_out_path = config_dir / "gtnh_recipes.generated.json"

    items_out_path.write_text(
        yaml.safe_dump(items_generated, sort_keys=True),
        encoding="utf-8",
    )
    blocks_out_path.write_text(
        yaml.safe_dump(blocks_generated, sort_keys=True),
        encoding="utf-8",
    )
    recipes_out_path.write_text(
        json.dumps(recipes_generated, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print(f"[ingest] Wrote {items_out_path}")
    print(f"[ingest] Wrote {blocks_out_path}")
    print(f"[ingest] Wrote {recipes_out_path}")


if __name__ == "__main__":
    main()
