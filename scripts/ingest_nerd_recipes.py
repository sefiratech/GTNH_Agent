# scripts/ingest_nerd_recipes.py
"""
Ingest NEI "recipes.json" + "recipe_stacks.json" (Nerd/NEI dump)
and convert to the canonical GTNH_Agent recipes format used by M3.

Inputs (from config/raw/):

  - recipes.json          (required)
      Shape (simplified):
        {
          "version": "...",
          "queries": [
            {
              "queryItem": "1d0",
              "handlers": [
                {
                  "id": "codechicken.nei.recipe.FurnaceRecipeHandler",
                  "name": "Smelting",
                  "tabName": "Smelting",
                  "recipes": [
                    {
                      "generic": {
                        "ingredients": ["4d0", {...}],
                        "otherStacks": [...],
                        "outItem": "1d0" or { "itemSlug": "1d0", "count": 9 }
                      }
                    }
                  ]
                },
                ...
              ]
            },
            ...
          ]
        }

  - recipe_stacks.json    (required)
      Expected shape (what you showed earlier):
        {
          "items": {
            "6574d0": {
              "id": 6574,
              "regName": "BuildCraft|Transport:item.buildcraftPipe.pipefluidsdiamond",
              "name": "item.PipeFluidsDiamond",
              "displayName": "Diamond Fluid Pipe",
              "nbt": { ... }
            },
            ...
          },
          "fluids": {
            ...
          }
        }

Output (to config/):

  - gtnh_recipes.generated.json

Canonical format (what SemanticsDB + crafting expect):

  {
    "recipes": [
      {
        "id": "handlerId:queryItem:index",
        "type": "machine" | "crafting" | "smelting" | "unknown",
        "machine": "macerator" | "smelting" | "assembler" | ...,
        "tier": null | "steam" | "lv" | "mv" | ...,
        "io": {
          "inputs": [
            { "item": "mod:item", "variant": "something" | null, "count": 1 }
          ],
          "outputs": [
            { "item": "mod:item", "variant": "something" | null, "count": 2 }
          ],
          "byproducts": [
            { "item": "mod:item", "variant": null, "count": 1 }
          ]
        }
      }
    ]
  }

Semantics:

  - Hand-authored gtnh_recipes.json still override these via SemanticsDB.
  - This ingester is conservative:
      * ingredients → inputs
      * outItem    → primary outputs
      * otherStacks are ignored for now (often fuels / helper stacks).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "config" / "raw"
OUT_DIR = PROJECT_ROOT / "config"


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------

def _load_json_required(path: Path) -> Any:
    """Load a JSON file or die loudly."""
    if not path.exists():
        raise FileNotFoundError(f"JSON not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Stack index (recipe_stacks.json)
# ---------------------------------------------------------------------------

def _build_stack_index(stacks_raw: Any) -> Dict[str, Dict[str, Any]]:
    """
    Build a lookup table from Nerd/NEI stack slugs to stack metadata.

    Expected stacks_raw (recipe_stacks.json):

      {
        "items": {
          "6574d0": {
            "id": 6574,
            "regName": "BuildCraft|Transport:item.buildcraftPipe.pipefluidsdiamond",
            "name": "item.PipeFluidsDiamond",
            "displayName": "Diamond Fluid Pipe",
            "nbt": { ... }
          },
          ...
        },
        "fluids": { ... }
      }

    Returns:
      { "6574d0": {stack_data}, ... }
    """
    if not isinstance(stacks_raw, dict):
        raise ValueError(f"recipe_stacks.json must be an object, got {type(stacks_raw)!r}")

    items = stacks_raw.get("items")
    if not isinstance(items, dict):
        raise ValueError("recipe_stacks.json missing top-level 'items' object")

    index: Dict[str, Dict[str, Any]] = {}
    for slug, entry in items.items():
        if not isinstance(slug, str):
            continue
        if not isinstance(entry, dict):
            continue
        index[slug] = entry

    print(f"[ingest] Stack index built with {len(index)} item entries")
    return index


def _slug_to_item_variant(
    slug: str,
    stack_index: Dict[str, Dict[str, Any]],
) -> Tuple[str, Optional[str]]:
    """
    Convert a Nerd stack slug ("7437d20470") into (item_id, variant).

    Heuristics:

      - item_id:
          Prefer stack["regName"] if present (e.g. "gregtech:gt.metaitem.02").
          Fallback to stack["name"].
          Fallback to "unknown:item".

      - variant:
          For GregTech meta-items, stack["name"] tends to be like "gt.metaitem.02.20470".
          We'll just use the full name as a variant token.
          If nothing useful exists, variant = None.
    """
    stack = stack_index.get(slug)
    if not isinstance(stack, dict):
        return "unknown:item", None

    item_id = stack.get("regName") or stack.get("name") or "unknown:item"
    if not isinstance(item_id, str):
        item_id = "unknown:item"

    variant: Optional[str] = None
    name_field = stack.get("name")
    if isinstance(name_field, str):
        variant = name_field

    return item_id, variant


def _normalize_stack_ref(
    ref: Any,
    stack_index: Dict[str, Dict[str, Any]],
    default_count: int = 1,
) -> Dict[str, Any]:
    """
    Normalize a single ingredient/output reference into canonical IO entry:

      { "item": "mod:item", "variant": <str|None>, "count": int }

    Supports:

      - "7437d20470"           (plain slug string)
      - {"itemSlug": "7437d0", "count": 2, ...}

    If stack slug is unknown, we fall back to "unknown:item" but keep structure valid.
    """
    slug: Optional[str] = None
    count = int(default_count)

    # Case 1: direct string slug ("1d0", "7437d20470")
    if isinstance(ref, str):
        slug = ref

    # Case 2: dict with itemSlug
    elif isinstance(ref, dict):
        if "itemSlug" in ref:
            slug = ref.get("itemSlug")
            count = int(ref.get("count", default_count))
        else:
            # Last resort: maybe it already looks like our canonical form
            if isinstance(ref.get("item"), str):
                item_id = ref["item"]
                variant = ref.get("variant")
                count = int(ref.get("count", default_count))
                if count <= 0:
                    count = 1
                return {"item": item_id, "variant": variant, "count": count}
            # Or maybe it has some "stack" style reference; punt for now.
            slug = str(ref.get("stack") or ref.get("id") or "")

    if not slug or not isinstance(slug, str):
        if count <= 0:
            count = 1
        return {"item": "unknown:item", "variant": None, "count": count}

    item_id, variant = _slug_to_item_variant(slug, stack_index)
    if count <= 0:
        count = 1
    return {"item": item_id, "variant": variant, "count": count}


def _normalize_io_list(
    entries: Optional[List[Any]],
    stack_index: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Normalize a list of Nerd IO references into canonical IO entries."""
    if not entries:
        return []
    normalized: List[Dict[str, Any]] = []
    for e in entries:
        normalized.append(_normalize_stack_ref(e, stack_index))
    return normalized


# ---------------------------------------------------------------------------
# Recipe type & machine heuristics
# ---------------------------------------------------------------------------

def _derive_recipe_type(handler_id: str, handler_name: str, tab_name: str) -> str:
    """
    Classify recipe type based on NEI handler.

    We only care about:
      - "crafting"
      - "smelting"
      - "machine"
      - "unknown"
    """
    hid = handler_id.lower()
    hname = handler_name.lower()
    tname = tab_name.lower()

    if "crafting" in hname or "crafting" in tname:
        return "crafting"
    if "furnace" in hid or "smelt" in hname or "smelt" in tname:
        return "smelting"
    if "gregtech.nei" in hid or "gtnei" in hid or "gregtech" in hid:
        return "machine"
    if "machine" in hname or "machine" in tname:
        return "machine"

    return "unknown"


def _derive_machine_id(handler_id: str, handler_name: str, tab_name: str) -> Optional[str]:
    """
    Derive a machine identifier string from NEI handler data.

    For crafting, this may be None.
    For machines, we try to produce a stable, lowercase id like "macerator" or "assembler".
    """
    raw = handler_name or tab_name or handler_id
    if not isinstance(raw, str) or not raw.strip():
        return None

    base = raw.strip().lower().replace(" ", "_")
    base = base.replace("shaped_", "").replace("shapeless_", "")
    return base or None


# ---------------------------------------------------------------------------
# Recipes ingestion
# ---------------------------------------------------------------------------

def ingest_recipes(
    recipes_raw: Any,
    stacks_raw: Any,
) -> Dict[str, Any]:
    """
    Convert NEI recipes.json + recipe_stacks.json into canonical recipes list.
    """
    if not isinstance(recipes_raw, dict):
        raise ValueError(f"recipes.json expected dict with 'queries', got {type(recipes_raw)!r}")

    queries = recipes_raw.get("queries")
    if not isinstance(queries, list):
        raise ValueError("recipes.json missing 'queries' list at top level.")

    stack_index = _build_stack_index(stacks_raw)

    normalized_recipes: List[Dict[str, Any]] = []

    for q_idx, query in enumerate(queries):
        if not isinstance(query, dict):
            continue

        query_item_slug = query.get("queryItem")
        handlers = query.get("handlers") or []
        if not isinstance(handlers, list):
            continue

        for h_idx, handler in enumerate(handlers):
            if not isinstance(handler, dict):
                continue

            handler_id = handler.get("id", "unknown_handler")
            handler_name = handler.get("name", "") or ""
            tab_name = handler.get("tabName", "") or ""

            rtype = _derive_recipe_type(str(handler_id), handler_name, tab_name)
            machine = _derive_machine_id(str(handler_id), handler_name, tab_name)

            recipes_list = handler.get("recipes") or []
            if not isinstance(recipes_list, list):
                continue

            for r_idx, recipe_entry in enumerate(recipes_list):
                if not isinstance(recipe_entry, dict):
                    continue

                generic = recipe_entry.get("generic") or recipe_entry
                if not isinstance(generic, dict):
                    continue

                ingredients_raw = generic.get("ingredients") or []
                other_stacks_raw = generic.get("otherStacks") or []  # currently ignored
                out_raw = generic.get("outItem")

                # Fallback: if outItem is missing, use queryItem as the output slug
                if out_raw is None and isinstance(query_item_slug, str):
                    out_raw = query_item_slug

                inputs = _normalize_io_list(ingredients_raw, stack_index)

                # outItem can be a single slug, a dict, or a list; normalize to list then process
                if isinstance(out_raw, list):
                    outputs_raw_list: List[Any] = out_raw
                elif out_raw is None:
                    outputs_raw_list = []
                else:
                    outputs_raw_list = [out_raw]

                outputs = _normalize_io_list(outputs_raw_list, stack_index)

                # For now, ignore otherStacks in canonical IO; often fuels / helper stacks.
                byproducts: List[Dict[str, Any]] = []

                if not outputs:
                    continue

                rid = f"{handler_id}:{query_item_slug}:{r_idx}"

                normalized_recipes.append(
                    {
                        "id": rid,
                        "type": rtype,
                        "machine": machine,
                        "tier": None,
                        "io": {
                            "inputs": inputs,
                            "outputs": outputs,
                            "byproducts": byproducts,
                        },
                    }
                )

    # Deduplicate by id, last one wins
    by_id: Dict[str, Dict[str, Any]] = {}
    for r in normalized_recipes:
        rid = r.get("id", "unknown_recipe")
        by_id[rid] = r

    print(f"[ingest] Normalized {len(normalized_recipes)} raw recipes -> {len(by_id)} unique by id")
    return {"recipes": list(by_id.values())}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    recipes_path = RAW_DIR / "recipes.json"
    stacks_path = RAW_DIR / "recipes_stacks.json"  # corrected filename
    out_path = OUT_DIR / "gtnh_recipes.generated.json"

    print(f"[ingest] RAW_DIR={RAW_DIR}")
    print(f"[ingest] Loading recipes from {recipes_path}")
    recipes_raw = _load_json_required(recipes_path)

    print(f"[ingest] Loading stacks from {stacks_path}")
    stacks_raw = _load_json_required(stacks_path)

    data = ingest_recipes(recipes_raw, stacks_raw)
    recipes = data.get("recipes", [])
    print(f"[ingest] Normalized {len(recipes)} canonical recipes")

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"[ingest] Wrote {out_path}")


