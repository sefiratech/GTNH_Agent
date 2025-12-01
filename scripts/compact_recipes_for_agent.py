# scripts/compact_recipes_for_agent.py
"""
Compact gtnh_recipes.generated.json into a smaller, agent-friendly subset.

Input:
  config/gtnh_recipes.generated.json

Output:
  config/gtnh_recipes.agent.json

Filtering rules (v1):
  - Drop recipes with type == "unknown"
  - Keep:
      * type == "crafting"
      * type == "smelting"
      * type == "machine" and machine is not None
  - Deduplicate by id (last wins)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "config"

GENERATED_PATH = CONFIG_DIR / "gtnh_recipes.generated.json"
AGENT_PATH = CONFIG_DIR / "gtnh_recipes.agent.json"


def _load_generated() -> Dict[str, Any]:
    if not GENERATED_PATH.exists():
        raise FileNotFoundError(f"Missing generated recipes: {GENERATED_PATH}")
    with GENERATED_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{GENERATED_PATH} must be an object with 'recipes'.")
    return data


def _should_keep(recipe: Dict[str, Any]) -> bool:
    """
    Basic filtering policy for v1.

    Keep:
      - type == "crafting"
      - type == "smelting"
      - type == "machine" and machine is not None

    Drop:
      - type == "unknown"
      - any recipe with obviously broken structure
    """
    rtype = recipe.get("type")
    machine = recipe.get("machine")

    if rtype not in ("crafting", "smelting", "machine"):
        return False

    if rtype == "machine" and not machine:
        return False

    io = recipe.get("io") or {}
    outputs = io.get("outputs") or []
    if not outputs:
        return False

    return True


def compact_recipes(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply filtering and deduplication to the generated recipes list.
    """
    raw_list = data.get("recipes", [])
    if not isinstance(raw_list, list):
        raise ValueError("gtnh_recipes.generated.json missing 'recipes' list.")

    kept: List[Dict[str, Any]] = []
    for r in raw_list:
        if not isinstance(r, dict):
            continue
        if _should_keep(r):
            kept.append(r)

    # Deduplicate by id (last wins)
    by_id: Dict[str, Dict[str, Any]] = {}
    for r in kept:
        rid = r.get("id")
        if isinstance(rid, str):
            by_id[rid] = r

    print(
        f"[compact] Raw recipes: {len(raw_list)} "
        f"→ kept: {len(kept)} → unique by id: {len(by_id)}"
    )
    return {"recipes": list(by_id.values())}


def main() -> None:
    print(f"[compact] Loading generated recipes from {GENERATED_PATH}")
    data = _load_generated()

    compacted = compact_recipes(data)
    recipes = compacted.get("recipes", [])
    print(f"[compact] Writing {len(recipes)} compact recipes to {AGENT_PATH}")

    with AGENT_PATH.open("w", encoding="utf-8") as f:
        json.dump(compacted, f, indent=2, ensure_ascii=False)

    print("[compact] Done.")


if __name__ == "__main__":
    main()

