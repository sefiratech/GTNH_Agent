# craftable_items and crafting helpers
# src/semantics/crafting.py

from collections import Counter
from typing import Dict, Any, List, Optional, Tuple, Set

from spec.types import WorldState                  # world representation from M1
from .schema import CraftOption, CraftResult       # craftability result types
from .loader import SemanticsDB                    # DB with recipes/index


# ---------------------------------------------------------------------------
# Inventory / machine helpers
# ---------------------------------------------------------------------------

def inventory_to_counts(inventory: List[Dict[str, Any]]) -> Counter:
    """
    Convert inventory list into a Counter keyed by (item_id, variant).

    Each entry in inventory is *expected* (but not strictly required) to have:
      {
        "item_id": "gregtech:gt.metaitem.01",
        "variant": "plate.copper",
        "count": 32
      }

    We try a few common keys for the item id:
      - "item_id"
      - "item"
      - "id"
    """
    counts: Counter = Counter()
    for stack in inventory:
        if not isinstance(stack, dict):
            continue

        item_id = (
            stack.get("item_id")
            or stack.get("item")
            or stack.get("id")
        )
        if not isinstance(item_id, str):
            continue

        variant = stack.get("variant")
        count = int(stack.get("count", 0))
        if count <= 0:
            continue

        key = (item_id, variant)
        counts[key] += count

    return counts


def _extract_machine_ids_from_context(context: Dict[str, Any]) -> Set[str]:
    """
    Extract machine ids from world.context["machines"].

    Expected patterns per machine entry:
      - { "id": "gregtech:lv_macerator", ... }
      - { "machine_id": "gregtech:steam_macerator", ... }
      - { "type": "gregtech:steam_macerator", ... }  # legacy / loose
    """
    machines_list = context.get("machines", [])
    machine_ids: Set[str] = set()

    if isinstance(machines_list, list):
        for m in machines_list:
            if not isinstance(m, dict):
                continue
            mid = (
                m.get("machine_id")
                or m.get("id")
                or m.get("type")
            )
            if isinstance(mid, str):
                machine_ids.add(mid)

    return machine_ids


# ---------------------------------------------------------------------------
# Craftability
# ---------------------------------------------------------------------------

def _io_entry_to_key(entry: Dict[str, Any]) -> Tuple[str, Optional[str]]:
    """
    Convert a recipe IO entry into (item_id, variant) key.

    Expected recipe IO entry:
      {
        "item": "gregtech:gt.metaitem.01",
        "variant": "dust.copper",
        "count": 2
      }
    """
    item_id = entry.get("item")
    if not isinstance(item_id, str):
        raise ValueError(f"Recipe IO entry missing valid 'item': {entry!r}")
    variant = entry.get("variant")
    return (item_id, variant)


def _build_craft_result(entry: Dict[str, Any]) -> CraftResult:
    """Build a CraftResult from a recipe IO entry."""
    item_id, variant = _io_entry_to_key(entry)
    count = int(entry.get("count", 1))
    if count <= 0:
        count = 1
    return CraftResult(item_id=item_id, variant=variant, count=count)


def craftable_items(world: WorldState, db: SemanticsDB) -> List[CraftOption]:
    """
    Determine which recipes are craftable given current inventory and machines.

    Primary expectation:
      - world.inventory is a list of normalized InventoryStack dicts
        with keys: "item_id", "variant", "count"
      - world.context["machines"] is a list of normalized MachineEntry dicts
        with keys: "id", "type", "tier", "extra"

    The function is tolerant of some legacy shapes (e.g. "item", "id"),
    but callers SHOULD go through observation.encoder.build_world_state.

    This is a coarse, static analysis:
      - It does *not* simulate time, positions, or power networks.
      - It only checks material availability + machine presence.

    WorldState expectations (loosely enforced):
      - world.inventory: list of item stacks (see inventory_to_counts)
      - world.context["machines"]: list of machine descriptors
    """
    inventory = getattr(world, "inventory", []) or []
    context = getattr(world, "context", {}) or {}

    inv_counts = inventory_to_counts(inventory)
    machine_ids = _extract_machine_ids_from_context(context)

    options: List[CraftOption] = []

    for recipe in db.recipes:
        recipe_id = recipe.get("id", "unknown")
        recipe_type = recipe.get("type")
        machine_id = recipe.get("machine")
        tier = recipe.get("tier")

        io_block: Dict[str, Any] = recipe.get("io", {}) or {}
        inputs: List[Dict[str, Any]] = io_block.get("inputs", []) or []
        outputs: List[Dict[str, Any]] = io_block.get("outputs", []) or []
        byproducts: List[Dict[str, Any]] = io_block.get("byproducts", []) or []

        # Skip malformed recipes
        if not outputs:
            continue

        # Check machine requirement (for machine-type recipes)
        if recipe_type == "machine" and machine_id:
            if machine_id not in machine_ids:
                # Required machine not present in world
                continue

        # Check all inputs are available in required quantities
        limiting_resource: Optional[str] = None
        limiting_ratio: float = float("inf")  # lower = more limiting

        missing = False
        for inp in inputs:
            key = _io_entry_to_key(inp)
            required = int(inp.get("count", 1))
            if required <= 0:
                required = 1

            available = inv_counts.get(key, 0)
            if available < required:
                missing = True
                item_id, variant = key
                v_str = f"{item_id}[{variant}]" if variant else item_id
                deficit = required - available
                limiting_resource = f"Missing {deficit}x {v_str}"
                break

            # Compute how many times this input allows the recipe to run
            ratio = available / required
            if ratio < limiting_ratio:
                limiting_ratio = ratio
                item_id, variant = key
                v_str = f"{item_id}[{variant}]" if variant else item_id
                limiting_resource = f"Limited by {v_str} (x{available})"

        if missing:
            # Not enough materials, skip this recipe
            continue

        # Build primary + secondary outputs
        primary_output_entry = outputs[0]
        primary_output = _build_craft_result(primary_output_entry)

        secondary_outputs: List[CraftResult] = []
        for extra in outputs[1:]:
            secondary_outputs.append(_build_craft_result(extra))
        for bp in byproducts:
            # Byproducts may be chance-based; we ignore chance for now and
            # treat them like secondary outputs for planning purposes.
            secondary_outputs.append(_build_craft_result(bp))

        # Notes for debugging / planner traces
        notes_parts: List[str] = []
        if recipe_type:
            notes_parts.append(f"type={recipe_type}")
        if machine_id:
            notes_parts.append(f"machine={machine_id}")
        if tier:
            notes_parts.append(f"tier={tier}")
        notes = "; ".join(notes_parts) if notes_parts else ""

        options.append(
            CraftOption(
                recipe_id=recipe_id,
                primary_output=primary_output,
                secondary_outputs=secondary_outputs,
                machine_id=machine_id,
                tier=tier,
                limiting_resource=limiting_resource,
                notes=notes,
            )
        )

    return options

