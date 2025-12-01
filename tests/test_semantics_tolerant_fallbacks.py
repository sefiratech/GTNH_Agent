# tests/test_semantics_tolerant_fallbacks.py
"""
Tests that M3 semantics gracefully tolerate sloppy / legacy upstream input
without relying on it as the primary contract.

These tests DO NOT use observation.encoder — they simulate badly-shaped
BotCore snapshots or legacy item stacks to ensure M3 does not hard-crash.
"""

from semantics.crafting import inventory_to_counts
from semantics.tech_state import _extract_machine_ids_from_context


def test_inventory_to_counts_tolerates_legacy_keys():
    """
    inventory_to_counts should accept stacks using "item" or "id"
    instead of the normalized "item_id" key.
    """
    inventory = [
        {"item": "minecraft:iron_ingot", "count": 3},
        {"id": "gregtech:gt.metaitem.01", "variant": "plate.copper", "count": 16},
    ]

    counts = inventory_to_counts(inventory)

    assert counts[("minecraft:iron_ingot", None)] == 3
    assert counts[("gregtech:gt.metaitem.01", "plate.copper")] == 16


def test_extract_machine_ids_tolerates_mixed_fields():
    """
    _extract_machine_ids_from_context should accept any of:
      - "machine_id"
      - "id"
      - "type"
    and produce a coherent set of machine identifiers.
    """
    context = {
        "machines": [
            {"machine_id": "gregtech:steam_macerator"},
            {"id": "gregtech:lv_macerator"},
            {"type": "gregtech:mv_macerator"},
        ]
    }

    ids = _extract_machine_ids_from_context(context)

    assert "gregtech:steam_macerator" in ids
    assert "gregtech:lv_macerator" in ids
    assert "gregtech:mv_macerator" in ids

