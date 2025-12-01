# tests/test_observation_worldstate_normalization.py
"""
Contract tests for observation.encoder.build_world_state.

Goal:
  - If BotCore / raw snapshot shape drifts, this test should fail loudly.
  - It enforces that build_world_state:
      * normalizes inventory into {item_id, variant, count}
      * normalizes context["machines"] into {id, type, tier, extra}
      * applies sane defaults for missing top-level fields
"""

from observation.encoder import build_world_state


def test_build_world_state_normalizes_inventory_and_machines():
    raw_snapshot = {
        # deliberately omit tick/position/dimension to test defaults
        "inventory": [
            {"item": "minecraft:iron_ingot", "count": 3},
            {"item_id": "gregtech:gt.metaitem.01", "variant": "plate.copper", "count": 16},
        ],
        "context": {
            "machines": [
                {"machine_id": "gregtech:steam_macerator"},
                {"type": "gregtech:lv_macerator", "tier": "lv"},
            ]
        },
    }

    ws = build_world_state(raw_snapshot)

    # Defaults for missing scalar / small-map fields
    assert ws.tick == 0
    assert ws.position == {"x": 0.0, "y": 0.0, "z": 0.0}
    assert ws.dimension == "overworld"

    # Inventory must be fully normalized and in the same order
    assert ws.inventory == [
        {"item_id": "minecraft:iron_ingot", "variant": None, "count": 3},
        {"item_id": "gregtech:gt.metaitem.01", "variant": "plate.copper", "count": 16},
    ]

    # Machines must be normalized into MachineEntry shape, preserving order
    machines = ws.context["machines"]
    assert len(machines) == 2

    # First came from "machine_id"
    assert machines[0]["id"] == "gregtech:steam_macerator"
    # type falls back to id when not explicitly provided
    assert machines[0]["type"] == "gregtech:steam_macerator"

    # Second came from "type"
    assert machines[1]["id"] == "gregtech:lv_macerator"
    assert machines[1]["type"] == "gregtech:lv_macerator"
    assert machines[1]["tier"] == "lv"

