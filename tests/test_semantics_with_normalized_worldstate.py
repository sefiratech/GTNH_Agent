# tests/test_semantics_with_normalized_worldstate.py

from semantics.crafting import craftable_items
from semantics.loader import SemanticsDB
from semantics.tech_state import infer_tech_state_from_world, TechGraph
from semantics.schema import TechState
from observation.encoder import build_world_state


def test_semantics_with_normalized_worldstate():
    # Fake raw snapshot coming from BotCore (intentionally a bit messy)
    raw_snapshot = {
        "tick": 123,
        "position": {"x": 0, "y": 64, "z": 0},
        "dimension": "overworld",
        "inventory": [
            # Uses "item" instead of "item_id" → should be normalized by encoder
            {"item": "gregtech:gt.metaitem.01", "variant": "dust.copper", "count": 4},
        ],
        "nearby_entities": [],
        "blocks_of_interest": [],
        "tech_state": {},
        "context": {
            "machines": [
                # Uses "machine_id" → should become {"id": ..., "type": ..., "tier": ...}
                {"machine_id": "gregtech:steam_macerator", "tier": "steam"},
            ]
        },
    }

    # M7: raw snapshot → normalized WorldState
    world = build_world_state(raw_snapshot)

    # Sanity checks on normalization
    assert world.inventory, "Inventory should not be empty after normalization."
    inv0 = world.inventory[0]
    assert inv0["item_id"] == "gregtech:gt.metaitem.01"
    assert inv0["variant"] == "dust.copper"
    assert inv0["count"] == 4

    machines = world.context.get("machines", [])
    assert machines, "Machines list should not be empty after normalization."
    m0 = machines[0]
    # At least id should be present; type/tier are best-effort
    assert m0["id"] == "gregtech:steam_macerator"
    assert m0["type"] is not None

    # M3: semantics should happily consume the normalized WorldState
    db = SemanticsDB()
    graph = TechGraph()

    craft_options = craftable_items(world, db)
    tech_state = infer_tech_state_from_world(world, graph)

    # Should return coherent types (content depends on config, so we don't over-assert)
    assert isinstance(craft_options, list)
    assert isinstance(tech_state, TechState)
    assert isinstance(tech_state.active, str)
    assert isinstance(tech_state.unlocked, list)

