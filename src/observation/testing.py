# src/observation/testing.py
"""
Testing helpers for M7 - observation_encoding.

Provides:
  - minimal RawWorldSnapshot factories
  - heavy RawWorldSnapshot factories (for perf / scale tests)
  - dummy SemanticsDB implementation
"""

from __future__ import annotations

from typing import Any, Dict, List

from bot_core.snapshot import RawWorldSnapshot
from semantics.loader import SemanticsDB
from semantics.schema import TechState


class DummySemanticsDB(SemanticsDB):
    """
    Minimal SemanticsDB stub for tests.

    Only implements get_item_info in a way that matches M7 encoder expectations:
      - returns an object with .category and .material attributes.
    """

    def get_item_info(self, item_id: str, variant: Any = None) -> Any:
        # Cheap heuristic: treat logs/wood as "raw_material"/"wood", everything else generic.
        category = "raw_material" if "log" in item_id or "wood" in item_id else "misc"
        material = "wood" if "log" in item_id or "wood" in item_id else "unknown"
        return type("Info", (), {"category": category, "material": material})()


def make_minimal_tech_state() -> TechState:
    """
    Tiny TechState instance suitable for basic tests.
    """
    return TechState(
        unlocked=["stone_age"],
        active="stone_age",
        evidence={},
    )


def make_minimal_snapshot() -> RawWorldSnapshot:
    """
    Minimal RawWorldSnapshot used in planner encoding tests.

    Keeps everything tiny but structurally valid.
    """
    return RawWorldSnapshot(
        tick=123,
        dimension="overworld",
        player_pos={"x": 0.0, "y": 64.0, "z": 0.0},
        player_yaw=0.0,
        player_pitch=0.0,
        on_ground=True,
        chunks={},
        # Use plain dicts for entities; encoder already supports this.
        entities=[
            {
                "entity_id": 1,
                "type": "item",
                "x": 2.0,
                "y": 64.0,
                "z": 2.0,
                "data": {"hostile": False},
            }
        ],
        inventory=[
            {"item_id": "minecraft:log", "variant": None, "count": 16},
        ],
        context={"machines": []},
    )


def make_heavy_snapshot(
    entity_count: int = 120,
    inventory_count: int = 130,
) -> RawWorldSnapshot:
    """
    Heavier RawWorldSnapshot for perf / scale tests.

    Builds:
      - ~entity_count entities in a grid around player
      - ~inventory_count inventory stacks with simple variation
    """
    entities: List[Dict[str, Any]] = []
    for i in range(entity_count):
        # Spread entities in a grid-ish pattern around the origin.
        x = float(i % 32)
        z = float((i // 32) % 32)
        y = 64.0
        entities.append(
            {
                "entity_id": i + 1,
                "type": "item" if i % 3 == 0 else "mob",
                "x": x,
                "y": y,
                "z": z,
                "data": {"hostile": i % 5 == 0},
            }
        )

    inventory: List[Dict[str, Any]] = []
    for i in range(inventory_count):
        item_id = "minecraft:log" if i % 2 == 0 else "minecraft:cobblestone"
        inventory.append(
            {
                "item_id": item_id,
                "variant": None,
                "count": 32,
            }
        )

    machines: List[Dict[str, Any]] = []
    for i in range(8):
        machines.append(
            {
                "type": "gregtech:steam_bronze_boiler" if i < 4 else "gregtech:lv_macerator",
                "tier": "steam" if i < 4 else "lv",
            }
        )

    return RawWorldSnapshot(
        tick=9999,
        dimension="overworld",
        player_pos={"x": 0.0, "y": 64.0, "z": 0.0},
        player_yaw=90.0,
        player_pitch=0.0,
        on_ground=True,
        chunks={},
        entities=entities,
        inventory=inventory,
        context={"machines": machines},
    )


__all__ = [
    "DummySemanticsDB",
    "make_minimal_tech_state",
    "make_minimal_snapshot",
    "make_heavy_snapshot",
]

