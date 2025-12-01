# src/bot_core/nav/__init__.py
"""
Navigation subsystem for bot_core_1_7_10.

Provides:
- NavGrid: walkability queries over a RawWorldSnapshot
- A* pathfinding: find_path
- Mover: path_to_actions to convert paths into move_to Actions
- current_coord_from_snapshot: helper to get (x, y, z) from snapshot
"""

from __future__ import annotations

from .grid import NavGrid, BlockSolidFn
from .pathfinder import Coord, find_path
from .mover import path_to_actions, current_coord_from_snapshot

__all__ = [
    "NavGrid",
    "BlockSolidFn",
    "Coord",
    "find_path",
    "path_to_actions",
    "current_coord_from_snapshot",
]

