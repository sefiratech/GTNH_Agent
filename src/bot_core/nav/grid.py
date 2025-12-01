# navigation grid abstraction based on world blocks
# src/bot_core/nav/grid.py
"""
NavGrid: navigation grid abstraction over RawWorldSnapshot.

This module does not know GTNH semantics. It only:
- Exposes walkability queries.
- Uses a pluggable is_solid callback to decide collisions.

Actual "what blocks are solid" logic belongs to M3 (semantics).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Tuple

from ..snapshot import RawWorldSnapshot, RawChunk

# (x, y, z) integer coordinates
Coord = Tuple[int, int, int]

# Signature for a block-solid callback:
#   is_solid(x, y, z, snapshot) -> bool
BlockSolidFn = Callable[[int, int, int, RawWorldSnapshot], bool]


@dataclass
class NavGrid:
    """
    Navigation grid built on top of a RawWorldSnapshot.

    Responsibilities:
    - Provide walkability tests (is_walkable).
    - Provide neighbor coordinates for pathfinding.

    It does NOT:
    - Interpret block types.
    - Perform semantics or tech reasoning.
    """

    snapshot: RawWorldSnapshot
    is_solid_block: BlockSolidFn

    # Optional constraints; these can be tuned later.
    max_fall_height: int = 4  # how far the bot is allowed to drop
    max_step_height: int = 1  # how high the bot can step up

    # ------------------------------------------------------------------
    # Core queries
    # ------------------------------------------------------------------

    def is_walkable(self, x: int, y: int, z: int) -> bool:
        """
        Determine if the bot can "stand" at (x, y, z).

        Simple rule:
        - Block at (x, y - 1, z) is solid (floor).
        - Blocks at (x, y, z) and (x, y + 1, z) are non-solid (no collision).
        """
        floor_y = y - 1

        if not self._is_coord_supported(x, floor_y, z):
            return False

        # Space must be free for body + head.
        if self._is_block_solid(x, y, z):
            return False
        if self._is_block_solid(x, y + 1, z):
            return False

        return True

    def neighbors_4dir(self, coord: Coord) -> list[Coord]:
        """
        Return potential 4-directional neighbors on the x-z plane,
        with basic vertical adjustment.

        We allow up/down steps within max_step_height and controlled falls.
        """
        x, y, z = coord
        candidates: list[Coord] = []

        # Offsets in x-z plane
        for dx, dz in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx = x + dx
            nz = z + dz

            # Try to find a suitable y at (nx, nz) near current y
            # First: small upward/downward steps
            for dy in range(-self.max_step_height, self.max_step_height + 1):
                ny = y + dy
                if self.is_walkable(nx, ny, nz):
                    candidates.append((nx, ny, nz))
                    break
            else:
                # If no small step works, see if we can safely fall
                fall_target = self._find_fall_target(nx, y, nz)
                if fall_target is not None:
                    candidates.append(fall_target)

        return candidates

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _is_block_solid(self, x: int, y: int, z: int) -> bool:
        """
        Delegate to is_solid_block callback.

        This is the only place RawWorldSnapshot → collision decision happens.
        """
        return bool(self.is_solid_block(x, y, z, self.snapshot))

    def _is_coord_supported(self, x: int, y: int, z: int) -> bool:
        """
        Check whether a block at (x, y, z) exists and is solid enough
        to act as a floor.
        """
        return self._is_block_solid(x, y, z)

    def _find_fall_target(self, x: int, start_y: int, z: int) -> Coord | None:
        """
        Try to find a valid landing spot when walking off an edge.

        We search downward from start_y to start_y - max_fall_height.
        """
        lowest_y = start_y - self.max_fall_height
        y = start_y

        # Already in air? Step down until floor or limit.
        while y >= lowest_y:
            # We want (x, y, z) to be walkable; if we find one, we accept it.
            if self.is_walkable(x, y, z):
                return (x, y, z)
            y -= 1

        return None

