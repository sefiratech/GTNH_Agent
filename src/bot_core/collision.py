# src/bot_core/collision.py
"""
Block collision helpers for bot_core_1_7_10.

This is a minimal interim layer that defines how the navigation system
decides whether a block is "solid" for movement purposes.

Long-term:
    - A richer implementation in M3 should provide block IDs / materials
      from RawWorldSnapshot.chunks and drive a more accurate collision
      model.

Short-term (minimal hack):
    - Treat "air-like" blocks as non-solid.
    - When we don't know the actual blocks, optionally fall back to a
      simple floor model (e.g., fixed Y-level).

This module is intentionally lightweight and does NOT:
    - Interpret GTNH tech semantics
    - Reason about hazards (lava, fire, etc.)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

from .snapshot import RawWorldSnapshot

# Signature for a "block lookup" function, if/when you implement one:
#   block_at(snapshot, x, y, z) -> Any
BlockAtFn = Callable[[RawWorldSnapshot, int, int, int], Any]


def _is_air_like(block: Any) -> bool:
    """
    Heuristic to decide if a block is "air-like".

    This is deliberately forgiving, since we don't yet know the real
    chunk/ID layout. It handles a few common patterns:

        - None or {} or []  → air
        - numeric 0         → air (old-school ID)
        - mapping with id in {"minecraft:air", "air"} → air

    Anything else is treated as non-air (i.e., potentially solid).
    """
    if block is None:
        return True

    # Empty containers are treated as "no block"
    if block == {} or block == []:
        return True

    # Classic numeric ID for air
    if isinstance(block, (int, float)) and block == 0:
        return True

    # Dict-like with an ID field
    if isinstance(block, dict):
        bid = block.get("id") or block.get("name")
        if isinstance(bid, str) and bid.lower() in ("minecraft:air", "air"):
            return True

    return False


@dataclass
class BlockCollisionProfile:
    """
    Encapsulates a collision policy for navigation.

    Parameters:
        block_at:
            Optional function for retrieving block data from a snapshot.

        default_floor_y:
            Optional Y-level that is treated as "solid floor everywhere"
            when block_at is not available. This is a crude fallback used
            for initial smoke tests and simple worlds.

    The main entrypoint is `is_solid_block(x, y, z, snapshot)`.
    """

    block_at: Optional[BlockAtFn] = None
    default_floor_y: Optional[int] = None

    def is_solid_block(
        self,
        x: int,
        y: int,
        z: int,
        snapshot: RawWorldSnapshot,
    ) -> bool:
        """
        Decide if the block at (x, y, z) should be treated as solid.

        Behavior:
            - If block_at is provided:
                * call it and treat any non-air-like value as solid.
            - Else if default_floor_y is set:
                * y <= default_floor_y → solid
                * y >  default_floor_y → non-solid
            - Else:
                * everything is non-solid (nav will generally fail).
        """
        if self.block_at is not None:
            block = self.block_at(snapshot, x, y, z)
            return not _is_air_like(block)

        if self.default_floor_y is not None:
            # Very crude: treat everything at or below this Y as solid.
            return y <= self.default_floor_y

        # No information: treat everything as non-solid.
        return False


# ---------------------------------------------------------------------------
# Default profile & convenience entry point
# ---------------------------------------------------------------------------

# For now, assume a flat solid floor at y=63 (common "ground" level in tests).
_DEFAULT_PROFILE = BlockCollisionProfile(
    block_at=None,
    default_floor_y=63,
)


def default_block_collision_profile() -> BlockCollisionProfile:
    """
    Return the default BlockCollisionProfile used by BotCoreImpl.

    This can later be replaced or decorated by M3 when a richer
    block-at lookup is available.
    """
    return _DEFAULT_PROFILE


def default_is_solid_block(
    x: int,
    y: int,
    z: int,
    snapshot: RawWorldSnapshot,
) -> bool:
    """
    Convenience function for use as a BlockSolidFn in nav/grid.

    Uses the module-level default profile.
    """
    return _DEFAULT_PROFILE.is_solid_block(x, y, z, snapshot)

