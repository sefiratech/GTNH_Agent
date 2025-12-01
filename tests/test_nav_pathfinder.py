# tests/test_nav_pathfinder.py
"""
Unit tests for NavGrid + A* pathfinder.

We build synthetic "worlds" by faking is_solid_block behavior,
so no real chunk data is required.
"""

from __future__ import annotations

from typing import Any

from bot_core.snapshot import RawWorldSnapshot
from bot_core.nav import NavGrid, find_path, Coord


def make_empty_snapshot() -> RawWorldSnapshot:
    return RawWorldSnapshot(
        tick=0,
        dimension="overworld",
        player_pos={"x": 0.0, "y": 64.0, "z": 0.0},
        player_yaw=0.0,
        player_pitch=0.0,
        on_ground=True,
        chunks={},
        entities=[],
        inventory=[],
        context={},
    )


def flat_floor_is_solid(x: int, y: int, z: int, snapshot: RawWorldSnapshot) -> bool:
    """
    Simple solidity function:
    - floor at y=63 is solid
    - everything else is non-solid
    """
    return y == 63


def wall_is_solid(x: int, y: int, z: int, snapshot: RawWorldSnapshot) -> bool:
    """
    Like flat_floor_is_solid, but with a vertical wall at x == 2, z in [0..4].

    We model:
      - floor at y=63 everywhere
      - solid column at x=2, z in [0..4], for y >= 63

    That way, positions directly above the wall (y=64) are NOT walkable
    because is_walkable sees solid at (x, y, z) and/or (x, y+1, z).
    """
    # Floor everywhere
    if y == 63:
        return True

    # Wall column: block body/head space at and above the floor
    if x == 2 and 0 <= z <= 4 and y >= 64:
        return True

    return False


def test_find_path_in_open_space() -> None:
    snap = make_empty_snapshot()
    grid = NavGrid(snapshot=snap, is_solid_block=flat_floor_is_solid)

    start: Coord = (0, 64, 0)
    goal: Coord = (4, 64, 0)

    result = find_path(grid, start=start, goal=goal, max_steps=100)

    assert result.success
    assert result.path[0] == start
    assert result.path[-1] == goal
    # path length should be >= 5 (0..4 inclusive)
    assert len(result.path) >= 5


def test_find_path_around_wall() -> None:
    snap = make_empty_snapshot()
    grid = NavGrid(snapshot=snap, is_solid_block=wall_is_solid)

    start: Coord = (0, 64, 0)
    goal: Coord = (4, 64, 0)

    result = find_path(grid, start=start, goal=goal, max_steps=200)

    assert result.success
    assert result.path[0] == start
    assert result.path[-1] == goal

    # Ensure at least one step avoids x == 2, z in [0..4] at y=64
    for (x, y, z) in result.path:
        # we only care about positions directly over the floor (y=64)
        if y == 64 and x == 2 and 0 <= z <= 4:
            raise AssertionError("Path incorrectly goes through the wall")


def test_find_path_max_steps_exhaustion() -> None:
    snap = make_empty_snapshot()
    # treat EVERYTHING as solid so no path is possible
    grid = NavGrid(
        snapshot=snap,
        is_solid_block=lambda x, y, z, s: True,  # type: ignore[arg-type]
    )

    start: Coord = (0, 64, 0)
    goal: Coord = (10, 64, 10)

    result = find_path(grid, start=start, goal=goal, max_steps=10)

    assert not result.success
    assert result.reason in ("no_path_found", "max_steps_exhausted")
    assert result.path == []

