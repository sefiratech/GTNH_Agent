# A* (or similar) pathfinding over NavGrid
# src/bot_core/nav/pathfinder.py
"""
A* pathfinding over NavGrid.

- Uses Manhattan distance heuristic.
- 4-directional neighbors (x-z plane) with limited vertical adjustment.
- max_steps guard to avoid infinite loops / huge searches.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .grid import NavGrid, Coord


@dataclass
class PathfindingResult:
    """Structured result for a pathfinding attempt."""

    path: List[Coord]
    success: bool
    reason: str | None = None


def _heuristic(a: Coord, b: Coord) -> float:
    """Manhattan distance heuristic for A*."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1]) + abs(a[2] - b[2])


def find_path(
    grid: NavGrid,
    start: Coord,
    goal: Coord,
    max_steps: int = 1024,
) -> PathfindingResult:
    """
    A* search for a path from start to goal on NavGrid.

    Returns a PathfindingResult with:
      - path: possibly empty list of coordinates including start and goal
      - success: bool
      - reason: if not success, a human-readable explanation

    This function does not send any packets or mutate world state.
    """
    if start == goal:
        return PathfindingResult(path=[start], success=True)

    open_heap: List[tuple[float, Coord]] = []
    heapq.heappush(open_heap, (0.0, start))

    came_from: Dict[Coord, Coord] = {}
    g_score: Dict[Coord, float] = {start: 0.0}

    steps_remaining = max_steps

    while open_heap and steps_remaining > 0:
        _, current = heapq.heappop(open_heap)
        steps_remaining -= 1

        if current == goal:
            # reconstruct path
            return PathfindingResult(
                path=_reconstruct_path(came_from, current),
                success=True,
            )

        for nxt in grid.neighbors_4dir(current):
            tentative_g = g_score[current] + 1.0

            if tentative_g < g_score.get(nxt, float("inf")):
                came_from[nxt] = current
                g_score[nxt] = tentative_g
                f_score = tentative_g + _heuristic(nxt, goal)
                heapq.heappush(open_heap, (f_score, nxt))

    # No path found or max_steps exhausted
    reason = (
        "max_steps_exhausted"
        if steps_remaining <= 0
        else "no_path_found"
    )
    return PathfindingResult(path=[], success=False, reason=reason)


def _reconstruct_path(
    came_from: Dict[Coord, Coord],
    current: Coord,
) -> List[Coord]:
    """Reconstruct full path from came_from map."""
    path: List[Coord] = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path

