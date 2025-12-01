# convert paths into low-level movement actions
# src/bot_core/nav/mover.py
"""
Mover: convert paths into high-level move_to Actions.

M6 only owns:
- path → sequence of move_to Actions
- basic parameter shaping (radius, etc.)

It does NOT send packets or talk to the world; that's ActionExecutor's job.
"""

from __future__ import annotations

from typing import List

from spec.types import Action  # Action is defined in M1 (spec.types)
from ..snapshot import RawWorldSnapshot
from .pathfinder import PathfindingResult, Coord


def path_to_actions(
    path_result: PathfindingResult,
    snapshot: RawWorldSnapshot,
    *,
    radius: float = 0.5,
) -> List[Action]:
    """
    Convert a PathfindingResult into a list of move_to Actions.

    Each step becomes a single move_to with:
      - x, y, z: integer block coordinates
      - radius: how close is "good enough" for arrival

    Higher layers (skills, planner) can choose whether to:
      - compress steps into fewer waypoints
      - run only a prefix of the path
    """
    if not path_result.success or not path_result.path:
        return []

    actions: List[Action] = []
    for (x, y, z) in path_result.path:
        actions.append(
            Action(
                type="move_to",
                params={
                    "x": int(x),
                    "y": int(y),
                    "z": int(z),
                    "radius": float(radius),
                },
            )
        )

    return actions


def current_coord_from_snapshot(snapshot: RawWorldSnapshot) -> Coord:
    """
    Helper: derive integer Coord from the current player position.

    Floor the floating position to an int grid; this keeps NavGrid
    and real position in sync enough for path planning.
    """
    px = snapshot.player_pos.get("x", 0.0)
    py = snapshot.player_pos.get("y", 0.0)
    pz = snapshot.player_pos.get("z", 0.0)

    return int(px), int(py), int(pz)

