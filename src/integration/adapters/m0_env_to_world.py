# src/integration/adapters/m0_env_to_world.py

"""
Phase 0 → Phase 1 adapter.

Takes a Phase 0-like "environment profile" object and turns it into a
Phase 1 WorldState suitable for offline planning.

Deliberately duck-typed:

- We do NOT import EnvProfile directly here.
- Anything with .name and a few optional attributes is acceptable.

This keeps Phase 1 loosely coupled to Phase 0 and avoids hard import cycles.
"""

from typing import Any, Dict, Protocol, runtime_checkable

from spec.types import WorldState


@runtime_checkable
class EnvProfileLike(Protocol):
    """
    Minimal interface we expect from a Phase 0 environment profile.

    In Phase 0 this is typically an EnvProfile dataclass loaded from env.yaml,
    but for Phase 1 we only care about a small subset of fields.
    """

    name: str  # required


def _get_attr(obj: Any, attr: str, default: Any = None) -> Any:
    """Safe getattr wrapper to avoid AttributeError spam."""
    return getattr(obj, attr, default)


def world_from_env_profile(profile: EnvProfileLike) -> WorldState:
    """
    Build a *planning* WorldState from a Phase 0-like environment profile.

    This is intentionally lossy / approximate in Phase 1:

    - It does NOT attempt to describe actual chunk contents.
    - It encodes only what the planner / semantics care about at this phase:
      - selected env name
      - modpack / GTNH version
      - optional world_seed
      - optional metadata path / profile source

    We construct a minimal-but-valid WorldState using the fields required
    by the WorldState dataclass:
        position, dimension, inventory, nearby_entities,
        blocks_of_interest, tech_state, context, tick
    """
    # Context payload: what Phase 1 planning actually cares about from P0
    context: Dict[str, Any] = {
        "env_name": profile.name,
        "modpack": _get_attr(profile, "modpack_id"),
        "gtnh_version": _get_attr(profile, "gtnh_version"),
        "world_seed": _get_attr(profile, "world_seed"),
        "meta": {
            "profile_path": str(_get_attr(profile, "source_path", "")),
            "phase": "p1_offline",
        },
    }

    # Minimal world-scene stub for offline planning.
    # Fields must match the WorldState dataclass signature.
    return WorldState(
        position=_get_attr(profile, "position", None),           # planner doesn't care yet
        dimension=_get_attr(profile, "dimension", "overworld"),
        inventory=_get_attr(profile, "inventory", {}),           # empty by default
        nearby_entities=_get_attr(profile, "nearby_entities", []),
        blocks_of_interest=_get_attr(profile, "blocks_of_interest", []),
        tech_state=_get_attr(profile, "tech_state", None),
        context=context,
        tick=_get_attr(profile, "tick", 0),                      # default to tick 0
    )

