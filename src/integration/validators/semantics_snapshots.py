# src/integration/validators/semantics_snapshots.py

from dataclasses import dataclass
from typing import Any, Callable, Dict


@dataclass
class SemanticsSnapshot:
    """
    Lightweight container for a 'semantics view' of the world.

    You can define this however you like in tests:
    - tech_state: string ID (e.g. "steam_age", "lv")
    - key_counts: number of items/blocks in certain semantic groups
    - any other scalar / small-structure information
    """
    tech_state: str
    features: Dict[str, Any]


def take_semantics_snapshot(
    *,
    world: Any,
    get_tech_state: Callable[[Any], Any],
    build_features: Callable[[Any, Any], Dict[str, Any]],
) -> SemanticsSnapshot:
    """
    Take a snapshot of semantics relevant for regression testing.

    Args:
        world:
            A world-like object / fixture.
        get_tech_state:
            Function that maps the world to a TechState-like object
            (must have .name or be convertible to string).
        build_features:
            Function that takes (world, tech_state) and returns a dict
            of scalar-ish features to compare against baselines.

    Returns:
        SemanticsSnapshot holding tech_state.name and the feature dict.
    """
    tech_state_obj = get_tech_state(world)
    tech_state_name = getattr(tech_state_obj, "name", str(tech_state_obj))

    features = build_features(world, tech_state_obj)

    return SemanticsSnapshot(
        tech_state=tech_state_name,
        features=features,
    )


def compare_semantics_snapshots(
    current: SemanticsSnapshot,
    expected: SemanticsSnapshot,
) -> Dict[str, str]:
    """
    Compare two SemanticsSnapshot instances and return a dict of differences.

    The returned dict maps a feature name to a human-readable difference description.
    Empty dict means "no differences detected" (within the feature sets compared).
    """
    diffs: Dict[str, str] = {}

    if current.tech_state != expected.tech_state:
        diffs["tech_state"] = f"{current.tech_state!r} != {expected.tech_state!r}"

    # Compare overlapping feature keys only
    keys = set(current.features.keys()) | set(expected.features.keys())
    for key in sorted(keys):
        cv = current.features.get(key)
        ev = expected.features.get(key)
        if cv != ev:
            diffs[key] = f"{cv!r} != {ev!r}"

    return diffs
