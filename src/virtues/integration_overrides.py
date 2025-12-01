# path: src/virtues/integration_overrides.py

from __future__ import annotations

from typing import Dict


def merge_virtue_weights(
    base_weights: Dict[str, float],
    overrides: Dict[str, float],
) -> Dict[str, float]:
    """
    Merge base virtue weights with per-phase overrides.

    Overrides are interpreted as *multipliers*:

        merged[v] = base[v] * overrides.get(v, 1.0)

    Any virtue not present in overrides keeps its base weight.

    Parameters
    ----------
    base_weights:
        The default virtue weights loaded from virtues.yaml (M4).
    overrides:
        Per-phase override factors from M11 (e.g. 1.2 for Efficiency).

    Returns
    -------
    Dict[str, float]
        New dict of merged weights.
    """
    merged: Dict[str, float] = {}

    for name, base in base_weights.items():
        factor = overrides.get(name, 1.0)
        try:
            factor_f = float(factor)
        except (TypeError, ValueError):
            factor_f = 1.0
        merged[name] = float(base) * factor_f

    # Include any virtues that only appear in overrides (if you want that behavior)
    for name, factor in overrides.items():
        if name not in merged:
            try:
                merged[name] = float(factor)
            except (TypeError, ValueError):
                continue

    return merged

