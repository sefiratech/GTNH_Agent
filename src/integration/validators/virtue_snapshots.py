# src/integration/validators/virtue_snapshots.py

from dataclasses import dataclass
from typing import Any, Callable, Dict, List


@dataclass
class VirtueScoreSnapshot:
    """
    Snapshot of virtue scores for a given plan in a given context.

    - context_id: e.g. 'lv_bootstrap'
    - plan_id: arbitrary identifier for the plan being evaluated
    - scores: raw scores from the virtue lattice (any JSON-serializable dict)
    """
    context_id: str
    plan_id: str
    scores: Dict[str, Any]


def take_virtue_snapshot(
    *,
    plan_summary: Any,
    context_id: str,
    virtue_config: Any,
    score_fn: Callable[[Any, str, Any], Dict[str, Any]],
) -> VirtueScoreSnapshot:
    """
    Call a scoring function to produce a VirtueScoreSnapshot.

    Args:
        plan_summary:
            Whatever object your virtue lattice uses to represent a plan.
        context_id:
            Virtue context identifier (e.g. 'lv_bootstrap').
        virtue_config:
            Config object / data structure for the lattice.
        score_fn:
            Callable of the form score_fn(plan_summary, context_id, virtue_config)
            that returns a dict of scores.

    Returns:
        VirtueScoreSnapshot bundling the context, a plan_id, and the scores.
    """
    scores = score_fn(plan_summary, context_id, virtue_config)
    plan_id = getattr(plan_summary, "id", "<unknown>")

    return VirtueScoreSnapshot(
        context_id=context_id,
        plan_id=plan_id,
        scores=scores,
    )


def compare_virtue_snapshots(
    baseline: VirtueScoreSnapshot,
    current: VirtueScoreSnapshot,
    *,
    tolerances: Dict[str, float] | None = None,
) -> Dict[str, str]:
    """
    Compare two VirtueScoreSnapshot instances and return differences.

    For numeric scores, you can optionally specify absolute tolerances per-key
    in the 'tolerances' dict. Non-numeric values are compared for exact equality.

    Returns:
        Dict[key, description] of differences. Empty dict means no detected change
        beyond tolerances.
    """
    diffs: Dict[str, str] = {}

    if baseline.context_id != current.context_id:
        diffs["context_id"] = f"{baseline.context_id!r} != {current.context_id!r}"

    if baseline.plan_id != current.plan_id:
        diffs["plan_id"] = f"{baseline.plan_id!r} != {current.plan_id!r}"

    base_scores = baseline.scores
    curr_scores = current.scores
    keys = set(base_scores.keys()) | set(curr_scores.keys())

    tol = tolerances or {}

    for key in sorted(keys):
        bv = base_scores.get(key)
        cv = curr_scores.get(key)

        if isinstance(bv, (int, float)) and isinstance(cv, (int, float)):
            limit = tol.get(key, 0.0)
            if abs(bv - cv) > limit:
                diffs[key] = f"{cv} (current) vs {bv} (baseline) exceeds tol={limit}"
        else:
            if bv != cv:
                diffs[key] = f"{cv!r} != {bv!r}"

    return diffs
