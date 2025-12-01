# src/integration/validators/planner_guardrails.py

from dataclasses import dataclass
from typing import Any, Callable, Dict, List


@dataclass
class PlannerRunResult:
    """Simple container for a single planner run."""
    plans: List[Dict[str, Any]]


def run_planner_deterministic(
    *,
    planner_call: Callable[[Dict[str, Any]], Dict[str, Any]],
    payload: Dict[str, Any],
) -> PlannerRunResult:
    """
    Run the planner backend in a deterministic configuration.

    The caller is responsible for:
    - setting temperature/top_p/etc. in the payload,
    - ensuring the backend is configured to honor those settings.

    This wrapper only normalizes the output format for testing.
    """
    output = planner_call(payload)
    plans = output.get("plans", [])
    if not isinstance(plans, list):
        raise TypeError(f"planner returned non-list 'plans': {type(plans)!r}")
    return PlannerRunResult(plans=plans)


def assert_planner_stable_across_runs(
    *,
    planner_call: Callable[[Dict[str, Any]], Dict[str, Any]],
    base_payload: Dict[str, Any],
    runs: int = 3,
) -> None:
    """
    Assert that the planner is reasonably stable given identical inputs.

    Compares:
    - plan count across runs
    - basic shape of the first plan's 'steps' list

    This is intentionally conservative: it doesn't assert full equality of
    every field, but it will catch obvious instability, such as:
    - run 1 returns 1 plan, run 2 returns 5 plans, etc.
    - steps missing in some runs.
    """
    results: List[PlannerRunResult] = []
    for _ in range(runs):
        results.append(run_planner_deterministic(planner_call=planner_call, payload=base_payload))

    first = results[0]
    first_count = len(first.plans)

    for idx, res in enumerate(results[1:], start=2):
        count = len(res.plans)
        if count != first_count:
            raise AssertionError(
                f"planner instability: run 1 had {first_count} plans, "
                f"run {idx} had {count} plans"
            )

    # If there are no plans, there's nothing further to compare.
    if first_count == 0:
        return

    # Compare basic shape of first plan's steps across runs
    def plan_signature(plan: Dict[str, Any]) -> Any:
        steps = plan.get("steps", [])
        if not isinstance(steps, list):
            return ("invalid", None)
        # Only consider the 'skill' key of each step for now
        return tuple(step.get("skill") for step in steps)

    first_sig = plan_signature(first.plans[0])

    for idx, res in enumerate(results[1:], start=2):
        sig = plan_signature(res.plans[0])
        if sig != first_sig:
            raise AssertionError(
                f"planner instability in steps: signature mismatch between "
                f"run 1 and run {idx}"
            )
