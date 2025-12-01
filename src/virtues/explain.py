# src/virtues/explain.py

from .schema import PlanScore


def explain_plan(plan_score: PlanScore) -> dict:
    """
    Turn a PlanScore into a structured explanation for logs/UI.
    """
    core = {
        node_id: {
            "raw": ns.raw,
            "propagated": ns.propagated,
            "rationale": ns.rationale,
        }
        for node_id, ns in plan_score.node_scores.items()
    }

    return {
        "plan_id": plan_score.plan_id,
        "context_id": plan_score.context_id,
        "allowed": plan_score.allowed,
        "disallowed_reason": plan_score.disallowed_reason,
        "overall_score": plan_score.overall_score,
        "core_virtues": core,
        "derived_virtues": plan_score.derived_virtues,
    }
