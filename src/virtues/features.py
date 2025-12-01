# src/virtues/features.py

from typing import Dict, Any

from .schema import PlanSummary
from spec.types import WorldState  # from M1/M3
from semantics.schema import TechState
from semantics.loader import SemanticsDB


def summarize_plan(
    plan: Dict[str, Any],
    world: WorldState,
    tech_state: TechState,
    db: SemanticsDB,
    skill_metadata: Dict[str, Dict[str, Any]],
) -> PlanSummary:
    """
    Turn a raw plan + semantics into a PlanSummary with numeric features.
    """

    steps = plan.get("steps", [])

    # Placeholder heuristics; you’ll tune later.
    time_cost = float(len(steps))
    resource_cost = 0.0
    risk_level = 0.0
    pollution_level = 0.0
    infra_reuse_score = 0.0
    infra_impact_score = 0.0
    novelty_score = 0.0
    aesthetic_score = plan.get("aesthetic_hint", 0.0)
    complexity_score = min(1.0, len(steps) / 20.0)
    tech_progress_score = 0.0
    stability_score = 0.5
    reversibility_score = 0.5

    for step in steps:
        skill = step.get("skill", "")
        meta = skill_metadata.get(skill, {})
        tags = meta.get("tags", [])

        resource_cost += float(meta.get("estimated_resource_cost", 0.0))
        risk_level += float(meta.get("risk_delta", 0.0))
        pollution_level += float(meta.get("pollution_delta", 0.0))
        infra_reuse_score += float(meta.get("infra_reuse", 0.0))
        infra_impact_score += float(meta.get("infra_impact", 0.0))
        novelty_score += float(meta.get("novelty_delta", 0.0))
        tech_progress_score += float(meta.get("progress_value", 0.0))
        stability_score += float(meta.get("stability_delta", 0.0))
        reversibility_score += float(meta.get("reversibility_delta", 0.0))

    # Clamp / normalize as needed
    risk_level = max(0.0, min(1.0, risk_level))
    pollution_level = max(0.0, min(1.0, pollution_level))
    tech_progress_score = max(0.0, min(1.0, tech_progress_score))

    context_features: Dict[str, float] = {
        "tech_tier": float(tech_state.tier.value) if hasattr(tech_state, "tier") else 0.0,
    }

    return PlanSummary(
        id=plan.get("id", "plan"),
        time_cost=time_cost,
        resource_cost=resource_cost,
        risk_level=risk_level,
        pollution_level=pollution_level,
        infra_reuse_score=infra_reuse_score,
        infra_impact_score=infra_impact_score,
        novelty_score=novelty_score,
        aesthetic_score=aesthetic_score,
        complexity_score=complexity_score,
        tech_progress_score=tech_progress_score,
        stability_score=stability_score,
        reversibility_score=reversibility_score,
        context_features=context_features,
    )
