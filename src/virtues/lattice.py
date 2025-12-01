# score_plan, compare_plans, weight merging
# src/virtues/lattice.py

from typing import Dict, List, Tuple

from .schema import (
    VirtueConfig,
    PlanSummary,
    NodeScore,
    PlanScore,
)
from .loader import load_virtue_config


def _normalize_feature(value: float, direction: str) -> float:
    """
    Map a raw feature value to [0, 1] where 1.0 is 'more virtuous'.
    Very simple for now; can be tuned later.
    """
    if direction == "higher_better":
        # Assume value in [0, inf); use 1 / (1 + x) inverse for cost-like,
        # but here we just clamp [0,1] assuming upstream normalized.
        return max(0.0, min(1.0, value))
    elif direction == "lower_better":
        # Assume input normalized [0,1] where 1 is worst
        return max(0.0, min(1.0, 1.0 - value))
    else:
        return 0.0


def compute_raw_node_scores(
    plan: PlanSummary,
    config: VirtueConfig,
) -> Dict[str, NodeScore]:
    """
    Use config.features to turn PlanSummary into raw node-local scores.
    """
    # Start everything at neutral 0.5
    scores: Dict[str, NodeScore] = {
        node_id: NodeScore(node_id=node_id, raw=0.5, propagated=0.5, rationale="")
        for node_id in config.nodes.keys()
    }

    feature_map = config.features

    # Map PlanSummary attributes by name
    plan_dict = {
      "time_cost": plan.time_cost,
      "resource_cost": plan.resource_cost,
      "risk_level": plan.risk_level,
      "pollution_level": plan.pollution_level,
      "infra_reuse_score": plan.infra_reuse_score,
      "infra_impact_score": plan.infra_impact_score,
      "novelty_score": plan.novelty_score,
      "aesthetic_score": plan.aesthetic_score,
      "complexity_score": plan.complexity_score,
      "tech_progress_score": plan.tech_progress_score,
      "stability_score": plan.stability_score,
      "reversibility_score": plan.reversibility_score,
    }

    for feature_name, spec in feature_map.items():
        if feature_name not in plan_dict:
            continue
        value = float(plan_dict[feature_name])
        for node_id, mapping in spec.get("affects", {}).items():
            direction = mapping.get("direction", "higher_better")
            weight = float(mapping.get("weight", 0.0))
            contrib = _normalize_feature(value, direction) * weight
            ns = scores[node_id]
            ns.raw += contrib
            scores[node_id] = ns

    # Simple clamp to [0,1]
    for node_id, ns in scores.items():
        ns.raw = max(0.0, min(1.0, ns.raw))
        ns.propagated = ns.raw

    return scores


def propagate_node_scores(
    raw_scores: Dict[str, NodeScore],
    config: VirtueConfig,
) -> Dict[str, NodeScore]:
    """
    Very simple propagation: parent nodes get a small contribution
    from their children, according to the edges.
    """
    scores = {k: NodeScore(**vars(v)) for k, v in raw_scores.items()}

    # For now: one forward pass over edges, no fancy ordering.
    for edge in config.edges:
        if edge.source not in scores or edge.target not in scores:
            continue
        src = scores[edge.source]
        tgt = scores[edge.target]
        # Let source pull up/down a bit based on target
        influence = 0.2  # small coupling
        src.propagated = max(
            0.0,
            min(1.0, (1 - influence) * src.propagated + influence * tgt.raw),
        )
        scores[edge.source] = src

    return scores


def compute_derived_virtues(
    node_scores: Dict[str, NodeScore],
    config: VirtueConfig,
) -> Dict[str, float]:
    """
    Combine core node scores into a small set of derived virtues
    using config.derived_virtues.
    """
    result: Dict[str, float] = {}
    for dv_id, spec in config.derived_virtues.items():
        num = 0.0
        den = 0.0
        for node_id, weight in spec.from_nodes.items():
            if node_id in node_scores:
                num += node_scores[node_id].propagated * weight
                den += abs(weight)
        result[dv_id] = num / den if den > 0 else 0.0
    return result


def score_plan(
    plan: PlanSummary,
    context_id: str,
    config: VirtueConfig | None = None,
) -> PlanScore:
    """
    Full scoring: raw → propagated → derived → overall scalar.
    """
    if config is None:
        config = load_virtue_config()

    context = config.contexts.get(context_id, config.contexts["default"])

    raw = compute_raw_node_scores(plan, config)
    propagated = propagate_node_scores(raw, config)

    # Apply hard constraints (very simple for now: just allowed/disallowed flag)
    allowed = True
    reason = None
    # TODO: actually use context.hard_constraints & plan.context_features

    # Overall scalar: weighted sum of propagated node scores
    overall = 0.0
    for node_id, ns in propagated.items():
        weight = context.node_weights.get(node_id, 0.0)
        overall += ns.propagated * weight

    # Derived virtues
    derived = compute_derived_virtues(propagated, config)

    return PlanScore(
        plan_id=plan.id,
        context_id=context_id,
        node_scores=propagated,
        derived_virtues=derived,
        overall_score=overall,
        allowed=allowed,
        disallowed_reason=reason,
    )


def compare_plans(
    plans: List[PlanSummary],
    context_id: str,
    config: VirtueConfig | None = None,
) -> Tuple[PlanSummary, List[PlanScore]]:
    """
    Score all plans and return (best_plan, all_scores).
    """
    if config is None:
        config = load_virtue_config()

    scores: List[PlanScore] = []
    best_idx = 0
    best_score = float("-inf")

    for idx, plan in enumerate(plans):
        ps = score_plan(plan, context_id, config)
        scores.append(ps)
        if not ps.allowed:
            continue
        if ps.overall_score > best_score:
            best_score = ps.overall_score
            best_idx = idx

    return plans[best_idx], scores
