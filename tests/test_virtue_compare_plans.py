# tests/test_virtue_compare_plans.py

from virtues.schema import PlanSummary
from virtues.lattice import compare_plans
from virtues.loader import load_virtue_config


def _plan(
    pid: str,
    *,
    risk_level: float,
    resource_cost: float,
    tech_progress_score: float,
    time_cost: float = 5.0,
) -> PlanSummary:
    return PlanSummary(
        id=pid,
        time_cost=time_cost,
        resource_cost=resource_cost,
        risk_level=risk_level,
        pollution_level=0.2,
        infra_reuse_score=0.5,
        infra_impact_score=0.5,
        novelty_score=0.5,
        aesthetic_score=0.5,
        complexity_score=0.3,
        tech_progress_score=tech_progress_score,
        stability_score=0.5,
        reversibility_score=0.5,
        context_features={},
    )


def test_compare_plans_prefers_safer_progress_plan():
    config = load_virtue_config()

    # Plan 0: high risk, slightly more progress
    risky = _plan(
        "risky",
        risk_level=0.9,
        resource_cost=2.0,
        tech_progress_score=0.8,
    )

    # Plan 1: much safer, slightly less progress
    safe = _plan(
        "safe",
        risk_level=0.1,
        resource_cost=2.0,
        tech_progress_score=0.7,
    )

    best, scores = compare_plans(
        plans=[risky, safe],
        context_id="lv_bootstrap",
        config=config,
    )

    # Should pick “safe” in LV context, where Discipline & Foundation are heavy
    assert best.id == "safe"

    # Derived virtues should reflect that too (e.g. patience, prudence)
    risky_score = next(s for s in scores if s.plan_id == "risky")
    safe_score = next(s for s in scores if s.plan_id == "safe")

    patience_safe = safe_score.derived_virtues.get("patience", 0.0)
    patience_risky = risky_score.derived_virtues.get("patience", 0.0)
    assert patience_safe > patience_risky
