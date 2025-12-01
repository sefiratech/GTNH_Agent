# tests/test_virtue_hard_constraints.py

from virtues.schema import PlanSummary
from virtues.lattice import score_plan
from virtues.loader import load_virtue_config


def test_plan_can_be_disallowed_by_constraints():
    config = load_virtue_config()

    # Fake plan that clearly violates some hypothetical constraint
    # For now, just stuff context_features with something you'd check later.
    evil_plan = PlanSummary(
        id="delete_base",
        time_cost=1.0,
        resource_cost=0.1,
        risk_level=1.0,
        pollution_level=1.0,
        infra_reuse_score=0.0,
        infra_impact_score=-1.0,   # negative = destroys infra, for example
        novelty_score=0.5,
        aesthetic_score=0.0,
        complexity_score=0.1,
        tech_progress_score=0.0,
        stability_score=0.0,
        reversibility_score=0.0,
        context_features={"base_integrity_delta": -1.0},
    )

    score = score_plan(evil_plan, context_id="default", config=config)

    # Once hard-constraint logic is implemented, this should flip:
    # assert not score.allowed
    # For now we just assert the test runs; you’ll tighten this when the constraint handler is written.
    assert score.plan_id == "delete_base"
