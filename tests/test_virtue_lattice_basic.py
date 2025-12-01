# tests/test_virtue_lattice_basic.py

from virtues.schema import PlanSummary
from virtues.lattice import score_plan
from virtues.loader import load_virtue_config


def _make_plan_summary(
    *,
    plan_id: str,
    time_cost: float,
    resource_cost: float,
    risk_level: float,
    pollution_level: float,
    tech_progress_score: float,
) -> PlanSummary:
    # Fill non-essential fields with neutral-ish defaults
    return PlanSummary(
        id=plan_id,
        time_cost=time_cost,
        resource_cost=resource_cost,
        risk_level=risk_level,
        pollution_level=pollution_level,
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


def test_basic_scoring_low_risk_high_progress_is_better():
    config = load_virtue_config()

    # Plan A: slower, more expensive, higher risk, low progress
    bad_plan = _make_plan_summary(
        plan_id="bad",
        time_cost=10.0,
        resource_cost=5.0,
        risk_level=0.8,           # high risk
        pollution_level=0.6,
        tech_progress_score=0.2,  # meh progress
    )

    # Plan B: cheaper, safer, better progress
    good_plan = _make_plan_summary(
        plan_id="good",
        time_cost=6.0,
        resource_cost=2.0,
        risk_level=0.1,           # low risk
        pollution_level=0.2,
        tech_progress_score=0.7,  # solid progress
    )

    bad_score = score_plan(bad_plan, context_id="lv_bootstrap", config=config)
    good_score = score_plan(good_plan, context_id="lv_bootstrap", config=config)

    assert bad_score.allowed
    assert good_score.allowed

    # sanity: scalar alignment should prefer the good plan
    assert good_score.overall_score > bad_score.overall_score

    # safety / “discipline” node should be better for the lower-risk plan
    good_disc = good_score.node_scores["discipline"].propagated
    bad_disc = bad_score.node_scores["discipline"].propagated
    assert good_disc > bad_disc

    # tech progress should show up in “drive” / “manifestation”
    assert good_score.node_scores["drive"].propagated >= bad_score.node_scores["drive"].propagated
    assert good_score.node_scores["manifestation"].propagated >= bad_score.node_scores["manifestation"].propagated
