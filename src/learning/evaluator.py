#"src/learning/evaluator.py"

"""
M10 Skill Evaluator

Compares baseline skill performance vs candidate performance using:
- success rate
- estimated time
- estimated resource cost
- virtue scores (via M4)

Evaluation logic:
1. Aggregate SkillPerformanceStats for a given skill across episodes.
2. Use M4 virtue lattice to compute virtue-aware metrics.
3. Emit a recommendation and reasons:
   - "promote_candidate"
   - "keep_baseline"
   - "reject_candidate"

Q1.2:
- The evaluator's recommendation is used by SkillLearningManager to decide
  whether to register and/or promote a candidate skill in the SkillRegistry.
"""

from __future__ import annotations

from typing import Any, Dict, List, Callable, Protocol, Optional

from .schema import ExperienceEpisode, SkillPerformanceStats, SkillName
from semantics.schema import TechState
from observation.trace_schema import PlanTrace

# This one you already have; it's harmless at import time.
from virtues.loader import load_virtue_config  # type: ignore[import]


# ---------------------------------------------------------------------------
# Protocols / type aliases
# ---------------------------------------------------------------------------

class PlanMetricsProtocol(Protocol):
    estimated_time: float
    estimated_resource_cost: float


ExtractMetricsFn = Callable[
    [Any, Any, TechState, Any, Dict[str, Dict[str, Any]]],
    PlanMetricsProtocol,
]
ScorePlanFn = Callable[
    [Any, str, PlanMetricsProtocol, Any],
    Dict[str, Any],
]


# ---------------------------------------------------------------------------
# Module-level shims (what your tests expect to patch)
# ---------------------------------------------------------------------------

def extract_plan_metrics(
    plan: Any,
    world: Any,
    tech_state: TechState,
    db: Any,
    skill_metadata: Dict[str, Dict[str, Any]],
) -> PlanMetricsProtocol:
    """
    Shim that lazily imports virtues.metrics.extract_plan_metrics.

    Tests are free to monkeypatch THIS symbol directly.
    """
    from virtues.metrics import extract_plan_metrics as _fn  # type: ignore[import]
    return _fn(
        plan=plan,
        world=world,
        tech_state=tech_state,
        db=db,
        skill_metadata=skill_metadata,
    )


def score_plan(
    plan: Any,
    context_id: str,
    metrics: PlanMetricsProtocol,
    config: Any,
) -> Dict[str, Any]:
    """
    Shim that lazily imports virtues.lattice.score_plan.

    Tests can monkeypatch THIS symbol as well.
    """
    from virtues.lattice import score_plan as _fn  # type: ignore[import]
    return _fn(
        plan=plan,
        context_id=context_id,
        metrics=metrics,
        config=config,
    )


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class SkillEvaluator:
    """
    Compute metrics and compare skill performance, including virtue scores.

    Responsibilities:
    - Aggregate SkillPerformanceStats for a single skill over many episodes.
    - Compare baseline vs candidate using simple heuristics.
    - Produce recommendation + reasons for M10/M11 to consume.

    You can:
    - Let it use the default `extract_plan_metrics` / `score_plan` shims, or
    - Inject custom functions (especially in tests).
    """

    def __init__(
        self,
        *,
        extract_metrics_fn: Optional[ExtractMetricsFn] = None,
        score_plan_fn: Optional[ScorePlanFn] = None,
        virtue_config: Any | None = None,
    ) -> None:
        """
        Parameters
        ----------
        extract_metrics_fn:
            Optional override for extracting metrics from a plan.
        score_plan_fn:
            Optional override for scoring a plan with virtues.
        virtue_config:
            Optional pre-loaded virtue config; defaults to load_virtue_config().
        """
        self._virtue_config = virtue_config if virtue_config is not None else load_virtue_config()
        self._extract_metrics: ExtractMetricsFn = extract_metrics_fn or extract_plan_metrics
        self._score_plan: ScorePlanFn = score_plan_fn or score_plan

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def aggregate_skill_stats(
        self,
        episodes: List[ExperienceEpisode],
        skill_name: SkillName,
        *,
        context_id: str,
        skill_metadata: Dict[str, Dict[str, Any]],
        semantics_db: Any | None = None,
    ) -> SkillPerformanceStats:
        """
        Aggregate performance stats for a given skill across episodes.
        """
        uses = 0
        successes = 0
        time_total = 0.0
        resource_total = 0.0
        virtue_accum: Dict[str, float] = {}

        for ep in episodes:
            if not self._episode_uses_skill(ep, skill_name):
                continue

            uses += 1
            if ep.success:
                successes += 1

            trace: PlanTrace = ep.trace
            tech: TechState = ep.tech_state

            world_before = None
            if getattr(trace, "steps", None):
                first_step = trace.steps[0]
                world_before = getattr(first_step, "world_before", None)

            # Extract lower-level metrics via injected / default function
            metrics = self._extract_metrics(
                plan=trace.plan,
                world=world_before,
                tech_state=tech,
                db=semantics_db,
                skill_metadata=skill_metadata,
            )

            # Score the plan against the virtue lattice
            scores = self._score_plan(
                plan=trace.plan,
                context_id=context_id,
                metrics=metrics,
                config=self._virtue_config,
            )

            # Accumulate scalar metrics
            time_total += float(getattr(metrics, "estimated_time", 0.0))
            resource_total += float(getattr(metrics, "estimated_resource_cost", 0.0))

            virtue_scores = scores.get("virtue_scores") or {}
            for vname, vscore in virtue_scores.items():
                try:
                    v = float(vscore)
                except (TypeError, ValueError):
                    continue
                virtue_accum[vname] = virtue_accum.get(vname, 0.0) + v

        if uses == 0:
            return SkillPerformanceStats.zero(skill_name)

        avg_time = time_total / uses if uses > 0 else 0.0
        avg_resource = resource_total / uses if uses > 0 else 0.0
        avg_virtues = {k: v / uses for k, v in virtue_accum.items()}

        return SkillPerformanceStats(
            skill_name=skill_name,
            uses=uses,
            success_rate=successes / uses if uses > 0 else 0.0,
            avg_time=avg_time,
            avg_resource_cost=avg_resource,
            avg_virtue_scores=avg_virtues,
        )

    def _episode_uses_skill(
        self,
        episode: ExperienceEpisode,
        skill_name: SkillName,
    ) -> bool:
        """
        Return True if the given skill appears in any trace step meta.

        Assumes trace.steps[*].meta["skill"] is set when skills are invoked.
        """
        trace: PlanTrace = episode.trace
        for step in getattr(trace, "steps", []) or []:
            meta = getattr(step, "meta", {}) or {}
            if meta.get("skill") == skill_name:
                return True
        return False

    # ------------------------------------------------------------------
    # Comparison
    # ------------------------------------------------------------------

    def compare_stats(
        self,
        baseline: SkillPerformanceStats,
        candidate: SkillPerformanceStats,
        *,
        min_uses: int = 5,
        max_success_drop: float = 0.05,
        improvement_threshold: float = 0.1,
    ) -> Dict[str, Any]:
        """
        Compare baseline vs candidate and return a recommendation.

        Heuristics:
        - If candidate has too few uses: "reject_candidate" (insufficient data).
        - If candidate success rate is significantly lower: "reject_candidate".
        - If candidate success is comparable and it improves on time/resource
          or virtue scores: "promote_candidate".
        - Otherwise: "keep_baseline".

        Q1.2:
        - The "promote_candidate" recommendation is used by the manager as
          the first gate before attempting registry-level promotion.
        """
        reasons: List[str] = []

        # 1. Data sufficiency
        if candidate.uses < min_uses:
            reasons.append("candidate_insufficient_data")
            return {
                "recommendation": "reject_candidate",
                "reasons": reasons,
                "baseline": baseline.to_dict(),
                "candidate": candidate.to_dict(),
            }

        # 2. Success rate guardrail
        success_delta = candidate.success_rate - baseline.success_rate
        if success_delta < -max_success_drop:
            reasons.append("candidate_success_rate_too_low")
            return {
                "recommendation": "reject_candidate",
                "reasons": reasons,
                "baseline": baseline.to_dict(),
                "candidate": candidate.to_dict(),
            }

        # 3. Efficiency & virtue improvements
        improved = False

        # Time: lower is better
        if candidate.avg_time < baseline.avg_time * (1.0 - improvement_threshold):
            reasons.append("candidate_faster")
            improved = True

        # Resource cost: lower is better
        if candidate.avg_resource_cost < baseline.avg_resource_cost * (1.0 - improvement_threshold):
            reasons.append("candidate_cheaper")
            improved = True

        # Virtue scores: higher is better for each dimension
        virtue_improvements = self._compare_virtues(baseline, candidate, improvement_threshold)
        reasons.extend(virtue_improvements)
        if virtue_improvements:
            improved = True

        # 4. Final decision
        if improved and success_delta >= -max_success_drop:
            recommendation = "promote_candidate"
        elif success_delta >= -max_success_drop:
            recommendation = "keep_baseline"
        else:
            recommendation = "reject_candidate"

        return {
            "recommendation": recommendation,
            "reasons": reasons,
            "baseline": baseline.to_dict(),
            "candidate": candidate.to_dict(),
        }

    def _compare_virtues(
        self,
        baseline: SkillPerformanceStats,
        candidate: SkillPerformanceStats,
        improvement_threshold: float,
    ) -> List[str]:
        """
        Compare virtue scores and return a list of improvement reason strings.

        Example outputs:
        - "candidate_better_safety"
        - "candidate_better_efficiency"
        """
        reasons: List[str] = []
        for vname, cand_score in candidate.avg_virtue_scores.items():
            base_score = baseline.avg_virtue_scores.get(vname, 0.0)
            # Additive check so we don't explode around zero
            if cand_score > base_score + improvement_threshold:
                reasons.append(f"candidate_better_{vname}")
        return reasons

