# path: src/virtues/evaluator.py

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from .schema import VirtueConfig, PlanSummary, PlanScore

logger = logging.getLogger(__name__)

try:
    # Preferred: a single scoring entrypoint defined in metrics.py
    from .metrics import compute_plan_score  # type: ignore
except Exception:  # pragma: no cover - defensive import
    compute_plan_score = None  # type: ignore[misc]


def _coerce_plan_summary(plan: Any, plan_id_fallback: str = "plan") -> PlanSummary:
    """
    Best-effort conversion of an arbitrary object into a PlanSummary.

    This is intentionally forgiving so that M8 / M3 / M11 can pass in either:
      - a PlanSummary instance (ideal)
      - a dict with numeric feature fields
      - a dict-like object with .get()

    Any missing numeric fields default to 0.0, and context_features defaults
    to an empty dict.
    """
    if isinstance(plan, PlanSummary):
        return plan

    if not isinstance(plan, dict):
        # Try to treat it like an object with attributes or .__dict__
        if hasattr(plan, "__dict__"):
            plan = dict(plan.__dict__)
        else:
            logger.warning(
                "virtues._coerce_plan_summary received unsupported plan type %r; "
                "using zeroed PlanSummary.",
                type(plan),
            )
            plan = {}

    # Allow id under multiple common keys
    plan_id = (
        str(plan.get("id"))
        if plan.get("id") is not None
        else str(plan.get("plan_id"))
        if plan.get("plan_id") is not None
        else plan_id_fallback
    )

    def _num(key: str) -> float:
        v = plan.get(key, 0.0)
        try:
            return float(v)
        except Exception:
            return 0.0

    context_features = plan.get("context_features") or {}
    if not isinstance(context_features, dict):
        context_features = {}

    return PlanSummary(
        id=plan_id,
        time_cost=_num("time_cost"),
        resource_cost=_num("resource_cost"),
        risk_level=_num("risk_level"),
        pollution_level=_num("pollution_level"),
        infra_reuse_score=_num("infra_reuse_score"),
        infra_impact_score=_num("infra_impact_score"),
        novelty_score=_num("novelty_score"),
        aesthetic_score=_num("aesthetic_score"),
        complexity_score=_num("complexity_score"),
        tech_progress_score=_num("tech_progress_score"),
        stability_score=_num("stability_score"),
        reversibility_score=_num("reversibility_score"),
        context_features=context_features,
    )


def _score_with_builtin_logic(
    plan_summary: PlanSummary,
    virtue_config: VirtueConfig,
    context_id: str,
) -> PlanScore:
    """
    Extremely simple fallback scoring logic used if virtues.metrics
    does not expose `compute_plan_score`.

    This is NOT meant to be clever; it's just a reasonable default so that
    callers always get something structured back.

    Strategy:
        - overall_score: heuristic combining tech_progress, stability,
          reversibility, inverse risk/resource/pollution.
        - derived_virtues: a single "prudence" and "ambition" as examples.
        - allowed: True unless risk_level is extremely high.
    """
    # Heuristic weights; these can be tuned or replaced by metrics.compute_plan_score
    tech = plan_summary.tech_progress_score
    stability = plan_summary.stability_score
    reversibility = plan_summary.reversibility_score
    risk = plan_summary.risk_level
    resource = plan_summary.resource_cost
    pollution = plan_summary.pollution_level

    # Normalize a bit so it doesn't go completely insane
    overall = (
        1.5 * tech
        + 1.0 * stability
        + 0.5 * reversibility
        - 1.0 * risk
        - 0.5 * resource
        - 0.5 * pollution
    )

    derived_virtues: Dict[str, float] = {
        "prudence": max(0.0, 1.0 - risk / 10.0),
        "ambition": max(0.0, min(1.0, tech / 10.0)),
    }

    allowed = risk < 8.0
    disallowed_reason: Optional[str] = None
    if not allowed:
        disallowed_reason = "risk_level_too_high"

    # Node-level detail is not computed here; we keep it minimal.
    node_scores: Dict[str, Any] = {}

    return PlanScore(
        plan_id=plan_summary.id,
        context_id=context_id,
        node_scores=node_scores,               # type: ignore[arg-type]
        derived_virtues=derived_virtues,
        overall_score=overall,
        allowed=allowed,
        disallowed_reason=disallowed_reason,
    )


def evaluate_plan_with_virtues(
    plan: Any,
    world_state: Any,
    virtue_config: VirtueConfig,
    context_id: str = "default",
) -> Dict[str, float]:
    """
    High-level entrypoint for virtue evaluation used by M8 / AgentLoop.

    This function is intentionally simple from the caller's perspective:

        scores = evaluate_plan_with_virtues(plan, world_state, virtue_config)

    Inputs:
        plan:
            Either:
              - a PlanSummary instance (preferred), or
              - a dict with numeric feature fields as defined in PlanSummary.

        world_state:
            Reserved for future use (e.g. tech band, biome, danger level).
            The current implementation does not inspect it directly; any
            required world-derived features should already be baked into
            PlanSummary.context_features by M3 / semantics.

        virtue_config:
            Loaded VirtueConfig describing the lattice, contexts, and derived
            virtues.

        context_id:
            Which VirtueContext to use (e.g. "eco_factory", "speedrun").
            If not found in virtue_config.contexts, a default weighting is
            assumed by the scoring function.

    Returns:
        Dict[str, float] mapping:
            virtue_id -> score

        Typically this is:
            - `PlanScore.derived_virtues`
        but it may also include an "overall" entry when convenient.
    """
    del world_state  # currently unused; kept for signature stability

    plan_summary = _coerce_plan_summary(plan)

    # Preferred path: delegate to virtues.metrics.compute_plan_score
    if compute_plan_score is not None:
        try:
            plan_score: PlanScore = compute_plan_score(
                plan_summary=plan_summary,
                virtue_config=virtue_config,
                context_id=context_id,
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(
                "virtues.evaluate_plan_with_virtues: compute_plan_score failed: %r; "
                "falling back to builtin heuristic.",
                exc,
            )
            plan_score = _score_with_builtin_logic(plan_summary, virtue_config, context_id)
    else:
        plan_score = _score_with_builtin_logic(plan_summary, virtue_config, context_id)

    scores: Dict[str, float] = dict(plan_score.derived_virtues)
    # Optionally expose overall score under a standard key
    scores.setdefault("overall", plan_score.overall_score)
    return scores

