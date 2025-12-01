# path: tests/test_q1_control_and_experience.py

"""
Q1 – Self-eval + Retry + Experience + Monitoring

These tests target the *contracts* for:
- ExperienceEpisode / ExperienceBuffer (M10)
- CriticResponse / ErrorModelResponse shapes (M2 spec)
- Monitoring event helpers (M9)

They are intentionally written so they can run before the full AgentLoop
Q1 control flow is implemented, but they encode the scenarios that
AgentLoop must eventually satisfy:

1. Single plan with no retries.
2. Plan rejected once, accepted on retry.
3. Plan abandoned based on severity.
4. Critic and ErrorModel responses using compatible schemas.
5. Monitoring events emitted in the correct order.
6. ExperienceEpisode containing both pre- and post-evaluation data.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest

# M10 experience / buffer
from learning.buffer import ExperienceBuffer
from learning.schema import ExperienceEpisode

# M2 spec: shared critic / error-model response shapes
from spec.llm import CriticResponse, ErrorModelResponse

# M9 monitoring
from monitoring.events import EventType
import monitoring.integration as integ


# ---------------------------------------------------------------------------
# Helpers for tests
# ---------------------------------------------------------------------------


class DummyTechState:
    """Minimal stand-in for semantics.schema.TechState."""
    def __init__(self, active: str = "steam_age") -> None:
        self.active = active

    def to_dict(self) -> Dict[str, Any]:
        return {"active": self.active}


class DummyTrace:
    """Minimal stand-in for observation.trace_schema.PlanTrace."""
    def __init__(self, steps: int) -> None:
        self.steps = [{"meta": {"skill": "dummy_skill"}} for _ in range(steps)]

    def to_dict(self) -> Dict[str, Any]:
        return {"steps": list(self.steps)}


def _tech_state_to_dict(ts: Any) -> Dict[str, Any]:
    if hasattr(ts, "to_dict"):
        return ts.to_dict()
    return {"raw": str(ts)}


def _tech_state_from_dict(d: Dict[str, Any]) -> Any:
    # For test purposes we don't reconstruct a full TechState.
    return d


def _trace_to_dict(tr: Any) -> Dict[str, Any]:
    if hasattr(tr, "to_dict"):
        return tr.to_dict()
    return {"raw": str(tr)}


def _trace_from_dict(d: Dict[str, Any]) -> Any:
    # For test purposes we don't reconstruct a full PlanTrace.
    return d


@pytest.fixture
def tmp_experience_buffer(tmp_path: Path) -> ExperienceBuffer:
    path = tmp_path / "experience.jsonl"
    return ExperienceBuffer(
        path=path,
        tech_state_to_dict=_tech_state_to_dict,
        tech_state_from_dict=_tech_state_from_dict,
        trace_to_dict=_trace_to_dict,
        trace_from_dict=_trace_from_dict,
    )


@pytest.fixture
def event_recorder(monkeypatch: pytest.MonkeyPatch) -> Tuple[List[Dict[str, Any]], Any]:
    """
    Capture all monitoring events emitted via monitoring.integration.log_event.

    Returns:
        (events, bus)

        events: list of dicts
            Each dict has keys: module, event_type, message, payload, correlation_id

        bus: EventBus instance (real or dummy)
    """
    recorded: List[Dict[str, Any]] = []

    def fake_log_event(
        bus: Any,
        module: str,
        event_type: EventType,
        message: str,
        payload: Dict[str, Any],
        correlation_id: str | None,
    ) -> None:
        recorded.append(
            {
                "module": module,
                "event_type": event_type,
                "message": message,
                "payload": payload,
                "correlation_id": correlation_id,
            }
        )

    # Patch the symbol imported in integration.py
    monkeypatch.setattr(integ, "log_event", fake_log_event, raising=True)

    # Instantiate the real EventBus (its internals don't matter here)
    from monitoring.bus import EventBus  # type: ignore

    return recorded, EventBus()


# ---------------------------------------------------------------------------
# 1 & 2 & 3. Monitoring sequences for single / retry / abandon
# ---------------------------------------------------------------------------


def _event_types(events: List[Dict[str, Any]]) -> List[EventType]:
    return [e["event_type"] for e in events]


def test_single_plan_no_retries_event_sequence(event_recorder) -> None:
    """
    Scenario: single plan, accepted, executed, outcome evaluated.

    Expected event type order (at minimum):
        PLAN_CREATED
        PLAN_EVALUATED
        PLAN_OUTCOME_EVALUATED
    """
    events, bus = event_recorder
    episode_id = "ep_single"
    context_id = "ctx"

    plan = {"id": "plan1", "steps": [{"skill": "s1", "params": {}}]}
    # Planner result
    integ.emit_plan_created(
        bus=bus,
        plan=plan,
        goal="build coke ovens",
        episode_id=episode_id,
        context_id=context_id,
    )

    # Pre-eval
    integ.emit_plan_evaluated(
        bus=bus,
        episode_id=episode_id,
        plan_id="plan1",
        attempt_index=0,
        virtue_scores={"prudence": 0.9},
        failure_type=None,
        severity=None,
        context_id=context_id,
    )

    # Post-outcome evaluation
    integ.emit_plan_outcome_evaluated(
        bus=bus,
        episode_id=episode_id,
        plan_id="plan1",
        attempt_index=0,
        outcome={
            "failure_type": None,
            "severity": None,
            "fix_suggestions": [],
            "notes": "ok",
        },
        context_id=context_id,
    )

    types = _event_types(events)
    assert types[0] == EventType.PLAN_CREATED
    assert EventType.PLAN_EVALUATED in types
    assert EventType.PLAN_OUTCOME_EVALUATED in types

    # Ensure ordering is at least monotonic: created -> evaluated -> outcome
    assert types.index(EventType.PLAN_CREATED) < types.index(EventType.PLAN_EVALUATED)
    assert types.index(EventType.PLAN_EVALUATED) < types.index(
        EventType.PLAN_OUTCOME_EVALUATED
    )


def test_plan_rejected_once_then_accepted_event_sequence(event_recorder) -> None:
    """
    Scenario: first plan attempt rejected and retried, second accepted.

    Expected *relative* event sequence:
        PLAN_CREATED (attempt 0)
        PLAN_EVALUATED (attempt 0)
        PLAN_RETRIED
        PLAN_CREATED (attempt 1)
        PLAN_EVALUATED (attempt 1)
        PLAN_OUTCOME_EVALUATED
    """
    events, bus = event_recorder
    episode_id = "ep_retry"
    context_id = "ctx"

    # First attempt
    integ.emit_plan_created(
        bus=bus,
        plan={"id": "plan1", "steps": []},
        goal="build coke ovens",
        episode_id=episode_id,
        context_id=context_id,
    )
    integ.emit_plan_evaluated(
        bus=bus,
        episode_id=episode_id,
        plan_id="plan1",
        attempt_index=0,
        virtue_scores={"prudence": 0.2},
        failure_type="virtue_violation",
        severity="medium",
        context_id=context_id,
    )
    integ.emit_plan_retried(
        bus=bus,
        episode_id=episode_id,
        plan_id="plan1",
        attempt_index=0,
        reason="virtue_score_too_low",
        remaining_budget=1,
        context_id=context_id,
    )

    # Second attempt
    integ.emit_plan_created(
        bus=bus,
        plan={"id": "plan2", "steps": [{"skill": "s1", "params": {}}]},
        goal="build coke ovens",
        episode_id=episode_id,
        context_id=context_id,
    )
    integ.emit_plan_evaluated(
        bus=bus,
        episode_id=episode_id,
        plan_id="plan2",
        attempt_index=1,
        virtue_scores={"prudence": 0.8},
        failure_type=None,
        severity=None,
        context_id=context_id,
    )
    integ.emit_plan_outcome_evaluated(
        bus=bus,
        episode_id=episode_id,
        plan_id="plan2",
        attempt_index=1,
        outcome={"failure_type": None, "severity": None, "fix_suggestions": []},
        context_id=context_id,
    )

    types = _event_types(events)

    # Shape checks
    assert types.count(EventType.PLAN_CREATED) == 2
    assert types.count(EventType.PLAN_EVALUATED) == 2
    assert EventType.PLAN_RETRIED in types
    assert EventType.PLAN_OUTCOME_EVALUATED in types

    # Order constraints: created0 < eval0 < retried < created1 < eval1 < outcome
    idx_created0 = types.index(EventType.PLAN_CREATED)
    idx_eval0 = types.index(EventType.PLAN_EVALUATED)
    idx_retried = types.index(EventType.PLAN_RETRIED)
    # second occurrences:
    idx_created1 = types.index(EventType.PLAN_CREATED, idx_retried + 1)
    idx_eval1 = types.index(EventType.PLAN_EVALUATED, idx_created1 + 1)
    idx_outcome = types.index(EventType.PLAN_OUTCOME_EVALUATED)

    assert idx_created0 < idx_eval0 < idx_retried < idx_created1 < idx_eval1 < idx_outcome


def test_plan_abandoned_based_on_severity_event_sequence(event_recorder) -> None:
    """
    Scenario: plan evaluated as too severe a failure risk and abandoned.

    Expected relative sequence:
        PLAN_CREATED
        PLAN_EVALUATED
        PLAN_ABANDONED
    """
    events, bus = event_recorder
    episode_id = "ep_abandon"
    context_id = "ctx"

    integ.emit_plan_created(
        bus=bus,
        plan={"id": "danger_plan", "steps": []},
        goal="do something reckless",
        episode_id=episode_id,
        context_id=context_id,
    )
    integ.emit_plan_evaluated(
        bus=bus,
        episode_id=episode_id,
        plan_id="danger_plan",
        attempt_index=0,
        virtue_scores={"prudence": 0.0},
        failure_type="high_risk",
        severity="high",
        context_id=context_id,
    )
    integ.emit_plan_abandoned(
        bus=bus,
        episode_id=episode_id,
        plan_id="danger_plan",
        attempt_index=0,
        reason="severity_high",
        context_id=context_id,
    )

    types = _event_types(events)
    assert EventType.PLAN_CREATED in types
    assert EventType.PLAN_EVALUATED in types
    assert EventType.PLAN_ABANDONED in types

    assert types.index(EventType.PLAN_CREATED) < types.index(EventType.PLAN_EVALUATED)
    assert types.index(EventType.PLAN_EVALUATED) < types.index(EventType.PLAN_ABANDONED)


# ---------------------------------------------------------------------------
# 4. Critic vs ErrorModel response schema compatibility
# ---------------------------------------------------------------------------


def test_critic_and_error_model_responses_share_failure_shape() -> None:
    """
    The CriticResponse and ErrorModelResponse types must share the
    `failure_type`, `severity`, and `fix_suggestions` fields so that
    they can both be mapped into a common PlanEvaluation / EpisodeOutcome
    reducer.
    """
    critic_resp = CriticResponse(
        ok=False,
        critique="too risky",
        suggested_modifications={"lower_risk": True},
        failure_type="risk_violation",
        severity="high",
        fix_suggestions=["reduce risk", "use safer setup"],
        raw_text="...",
    )

    error_resp = ErrorModelResponse(
        classification="execution_failure",
        summary="bot fell into lava",
        suggested_fix={"hint": "avoid open lava path"},
        retry_advised=False,
        failure_type="execution_failure",
        severity="high",
        fix_suggestions=["adjust pathfinding", "build guard rails"],
        raw_text="...",
    )

    # Critical fields exist and are populated:
    for obj in (critic_resp, error_resp):
        assert hasattr(obj, "failure_type")
        assert hasattr(obj, "severity")
        assert hasattr(obj, "fix_suggestions")

        # Values are JSON-serializable in the naive sense
        json.dumps(
            {
                "failure_type": obj.failure_type,
                "severity": obj.severity,
                "fix_suggestions": obj.fix_suggestions,
            }
        )


# ---------------------------------------------------------------------------
# 6. ExperienceEpisode & ExperienceBuffer – pre & post eval data
# ---------------------------------------------------------------------------


def test_experience_episode_roundtrip_has_pre_and_post_eval(
    tmp_experience_buffer: ExperienceBuffer,
) -> None:
    """
    Ensure that ExperienceEpisode includes both pre- and post-evaluation
    data when serialized to / from ExperienceBuffer.

    This covers:
    - goal
    - plan
    - pre_eval
    - post_eval
    - final_outcome
    - virtue_scores
    - failure_type / severity
    """
    buffer = tmp_experience_buffer

    ep = ExperienceEpisode(
        id="ep1",
        goal="build coke ovens",
        plan={"id": "plan1", "steps": [{"skill": "build_coke_ovens", "params": {}}]},
        pre_eval={
            "ok": True,
            "failure_type": None,
            "severity": None,
            "virtue_scores": {"prudence": 0.9},
        },
        post_eval={
            "failure_type": None,
            "severity": None,
            "fix_suggestions": [],
        },
        final_outcome={
            "result": "success",
            "notes": "coke ovens built",
        },
        tech_state=DummyTechState(active="steam_age"),
        trace=DummyTrace(steps=3),
        virtue_scores={"prudence": 0.9, "overall": 0.85},
        success=True,
        failure_type=None,
        severity=None,
    )

    # Append via the canonical Q1-style entrypoint
    buffer.append_experience(ep)

    # Raw read: ensure keys are present
    raws = list(buffer.load_all_raw())
    assert len(raws) == 1
    raw = raws[0]

    for key in ("pre_eval", "post_eval", "final_outcome", "plan"):
        assert key in raw, f"missing key {key} in raw experience dict"

    assert raw["goal"] == "build coke ovens"
    assert raw["plan"]["id"] == "plan1"
    assert "virtue_scores" in raw
    assert raw["virtue_scores"]["prudence"] == pytest.approx(0.9)
    assert "failure_type" in raw
    assert "severity" in raw

    # Full typed roundtrip
    ep_roundtrip = next(buffer.load_all())
    assert ep_roundtrip.goal == ep.goal
    assert ep_roundtrip.plan["id"] == ep.plan["id"]
    assert ep_roundtrip.pre_eval["ok"] is True
    assert ep_roundtrip.post_eval["failure_type"] is None
    assert ep_roundtrip.final_outcome["result"] == "success"
    assert ep_roundtrip.virtue_scores["overall"] == pytest.approx(0.85)

