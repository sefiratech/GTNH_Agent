# path: src/monitoring/integration.py
"""
Integration helpers for M9 – monitoring_and_tools.

This module provides convenience functions for emitting well-structured
MonitoringEvents from:

- M8 AgentLoop (phases, plans, steps, critic, virtues, experiences)
- M7 Observation encoding (planner payloads, trace structure)
- M6 BotCore (observations, actions)
- M3/M4 Semantics & Virtues (TechState inference, virtue scoring)

All functions are thin wrappers around monitoring.logger.log_event
and enforce consistent payload shapes across the codebase.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .bus import EventBus
from .events import EventType
from .logger import log_event


JsonDict = Dict[str, Any]


# ============================================================
# M8 – AgentLoop integration
# ============================================================

def emit_agent_phase_change(
    bus: EventBus,
    phase: str,
    episode_id: Optional[str] = None,
    context_id: Optional[str] = None,
) -> None:
    """
    Emit an AGENT_PHASE_CHANGE event when AgentLoop changes phase.

    Expected phases: "IDLE", "PLANNING", "EXECUTING", "RECOVERING", etc.
    """
    payload: JsonDict = {
        "phase": phase,
        "episode_id": episode_id,
        "context_id": context_id,
    }
    log_event(
        bus=bus,
        module="M8.agent_loop",
        event_type=EventType.AGENT_PHASE_CHANGE,
        message=f"Agent phase changed to {phase}",
        payload=payload,
        correlation_id=episode_id,
    )


def emit_plan_created(
    bus: EventBus,
    plan: JsonDict,
    goal: Optional[str],
    episode_id: str,
    context_id: Optional[str] = None,
) -> None:
    """
    Emit a PLAN_CREATED event when the planner returns a plan.

    `plan` is expected to be the raw planner output (JSON-like).
    """
    steps = plan.get("steps") or []
    payload: JsonDict = {
        "plan": plan,
        "goal": goal or "",
        "step_count": len(steps),
        "episode_id": episode_id,
        "context_id": context_id,
    }
    log_event(
        bus=bus,
        module="M8.agent_loop",
        event_type=EventType.PLAN_CREATED,
        message="New plan created",
        payload=payload,
        correlation_id=episode_id,
    )


def emit_plan_step_executed(
    bus: EventBus,
    episode_id: str,
    step_index: int,
    step_spec: JsonDict,
    trace_step_result: JsonDict,
    context_id: Optional[str] = None,
) -> None:
    """
    Emit a PLAN_STEP_EXECUTED event when a single plan step has been run.

    `step_spec` is the planner's step spec.
    `trace_step_result` should be a JSON-like representation of TraceStep.
    """
    payload: JsonDict = {
        "episode_id": episode_id,
        "context_id": context_id,
        "step_index": step_index,
        "step_spec": step_spec,
        "trace_step": trace_step_result,
    }
    log_event(
        bus=bus,
        module="M8.agent_loop",
        event_type=EventType.PLAN_STEP_EXECUTED,
        message=f"Plan step executed (index={step_index})",
        payload=payload,
        correlation_id=episode_id,
    )


def emit_plan_failed(
    bus: EventBus,
    episode_id: str,
    reason: str,
    failing_step_index: Optional[int] = None,
    context_id: Optional[str] = None,
) -> None:
    """
    Emit a PLAN_FAILED event when AgentLoop aborts execution of a plan.
    """
    payload: JsonDict = {
        "episode_id": episode_id,
        "context_id": context_id,
        "reason": reason,
        "step_index": failing_step_index,
    }
    log_event(
        bus=bus,
        module="M8.agent_loop",
        event_type=EventType.PLAN_FAILED,
        message=f"Plan failed: {reason}",
        payload=payload,
        correlation_id=episode_id,
    )


def emit_plan_evaluated(
    bus: EventBus,
    episode_id: str,
    plan_id: str,
    attempt_index: int,
    virtue_scores: Dict[str, float],
    failure_type: Optional[str],
    severity: Optional[str],
    context_id: Optional[str] = None,
) -> None:
    """
    Emit a PLAN_EVALUATED event after a pre-execution evaluation of a plan
    attempt (Critic + virtues).

    This should be called once per call to AgentLoop._evaluate_plan.
    """
    payload: JsonDict = {
        "episode_id": episode_id,
        "context_id": context_id,
        "plan_id": plan_id,
        "attempt_index": attempt_index,
        "virtue_scores": virtue_scores,
        "failure_type": failure_type,
        "severity": severity,
    }
    log_event(
        bus=bus,
        module="M8.agent_loop",
        event_type=EventType.PLAN_EVALUATED,
        message=f"Plan evaluated (attempt={attempt_index}, plan_id={plan_id})",
        payload=payload,
        correlation_id=episode_id,
    )


def emit_plan_retried(
    bus: EventBus,
    episode_id: str,
    plan_id: str,
    attempt_index: int,
    reason: str,
    remaining_budget: int,
    context_id: Optional[str] = None,
) -> None:
    """
    Emit a PLAN_RETRIED event when the self-eval loop decides to retry
    planning instead of executing the current attempt.
    """
    payload: JsonDict = {
        "episode_id": episode_id,
        "context_id": context_id,
        "plan_id": plan_id,
        "attempt_index": attempt_index,
        "reason": reason,
        "remaining_retry_budget": remaining_budget,
    }
    log_event(
        bus=bus,
        module="M8.agent_loop",
        event_type=EventType.PLAN_RETRIED,
        message=f"Plan retry scheduled (attempt={attempt_index}, plan_id={plan_id})",
        payload=payload,
        correlation_id=episode_id,
    )


def emit_plan_abandoned(
    bus: EventBus,
    episode_id: str,
    plan_id: str,
    attempt_index: int,
    reason: str,
    context_id: Optional[str] = None,
) -> None:
    """
    Emit a PLAN_ABANDONED event when the self-eval loop decides that
    planning should stop without executing the current attempt.
    """
    payload: JsonDict = {
        "episode_id": episode_id,
        "context_id": context_id,
        "plan_id": plan_id,
        "attempt_index": attempt_index,
        "reason": reason,
    }
    log_event(
        bus=bus,
        module="M8.agent_loop",
        event_type=EventType.PLAN_ABANDONED,
        message=f"Plan abandoned (attempt={attempt_index}, plan_id={plan_id})",
        payload=payload,
        correlation_id=episode_id,
    )


def emit_plan_outcome_evaluated(
    bus: EventBus,
    episode_id: str,
    plan_id: str,
    attempt_index: int,
    outcome: Dict[str, Any],
    context_id: Optional[str] = None,
) -> None:
    """
    Emit a PLAN_OUTCOME_EVALUATED event at the end of post-execution
    outcome evaluation (ErrorModel).

    `outcome` is expected to follow the ErrorModelResponse/CriticResponse-like
    dict shape:
        {
          "failure_type": ...,
          "severity": ...,
          "fix_suggestions": [...],
          "notes": "...",
          "raw_text": "...",
          ...
        }
    """
    payload: JsonDict = {
        "episode_id": episode_id,
        "context_id": context_id,
        "plan_id": plan_id,
        "attempt_index": attempt_index,
        "outcome": outcome,
    }
    log_event(
        bus=bus,
        module="M8.agent_loop",
        event_type=EventType.PLAN_OUTCOME_EVALUATED,
        message=f"Plan outcome evaluated (attempt={attempt_index}, plan_id={plan_id})",
        payload=payload,
        correlation_id=episode_id,
    )


def emit_critic_result(
    bus: EventBus,
    episode_id: str,
    critic_result: JsonDict,
    context_id: Optional[str] = None,
) -> None:
    """
    Emit a CRITIC_RESULT event after the critic evaluates the plan/trace.
    """
    payload: JsonDict = {
        "episode_id": episode_id,
        "context_id": context_id,
        "critic_result": critic_result,
    }
    log_event(
        bus=bus,
        module="M8.agent_loop",
        event_type=EventType.CRITIC_RESULT,
        message="Critic evaluation completed",
        payload=payload,
        correlation_id=episode_id,
    )


def emit_virtue_scores(
    bus: EventBus,
    episode_id: str,
    scores: Dict[str, Any],
    context_id: Optional[str] = None,
) -> None:
    """
    Emit a VIRTUE_SCORES event when VirtueEngine returns scores for a trace.
    """
    payload: JsonDict = {
        "episode_id": episode_id,
        "context_id": context_id,
        "scores": scores,
    }
    log_event(
        bus=bus,
        module="M8.agent_loop",
        event_type=EventType.VIRTUE_SCORES,
        message="Virtue scores computed for episode",
        payload=payload,
        correlation_id=episode_id,
    )


def emit_experience_recorded(
    bus: EventBus,
    experience_id: str,
    episode_id: str,
    env_profile_name: str,
    context_id: Optional[str],
    meta: JsonDict,
) -> None:
    """
    Emit a LOG event marking that an Experience has been added to the buffer.

    We don't define a dedicated EventType for this (yet); instead we rely on
    LOG + subtype for compatibility with the existing EventType enum.
    """
    payload: JsonDict = {
        "subtype": "EXPERIENCE_RECORDED",
        "experience_id": experience_id,
        "episode_id": episode_id,
        "env_profile_name": env_profile_name,
        "context_id": context_id,
        "meta": meta,
    }
    log_event(
        bus=bus,
        module="M8.agent_loop",
        event_type=EventType.LOG,
        message="Experience recorded",
        payload=payload,
        correlation_id=episode_id,
    )


# ============================================================
# M7 – Observation Encoding integration
# ============================================================

def emit_planner_observation_snapshot(
    bus: EventBus,
    episode_id: str,
    planner_payload: JsonDict,
    tech_state: Optional[JsonDict],
    context_id: Optional[str] = None,
) -> None:
    """
    Emit a SNAPSHOT event for the planner-side observation payload.

    Intended to be called just after the observation encoder builds
    the planner input payload.
    """
    payload: JsonDict = {
        "episode_id": episode_id,
        "context_id": context_id,
        "planner_payload": planner_payload,
        "tech_state": tech_state or {},
    }
    log_event(
        bus=bus,
        module="M7.observation",
        event_type=EventType.SNAPSHOT,
        message="Planner observation snapshot",
        payload=payload,
        correlation_id=episode_id,
    )


def emit_trace_structure_snapshot(
    bus: EventBus,
    episode_id: str,
    trace_summary: JsonDict,
    context_id: Optional[str] = None,
) -> None:
    """
    Emit a SNAPSHOT event describing the overall structure of a PlanTrace.

    `trace_summary` should be a compressed/summary representation of:
    - steps count
    - key timestamps
    - failure flags
    - any other relevant structural info
    """
    payload: JsonDict = {
        "episode_id": episode_id,
        "context_id": context_id,
        "trace": trace_summary,
    }
    log_event(
        bus=bus,
        module="M7.observation",
        event_type=EventType.SNAPSHOT,
        message="Trace structure snapshot",
        payload=payload,
        correlation_id=episode_id,
    )


# ============================================================
# M6 – BotCore integration
# ============================================================

def emit_observation_metadata(
    bus: EventBus,
    world_meta: JsonDict,
    context_id: Optional[str] = None,
) -> None:
    """
    Emit a LOG event summarizing the latest RawWorldSnapshot / WorldState.

    `world_meta` is expected to contain:
    - dimension
    - position
    - loaded_chunk_count
    - entity_counts, etc.
    """
    payload: JsonDict = {
        "subtype": "OBSERVATION_METADATA",
        "world": world_meta,
        "context_id": context_id,
    }
    log_event(
        bus=bus,
        module="M6.bot_core",
        event_type=EventType.LOG,
        message="Observation metadata",
        payload=payload,
        correlation_id=None,
    )


def emit_action_executed_from_botcore(
    bus: EventBus,
    action_type: str,
    params: JsonDict,
    success: bool,
    error: Optional[str],
    context_id: Optional[str] = None,
) -> None:
    """
    Emit an ACTION_EXECUTED event from BotCore when an action completes.

    This is lower-level than PLAN_STEP_EXECUTED and is focused on the
    BotCore action primitives.
    """
    payload: JsonDict = {
        "context_id": context_id,
        "action_type": action_type,
        "params": params,
        "success": success,
        "error": error,
    }
    log_event(
        bus=bus,
        module="M6.bot_core",
        event_type=EventType.ACTION_EXECUTED,
        message=f"Action executed: {action_type}",
        payload=payload,
        correlation_id=None,
    )


def emit_action_failure_short(
    bus: EventBus,
    action_type: str,
    reason: str,
    context_id: Optional[str] = None,
) -> None:
    """
    Emit a PLAN_FAILED-style LOG event for low-level action failures that
    don't map cleanly to a specific episode/step.
    """
    payload: JsonDict = {
        "subtype": "ACTION_FAILURE",
        "action_type": action_type,
        "reason": reason,
        "context_id": context_id,
    }
    log_event(
        bus=bus,
        module="M6.bot_core",
        event_type=EventType.LOG,
        message=f"Action failure: {action_type}",
        payload=payload,
        correlation_id=None,
    )


# ============================================================
# M3 & M4 – Semantics & Virtues integration
# ============================================================

def emit_tech_state_updated(
    bus: EventBus,
    tech_state_dict: JsonDict,
    episode_id: Optional[str] = None,
    context_id: Optional[str] = None,
) -> None:
    """
    Emit a TECH_STATE_UPDATED event when semantics.tech_state infers a new
    TechState for the agent.
    """
    payload: JsonDict = {
        "tech_state": tech_state_dict,
        "episode_id": episode_id,
        "context_id": context_id,
    }
    log_event(
        bus=bus,
        module="M3.semantics.tech_state",
        event_type=EventType.TECH_STATE_UPDATED,
        message="Tech state inferred/updated",
        payload=payload,
        correlation_id=episode_id,
    )


def emit_virtue_scores_from_engine(
    bus: EventBus,
    scores: Dict[str, Any],
    trace_meta: JsonDict,
    episode_id: Optional[str] = None,
    context_id: Optional[str] = None,
) -> None:
    """
    Emit a VIRTUE_SCORES event specifically from virtues.lattice / metrics,
    including trace metadata if available.
    """
    payload: JsonDict = {
        "scores": scores,
        "trace_meta": trace_meta,
        "episode_id": episode_id,
        "context_id": context_id,
    }
    log_event(
        bus=bus,
        module="M4.virtues",
        event_type=EventType.VIRTUE_SCORES,
        message="Virtue scores computed",
        payload=payload,
        correlation_id=episode_id,
    )

