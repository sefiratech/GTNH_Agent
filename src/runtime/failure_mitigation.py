# path: src/runtime/failure_mitigation.py

"""
Cross-phase failure handling helpers for GTNH_Agent.

This module centralizes how we turn failures into structured monitoring
events for M9, matching the "Failure points & mitigations (cross-phase)"
design.

It DOES NOT try to detect failures itself. Instead, it provides small,
explicit helpers that other modules can call when they hit trouble.

Failure classes covered:

1) Config / model mismatch (Phase 0 / 1)
   - Env loader or validators can call:
       emit_config_error(...)
       emit_model_path_error(...)

2) LLM failures (Phase 1 / AgentLoop)
   - LLM wrappers or planner/critic code can call:
       emit_llm_failure(...)
   - Uses LOG events with subtypes like:
       "LLM_TIMEOUT", "LLM_OOM", "LLM_BAD_OUTPUT", "LLM_BACKEND_ERROR"

3) BotCore / Minecraft IO (Phase 2)
   - BotCore action wrappers can call:
       emit_action_failure(...)
   - Optionally escalate to plan failure:
       emit_plan_failed_due_to_actions(...)

4) AgentLoop stepping (Phase 3)
   - Already handled by runtime.error_handling.safe_step_with_logging()
   - That helper logs "AGENT_STEP_EXCEPTION" events.
"""

from __future__ import annotations  # forward type references in type hints

from typing import Any, Dict, Iterable, Optional  # type hints

from monitoring.bus import EventBus                    # event bus used across system
from monitoring.events import EventType                # monitoring event type enum
from monitoring.logger import log_event                # convenience helper for JSONL logging


JsonDict = Dict[str, Any]


# ------------------------------------------------------------------------------
# 1. Config / model mismatch (Phase 0 / 1)
# ------------------------------------------------------------------------------

def emit_config_error(
    bus: EventBus,
    message: str,
    *,
    env_profile_repr: Optional[str] = None,
    error_repr: Optional[str] = None,
) -> None:
    """
    Emit a LOG event for a general configuration error.

    Typical usage (Phase 0 / 1):

        try:
            env_profile = load_env_profile(...)
        except Exception as exc:
            emit_config_error(bus, "Failed to load env profile", error_repr=repr(exc))
            raise

    This does NOT exit the process by itself; caller decides whether to abort.
    """
    payload: JsonDict = {
        "subtype": "CONFIG_ERROR",
        "env_profile": env_profile_repr,
        "error": error_repr,
    }

    log_event(
        bus=bus,
        module="runtime.config",
        event_type=EventType.LOG,
        message=message,
        payload=payload,
        correlation_id=None,
    )


def emit_model_path_error(
    bus: EventBus,
    message: str,
    *,
    model_id: str,
    expected_path: str,
    error_repr: Optional[str] = None,
) -> None:
    """
    Emit a LOG event for a model path mismatch / missing file.

    Example usage:

        if not model_path.exists():
            emit_model_path_error(
                bus,
                "Model path does not exist",
                model_id=model_id,
                expected_path=str(model_path),
                error_repr=None,
            )
            raise FileNotFoundError(...)

    This corresponds to the "Config / model mismatch (Phase 0 / 1)" bullet.
    """
    payload: JsonDict = {
        "subtype": "MODEL_PATH_ERROR",
        "model_id": model_id,
        "expected_path": expected_path,
        "error": error_repr,
    }

    log_event(
        bus=bus,
        module="runtime.models",
        event_type=EventType.LOG,
        message=message,
        payload=payload,
        correlation_id=None,
    )


# ------------------------------------------------------------------------------
# 2. LLM failures (Phase 1 / AgentLoop)
# ------------------------------------------------------------------------------

def emit_llm_failure(
    bus: EventBus,
    *,
    subtype: str,
    role: str,
    model: str,
    episode_id: Optional[str],
    context_id: Optional[str],
    error_repr: str,
    meta: Optional[JsonDict] = None,
) -> None:
    """
    Emit a LOG event describing an LLM failure.

    Expected subtypes include (but are not limited to):
        - "LLM_TIMEOUT"
        - "LLM_OOM"
        - "LLM_BAD_OUTPUT"
        - "LLM_BACKEND_ERROR"

    Example usage in M2 / M8:

        try:
            result = planner_llm.call(prompt)
        except TimeoutError as exc:
            emit_llm_failure(
                bus,
                subtype="LLM_TIMEOUT",
                role="planner",
                model=planner_model_id,
                episode_id=episode_id,
                context_id=context_id,
                error_repr=repr(exc),
                meta={"timeout_s": timeout_value},
            )
            raise
    """
    payload: JsonDict = {
        "subtype": subtype,
        "llm_role": role,
        "model": model,
        "episode_id": episode_id,
        "context_id": context_id,
        "error": error_repr,
        "meta": meta or {},
    }

    log_event(
        bus=bus,
        module="runtime.llm",
        event_type=EventType.LOG,
        message=f"LLM failure ({subtype}) for role={role}, model={model}",
        payload=payload,
        correlation_id=episode_id,
    )


# ------------------------------------------------------------------------------
# 3. BotCore / Minecraft IO (Phase 2)
# ------------------------------------------------------------------------------

def emit_action_failure(
    bus: EventBus,
    *,
    episode_id: Optional[str],
    context_id: Optional[str],
    action_name: str,
    action_args: JsonDict,
    error_repr: str,
    module_name: str = "M6.bot_core",
) -> None:
    """
    Emit an ACTION_EXECUTED event for a failed action.

    This corresponds to "Emit ACTION_EXECUTED with success=False, error=...".

    Intended usage inside BotCore or its wrappers:

        try:
            res = self._do_click_block(...)
        except Exception as exc:
            emit_action_failure(
                bus=self.event_bus,
                episode_id=episode_id,
                context_id=context_id,
                action_name="click_block",
                action_args={"pos": pos.to_tuple()},
                error_repr=repr(exc),
            )
            raise

    Note:
    - success is explicitly False
    - error field includes a string representation of the cause
    """
    payload: JsonDict = {
        "episode_id": episode_id,
        "context_id": context_id,
        "action_name": action_name,
        "action_args": action_args,
        "success": False,
        "error": error_repr,
    }

    log_event(
        bus=bus,
        module=module_name,
        event_type=EventType.ACTION_EXECUTED,
        message=f"Action failed: {action_name}",
        payload=payload,
        correlation_id=episode_id,
    )


def emit_plan_failed_due_to_actions(
    bus: EventBus,
    *,
    episode_id: Optional[str],
    context_id: Optional[str],
    reason: str,
    failing_actions: Iterable[JsonDict],
    module_name: str = "M8.agent_loop",
) -> None:
    """
    Emit a PLAN_FAILED event when repeated action failures make
    continuing the plan pointless.

    This corresponds to:
        "Possibly PLAN_FAILED when repeatedly failing."

    `failing_actions` is a list (or other iterable) of summaries, e.g.:

        failing_actions = [
            {"action": "move_to", "error": "path_blocked"},
            {"action": "move_to", "error": "path_blocked"},
        ]

    Caller decides when to trigger this (e.g. after N retries).
    """
    payload: JsonDict = {
        "episode_id": episode_id,
        "context_id": context_id,
        "reason": reason,
        "failing_actions": list(failing_actions),
    }

    log_event(
        bus=bus,
        module=module_name,
        event_type=EventType.PLAN_FAILED,
        message=reason,
        payload=payload,
        correlation_id=episode_id,
    )


# ------------------------------------------------------------------------------
# 4. Monitoring & TUI overload
# ------------------------------------------------------------------------------

def emit_monitoring_overload_warning(
    bus: EventBus,
    *,
    approx_rate_hz: float,
    module_name: str = "M9.monitoring",
    context_id: Optional[str] = None,
) -> None:
    """
    Optional helper to log when you detect excessive event volume.

    This does NOT implement rate limiting by itself, but gives you
    a structured way to mark that you think you're spamming the bus/logs.

    Example usage:

        if events_per_second > 500:
            emit_monitoring_overload_warning(bus, approx_rate_hz=events_per_second)

    The design text notes:
    - Event bus is simple & in-process.
    - TUI samples latest state; worst case: slightly laggy.
    - Logger can drop events silently on disk failure.

    This helper just makes such conditions visible in the logs.
    """
    payload: JsonDict = {
        "subtype": "MONITORING_OVERLOAD",
        "approx_rate_hz": approx_rate_hz,
        "context_id": context_id,
    }

    log_event(
        bus=bus,
        module=module_name,
        event_type=EventType.LOG,
        message="High monitoring event rate detected",
        payload=payload,
        correlation_id=None,
    )

