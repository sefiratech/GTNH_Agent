# path: src/monitoring/events.py
"""
Event and command schemas for M9 – monitoring_and_tools.

This module defines:
- MonitoringEvent (structured system events)
- EventType enum
- ControlCommandType enum
- ControlCommand for human/system-issued controls

All events are JSON-serializable via `.to_dict()` and are intended
for use with monitoring.bus.EventBus and monitoring.logger.JsonFileLogger.
"""

from __future__ import annotations

import time
import logging
from dataclasses import dataclass, asdict
from enum import Enum, auto
from typing import Any, Dict, Optional

from spec.monitoring import EventSink, event_to_dict

logger = logging.getLogger(__name__)


# ============================================================
# Event Types
# ============================================================

class LoggingEventSink(EventSink):
    """
    Minimal EventSink that logs events via the standard logging module.

    This is enough for:
      - debugging hierarchical planning
      - quick grep-able traces
    """

    def emit(self, event: Any) -> None:
        payload = event_to_dict(event)
        logger.info("AgentEvent: %s", payload)

class EventType(Enum):
    """Typed monitoring events emitted throughout the agent system."""

    # Agent lifecycle phases (Idle, Planning, Executing, Recovering)
    AGENT_PHASE_CHANGE = auto()

    # Planner & plan structure
    PLAN_CREATED = auto()
    PLAN_STEP_EXECUTED = auto()
    PLAN_FAILED = auto()

    # Self-evaluation + retry loop (Q1.1)
    PLAN_EVALUATED = auto()          # pre-execution evaluation for an attempt
    PLAN_RETRIED = auto()            # a plan attempt was rejected and retried
    PLAN_ABANDONED = auto()          # plan was abandoned (no further retries)
    PLAN_OUTCOME_EVALUATED = auto()  # post-execution outcome evaluation

    # Low-level action execution (BotCore)
    ACTION_EXECUTED = auto()

    # World / semantics updates
    TECH_STATE_UPDATED = auto()

    # Virtue engine scoring
    VIRTUE_SCORES = auto()

    # Critic analysis
    CRITIC_RESULT = auto()

    # Full state snapshot (rare, expensive)
    SNAPSHOT = auto()

    # Control surface events
    CONTROL_COMMAND = auto()

    # Generic log messages
    LOG = auto()


# ============================================================
# Monitoring Event Structure
# ============================================================

@dataclass
class MonitoringEvent:
    """
    Runtime event emitted by the agent, planner, critic, BotCore,
    observation pipeline, or the control surface.

    All fields must be JSON-safe.
    """

    ts: float                   # UNIX timestamp (seconds)
    module: str                 # Source module string ("M8.agent_loop", "bot_core", etc.)
    event_type: EventType       # Enum describing the event class
    message: str                # Short human-readable description
    payload: Dict[str, Any]     # Structured data (plan, action, tech state, scores)
    correlation_id: Optional[str] = None  # Used for grouping events per plan/episode

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a JSON-safe dict for loggers."""
        data = asdict(self)
        data["event_type"] = self.event_type.name  # store name, not enum
        return data


# ============================================================
# Control Commands
# ============================================================

class ControlCommandType(Enum):
    """
    Commands that humans or automated tools can send to control the agent loop.
    """

    PAUSE = auto()          # Freeze AgentLoop
    RESUME = auto()         # Unpause loop
    SINGLE_STEP = auto()    # Execute exactly one planning/execution iteration
    CANCEL_PLAN = auto()    # Drop current plan
    SET_GOAL = auto()       # Set a new goal in the AgentLoop
    DUMP_STATE = auto()     # Emit a full snapshot event


@dataclass
class ControlCommand:
    """
    Represents an external command for the agent.

    Sent through EventBus.publish_command(), then interpreted by
    monitoring.controller.AgentController.
    """

    cmd: ControlCommandType             # The specific command
    args: Dict[str, Any]                # Additional arguments for command execution

    @staticmethod
    def pause() -> "ControlCommand":
        return ControlCommand(ControlCommandType.PAUSE, {})

    @staticmethod
    def resume() -> "ControlCommand":
        return ControlCommand(ControlCommandType.RESUME, {})

    @staticmethod
    def single_step() -> "ControlCommand":
        return ControlCommand(ControlCommandType.SINGLE_STEP, {})

    @staticmethod
    def cancel_plan() -> "ControlCommand":
        return ControlCommand(ControlCommandType.CANCEL_PLAN, {})

    @staticmethod
    def set_goal(goal: str) -> "ControlCommand":
        return ControlCommand(ControlCommandType.SET_GOAL, {"goal": goal})

    @staticmethod
    def dump_state() -> "ControlCommand":
        return ControlCommand(ControlCommandType.DUMP_STATE, {})

