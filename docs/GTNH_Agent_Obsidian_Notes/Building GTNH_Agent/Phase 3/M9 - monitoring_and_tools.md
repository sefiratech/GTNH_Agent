# M9 – monitoring_and_tools

## Purpose
Provide full observability, structured logging, and a control surface for the GTNH Agent.  
This module ensures the agent cannot act as a black box and allows real‑time inspection of plans, actions, virtue scores, tech state, and system health.

## Role in Architecture
- Belongs to **Phase P3 — Agent Orchestration & Tooling**  
- Sits **on top of M8 (AgentLoop)**  
- Reads semantic and structural information from earlier modules  
- Emits events consumable by dashboards, CLIs, loggers, and dev tools

---

# 1. Responsibilities & Boundaries

## 1.1 What M9 Owns
- A unified **structured logging subsystem** (JSONL)
- A **monitoring event bus** for in‑process pub/sub:
  - Plan creation
  - Step execution
  - Tech state updates
  - Virtue scoring
  - Action results
  - Critic evaluations
  - System errors
- A **control surface** that can:
  - Pause/resume the agent
  - Single‑step the agent
  - Cancel the current plan
  - Set a new goal
  - Dump the agent’s internal state
- A **TUI dashboard** for live inspection

## 1.2 What M9 Does *Not* Own
- No LLM inference
- No planning logic
- No GTNH semantics
- No direct world manipulation or packet processing
- No skill logic

M9 observes and controls, it never *interferes* with domain logic.

---

# 2. Module Structure & Files

```
src/
  monitoring/
    __init__.py
    events.py         # Event & command definitions
    bus.py            # Unified event bus
    logger.py         # JSON-lines logger
    controller.py     # Agent control surface
    dashboard_tui.py  # Live TUI dashboard
```

This structure ensures clean separation of concerns:  
events → message passing → logging → control → UI.

---

# 3. Event Model

## 3.1 Event Types
Events must be typed and structured. Examples:

- `AGENT_PHASE_CHANGE`
- `PLAN_CREATED`
- `PLAN_STEP_EXECUTED`
- `ACTION_EXECUTED`
- `PLAN_FAILED`
- `TECH_STATE_UPDATED`
- `VIRTUE_SCORES`
- `CRITIC_RESULT`
- `SNAPSHOT`
- `CONTROL_COMMAND`
- `LOG`

Each event carries:
- timestamp  
- module name  
- event_type  
- human-readable message  
- structured `payload`  
- optional correlation_id (plan/episode context)

```python
#src/monitoring/events.py
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
from dataclasses import dataclass, asdict
from enum import Enum, auto
from typing import Any, Dict, Optional


# ============================================================
# Event Types
# ============================================================

class EventType(Enum):
    """Typed monitoring events emitted throughout the agent system."""

    # Agent lifecycle phases (Idle, Planning, Executing, Recovering)
    AGENT_PHASE_CHANGE = auto()

    # Planner & plan structure
    PLAN_CREATED = auto()
    PLAN_STEP_EXECUTED = auto()
    PLAN_FAILED = auto()

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
    module: str                # Source module string ("M8.agent_loop", "bot_core", etc.)
    event_type: EventType      # Enum describing the event class
    message: str               # Short human-readable description
    payload: Dict[str, Any]    # Structured data (plan, action, tech state, scores)
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

```


## 3.2 Control Commands
Commands are issued by dashboards, CLI tools, or automated scripts:

- `PAUSE`
- `RESUME`
- `SINGLE_STEP`
- `CANCEL_PLAN`
- `SET_GOAL`
- `DUMP_STATE`

They are handled by the **AgentController**.

```python
#src/monitoring/controller.py
"""
Controller for M9 – monitoring_and_tools.

This module defines AgentController, which listens for ControlCommand
instances on the monitoring EventBus and safely manipulates the AgentLoop.

Supported commands (ControlCommandType):
- PAUSE
- RESUME
- SINGLE_STEP
- CANCEL_PLAN
- SET_GOAL
- DUMP_STATE
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, Protocol

from .bus import EventBus
from .events import (
    ControlCommand,
    ControlCommandType,
    EventType,
    MonitoringEvent,
)
from .logger import log_event


# ============================================================
# Agent interface for control
# ============================================================

class AgentLoopControl(Protocol):
    """
    Structural protocol describing the minimum interface the controller
    expects from an AgentLoop-like object.

    Any concrete AgentLoop should implement these methods.
    """

    def step(self) -> None:
        """Execute one iteration of the agent loop."""

    def cancel_current_plan(self) -> None:
        """Cancel the currently active plan, if any."""

    def set_goal(self, goal: str) -> None:
        """Set a new high-level goal for the agent."""

    def debug_state(self) -> Dict[str, Any]:
        """
        Return a JSON-serializable debug snapshot of internal state:
        - current phase
        - current/last plan
        - last tech_state
        - any other useful metadata
        """


# ============================================================
# Agent Controller
# ============================================================

class AgentController:
    """
    Control surface for the AgentLoop.

    Listens for ControlCommands on the EventBus and:
    - pauses/resumes agent stepping
    - runs single-step iterations
    - cancels the current plan
    - sets new goals
    - emits state snapshots

    The main loop should call `maybe_step_agent()` instead of calling
    `agent.step()` directly, so control flags are respected.
    """

    def __init__(self, agent: AgentLoopControl, bus: EventBus) -> None:
        self._agent = agent
        self._bus = bus

        # Internal flags
        self._paused: bool = False
        self._single_step: bool = False

        # Subscribe to incoming control commands
        self._bus.subscribe_commands(self._handle_command)

    # --------------------------------------------------------
    # Command handling
    # --------------------------------------------------------

    def _handle_command(self, cmd: ControlCommand) -> None:
        """
        Callback invoked when a ControlCommand is published to the EventBus.
        Dispatches the command and logs the outcome as a monitoring event.
        """
        if cmd.cmd == ControlCommandType.PAUSE:
            self._paused = True
            self._single_step = False
            self._log_control("PAUSE", {"paused": True})

        elif cmd.cmd == ControlCommandType.RESUME:
            self._paused = False
            self._single_step = False
            self._log_control("RESUME", {"paused": False})

        elif cmd.cmd == ControlCommandType.SINGLE_STEP:
            # Execute exactly one step on the next maybe_step_agent() call
            self._single_step = True
            self._paused = False
            self._log_control("SINGLE_STEP", {"single_step": True})

        elif cmd.cmd == ControlCommandType.CANCEL_PLAN:
            self._agent.cancel_current_plan()
            self._log_control("CANCEL_PLAN", {})

        elif cmd.cmd == ControlCommandType.SET_GOAL:
            goal = cmd.args.get("goal", "")
            self._agent.set_goal(goal)
            self._log_control("SET_GOAL", {"goal": goal})

        elif cmd.cmd == ControlCommandType.DUMP_STATE:
            state = self._safe_debug_state()
            self._log_snapshot(state)

    # --------------------------------------------------------
    # Public stepping API
    # --------------------------------------------------------

    def maybe_step_agent(self) -> None:
        """
        Entry point for the main loop.

        Instead of calling agent.step() directly, the runtime should call this.
        This respects pause and single-step semantics:
        - If paused: do nothing.
        - If single_step: run one step, then re-pause.
        - Otherwise: run one step as usual.
        """
        if self._paused:
            return

        # Perform one agent loop step
        self._agent.step()

        # If we were in single-step mode, pause again after the step
        if self._single_step:
            self._paused = True
            self._single_step = False
            self._log_control("SINGLE_STEP_COMPLETED", {"paused": True})

    # --------------------------------------------------------
    # Logging helpers
    # --------------------------------------------------------

    def _log_control(self, cmd_name: str, payload: Dict[str, Any]) -> None:
        """
        Emit a CONTROL_COMMAND monitoring event describing a control action.
        """
        log_event(
            bus=self._bus,
            module="M9.controller",
            event_type=EventType.CONTROL_COMMAND,
            message=f"Control command: {cmd_name}",
            payload={"cmd": cmd_name, **payload},
            correlation_id=None,
        )

    def _log_snapshot(self, state: Dict[str, Any]) -> None:
        """
        Emit a SNAPSHOT monitoring event containing debug state.
        """
        evt = MonitoringEvent(
            ts=time.time(),
            module="M9.controller",
            event_type=EventType.SNAPSHOT,
            message="Agent state snapshot",
            payload={"state": state},
            correlation_id=None,
        )
        self._bus.publish(evt)

    def _safe_debug_state(self) -> Dict[str, Any]:
        """
        Call agent.debug_state() and ensure the result is JSON-serializable
        (or at least safely convertible to a dict).
        """
        try:
            state = self._agent.debug_state()
        except Exception as exc:  # pragma: no cover - defensive
            # Fall back to a minimal error snapshot
            return {
                "error": "debug_state_failed",
                "details": repr(exc),
            }

        # Best-effort normalization: dataclasses → dict, everything else passes through
        try:
            return asdict(state)  # type: ignore[arg-type]
        except TypeError:
            return state

```

---

# 4. Event Bus

A minimal, thread‑safe pub/sub mechanism:

- Subscribers receive **MonitoringEvent** objects  
- Command handlers receive **ControlCommand** objects  
- Used by:
  - TUI dashboard
  - File-based logger
  - M8 AgentLoop instrumentation
  - Dev scripts

This acts as the “nervous system” of the entire agent.

```python
#src/monitoring/bus.py
"""
Event bus for M9 – monitoring_and_tools.

Provides a minimal, thread-safe, in-process pub/sub mechanism:

- Subscribers receive MonitoringEvent objects.
- Command handlers receive ControlCommand objects.
- Used by:
    - TUI dashboard
    - File-based logger
    - AgentLoop (M8) instrumentation
    - Dev tools / scripts
"""

from __future__ import annotations

from threading import Lock
from typing import Callable, List

from .events import MonitoringEvent, ControlCommand


# ============================================================
# Type aliases
# ============================================================

SubscriberFn = Callable[[MonitoringEvent], None]
CommandHandlerFn = Callable[[ControlCommand], None]


# ============================================================
# Event Bus
# ============================================================

class EventBus:
    """
    Simple in-process event bus for monitoring events and control commands.

    Design goals:
    - Minimal: no external dependencies or IPC.
    - Thread-safe: subscribers list protected by a Lock.
    - Non-blocking-ish: each publish iterates over a snapshot of subscribers.
    """

    def __init__(self) -> None:
        # Registered monitoring event subscribers
        self._subscribers: List[SubscriberFn] = []
        # Registered control command handlers
        self._cmd_handlers: List[CommandHandlerFn] = []
        # Single lock guarding both lists
        self._lock = Lock()

    # --------------------------------------------------------
    # Subscription API: Monitoring Events
    # --------------------------------------------------------

    def subscribe(self, fn: SubscriberFn) -> None:
        """
        Register a subscriber to receive MonitoringEvent instances.

        Subscribers MUST NOT throw exceptions; if they do, it's on them.
        """
        with self._lock:
            self._subscribers.append(fn)

    def unsubscribe(self, fn: SubscriberFn) -> None:
        """
        Remove a previously registered MonitoringEvent subscriber.

        Safe to call even if `fn` is not present.
        """
        with self._lock:
            if fn in self._subscribers:
                self._subscribers.remove(fn)

    # --------------------------------------------------------
    # Subscription API: Control Commands
    # --------------------------------------------------------

    def subscribe_commands(self, fn: CommandHandlerFn) -> None:
        """
        Register a handler to receive ControlCommand instances.
        """
        with self._lock:
            self._cmd_handlers.append(fn)

    def unsubscribe_commands(self, fn: CommandHandlerFn) -> None:
        """
        Remove a previously registered ControlCommand handler.

        Safe to call even if `fn` is not present.
        """
        with self._lock:
            if fn in self._cmd_handlers:
                self._cmd_handlers.remove(fn)

    # --------------------------------------------------------
    # Publish API: Monitoring Events
    # --------------------------------------------------------

    def publish(self, event: MonitoringEvent) -> None:
        """
        Publish a MonitoringEvent to all subscribers.

        Takes a snapshot of subscribers under the lock, then iterates without
        holding the lock to avoid deadlocks if subscribers call back into the bus.
        """
        with self._lock:
            subscribers = list(self._subscribers)

        for fn in subscribers:
            try:
                fn(event)
            except Exception:
                # Deliberately swallow exceptions to avoid one bad subscriber
                # killing the event stream. If you want to log this, hook a
                # dedicated error subscriber.
                pass

    # --------------------------------------------------------
    # Publish API: Control Commands
    # --------------------------------------------------------

    def publish_command(self, cmd: ControlCommand) -> None:
        """
        Publish a ControlCommand to all registered command handlers.
        """
        with self._lock:
            handlers = list(self._cmd_handlers)

        for fn in handlers:
            try:
                fn(cmd)
            except Exception:
                # Same rationale as for events: bad handlers shouldn't break others.
                pass

    # --------------------------------------------------------
    # Utility
    # --------------------------------------------------------

    def clear(self) -> None:
        """
        Clear all subscribers and handlers.

        Mostly useful for tests; probably not what you want in production.
        """
        with self._lock:
            self._subscribers.clear()
            self._cmd_handlers.clear()


# ============================================================
# Optional: shared bus instance
# ============================================================

# Many runtimes will prefer to use a single, process-wide bus.
# You can import `default_bus` where DI is annoying, or ignore this
# and manage EventBus instances explicitly.
default_bus = EventBus()

```

---

# 5. Structured Logging

M9 provides a single, unified JSONL logger:

- Writes each MonitoringEvent as one JSON line
- Enforced schema ensures logs are machine‑readable
- Lives in `logs/monitoring/` or `logs/` (configurable)
- Automatically subscribes to event bus

Logged topics include:
- Plan metadata
- Critic verdicts
- Virtue scores
- Failures
- Tech state transitions
- Action results
- Timing info (timestamps, latency)

```python
#src/monitoring/logger.py
"""
Structured logging for M9 – monitoring_and_tools.

Provides:
- JsonFileLogger: subscribes to an EventBus and writes MonitoringEvents as JSONL.
- log_event: convenience helper for publishing MonitoringEvents via the EventBus.

Usage patterns:

    from pathlib import Path
    from monitoring.bus import EventBus
    from monitoring.logger import JsonFileLogger, log_event
    from monitoring.events import EventType

    bus = EventBus()
    logger = JsonFileLogger(Path("logs/monitoring/events.log"), bus)

    log_event(
        bus=bus,
        module="example.module",
        event_type=EventType.LOG,
        message="Something happened",
        payload={"foo": "bar"},
    )
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

from .bus import EventBus
from .events import EventType, MonitoringEvent


# ============================================================
# JSONL File Logger
# ============================================================

class JsonFileLogger:
    """
    JSON-lines logger for MonitoringEvent instances.

    - Subscribes to an EventBus and writes one JSON object per line.
    - Ensures UTF-8 encoding.
    - Ensures parent directory exists.
    """

    def __init__(self, path: Path, bus: EventBus) -> None:
        """
        Initialize the logger and subscribe to the event bus.

        Parameters
        ----------
        path:
            Path to the log file (e.g. logs/monitoring/events.log).
        bus:
            EventBus instance to subscribe to.
        """
        self._path = path
        self._ensure_parent_dir(path)
        # Open file in append mode
        self._file = path.open("a", encoding="utf-8")
        # Subscribe as a MonitoringEvent consumer
        bus.subscribe(self._on_event)

    @staticmethod
    def _ensure_parent_dir(path: Path) -> None:
        """
        Create parent directories for `path` if they don't exist.
        """
        parent = path.parent
        if not parent.exists():
            parent.mkdir(parents=True, exist_ok=True)

    def _on_event(self, event: MonitoringEvent) -> None:
        """
        Callback invoked for each MonitoringEvent published on the EventBus.
        Writes the event as a JSON object to the log file.
        """
        data = event.to_dict()
        line = json.dumps(data, ensure_ascii=False)
        try:
            self._file.write(line + "\n")
            self._file.flush()
        except Exception:
            # Logging must not crash the process.
            # If disk is full or handle is invalid, we silently drop.
            # You can add a fallback stderr logger here if desired.
            pass

    def close(self) -> None:
        """
        Close the underlying file handle.

        Should be called at graceful shutdown.
        """
        try:
            self._file.close()
        except Exception:
            pass


# ============================================================
# Convenience helper for emitting events
# ============================================================

def log_event(
    bus: EventBus,
    module: str,
    event_type: EventType,
    message: str,
    payload: Optional[Dict[str, Any]] = None,
    correlation_id: Optional[str] = None,
) -> None:
    """
    Convenience function to create and publish a MonitoringEvent.

    Intended usage in other modules:

        log_event(
            bus=self._bus,
            module="M8.agent_loop",
            event_type=EventType.PLAN_CREATED,
            message="New plan created",
            payload={"plan": plan_dict, "goal": self._goal},
            correlation_id=plan_id,
        )

    Parameters
    ----------
    bus:
        EventBus instance to publish the event to.
    module:
        String identifying the source module ("M8.agent_loop", "bot_core", "llm_stack").
    event_type:
        EventType enum member describing what kind of event this is.
    message:
        Short human-readable description.
    payload:
        Structured JSON-safe data attached to this event.
    correlation_id:
        Optional ID linking related events (per-episode, per-plan, etc.).
    """
    event = MonitoringEvent(
        ts=time.time(),
        module=module,
        event_type=event_type,
        message=message,
        payload=payload or {},
        correlation_id=correlation_id,
    )
    bus.publish(event)

```


---

# 6. Control Surface

`AgentController` wraps M8’s AgentLoop to provide safe external control.

It can:

### • Pause / Resume
Freezes the loop or resumes normal stepping.

### • Single-step execution
Executes exactly one AgentLoop step before pausing again.

### • Cancel plan
Drops the active plan and returns to planning state.

### • Set goal
Pushes a new high‑level objective into the planner loop.

### • Dump state
Outputs active plan, current phase, latest tech state, and internal metadata.

The control surface never calls agent internals directly; it uses
AgentLoop‑provided methods:
- `cancel_current_plan()`
- `set_goal()`
- `debug_state()`
- `step()` (wrapped by controller logic)


```python
#src/monitoring/controller.py
"""
Control surface for M9 – monitoring_and_tools.

AgentController wraps an AgentLoop-like object and exposes safe external
control via ControlCommand messages on the EventBus.

Supported commands (ControlCommandType):
- PAUSE          -> freeze stepping
- RESUME         -> unfreeze stepping
- SINGLE_STEP    -> run exactly one AgentLoop step, then pause again
- CANCEL_PLAN    -> drop the current plan
- SET_GOAL       -> update the agent's high-level goal
- DUMP_STATE     -> emit a debug snapshot as a SNAPSHOT event
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Protocol

from .bus import EventBus
from .events import (
    ControlCommand,
    ControlCommandType,
    EventType,
)
from .logger import log_event


# ============================================================
# Agent interface expected by the controller
# ============================================================

class AgentLoopControl(Protocol):
    """
    Minimal protocol describing what the controller expects from M8's AgentLoop.

    Any concrete agent loop that wants to be controlled by AgentController
    must implement these methods.
    """

    def step(self) -> None:
        """Execute one iteration of the agent loop."""

    def cancel_current_plan(self) -> None:
        """Cancel the currently active plan, if any."""

    def set_goal(self, goal: str) -> None:
        """Set a new high-level goal for the agent."""

    def debug_state(self) -> Dict[str, Any]:
        """
        Return a JSON-serializable snapshot of internal state.

        Recommended contents:
        - current phase
        - current/last plan
        - latest tech_state snapshot
        - env_profile_name / context_id
        - any other useful metadata
        """
        ...


# ============================================================
# Agent Controller
# ============================================================

class AgentController:
    """
    Control surface for the AgentLoop.

    - Listens for ControlCommand instances on the EventBus.
    - Provides pause/resume/single-step semantics for stepping.
    - Forwards CANCEL_PLAN and SET_GOAL into the agent.
    - Emits state snapshots as SNAPSHOT events when requested.

    The main runtime should call `maybe_step_agent()` instead of calling
    `agent.step()` directly, so pause/single-step flags are respected.
    """

    def __init__(self, agent: AgentLoopControl, bus: EventBus) -> None:
        self._agent = agent
        self._bus = bus

        # Internal control flags
        self._paused: bool = False
        self._single_step: bool = False

        # Subscribe to control commands
        self._bus.subscribe_commands(self._handle_command)

    # --------------------------------------------------------
    # Command handling
    # --------------------------------------------------------

    def _handle_command(self, cmd: ControlCommand) -> None:
        """
        Process an incoming ControlCommand from dashboards, CLIs, or scripts.
        """
        if cmd.cmd == ControlCommandType.PAUSE:
            self._paused = True
            self._single_step = False
            self._log_control("PAUSE", {"paused": True})

        elif cmd.cmd == ControlCommandType.RESUME:
            self._paused = False
            self._single_step = False
            self._log_control("RESUME", {"paused": False})

        elif cmd.cmd == ControlCommandType.SINGLE_STEP:
            # Next maybe_step_agent() call will run exactly one step
            self._single_step = True
            self._paused = False
            self._log_control("SINGLE_STEP", {"single_step": True})

        elif cmd.cmd == ControlCommandType.CANCEL_PLAN:
            self._agent.cancel_current_plan()
            self._log_control("CANCEL_PLAN", {})

        elif cmd.cmd == ControlCommandType.SET_GOAL:
            goal = cmd.args.get("goal", "")
            self._agent.set_goal(goal)
            self._log_control("SET_GOAL", {"goal": goal})

        elif cmd.cmd == ControlCommandType.DUMP_STATE:
            state = self._safe_debug_state()
            self._log_snapshot(state)

    # --------------------------------------------------------
    # Stepping API for the main loop
    # --------------------------------------------------------

    def maybe_step_agent(self) -> None:
        """
        Replacement for direct agent.step() calls.

        Behavior:
        - If paused: do nothing.
        - If single_step is set: run one step, then re-pause.
        - Otherwise: run one step as normal.
        """
        if self._paused:
            return

        # Run exactly one iteration of the agent loop
        self._agent.step()

        if self._single_step:
            # After one step, go back to PAUSE
            self._paused = True
            self._single_step = False
            self._log_control("SINGLE_STEP_COMPLETED", {"paused": True})

    # --------------------------------------------------------
    # Introspection helpers (optional)
    # --------------------------------------------------------

    @property
    def paused(self) -> bool:
        """Return whether the controller is currently paused."""
        return self._paused

    @property
    def single_step_pending(self) -> bool:
        """Return whether the next step will be a single-step execution."""
        return self._single_step

    # --------------------------------------------------------
    # Logging helpers
    # --------------------------------------------------------

    def _log_control(self, cmd_name: str, payload: Dict[str, Any]) -> None:
        """
        Emit a CONTROL_COMMAND monitoring event describing a control action.
        """
        log_event(
            bus=self._bus,
            module="M9.controller",
            event_type=EventType.CONTROL_COMMAND,
            message=f"Control command: {cmd_name}",
            payload={"cmd": cmd_name, **payload},
            correlation_id=None,
        )

    def _log_snapshot(self, state: Dict[str, Any]) -> None:
        """
        Emit a SNAPSHOT monitoring event containing debug state.
        """
        log_event(
            bus=self._bus,
            module="M9.controller",
            event_type=EventType.SNAPSHOT,
            message="Agent state snapshot",
            payload={"state": state},
            correlation_id=None,
        )

    def _safe_debug_state(self) -> Dict[str, Any]:
        """
        Call agent.debug_state() and do a best-effort normalization so
        it is JSON-serializable for logging.
        """
        try:
            state = self._agent.debug_state()
        except Exception as exc:  # pragma: no cover - defensive fallback
            return {
                "error": "debug_state_failed",
                "details": repr(exc),
            }

        # If it's a dataclass, convert it to a dict
        if is_dataclass(state):
            return asdict(state)

        # Otherwise assume it's already JSON-like
        return state

```



---

# 7. Dashboard (TUI)

A lightweight TUI (built with **rich**) subscribes to the event bus and renders:

### • Agent status
- Phase
- Current goal
- Last plan

### • Tech State
- Tier
- Missing unlocks
- Semantic traits

### • Virtue scores
- Per virtue
- Aggregated plan quality

### • Plan summary
- Step count
- Last executed step
- Failures

This requires no external services, no web server, and runs entirely offline.

```python
#src/monitoring/dashboard_tui.py
"""
TUI dashboard for M9 – monitoring_and_tools.

A lightweight terminal UI (using `rich`) that subscribes to the monitoring
EventBus and renders:

- Agent status:
    - Phase
    - Current goal
    - Last plan id/summary

- Tech State:
    - Tier
    - Missing unlocks
    - Semantic traits (as provided by payload)

- Virtue scores:
    - Per virtue
    - Aggregated quality (simple summary)

- Plan summary:
    - Step count
    - Last executed step index
    - Failure info (if any)

This runs entirely offline. No web server, no external services.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .bus import EventBus, default_bus
from .events import EventType, MonitoringEvent


# ============================================================
# TUI Dashboard
# ============================================================

class TuiDashboard:
    """
    Live terminal dashboard bound to a monitoring.EventBus.

    It consumes MonitoringEvents and keeps a small in-memory state
    representation, which is rendered periodically via rich.
    """

    def __init__(self, bus: EventBus) -> None:
        self._bus = bus
        self._console = Console()

        # Internal state snapshot for display
        self._state: Dict[str, Any] = {
            "agent_phase": "UNKNOWN",
            "goal": "",
            "plan_id": None,
            "plan_step_count": 0,
            "last_step_index": None,
            "last_failure": None,
            "tech_state": {
                "tier": None,
                "missing_unlocks": [],
                "traits": {},
            },
            "virtue_scores": {},       # {virtue_name: score}
            "virtue_summary": None,    # e.g. average / rating string
        }

        # Subscribe to events
        self._bus.subscribe(self._on_event)

    # --------------------------------------------------------
    # Event handler
    # --------------------------------------------------------

    def _on_event(self, event: MonitoringEvent) -> None:
        """
        Update dashboard state based on a MonitoringEvent.
        This should be cheap and non-blocking.
        """
        et = event.event_type

        # Agent phase changes
        if et == EventType.AGENT_PHASE_CHANGE:
            phase = event.payload.get("phase", "UNKNOWN")
            self._state["agent_phase"] = phase

        # A new plan has been created
        elif et == EventType.PLAN_CREATED:
            plan = event.payload.get("plan", {}) or {}
            goal = event.payload.get("goal", "")
            # Pull plan id if the planner provides one
            plan_id = plan.get("id") or event.correlation_id
            steps = plan.get("steps") or []
            self._state["goal"] = goal
            self._state["plan_id"] = plan_id
            self._state["plan_step_count"] = len(steps)
            self._state["last_step_index"] = None
            self._state["last_failure"] = None

        # Each step of a plan execution
        elif et == EventType.PLAN_STEP_EXECUTED:
            idx = event.payload.get("step_index")
            if idx is not None:
                self._state["last_step_index"] = idx

        # Plan failure information
        elif et == EventType.PLAN_FAILED:
            reason = event.payload.get("reason", "unknown")
            step_index = event.payload.get("step_index")
            self._state["last_failure"] = {
                "reason": reason,
                "step_index": step_index,
            }

        # Tech state updates
        elif et == EventType.TECH_STATE_UPDATED:
            tech_payload = event.payload.get("tech_state", {}) or {}
            tier = tech_payload.get("tier") or tech_payload.get("current_tier")
            missing = tech_payload.get("missing_unlocks") or []
            traits = tech_payload.get("traits") or {}
            self._state["tech_state"] = {
                "tier": tier,
                "missing_unlocks": missing,
                "traits": traits,
            }

        # Virtue scores from the virtue engine
        elif et == EventType.VIRTUE_SCORES:
            scores = event.payload.get("scores", {}) or {}
            self._state["virtue_scores"] = scores
            self._state["virtue_summary"] = self._summarize_virtues(scores)

        # Critic result could be used to enhance failure panel later
        elif et == EventType.CRITIC_RESULT:
            # Optional: stash critic summary if provided
            critic = event.payload.get("critic_result", {})
            if critic:
                # Only store lightweight summary to avoid clutter
                self._state["last_failure"] = self._state["last_failure"] or {}
                self._state["last_failure"]["critic_summary"] = critic.get(
                    "summary", ""
                )

    # --------------------------------------------------------
    # Rendering helpers
    # --------------------------------------------------------

    def _summarize_virtues(self, scores: Dict[str, Any]) -> Optional[str]:
        """
        Build a simple summary string for virtue scores.
        E.g. "avg=0.82" if numeric, otherwise None.
        """
        numeric_scores: List[float] = []
        for v in scores.values():
            try:
                numeric_scores.append(float(v))
            except (TypeError, ValueError):
                continue
        if not numeric_scores:
            return None
        avg = sum(numeric_scores) / len(numeric_scores)
        return f"avg={avg:.2f}"

    def _render_agent_panel(self) -> Panel:
        """
        Top-left: agent phase + current goal + plan id.
        """
        phase = self._state["agent_phase"]
        goal = self._state["goal"] or "<none>"
        plan_id = self._state["plan_id"] or "<none>"

        txt = Text()
        txt.append("Phase: ", style="bold")
        txt.append(f"{phase}\n")
        txt.append("Goal: ", style="bold")
        txt.append(f"{goal}\n")
        txt.append("Plan ID: ", style="bold")
        txt.append(f"{plan_id}\n")

        return Panel(txt, title="Agent Status", border_style="cyan")

    def _render_tech_panel(self) -> Panel:
        """
        Middle-left: tech tier + missing unlocks + traits.
        """
        tech = self._state["tech_state"]
        tier = tech.get("tier") or "<unknown>"
        missing = tech.get("missing_unlocks") or []
        traits = tech.get("traits") or {}

        table = Table.grid(pad_edge=False)
        table.add_column(justify="left")

        table.add_row(f"[bold]Tier:[/bold] {tier}")

        if missing:
            missing_str = ", ".join(str(m) for m in missing[:8])
            if len(missing) > 8:
                missing_str += ", …"
            table.add_row(f"[bold]Missing:[/bold] {missing_str}")
        else:
            table.add_row("[bold]Missing:[/bold] <none>")

        if traits:
            trait_str = ", ".join(f"{k}={v}" for k, v in list(traits.items())[:6])
            if len(traits) > 6:
                trait_str += ", …"
            table.add_row(f"[bold]Traits:[/bold] {trait_str}")
        else:
            table.add_row("[bold]Traits:[/bold] <none>")

        return Panel(table, title="Tech State", border_style="green")

    def _render_virtue_panel(self) -> Panel:
        """
        Middle-center: virtue scores per virtue + summary.
        """
        scores = self._state["virtue_scores"] or {}
        summary = self._state["virtue_summary"]

        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Virtue", style="bold", width=16)
        table.add_column("Score", justify="right")

        if scores:
            for name, score in scores.items():
                table.add_row(str(name), f"{score}")
        else:
            table.add_row("<none>", "-")

        footer_text = Text()
        if summary:
            footer_text.append(f"Summary: {summary}")
        else:
            footer_text.append("No virtue scores yet")

        panel = Panel(
            renderable=table,
            title="Virtue Scores",
            border_style="magenta",
        )
        # Wrap with another panel that carries the footer? Keep it simple:
        return Panel(
            panel,
            title="Virtue Scores",
            subtitle=footer_text,
            border_style="magenta",
        )

    def _render_plan_panel(self) -> Panel:
        """
        Middle-right: plan step count, last executed step, and failures.
        """
        step_count = self._state["plan_step_count"]
        last_idx = self._state["last_step_index"]
        failure = self._state["last_failure"]

        table = Table.grid()
        table.add_column(justify="left")

        table.add_row(f"[bold]Steps:[/bold] {step_count}")
        table.add_row(
            f"[bold]Last Step Index:[/bold] "
            f"{last_idx if last_idx is not None else '-'}"
        )

        if failure:
            reason = failure.get("reason", "unknown")
            s_idx = failure.get("step_index")
            critic_summary = failure.get("critic_summary")
            fail_lines = [f"[bold]Reason:[/bold] {reason}"]
            if s_idx is not None:
                fail_lines.append(f"[bold]At step:[/bold] {s_idx}")
            if critic_summary:
                fail_lines.append(f"[bold]Critic:[/bold] {critic_summary}")
            table.add_row("")
            table.add_row("[bold red]Failure:[/bold red]")
            for line in fail_lines:
                table.add_row(line)
        else:
            table.add_row("")
            table.add_row("[bold green]No failures recorded.[/bold green]")

        return Panel(table, title="Plan Summary", border_style="yellow")

    def _build_layout(self) -> Layout:
        """
        Construct the overall layout for the dashboard.
        """
        layout = Layout()

        # Overall layout: top bar + middle row
        layout.split(
            Layout(name="top", size=5),
            Layout(name="middle", ratio=1),
        )

        # Top: agent status
        layout["top"].update(self._render_agent_panel())

        # Middle row: tech | virtues | plan
        layout["middle"].split_row(
            Layout(name="tech"),
            Layout(name="virtues"),
            Layout(name="plan"),
        )
        layout["tech"].update(self._render_tech_panel())
        layout["virtues"].update(self._render_virtue_panel())
        layout["plan"].update(self._render_plan_panel())

        return layout

    # --------------------------------------------------------
    # Main loop
    # --------------------------------------------------------

    def run(self, refresh_per_second: float = 4.0) -> None:
        """
        Run the TUI event loop.

        This blocks the current thread. Use a separate thread if needed.
        """
        refresh_delay = 1.0 / max(refresh_per_second, 0.1)
        with Live(self._build_layout(), console=self._console, refresh_per_second=refresh_per_second) as live:
            while True:
                # Re-render using the latest state
                live.update(self._build_layout())
                time.sleep(refresh_delay)


# ============================================================
# Optional: CLI entrypoint
# ============================================================

def run_dashboard_with_default_bus() -> None:
    """
    Convenience function to spawn a dashboard bound to monitoring.bus.default_bus.

    This assumes the rest of the system is using `default_bus` for publishing events.
    """
    dashboard = TuiDashboard(default_bus)
    dashboard.run()


if __name__ == "__main__":
    # Running this file directly will attach to default_bus.
    # In a real system you will likely start the agent runtime and dashboard
    # in the same process wired to default_bus, or in separate processes with
    # some IPC wrapper. For M9 v1, in-process is enough.
    run_dashboard_with_default_bus()

```



---

# 8. Integration Points with Previous Modules

## 8.1 M8 – AgentLoop
M9 listens for:
- Phase changes
- Plan creation
- Step execution
- Critic results
- Virtue scoring
- Experience creation

## 8.2 M7 – Observation Encoding
M9 displays:
- Planner payload summaries  
- Step world-state transitions  
- Trace structure  

## 8.3 M6 – BotCore
M9 captures:
- Raw observation metadata  
- Movement/action summaries  
- Success/failure states  

## 8.4 M3 & M4 – Semantics & Virtues
Displayed in the dashboard:
- TechState inference
- Virtue scoring details

```python
#src/monitoring/integration.py
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



```





---

# 9. Tools for Humans

M9 provides practical utilities:

### • Episode inspector
Loads last N episodes, showing plan, trace, critic result, and virtue scores.

### • LLM log viewer
Filters logs by:
- context_id
- episode_id
- planner vs critic vs scribe

### • State dumper
Exports a full JSON bundle of:
- active plan
- recent trace
- tech state
- semantic snapshot

### • Monitoring CLI (future extension)
Commands to issue control signals from the terminal.

```python
#src/monitoring/tools.py
"""
Human-facing utilities for M9 – monitoring_and_tools.

Provides:

- Episode inspector:
    - Load last N episodes from monitoring JSONL logs.
    - Summarize plan, trace, critic result, virtue scores.

- LLM log viewer:
    - Filter logs by:
        - context_id
        - episode_id
        - role (planner / critic / scribe / error_model, etc.)

- State dumper:
    - Export a JSON bundle of:
        - active plan
        - recent trace
        - tech state
        - semantic snapshot (if provided)

- Monitoring CLI:
    - issue control signals from the terminal using argparse:
        - pause
        - resume
        - single-step
        - cancel-plan
        - set-goal
        - dump-state
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .bus import EventBus, default_bus
from .events import (
    ControlCommand,
    ControlCommandType,
    EventType,
    MonitoringEvent,
)
from .logger import log_event


JsonDict = Dict[str, Any]


# ============================================================
# Episode inspector
# ============================================================

@dataclass
class EpisodeSummary:
    """
    Human-friendly summary of an episode reconstructed from monitoring logs.
    """
    episode_id: str
    context_id: Optional[str]
    phase_sequence: List[str]
    goal: Optional[str]
    plan_step_count: int
    last_step_index: Optional[int]
    failed: bool
    fail_reason: Optional[str]
    virtue_scores: JsonDict
    critic_result_summary: Optional[str]

    def to_dict(self) -> JsonDict:
        return asdict(self)


def _load_monitoring_events_from_jsonl(path: Path) -> Iterable[MonitoringEvent]:
    """
    Load MonitoringEvents from a JSONL file produced by JsonFileLogger.
    """
    if not path.exists():
        return []

    events: List[MonitoringEvent] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            try:
                etype = EventType[data["event_type"]]
            except Exception:
                continue

            evt = MonitoringEvent(
                ts=data.get("ts", 0.0),
                module=data.get("module", ""),
                event_type=etype,
                message=data.get("message", ""),
                payload=data.get("payload") or {},
                correlation_id=data.get("correlation_id"),
            )
            events.append(evt)
    return events


def _group_events_by_episode(events: Iterable[MonitoringEvent]) -> Dict[str, List[MonitoringEvent]]:
    """
    Group events by episode_id inferred from payload or correlation_id.

    Priority:
    - payload.episode_id
    - correlation_id
    """
    grouped: Dict[str, List[MonitoringEvent]] = {}
    for evt in events:
        payload = evt.payload or {}
        episode_id = payload.get("episode_id") or evt.correlation_id
        if not episode_id:
            # Ignore events with no episode context for this inspector
            continue
        grouped.setdefault(episode_id, []).append(evt)
    return grouped


def _build_episode_summary(episode_id: str, events: List[MonitoringEvent]) -> EpisodeSummary:
    """
    Derive an EpisodeSummary from a list of events associated with that episode_id.
    """
    phase_sequence: List[str] = []
    context_id: Optional[str] = None
    goal: Optional[str] = None
    plan_step_count = 0
    last_step_index: Optional[int] = None
    failed = False
    fail_reason: Optional[str] = None
    virtue_scores: JsonDict = {}
    critic_result_summary: Optional[str] = None

    for evt in events:
        payload = evt.payload or {}

        # track context_id if present
        cid = payload.get("context_id")
        if cid and context_id is None:
            context_id = cid

        if evt.event_type == EventType.AGENT_PHASE_CHANGE:
            phase = payload.get("phase")
            if phase:
                phase_sequence.append(phase)

        elif evt.event_type == EventType.PLAN_CREATED:
            goal = payload.get("goal", goal)
            plan_step_count = payload.get("step_count", plan_step_count)

        elif evt.event_type == EventType.PLAN_STEP_EXECUTED:
            idx = payload.get("step_index")
            if isinstance(idx, int):
                last_step_index = idx

        elif evt.event_type == EventType.PLAN_FAILED:
            failed = True
            fail_reason = payload.get("reason", fail_reason)

        elif evt.event_type == EventType.VIRTUE_SCORES:
            virtue_scores = payload.get("scores") or virtue_scores

        elif evt.event_type == EventType.CRITIC_RESULT:
            critic = payload.get("critic_result") or {}
            critic_result_summary = critic.get("summary") or critic_result_summary

    return EpisodeSummary(
        episode_id=episode_id,
        context_id=context_id,
        phase_sequence=phase_sequence,
        goal=goal,
        plan_step_count=plan_step_count,
        last_step_index=last_step_index,
        failed=failed,
        fail_reason=fail_reason,
        virtue_scores=virtue_scores,
        critic_result_summary=critic_result_summary,
    )


def load_last_n_episode_summaries(
    log_path: Path,
    last_n: int,
) -> List[EpisodeSummary]:
    """
    Load the last N episodes from a monitoring JSONL file and return summaries.

    Episodes are sorted by the timestamp of their last event.
    """
    events = list(_load_monitoring_events_from_jsonl(log_path))
    grouped = _group_events_by_episode(events)

    # Sort episodes by max ts
    def episode_last_ts(item: Tuple[str, List[MonitoringEvent]]) -> float:
        _, evts = item
        return max((e.ts for e in evts), default=0.0)

    sorted_items = sorted(grouped.items(), key=episode_last_ts, reverse=True)
    selected = sorted_items[:last_n]

    summaries: List[EpisodeSummary] = []
    for episode_id, evts in selected:
        summaries.append(_build_episode_summary(episode_id, evts))

    return summaries


# ============================================================
# LLM log viewer
# ============================================================

def iter_llm_logs(
    log_dir: Path,
    context_id: Optional[str] = None,
    episode_id: Optional[str] = None,
    role: Optional[str] = None,
) -> Iterable[JsonDict]:
    """
    Iterate over LLM log JSON files in `log_dir` and yield entries matching
    optional filters:

    - context_id
    - episode_id
    - role   (e.g. "planner", "critic", "scribe", "error_model")

    Assumes each file contains a single JSON object per file.
    """
    if not log_dir.exists() or not log_dir.is_dir():
        return []

    for path in sorted(log_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue

        if context_id is not None and data.get("context_id") != context_id:
            continue

        if episode_id is not None and data.get("episode_id") != episode_id:
            continue

        # role key name may differ; we try a few common fields
        if role is not None:
            log_role = (
                data.get("role")
                or data.get("llm_role")
                or data.get("kind")
            )
            if log_role != role:
                continue

        yield data


# ============================================================
# State dumper
# ============================================================

def build_state_bundle(
    agent_state: JsonDict,
    tech_state: Optional[JsonDict] = None,
    semantics_snapshot: Optional[JsonDict] = None,
) -> JsonDict:
    """
    Construct a JSON bundle with:

    - active plan (if present in agent_state)
    - recent trace (if present)
    - tech_state
    - semantic snapshot
    - raw debug_state for completeness

    agent_state is expected to come from agent.debug_state().
    """
    plan = agent_state.get("plan") or agent_state.get("current_plan")
    trace = agent_state.get("trace") or agent_state.get("recent_trace")

    bundle: JsonDict = {
        "meta": {
            "built_at": time.time(),
        },
        "agent_state": agent_state,
        "plan": plan,
        "trace": trace,
        "tech_state": tech_state or agent_state.get("tech_state") or {},
        "semantics": semantics_snapshot or {},
    }
    return bundle


def save_state_bundle(path: Path, bundle: JsonDict) -> None:
    """
    Persist a state bundle as pretty-printed JSON.
    """
    if not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(bundle, f, indent=2, sort_keys=True)


# ============================================================
# Monitoring CLI
# ============================================================

def _send_control_command(bus: EventBus, cmd: ControlCommand) -> None:
    """
    Publish a ControlCommand over the EventBus and log a corresponding event.
    """
    # Log a control event (optional; controller also logs its own)
    log_event(
        bus=bus,
        module="M9.cli",
        event_type=EventType.CONTROL_COMMAND,
        message=f"CLI control command: {cmd.cmd.name}",
        payload={"cmd": cmd.cmd.name, "args": cmd.args},
        correlation_id=None,
    )
    bus.publish_command(cmd)


def _cmd_pause(args: argparse.Namespace) -> None:
    _send_control_command(default_bus, ControlCommand(cmd=ControlCommandType.PAUSE, args={}))


def _cmd_resume(args: argparse.Namespace) -> None:
    _send_control_command(default_bus, ControlCommand(cmd=ControlCommandType.RESUME, args={}))


def _cmd_single_step(args: argparse.Namespace) -> None:
    _send_control_command(default_bus, ControlCommand(cmd=ControlCommandType.SINGLE_STEP, args={}))


def _cmd_cancel_plan(args: argparse.Namespace) -> None:
    _send_control_command(default_bus, ControlCommand(cmd=ControlCommandType.CANCEL_PLAN, args={}))


def _cmd_set_goal(args: argparse.Namespace) -> None:
    goal = args.goal or ""
    _send_control_command(
        default_bus,
        ControlCommand(cmd=ControlCommandType.SET_GOAL, args={"goal": goal}),
    )


def _cmd_dump_state(args: argparse.Namespace) -> None:
    """
    Send DUMP_STATE and briefly listen to SNAPSHOT events to print them.

    This is a simplistic approach: it assumes we are running in the same
    process as the agent, with a controller wired to default_bus.
    """
    snapshots: List[MonitoringEvent] = []

    def collector(evt: MonitoringEvent) -> None:
        if evt.event_type == EventType.SNAPSHOT and evt.module == "M9.controller":
            snapshots.append(evt)

    default_bus.subscribe(collector)
    try:
        _send_control_command(
            default_bus,
            ControlCommand(cmd=ControlCommandType.DUMP_STATE, args={}),
        )
        # Wait a short time for the snapshot to arrive
        time.sleep(0.5)
    finally:
        default_bus.unsubscribe(collector)

    if not snapshots:
        print("No state snapshot received.", file=sys.stderr)
        return

    # Use the latest snapshot
    evt = snapshots[-1]
    state = evt.payload.get("state") or {}
    json.dump(state, sys.stdout, indent=2, sort_keys=True)
    print()


def _cmd_inspect_episodes(args: argparse.Namespace) -> None:
    log_path = Path(args.log_path)
    summaries = load_last_n_episode_summaries(log_path, last_n=args.n)
    out = [s.to_dict() for s in summaries]
    json.dump(out, sys.stdout, indent=2, sort_keys=True)
    print()


def _cmd_view_llm_logs(args: argparse.Namespace) -> None:
    log_dir = Path(args.log_dir)
    entries = list(
        iter_llm_logs(
            log_dir=log_dir,
            context_id=args.context_id,
            episode_id=args.episode_id,
            role=args.role,
        )
    )
    json.dump(entries, sys.stdout, indent=2, sort_keys=True)
    print()


def build_arg_parser() -> argparse.ArgumentParser:
    """
    Build the monitoring CLI argument parser.
    """
    parser = argparse.ArgumentParser(
        prog="gtnh-monitor",
        description="Monitoring CLI for GTNH_Agent (M9 tools for humans).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # Control commands
    p_pause = sub.add_parser("pause", help="Pause the agent loop.")
    p_pause.set_defaults(func=_cmd_pause)

    p_resume = sub.add_parser("resume", help="Resume the agent loop.")
    p_resume.set_defaults(func=_cmd_resume)

    p_step = sub.add_parser("single-step", help="Run a single agent loop step.")
    p_step.set_defaults(func=_cmd_single_step)

    p_cancel = sub.add_parser("cancel-plan", help="Cancel the current plan.")
    p_cancel.set_defaults(func=_cmd_cancel_plan)

    p_goal = sub.add_parser("set-goal", help="Set a new high-level goal.")
    p_goal.add_argument("goal", type=str, help="Goal string.")
    p_goal.set_defaults(func=_cmd_set_goal)

    p_dump = sub.add_parser("dump-state", help="Request and print a state snapshot.")
    p_dump.set_defaults(func=_cmd_dump_state)

    # Episode inspector
    p_epi = sub.add_parser("inspect-episodes", help="Inspect last N episodes from monitoring logs.")
    p_epi.add_argument(
        "--log-path",
        type=str,
        default="logs/monitoring/events.log",
        help="Path to monitoring JSONL log file.",
    )
    p_epi.add_argument(
        "-n",
        type=int,
        default=5,
        help="Number of recent episodes to show.",
    )
    p_epi.set_defaults(func=_cmd_inspect_episodes)

    # LLM log viewer
    p_llm = sub.add_parser("view-llm", help="View filtered LLM logs from logs/llm.")
    p_llm.add_argument(
        "--log-dir",
        type=str,
        default="logs/llm",
        help="Directory containing LLM JSON logs.",
    )
    p_llm.add_argument(
        "--context-id",
        type=str,
        default=None,
        help="Filter by context_id.",
    )
    p_llm.add_argument(
        "--episode-id",
        type=str,
        default=None,
        help="Filter by episode_id.",
    )
    p_llm.add_argument(
        "--role",
        type=str,
        default=None,
        help="Filter by role (planner, critic, scribe, error_model...).",
    )
    p_llm.set_defaults(func=_cmd_view_llm_logs)

    return parser


def main(argv: Optional[List[str]] = None) -> None:
    """
    Entry point for the monitoring CLI.

    Example usage:

        python -m monitoring.tools pause
        python -m monitoring.tools resume
        python -m monitoring.tools single-step
        python -m monitoring.tools set-goal "Automate coke ovens"
        python -m monitoring.tools dump-state
        python -m monitoring.tools inspect-episodes -n 3
        python -m monitoring.tools view-llm --role planner
    """
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    func = getattr(args, "func", None)
    if func is None:
        parser.print_help()
        sys.exit(1)
    func(args)


if __name__ == "__main__":
    main()

```

---

# 10. Testing Strategy

### 10.1 Event Bus
- Publish/subscribe behavior
- Thread safety
- Ordering guarantees

### 10.2 Logger
- JSON structure validity
- Correct field encoding
- Flush behavior

### 10.3 Controller
- pause/resume
- single-step correctness
- plan cancellation
- goal setting

### 10.4 TUI Render Smoke Tests
- Template builds
- Layout does not crash
- Patches structured state correctly

No Minecraft or LLMs required for M9’s test suite.

```python
#tests/test_monitoring_event_bus.py
"""
Tests for monitoring.bus.EventBus

Covers:
- Publish/subscribe behavior
- Unsubscribe behavior
- Ordering guarantees
- Basic thread-safety smoke check
"""

from __future__ import annotations

import threading
import time
from typing import List

from monitoring.bus import EventBus
from monitoring.events import MonitoringEvent, EventType


def make_event(ts: float, module: str = "test", msg: str = "msg") -> MonitoringEvent:
    return MonitoringEvent(
        ts=ts,
        module=module,
        event_type=EventType.LOG,
        message=msg,
        payload={},
        correlation_id=None,
    )


def test_event_bus_publish_subscribe_basic():
    bus = EventBus()
    received: List[MonitoringEvent] = []

    def subscriber(evt: MonitoringEvent) -> None:
        received.append(evt)

    bus.subscribe(subscriber)

    e1 = make_event(1.0, msg="first")
    e2 = make_event(2.0, msg="second")

    bus.publish(e1)
    bus.publish(e2)

    assert len(received) == 2
    assert received[0].message == "first"
    assert received[1].message == "second"


def test_event_bus_unsubscribe():
    bus = EventBus()
    received: List[MonitoringEvent] = []

    def subscriber(evt: MonitoringEvent) -> None:
        received.append(evt)

    bus.subscribe(subscriber)
    bus.unsubscribe(subscriber)

    bus.publish(make_event(1.0))

    # Should not receive anything after unsubscribe
    assert received == []


def test_event_bus_ordering_guarantee():
    bus = EventBus()
    seen: List[int] = []

    def subscriber(evt: MonitoringEvent) -> None:
        seen.append(int(evt.ts))

    bus.subscribe(subscriber)

    # Publish in ascending order
    for ts in [1, 2, 3, 4, 5]:
        bus.publish(make_event(float(ts)))

    assert seen == [1, 2, 3, 4, 5]


def test_event_bus_thread_safety_smoke():
    """
    Smoke test: multiple threads publishing simultaneously should not crash
    and subscribers should receive the correct number of events.
    """
    bus = EventBus()
    count = 100

    received: List[MonitoringEvent] = []
    lock = threading.Lock()

    def subscriber(evt: MonitoringEvent) -> None:
        with lock:
            received.append(evt)

    bus.subscribe(subscriber)

    def publisher_thread(start: int) -> None:
        for i in range(start, start + count):
            bus.publish(make_event(float(i)))

    threads = [
        threading.Thread(target=publisher_thread, args=(0,)),
        threading.Thread(target=publisher_thread, args=(1000,)),
    ]

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # We expect 2 * count events
    assert len(received) == 2 * count


```


```python
#tests/test_monitoring_logger.py
"""
Tests for monitoring.logger.JsonFileLogger and log_event.

Covers:
- JSON structure validity
- Correct field encoding
- Flush behavior (file actually gets data)
"""

from __future__ import annotations

import json
from pathlib import Path

from monitoring.bus import EventBus
from monitoring.logger import JsonFileLogger, log_event
from monitoring.events import EventType


def test_json_file_logger_writes_valid_json(tmp_path: Path):
    bus = EventBus()
    log_path = tmp_path / "events.log"

    logger = JsonFileLogger(log_path, bus)

    # Emit one event
    log_event(
        bus=bus,
        module="test.module",
        event_type=EventType.LOG,
        message="hello world",
        payload={"a": 1, "b": "x"},
        correlation_id="episode-123",
    )

    # Explicit close to ensure file handle is flushed
    logger.close()

    content = log_path.read_text(encoding="utf-8").strip()
    lines = content.splitlines()
    assert len(lines) == 1

    data = json.loads(lines[0])

    assert data["module"] == "test.module"
    assert data["event_type"] == "LOG"
    assert data["message"] == "hello world"
    assert data["payload"]["a"] == 1
    assert data["payload"]["b"] == "x"
    assert data["correlation_id"] == "episode-123"
    assert isinstance(data["ts"], (int, float))


def test_logger_parent_dir_created(tmp_path: Path):
    # Create nested path that doesn't exist initially
    log_dir = tmp_path / "nested" / "logs"
    log_path = log_dir / "events.log"

    bus = EventBus()
    logger = JsonFileLogger(log_path, bus)

    # Emit something
    log_event(
        bus=bus,
        module="test.module",
        event_type=EventType.LOG,
        message="hello",
        payload={},
    )
    logger.close()

    assert log_path.exists()
    content = log_path.read_text(encoding="utf-8").strip()
    assert content  # not empty

```


```python
#tests/test_monitoring_controller.py
"""
Tests for monitoring.controller.AgentController.

Covers:
- pause/resume semantics
- single-step correctness
- plan cancellation
- goal setting
- DUMP_STATE wiring (basic)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from monitoring.bus import EventBus
from monitoring.controller import AgentController
from monitoring.events import (
    ControlCommand,
    ControlCommandType,
    EventType,
    MonitoringEvent,
)


@dataclass
class FakeAgent:
    """
    Simple fake AgentLoop implementing the AgentLoopControl protocol.
    """
    calls: Dict[str, Any]

    def __init__(self) -> None:
        self.calls = {
            "step": 0,
            "cancel_current_plan": 0,
            "set_goal": [],
            "debug_state": 0,
        }

    def step(self) -> None:
        self.calls["step"] += 1

    def cancel_current_plan(self) -> None:
        self.calls["cancel_current_plan"] += 1

    def set_goal(self, goal: str) -> None:
        self.calls["set_goal"].append(goal)

    def debug_state(self) -> Dict[str, Any]:
        self.calls["debug_state"] += 1
        return {
            "phase": "TEST",
            "plan": {"id": "plan-xyz", "steps": []},
            "tech_state": {"tier": "LV"},
        }


def test_controller_pause_resume_and_single_step():
    bus = EventBus()
    agent = FakeAgent()
    controller = AgentController(agent, bus)

    # Initially not paused, not single step
    assert controller.paused is False

    # PAUSE -> maybe_step_agent should do nothing
    bus.publish_command(ControlCommand(cmd=ControlCommandType.PAUSE, args={}))
    assert controller.paused is True
    controller.maybe_step_agent()
    assert agent.calls["step"] == 0

    # RESUME -> normal stepping
    bus.publish_command(ControlCommand(cmd=ControlCommandType.RESUME, args={}))
    assert controller.paused is False
    controller.maybe_step_agent()
    assert agent.calls["step"] == 1

    # SINGLE_STEP -> one step, then back to paused
    bus.publish_command(ControlCommand(cmd=ControlCommandType.SINGLE_STEP, args={}))
    assert controller.paused is False
    controller.maybe_step_agent()
    assert agent.calls["step"] == 2
    # After single step, should be paused again
    assert controller.paused is True


def test_controller_cancel_plan_and_set_goal_and_dump_state():
    bus = EventBus()
    agent = FakeAgent()
    controller = AgentController(agent, bus)

    # Capture monitoring events for inspection
    received: List[MonitoringEvent] = []

    def subscriber(evt: MonitoringEvent) -> None:
        received.append(evt)

    bus.subscribe(subscriber)

    # CANCEL_PLAN
    bus.publish_command(ControlCommand(cmd=ControlCommandType.CANCEL_PLAN, args={}))
    assert agent.calls["cancel_current_plan"] == 1

    # SET_GOAL
    bus.publish_command(
        ControlCommand(cmd=ControlCommandType.SET_GOAL, args={"goal": "Automate LV base"})
    )
    assert agent.calls["set_goal"] == ["Automate LV base"]

    # DUMP_STATE
    bus.publish_command(ControlCommand(cmd=ControlCommandType.DUMP_STATE, args={}))
    assert agent.calls["debug_state"] == 1

    # There should be at least one SNAPSHOT event from controller
    snapshot_events = [e for e in received if e.event_type == EventType.SNAPSHOT]
    assert snapshot_events, "Expected at least one SNAPSHOT event"

    # And the payload should have a 'state'
    latest = snapshot_events[-1]
    assert "state" in latest.payload
    assert latest.payload["state"]["phase"] == "TEST"

```


```python
#tests/test_monitoring_dashboard_tui.py
"""
Smoke tests for monitoring.dashboard_tui.TuiDashboard.

Covers:
- Layout builds cleanly
- Event updates patch internal state
- Rendering functions do not crash
"""

from __future__ import annotations

from monitoring.bus import EventBus
from monitoring.dashboard_tui import TuiDashboard
from monitoring.events import EventType, MonitoringEvent


def make_event(event_type: EventType, payload: dict) -> MonitoringEvent:
    return MonitoringEvent(
        ts=0.0,
        module="test",
        event_type=event_type,
        message="",
        payload=payload,
        correlation_id=None,
    )


def test_dashboard_handles_basic_events_and_renders():
    bus = EventBus()
    dashboard = TuiDashboard(bus)

    # Phase change
    bus.publish(make_event(EventType.AGENT_PHASE_CHANGE, {"phase": "PLANNING"}))

    # Plan created
    bus.publish(
        make_event(
            EventType.PLAN_CREATED,
            {
                "goal": "Build coke ovens",
                "plan": {"id": "plan-1", "steps": [1, 2, 3]},
                "step_count": 3,
            },
        )
    )

    # Plan step executed
    bus.publish(
        make_event(
            EventType.PLAN_STEP_EXECUTED,
            {"step_index": 1, "step_spec": {}, "trace_step": {}},
        )
    )

    # Tech state update
    bus.publish(
        make_event(
            EventType.TECH_STATE_UPDATED,
            {
                "tech_state": {
                    "tier": "LV",
                    "missing_unlocks": ["Basic Steam Turbine"],
                    "traits": {"energy": "steam"},
                }
            },
        )
    )

    # Virtue scores
    bus.publish(
        make_event(
            EventType.VIRTUE_SCORES,
            {"scores": {"prudence": 0.9, "temperance": 0.7}},
        )
    )

    # Plan failure
    bus.publish(
        make_event(
            EventType.PLAN_FAILED,
            {"reason": "Out of steam", "step_index": 2},
        )
    )

    # Critic result (optional)
    bus.publish(
        make_event(
            EventType.CRITIC_RESULT,
            {"critic_result": {"summary": "Good plan, bad resources"}},
        )
    )

    # Now try building the layout; it should not throw.
    layout = dashboard._build_layout()  # type: ignore[attr-defined]
    assert layout is not None

```


---

# 11. Completion Criteria (Definition of Done)

M9 is considered complete when:

- **Structured JSON logging** exists and captures:
  - Plan creation
  - Action execution
  - Tech state
  - Virtue evaluations
  - Critic results
  - Failures

- **Event bus** is fully wired across:
  - M6 (BotCore)
  - M7 (Observation)
  - M8 (AgentLoop)
  - M4 (Virtues)
  - M2 (LLM stack logs)

- **Control surface** can:
  - pause/resume  
  - step  
  - set goal  
  - cancel plan  
  - dump state  

- **TUI dashboard** displays:
  - phase  
  - goal  
  - plan summary  
  - tech state  
  - virtue scores  

- Monitoring layer has zero GTNH-specific logic.

- All new tests pass.

M9 gives you a live HUD for GTNH Agent, making the entire system transparent, debuggable, and inspectable.
