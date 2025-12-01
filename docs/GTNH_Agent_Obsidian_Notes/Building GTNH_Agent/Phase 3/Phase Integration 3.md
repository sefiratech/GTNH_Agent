## High-level architecture: M8 ↔ M9 data flow

**Core pieces**

- **M8: `AgentLoopV1`**
    
    - Owns: episode loop, planning, executing, recovering, experience.
        
    - Emits events via `monitoring.integration` helpers:
        
        - `emit_agent_phase_change`
            
        - `emit_plan_created`
            
        - `emit_plan_step_executed`
            
        - `emit_plan_failed`
            
        - `emit_critic_result`
            
        - `emit_virtue_scores`
            
        - `emit_experience_recorded`
            
    - Consumes control via `AgentController.maybe_step_agent()` instead of calling `step()` directly.
        
- **M9: Monitoring & tools**
    
    - **`EventBus`** (`monitoring.bus`)
        
        - Central pub/sub.
            
        - Producers: M8 (AgentLoop), M6, M7, M3, M4, CLI tools.
            
        - Consumers: logger, TUI, tests, dev tools, controller.
            
    - **`JsonFileLogger`** (`monitoring.logger`)
        
        - Subscribes to bus, logs every `MonitoringEvent` as JSONL.
            
    - **`AgentController`** (`monitoring.controller`)
        
        - Subscribes to `ControlCommand`s.
            
        - Wraps AgentLoop with pause / resume / single-step / cancel plan / set goal / dump state.
            
    - **`TuiDashboard`** (`monitoring.dashboard_tui`)
        
        - Subscribes to `MonitoringEvent`s.
            
        - Renders phase, goal, plan summary, tech state, virtue scores.
            
    - **`tools.py`**
        
        - CLI for control (pause, resume, step, set-goal, dump-state).
            
        - Episode inspector & LLM log viewer over JSON/JSONL logs.
            

**Event flow (simplified)**

1. **M8 changes phase** → calls `emit_agent_phase_change(bus, phase, episode_id, context_id)`  
    → `EventBus.publish()`  
    → `JsonFileLogger` writes JSONL, `TuiDashboard` updates phase.
    
2. **M8 creates a plan** → `emit_plan_created(...)`  
    → bus → logger + TUI (goal, plan id, step count).
    
3. **Each plan step runs** → `emit_plan_step_executed(...)`  
    → bus → logger + TUI (last step index).
    
4. **Failure / critic / virtues / experience**  
    → same pattern via their helper functions → logger + TUI + episode inspector.
    
5. **Human control (CLI / tools / future UI)**  
    → send `ControlCommand` on bus → `AgentController` receives → calls:
    
    - `agent.cancel_current_plan()`
        
    - `agent.set_goal(...)`
        
    - `agent.debug_state()`
        
    - `agent.step()` (wrapped by `maybe_step_agent()`)
        
6. **DUMP_STATE**  
    → controller calls `agent.debug_state()` → emits `SNAPSHOT` event  
    → logger captures it, CLI can listen for it and print / save.
    

---

## Integration script (example runtime)

Below is a **complete Python module** that wires together:

- `EventBus`
    
- `JsonFileLogger`
    
- `AgentLoopV1` (from M8)
    
- `AgentController`
    
- `TuiDashboard` (in a background thread)
    
- A simple main loop that calls `controller.maybe_step_agent()`
    

Every line is commented so your future sleep-deprived self knows what the hell it does.

```python
"""
gtnh_runtime_main.py

Unified runtime wiring M8 (AgentLoop) and M9 (monitoring & tools).

This script shows:
- How the EventBus, Logger, AgentLoop, Controller, and TUI Dashboard
  fit together.
- How control commands affect the agent loop.
- Where monitoring events flow and how logs are produced.
"""

from __future__ import annotations  # allow forward type references in annotations

import threading                    # for running the TUI in a background thread
import time                         # for sleep in the main loop
from pathlib import Path            # for filesystem paths

from monitoring.bus import EventBus # central event bus type
from monitoring.bus import default_bus  # optional shared global bus instance
from monitoring.logger import JsonFileLogger  # JSONL file logger
from monitoring.controller import AgentController  # control surface wrapper
from monitoring.dashboard_tui import TuiDashboard  # terminal HUD

# You should already have this from M8:
# `AgentLoopV1` implements AgentLoopControl protocol:
#   - step()
#   - cancel_current_plan()
#   - set_goal(goal: str)
#   - debug_state() -> dict
from agent_loop.loop import AgentLoopV1  # your core agent loop implementation


def start_tui_in_background(bus: EventBus) -> threading.Thread:
    """
    Start the TuiDashboard in a separate daemon thread.

    The dashboard listens to MonitoringEvents on the given bus and renders
    a live HUD. Running it in a background thread avoids blocking the
    main agent loop.
    """
    # Create a TUI instance bound to the event bus
    dashboard = TuiDashboard(bus)          # TUI will subscribe to events from the bus

    # Define a thread target that simply runs the dashboard loop
    def _run() -> None:
        dashboard.run(refresh_per_second=4.0)  # 4 FPS is enough for monitoring

    # Create the thread object
    t = threading.Thread(target=_run, name="TuiDashboardThread")
    # Mark as daemon so it will not block process exit
    t.daemon = True
    # Start the TUI thread
    t.start()
    # Return the thread in case the caller wants to join or inspect it
    return t


def build_monitoring_stack(
    use_default_bus: bool = True,
    log_path: Path | None = None,
) -> tuple[EventBus, JsonFileLogger]:
    """
    Construct the monitoring stack used by the runtime:

    - Choose an EventBus (either shared default_bus or a fresh one).
    - Attach a JsonFileLogger that writes MonitoringEvents to JSONL.

    Returns:
        (bus, logger)
    """
    # Choose which bus instance to use (shared vs. local)
    if use_default_bus:
        bus = default_bus                 # reuse global bus used by tools CLI
    else:
        bus = EventBus()                  # create a private bus

    # Decide where to write monitoring logs
    if log_path is None:
        # Default location: logs/monitoring/events.log
        log_path = Path("logs") / "monitoring" / "events.log"

    # Create a JSON-lines logger subscribed to the bus
    logger = JsonFileLogger(log_path, bus)  # ensures parent dirs and open file

    # Return both bus and logger so caller can keep references
    return bus, logger


def build_agent_loop(bus: EventBus) -> AgentLoopV1:
    """
    Construct the AgentLoopV1 and wire it to monitoring.

    Assumes your AgentLoopV1 is already instrumented to:
    - Call integration helpers like emit_agent_phase_change(),
      emit_plan_created(), emit_plan_step_executed(), etc.
    - Use the provided EventBus instance for those emissions.
    """
    # Example: AgentLoopV1 might accept bus and env_profile_name as constructor args
    # Replace with your real constructor as needed.
    agent = AgentLoopV1(event_bus=bus)     # pass the monitoring bus into M8
    return agent


def run_agent_runtime() -> None:
    """
    Main entrypoint tying M8 and M9 together.

    Runtime responsibilities:
    - Build monitoring stack (bus + logger).
    - Build AgentLoopV1 bound to the bus.
    - Wrap the agent in AgentController for pause/step/goal/etc.
    - Start TUI dashboard in the background.
    - Enter a main loop that calls controller.maybe_step_agent().
    """
    # Build monitoring infrastructure
    bus, logger = build_monitoring_stack(  # create bus + JSONL logger
        use_default_bus=True,              # use shared bus for CLI compatibility
        log_path=None,                     # use default logs/monitoring/events.log
    )

    # Build agent loop (M8) wired to the same EventBus
    agent = build_agent_loop(bus)          # AgentLoopV1 uses this bus for events

    # Wrap agent loop in AgentController for safe external control
    controller = AgentController(agent=agent, bus=bus)  # listens to ControlCommand on bus

    # Start TUI dashboard in a background thread (optional but nice)
    tui_thread = start_tui_in_background(bus)  # daemon thread, no need to join

    # Main loop: drive the agent via controller.maybe_step_agent()
    try:
        while True:
            # Let the controller decide whether to step the agent:
            # - If paused: no-op.
            # - If single-step pending: run one step then pause again.
            # - Otherwise: run one step as usual.
            controller.maybe_step_agent()

            # Sleep a bit to avoid spinning the CPU at 100%.
            # The actual cadence depends on how reactive you want the bot.
            time.sleep(0.1)
    except KeyboardInterrupt:
        # Handle Ctrl+C to shut down cleanly
        print("Shutting down agent runtime...")

    finally:
        # Close the logger file handle to flush remaining events
        logger.close()
        # Optionally: join the TUI thread if you want a clean shutdown
        # (it's a daemon thread, so it's not strictly required)
        # tui_thread.join(timeout=1.0)


if __name__ == "__main__":
    # If this module is executed directly, start the runtime.
    run_agent_runtime()


```


```python
# path: src/runtime/__init__.py

"""
Runtime wiring package for GTNH_Agent.

Holds entrypoints that stitch together:
- M8 (AgentLoop)
- M9 (monitoring_and_tools)
and any other orchestration/runtime glue.

Usage:
    python -m runtime.agent_runtime_main
"""


```

```python
# path: src/runtime/agent_runtime_main.py

"""
Unified runtime wiring M8 (AgentLoop) and M9 (monitoring & tools).

This script shows:
- How the EventBus, Logger, AgentLoop, Controller, and TUI Dashboard fit together.
- How control commands affect the agent loop.
- Where monitoring events flow and how logs are produced.

NOTE:
- This is a reference runtime.
- You may need to adjust the import for AgentLoopV1 based on where M8 actually lives.
"""

from __future__ import annotations  # allow forward type references in type hints

import threading                    # for running the TUI dashboard in a background thread
import time                         # for sleep timing in the main loop
from pathlib import Path            # for filesystem path handling

from monitoring.bus import EventBus         # central in-process event bus type
from monitoring.bus import default_bus      # optional shared global bus instance
from monitoring.logger import JsonFileLogger  # JSONL file logger for MonitoringEvents
from monitoring.controller import AgentController  # control surface wrapper around AgentLoop
from monitoring.dashboard_tui import TuiDashboard  # terminal HUD that subscribes to MonitoringEvents

# You should already have this from M8:
# `AgentLoopV1` implements the AgentLoopControl protocol:
#   - step()
#   - cancel_current_plan()
#   - set_goal(goal: str)
#   - debug_state() -> dict
from agent_loop.loop import AgentLoopV1     # core agent loop implementation (M8)


def start_tui_in_background(bus: EventBus) -> threading.Thread:
    """
    Start the TuiDashboard in a separate daemon thread.

    The dashboard listens to MonitoringEvents on the given bus and renders
    a live HUD. Running it in a background thread avoids blocking the
    main agent loop.
    """
    # Create a TUI instance bound to the event bus so it can receive MonitoringEvents
    dashboard = TuiDashboard(bus)

    # Define a function that simply runs the dashboard render loop
    def _run() -> None:
        # Run the dashboard event loop; 4 FPS is plenty for monitoring purposes
        dashboard.run(refresh_per_second=4.0)

    # Create a dedicated thread for the TUI
    t = threading.Thread(
        target=_run,                 # function to execute in the thread
        name="TuiDashboardThread",   # helpful name for debugging / profiling
    )
    # Mark the thread as daemon so it won't block process exit
    t.daemon = True
    # Start the TUI thread so it begins rendering
    t.start()
    # Return the thread in case the caller wants to track or join it
    return t


def build_monitoring_stack(
    use_default_bus: bool = True,
    log_path: Path | None = None,
) -> tuple[EventBus, JsonFileLogger]:
    """
    Construct the monitoring stack used by the runtime.

    Responsibilities:
    - Choose an EventBus (either the shared default_bus or a fresh instance).
    - Attach a JsonFileLogger that writes MonitoringEvents as JSONL.

    Returns:
        (bus, logger)
    """
    # Choose which EventBus instance to use
    if use_default_bus:
        # Use the shared global bus so tools (CLI, etc.) can talk to the same stream
        bus = default_bus
    else:
        # Create a private bus if you want isolated monitoring in this process only
        bus = EventBus()

    # Decide where to write monitoring logs on disk
    if log_path is None:
        # Default location: logs/monitoring/events.log (created if needed)
        log_path = Path("logs") / "monitoring" / "events.log"

    # Create a JSON-lines logger subscribed to the bus
    logger = JsonFileLogger(
        path=log_path,  # file where MonitoringEvents will be appended as JSONL
        bus=bus,        # EventBus instance to subscribe to
    )

    # Return both the bus and the logger so the caller can hold references
    return bus, logger


def build_agent_loop(bus: EventBus) -> AgentLoopV1:
    """
    Construct the AgentLoopV1 and wire it to the monitoring EventBus.

    Assumptions:
    - AgentLoopV1 is already instrumented to call integration helpers like:
        - emit_agent_phase_change()
        - emit_plan_created()
        - emit_plan_step_executed()
        - emit_plan_failed()
        - emit_critic_result()
        - emit_virtue_scores()
        - emit_experience_recorded()
      using the EventBus passed here.

    You may need to adjust this function if your AgentLoopV1 signature differs.
    """
    # Example constructor: AgentLoopV1(event_bus=bus, ...)
    agent = AgentLoopV1(
        event_bus=bus,  # pass the monitoring bus so M8 can emit events
    )
    # Return the fully constructed agent loop
    return agent


def run_agent_runtime() -> None:
    """
    Main entrypoint tying M8 and M9 together.

    Runtime responsibilities:
    - Build the monitoring stack (EventBus + JsonFileLogger).
    - Build AgentLoopV1 bound to that bus.
    - Wrap the agent in AgentController for pause/step/goal/etc.
    - Start TUI dashboard in the background.
    - Enter a main loop that calls controller.maybe_step_agent().
    """
    # Build monitoring infrastructure: event bus + JSONL file logger
    bus, logger = build_monitoring_stack(
        use_default_bus=True,  # use shared bus so tools/CLI can send commands
        log_path=None,         # None -> default path logs/monitoring/events.log
    )

    # Build the agent loop (M8) wired to the same EventBus
    agent = build_agent_loop(bus)

    # Wrap the agent loop in an AgentController to handle control commands
    controller = AgentController(
        agent=agent,  # the underlying AgentLoopV1 instance
        bus=bus,      # bus where ControlCommands and MonitoringEvents flow
    )

    # Start the TUI dashboard in a background daemon thread
    tui_thread = start_tui_in_background(bus)

    # Main loop: drive the agent via controller.maybe_step_agent()
    try:
        while True:
            # Let the controller decide whether to step the agent:
            # - If paused: no-op.
            # - If single_step_pending: run one step then re-pause.
            # - Otherwise: run one step as normal.
            controller.maybe_step_agent()

            # Sleep briefly to avoid pegging a CPU core at 100%
            # and to give IO/logging a chance to breathe.
            time.sleep(0.1)
    except KeyboardInterrupt:
        # Handle Ctrl+C for graceful shutdown in a dev environment
        print("Shutting down GTNH agent runtime...")

    finally:
        # Close the JSONL logger so buffered events are flushed to disk
        logger.close()
        # Optionally, you could join the TUI thread for a tidier shutdown:
        # tui_thread.join(timeout=1.0)


if __name__ == "__main__":
    # If this module is executed directly (e.g. python -m runtime.agent_runtime_main),
    # start the unified runtime.
    run_agent_runtime()

```


---


## Testing, logging, and failure points

**Testing hooks**

- M9 tests already cover:
    
    - `EventBus` publish/subscribe, ordering, thread-safety.
        
    - `JsonFileLogger` JSON correctness & file flush.
        
    - `AgentController` pause/resume/single-step/cancel/goal/dump-state.
        
    - `TuiDashboard` smoke tests (layout builds, state updated by events).
        
- For M8 + M9 integration:
    
    - Add a small test that:
        
        - Creates a `EventBus` + `JsonFileLogger` + fake `AgentLoopControl`.
            
        - Wraps it in `AgentController`.
            
        - Publishes a few `ControlCommand`s and `MonitoringEvent`s.
            
        - Asserts the JSONL file contains coherent episodes.
            

**Logging behavior**

- **All monitoring goes through `log_event`**:
    
    - Producers: M8, M6, M7, M3, M4, CLI.
        
    - Consumers: `JsonFileLogger`, TUI, tests, dev scripts.
        
- Failure semantics:
    
    - If a subscriber crashes, EventBus swallows the exception so the whole process doesn’t die.
        
    - If disk is full or logger fails to write, `JsonFileLogger` drops the event silently (by design, to avoid cascading failures). If you want loud failure, you can add stderr logging.
        

**Potential failure points & mitigation**

- **AgentLoop exceptions**:
    
    - If `agent.step()` raises, `maybe_step_agent()` will propagate that up and blow the loop.
        
    - Recommended: wrap `agent.step()` in a try/except in your real runtime, emit a `PLAN_FAILED` or `LOG` event with subtype `"AGENT_STEP_EXCEPTION"`, then decide whether to continue or abort.
        
- **`debug_state()` lying or returning non-JSON-safe data**:
    
    - `_safe_debug_state()` in `AgentController` tries to normalize dataclasses → dict.
        
    - If it explodes, you still get a `"debug_state_failed"` snapshot payload.
        
- **TUI performance**:
    
    - If event volume is insane, the dashboard just reads the latest state and re-renders; worst case you get slightly stale but still sane UI.
        
    - It runs in its own thread, so it won’t block the agent.
        
- **Log volume / disk usage**:
    
    - JSONL grows forever.
        
    - You’ll want rotation later (simple “new file per day” or size-based rotation is enough). That can be bolted into `JsonFileLogger` without touching the rest of the architecture.


```python
# path: tests/test_runtime_integration.py

"""
Integration tests for M8 (AgentLoop) + M9 (monitoring & tools).

Covers:
- EventBus + JsonFileLogger end-to-end.
- AgentController driving a fake AgentLoopControl implementation.
- Emitting a small set of MonitoringEvents that form a coherent "episode".
- Verifying that the JSONL log contains consistent, structured data.

This does NOT require:
- Minecraft
- Real LLMs
- Real AgentLoopV1 implementation
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from monitoring.bus import EventBus
from monitoring.controller import AgentController
from monitoring.events import (
    ControlCommand,
    ControlCommandType,
    EventType,
    MonitoringEvent,
)
from monitoring.logger import JsonFileLogger, log_event


class FakeAgentLoop:
    """
    Minimal fake implementing the AgentLoopControl protocol.

    Behaviors:
    - `step()` increments a counter.
    - `cancel_current_plan()` toggles a flag.
    - `set_goal()` stores the last goal.
    - `debug_state()` returns a JSON-safe dict.
    """

    def __init__(self) -> None:
        self.step_count: int = 0
        self.cancelled: bool = False
        self.goals: List[str] = []

    def step(self) -> None:
        self.step_count += 1

    def cancel_current_plan(self) -> None:
        self.cancelled = True

    def set_goal(self, goal: str) -> None:
        self.goals.append(goal)

    def debug_state(self) -> Dict[str, Any]:
        # Return a JSON-safe diagnostic snapshot
        return {
            "phase": "EXECUTING",
            "current_plan": {"id": "plan-ep1", "steps": [1, 2, 3]},
            "step_count": self.step_count,
            "tech_state": {"tier": "LV"},
            "episode_id": "ep-1",
        }


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """
    Helper: read a JSONL file into a list of dicts.
    """
    if not path.exists():
        return []
    items: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            items.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return items


def test_m8_m9_integration_basic_episode(tmp_path: Path) -> None:
    """
    End-to-end integration test:

    - Build EventBus and JsonFileLogger.
    - Wrap a FakeAgentLoop with AgentController.
    - Emit events that represent a simple episode:
        - phase change
        - plan created
        - one step executed
        - virtue scores
        - critic result
    - Drive controller with a few ControlCommand messages.
    - Assert the JSONL log contains a coherent episode for episode_id "ep-1".
    """
    # Arrange: event bus + logger
    bus = EventBus()
    log_path = tmp_path / "events.log"
    logger = JsonFileLogger(log_path, bus)

    # Arrange: fake agent + controller
    agent = FakeAgentLoop()
    controller = AgentController(agent=agent, bus=bus)

    # Episode identifiers
    episode_id = "ep-1"
    context_id = "test-context-1"

    # Act: emit a phase change
    log_event(
        bus=bus,
        module="M8.agent_loop",
        event_type=EventType.AGENT_PHASE_CHANGE,
        message="Agent entering PLANNING",
        payload={"phase": "PLANNING", "episode_id": episode_id, "context_id": context_id},
        correlation_id=episode_id,
    )

    # Act: emit a plan created event
    plan = {"id": "plan-ep1", "steps": [{"idx": 0}, {"idx": 1}, {"idx": 2}]}
    log_event(
        bus=bus,
        module="M8.agent_loop",
        event_type=EventType.PLAN_CREATED,
        message="New plan created",
        payload={
            "plan": plan,
            "goal": "Automate LV steam",
            "step_count": 3,
            "episode_id": episode_id,
            "context_id": context_id,
        },
        correlation_id=episode_id,
    )

    # Act: simulate one executed step
    log_event(
        bus=bus,
        module="M8.agent_loop",
        event_type=EventType.PLAN_STEP_EXECUTED,
        message="Executed step 0",
        payload={
            "episode_id": episode_id,
            "context_id": context_id,
            "step_index": 0,
            "step_spec": {"idx": 0},
            "trace_step": {"status": "ok"},
        },
        correlation_id=episode_id,
    )

    # Act: virtue scores for this episode
    log_event(
        bus=bus,
        module="M4.virtues",
        event_type=EventType.VIRTUE_SCORES,
        message="Virtue scores computed",
        payload={
            "episode_id": episode_id,
            "context_id": context_id,
            "scores": {"prudence": 0.9, "temperance": 0.7},
        },
        correlation_id=episode_id,
    )

    # Act: critic result summary
    log_event(
        bus=bus,
        module="M8.agent_loop",
        event_type=EventType.CRITIC_RESULT,
        message="Critic evaluation completed",
        payload={
            "episode_id": episode_id,
            "context_id": context_id,
            "critic_result": {"summary": "Good plan, watch resources"},
        },
        correlation_id=episode_id,
    )

    # Act: basic control interactions
    bus.publish_command(ControlCommand(cmd=ControlCommandType.SET_GOAL, args={"goal": "Automate LV steam"}))
    bus.publish_command(ControlCommand(cmd=ControlCommandType.SINGLE_STEP, args={}))
    controller.maybe_step_agent()  # should perform one step and then pause again
    bus.publish_command(ControlCommand(cmd=ControlCommandType.DUMP_STATE, args={}))

    # Cleanup / flush
    logger.close()

    # Assert: agent received commands
    assert agent.goals == ["Automate LV steam"]
    assert agent.step_count == 1

    # Assert: log file exists and has multiple events
    events = _read_jsonl(log_path)
    assert len(events) >= 5, "Expected multiple monitoring events in JSONL log"

    # Filter down to just this episode_id
    ep_events = [e for e in events if e.get("payload", {}).get("episode_id") == episode_id]
    assert ep_events, "Expected events associated with episode_id ep-1"

    # Check that we have at least one of each important type
    types = {e["event_type"] for e in ep_events}
    assert "AGENT_PHASE_CHANGE" in types
    assert "PLAN_CREATED" in types
    assert "PLAN_STEP_EXECUTED" in types
    assert "VIRTUE_SCORES" in types
    assert "CRITIC_RESULT" in types

    # Check that correlation_id is consistent across the main episode events
    corr_ids = {e.get("correlation_id") for e in ep_events}
    assert corr_ids == {episode_id}, "Expected a single coherent correlation_id for the episode"


```


```python
# path: src/runtime/error_handling.py

"""
Error handling helpers for the GTNH agent runtime.

Implements the "Potential failure points & mitigation" suggestions:
- Wrap agent stepping in a guard that logs AGENT_STEP_EXCEPTION events.
- Keep the monitoring/logging behavior consistent with the rest of M9.

This does NOT change core semantics of AgentController; it wraps calls
from your runtime loop.
"""

from __future__ import annotations

from typing import Optional

from monitoring.bus import EventBus
from monitoring.events import EventType
from monitoring.logger import log_event
from monitoring.controller import AgentController


def safe_step_with_logging(
    controller: AgentController,
    bus: EventBus,
    episode_id: Optional[str] = None,
    context_id: Optional[str] = None,
) -> None:
    """
    Call controller.maybe_step_agent() inside a try/except block.

    If agent.step() (inside the controller) throws, we:
    - Emit a LOG event with subtype "AGENT_STEP_EXCEPTION".
    - Re-raise the exception so the runtime can decide whether to abort
      or continue.

    This keeps the monitoring layer informed without silently hiding failures.
    """
    try:
        # Normal stepping: AgentController will decide whether to run a step
        controller.maybe_step_agent()
    except Exception as exc:
        # Emit a structured monitoring event describing the failure
        log_event(
            bus=bus,
            module="runtime.safe_step",
            event_type=EventType.LOG,
            message="Agent step raised an exception",
            payload={
                "subtype": "AGENT_STEP_EXCEPTION",
                "episode_id": episode_id,
                "context_id": context_id,
                "exception_repr": repr(exc),
            },
            correlation_id=episode_id,
        )
        # Re-raise so the caller decides how fatal this is
        raise


```


```python
# path: src/runtime/agent_runtime_main.py

"""
Unified runtime wiring M8 (AgentLoop) and M9 (monitoring & tools).

This script shows:
- How the EventBus, Logger, AgentLoop, Controller, and TUI Dashboard fit together.
- How control commands affect the agent loop.
- Where monitoring events flow and how logs are produced.

NOTE:
- This is a reference runtime.
- You may need to adjust the import for AgentLoopV1 based on where M8 actually lives.
"""

from __future__ import annotations  # allow forward type references in type hints

import threading                    # for running the TUI dashboard in a background thread
import time                         # for sleep timing in the main loop
from pathlib import Path            # for filesystem path handling

from monitoring.bus import EventBus         # central in-process event bus type
from monitoring.bus import default_bus      # optional shared global bus instance
from monitoring.logger import JsonFileLogger  # JSONL file logger for MonitoringEvents
from monitoring.controller import AgentController  # control surface wrapper around AgentLoop
from monitoring.dashboard_tui import TuiDashboard  # terminal HUD that subscribes to MonitoringEvents

# Runtime-level error handling helper that logs AGENT_STEP_EXCEPTION events
from runtime.error_handling import safe_step_with_logging

# You should already have this from M8:
# `AgentLoopV1` implements the AgentLoopControl protocol:
#   - step()
#   - cancel_current_plan()
#   - set_goal(goal: str)
#   - debug_state() -> dict
from agent_loop.loop import AgentLoopV1     # core agent loop implementation (M8)


def start_tui_in_background(bus: EventBus) -> threading.Thread:
    """
    Start the TuiDashboard in a separate daemon thread.

    The dashboard listens to MonitoringEvents on the given bus and renders
    a live HUD. Running it in a background thread avoids blocking the
    main agent loop.
    """
    # Create a TUI instance bound to the event bus so it can receive MonitoringEvents
    dashboard = TuiDashboard(bus)

    # Define a function that simply runs the dashboard render loop
    def _run() -> None:
        # Run the dashboard event loop; 4 FPS is plenty for monitoring purposes
        dashboard.run(refresh_per_second=4.0)

    # Create a dedicated thread for the TUI
    t = threading.Thread(
        target=_run,                 # function to execute in the thread
        name="TuiDashboardThread",   # helpful name for debugging / profiling
    )
    # Mark the thread as daemon so it won't block process exit
    t.daemon = True
    # Start the TUI thread so it begins rendering
    t.start()
    # Return the thread in case the caller wants to track or join it
    return t


def build_monitoring_stack(
    use_default_bus: bool = True,
    log_path: Path | None = None,
) -> tuple[EventBus, JsonFileLogger]:
    """
    Construct the monitoring stack used by the runtime.

    Responsibilities:
    - Choose an EventBus (either the shared default_bus or a fresh instance).
    - Attach a JsonFileLogger that writes MonitoringEvents as JSONL.

    Returns:
        (bus, logger)
    """
    # Choose which EventBus instance to use
    if use_default_bus:
        # Use the shared global bus so tools (CLI, etc.) can talk to the same stream
        bus = default_bus
    else:
        # Create a private bus if you want isolated monitoring in this process only
        bus = EventBus()

    # Decide where to write monitoring logs on disk
    if log_path is None:
        # Default location: logs/monitoring/events.log (created if needed)
        log_path = Path("logs") / "monitoring" / "events.log"

    # Create a JSON-lines logger subscribed to the bus
    logger = JsonFileLogger(
        path=log_path,  # file where MonitoringEvents will be appended as JSONL
        bus=bus,        # EventBus instance to subscribe to
    )

    # Return both the bus and the logger so the caller can hold references
    return bus, logger


def build_agent_loop(bus: EventBus) -> AgentLoopV1:
    """
    Construct the AgentLoopV1 and wire it to the monitoring EventBus.

    Assumptions:
    - AgentLoopV1 is already instrumented to call integration helpers like:
        - emit_agent_phase_change()
        - emit_plan_created()
        - emit_plan_step_executed()
        - emit_plan_failed()
        - emit_critic_result()
        - emit_virtue_scores()
        - emit_experience_recorded()
      using the EventBus passed here.

    You may need to adjust this function if your AgentLoopV1 signature differs.
    """
    # Example constructor: AgentLoopV1(event_bus=bus, ...)
    agent = AgentLoopV1(
        event_bus=bus,  # pass the monitoring bus so M8 can emit events
    )
    # Return the fully constructed agent loop
    return agent


def run_agent_runtime() -> None:
    """
    Main entrypoint tying M8 and M9 together.

    Runtime responsibilities:
    - Build the monitoring stack (EventBus + JsonFileLogger).
    - Build AgentLoopV1 bound to that bus.
    - Wrap the agent in AgentController for pause/step/goal/etc.
    - Start TUI dashboard in the background.
    - Enter a main loop that calls controller.maybe_step_agent() via
      safe_step_with_logging(), so exceptions are logged as events.
    """
    # Build monitoring infrastructure: event bus + JSONL file logger
    bus, logger = build_monitoring_stack(
        use_default_bus=True,  # use shared bus so tools/CLI can send commands
        log_path=None,         # None -> default path logs/monitoring/events.log
    )

    # Build the agent loop (M8) wired to the same EventBus
    agent = build_agent_loop(bus)

    # Wrap the agent loop in an AgentController to handle control commands
    controller = AgentController(
        agent=agent,  # the underlying AgentLoopV1 instance
        bus=bus,      # bus where ControlCommands and MonitoringEvents flow
    )

    # Start the TUI dashboard in a background daemon thread
    tui_thread = start_tui_in_background(bus)

    # Main loop: drive the agent via controller, with error logging
    try:
        while True:
            # Use the runtime helper so any exceptions from agent.step()
            # (triggered inside controller.maybe_step_agent) are:
            # - logged as AGENT_STEP_EXCEPTION events
            # - re-raised for the runtime to decide what to do
            safe_step_with_logging(
                controller=controller,
                bus=bus,
                episode_id=None,   # optionally thread real episode IDs from AgentLoop
                context_id=None,   # optionally thread context IDs from env/profile
            )

            # Sleep briefly to avoid pegging a CPU core at 100%
            # and to give IO/logging a chance to breathe.
            time.sleep(0.1)
    except KeyboardInterrupt:
        # Handle Ctrl+C for graceful shutdown in a dev environment
        print("Shutting down GTNH agent runtime...")
    finally:
        # Close the JSONL logger so buffered events are flushed to disk
        logger.close()
        # Optionally, you could join the TUI thread for a tidier shutdown:
        # tui_thread.join(timeout=1.0)


if __name__ == "__main__":
    # If this module is executed directly (e.g. python -m runtime.agent_runtime_main),
    # start the unified runtime.
    run_agent_runtime()


```


---

You now have a clear story:

- **M8 thinks, acts, and emits events.**
    
- **M9 watches, logs, visualizes, and controls.**
    

The robot is no longer a mysterious gremlin operating in the dark. It’s a _well-instrumented_ gremlin.