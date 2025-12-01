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

