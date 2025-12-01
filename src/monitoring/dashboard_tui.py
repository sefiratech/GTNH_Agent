# rich-based TUI dashboard
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

