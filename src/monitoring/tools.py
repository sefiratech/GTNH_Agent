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
