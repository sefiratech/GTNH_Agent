# Path: tools/agent_demo.py

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from agent.bootstrap import build_agent_runtime, Phase012BootstrapConfig
from agent.logging_config import configure_logging
from agent.loop import AgentLoop, AgentLoopConfig
from learning.buffer import ExperienceBuffer as ReplayBuffer
from spec.agent_loop import AgentGoal


class DemoCurriculum:
    """
    Minimal curriculum implementation for smoke testing.

    Responsibilities for Pass B:
      - Provide `next_goal(tech_state=..., experience_summary=...)`
      - Optionally provide `get_skill_view_for_goal(goal)`
      - Optionally accept `on_episode_complete(...)`

    This does NOT:
      - Read any curriculum YAML
      - Adapt based on experience_summary
      - Coordinate SkillPolicy / LearningManager
    """

    def __init__(self) -> None:
        self._counter = 0

    def next_goal(
        self,
        *,
        tech_state: Any,
        experience_summary: Optional[Dict[str, Any]] = None,
    ) -> AgentGoal:
        """
        Always returns a simple AgentGoal derived from the current tech_state.

        This satisfies the Pass B contract:
          AgentLoop → curriculum.next_goal(tech_state, experience_summary)
        without pulling in the full M11 stack.
        """
        self._counter += 1
        tier = getattr(tech_state, "active", None) or getattr(tech_state, "tier", "unknown")

        return AgentGoal(
            id=f"demo-{self._counter}",
            text=f"[DEMO] stabilize base at tech tier '{tier}'",
            phase="demo_p0",
            source="demo_curriculum",
        )

    def get_skill_view_for_goal(self, goal: AgentGoal) -> None:
        """
        For this smoke test, we don’t filter skills at all.
        Returning None tells AgentLoop there is no special SkillView.
        """
        return None

    def on_episode_complete(
        self,
        *,
        goal: AgentGoal,
        episode_result: Any,
    ) -> None:
        """
        Placeholder hook for compatibility with AgentLoop._maybe_notify_curriculum.
        No-op for the demo.
        """
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal AgentLoop smoke test (Pass B: data flow).")
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes to run.")
    parser.add_argument(
        "--profile",
        type=str,
        default="dev_local",
        help="Env profile name (from config/env.yaml).",
    )
    parser.add_argument(
        "--use-dummy-semantics",
        action="store_true",
        default=True,
        help="Use dummy semantics instead of full GTNH semantics.",
    )
    parser.add_argument(
        "--replay-path",
        type=Path,
        default=Path("data/experiences/replay.jsonl"),
        help="Path for the JSONL replay buffer used by ExperienceBuffer.",
    )
    args = parser.parse_args()

    # Logging
    configure_logging(logging.DEBUG)

    # Build base runtime (M0–M3, M6–M7 stack)
    bootstrap_cfg = Phase012BootstrapConfig(
        context_id="dev_dummy",
        logging_level=logging.DEBUG,
    )
    runtime = build_agent_runtime(
        profile=args.profile,
        use_dummy_semantics=args.use_dummy_semantics,
        bootstrap_config=bootstrap_cfg,
    )

    # M10 replay buffer: JSONL-backed ExperienceBuffer
    replay_buffer = ReplayBuffer(path=args.replay_path)

    # Minimal curriculum implementation for Pass B
    curriculum = DemoCurriculum()

    # AgentLoop config tuned for smoke testing
    loop_cfg = AgentLoopConfig(
        enable_critic=True,
        enable_retry_loop=False,
        store_experiences=True,   # store in-episode Experience objects
        max_planner_calls=1,
        max_skill_steps=8,
        fail_fast_on_invalid_plan=False,
        log_virtue_scores=False,
        log_traces=True,
    )

    # AgentLoop: now wired with curriculum + replay_buffer
    loop = AgentLoop(
        runtime=runtime,
        planner=None,          # AgentLoop will use fallback planner behavior if needed
        curriculum=curriculum,  # Pass B: curriculum is the ONLY goal authority
        skills=getattr(runtime, "skills", None),
        config=loop_cfg,
        replay_buffer=replay_buffer,
    )

    for ep in range(args.episodes):
        result = loop.run_episode(episode_id=ep)
        plan = result.plan or {}
        goal_text = plan.get("goal_text", "<no goal text>")
        steps = plan.get("steps", []) or []

        print(f"\n=== Episode {ep} ===")
        print(f"Goal: {goal_text}")
        print(f"Steps: {len(steps)}")
        if steps:
            print("First step:", steps[0])

    print(
        "\nAgent demo completed without crashing. "
        "If you got this far, the core stack boots and writes experiences to the replay buffer."
    )


if __name__ == "__main__":
    main()

