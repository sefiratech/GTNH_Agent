# src/cli/phase1_offline.py

import argparse
import json

from env_loader import load_env_profile  # M0
from integration.adapters.m0_env_to_world import world_from_env_profile
from integration.phase1_integration import run_phase1_planning_episode


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 1 offline planner (P0 env + P1 integration)."
    )
    parser.add_argument("--env", required=True, help="Env profile name (from env.yaml)")
    parser.add_argument("--goal", required=True, help="Natural language planning goal")
    parser.add_argument(
        "--virtue-context-id",
        default="lv_coke_ovens",
        help="Virtue context id (e.g. lv_coke_ovens, lv_resources)",
    )
    args = parser.parse_args()

    # P0: load env profile
    profile = load_env_profile(args.env)

    # Adapter: EnvProfile -> WorldState
    world = world_from_env_profile(profile)

    # P1: run planning episode
    best_plan, scores = run_phase1_planning_episode(
        world=world,
        goal=args.goal,
        virtue_context_id=args.virtue_context_id,
    )

    print(json.dumps(
        {
            "best_plan": best_plan,
            "scores": scores,
        },
        indent=2,
        sort_keys=True,
    ))


if __name__ == "__main__":
    main()
