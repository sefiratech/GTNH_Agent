# scripts/smoke_llm_stack.py

from __future__ import annotations

import pprint
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Make sure src/ is on sys.path so imports like `llm_stack` and `spec` work
# when running this script directly from the project root.
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

# Now we can import from the src/ packages
from env.loader import load_environment
from llm_stack.stack import LLMStack
from spec.types import Observation


def main() -> int:
    # -----------------------------------------------------------------------
    # Load environment info so we can see which model/profile we're exercising
    # -----------------------------------------------------------------------
    env = load_environment()
    mp = env.model_profile
    primary = mp.models.get("primary")

    print("=== EnvProfile ===")
    print(f"Active env profile : {env.name}")
    print(f"Bot mode           : {env.bot_mode}")
    print(f"Model profile      : {mp.name}")
    print(f"Backend            : {mp.backend}")
    if primary is not None:
        print(f"Primary model path : {primary.path}")
        print(f"Context length     : {primary.context_length}")
    print()

    # Instantiate the LLM stack (uses env.yaml + models.yaml + hardware.yaml)
    stack = LLMStack()

    # Minimal fake observation for planning
    obs = Observation(
        json_payload={
            "tech_state": "LV",
            "machines": ["coke_oven", "steam_boiler"],
            "notes": "Small LV steam base with coke ovens and a boiler.",
        },
        text_summary=(
            "LV steam base with coke ovens and a boiler; "
            "wants stable fuel/creosote."
        ),
    )

    # A tiny skill set so Qwen has something to reference
    skill_descriptions = {
        "feed_coke_ovens": {
            "description": (
                "Keep coke ovens supplied with logs so they run continuously."
            ),
            "params": {
                "target_ovens": "list of coke oven positions",
                "log_type": "preferred log type",
            },
        },
        "empty_creosote": {
            "description": (
                "Move creosote oil from coke ovens into storage tanks."
            ),
            "params": {
                "source_ovens": "list of coke oven positions",
                "target_tank": "storage tank location",
            },
        },
    }

    constraints = {
        "max_steps": 8,
        "avoid": ["breaking_critical_machines"],
    }

    print("Calling LLMStack().plan_code.plan(...)\n")
    start = time.perf_counter()
    plan = stack.plan_code.plan(
        observation=obs,
        goal=(
            "Maintain continuous charcoal and creosote production "
            "at the LV steam base."
        ),
        skill_descriptions=skill_descriptions,
        constraints=constraints,
    )
    end = time.perf_counter()
    elapsed = end - start

    print("Received plan dict:\n")
    pprint.pprint(plan, width=100)

    print(f"\nElapsed wall time: {elapsed:.3f} seconds")

    # Basic sanity checks (not strict tests, just smoke)
    if not isinstance(plan, dict):
        print("\n[WARN] Plan is not a dict.")
        return 1

    if "steps" not in plan:
        print("\n[WARN] Plan has no 'steps' key.")
        return 1

    notes = plan.get("notes")
    if notes == "json_parse_error":
        print(
            "\n[WARN] PlanCodeModel returned non-JSON output; "
            "see 'raw_text' above for inspection."
        )

    print("\nSmoke test completed without hard failures.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

