# scripts/smoke_scribe_model.py

from __future__ import annotations

import pprint
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure src/ is on sys.path so imports like `llm_stack` and `spec` work
# when running this script directly from the project root.
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

# Now we can import from the src/ packages
from llm_stack.stack import LLMStack
from llm_stack.schema import TraceSummaryRequest


def main() -> int:
    stack = LLMStack()

    # Fake trace resembling what M8/M10 might log later
    trace = {
        "goal": "Maintain continuous charcoal and creosote production.",
        "tech_state_before": "LV",
        "tech_state_after": "LV",
        "plan": {
            "steps": [
                {
                    "skill": "feed_coke_ovens",
                    "params": {
                        "target_ovens": ["coke_oven_1", "coke_oven_2"],
                        "log_type": "oak",
                    },
                },
                {
                    "skill": "empty_creosote",
                    "params": {
                        "source_ovens": ["coke_oven_1", "coke_oven_2"],
                        "target_tank": "storage_tank_1",
                    },
                },
            ]
        },
        "actions": [
            {"skill": "feed_coke_ovens", "status": "success", "duration_ticks": 200},
            {"skill": "empty_creosote", "status": "success", "duration_ticks": 120},
        ],
        "outcome": {
            "charcoal_stock": 4 * 64,
            "creosote_tank_level": "75%",
            "notes": "Base now has stable charcoal + creosote flow.",
        },
    }

    req = TraceSummaryRequest(
        trace=trace,
        purpose="context_chunk",  # could also be "human_doc" or "debug"
    )

    print("Calling LLMStack().scribe.summarize_trace(...)\n")

    start = time.time()
    resp = stack.scribe.summarize_trace(req)
    elapsed = time.time() - start

    print("Received TraceSummaryResponse:\n")
    # resp is a dataclass; dump its fields
    pprint.pprint(resp.__dict__)

    print(f"\nElapsed wall time: {elapsed:.3f} seconds")

    # Weak sanity checks
    if not resp.summary:
        print("\n[WARN] summary is empty.")
    if not isinstance(resp.keywords, list):
        print("\n[WARN] keywords is not a list.")
    if not isinstance(resp.suggested_tags, list):
        print("\n[WARN] suggested_tags is not a list.")

    print("\nScribeModel smoke test completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

