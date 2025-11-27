# scripts/smoke_error_model.py

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
from llm_stack.stack import LLMStack
from llm_stack.schema import ErrorContext  # dataclass you already defined


def main() -> int:
    stack = LLMStack()

    # Simulate a classic JSON failure from PlanCodeModel
    fake_prompt = "You MUST return only valid JSON with a 'steps' list."
    fake_raw_response = '{ "steps": [ { "skill": "feed_coke_ovens" '  # truncated on purpose

    ctx = ErrorContext(
        role="plan_code",
        operation="plan",
        prompt=fake_prompt,
        raw_response=fake_raw_response,
        error_type="json_decode_error",
        metadata={
            "exception": "JSONDecodeError: Expecting ',' delimiter",
            "module": "PlanCodeModel.plan",
        },
    )

    print("Calling LLMStack().error.analyze_failure(...)\n")

    start = time.time()
    analysis = stack.error.analyze_failure(ctx)
    elapsed = time.time() - start

    print("Received ErrorAnalysis:\n")
    # ErrorAnalysis is a dataclass; pprint its __dict__ for readability
    pprint.pprint(analysis.__dict__)

    print(f"\nElapsed wall time: {elapsed:.3f} seconds")

    # Weak sanity checks
    if not analysis.classification:
        print("\n[WARN] classification is empty.")
    if not isinstance(analysis.retry_advised, bool):
        print("\n[WARN] retry_advised is not a bool.")

    print("\nErrorModel smoke test completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

