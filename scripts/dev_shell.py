# scripts/dev_shell.py
import sys
from pathlib import Path

# add src/ to sys.path
ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

def main() -> None:
    from llm_stack import LLMStack
    from spec.types import Observation

    stack = LLMStack()
    obs = Observation(json_payload={}, text_summary="test gpu offload")

    plan = stack.plan_code.plan(
        observation=obs,
        goal="Place a chest in front of the player.",
        skill_descriptions={},
        constraints={},
    )

    print("Plan result:", plan)

if __name__ == "__main__":
    main()

