# tools/validate_env.py

import sys           # for exit codes
from pprint import pprint  # for structured printing
import os            # for CI detection

# make src discoverable if running as a script
from pathlib import Path
# __file__ is .../config/tools/validate_env.py
# parents[2] is the project root; append ROOT/src
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT / "src"))

from env.loader import load_environment  # import our loader

IS_CI = os.getenv("CI") == "true"


def main() -> None:
    """Load and print the resolved environment, failing fast on errors."""
    try:
        env = load_environment()             # try to resolve the active env profile
    except Exception as e:                   # catch *any* error for debugging
        # Special case: in CI, ignore missing *local* model files
        is_missing_model = isinstance(e, FileNotFoundError) and "Missing model file" in str(e)
        if IS_CI and is_missing_model:
            print("Environment validation WARNING (CI, missing model file):", file=sys.stderr)
            print(repr(e), file=sys.stderr)
            sys.exit(0)                      # treat as success in CI

        print("Environment validation FAILED:", file=sys.stderr)
        print(repr(e), file=sys.stderr)
        sys.exit(1)                          # non-zero exit: CI will mark as failed

    print("Environment validation OK.")
    print("\nActive profile:", env.name)
    print("\nBot mode:", env.bot_mode)
    print("\nMinecraft profile:")
    pprint(env.minecraft)
    print("\nModel profile:")
    pprint(env.model_profile)
    print("\nHardware profile:")
    pprint(env.hardware_profile)


if __name__ == "__main__":
    main()  # run main() only when script is executed directly

