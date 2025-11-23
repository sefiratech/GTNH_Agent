# tools/validate_env.py

import sys           # for exit codes
from pprint import pprint  # for structured printing

# make src discoverable if running as a script
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from env.loader import load_environment  # import our loader


def main() -> None:
    """Load and print the resolved environment, failing fast on errors."""
    try:
        env = load_environment()             # try to resolve the active env profile
    except Exception as e:                   # catch *any* error for debugging
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
