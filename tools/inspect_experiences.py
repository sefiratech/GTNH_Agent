# Path: tools/inspect_experiences.py

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

from learning.buffer import ExperienceBuffer


def _summarize_last_experience(raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract a small, human-readable summary from a raw Experience dict.
    """
    goal = raw.get("goal") or {}
    if not isinstance(goal, dict):
        goal = {"id": "", "text": str(goal), "phase": "", "source": ""}

    final_outcome = raw.get("final_outcome") or {}
    problem_signature = raw.get("problem_signature") or {}

    summary: Dict[str, Any] = {
        "goal_id": str(goal.get("id", "")),
        "goal_text": str(goal.get("text", "")),
        "goal_phase": str(goal.get("phase", "")),
        "goal_source": str(goal.get("source", "")),
        "success": bool(final_outcome.get("success", False)),
    }

    # Optional tech info if present in problem_signature
    if isinstance(problem_signature, dict):
        tech_tier = problem_signature.get("tech_tier")
        if tech_tier is None:
            tech_state = problem_signature.get("tech_state")
            if isinstance(tech_state, dict):
                tech_tier = tech_state.get("tier")
        if tech_tier is not None:
            summary["tech_tier"] = str(tech_tier)

    return summary


def inspect_experiences(path: Path) -> None:
    """
    Load the JSONL replay buffer and print:
      - total experience count
      - summary of the last experience (if any)
    """
    buffer = ExperienceBuffer(path)

    # Count episodes via buffer API
    total = buffer.count()

    print(f"Experience file: {path}")
    print(f"Total experiences (non-empty lines): {total}")

    if total == 0:
        print("Buffer is empty. Run a few episodes first.")
        return

    # Grab the last non-empty line and parse it
    last_raw: Optional[Dict[str, Any]] = None
    for raw in buffer.load_all_raw():
        last_raw = raw

    if last_raw is None:
        print("No valid JSON lines found (all corrupted?).")
        return

    summary = _summarize_last_experience(last_raw)

    print("\nLast experience summary:")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect the M10 experience replay buffer (JSONL).\n"
            "Prints total count and a summary of the last experience."
        )
    )
    parser.add_argument(
        "--path",
        type=Path,
        required=True,
        help="Path to the JSONL experience buffer (e.g. data/experiences/replay.jsonl)",
    )

    args = parser.parse_args()
    inspect_experiences(args.path)


if __name__ == "__main__":
    main()

