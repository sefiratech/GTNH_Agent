# path: tools/audit/llm_role_usage_report.py
#!/usr/bin/env python3
"""
LLM Role Usage Reporter

Helper for:
- Quality 1 (Self-eval + retry)
- Quality 6 (Structured LLM-role separation)
- Collision detection between Planner / Critic / ErrorModel / Scribe

Scans the repo for:
- Generic vs role-specific LLMStack calls
- Cross-role references (e.g. ErrorModel imported/used in planner)
- Places where CriticModel is referenced or missing

Outputs a markdown report to stdout.
"""

from pathlib import Path
import re
from typing import Dict, List, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]


ROLE_CALL_PATTERNS = {
    "generic_call": re.compile(r"llm_stack\.call\("),
    "call_planner": re.compile(r"llm_stack\.call_planner\("),
    "call_critic": re.compile(r"llm_stack\.call_critic\("),
    "call_error_model": re.compile(r"llm_stack\.call_error_model\("),
    "call_scribe": re.compile(r"llm_stack\.call_scribe\("),
}

ROLE_NAME_PATTERNS = {
    "Planner": re.compile(r"\bPlanner\b"),
    "PlanCodeModel": re.compile(r"\bPlanCodeModel\b"),
    "CriticModel": re.compile(r"\bCriticModel\b"),
    "ErrorModel": re.compile(r"\bErrorModel\b"),
    "Scribe": re.compile(r"\bScribe\b"),
}


def iter_py_files() -> List[Path]:
    return [p for p in REPO_ROOT.rglob("*.py") if p.is_file()]


def scan_llm_calls(files: List[Path]) -> Dict[str, Dict[str, List[int]]]:
    """
    Returns:
        {relative_path: {pattern_name: [line_numbers...]}}
    """
    results: Dict[str, Dict[str, List[int]]] = {}
    for f in files:
        try:
            lines = f.read_text(encoding="utf-8").splitlines()
        except Exception:
            continue

        rel = str(f.relative_to(REPO_ROOT))
        file_hits: Dict[str, List[int]] = {}

        for lineno, line in enumerate(lines, start=1):
            for tag, pattern in ROLE_CALL_PATTERNS.items():
                if pattern.search(line):
                    file_hits.setdefault(tag, []).append(lineno)

        if file_hits:
            results[rel] = file_hits
    return results


def scan_role_name_collisions(files: List[Path]) -> Dict[str, Dict[str, List[int]]]:
    """
    Looks for references to role names anywhere, useful for collisions:
    - ErrorModel in planner
    - CriticModel missing entirely
    """
    results: Dict[str, Dict[str, List[int]]] = {}
    for f in files:
        try:
            lines = f.read_text(encoding="utf-8").splitlines()
        except Exception:
            continue
        rel = str(f.relative_to(REPO_ROOT))
        file_hits: Dict[str, List[int]] = {}
        for lineno, line in enumerate(lines, start=1):
            for tag, pattern in ROLE_NAME_PATTERNS.items():
                if pattern.search(line):
                    file_hits.setdefault(tag, []).append(lineno)
        if file_hits:
            results[rel] = file_hits
    return results


def summarize_llm_calls(call_hits: Dict[str, Dict[str, List[int]]]) -> None:
    print("# LLM Role Usage Report\n")

    print("## Files using generic vs role-specific calls\n")
    print("| File | generic `llm_stack.call` | call_planner | call_critic | call_error_model | call_scribe |")
    print("|------|--------------------------|--------------|-------------|------------------|------------|")

    for rel, hits in sorted(call_hits.items()):
        generic = len(hits.get("generic_call", []))
        planner = len(hits.get("call_planner", []))
        critic = len(hits.get("call_critic", []))
        err = len(hits.get("call_error_model", []))
        scribe = len(hits.get("call_scribe", []))
        print(f"| {rel} | {generic} | {planner} | {critic} | {err} | {scribe} |")

    print("\n> Any non-zero count in `generic llm_stack.call` is a Q1 refactor target for Quality 6.\n")


def summarize_role_collisions(role_hits: Dict[str, Dict[str, List[int]]]) -> None:
    print("## Role name references (collision hints)\n")
    print("| File | Planner | PlanCodeModel | CriticModel | ErrorModel | Scribe |")
    print("|------|---------|--------------|-------------|------------|--------|")

    for rel, hits in sorted(role_hits.items()):
        planner = len(hits.get("Planner", []))
        plan_code = len(hits.get("PlanCodeModel", []))
        critic = len(hits.get("CriticModel", []))
        err = len(hits.get("ErrorModel", []))
        scribe = len(hits.get("Scribe", []))
        print(f"| {rel} | {planner} | {plan_code} | {critic} | {err} | {scribe} |")

    print(
        "\n> Use this to spot collisions: "
        "e.g. ErrorModel referenced inside planner modules, or CriticModel never appearing anywhere."
    )


def main() -> None:
    files = iter_py_files()
    call_hits = scan_llm_calls(files)
    role_hits = scan_role_name_collisions(files)
    summarize_llm_calls(call_hits)
    summarize_role_collisions(role_hits)


if __name__ == "__main__":
    main()
