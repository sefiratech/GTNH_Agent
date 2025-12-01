# tools/audit/check_llm_roles.py
#!/usr/bin/env python
"""
Quick LLM role boundary audit.

Checks:
  1) No `llm_stack.call(` in src/ (generic LLM usage).
  2) No obvious role-bleed:
       - PlanCodeModel + ErrorModel names in the same *implementation* file.

This is intentionally dumb but fast. You can refine later with AST parsing.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"

# Files that are allowed to mention multiple roles because they are
# pure orchestrators/wiring layers, not role implementations.
ROLE_BLEED_WHITELIST: set[str] = {
    "src/llm_stack/stack.py",
}

# 1) Ban generic LLM calls in src
def check_generic_calls() -> list[str]:
    offenders: list[str] = []
    for path in SRC_ROOT.rglob("*.py"):
        text = path.read_text(encoding="utf-8", errors="ignore")
        if "llm_stack.call(" in text:
            rel = path.relative_to(PROJECT_ROOT)
            offenders.append(str(rel))
    return offenders


# 2) Crude role-bleed check: PlanCodeModel + ErrorModel in same file
def check_role_bleed() -> list[str]:
    offenders: list[str] = []
    for path in (SRC_ROOT / "llm_stack").rglob("*.py"):
        text = path.read_text(encoding="utf-8", errors="ignore")
        has_plan_code = "PlanCodeModel" in text or "PlanCodeModelImpl" in text
        has_error_model = "ErrorModel" in text or "ErrorModelImpl" in text
        if has_plan_code and has_error_model:
            rel = path.relative_to(PROJECT_ROOT)
            rel_str = str(rel)
            # Orchestrator modules are allowed to reference multiple roles.
            if rel_str in ROLE_BLEED_WHITELIST:
                continue
            offenders.append(rel_str)
    return offenders


def main() -> int:
    generic = check_generic_calls()
    bleed = check_role_bleed()

    failed = False

    if generic:
        failed = True
        print("❌ Generic llm_stack.call() found in runtime code:")
        for p in generic:
            print(f"   - {p}")

    if bleed:
        failed = True
        print("❌ Potential PlanCodeModel/ErrorModel role bleed in:")
        for p in bleed:
            print(f"   - {p}")

    if not failed:
        print("✅ LLM role boundary checks passed.")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

