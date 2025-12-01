# src/integration/validators/__init__.py

"""
Validation and regression helpers for Phase 1 integration.

This package collects small utilities that:
- verify Skill Pack / Skill Registry integrity (M5)
- support snapshot-style regression tests for semantics (M3)
- add guardrails for planner stability (M2)
- help track virtue config regressions (M4)
"""

from .skill_integrity import (
    SkillPackIntegrityResult,
    validate_skill_packs_against_registry,
)
