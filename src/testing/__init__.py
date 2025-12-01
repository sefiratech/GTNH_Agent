# src/integration/testing/__init__.py

"""
Helpers and fakes for Phase 1 integration testing.

This package provides:

- Fake world / tech state types
- Fake planner backend (deterministic, no real LLM)
- Fake virtue / semantics behavior

These are used to:
- exercise integration.phase1_integration.run_phase1_planning_episode()
- define repeatable offline regression scenarios (e.g. LV coke ovens)
"""

from .fakes import (
    FakeWorldState,
    FakeTechState,
    FakePlannerBackend,
    fake_get_current_tech_state,
    fake_load_semantics_db,
    fake_load_virtue_config,
    fake_summarize_plan,
    fake_compare_plans,
)
