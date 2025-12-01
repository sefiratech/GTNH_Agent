## Phase 1 Integration Contract

**Entry point**

- `integration.phase1_integration.run_phase1_planning_episode(world, goal, virtue_context_id)`

**Assumptions (Phase 1 only)**

- Planner (M2):
  - Uses a **fake / deterministic planner backend** (`FakePlannerBackend` from `integration.testing.fakes`)
  - No real LLM calls in Phase 1

- Virtue lattice (M4):
  - Uses **fake summarizer and comparer** from `integration.testing.fakes`
  - Real virtue configs can come later; Phase 1 just cares about the shape

- Semantics (M3):
  - Uses the **real semantics DB** and tech-state inference

- Skills (M5):
  - Uses **real SkillSpecs** (YAML), Skill Packs, and Python implementations
  - Registry + integrity tests must pass

**Stability**

- The `run_phase1_planning_episode(world, goal, virtue_context_id)` signature is considered stable for Phase 1.
- Future phases may add keyword-only parameters, but the three positional arguments must remain compatible.
