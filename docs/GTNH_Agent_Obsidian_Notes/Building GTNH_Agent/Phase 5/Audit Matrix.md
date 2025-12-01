
```markdown

1. Self-evaluation + Retry Loop
    

What exists:  
No functions found for propose/evaluate/retry in M8.  
No CriticModel fields (failure_type, severity, fix_suggestions) in M2.  
No M9 events (PlanEvaluated, PlanRetried, PlanAbandoned).

What’s missing:  
Entire M8 self-eval pipeline.  
CriticModel structured outputs.  
All plan evaluation event types.  
Virtue-based evaluation helper (evaluate_plan_with_virtues).

Where it lives:  
Expected in: src/agent_loop, src/llm_stack, src/monitoring, src/virtues.

Landmines:  
ErrorModel referenced in planning modules → role collision.  
No CriticModel implementation → planner and error model overloaded.

2. Skill Evolution & Versioning
    

What exists:  
Skill YAMLs contain version, status, origin, and metrics (inconsistently).  
Registry functions partially present.  
SkillLearningManager exists in M10.

What’s missing:  
No function to create candidate skill versions.  
No metric-update functions.  
No promotion/demotion logic wired to learning.  
Registry not version-aware in actual use.

Where it lives:  
Expected in: src/skills/schema.py, src/skills/registry.py, src/learning/manager.py, src/learning/synthesizer.py.

Landmines:  
YAML metadata inconsistent across skills.  
Learning cannot evolve skills safely because versioning is incomplete.

3. Hierarchical Planning
    

What exists:  
No spec-layer types for AgentGoal, TaskPlan, SkillInvocation.  
No planner modes (plan_goal, plan_task).  
No state-machine markers in M8.

What’s missing:  
Entire hierarchical planning flow.  
All spec types.  
State-machine segmentation in the agent loop.  
Planner presets for multi-level planning.

Where it lives:  
Expected in: src/spec/agent_loop.py, src/spec/skills.py, src/llm_stack/planner.py, src/agent_loop/loop.py.

Landmines:  
Planner is monolithic; hierarchy will require structural refactors.  
Risk of planner doing both high-level and low-level planning without boundaries.

4. Experience Memory
    

What exists:  
Replay buffer exists with append_experience.  
Some experience-related references in learning schema.

What’s missing:  
Experience schema fields (problem_signature, final_outcome, virtue_scores, lessons).  
No query_similar_experiences.  
No build_experience_from_episode in M8.  
No summarize_episode_for_memory in Scribe.  
No call sites linking M8 episodes into experience buffer.

Where it lives:  
Expected in: src/learning/buffer.py, src/learning/schema.py, src/agent_loop/loop.py, src/llm_stack/scribe.py.

Landmines:  
Without proper episode-to-experience processing, learning is nearly blind.  
Scribe’s role in memory summarization is undefined and drifting.

5. Curriculum-Driven Goal Selection
    

What exists:  
CurriculumConfig and PhaseConfig exist.  
Basic curriculum engine present.

What’s missing:  
Metadata fields such as required_tech_state, preferred_virtue_context, entry_conditions, exit_conditions not consistently detected.  
No next_goal integration at the top of M8’s loop.  
No clear M8 → M11 → M8 loop for goal selection.

Where it lives:  
Expected in: src/curriculum/engine.py, src/curriculum/schema.py, src/agent_loop/loop.py.

Landmines:  
Agent loop currently uses implicit goals rather than curriculum-authoritative ones.  
Planner doesn’t receive goal IDs, causing misalignment with learning and curriculum logic.

6. Structured LLM-Role Separation
    

What exists:  
Some role references exist (planner, critic, error model, scribe).  
Some role-specific call functions exist but inconsistently.

What’s missing:  
Role definitions missing in llm_roles.yaml.  
Many modules still not using call_planner, call_critic, call_error_model, call_scribe.  
CriticModel effectively unimplemented.  
Role scheduling rules do not exist.

Where it lives:  
Expected in: src/llm_stack/stack.py, src/llm_stack/planner.py, src/llm_stack/critic.py, config/llm_roles.yaml, src/spec/llm.py.

Landmines:  
PlanCodeModel imports ErrorModel → role contamination.  
CriticModel referenced but not actually implemented.  
Missing role definitions block future self-eval and retry integration.

7. Predictive World-Model
    

What exists:  
None of the world-model components exist (no WorldModel class, no predictive functions).

What’s missing:  
Entire world-model layer.  
All predictive functions (simulate_tech_progress, estimate_infra_effect, estimate_resource_trajectory).  
No integration in M8, M4, M11.

Where it lives:  
Expected in: src/world/world_model.py, src/semantics, src/agent_loop/loop.py, src/virtues, src/curriculum.

Landmines:  
No predictive modeling means long-horizon planning and project evaluation remain brittle.  
TechGraph exists but is not used predictively, which complicates future integrations.
```

---
# Q0 Auto-scan Summary

> This is heuristic: it only checks symbol presence, not correctness.

## Quality 1: Self-evaluation + Retry Loop

| Module | Status    | Present / Total | Notes (auto-generated)                                                                                                                                                                        |
| ------ | --------- | --------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| M2     | `missing` | 0 / 3           | string 'failure_type' MISSING (expected in ['src/llm_stack']) – Error/Critic response field<br>string 'severity' MISSING (expected in ['src/llm_stack']) – Error/Critic response field        |
| M4     | `missing` | 0 / 1           | function 'evaluate_plan_with_virtues' MISSING (expected in ['src/virtues', 'src/integration']) – Virtue-aware plan evaluation                                                                 |
| M8     | `missing` | 0 / 7           | function 'propose_plan' MISSING (expected in ['src/agent_loop']) – Agent loop planning step<br>function 'evaluate_plan' MISSING (expected in ['src/agent_loop']) – Agent loop evaluation step |
| M9     | `missing` | 0 / 3           | event 'PlanEvaluated' MISSING (expected in ['src/monitoring']) – Monitoring event<br>event 'PlanRetried' MISSING (expected in ['src/monitoring']) – Monitoring event                          |

## Quality 2: Dynamic Skill Evolution & Versioning

| Module | Status    | Present / Total | Notes (auto-generated)                                                                                                                                                                                                                                                                                                                             |
| ------ | --------- | --------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| M10    | `stub`    | 1 / 3           | string 'SkillLearningManager' FOUND in src/learning/manager.py, src/learning/curriculum_hooks.py – Learning manager for skills<br>string 'create_candidate_skill' MISSING (expected in ['src/learning']) – Any method that creates candidate skills                                                                                                |
| M5     | `partial` | 4 / 8           | config_key 'version' FOUND in config/skills/plant_sapling.yaml, config/skills/maintain_coke_ovens.yaml, config/skills/feed_coke_ovens.yaml – Skill version field in YAML<br>config_key 'status' FOUND in config/skills/plant_sapling.yaml, config/skills/maintain_coke_ovens.yaml, config/skills/feed_coke_ovens.yaml – Skill status field in YAML |

## Quality 3: Hierarchical Planning

| Module | Status | Present / Total | Notes (auto-generated) |
|--------|--------|-----------------|-------------------------|
| M2 | `missing` | 0 / 2 | function 'plan_goal' MISSING (expected in ['src/llm_stack']) – Planner mode: goal → tasks<br>function 'plan_task' MISSING (expected in ['src/llm_stack']) – Planner mode: task → skills |
| M8 | `missing` | 0 / 5 | string 'GoalSelection' MISSING (expected in ['src/agent_loop']) – Agent loop state name<br>string 'TaskPlanning' MISSING (expected in ['src/agent_loop']) – Agent loop state name |
| spec/* | `missing` | 0 / 3 | class 'AgentGoal' MISSING (expected in ['src/spec']) – Spec: Agent goal type<br>class 'TaskPlan' MISSING (expected in ['src/spec']) – Spec: Task-level plan |

## Quality 4: Experience Memory

| Module | Status | Present / Total | Notes (auto-generated) |
|--------|--------|-----------------|-------------------------|
| M10 | `stub` | 1 / 6 | config_key 'problem_signature' MISSING (expected in ['src/learning', 'tests']) – Experience field<br>config_key 'final_outcome' MISSING (expected in ['src/learning', 'tests']) – Experience field |
| M2 | `missing` | 0 / 1 | function 'summarize_episode_for_memory' MISSING (expected in ['src/llm_stack']) – Scribe helper for memory |
| M8 | `missing` | 0 / 2 | function 'build_experience_from_episode' MISSING (expected in ['src/agent_loop', 'src/integration']) – Mapper from episode → experience<br>string 'append_experience' MISSING (expected in ['src/agent_loop', 'src/integration']) – Call site into buffer |

## Quality 5: Curriculum-driven Goal Selection

| Module | Status | Present / Total | Notes (auto-generated) |
|--------|--------|-----------------|-------------------------|
| M11 | `partial` | 3 / 7 | class 'CurriculumConfig' FOUND in src/curriculum/schema.py – Core curriculum schema<br>class 'PhaseConfig' FOUND in src/curriculum/schema.py – Per-phase config |
| M8 | `missing` | 0 / 1 | string 'curriculum.next_goal' MISSING (expected in ['src/agent_loop', 'src/curriculum']) – Curriculum is goal source |

## Quality 6: Structured LLM-role Separation

| Module | Status | Present / Total | Notes (auto-generated) |
|--------|--------|-----------------|-------------------------|
| M2 | `stub` | 1 / 8 | config_key 'planner' MISSING (expected in ['config']) – Planner role definition<br>config_key 'critic' MISSING (expected in ['config']) – Critic role definition |
| runtime/* | `missing` | 0 / 1 | string 'llm_stack.call(' MISSING (expected in ['src']) – Generic LLM calls (should be phased out) |

## Quality 7: Lightweight Predictive World-model

| Module | Status | Present / Total | Notes (auto-generated) |
|--------|--------|-----------------|-------------------------|
| M11 | `missing` | 0 / 1 | string 'estimate_resource_trajectory' MISSING (expected in ['src/curriculum']) – Curriculum using world-model |
| M3 | `missing` | 0 / 4 | class 'WorldModel' MISSING (expected in ['src/world', 'src/semantics']) – World model main class<br>function 'simulate_tech_progress' MISSING (expected in ['src/world', 'src/semantics']) – Tech progress simulation |
| M4 | `missing` | 0 / 1 | string 'estimate_infra_effect' MISSING (expected in ['src/virtues']) – Virtue scoring using world-model |
| M8 | `missing` | 0 / 1 | string 'simulate_tech_progress' MISSING (expected in ['src/agent_loop']) – Agent loop using world-model |

---

# LLM Role Usage Report

## Files using generic vs role-specific calls

| File | generic `llm_stack.call` | call_planner | call_critic | call_error_model | call_scribe |
|------|--------------------------|--------------|-------------|------------------|------------|
| tools/audit/q0_auto_scan.py | 1 | 0 | 0 | 0 | 0 |

> Any non-zero count in `generic llm_stack.call` is a Q1 refactor target for Quality 6.

## Role name references (collision hints)

| File | Planner | PlanCodeModel | CriticModel | ErrorModel | Scribe |
|------|---------|--------------|-------------|------------|--------|
| bootstrap_structure.py | 0 | 0 | 2 | 0 | 0 |
| scripts/smoke_error_model.py | 0 | 2 | 0 | 0 | 0 |
| scripts/smoke_llm_stack.py | 0 | 1 | 0 | 0 | 0 |
| src/agent/bootstrap.py | 1 | 0 | 2 | 0 | 0 |
| src/agent/loop.py | 1 | 0 | 0 | 0 | 0 |
| src/agent/runtime_m6_m7.py | 1 | 0 | 7 | 0 | 0 |
| src/app/runtime.py | 0 | 0 | 4 | 0 | 0 |
| src/integration/episode_logging.py | 2 | 0 | 0 | 0 | 0 |
| src/integration/phase1_integration.py | 1 | 0 | 0 | 0 | 0 |
| src/llm_stack/critic.py | 0 | 0 | 1 | 0 | 0 |
| src/llm_stack/error_model.py | 0 | 0 | 0 | 7 | 0 |
| src/llm_stack/plan_code.py | 0 | 10 | 0 | 1 | 0 |
| src/llm_stack/schema.py | 0 | 3 | 0 | 2 | 0 |
| src/llm_stack/stack.py | 0 | 4 | 0 | 3 | 0 |
| src/monitoring/events.py | 1 | 0 | 0 | 0 | 0 |
| src/monitoring/integration.py | 1 | 0 | 0 | 0 | 0 |
| src/observation/encoder.py | 2 | 0 | 0 | 0 | 0 |
| src/observation/pipeline.py | 1 | 0 | 6 | 0 | 0 |
| src/observation/schema.py | 0 | 0 | 2 | 0 | 0 |
| src/skills/registry.py | 1 | 0 | 0 | 0 | 0 |
| src/skills/schema.py | 1 | 0 | 0 | 0 | 0 |
| src/spec/__init__.py | 1 | 0 | 2 | 0 | 0 |
| src/spec/agent_loop.py | 0 | 0 | 3 | 0 | 0 |
| src/spec/llm.py | 1 | 1 | 1 | 1 | 1 |
| tests/fakes/fake_runtime.py | 1 | 0 | 0 | 0 | 0 |
| tests/test_agent_loop_v1.py | 1 | 0 | 0 | 0 | 0 |
| tests/test_architecture_integration.py | 0 | 0 | 3 | 0 | 0 |
| tests/test_observation_pipeline.py | 0 | 0 | 2 | 0 | 0 |
| tests/test_observation_planner_encoding.py | 1 | 0 | 0 | 0 | 0 |
| tests/test_runtime_m6_m7_smoke.py | 0 | 0 | 2 | 0 | 0 |
| tools/agent_demo.py | 3 | 0 | 0 | 0 | 0 |
| tools/audit/llm_role_usage_report.py | 4 | 3 | 6 | 7 | 4 |
| tools/audit/q0_auto_scan.py | 4 | 0 | 0 | 0 | 4 |

> Use this to spot collisions: e.g. ErrorModel referenced inside planner modules, or CriticModel never appearing anywhere.


- PlanCodeModel references ErrorModel → possible role bleed.
- CriticModel referenced but not implemented in any meaningful way.
- No plan_goal / plan_task modes in M2 → Planner is monolithic.
- No M8 goal selection referencing curriculum → possible duplication between M8 and M11.
- Experience extraction absent in M8 → Scribe and M10 may be stepping on each other.



Role Boundary Spec (LLM Roles) — Skeleton
```yaml
# LLM Role Boundary Spec (Q0)

## Planner
- Responsibilities:
- Inputs:
- Outputs:
- Must NOT do:

## PlanCodeModel
- Responsibilities:
- Inputs:
- Outputs:
- Must NOT do:

## CriticModel
- Responsibilities:
- Inputs:
- Outputs:
- Current status: missing | stub | partial | complete
- Must NOT do:

## ErrorModel
- Responsibilities:
- Inputs:
- Outputs:
- Must NOT do:

## Scribe
- Responsibilities:
- Inputs:
- Outputs:
- Must NOT do:

## Scheduling Rules
1. Planner runs first and produces: …
2. CriticModel runs on the planner output to: …
3. PlanCodeModel transforms: …
4. ErrorModel only runs after: …
5. Scribe runs at: …

```

