
| Quality                    | Modules         | Status | Notes |
| -------------------------- | --------------- | ------ | ----- |
| 1. Self-eval + retry       | M8, M2, M9      |        |       |
| 2. Skill evolution         | M5, M10         |        |       |
| 3. Hier planning           | M8, M2, spec    |        |       |
| 4. Experience memory       | M10, M8, Scribe |        |       |
| 5. Curriculum-driven goals | M11, M8, M3, M4 |        |       |
| 6. LLM-role separation     | M2, config      |        |       |
| 7. Predictive world-model  | M3, M4, M8, M11 |        |       |


## Ground Truth

### 1. Self-evaluation + retry loop

**Target modules:**

- Primary: `M8`
    
- Support: `M2`, `M9`
    

**Audit checklist:**

- In **M8**:
    
    - Does the loop have explicit steps:
        
        - `propose_plan`
            
        - `evaluate_plan` (separate from planning)
            
        - `maybe_retry` / retry budget
            
        - `evaluate_outcome` at end of episode
            
    - Are dataclasses present:
        
        - `PlanAttempt`
            
        - `PlanEvaluation`
            
        - `RetryPolicy`
            
    - Are functions present:
        
        - `evaluate_plan_with_virtues`
            
        - `maybe_retry_plan`
            
        - `postmortem_plan_failure`
            
- In **M2**:
    
    - `CriticModel` & `ErrorModel` return structured fields:
        
        - `failure_type`
            
        - `severity`
            
        - `fix_suggestions`
            
- In **M9**:
    
    - Events / logging types:
        
        - `PlanEvaluated`
            
        - `PlanRetried`
            
        - `PlanAbandoned`
            

**Status outcome:** for each bullet, mark: `missing | stub | partial | complete`.

---

### 2. Dynamic skill evolution & versioning

**Target modules:**

- Primary: `M5`
    
- Secondary: `M10`
    

**Audit checklist:**

- **Skill schema (M5) includes fields:**
    
    - `version`
        
    - `status: active | deprecated | candidate`
        
    - `origin: hand_authored | auto_synthesized`
        
    - `metrics: { success_rate, avg_cost, avg_risk, last_used_at }`
        
- **Registry functions exist:**
    
    - `get_latest_skill(skill_name)`
        
    - `list_skill_versions(skill_name)`
        
    - `register_skill_candidate(candidate_skill)`
        
    - `mark_skill_version_deprecated(skill_id_or_version)`
        
- **M10 integration:**
    
    - Learning output can:
        
        - create new candidate skill version
            
        - update metrics
            
        - promote or demote versions
            

Again, annotate each of these in your status matrix.

---

### 3. Hierarchical planning

**Target modules:**

- Primary: `M8`
    
- Secondary: `M2`
    
- Spec: `spec/agent_loop.py`, `spec/skills.py`, `spec/types.py`
    

**Audit checklist:**

- **Spec layer defines:**
    
    - `AgentGoal`
        
    - `TaskPlan`
        
    - `SkillInvocation`
        
- **M2 planner supports modes:**
    
    - `plan_goal → tasks`
        
    - `plan_task → skills`
        
- **M8 loop shaped like a state machine:**
    
    - `GoalSelection`
        
    - `TaskPlanning`
        
    - `SkillResolution`
        
    - `Execution`
        
    - `Review`
        

Right now you likely have “goal-ish” and “plan-ish” behavior. The checklist forces you to see: is this actually _hierarchical_, or just one long plan?

---

### 4. Experience memory

**Target modules:**

- Primary: `M10`
    
- Hooks: `M8`, Scribe, `M9`
    

**Audit checklist:**

- **Experience schema (e.g. `learning/buffer.py`) has fields:**
```python
Experience = {
    "problem_signature": ...,
    "goal": ...,
    "plan": ...,
    "attempts": ...,
    "final_outcome": ...,
    "virtue_scores": ...,
    "lessons": ...,
}

```


- **Buffer supports:**
    
    - `append_experience(experience)`
        
    - `query_similar_experiences(problem_signature, goal, ...)`
        
- **M8**:
    
    - End of episode calls something like:
        
        - `experience = build_experience_from_episode(...)`
            
        - `replay_store.append_experience(experience)`
            
- **Scribe / M2**:
    
    - Function present: `summarize_episode_for_memory(trace) -> (problem_signature, lessons, etc.)`
        

---

### 5. Curriculum-driven goal selection

This one’s already partly done via **M11 + M8 + M10**.

**Target modules:**

- Primary: `M11`
    
- Integrated with: `M8`, `M3`, `M4`
    

**Audit checklist:**

- Curriculum engine:
    
    - Can output something like `AgentGoal` or at least:
        
        - goal text
            
        - goal id
            
        - phase context
            
- Schema has per-unit metadata:
    
    - `required_tech_state`
        
    - `preferred_virtue_context`
        
    - `entry_conditions`
        
    - `exit_conditions`
        
- M8:
    
    - Top of loop:
        
        - `goal = curriculum.next_goal(tech_state, experience_memory?)`
            
    - Planning step uses that goal as primary objective, not some ad-hoc prompt.
        

You already did the **learning scheduling** part with M10; this is about ensuring _goal selection itself_ is explicitly curriculum-driven.

---

### 6. Structured LLM-role separation

**Target modules:**

- Primary: `M2`
    
- Config: `config/llm_roles.yaml`
    
- Spec: `spec/llm.py`
    

**Audit checklist:**

- `config/llm_roles.yaml` defines, for each role:
    
    - system prompt
        
    - temperature
        
    - stop tokens
        
    - output schema
        
    - tool permissions
        
- `LLMStack` exposes role-specific calls:
    
    - `call_planner(...)`
        
    - `call_critic(...)`
        
    - `call_scribe(...)`
        
    - `call_error_model(...)`
        
- All call sites in **M8**, **M10**, **M11** use these instead of “raw” generic LLM calls.
    

If you grep for `llm_stack.call(` and find anything that doesn’t go through a role-specific method, that’s a target for refactor.

---

### 7. Lightweight predictive world-model

**Target modules:**

- Primary: `M3` (`world_model.py`)
    
- Consumers: `M4`, `M8`, `M11`
    

**Audit checklist:**

- `src/world/world_model.py` (or similar) provides:
    
    - `simulate_tech_progress(current_tech_state, candidate_goal)`
        
    - `estimate_infra_effect(factory_layout, change)`
        
    - `estimate_resource_trajectory(inventory, consumption_rates)`
        
- This world-model uses:
    
    - `TechGraph`
        
    - semantic item/block info (M3)
        
- Integration:
    
    - M8: uses predictions to reject obviously bad plans early.
        
    - M4: uses predicted risk/throughput for virtue scoring.
        
    - M11: uses predicted returns when weighing big goals / long-horizon projects.



## Methodology
### Phase Q0 – Audit-only

1. For each quality (1–7):
    
    - Walk the relevant modules.
        
    - Fill your status matrix (missing/stub/partial/complete).
        
    - Note any “collisions” (places two modules think they own the same concern).
        
2. End result: you have a **truth map** of where you actually are vs the nice manifesto.
3. Subtask: We have PlanCodeModel, ErrorModel, and ScribeModel implemented. We also need to implement CriticModel perhaps tighten the boundaries between each role so they never bleed into each other and have clear scheduling rules.
    

### Phase Q1 – Integration order

Recommended order to actually implement deeply:

1. **LLM-role separation (6)**
    
    - M2-only mostly, unlocks clarity for everything else.
        
2. **Self-eval + retry (1)**
    
    - With roles in place, the planner/critic split is clean.
        
3. **Experience memory (4)**
    
    - So the agent can remember past failures / successes.
        
4. **Curriculum-driven goals (5)**
    
    - Hook curriculum into goal selection, leaning on memory.
        
5. **Skill evolution & versioning (2)**
    
    - Once you have memory, you can evaluate skill performance.
        
6. **Hierarchical planning (3)**
    
    - Add richer structure on top of the now-competent base loop.
        
7. **Predictive world-model (7)**
    
    - Top-shelf polish that helps everything else behave sanely.
        

This way you don’t constantly rewire fundamentals after you’ve already built higher-order stuff.




## Other Important Context

Filestructure:

```

. # Repo root: GTNH_Agent top-level directory  
├── bootstrap_structure.py # Script to (re)create baseline repo structure / scaffolding  
├── config # All YAML / JSON config inputs that shape the agent’s behavior and environment  
│ ├── curricula # Phase/goal curricula for different playstyles or tech paths  
│ │ ├── aesthetic_megabase.yaml # Curriculum tuned for “pretty megabase” / aesthetics-first progression  
│ │ ├── default_speedrun.yaml # Default mainline progression curriculum (speed / efficiency biased)  
│ │ └── eco_factory.yaml # Curriculum focused on eco-friendly / sustainable factory builds  
│ ├── env.yaml # Environment profiles: modpack version, paths, runtime switches, hardware profile bindings  
│ ├── gtnh_blocks.generated.yaml # Autogenerated block semantics for GTNH world (from CSV / dumps)  
│ ├── gtnh_blocks.yaml # Hand-edited overrides / corrections for GTNH block semantics  
│ ├── gtnh_items.generated.yaml # Autogenerated item semantics for GTNH (raw ingest output)  
│ ├── gtnh_items.yaml # Hand-tuned item semantics (categories, tags, corrections)  
│ ├── gtnh_recipes.agent.json # Recipes compressed / transformed for the agent’s planning & craftability checks  
│ ├── gtnh_recipes.generated.json # Bulk recipe dump from ingestors before agent-specific compaction  
│ ├── gtnh_recipes.json # Canonical recipe view / merged recipes used by semantics and planning  
│ ├── gtnh_tech_graph.yaml # Directed graph of GTNH tech progression (nodes = tech states, edges = requirements)  
│ ├── hardware.yaml # Hardware capabilities (CPU/GPU/memory) for choosing models and batch sizes  
│ ├── llm_roles.yaml # Role definitions for planner / critic / scribe / error_model (prompts, temps, tools)  
│ ├── minecraft.yaml # Minecraft-specific config: ports, world name, bot connection settings, etc.  
│ ├── models.yaml # Local model catalog: which GGUFs / backends correspond to which logical roles  
│ ├── raw # Raw input artifacts used to generate the processed semantics configs  
│ │ ├── block.csv # Raw block dump from GTNH / Nerd / Tellme (pre-normalized)  
│ │ ├── item.csv # Raw item dump used to build item semantics  
│ │ ├── recipes.json # Raw recipe dump from NEI/NERD before agent-friendly processing  
│ │ └── recipes_stacks.json # Variant dump with stack sizes / stack-specific recipe info  
│ ├── skill_packs # Declarative groupings of skills for different stages / setups  
│ │ ├── lv_core.yaml # Skill pack for LV core progression (standard early/mid LV toolkit)  
│ │ └── steam_age.yaml # Skill pack for Steam Age baseline (boilers, coke ovens, wood loop, etc.)  
│ ├── skills # Individual skill definitions in YAML (metadata, arguments, contracts)  
│ │ ├── basic_crafting.yaml # High-level definition for generic inventory → crafting operations  
│ │ ├── chop_tree.yaml # Skill config for tree chopping behavior and constraints  
│ │ ├── feed_coke_ovens.yaml # Config for coke oven feeding skill (items, timing, failure modes)  
│ │ ├── feed_steam_boiler.yaml # Steam boiler fuel-feeding skill config  
│ │ ├── maintain_coke_ovens.yaml # Maintenance skill for coke ovens (output clearing, uptime)  
│ │ ├── plant_sapling.yaml # Sapling placement skill config (spacing, targeting, terrain constraints)  
│ │ └── refill_water_tanks.yaml # Water resupply skill config (buckets, tanks, sources)  
│ ├── skills_candidates # Staging area for auto-synthesized / experimental skills before promotion  
│ ├── tools # Small helper scripts for config / environment sanity  
│ │ ├── print_env.py # Debug utility to print resolved env profile and paths  
│ │ └── validate_env.py # Validation script for env.yaml / models.yaml / hardware.yaml coherence  
│ └── virtues.yaml # Base virtue lattice config: weights, names, and default tradeoffs  
├── docs # Human-facing documentation for architecture and integration  
│ ├── architecture.md # High-level system architecture overview and module breakdown  
│ ├── ipc_protocol_m6.md # IPC protocol spec between Forge mod / bot core and runtime  
│ ├── m6_bot_core_1_7_10.md # Design notes for the 1.7.10 bot core (M6)  
│ └── phase1_integration.md # Phase 1 integration notes: env ↔ semantics ↔ bot_core wiring  
├── .github # GitHub-specific configuration  
│ └── workflows # CI/CD runs and checks  
│ └── ci.yml # Main CI pipeline (lint, tests, maybe type-checking)  
├── .gitignore # Files and directories that Git should ignore  
├── logs # Runtime logs; LLM traces, failures, planner runs, etc. (you said no comments inside, so I’m behaving)  
│ └── llm  
│ ├── 20251127T154508_30389_error_model_analyze_failure.json  
│ ├── 20251127T154508_30389_plan_code_plan.json  
│ ├── 20251127T154508_30389_scribe_summarize_trace.json  
│ ├── 20251127T154653_30421_plan_code_plan.json  
│ ├── 20251127T154929_30485_error_model_analyze_failure.json  
│ ├── 20251127T155054_30542_scribe_summarize_trace.json  
│ ├── 20251127T171641_54578_plan_code_plan.json  
│ ├── 20251127T172238_54966_plan_code_plan.json  
│ ├── 20251127T172404_55071_plan_code_plan.json  
│ ├── 20251127T181107_57121_plan_code_plan.json  
│ ├── 20251127T181432_57416_plan_code_plan.json  
│ ├── 20251127T202123_80866_error_model_analyze_failure.json  
│ ├── 20251127T202123_80866_plan_code_plan.json  
│ ├── 20251127T202123_80866_scribe_summarize_trace.json  
│ ├── 20251127T202334_80953_error_model_analyze_failure.json  
│ ├── 20251127T202334_80953_plan_code_plan.json  
│ ├── 20251127T202334_80953_scribe_summarize_trace.json  
│ ├── 20251127T223951_107281_error_model_analyze_failure.json  
│ ├── 20251127T223951_107281_plan_code_plan.json  
│ ├── 20251127T223951_107281_scribe_summarize_trace.json  
│ ├── 20251128T001446_136742_error_model_analyze_failure.json  
│ ├── 20251128T001446_136742_plan_code_plan.json  
│ ├── 20251128T001446_136742_scribe_summarize_trace.json  
│ ├── 20251128T003700_146815_error_model_analyze_failure.json  
│ ├── 20251128T003700_146815_plan_code_plan.json  
│ ├── 20251128T003700_146815_scribe_summarize_trace.json  
│ ├── 20251128T115106_24456_error_model_analyze_failure.json  
│ ├── 20251128T115106_24456_plan_code_plan.json  
│ ├── 20251128T115106_24456_scribe_summarize_trace.json  
│ ├── 20251128T143640_52852_error_model_analyze_failure.json  
│ ├── 20251128T143640_52852_plan_code_plan.json  
│ ├── 20251128T143640_52852_scribe_summarize_trace.json  
│ ├── 20251128T143816_52937_error_model_analyze_failure.json  
│ ├── 20251128T143816_52937_plan_code_plan.json  
│ ├── 20251128T143816_52937_scribe_summarize_trace.json  
│ ├── 20251128T153451_109564_error_model_analyze_failure.json  
│ ├── 20251128T153451_109564_plan_code_plan.json  
│ ├── 20251128T153451_109564_scribe_summarize_trace.json  
│ ├── 20251128T154112_109913_error_model_analyze_failure.json  
│ ├── 20251128T154112_109913_plan_code_plan.json  
│ ├── 20251128T154112_109913_scribe_summarize_trace.json  
│ ├── 20251128T154622_110074_error_model_analyze_failure.json  
│ ├── 20251128T154622_110074_plan_code_plan.json  
│ ├── 20251128T154622_110074_scribe_summarize_trace.json  
│ ├── 20251128T155834_129265_error_model_analyze_failure.json  
│ ├── 20251128T155834_129265_plan_code_plan.json  
│ ├── 20251128T155834_129265_scribe_summarize_trace.json  
│ ├── 20251128T160155_129475_error_model_analyze_failure.json  
│ ├── 20251128T160155_129475_plan_code_plan.json  
│ ├── 20251128T160155_129475_scribe_summarize_trace.json  
│ ├── 20251128T160633_142703_error_model_analyze_failure.json  
│ ├── 20251128T160633_142703_plan_code_plan.json  
│ ├── 20251128T160633_142703_scribe_summarize_trace.json  
│ ├── 20251128T162316_143085_error_model_analyze_failure.json  
│ ├── 20251128T162316_143085_plan_code_plan.json  
│ ├── 20251128T162316_143085_scribe_summarize_trace.json  
│ ├── 20251128T162956_163507_error_model_analyze_failure.json  
│ ├── 20251128T162956_163507_plan_code_plan.json  
│ ├── 20251128T162956_163507_scribe_summarize_trace.json  
│ ├── 20251128T164527_184688_error_model_analyze_failure.json  
│ ├── 20251128T164527_184688_plan_code_plan.json  
│ ├── 20251128T164527_184688_scribe_summarize_trace.json  
│ ├── 20251128T164725_184780_error_model_analyze_failure.json  
│ ├── 20251128T164725_184780_plan_code_plan.json  
│ ├── 20251128T164725_184780_scribe_summarize_trace.json  
│ ├── 20251128T164846_185052_error_model_analyze_failure.json  
│ ├── 20251128T164846_185052_plan_code_plan.json  
│ ├── 20251128T164846_185052_scribe_summarize_trace.json  
│ ├── 20251128T190228_210275_error_model_analyze_failure.json  
│ ├── 20251128T190228_210275_plan_code_plan.json  
│ ├── 20251128T190228_210275_scribe_summarize_trace.json  
│ ├── 20251128T190457_210363_error_model_analyze_failure.json  
│ ├── 20251128T190457_210363_plan_code_plan.json  
│ ├── 20251128T190457_210363_scribe_summarize_trace.json  
│ ├── 20251128T191608_225560_error_model_analyze_failure.json  
│ ├── 20251128T191608_225560_plan_code_plan.json  
│ ├── 20251128T191608_225560_scribe_summarize_trace.json  
│ ├── 20251128T192046_233385_error_model_analyze_failure.json  
│ ├── 20251128T192046_233385_plan_code_plan.json  
│ ├── 20251128T192046_233385_scribe_summarize_trace.json  
│ ├── 20251128T192251_233478_error_model_analyze_failure.json  
│ ├── 20251128T192251_233478_plan_code_plan.json  
│ ├── 20251128T192251_233478_scribe_summarize_trace.json  
│ ├── 20251128T204447_291001_error_model_analyze_failure.json  
│ ├── 20251128T204447_291001_plan_code_plan.json  
│ ├── 20251128T204501_291001_scribe_summarize_trace.json  
│ ├── 20251128T204732_291164_error_model_analyze_failure.json  
│ ├── 20251128T204732_291164_plan_code_plan.json  
│ ├── 20251128T204807_291164_scribe_summarize_trace.json  
│ ├── 20251128T212201_317178_error_model_analyze_failure.json  
│ ├── 20251128T212201_317178_plan_code_plan.json  
│ ├── 20251128T212258_317178_scribe_summarize_trace.json  
│ ├── 20251128T214908_353050_error_model_analyze_failure.json  
│ ├── 20251128T214908_353050_plan_code_plan.json  
│ ├── 20251128T215005_353050_scribe_summarize_trace.json  
│ ├── 20251128T215508_371511_error_model_analyze_failure.json  
│ ├── 20251128T215508_371511_plan_code_plan.json  
│ ├── 20251128T215605_371511_scribe_summarize_trace.json  
│ ├── 20251128T215741_371600_error_model_analyze_failure.json  
│ ├── 20251128T215741_371600_plan_code_plan.json  
│ ├── 20251128T215838_371600_scribe_summarize_trace.json  
│ ├── 20251129T120138_42538_error_model_analyze_failure.json  
│ ├── 20251129T120138_42538_plan_code_plan.json  
│ ├── 20251129T120235_42538_scribe_summarize_trace.json  
│ ├── 20251129T121735_52787_error_model_analyze_failure.json  
│ ├── 20251129T121735_52787_plan_code_plan.json  
│ ├── 20251129T121832_52787_scribe_summarize_trace.json  
│ ├── 20251129T122136_52946_error_model_analyze_failure.json  
│ ├── 20251129T122136_52946_plan_code_plan.json  
│ ├── 20251129T122233_52946_scribe_summarize_trace.json  
│ ├── 20251129T131642_83734_error_model_analyze_failure.json  
│ ├── 20251129T131642_83734_plan_code_plan.json  
│ ├── 20251129T131738_83734_scribe_summarize_trace.json  
│ ├── 20251129T131934_83828_error_model_analyze_failure.json  
│ ├── 20251129T131934_83828_plan_code_plan.json  
│ ├── 20251129T132032_83828_scribe_summarize_trace.json  
│ ├── 20251129T134009_114623_error_model_analyze_failure.json  
│ ├── 20251129T134009_114623_plan_code_plan.json  
│ ├── 20251129T134106_114623_scribe_summarize_trace.json  
│ ├── 20251129T135607_136708_error_model_analyze_failure.json  
│ ├── 20251129T135607_136708_plan_code_plan.json  
│ ├── 20251129T135703_136708_scribe_summarize_trace.json  
│ ├── 20251129T140350_168767_error_model_analyze_failure.json  
│ ├── 20251129T140350_168767_plan_code_plan.json  
│ ├── 20251129T140447_168767_scribe_summarize_trace.json  
│ ├── 20251129T145028_202536_error_model_analyze_failure.json  
│ ├── 20251129T145028_202536_plan_code_plan.json  
│ ├── 20251129T145124_202536_scribe_summarize_trace.json  
│ ├── 20251129T145252_202626_error_model_analyze_failure.json  
│ ├── 20251129T145252_202626_plan_code_plan.json  
│ ├── 20251129T145349_202626_scribe_summarize_trace.json  
│ ├── 20251129T153928_260438_error_model_analyze_failure.json  
│ ├── 20251129T153928_260438_plan_code_plan.json  
│ ├── 20251129T154026_260438_scribe_summarize_trace.json  
│ ├── 20251129T154319_260642_error_model_analyze_failure.json  
│ ├── 20251129T154319_260642_plan_code_plan.json  
│ ├── 20251129T154415_260642_scribe_summarize_trace.json  
│ ├── 20251129T154536_260720_error_model_analyze_failure.json  
│ ├── 20251129T154536_260720_plan_code_plan.json  
│ ├── 20251129T154633_260720_scribe_summarize_trace.json  
│ ├── 20251129T160320_284771_error_model_analyze_failure.json  
│ ├── 20251129T160320_284771_plan_code_plan.json  
│ ├── 20251129T160417_284771_scribe_summarize_trace.json  
│ ├── 20251129T160525_284855_error_model_analyze_failure.json  
│ ├── 20251129T160525_284855_plan_code_plan.json  
│ └── 20251129T160622_284855_scribe_summarize_trace.json  
├── pyproject.toml # Project metadata + dependencies + pytest config (if using poetry/pdm-style layout)  
├── .pytest_cache # Pytest internal cache; speeds up subsequent test runs  
│ ├── CACHEDIR.TAG # Marks this directory as a cache for tools that care  
│ ├── .gitignore # Keeps pytest cache from polluting git  
│ ├── README.md # Pytest cache explanation  
│ └── v # Versioned cache structure  
│ └── cache  
│ ├── lastfailed # Records which tests failed last run  
│ └── nodeids # Cached test node IDs  
├── .python-version # Pyenv / version manager pin for Python interpreter version  
├── README.md # Top-level project description, setup steps, and usage hints  
├── scripts # CLI helpers for data ingest, smoke tests, and demos  
│ ├── compact_recipes_for_agent.py # Compacts raw GTNH recipes into agent-friendly format  
│ ├── demo_offline_agent_step.py # Demonstration script for running a single offline agent step  
│ ├── dev_shell.py # Dev REPL / shell bootstrapper into project environment  
│ ├── ingest_gtnh_semantics.py # ETL script to ingest GTNH semantics from raw dumps  
│ ├── ingest_nerd_csv_semantics.py # Ingest semantics from Nerd CSV exports  
│ ├── ingest_nerd_recipes.py # Ingest and normalize Nerd recipe exports  
│ ├── smoke_error_model.py # Quick smoke test of the error model alone  
│ ├── smoke_llm_stack.py # Smoke test of end-to-end LLM stack behavior  
│ └── smoke_scribe_model.py # Smoke test for the scribe summarization model  
├── src # All actual library / runtime code for GTNH_Agent  
│ ├── agent # Older / higher-level agent runtime bits (M6-M7 bridge)  
│ │ ├── bootstrap.py # Agent startup wiring using legacy runtime path  
│ │ ├── experience.py # Early experience / trace handling for the older agent path  
│ │ ├── logging_config.py # Logging setup for agent processes  
│ │ ├── loop.py # Legacy agent loop implementation (pre-M8 refactor)  
│ │ └── runtime_m6_m7.py # Specialized runtime that bridges M6 bot_core with M7 observation  
│ ├── agent_loop # Newer M8-style agent loop implementation  
│ │ ├── **init**.py # Package marker for agent_loop module  
│ │ ├── loop.py # Main agent loop v1: observe → plan → act → summarize  
│ │ ├── schema.py # Dataclasses / types for the agent loop (episodes, steps, traces)  
│ │ └── state.py # State tracking across steps/episodes (tech state, curriculum context, etc.)  
│ ├── app # Thin application wrapper layer for running the agent as an app  
│ │ ├── **init**.py # Package marker for app module  
│ │ └── runtime.py # Entry wiring of CLI → runtime → agent loop  
│ ├── bot_core # M6: Minecraft-side bot and IPC client abstraction  
│ │ ├── actions.py # High-level bot actions interface (what the bot can do)  
│ │ ├── collision.py # Collision detection / resolution utilities for movement  
│ │ ├── core.py # Core bot client logic & main control object  
│ │ ├── **init**.py # Package marker for bot_core  
│ │ ├── nav # Navigation system for pathfinding and movement  
│ │ │ ├── grid.py # Discrete grid representation of the world for nav  
│ │ │ ├── **init**.py # Package marker for bot_core.nav  
│ │ │ ├── mover.py # Movement primitive execution and state updates  
│ │ │ └── pathfinder.py # Pathfinding algorithms (A*, etc.) over the nav grid  
│ │ ├── net # Networking / IPC layer between Forge server and agent  
│ │ │ ├── client.py # Primary client for connecting to the bot server  
│ │ │ ├── external_client.py # Optional external client variant / wrapper  
│ │ │ ├── **init**.py # Package marker for bot_core.net  
│ │ │ └── ipc.py # Message encoding/decoding and IPC protocol helpers  
│ │ ├── runtime.py # Runtime harness for running bot_core loop independently  
│ │ ├── snapshot.py # Snapshot representation of the world/bot state for planning  
│ │ ├── testing # Bot core test utilities  
│ │ │ └── fakes.py # Fake bot implementations for tests and offline simulation  
│ │ ├── tracing.py # Detailed traces of bot actions for debugging and learning  
│ │ └── world_tracker.py # Tracks world entities/blocks across ticks from bot’s perspective  
│ ├── cli # Command-line entrypoints  
│ │ └── phase1_offline.py # CLI to run Phase 1 offline integration / smoke flows  
│ ├── curriculum # M11: curriculum engine and integration  
│ │ ├── engine.py # Core curriculum engine: resolve phases, goals, virtue overrides, projects  
│ │ ├── example_workflow.py # Example showing how M8/M10/M11 interact with curriculum  
│ │ ├── **init**.py # Package marker for curriculum  
│ │ ├── integration_agent_loop.py # Glue from curriculum into agent_loop (goal selection, hints)  
│ │ ├── loader.py # YAML loader for curriculum configs into dataclasses  
│ │ └── schema.py # Dataclasses: CurriculumConfig, PhaseConfig, LongHorizonProject, etc.  
│ ├── env # M0/M1 environment layer  
│ │ ├── **init**.py # Package marker for env  
│ │ ├── loader.py # Loads env.yaml / hardware.yaml / models.yaml into EnvProfile  
│ │ └── schema.py # Data structures describing env, hardware, models, and profiles  
│ ├── gtnh_agent.egg-info # Packaging metadata for installing this as a Python package  
│ │ ├── dependency_links.txt # Dependency linkage info for setuptools  
│ │ ├── PKG-INFO # Package metadata: name, version, description  
│ │ ├── requires.txt # Runtime requirements derived from pyproject/setup  
│ │ ├── SOURCES.txt # Source files included in the package  
│ │ └── top_level.txt # Top-level package names provided by this distribution  
│ ├── **init**.py # Root package marker for src/gtnh_agent namespace  
│ ├── integration # Phase1+ integration glue between subsystems  
│ │ ├── adapters # Mappers between different internal schemas  
│ │ │ └── m0_env_to_world.py # Adapter: env profiles → initial world/semantics config  
│ │ ├── episode_logging.py # Helpers to log/store episodes in a consistent format  
│ │ ├── **init**.py # Package marker for integration  
│ │ ├── phase1_integration.py # Main Phase 1 integration orchestration script  
│ │ ├── testing # Testing helpers for integration-level tests  
│ │ │ ├── fakes.py # Fake world/env/llm objects for integration tests  
│ │ │ └── **init**.py # Package marker for integration.testing  
│ │ └── validators # Guardrail & validation helpers  
│ │ ├── **init**.py # Package marker for validators  
│ │ ├── planner_guardrails.py # Check planner outputs for safety / sanity  
│ │ ├── semantics_snapshots.py # Validate semantics snapshots for consistency across runs  
│ │ ├── skill_integrity.py # Validate skill configs and registry consistency  
│ │ └── virtue_snapshots.py # Validate virtue lattice / overrides vs snapshots  
│ ├── learning # M10: skill learning, replay, evaluation, synthesis  
│ │ ├── buffer.py # Experience / replay buffer implementation (episodes, experiences)  
│ │ ├── curriculum_hooks.py # Hooks connecting curriculum phases to learning triggers  
│ │ ├── evaluator.py # Skill evaluator from replay episodes and metrics  
│ │ ├── **init**.py # Package marker for learning  
│ │ ├── manager.py # SkillLearningManager: schedules learning cycles per skill/goal  
│ │ ├── schema.py # Dataclasses for learning tasks, results, and experience records  
│ │ └── synthesizer.py # Auto-synthesis / refinement of new skill variants from experience  
│ ├── llm_stack # M2: local LLM stack abstraction and role orchestration  
│ │ ├── backend_llamacpp.py # llama.cpp backend bridge for local models  
│ │ ├── backend.py # Abstract backend interface and base types  
│ │ ├── codegen.py # Utilities for code-generation prompts / flows  
│ │ ├── config.py # LLM stack config loader (from models.yaml / llm_roles.yaml)  
│ │ ├── critic.py # Critic role wrapper: evaluates plans / ideas  
│ │ ├── error_model.py # Error model role: classifies failures and root causes  
│ │ ├── **init**.py # Package marker for llm_stack  
│ │ ├── json_utils.py # Safe JSON parsing / formatting helpers for LLM outputs  
│ │ ├── log_files.py # Helpers for logging LLM interactions into files (e.g., logs/llm)  
│ │ ├── plan_code.py # Planner + coder interaction logic (plan actions / code steps)  
│ │ ├── planner.py # Planner role wrapper: generates plans from goals/worldstate  
│ │ ├── presets.py # Prompt / config presets for different roles and tasks  
│ │ ├── schema.py # Dataclasses / Typed structures for LLM requests/responses  
│ │ ├── scribe.py # Scribe role: summarization, explanations, memory writeups  
│ │ └── stack.py # Central LLMStack orchestrator: dispatches calls to correct role/backend  
│ ├── monitoring # M9: monitoring, dashboards, and event bus  
│ │ ├── bus.py # Event bus implementation for monitoring events  
│ │ ├── controller.py # High-level monitoring controller and lifecycle orchestration  
│ │ ├── dashboard_tui.py # Terminal UI dashboard for live agent monitoring  
│ │ ├── events.py # Event type definitions (PlanEvaluated, PlanRetried, etc.)  
│ │ ├── **init**.py # Package marker for monitoring  
│ │ ├── integration.py # Integration helpers between agent loop and monitoring bus  
│ │ ├── llm_logging.py # Specialized logging of LLM calls and traces  
│ │ ├── logger.py # Monitoring-aware logger facade  
│ │ └── tools.py # Misc tools for debugging / monitoring pipeline  
│ ├── observation # M7: eyes & ears – encode Minecraft world into planner-friendly state  
│ │ ├── encoder.py # Encodes raw snapshots into feature-rich representations  
│ │ ├── **init**.py # Package marker for observation  
│ │ ├── pipeline.py # End-to-end observation pipeline orchestration  
│ │ ├── schema.py # Dataclasses for observations, tiles, entities, etc.  
│ │ ├── testing.py # Observation-focused test helpers / fixtures  
│ │ └── trace_schema.py # Trace structure for observation → planning data flows  
│ ├── runtime # Top-level runtime orchestration (phases, error handling, curriculum hooks)  
│ │ ├── agent_runtime_main.py # Main entrypoint for running the agent runtime  
│ │ ├── bootstrap_phases.py # Bootstraps per-phase configuration and setup  
│ │ ├── curriculum_learning_coordinator.py # Orchestrates M10 learning using curriculum context (M11)  
│ │ ├── curriculum_learning_triggers.py # Trigger manager: when to call M10 based on episodes, phases, projects  
│ │ ├── error_handling.py # Centralized error handling utilities for runtime  
│ │ ├── failure_mitigation.py # Strategies to recover from runtime failures (restart, degrade, etc.)  
│ │ ├── **init**.py # Package marker for runtime  
│ │ └── phase4_curriculum_learning_orchestrator.py # Phase 4 orchestrator: M10 ↔ M11 integration logic  
│ ├── semantics # M3: world semantics, craftability, and tech inference  
│ │ ├── cache.py # Semantics caching layer for performance and reuse  
│ │ ├── categorize.py # Categorization utilities for items/blocks (tags, classes)  
│ │ ├── crafting.py # Craftability logic and recipe resolution  
│ │ ├── ingest # Ingest helpers for semantics data  
│ │ │ └── **init**.py # Package marker for semantics.ingest  
│ │ ├── **init**.py # Package marker for semantics  
│ │ ├── loader.py # Load semantics configs (blocks/items/recipes) into runtime structures  
│ │ ├── schema.py # Dataclasses for semantics entities and relationships  
│ │ └── tech_state.py # TechState inference / transitions from observed world and recipes  
│ ├── skills # M5: skills implementation and registry  
│ │ ├── base # Core hand-authored low-level skill implementations  
│ │ │ ├── basic_crafting.py # Implementation of generic crafting behavior  
│ │ │ ├── chop_tree.py # Code for tree chopping skill  
│ │ │ ├── feed_coke_ovens.py # Implementation for feeding coke ovens  
│ │ │ ├── feed_steam_boiler.py # Implementation for feeding steam boilers  
│ │ │ ├── **init**.py # Package marker for skills.base  
│ │ │ ├── maintain_coke_ovens.py # Implementation for maintaining coke oven uptime  
│ │ │ ├── plant_sapling.py # Implementation of sapling planting skill  
│ │ │ └── refill_water_tanks.py # Implementation of water refilling behavior  
│ │ ├── **init**.py # Package marker for skills  
│ │ ├── loader.py # Loads YAML skill definitions into skill objects  
│ │ ├── packs.py # Skill pack loader/manager that groups skills for phases / profiles  
│ │ ├── registry.py # Skill registry: lookup, registration, (eventually) versioning  
│ │ └── schema.py # Dataclasses for skill definitions, arguments, and metadata  
│ ├── spec # Formal-ish specs for internal interfaces  
│ │ ├── agent_loop.py # Spec for M8 agent loop interfaces and contracts  
│ │ ├── bot_core.py # Spec for bot core interface (what the loop expects from the bot)  
│ │ ├── experience.py # Spec for experience / episode trace structures  
│ │ ├── **init**.py # Package marker for spec  
│ │ ├── llm.py # Spec for LLMStack role APIs and response structures  
│ │ ├── skills.py # Spec for skill interface, arguments, and lifecycle  
│ │ └── types.py # Shared type aliases, small dataclasses used across specs  
│ ├── testing # Shared testing utilities / helpers  
│ │ └── **init**.py # Package marker for src.testing  
│ └── virtues # M4: virtue lattice, metrics, overrides, and sanity checks  
│ ├── explain.py # Human-readable explanations of virtue decisions / scores  
│ ├── features.py # Feature extraction for virtue evaluation (inputs to scoring)  
│ ├── **init**.py # Package marker for virtues  
│ ├── integration_overrides.py # Integration-level overrides for specific phases / scenarios  
│ ├── lattice.py # Core virtue lattice implementation and combination logic  
│ ├── loader.py # Load virtues.yaml into structured configs  
│ ├── metrics.py # Virtue metrics and scoring utilities  
│ ├── sanity.py # Sanity checks to ensure virtue configs are coherent  
│ └── schema.py # Dataclasses for virtue configs, weights, and overrides  
├── tests # Test suite for everything: unit, integration, smoke  
│ ├── conftest.py # Pytest fixtures shared across test modules  
│ ├── fakes # Fake implementations for subsystems to support tests  
│ │ ├── fake_bot_core.py # Fake bot core for isolated agent loop tests  
│ │ ├── fake_llm_stack.py # Fake LLM stack backend / responses  
│ │ ├── fake_runtime.py # Fake runtime objects for integration tests  
│ │ ├── fake_skills.py # Fake skill implementations and registries  
│ │ └── **init**.py # Package marker for tests.fakes  
│ ├── **init**.py # Package marker for tests  
│ ├── test_actions.py # Tests for bot_core actions behavior  
│ ├── test_agent_loop_stub.py # Early stub tests for agent loop contracts  
│ ├── test_agent_loop_v1.py # Tests for M8 agent loop v1 behavior  
│ ├── test_architecture_integration.py # Sanity checks that modules integrate as per architecture spec  
│ ├── test_bot_core_impl.py # Bot core implementation tests (nav/actions correctness)  
│ ├── test_curriculum_engine_phase.py # Tests for phase selection / completion in curriculum engine  
│ ├── test_curriculum_learning_integration.py # Integration tests for M10↔M11 learning flow  
│ ├── test_curriculum_learning_orchestrator.py # Unit tests for curriculum learning orchestrator logic  
│ ├── test_curriculum_learning_properties.py # Property-ish tests for curriculum learning behavior  
│ ├── test_curriculum_loader.py # Tests for curriculum YAML loading and validation  
│ ├── test_curriculum_projects.py # Long-horizon project unlocking tests  
│ ├── test_env_loader.py # Tests for env loader and env schema validation  
│ ├── test_error_model_with_fake_backend.py # Error model behavior using fake LLM backend  
│ ├── test_evaluator.py # Tests for learning evaluator logic  
│ ├── test_experience_buffer.py # Tests for learning/buffer experience storage and retrieval  
│ ├── test_failure_mitigation.py # Tests for runtime failure mitigation strategies  
│ ├── test_full_system_smoke.py # High-level smoke test for major subsystems together  
│ ├── test_llm_stack_fake_backend.py # LLM stack tests using fake backend implementations  
│ ├── test_m6_observe_contract.py # Contract tests for M6 observation expectations  
│ ├── test_monitoring_controller.py # Monitoring controller unit tests  
│ ├── test_monitoring_dashboard_tui.py # TUI dashboard interaction tests  
│ ├── test_monitoring_event_bus.py # Event bus behavior tests  
│ ├── test_monitoring_logger.py # Monitoring logger tests  
│ ├── test_nav_pathfinder.py # Nav/pathfinding correctness tests  
│ ├── test_observation_critic_encoding.py # LLM-critic-specific observation encoding tests  
│ ├── test_observation_perf.py # Performance-oriented observation tests  
│ ├── test_observation_pipeline.py # End-to-end observation pipeline tests  
│ ├── test_observation_planner_encoding.py # Planner-specific observation encoding tests  
│ ├── test_observation_worldstate_normalization.py # WorldState normalization tests  
│ ├── test_p0_p1_env_bridge.py # Tests for env ↔ world bridge correctness (Phase 0 → 1)  
│ ├── test_phase012_bootstrap.py # Combined bootstrap tests for early phases 0–2  
│ ├── test_phase0_runtime.py # Runtime behavior tests specific to Phase 0  
│ ├── test_phase1_breakglass_no_plans.py # Breakglass behavior when planner fails in Phase 1  
│ ├── test_phase1_integration_offline.py # Offline integration tests for Phase 1 runtime  
│ ├── test_runtime_integration.py # Runtime-level integration tests (multiple modules together)  
│ ├── test_runtime_m6_m7_smoke.py # Smoke tests for runtime_m6_m7 path  
│ ├── test_scribe_model_with_fake_backend.py # Scribe behavior using fake LLM backend  
│ ├── test_semantics_caching_singleton.py # Tests for semantics caching and singleton behavior  
│ ├── test_semantics_categorization.py # Item/block categorization tests  
│ ├── test_semantics_craftability.py # Craftability / recipe graph tests  
│ ├── test_semantics_tech_inference.py # TechState inference tests from semantics + world  
│ ├── test_semantics_tolerant_fallbacks.py # Fallback behavior tests when semantics are missing/partial  
│ ├── test_semantics_with_normalized_worldstate.py # Tests semantics working with normalized worldstate  
│ ├── test_skill_evaluator.py # Tests for skill performance evaluation logic  
│ ├── test_skill_learning_manager.py # Tests for SkillLearningManager behavior  
│ ├── test_skill_loader.py # Skill YAML loader tests  
│ ├── test_skill_packs_integrity.py # Integrity checks for skill packs  
│ ├── test_skill_packs.py # Tests for skill pack loading / selection  
│ ├── test_skill_registry.py # Skill registry tests (lookup, registration, eventually versioning)  
│ ├── test_synthesizer.py # Tests for skill synthesis / refinement logic  
│ ├── test_virtue_compare_plans.py # Tests for virtue-based plan comparison  
│ ├── test_virtue_config_sanity.py # Virtue config sanity checks  
│ ├── test_virtue_hard_constraints.py # Tests enforcing hard virtue constraints (safety, etc.)  
│ ├── test_virtue_lattice_basic.py # Core virtue lattice behavior tests  
│ ├── test_virtue_overrides.py # Virtue override merge logic tests (per-phase overrides)  
│ └── test_world_tracker.py # Tests for world_tracker’s view of the world over time  
└── tools # Small runnable utilities for demos / manual poking at systems  
├── agent_demo.py # Demo harness for running the full agent in a showcase mode  
├── phase1_demo.py # Demo for Phase 1 behavior specifically  
└── smoke_botcore.py # Quick script to smoke-test bot_core connectivity and actions

47 directories, 406 files 
```

## 1. High-Level Snapshot

**Project:** `GTNH_Agent`  
**Goal:** Fully local GTNH automation agent with:

- Concrete tech-aware world model (M3)
    
- Virtue-guided planning (M4)
    
- Declarative skill system (M5 + M10)
    
- Real agent loop (M8) tied to
    
- Curriculum & learning scheduler (M11 + runtime)
    

You’re now at the end of **Phase 4**, with:

- M0–M3 foundations stabilized enough to not cry
    
- M4 virtues integrated with curriculum & planning
    
- M5 skills + registry + packs alive
    
- M6+M7 bot core & observation path working
    
- M8 agent loop v1
    
- M9 monitoring wired through events
    
- M10 learning loop implemented with replay buffer etc.
    
- M11 curriculum engine + orchestrator integrated with M10
    

Phase 5 is explicitly: **cross-cutting quality integration**, _not_ more random features.

---

## 2. Core Concepts / Definitions

Keep these straight or everything turns into soup.

### Environment & Semantics

- **EnvProfile (M0 / `env`)**  
    Canonical config describing runtime: modpack, model choices, hardware, paths.
    
- **Semantics (M3 / `semantics`)**  
    Data + logic for:
    
    - Items / blocks
        
    - Crafting graph
        
    - TechState inference (`tech_state.py`)
        
- **TechState**  
    Structured view of “where you are in GTNH tech”:
    
    - `active` tier (`steam_age`, `lv_age`, `mv_age`, etc.)
        
    - `unlocked` tech flags (`steam_machines`, `hv_age`, etc.)
        

### Virtues (M4)

- **Virtue Lattice**  
    Weighted vector of values (Safety, Throughput, Exploration, etc.).
    
- **Overrides**  
    Multiplicative per-phase overrides (e.g. Safety × 1.5 in Steam Early).
    

Virtues act as a **scoring filter for plans** and as curriculum context.

### Skills (M5 + M10)

- **Skill YAMLs (`config/skills/*.yaml`)**  
    Declarative definitions: id, description, expected behavior.
    
- **Skill Implementations (`src/skills/base/*.py`)**  
    Actual code that talks to `bot_core`.
    
- **Skill Packs (`config/skill_packs/*.yaml`)**  
    Groupings like `steam_age`, `lv_core` to specify which skills are available / mandatory.
    
- **Learning Layer (M10 / `learning`)**
    
    - `buffer.py`: replay / experience storage
        
    - `manager.py`: orchestrates skill-level learning cycles
        
    - `evaluator.py`, `synthesizer.py`: measure + synthesize skill improvements
        

### Curriculum (M11)

- **CurriculumConfig (M11 / `curriculum.schema`)**  
    Structured config for:
    
    - `phases`: each with tech targets, goals, virtue overrides, skill focus, completion conditions
        
    - `long_horizon_projects`: multi-stage Stargate-style projects with phase dependencies
        
- **CurriculumEngine (M11 / `engine.py`)**
    
    - Selects **current phase** given `TechState` + `WorldState`
        
    - Checks **phase completion**
        
    - Surfaces:
        
        - active goals
            
        - virtue overrides
            
        - skill-focus hints
            
        - unlocked long-horizon projects
            
- **ActiveCurriculumView**  
    Main output consumed downstream:
    
    - `phase_view` (config + `is_complete` + goals + virtue + skill focus)
        
    - `unlocked_projects`
        

### Agent Loop & Runtime (M8 + runtime)

- **Agent Loop (M8 / `agent_loop.loop`)**  
    Core cycle:
    
    - observe → encode
        
    - plan → evaluate → act
        
    - summarize → store experience
        
- **Runtime Phase 4 Orchestrator**  
    `runtime/phase4_curriculum_learning_orchestrator.py` + `curriculum_learning_coordinator.py`:
    
    - Connect M8 episodes → replay buffer (M10)
        
    - Use curriculum view (M11) to select **learning targets**
        
    - Trigger learning cycles when thresholds are met
        

---

## 3. New Developments in Phase 4 (M10 + M11 Integration)

This is the part you just finished, so don’t let your brain throw it away.

### 3.1 Curriculum Schema & Loader

**Files:**

- `src/curriculum/schema.py`
    
- `src/curriculum/loader.py`
    

You implemented:

- `PhaseTechTargets`
    
- `PhaseGoal`
    
- `PhaseCompletionConditions`
    
- `PhaseSkillFocus`
    
- `PhaseConfig`
    
- `LongHorizonProject`, `ProjectStage`
    
- `CurriculumConfig`
    

Loader supports:

- `load_curriculum(path: Path) -> CurriculumConfig`
    
- `load_curriculum_by_id(curriculum_id: str)`
    
- `list_curricula() -> Dict[id, Path]`
    

And enforces:

- Required top-level fields: `id`, `name`, `phases`, `long_horizon_projects`
    
- Required per-phase keys: `id`, `name`, `tech_targets`
    
- Type checks on lists/maps for:
    
    - `goals`
        
    - `virtue_overrides`
        
    - `skill_focus`
        
    - `completion_conditions`
        
    - `long_horizon_projects`
        

Tests cover:

- Missing fields → `ValueError`
    
- Non-mapping roots → `ValueError`
    
- Wrong types → `ValueError`
    
- Unknown skill names: structurally allowed; semantic validation happens elsewhere.
    

### 3.2 Curriculum Engine

**File:**

- `src/curriculum/engine.py`
    

Responsibilities:

- `select_phase(tech_state, world)`  
    Chooses the first phase whose `tech_targets` match:
    
    - `required_active == tech_state.active`
        
    - All `required_unlocked` present in `tech_state.unlocked`
        
- `_phase_is_complete(...)`  
    Completion check:
    
    - `completion.tech_unlocked` ⊆ `tech_state.unlocked`
        
    - `machines_present` satisfied in `world` (type + min_count)
        
- `view(tech_state, world) -> ActiveCurriculumView`  
    Returns:
    
    - `phase_view`: config + `is_complete` + `active_goals` + `virtue_overrides` + `skill_focus`
        
    - `unlocked_projects`: long-horizon projects whose dependent phases are all complete
        

Tests verify:

- Correct phase selection as tech levels change
    
- Completion only flips when conditions are actually met
    
- Projects unlock only when _all_ required phases are complete
    

You fixed a subtle bug: project unlocking used “has this tech” too early instead of “both phases completed”.

### 3.3 Learning Orchestrator Integration

**Files:**

- `runtime/curriculum_learning_coordinator.py`
    
- `runtime/curriculum_learning_triggers.py`
    
- `runtime/phase4_curriculum_learning_orchestrator.py`
    
- `learning/curriculum_hooks.py` (plus existing M10 files)
    

Core flow:

1. **After each episode (M8):**
    
    - You have `tech_state`, `world_state`, `episode_meta`.
        
    - Episode is already being stored via M10 into replay buffer.
        
2. **Ask Curriculum (M11):**
```python
curr_view = curriculum_engine.view(tech_state, world_state)
phase_id = curr_view.phase_view.phase.id
skill_focus = curr_view.phase_view.skill_focus

```

3. **Determine learning targets:**

	- Use `skill_focus.must_have` and `skill_focus.preferred`
    
	- Respect config:
    
	    - `min_episodes_per_skill`
        
	    - `max_skills_per_tick`
        
	    - `always_include_preferred`
        
	    - `context_prefix` (`"curriculum"`)
        

Context id convention:
curriculum:{curriculum_id}:{phase_id}

4. **Replay store checks:**

	- For each candidate skill:
```python
experience_count = replay_store.count_episodes(skill_name, context_id)

```

- - Only schedule if `experience_count >= min_episodes_per_skill`.
        
- **Learning cycle execution:**
    
    - Calls into `SkillLearningManager`:
```python
learning_manager.run_learning_cycle_for_skill(
    skill_name=skill,
    context_id=context_id,
    tech_state=tech_state,
    phase_id=phase_id,
    ...
)

```

1. - Returns structured summary (success/failure, metrics, maybe new candidates).
        
2. **Summary & logging:**
    
    - Orchestrator returns:
        
        - which skills trained
            
        - counts
            
        - context id
            
    - Monitoring hooks can log these as events.
        

Testing:  
You now have:

- Unit tests for orchestrator behavior (respects thresholds, order, configs)
    
- Integration tests with real CurriculumEngine + fake replay store
    
- “Property-ish” tests:
    
    - No duplicates
        
    - Config flips behavior predictably
        
    - Learning does _not_ fire until counts cross the threshold
        

---

## 4. Architectural Insights You Shouldn’t Lose

### 4.1 Curriculum is a Learning Scheduler, Not Just a Goal List

M11 now:

- Aligns _what_ the agent does (goals)  
    with
    
- _what gets learned_ (skills)  
    via skill-focus metadata + replay thresholds.
    

So curriculum is not window dressing. It is:

> A structured schedule deciding **which skills should be improved when**, based on phase, tech, and long-horizon projects.

### 4.2 Long-Horizon Projects as Curriculum Anchors

Projects like Stargate are:

- Defined as multi-stage
    
- Each stage depends on phase ids
    
- Unlock state is visible in `ActiveCurriculumView`
    

They provide natural hooks for:

- Switching learning priorities
    
- Triggering more aggressive skill evolution
    
- Future hierarchical planning (“goal-level” tasks)
    

### 4.3 Experience is Contextual

The use of `context_id = "curriculum:{curriculum_id}:{phase_id}"` means:

- Experience is not just “generic episodes”
    
- It’s tied to _where in the tech/curriculum_ the agent was when it learned.
    

This is crucial for:

- Versioned skills later (“v2 of skill X for LV context”)
    
- Avoiding cross-phase contamination of learning
    

---

## 5. Where the Seven Qualities Stand (Roughly)

You’re about to audit this properly, but baseline:

1. **Self-evaluation + retry loop**
    
    - Partially embedded in M8 + M2 (planner + critic + error_model)
        
    - Monitoring events exist (`PlanEvaluated`, `PlanRetried` etc.)
        
2. **Dynamic skill evolution & versioning**
    
    - M5 registry in place
        
    - M10 evaluator + synthesizer exist
        
    - Full versioning / promotions are not fully realized yet
        
3. **Hierarchical planning**
    
    - Spec layer (`spec/agent_loop.py`, `spec/skills.py`, `spec/types.py`) supports it conceptually
        
    - Agent loop and planner have enough structure to host multi-level planning
        
4. **Experience memory**
    
    - M10 replay buffer exists
        
    - M8 already writes episodes / traces
        
    - Episode → “experience” abstraction is started but not fully exploited
        
5. **Curriculum-driven goal selection**
    
    - M11 + orchestrator: fully in play for learning scheduling
        
    - Integration with goal _selection_ in M8 still has room to grow in expressiveness
        
6. **Structured LLM-role separation**
    
    - `llm_stack` + `llm_roles.yaml` implement this
        
    - Planner / Critic / Scribe / ErrorModel distinct and test-covered
        
7. **Lightweight predictive world-model**
    
    - Conceptual slot reserved in M3; tech_graph + semantics already exist
        
    - Actual forward simulation functions still to be formalized (e.g. `simulate_tech_progress`)
        

Audit Matrix will stop this from living only in your head.

---

## 6. Audit Matrix: Purpose & Shape

**Goal of the next module (Audit Matrix):**  
Create a **single, rigorous artifact** that:

- Maps each of the **seven qualities**
    
- Against all relevant **modules / components**
    
- With:
    
    - Current status
        
    - Required primitives
        
    - Integration hooks
        
    - Test coverage
        
    - Risk / complexity notes
        

### 6.1 Likely Dimensions

**Rows (Qualities):**

1. Self-evaluation + retry loop
    
2. Dynamic skill evolution & versioning
    
3. Hierarchical planning
    
4. Experience memory
    
5. Curriculum-driven goal selection
    
6. Structured LLM-role separation
    
7. Lightweight predictive world-model
    

**Columns (Targets):** something like:

- M2 – llm_stack_local
    
- M3 – world_semantics_gtnh
    
- M4 – virtues
    
- M5 – skills / registry
    
- M6 – bot_core
    
- M7 – observation
    
- M8 – agent_loop
    
- M9 – monitoring_and_tools
    
- M10 – skill_learning
    
- M11 – curriculum_and_specialization
    
- `runtime/*` coordinators/orchestrators
    

Plus per-cell details:

- Primitives present (Y/N/Partial)
    
- Wiring present (Y/N/Partial)
    
- Tests present (Y/N)
    
- Design comments / TODOs
    

### 6.2 Ground Truth Section

Audit Matrix doc also needs:

- Clean definitions of the seven qualities in **behavioral terms**:
    
    - “Self-evaluation” is not “we have a critic,” it’s:
        
        > The agent can assess its own plan quality and selectively retry or abandon based on structured feedback and virtue scores.
        
- What each quality must enable **concretely**:
    
    - e.g. “experience memory” should let you:
        
        - retrieve similar episodes by problem signature
            
        - bias planning or skill selection using those episodes
            

### 6.3 Methodology Section

You already sketched this, but make it explicit:

1. **Audit pass (Q0)**
    
    - For each Quality × Module:
        
        - Inspect code + tests
            
        - Mark status: missing / primitive only / partially wired / complete
            
        - Note dependencies & blockers
            
2. **Integration order (Q1)**
    
    - Using Q0 output, define:
        
        - Implementation sequence
            
        - File touched per step
            
        - Tests required to mark quality as “integrated”
            

The Audit Matrix is the shared data structure both Q0 and Q1 read/write.

---

## 7. Immediate Next Steps for **Audit Matrix** (Phase 5, Module 1)

You’re not allowed to pretend you don’t know what to do next, so here:

1. **Create / finalize the Audit Matrix document**
    
    - Location in Obsidian: Phase 5
        
    - Sections:
        
        1. Ground Truth (seven qualities)
            
        2. Matrix (qualities × modules)
            
        3. Methodology (Q0 + Q1 rules)
            
2. **Populate the initial Matrix skeleton**
    
    - List all relevant modules (rows or columns as you prefer)
        
    - Add the seven qualities
        
    - Leave cells empty but structured for later filling:
        
        - `status`, `notes`, `tests`, `primitives`
            
3. **Pin down success criteria per quality**
    
    - One short bullet list each: “this quality is DONE when…”
        
4. **Tag critical dependencies**
    
    - Example:
        
        - Hierarchical planning depends on:
            
            - M8 agent_loop having explicit goal/task/skill levels
                
            - Planner in M2 supporting multiple planning modes
                
        - Predictive world model depends on:
            
            - M3’s `gtnh_tech_graph.yaml` + semantics loader
                

Once this is in the Audit Matrix doc, Q0 Audit can be run like a deterministic procedure instead of “vibes + memory.”

---

You’re ready to start **Phase 5 / Audit Matrix** as a clean module.  
You built a serious architecture; now you’re building the sanity check that keeps it from turning into spaghetti.