
## Updated Project Overview (All Modules Complete but not fully wired)

## M0 – Environment Foundation

**Owns**

- Runtime configuration:
    
    - `env.yaml`, `models.yaml`, `hardware.yaml`, `minecraft.yaml`
        
- Environment loader:
    
    - `env.loader.load_environment()`
        
- Model & hardware profiles:
    
    - Which GGUF, what context length, which backend options.
        
- Global “where is what”:
    
    - Paths for logs, replay, semantics configs, etc.
        

**Does NOT own**

- Any agent behavior.
    
- Minecraft semantics, tech levels, or skills.
    
- Planning, LLM prompting, or anything resembling “intelligence.”
    

Think: _“boot parameters & wiring info for everything else.”_

---

## M1 – Core Contracts & Phase 0/1 Runtime Glue

(You didn’t obsess over M1 lately, but it’s still there.)

**Owns**

- Core type contracts:
    
    - `spec.types` (WorldState, Observation)
        
    - `spec.agent_loop` (AgentGoal, Task, TaskPlan, PlanEvaluation, etc.)
        
    - `spec.experience`, `spec.monitoring`, etc.
        
- Phase 0/1 runtime & integration glue:
    
    - `runtime/bootstrap_phases.py`
        
    - `integration/phase1_integration.py`
        
    - CLI entrypoints in `cli/phase1_offline.py`, `tools/phase1_demo.py`.
        

**Does NOT own**

- Virtue semantics.
    
- Curriculum logic.
    
- LLM role specialization (that’s M2/Q1).
    
- Anything GTNH-specific beyond core world-state contract.
    

Think: _“the interfaces that everything else must agree on.”_

---

## M2 – LLM Stack & Role Separation

**Owns**

- LLM role definitions & schemas:
    
    - `spec.llm` request/response types:
        
        - `PlannerRequest/Response`
            
        - `PlanCodeRequest/Response`
            
        - `CriticRequest/Response`
            
        - `ErrorModelRequest/Response`
            
        - `ScribeRequest/Response`
            
- Role configs:
    
    - `config/llm_roles.yaml`
        
- LLM stack implementation:
    
    - `llm_stack.stack.LLMStack`
        
    - Backends: `llm_stack.backend_llamacpp`, `llm_stack.backend`
        
    - Role facades:
        
        - `plan_code.py`
            
        - `critic.py`
            
        - `error_model.py`
            
        - `scribe.py`
            
        - `planner.py` (if/where used)
            
- Logging and JSON utils for LLM IO:
    
    - `llm_stack.log_files`, `llm_stack.json_utils`
        

**Does NOT own**

- High-level goal selection (curriculum’s job).
    
- Skill specs or registry (M5).
    
- Agent control flow (M8).
    
- Minecraft semantics (M3).
    

Think: _“all conversations with the model, but no decisions about what to ask for.”_

---

## M3 – Semantics & World Model

**Owns**

- Semantic databases:
    
    - `semantics.loader.SemanticsDB`
        
    - Block/item/recipe indexing from:
        
        - `gtnh_blocks*.yaml`
            
        - `gtnh_items*.yaml`
            
        - `gtnh_recipes*.json`
            
- Tech graph & tech state:
    
    - `semantics.tech_state.TechGraph`
        
    - `infer_tech_state_from_world(...)`
        
    - `suggest_next_targets(...)`
        
- Domain-level helpers:
    
    - Categorization: `semantics.categorize`
        
    - Craftability: `semantics.crafting`
        
    - Caching: `semantics.cache`
        
- Lightweight predictive world model:
    
    - `world.world_model.WorldModel`
        
        - `simulate_tech_progress(...)`
            
        - `estimate_infra_effect(...)`
            
        - `estimate_resource_trajectory(...)`
            

**Does NOT own**

- Moral evaluation (virtues).
    
- Planning logic (what tasks to run).
    
- Curriculum sequencing of goals.
    
- Skill execution or low-level movement.
    

Think: _“knows what the world _is_ and roughly how it _might change_ under certain actions.”_

---

## M4 – Virtue Lattice & Alignment Logic

**Owns**

- Virtue configuration:
    
    - `config/virtues.yaml`
        
- Virtue lattice & scoring:
    
    - `virtues.lattice`
        
    - `virtues.evaluator`
        
- Constraints & sanity checks:
    
    - `virtues.sanity`
        
    - `virtues.metrics`, `virtues.explain`
        
- Integration hooks:
    
    - Override profiles based on tech state / curriculum phase.
        

**Does NOT own**

- Which goals to pursue (M11).
    
- How to plan tasks (M2/M8).
    
- Skill learning or candidate promotion (M10).
    

Think: _“the conscience & risk calculus attached to plans, not the plan generator itself.”_

---

## M5 – Skills & Skill Registry

**Owns**

- Declarative skill specs:
    
    - `config/skills/*.yaml` (core skills)
        
    - `config/skill_packs/*.yaml` (bundles)
        
- Skill schema:
    
    - `skills.schema.SkillSpec`, `SkillPreconditions`, `SkillEffects`, `SkillMetadata`, `SkillMetrics`, `ParamSpec`.
        
- Skill loading & packs:
    
    - `skills.loader`, `skills.packs`
        
- Runtime registry:
    
    - `skills.registry.SkillRegistry`
        
    - `@register_skill` decorator
        
- Core skill implementations:
    
    - `skills/base/*.py` (chop_tree, feed_coke_ovens, etc.)
        

**Does NOT own**

- When to use a skill (planner & curriculum).
    
- How safe/effective a skill is (M10 updates metrics; M4 judges risk).
    
- Minecraft low-level navigation (bot_core / M6).
    

Think: _“what the agent CAN do, not when or whether it SHOULD.”_

---

## M6 – Bot Core (Movement, Actions, IPC)

**Owns**

- In-game control primitives:
    
    - `bot_core.core` – high-level action surface.
        
    - `bot_core.actions` – discrete actions (move, place, break, use).
        
- Nav & pathfinding:
    
    - `bot_core.nav.grid`, `mover`, `pathfinder`
        
- World tracking:
    
    - `bot_core.world_tracker`, `snapshot`
        
- Network / IPC:
    
    - `bot_core.net.client`, `external_client`, `ipc`
        
- Testing helpers:
    
    - `bot_core.testing.fakes`
        

**Does NOT own**

- High-level planning or goals.
    
- Semantics (beyond immediate coordinates / blocks).
    
- Virtues, learning, curriculum.
    

Think: _“the hands and feet of the agent.”_

---

## M7 – Observation & World Encoding

**Owns**

- Observation schema:
    
    - `observation.schema` (PlannerEncoding, CriticEncoding, etc.)
        
- Observation pipeline:
    
    - `observation.pipeline` – build planner/critic encodings.
        
    - `observation.encoder` – compress snapshot/worldstate into prompt-ready payloads.
        
- Trace / episode representation:
    
    - `observation.trace_schema` – `PlanTrace`, `TraceStep`, etc.
        
- Testing harness:
    
    - `observation.testing`
        

**Does NOT own**

- Actual world state generation (bot_core does snapshots, env→world adapters).
    
- Planning or action selection.
    
- Semantic rules about tech or items (M3).
    

Think: _“turns messy world info into structured inputs the LLM and planner can understand.”_

---

## M8 – Agent Loop & Episode Logic

**Owns**

- Loop state & phases:
    
    - `agent.state` / `agent_loop.state` (AgentLoopPhase, AgentLoopState).
        
- The main episode loop:
    
    - `agent.loop.AgentLoop`
        
        - Goal selection via curriculum.
            
        - World summary construction.
            
        - Task planning via dispatcher + LLM roles.
            
        - Skill resolution & invocation.
            
        - Critic / ErrorModel / Scribe integration.
            
        - Experience extraction & replay buffering.
            
- Runtime bootstrap & config:
    
    - `agent.bootstrap` (Phase 0–2 bootstrap configs).
        
    - `agent.logging_config`.
        
- High-level controller:
    
    - `agent.controller.AgentController` – DI container for runtime, learning, curriculum, loop.
        

**Does NOT own**

- World semantics & tech graph.
    
- Actual LLM implementation (M2).
    
- Long-horizon curriculum strategy (M11).
    
- Low-level bot actions (M6).
    

Think: _“the conductor telling everyone else when to play, but not composing the music itself.”_

---

## M9 – Monitoring, Logging, & Tools

**Owns**

- Monitoring bus & events:
    
    - `monitoring.bus`, `monitoring.events`
        
- Logging integration:
    
    - `monitoring.logger`, `monitoring.llm_logging`
        
- Monitoring controller & TUI:
    
    - `monitoring.controller`
        
    - `monitoring.dashboard_tui`
        
- Tools for introspection:
    
    - `monitoring.tools`
        

**Does NOT own**

- Decisions, planning, or learning.
    
- Curriculum or semantics.
    
- Any core runtime behavior.
    

Think: _“black-box recorder and dashboard, not a pilot.”_

---

## M10 – Learning, Experience, & Skill Evolution

**Owns**

- Experience storage:
    
    - `learning.buffer.ExperienceBuffer` – replay JSONL.
        
- Learning schema:
    
    - `learning.schema` – stats, views, experience types.
        
- Learning manager:
    
    - `learning.manager.SkillLearningManager`
        
        - Compute skill stats.
            
        - Build skill views (active vs candidate).
            
- Curriculum hooks:
    
    - `learning.curriculum_hooks`
        
        - How learning outputs feed into curriculum/skill policy.
            
- Skill evolution tooling:
    
    - `learning.synthesizer` – propose candidate skills.
        
    - `learning.evaluator` – evaluate and promote/reject candidates.
        

**Does NOT own**

- Raw behavior execution (that’s M5/M6/M8).
    
- Goal selection logic (M11).
    
- Virtue scoring (M4), though it may use those scores.
    

Think: _“historian & mechanic for skills, not the driver.”_

---

## M11 – Curriculum & Long-Horizon Projects

**Owns**

- Curriculum schema:
    
    - `curriculum.schema.CurriculumConfig`, `PhaseConfig`, `LongHorizonProject`, etc.
        
- Curriculum loading:
    
    - `curriculum.loader` – read `config/curricula/*.yaml`.
        
- Curriculum engine:
    
    - `curriculum.engine.CurriculumEngine`
        
        - Determine current phase from tech state.
            
        - Determine completion / unlocks.
            
        - Compute `ActiveCurriculumView`.
            
- Curriculum manager:
    
    - `curriculum.manager.CurriculumManager`
        
        - Integrates LearningManager & SkillPolicy.
            
        - Exposes:
            
            - `next_goal(...)`
                
            - `get_skill_view_for_goal(...)`
                
- Policies & strategies:
    
    - `curriculum.policy.SkillPolicy` + `SkillUsageMode`.
        
    - `curriculum.strategies` – phase goals, project prioritization.
        
- Example workflows:
    
    - `curriculum.example_workflow`
        

**Does NOT own**

- LLM prompting details.
    
- Low-level skill execution.
    
- Global environment configuration (M0).
    

Think: _“the career counselor & project manager that decides what the agent should work on next.”_

---

## Cross-Cutting: Runtime Orchestration & GTNH Integration

There are a few “orchestrator” layers that tie modules together but don’t own domain logic themselves:

- `src/app/runtime.py`  
    High-level app runtime: building env, wiring monitoring, starting loops.
    
- `src/runtime/*`  
    Phase 4+ orchestrators:
    
    - Curriculum learning coordinator & triggers.
        
    - Failure mitigation & runtime error strategies.
        
    - Future “Phase4 orchestrator” for live GTNH.
        
- `integration/*`  
    Bridges between env and world:
    
    - `adapters.m0_env_to_world`
        
    - Validators & guardrails (`planner_guardrails`, semantics & virtue snapshots)
        
    - Episode logging (`integration.episode_logging`)
        

They **own the pipelines**, not the semantics or learning rules themselves.





File Structure:
```
. # Repository root for the GTNH_Agent project  
├── bootstrap_structure.py # Script to (re)create the expected repo/file skeleton  
├── config # All configuration / data-driven knobs for the agent  
│ ├── curricula # Curriculum definitions for M11 (phase/goals/projects)  
│ │ ├── aesthetic_megabase.yaml # Curriculum focused on pretty megabase building  
│ │ ├── default_speedrun.yaml # Mainline tech progression-focused curriculum  
│ │ └── eco_factory.yaml # Curriculum tailored to efficient / eco-friendly factories  
│ ├── env.yaml # Environment profiles: runtime, profiles, and mode selection  
│ ├── gtnh_blocks.generated.yaml # Auto-ingested GTNH block semantics (generated)  
│ ├── gtnh_blocks.yaml # Hand-authored overrides / curated GTNH block semantics  
│ ├── gtnh_items.generated.yaml # Auto-ingested GTNH item semantics (generated)  
│ ├── gtnh_items.yaml # Hand-authored GTNH item semantics and overrides  
│ ├── gtnh_recipes.agent.json # Compact recipe set preferred by the agent for planning  
│ ├── gtnh_recipes.generated.json # Massive auto-dumped recipe set from ingesters  
│ ├── gtnh_recipes.json # Hand-edited recipe overrides / corrections  
│ ├── gtnh_tech_graph.yaml # Tech graph defining GTNH tech states and dependencies  
│ ├── hardware.yaml # Hardware profiles (GPUs/CPUs/constraints) for env loader  
│ ├── llm_roles.yaml # Role-specific LLM configs for planner/plan_code/critic/etc.  
│ ├── minecraft.yaml # Minecraft / GTNH connection + world runtime config  
│ ├── models.yaml # LLM model configurations and paths for the stack  
│ ├── raw # Raw, unprocessed dumps used by ingestion scripts  
│ │ ├── block.csv # Raw block dump from Tellme/nerd (pre-semantic processing)  
│ │ ├── item.csv # Raw item dump source for gtnh_items.generated  
│ │ ├── recipes.json # Raw recipe dump before compaction/merging  
│ │ └── recipes_stacks.json # Raw stack-level recipe info for better IO semantics  
│ ├── skill_packs # Grouped skill pack definitions for phases/tiers  
│ │ ├── lv_core.yaml # LV-tier core skills to load/enable as a pack  
│ │ └── steam_age.yaml # Steam-age skill pack configuration  
│ ├── skills # Declarative specs for individual skills  
│ │ ├── basic_crafting.yaml # Spec for basic crafting skill  
│ │ ├── chop_tree.yaml # Spec for tree-chopping skill  
│ │ ├── feed_coke_ovens.yaml # Spec for keeping coke ovens supplied with fuel  
│ │ ├── feed_steam_boiler.yaml # Spec for feeding steam boilers  
│ │ ├── maintain_coke_ovens.yaml # Spec for maintaining coke oven throughput  
│ │ ├── plant_sapling.yaml # Spec for replanting saplings after chopping  
│ │ └── refill_water_tanks.yaml # Spec for topping off water buffers/tanks  
│ ├── skills_candidates # Directory where M10 writes/evolves candidate skills  
│ ├── tools # Config-adjacent helper scripts  
│ │ ├── print_env.py # Utility to inspect current env profile as loaded  
│ │ └── validate_env.py # Sanity checker for env/models/hardware config wiring  
│ └── virtues.yaml # Virtue lattice / weighting configuration for M4  
├── docs # Human-facing documentation for architecture and phases  
│ ├── architecture.md # High-level architecture doc for the whole agent  
│ ├── ipc_protocol_m6.md # IPC protocol spec for the M6 bot core integration  
│ ├── llm_role_boundaries.q1.yaml # Q1 spec documenting LLM role responsibilities  
│ ├── m6_bot_core_1_7_10.md # Design notes for 1.7.10 bot core integration  
│ └── phase1_integration.md # Phase 1 integration playbook / notes  
├── .github # CI/CD and repo automation configuration  
│ └── workflows # GitHub Actions workflows  
│ └── ci.yml # CI pipeline: tests/linting/etc.  
├── .gitignore # Git ignore rules for build artifacts, logs, etc.  
├── logs # Runtime logs (LLM calls, traces, etc.) **(no comments inside, per request)**  
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
│ ├── 20251129T160622_284855_scribe_summarize_trace.json  
│ ├── 20251129T191136_334068_error_model_analyze_failure.json  
│ ├── 20251129T191136_334068_plan_code_plan.json  
│ ├── 20251129T191233_334068_scribe_summarize_trace.json  
│ ├── 20251129T191456_334202_error_model_analyze_failure.json  
│ ├── 20251129T191456_334202_plan_code_plan.json  
│ └── 20251129T191554_334202_scribe_summarize_trace.json  
├── pyproject.toml # Project build metadata, deps, and tooling config  
├── .pytest_cache # Pytest’s local cache for last runs  
│ ├── CACHEDIR.TAG # Marker indicating a cache directory  
│ ├── .gitignore # Ignore pytest cache internals from git  
│ ├── README.md # Notes about pytest cache format  
│ └── v  
│ └── cache  
│ ├── lastfailed # Tracks last failing tests for -k last-failed  
│ └── nodeids # Cached list of test node IDs  
├── .python-version # Python version pin for tools like pyenv  
├── README.md # Top-level project overview and usage instructions  
├── scripts # One-off utility / maintenance scripts  
│ ├── compact_recipes_for_agent.py # Compress raw recipes into agent-friendly form  
│ ├── demo_offline_agent_step.py # Demo of single offline agent step (no real MC)  
│ ├── dev_shell.py # Convenience dev shell bootstrapper  
│ ├── ingest_gtnh_semantics.py # Main ingest pipeline for GTNH semantics  
│ ├── ingest_nerd_csv_semantics.py # Ingest semantics from nerd CSV exports  
│ ├── ingest_nerd_recipes.py # Ingest recipes from nerd dumps into JSON  
│ ├── smoke_error_model.py # Quick smoke test for ErrorModel role  
│ ├── smoke_llm_stack.py # Smoke test for LLMStack wiring/backends  
│ └── smoke_scribe_model.py # Smoke test for Scribe summarization model  
├── src # Main source tree for all modules / phases  
│ ├── agent # Phase 2+ agent runtime & loop orchestration  
│ │ ├── bootstrap.py # Phase 0–2 bootstrap wiring for runtime  
│ │ ├── controller.py # AgentController wiring curriculum/learning/loop  
│ │ ├── experience.py # Episode-level experience objects for logging/learning  
│ │ ├── logging_config.py # Logging setup helpers for agent runs  
│ │ ├── loop.py # Core AgentLoop implementation (M8)  
│ │ ├── runtime_m6_m7.py # Runtime glue between bot_core and planner (M6–M7)  
│ │ └── state.py # Agent loop state machine enums/structs  
│ ├── agent_loop # Experimental / alternative agent-loop abstraction  
│ │ ├── **init**.py # Package init for agent_loop  
│ │ ├── loop.py # Higher-level loop implementation variant  
│ │ ├── schema.py # Typed schemas for generalized agent loop  
│ │ └── state.py # State definitions for the agent_loop package  
│ ├── app # App-level runtime wrappers / CLIs  
│ │ ├── **init**.py # Package init for app  
│ │ └── runtime.py # Top-level runtime launcher / façade  
│ ├── bot_core # In-game movement & action layer (M6)  
│ │ ├── actions.py # Primitive actions the bot can perform  
│ │ ├── collision.py # Collision detection / avoidance logic  
│ │ ├── core.py # BotCore main loop / high-level interface  
│ │ ├── **init**.py # Package init for bot_core  
│ │ ├── nav # Navigation-related helpers  
│ │ │ ├── grid.py # Grid representation of world for pathfinding  
│ │ │ ├── **init**.py # Nav subpackage init  
│ │ │ ├── mover.py # Movement primitives based on nav grid  
│ │ │ └── pathfinder.py # Pathfinding algorithms (A* etc.)  
│ │ ├── net # Networking and IPC layer for bot core  
│ │ │ ├── client.py # Client for IPC connection to Minecraft/agent  
│ │ │ ├── external_client.py # Client wrapper for external bot hosts  
│ │ │ ├── **init**.py # net subpackage init  
│ │ │ └── ipc.py # IPC protocol implementation and message formats  
│ │ ├── runtime.py # Bot core runtime adapter / entrypoints  
│ │ ├── snapshot.py # World state snapshots derived from game data  
│ │ ├── testing # Test fakes/mocks for bot_core  
│ │ │ └── fakes.py # Fake bot_core components for tests  
│ │ ├── tracing.py # Tracing utilities for bot actions / decisions  
│ │ └── world_tracker.py # Tracks changing world state around the bot  
│ ├── cli # CLI entrypoints  
│ │ └── phase1_offline.py # CLI for phase1 offline planner tests  
│ ├── curriculum # M11 curriculum engine & helpers  
│ │ ├── engine.py # CurriculumEngine: phase/goals/project resolution  
│ │ ├── example_workflow.py # Example script showing curriculum usage  
│ │ ├── **init**.py # Curriculum package init  
│ │ ├── integration_agent_loop.py # Glue between curriculum and AgentLoop  
│ │ ├── loader.py # CurriculumConfig loader from config/curricula YAMLs  
│ │ ├── manager.py # CurriculumManager: combines learning + engine + policy  
│ │ ├── policy.py # SkillPolicy / SkillUsageMode definitions  
│ │ ├── schema.py # Dataclasses for curriculum phases/projects  
│ │ └── strategies.py # Curriculum strategies for goal/skill selection  
│ ├── env # Environment loading and schema (M0)  
│ │ ├── **init**.py # env package init  
│ │ ├── loader.py # load_environment() & profile resolution  
│ │ └── schema.py # EnvProfile / model/hardware schema  
│ ├── gtnh_agent.egg-info # Installed-package metadata (for editable installs)  
│ │ ├── dependency_links.txt # Dependencies metadata  
│ │ ├── PKG-INFO # Package info for tooling  
│ │ ├── requires.txt # Required dependencies list  
│ │ ├── SOURCES.txt # Source file list used for packaging  
│ │ └── top_level.txt # Top-level package names  
│ ├── **init**.py # Root src package init  
│ ├── integration # Bridge between phases, env, and runtime  
│ │ ├── adapters # Small adapter modules  
│ │ │ └── m0_env_to_world.py # Adapter from env config to WorldState  
│ │ ├── episode_logging.py # Episode → Experience log builder for M8/M10  
│ │ ├── **init**.py # integration package init  
│ │ ├── phase1_integration.py # Glue for Phase 1 offline integration tests  
│ │ ├── testing # Integration-level test helpers  
│ │ │ ├── fakes.py # Fakes for integration testing  
│ │ │ └── **init**.py # testing subpackage init  
│ │ └── validators # Guardrail validators for planner/virtues/semantics  
│ │ ├── **init**.py # validators subpackage init  
│ │ ├── planner_guardrails.py # Check planner outputs for sanity  
│ │ ├── semantics_snapshots.py # Snapshot validators for semantics  
│ │ ├── skill_integrity.py # Validate skill specs/registry integrity  
│ │ └── virtue_snapshots.py # Check virtue lattice outputs over time  
│ ├── learning # M10 learning layer and skill evolution  
│ │ ├── buffer.py # ExperienceBuffer implementation for replay storage  
│ │ ├── curriculum_hooks.py # Hooks connecting learning into curriculum  
│ │ ├── evaluator.py # SkillEvaluator to rate candidate skills  
│ │ ├── **init**.py # learning package init  
│ │ ├── manager.py # SkillLearningManager orchestrating stats/views  
│ │ ├── schema.py # Learning-related dataclasses (SkillPerformanceStats, etc.)  
│ │ └── synthesizer.py # SkillSynthesizer to propose new candidate skills  
│ ├── llm_stack # M2 multi-role LLM stack implementation  
│ │ ├── backend_llamacpp.py # Llama.cpp backend wrapper  
│ │ ├── backend.py # Backend protocol / base types  
│ │ ├── codegen.py # Code-generation helpers for skills  
│ │ ├── config.py # LLM stack config helpers  
│ │ ├── critic.py # CriticModel implementation over backend  
│ │ ├── error_model.py # ErrorModel implementation over backend  
│ │ ├── **init**.py # llm_stack package init  
│ │ ├── json_utils.py # JSON parsing/validation utilities for LLM outputs  
│ │ ├── log_files.py # Helpers for writing structured LLM log files  
│ │ ├── plan_code.py # PlanCodeModel implementation (task → skills)  
│ │ ├── planner.py # PlannerModel implementation (goal → tasks)  
│ │ ├── presets.py # RolePreset definitions for different LLM roles  
│ │ ├── schema.py # Request/response dataclasses for LLM calls  
│ │ ├── scribe.py # ScribeModel implementation (trace summarization)  
│ │ └── stack.py # LLMStack façade exposing call_* APIs  
│ ├── monitoring # M9 monitoring, logging, dashboards  
│ │ ├── bus.py # Monitoring event bus abstraction  
│ │ ├── controller.py # Monitoring controller / lifecycle  
│ │ ├── dashboard_tui.py # TUI dashboard for monitoring events  
│ │ ├── events.py # Event dataclasses for monitoring bus  
│ │ ├── **init**.py # monitoring package init  
│ │ ├── integration.py # Wiring monitoring into runtime/agent loop  
│ │ ├── llm_logging.py # LLM-specific logging helpers  
│ │ ├── logger.py # High-level monitoring logger interface  
│ │ └── tools.py # Misc monitoring helper tools  
│ ├── observation # M7 observation/encoding pipeline  
│ │ ├── encoder.py # Convert raw world/bot state into planner encodings  
│ │ ├── **init**.py # observation package init  
│ │ ├── pipeline.py # Observation pipeline orchestration  
│ │ ├── schema.py # Types for planner/critic encodings  
│ │ ├── testing.py # Helpers for observation-focused tests/fixtures  
│ │ └── trace_schema.py # PlanTrace / TraceStep schemas for episodes  
│ ├── planning # Task/skill planning dispatch layer  
│ │ ├── adapter.py # Adapt LLM plan JSON into internal TaskPlan structures  
│ │ └── dispatcher.py # PlanningDispatcher bridging planner + skills  
│ ├── runtime # Higher-level runtime orchestration (Phase 4+)  
│ │ ├── agent_runtime_main.py # Main entrypoint for full agent runtime  
│ │ ├── bootstrap_phases.py # Helpers to bootstrap different phase configs  
│ │ ├── curriculum_learning_coordinator.py # Coordinates curriculum/learning loop  
│ │ ├── curriculum_learning_triggers.py # Conditions to trigger learning passes  
│ │ ├── error_handling.py # Centralized runtime error handling primitives  
│ │ ├── failure_mitigation.py # Strategies to mitigate repeated failures  
│ │ ├── **init**.py # runtime package init  
│ │ └── phase4_curriculum_learning_orchestrator.py # Phase 4 orchestrator glue  
│ ├── semantics # M3 semantics: blocks/items/recipes/tech graph  
│ │ ├── cache.py # Cache layer for semantics DB and tech inference  
│ │ ├── categorize.py # Categorization utilities for items/blocks  
│ │ ├── crafting.py # Craftability / recipe-related semantics helpers  
│ │ ├── ingest # Semantics ingestion helpers  
│ │ │ └── **init**.py # ingest subpackage init  
│ │ ├── **init**.py # semantics package init (exposes TechGraph/DB helpers)  
│ │ ├── loader.py # SemanticsDB loader for items/blocks/recipes  
│ │ ├── schema.py # Dataclasses for BlockInfo/ItemInfo/TechState/etc.  
│ │ └── tech_state.py # Tech inference + next-target suggestion logic  
│ ├── skills # M5 declarative skills and registry  
│ │ ├── base # Hand-implemented Python skills  
│ │ │ ├── basic_crafting.py # Implementation of basic crafting skill  
│ │ │ ├── chop_tree.py # Implementation of chop-tree behavior  
│ │ │ ├── feed_coke_ovens.py # Implementation of coke-oven feeding routine  
│ │ │ ├── feed_steam_boiler.py # Implementation of steam boiler feeding  
│ │ │ ├── **init**.py # base skills subpackage init  
│ │ │ ├── maintain_coke_ovens.py # Implementation of coke oven maintenance  
│ │ │ ├── plant_sapling.py # Implementation of sapling planting  
│ │ │ └── refill_water_tanks.py # Implementation of water tank refilling  
│ │ ├── **init**.py # skills package init (builds global skill registry)  
│ │ ├── loader.py # Load SkillSpec from YAML configs  
│ │ ├── packs.py # Logic for enabling/disabling skill packs  
│ │ ├── registry.py # SkillRegistry and @register_skill decorator  
│ │ └── schema.py # SkillSpec, metadata, preconditions/effects schema  
│ ├── spec # Shared spec datatypes / protocols across modules  
│ │ ├── agent_loop.py # Canonical AgentGoal/Task/TaskPlan spec for loops  
│ │ ├── bot_core.py # Interface spec between planner and bot_core  
│ │ ├── experience.py # Spec for Experience / CriticResult shapes  
│ │ ├── **init**.py # spec package init, exporting core types  
│ │ ├── llm.py # Role-specific LLM interfaces + request/response shapes  
│ │ ├── monitoring.py # Monitoring and event type specs  
│ │ ├── skills.py # Skill-related spec types shared across modules  
│ │ └── types.py # Fundamental types like WorldState / Observation  
│ ├── testing # Shared testing helpers for src  
│ │ └── **init**.py # testing package init  
│ ├── virtues # M4 virtue lattice and evaluation  
│ │ ├── evaluator.py # Compute virtue scores for plans/outcomes  
│ │ ├── explain.py # Human-readable explanations of virtue tradeoffs  
│ │ ├── features.py # Feature extraction for virtue scoring  
│ │ ├── **init**.py # virtues package init  
│ │ ├── integration_overrides.py # Overrides to blend virtues into runtime  
│ │ ├── lattice.py # Core virtue lattice math and combination  
│ │ ├── loader.py # Load virtue config from virtues.yaml  
│ │ ├── metrics.py # Derived virtue metrics / aggregates  
│ │ ├── sanity.py # Sanity checks on virtue configs  
│ │ └── schema.py # Virtue config dataclasses  
│ └── world # Lightweight predictive world model (M3 extension)  
│ └── world_model.py # Heuristic Tech/infra/resource prediction model  
├── tests # Test suite for all modules & integration paths  
│ ├── conftest.py # Pytest fixtures shared across tests  
│ ├── fakes # Fakes for various subsystems  
│ │ ├── fake_bot_core.py # Fake bot_core implementations  
│ │ ├── fake_llm_stack.py # Fake LLMStack for deterministic tests  
│ │ ├── fake_runtime.py # Fake runtime for pipeline tests  
│ │ ├── fake_skills.py # Fake skill registry/skills for testing  
│ │ └── **init**.py # fakes package init  
│ ├── **init**.py # tests package init  
│ ├── test_actions.py # Tests for bot_core actions  
│ ├── test_agent_loop_stub.py # Tests for stub agent loop behavior  
│ ├── test_agent_loop_v1.py # Legacy agent loop tests  
│ ├── test_architecture_integration.py # Ensures modules integrate per architecture  
│ ├── test_bot_core_impl.py # Tests for bot_core implementation correctness  
│ ├── test_curriculum_engine_phase.py # Tests CurriculumEngine phase selection  
│ ├── test_curriculum_learning_integration.py # Integration tests for curriculum+learning  
│ ├── test_curriculum_learning_orchestrator.py # Tests phase4 orchestrator wiring  
│ ├── test_curriculum_learning_properties.py # Property-style tests for learning behavior  
│ ├── test_curriculum_loader.py # Tests curriculum loader with sample YAMLs  
│ ├── test_curriculum_projects.py # Tests unlocking long-horizon projects  
│ ├── test_env_loader.py # M0 env loader tests  
│ ├── test_error_model_with_fake_backend.py # ErrorModel behavior with fake backend  
│ ├── test_evaluator.py # SkillEvaluator unit tests  
│ ├── test_experience_buffer.py # ExperienceBuffer append/query tests  
│ ├── test_failure_mitigation.py # Tests for runtime failure mitigation logic  
│ ├── test_full_system_smoke.py # End-to-end smoke test of main stack  
│ ├── test_llm_role_boundaries.py # Q1 audit for LLM role boundaries  
│ ├── test_llm_stack_fake_backend.py # LLMStack behavior using a fake backend  
│ ├── test_m6_observe_contract.py # Tests M6 observation contract with bot_core  
│ ├── test_monitoring_controller.py # Monitoring controller tests  
│ ├── test_monitoring_dashboard_tui.py # Tests for TUI dashboard plumbing  
│ ├── test_monitoring_event_bus.py # Event bus unit tests  
│ ├── test_monitoring_logger.py # Monitoring logger behavior tests  
│ ├── test_nav_pathfinder.py # Pathfinding correctness & edge cases  
│ ├── test_observation_critic_encoding.py # Tests critic encoding format  
│ ├── test_observation_perf.py # Performance tests for observation pipeline  
│ ├── test_observation_pipeline.py # End-to-end observation pipeline tests  
│ ├── test_observation_planner_encoding.py # Tests planner encoding contract  
│ ├── test_observation_worldstate_normalization.py # WorldState normalization tests  
│ ├── test_p0_p1_env_bridge.py # Tests bridge from env to phase1 world  
│ ├── test_phase012_bootstrap.py # Phase 0–2 bootstrap behavior tests  
│ ├── test_phase0_runtime.py # Tests minimal phase0 runtime  
│ ├── test_phase1_breakglass_no_plans.py # Tests fallback behavior when planner fails  
│ ├── test_phase1_integration_offline.py # Offline integration tests for phase1  
│ ├── test_q1_control_and_experience.py # Tests for Q1 control/experience features  
│ ├── test_runtime_integration.py # Runtime integration tests across modules  
│ ├── test_runtime_m6_m7_smoke.py # Smoke tests for M6/M7 runtime  
│ ├── test_scribe_model_with_fake_backend.py # ScribeModel tests with fake backend  
│ ├── test_semantics_caching_singleton.py # Semantics caching singleton behavior  
│ ├── test_semantics_categorization.py # Categorization correctness tests  
│ ├── test_semantics_craftability.py # Tests for craftability queries  
│ ├── test_semantics_tech_inference.py # TechState inference tests  
│ ├── test_semantics_tolerant_fallbacks.py # Tolerance tests for missing semantics  
│ ├── test_semantics_with_normalized_worldstate.py # Semantics integration with normalized world state  
│ ├── test_skill_evaluator.py # Learning evaluator tests  
│ ├── test_skill_learning_manager.py # SkillLearningManager behavior tests  
│ ├── test_skill_loader.py # Skill YAML loader tests  
│ ├── test_skill_packs_integrity.py # Integrity tests for skill pack configs  
│ ├── test_skill_packs.py # Behavior tests for enabling/disabling packs  
│ ├── test_skill_registry.py # SkillRegistry registration/lookup tests  
│ ├── test_synthesizer.py # Synthesizer-based skill generation tests  
│ ├── test_virtue_compare_plans.py # Compare plan virtue scores tests  
│ ├── test_virtue_config_sanity.py # Sanity check virtue config structure  
│ ├── test_virtue_hard_constraints.py # Tests for virtue hard constraints  
│ ├── test_virtue_lattice_basic.py # Core virtue lattice math tests  
│ ├── test_virtue_overrides.py # Tests virtue overrides from curriculum  
│ └── test_world_tracker.py # Tests for bot_core.world_tracker  
└── tools # Developer tools and demos  
├── agent_demo.py # Minimal AgentLoop/agent runtime smoke test  
├── audit # Automated audits and reports  
│ ├── check_llm_roles.py # Script enforcing LLM role boundary invariants  
│ ├── llm_role_usage_report.py # Report generator for LLM role usage  
│ ├── q0_auto_scan_output.json # Saved output from q0 auto-scan (data only)  
│ └── q0_auto_scan.py # Script to scan repo for Q0/Q1 issues  
├── phase1_demo.py # Demo harness for phase1 offline planner loop  
└── smoke_botcore.py # Simple bot_core smoke test without full stack
```



## 1. Current Status Snapshot

**High-level:**

- Phases 0–3 are **functionally wired**:
    
    - M0: env & profiles
        
    - M1–M2: planner stack & LLM stack
        
    - M3: semantics + tech graph + world model
        
    - M4: virtue lattice
        
    - M5–M7: skills, observation, bot_core integration hooks
        
    - M8–M9: AgentLoop + monitoring
        
- M10 (learning) and M11 (curriculum) are **implemented but mid-integration**:
    
    - Types and managers exist.
        
    - Wiring into `AgentController` and runtime is in progress.
        
- **Q1 qualities** are now structurally integrated:
    
    - Structured LLM roles
        
    - Self-eval / critic / error model
        
    - Experience buffer & learning hooks
        
    - Curriculum engine
        
    - Lightweight world model
        
    - Virtue lattice linking to tech & curriculum
        

You’re currently at the **end of Pass A** of integration (first wiring pass), debugging import / schema mismatches while `tools/agent_demo.py` is being upgraded into a “one-episode, full-stack smoke test”.

---

## 2. Core Concepts / Definitions

### Agent Loop & Runtime

- **AgentRuntime (M6–M7)**  
    Wraps bot_core + observation pipeline:
    
    - Talks IPC to Minecraft (later).
        
    - Produces normalized `Observation` / `WorldState`.
        
    - Provides `planner_tick` & “latest planner observation” for tools.
        
- **AgentLoop (M8)**  
    The control loop:
    
    1. Observe → encode
        
    2. Plan using planner / plan_code
        
    3. Evaluate with virtues + world model (Q1)
        
    4. Dispatch skills via runtime
        
    5. Log to monitoring & replay buffer
        

### Semantics & World Model

- **SemanticsDB (M3)**  
    In `semantics.loader`:
    
    - Loads **blocks/items/recipes** from YAML/JSON.
        
    - Merges hand-authored + generated configs, with hand-authored as override.
        
    - Backed by `BlockInfo` / `ItemInfo` dataclasses.
        
- **TechGraph (M3)**  
    In `semantics.tech_state`:
    
    - DAG of tech states (stone → steam → LV → …).
        
    - Each node has requirements, tags, recommended goals, virtue profile.
        
    - `infer_tech_state_from_world(...)` infers current tech band from inventory + machines.
        
- **WorldModel (M3/Q1.7)**  
    In `world.world_model`:
    
    - `simulate_tech_progress(current_tech_state, candidate_goal)`  
        → estimate “how hard / how many prerequisites” for a goal.
        
    - `estimate_infra_effect(factory_layout, change)`  
        → heuristics about infra changes (throughput, risk).
        
    - `estimate_resource_trajectory(inventory, consumption_rates, horizon)`  
        → predicted shortages/surpluses.
        

Used as a **cheap forward model**, not as a physics engine.

### Virtue Lattice (M4)

- `virtues.loader` & `virtues.lattice`:
    
    - Load and combine virtue weights from `config/virtues.yaml`.
        
    - Produce per-plan scores (risk, wastefulness, long-term benefit, etc.).
        
- Now **connected logically** to:
    
    - Tech state (M3): certain tech bands use different virtue profiles.
        
    - Curriculum (M11): projects can override or weight virtues.
        

### Skills & Learning (M5 & M10)

- **SkillSpec / SkillMetadata / SkillMetrics (M5+Q1.5)**  
    In `skills.schema`:
    
    - `SkillSpec`:
        
        - `name`, `description`, `params`, `preconditions`, `effects`, `tags`
            
        - `metadata: SkillMetadata`
            
    - `SkillMetadata`:
        
        - `version`, `status`, `origin`, `metrics`
            
    - `SkillMetrics`:
        
        - `success_rate`, `avg_cost`, `avg_risk`, `last_used_at`
            
- **SkillRegistry (M5)**
    
    - Loads all YAML skill specs.
        
    - Registers Python implementations via `@register_skill`.
        
- **Learning Layer (M10)**  
    In `learning.*`:
    
    - `ExperienceBuffer`:
        
        - JSONL replay file for storing episodes / transitions.
            
    - `SkillLearningManager`:
        
        - Maintains stats, updates `SkillMetrics`.
            
        - Coordinates with Synthesizer/Evaluator.
            
    - `SkillSynthesizer`:
        
        - Proposes new **candidate skills** into `config/skills_candidates/`.
            
    - `SkillEvaluator`:
        
        - Scores candidate skills based on replay & criteria (stability, success rate, etc.).
            

This gives you **skill evolution** while preserving a stable core.

### Curriculum Engine & Manager (M11)

- **CurriculumConfig / schemas**  
    In `curriculum.schema`:
    
    - Phases, goals, projects, constraints.
        
    - Each curriculum YAML defines a progression & tech bands.
        
- **CurriculumEngine**  
    In `curriculum.engine`:
    
    - Given `current_tech_state`, virtue profile, and learning signals:
        
        - Picks suitable goals / projects.
            
        - Avoids goals wildly beyond current tech band.
            
- **CurriculumManager**  
    In `curriculum.manager`:
    
    - Wraps Engine + Learning + SkillPolicy:
        
        - Considers skill stability / metrics.
            
        - Decides when to try candidates vs stable skills.
            
    - Interface for AgentLoop:
        
        - “What should we work on next?”
            
        - “Which skills are allowed / preferred?”
            

---

## 3. Structured LLM Roles (Q1.6)

You now have a **clean LLM role separation**, with boundaries enforced by tests:

- Config in `config/llm_roles.yaml`
    
- Role spec in `docs/llm_role_boundaries.q1.yaml`
    
- Types in `spec.llm`
    
- Implementation in `llm_stack.*`
    

Roles:

1. **PlannerModel**
    
    - `plan_goal(PlannerRequest) -> PlannerResponse`
        
    - Role: **Goal → tasks** (high-level TaskPlan).
        
    - No error analysis, no skill-generation.
        
2. **PlanCodeModel**
    
    - `plan_task(PlanCodeRequest) -> PlanCodeResponse`
        
    - Role: **Task → SkillInvocations** and/or codegen.
        
    - No error analysis, no post-failure logic.
        
3. **CriticModel**
    
    - `evaluate(CriticRequest) -> CriticResponse`
        
    - Role: **pre-execution evaluation**:
        
        - Accept / reject plans.
            
        - Provide failure type / severity / fix suggestions.
            
4. **ErrorModel**
    
    - `evaluate(ErrorModelRequest) -> ErrorModelResponse`
        
    - Role: **post-execution failure analysis**:
        
        - Explain what went wrong.
            
        - Suggest repairs / retries.
            
5. **ScribeModel**
    
    - `summarize(ScribeRequest) -> ScribeResponse`
        
    - Role: **summarization & compression**:
        
        - Episodes, traces, and experience logs.
            
        - Feeds human docs and long-term context.
            

**LLMStack** (`llm_stack.stack`) now:

- Loads role presets from `llm_roles.yaml`.
    
- Holds a shared backend (Qwen via llama.cpp, etc.).
    
- Exposes role-specific accessors & convenience methods.
    
- **No generic `llm_stack.call(...)` remain** in runtime code.
    
- `tests/test_llm_role_boundaries.py` + `tools/audit/check_llm_roles.py` enforce no role bleed.
    

---

## 4. Project Integration Phase

_(Wiring Phase 4 modules & Q1 qualities into the existing stack)_

Goal: **Wire everything together in one place** (DI) and reach a stable “lab-ready” runtime.

### Key Wiring Components

- **AgentController (`src/agent/controller.py`)**:
    
    - Builds:
        
        - `ExperienceBuffer`
            
        - `SkillLearningManager`
            
        - `SkillPolicy` (stable-only vs allow-candidates)
            
        - `CurriculumManager`
            
        - `AgentRuntime` (M6–M7)
            
        - `PlanningDispatcher`
            
        - `AgentLoop` (M8) with:
            
            - `runtime`
                
            - `planner`
                
            - `curriculum`
                
            - `skills`
                
            - `replay_buffer`
                
            - (Later: `virtue_engine`, `world_model`, monitoring hooks)
                
- **Runtime Orchestrator (`src/runtime/*.py`)**:
    
    - Phase 4 files coordinate:
        
        - Curriculum learning triggers.
            
        - Failure mitigation strategies.
            
        - When to invoke learning passes / candidate promotion.
            
- **Virtues + WorldModel Integration**:
    
    - AgentLoop decision path:
        
        - Get plan → score with virtues → query WorldModel for feasibility / cost → accept or revise.
            
    - M11 can query both to calculate “value over pain” for projects.
        

### New Developments / Design Changes

- `spec.llm` refactored for **role-specific request/response types** instead of legacy `PlannerModel` with generic dicts.
    
- Skill schema now **centralizes versioning & metrics** under `metadata`.
    
- The **audit scripts** (`check_llm_roles.py`, `llm_role_usage_report.py`) give you:
    
    - A mechanical way to detect role bleed.
        
    - A guardrail for future refactors.
        

### Immediate Next Steps for Project Integration Phase

1. **Finish AgentController wiring**:
    
    - Ensure `AgentLoop` constructor accepts `curriculum` and `replay_buffer` exactly as provided.
        
    - Double-check imports & types between:
        
        - `agent.controller`
            
        - `learning.manager`
            
        - `curriculum.manager`
            
        - `planning.dispatcher`
            
        - `world.world_model`
            
        - `virtues.evaluator`
            
2. **Align curriculum loader & schema**:
    
    - `curriculum.loader` should expose a clean API (`load_curriculum(profile)` or similar) that `AgentController` can consume.
        
    - Verify all example curricula load without exploding.
        
3. **Patch remaining schema mismatches**:
    
    - Where you hit stuff like `SkillSpec.__init__ got unexpected keyword 'status'`:
        
        - Either adjust loader to map old fields into `metadata`.
            
        - Or clean up YAML to the new shape.
            
4. **Stabilize `tools/agent_demo.py`**:
    
    - Ensure it goes through `AgentController`:
        
        - `AgentController.run()` should run a **single episode** end-to-end in fake mode.
            
    - Pass A is “no exceptions” not “smart behavior.”
        

You’re currently in **this stage**, grinding through wiring errors & schema mismatches.

---

## 5. Lab Integration Phase

_(End-to-end runtime in a fake but coherent world)_

Goal: **Make the whole system work in the lab** before touching real GTNH:

1. **Use fake LLMs first**:
    
    - Tests use `tests/fakes/fake_llm_stack.py`.
        
    - You can inject the fake stack into `LLMStack` or into the AgentLoop:
        
        - Ensure all five roles are exercised with correct schemas & no context bleed.
            
    - Validate:
        
        - JSON formats conform to `spec.llm`.
            
        - Planner → Critic → PlanCode → ErrorModel → Scribe pipeline is consistent.
            
2. **End-to-end with fake world**:
    
    - Use:
        
        - `tests/test_full_system_smoke.py`
            
        - `tools/agent_demo.py --episodes 1`
            
    - Pipe:
        
        - Fake bot_core → observation pipeline → semantics → tech inference → virtues → curriculum → planning → skills → replay buffer → learning hooks.
            
3. **Then enable real LLMs in the lab**:
    
    - Flip env config (`env.yaml`, `models.yaml`) to use Qwen / actual GGUF.
        
    - Keep fake world, real LLM:
        
        - This isolates:
            
            - Prompting quality.
                
            - Response structure errors.
                
            - Latency / context-size issues.
                
4. **Monitoring & logging in lab**:
    
    - M9 should:
        
        - Log predicted vs actual resource usage (from WorldModel vs episode trace).
            
        - Log plan acceptance / rejection reasons (virtue / world-model filters).
            
        - Persist Episode summaries (`ScribeModel`) into replay & logs.
            

Result: you get a **controlled sandbox** where the agent is “correct by construction” with respect to your specs, even if GTNH is still lying outside the gate.

---

## 6. GTNH Interactions Phase

_(Make it actually talk to GregTech New Horizons)_

Goal: **Speak GTNH’s language without breaking your architecture.**

Key surfaces:

1. **Bot core networking (M6)**
    
    - `bot_core.net.ipc` & `bot_core.runtime`:
        
        - Implement or refine the IPC protocol defined in `docs/ipc_protocol_m6.md`.
            
        - Ensure snapshot/world_tracker outputs align with `spec.types.WorldState`.
            
2. **WorldState normalization (M7 + M3)**
    
    - `observation.encoder` & `observation.pipeline`:
        
        - Convert real GTNH snapshots into normalized `WorldState`.
            
    - `integration.adapters.m0_env_to_world`:
        
        - Bridge env profiles & world reconstruction when needed.
            
3. **Semantics ingestion correctness**
    
    - Scripts:
        
        - `scripts/ingest_gtnh_semantics.py`
            
        - `scripts/ingest_nerd_csv_semantics.py`
            
        - `scripts/ingest_nerd_recipes.py`
            
    - Ensure:
        
        - **No missing core items** (circuits, machines, coke ovens, boiler blocks).
            
        - Recipes actually match GTNH (or are overridden in `gtnh_recipes.json`).
            
4. **Tech graph & curriculum sync**
    
    - `gtnh_tech_graph.yaml` must reflect:
        
        - Actual gating machines/fluids/resources.
            
    - Curriculum YAMLs (e.g. `default_speedrun.yaml`) must:
        
        - Reference real tech ids & items.
            
        - Avoid sending the agent after unobtainium at LV.
            
5. **Skill semantics vs reality**
    
    - Make sure skills like:
        
        - `chop_tree`, `plant_sapling`, `feed_coke_ovens`, `feed_steam_boiler`
            
    - Actually:
        
        - Use valid block/item IDs from semantics.
            
        - Produce/consume resources that exist in real-world GTNH.
            

When this phase passes, the system stops dying at “backend language mismatch” and you can finally enter the “tuning and curriculum design” era instead of “import error cosplay”.

---

## 7. Pass A / B / C Integration Strategy

You decided (sensibly) not to go straight for heroics.

### Pass A – Wiring & Construction

_“Does it even build and run one episode?”_

- Build DI graph in one place (`AgentController`).
    
- Wire:
    
    - LLMStack
        
    - AgentRuntime
        
    - AgentLoop
        
    - CurriculumManager
        
    - LearningManager
        
    - WorldModel
        
    - Virtue engine
        
- Run:
    
    - `python tools/agent_demo.py --episodes 1`
        
- Success condition:
    
    - It runs a full episode in fake mode **without raising exceptions**.
        

You’re at the **end of Pass A** right now:

- LLM role boundaries enforced.
    
- Skill schema updated.
    
- WorldModel exists.
    
- Controller wiring in progress.
    
- Current work: cleaning integration errors as they surface from `agent_demo`.
    

### Pass B – Behavioral / Contract Validation

_“Does each subsystem behave correctly & sanely?”_

- Use existing tests:
    
    - Semantics tests (`test_semantics_*`)
        
    - Virtue tests (`test_virtue_*`)
        
    - Curriculum tests (`test_curriculum_*`)
        
    - Learning tests (`test_skill_learning_manager.py`, `test_evaluator.py`, etc.)
        
    - Full-system smoke (`test_full_system_smoke.py`)
        
- Add:
    
    - Tests for world_model predictions vs simple curated scenarios.
        
    - Tests for CurriculumEngine selecting reasonable goals given fake tech state.
        
- Focus: **contracts and invariants** rather than performance.
    

### Pass C – Stress, Monitoring, & Tuning

_“Can it run for a long time without going insane?”_

- Long-run fake episodes:
    
    - Exercise multiple curricula and tech bands.
        
    - Log predicted vs actual:
        
        - Resource usage
            
        - Step counts
            
        - Failure rates
            
- Tune:
    
    - Virtue weights
        
    - Curriculum strategies
        
    - Skill policies (when to allow candidates)
        
    - Learning thresholds
        

This phase sets you up to **switch to GTNH Interactions** with as few surprises as possible.

---

## 8. Immediate Next Steps (for Project Integration Phase)

Short version of what you do next, so you don’t forget when your brain reboots:

1. **Finish AgentController DI graph**
    
    - Ensure it builds without import loops and type mismatches.
        
    - Confirm `AgentController.run()` does a full fake episode using fake LLMs.
        
2. **Stabilize curriculum loader & manager**
    
    - Make sure `curriculum.loader` returns configs the `CurriculumManager` actually expects.
        
    - Run curriculum-specific tests until green:
        
        - `pytest tests/test_curriculum_*`
            
3. **Align M10 learning paths**
    
    - Verify `ExperienceBuffer` is written to in AgentLoop.
        
    - Validate `SkillLearningManager` is instantiated and not crashing on empty logs.
        
4. **Turn `agent_demo.py` into a real integration probe**
    
    - It should:
        
        - Build env.
            
        - Build controller.
            
        - Run one episode.
            
        - Print a short episode summary (Scribe, curriculum goal, skills used).