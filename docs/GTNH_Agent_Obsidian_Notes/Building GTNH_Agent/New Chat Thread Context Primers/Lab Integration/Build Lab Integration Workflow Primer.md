
# Project Overview

## P0 – Foundations (M0–M1)

### M0 – Environment Foundation

**What M0 owns**

- Runtime & hardware configuration:
    
    - `config/env.yaml`, `minecraft.yaml`, `hardware.yaml`, `models.yaml`, `llm_roles.yaml`.
        
- Environment schema & loader:
    
    - `env/schema.py`, `env/loader.py`.
        
- Basic profile selection & validation:
    
    - “profile_name”, model paths, runtime modes (offline / lab / future live).
        
- Bridge from env config → world abstraction:
    
    - `integration.adapters.m0_env_to_world` type shape.
        

**What M0 does _not_ own**

- No LLM prompting or role logic (that’s M2 / llm_stack).
    
- No game semantics (items, blocks, tech tiers → M3).
    
- No agent loop or curriculum decisions (M8/M11).
    
- No learning / experience persistence (M5/M10).
    

**Capabilities M0 affords**

- The agent can:
    
    - Bootstrap a **consistent runtime profile** (which models, which env, which log dirs).
        
    - Load and validate which LLM roles exist and which backend to use.
        
    - Run in different operating modes (offline test vs future live agent) _without_ changing code.
        

**Direct integrations**

- **M1**: runtime bootstrap (`runtime/bootstrap_phases.py`, `agent/bootstrap.py`) reads env profile & passes it into the runtime.
    
- **M2**: llm_stack config (`llm_stack/config.py`) reads `models.yaml` + `llm_roles.yaml`.
    
- **M3**: env → world mapping (adapter uses env profile to pick semantics / tech graph).
    
- **M8**: AgentLoop config (episode limits, debug flags) can be read from env profile and passed down.
    

---

### M1 – Runtime & Process Shell (Phase 0/1 runtime)

**What M1 owns**

- Orchestration entrypoints:
    
    - `runtime/agent_runtime_main.py`
        
    - `runtime/bootstrap_phases.py`
        
- Phase bootstrap logic:
    
    - Wiring P0/P1 flows for tests like `test_phase0_runtime`, `test_phase1_integration_offline`.
        
- Process-level concerns:
    
    - Logging setup (`agent/logging_config.py`), seeding random, basic CLI / flags.
        

**What M1 does _not_ own**

- No game semantics or tech state (M3).
    
- No LLM logic or prompts (M2).
    
- No skill logic or learning decisions (M5/M10).
    
- No detailed AgentLoop control (M8 owns the loop itself).
    

**Capabilities M1 affords**

- The agent can:
    
    - Start a **full offline pipeline** from CLI.
        
    - Choose which “phase mode” to run (e.g. P1 offline integration vs later lab modes).
        
    - Run smoke tests / demos via scripts like `scripts/phase1_offline.py`.
        

**Direct integrations**

- **M0**: reads env profile, passes config into downstream runtime.
    
- **M2**: initializes LLM stack via `llm_stack.stack.LLMStack` with role presets from config.
    
- **M3**: ensures semantics loader is initialized during runtime bootstrap as needed.
    
- **M6/M7**: in P1 offline tests, plugs in fake BotCore / fake observation pipeline.
    
- **M8**: constructs `AgentLoop` / `AgentRuntime` in integration flows and hands control to it.
    

---

## P1 – Cognitive Stack & Knowledge (M2–M5)

### M2 – LLM Stack & Role Contracts

**What M2 owns**

- LLM role interfaces & schemas:
    
    - `spec/llm.py` (Planner, PlanCode, Critic, ErrorModel, Scribe contracts).
        
- LLMStack implementation:
    
    - `llm_stack/stack.py`, `backend.py`, `backend_llamacpp.py`, `presets.py`.
        
- Individual role wrappers:
    
    - `planner.py`, `plan_code.py`, `critic.py`, `error_model.py`, `scribe.py`.
        
- JSON parsing, log files, error handling:
    
    - `llm_stack/json_utils.py`
        
    - `llm_stack/log_files.py`
        
    - `llm_stack/error_model.py` internals.
        

**What M2 does _not_ own**

- It does **not** decide _when_ to call LLMs (that’s M8).
    
- It does not own skills or curriculum content (M5/M11).
    
- It does not own tech-state semantics (M3).
    
- It does not own virtue math (M4).
    

**Capabilities M2 affords**

- The agent can:
    
    - Turn observations + goals into a JSON plan (Planner).
        
    - Turn tasks into skill invocations (PlanCode).
        
    - Pre-check plans for safety / quality (Critic).
        
    - Analyze failures post-execution (ErrorModel).
        
    - Summarize episodes / traces for memory (Scribe).
        

**Direct integrations**

- **M0**: uses LLM model & role config from `models.yaml` / `llm_roles.yaml`.
    
- **M3**: planner payloads often embed semantic views (items, craftability) into prompts.
    
- **M4**: Critic/ErrorModel may take virtue scores / context_id as part of request.
    
- **M5**: Scribe / ErrorModel outputs are used by Experience building / replay and SkillSynthesizer context.
    
- **M7**: observation encoder provides the world summary passed into Planner/Critic.
    
- **M8**: AgentLoop invokes `llm_stack` roles when planning, critiquing, diagnosing, summarizing.
    

---

### M3 – GTNH World Semantics & Tech State

**What M3 owns**

- Semantics & tech state modeling:
    
    - `semantics/schema.py`, `tech_state.py`, `loader.py`, `crafting.py`, `categorize.py`.
        
- Caches & performance:
    
    - `semantics/cache.py`.
        
- Ingest and normalization from raw data:
    
    - Driven by `scripts/ingest_gtnh_semantics.py`, `ingest_nerd_*`.
        

**What M3 does _not_ own**

- No LLM prompts or models (M2).
    
- No skill procedures (M5).
    
- No curriculum logic (M11).
    
- No actual Minecraft I/O (M6).
    

**Capabilities M3 affords**

- The agent can:
    
    - Understand **what items/blocks are**, their categories, and approximate use.
        
    - Infer **tech stage** from available machines / items (LV vs MV vs HV etc).
        
    - Query **craftability** and dependencies (what machine/tech is needed).
        
    - Represent GTNH tech state as a structured `TechState` injected into goals, curriculum, and learning.
        

**Direct integrations**

- **M0**: env profile selects which GTNH dataset / variants to load.
    
- **M2**: semantics snapshots are embedded into planner / scribe prompts.
    
- **M4**: virtue evaluation may refer to tech_state (e.g. “too greedy for this stage”).
    
- **M5**: Experience episodes store `TechState` to contextualize skills and outcomes.
    
- **M7**: observation pipeline normalizes world snapshots into `TechState` via semantics.
    
- **M8**: AgentLoop reads tech_state to drive curriculum / goal selection and planning context.
    
- **M11**: curriculum engine uses `TechState.active` & `.unlocked` to gate phases and projects.
    

---

### M4 – Virtue Lattice & Ethical Evaluation

**What M4 owns**

- Virtue schema & metrics:
    
    - `virtues/schema.py`, `metrics.py`.
        
- Lattice structure & comparison logic:
    
    - `virtues/lattice.py`, `features.py`, `compare_plans`, etc.
        
- Config & overrides:
    
    - `virtues/loader.py`, `integration_overrides.py`, `virtues.yaml`.
        

**What M4 does _not_ own**

- No agent loop control (M8).
    
- No curriculum goals (M11).
    
- No LLM itself (M2), just provides numbers / explanations.
    
- No direct skill code (M5), only evaluation of usage.
    

**Capabilities M4 affords**

- The agent can:
    
    - Assign **virtue scores** to plans and actions (prudence, justice, temperance, etc).
        
    - Apply **hard constraints** (e.g. “never nuke your base”).
        
    - Compare alternate plans by virtue balance, not just success probability.
        
    - Explain decisions in virtue terms (for logs & debugging).
        

**Direct integrations**

- **M2**: Critic/ErrorModel requests may include virtue context & receive virtue-related metadata.
    
- **M3**: virtue features may depend on tech_state (e.g. resource scarcity at LV).
    
- **M5**: Experience episodes include virtue metrics to feed learning / replay.
    
- **M8**: AgentLoop calls into virtue evaluator during plan comparison / mitigation.
    
- **M9**: monitoring / dashboard displays virtue-related metrics / events.
    
- **M11**: Curriculum phase configs can override virtue weights by phase.
    

---

### M5 – Experience & Core Skill Layer (Replay basics)

**What M5 owns**

- Experience / replay schema:
    
    - `learning/schema.py`, `agent/experience.py`.
        
- Experience buffer:
    
    - `learning/buffer.py` (JSONL-style `replay.jsonl`).
        
- Minimal learning hooks:
    
    - `learning/curriculum_hooks.py` (where curriculum and experience meet).
        
- Baseline skill world:
    
    - `skills/schema.py`, `skills/loader.py`, `skills/registry.py`, `skills/packs.py`.
        
- Specs for base hand-authored skills:
    
    - `config/skills/*.yaml`
        
    - `config/skill_packs/*.yaml`
        
    - Implementations in `skills/base/*.py`.
        

**What M5 does _not_ own**

- No high-level learning strategy / manager (that’s M10).
    
- No curriculum engine (M11).
    
- No underlying LLM code generation (M2).
    
- No AgentLoop iteration control (M8).
    

**Capabilities M5 affords**

- The agent can:
    
    - Record episodes into a stable replay format for analysis or future learning.
        
    - Query its experiences, at least sequentially, for debugging / offline tools.
        
    - Access a **stable set of base skills** with specs & implementations.
        
    - Use skill metadata (preconditions, effects, tags) as prompts for planners.
        

**Direct integrations**

- **M2**: Scribe / ErrorModel outputs are folded into Experience metadata.
    
- **M3**: Experience episodes embed `TechState`.
    
- **M4**: virtue metrics stored in Experience.
    
- **M6**: Skills translate into BotCore actions; Experience captures traces of this.
    
- **M7**: Observation & trace schemas are recorded in episodes.
    
- **M8**: AgentLoop writes episodes into `ExperienceBuffer`.
    
- **M10**: SkillLearningManager & Synthesizer read from experiences and skill registry.
    
- **M9**: monitoring may tap into Experience to display episode status.
    

---

## P2 – Embodiment & Sensing (M6–M7)

### M6 – Bot Core & Minecraft IPC Shell

**What M6 owns**

- BotCore abstraction:
    
    - `bot_core/core.py`, `bot_core/runtime.py`, `bot_core/testing/fakes.py`.
        
- Low-level actions & movement:
    
    - `bot_core/actions.py`, `bot_core/nav/*`, `collision.py`, `world_tracker.py`.
        
- Net / IPC interfaces:
    
    - `bot_core/net/ipc.py`, `external_client.py`, `client.py`.
        
- World snapshot capture:
    
    - `bot_core/snapshot.py`.
        

**What M6 does _not_ own**

- No semantic interpretation of blocks/items (M3).
    
- No decision-making (that’s M8 + M2).
    
- No skill semantics (that’s M5), only action-level primitives.
    
- No direct curriculum or learning logic (M10/M11).
    

**Capabilities M6 affords**

- The agent can:
    
    - Turn high-level “skill actions” into concrete game actions:
        
        - Move, break block, place block, interact, etc.
            
    - Track position, collisions, and world snapshots at a low level.
        
    - Eventually talk to a Forge mod / external client via IPC.
        

**Direct integrations**

- **M5**: Skills’ `suggest_actions` produce `Action` objects consumed by BotCore.
    
- **M7**: Observation pipeline ingests snapshots/world_tracker info as raw input.
    
- **M8**: AgentLoop sequences actions through BotCore and receives step results.
    
- **M9**: monitoring can tap BotCore events for diagnostics.
    
- **M1**: runtime tests (`test_runtime_m6_m7_smoke.py`, `tools/smoke_botcore.py`) exercise BotCore in isolation.
    

---

### M7 – Observation & Trace Encoding

**What M7 owns**

- Observation schema & normalization:
    
    - `observation/schema.py`, `worldstate`-like normalized views.
        
- Observation pipeline:
    
    - `observation/pipeline.py`, `observation/encoder.py`.
        
- Trace schema:
    
    - `observation/trace_schema.py` (PlanTrace, TraceStep, etc.).
        
- Test utilities: `observation/testing.py`.
    

**What M7 does _not_ own**

- No planning or skill selection (M2/M8).
    
- No semantics data loading (M3).
    
- No learning decisions (M10).
    
- No direct LLM logic.
    

**Capabilities M7 affords**

- The agent can:
    
    - Turn raw bot snapshots & environment data into a structured `WorldState`.
        
    - Encode planner/critic/scribe views from the same underlying world.
        
    - Produce PlanTrace objects that track:
        
        - plan
            
        - steps
            
        - tech_state
            
        - virtue scores (attached by others).
            

**Direct integrations**

- **M3**: uses semantics to enrich observation with tech_state, item groups, etc.
    
- **M2**: provides encoded inputs for planner, critic, scribe requests.
    
- **M5**: PlanTrace attached to Experience episodes.
    
- **M6**: consumes snapshots from BotCore as raw input.
    
- **M8**: AgentLoop uses observation pipeline to get current world_before / world_after.
    

---

## P3 – Control & Monitoring (M8–M9)

### M8 – Agent Loop & Planning Orchestration

**What M8 owns**

- Agent loop schema:
    
    - `spec/agent_loop.py` (AgentGoal, EpisodeResult, etc.).
        
- AgentLoop implementation:
    
    - `agent_loop/loop.py`, `agent_loop/schema.py`, `agent_loop/state.py`.
        
- Integration with runtime:
    
    - `agent/controller.py`, `agent/state.py`, `agent/experience.py` connection points.
        

**What M8 does _not_ own**

- No low-level BotCore (M6).
    
- No semantics definitions (M3).
    
- No virtue math (M4).
    
- No curriculum content (M11) or learning algorithms (M10).
    
- No specific LLM prompts / parsing (delegated to M2).
    

**Capabilities M8 affords**

- The agent can:
    
    - Run a full **episode**:
        
        - observe → plan → (optionally critic) → execute steps → summarize outcome → store experience.
            
    - Track episode lifecycle, including:
        
        - Episode IDs
            
        - PlanTrace
            
        - Pre/post evaluations (plan & outcome).
            
    - Plug into Lab / Doctor to run minimal episodes as smoke tests.
        

**Direct integrations**

- **M0/M1**: receives runtime config, env profile, debug flags.
    
- **M2**: calls Planner, PlanCode, Critic, ErrorModel, Scribe via LLMStack.
    
- **M3**: uses tech_state (via observation / semantics) to contextualize goals and plans.
    
- **M4**: queries virtue evaluator for plan comparison / constraints.
    
- **M5**: writes `ExperienceEpisode` into ExperienceBuffer.
    
- **M6**: dispatches skill-generated `Action` lists into BotCore and receives results.
    
- **M7**: uses observation pipeline and PlanTrace types to build and record traces.
    
- **M9**: emits monitoring events for episode start/finish, plan selections, failures.
    
- **M11**: requests next AgentGoal from curriculum engine.
    

---

### M9 – Monitoring, Events & Tools

**What M9 owns**

- Monitoring/event infrastructure:
    
    - `monitoring/bus.py`, `events.py`, `integration.py`.
        
- Logging facade:
    
    - `monitoring/logger.py`, `llm_logging.py`.
        
- Dashboard / tools:
    
    - `monitoring/controller.py`, `dashboard_tui.py`, `monitoring/tools.py`.
        

**What M9 does _not_ own**

- No decision-making for the agent (M8).
    
- No semantics or tech state (M3).
    
- No LLM logic (M2).
    
- No curriculum / learning logic (M10/M11).
    

**Capabilities M9 affords**

- The agent/developer can:
    
    - Observe **live event streams**:
        
        - Episode start / end
            
        - Plan chosen
            
        - Skill invoked
            
        - Error events.
            
    - Run non-interfering monitoring tools:
        
        - TUI dashboards
            
        - LLM call summaries
            
        - System-level audits.
            
    - Pipe LLM logs & runtime metrics into a coherent stream keyed by episode_id.
        

**Direct integrations**

- **M2**: LLM calls log to `logs/llm/*.json` via `log_llm_call` and are interpreted by monitoring tools.
    
- **M4**: virtue-related events (constraints triggered, inequality warnings) are emitted as monitoring events.
    
- **M5**: monitoring tools may read Experience/replay for offline reports.
    
- **M6**: BotCore events (movement, collisions) can be emitted on the event bus.
    
- **M7**: observation anomalies can be surfaced as monitoring events.
    
- **M8**: AgentLoop emits high-level events, consumed by dashboard / logger.
    
- **M1**: runtime config plugs monitoring mode / log level.
    

---

## P4 – Learning & Curriculum (M10–M11)

### M10 – Skill Learning & Evolution

**What M10 owns**

- Learning manager & evaluator:
    
    - `learning/manager.py`, `evaluator.py`.
        
- Evolution schema:
    
    - `learning/schema.py` fields for SkillCandidate, metrics_before/after, etc.
        
- Synthesizer:
    
    - `learning/synthesizer.py` (CodeModel → SkillCandidate).
        
- Curriculum learning hooks:
    
    - `runtime/curriculum_learning_coordinator.py`
        
    - `runtime/curriculum_learning_triggers.py`.
        

**What M10 does _not_ own**

- No base skill definitions (M5 owns those).
    
- No curriculum phases or goals (M11).
    
- No core Experience format (M5) although it uses it.
    
- No LLM internals (M2), just uses CodeModel API for synthesis.
    

**Capabilities M10 affords**

- The agent can:
    
    - Propose **new skill candidates** from clusters of successful episodes.
        
    - Evaluate skill variants using metrics (success_rate, cost, risk).
        
    - Update skill metadata and mark versions as **candidate → active → deprecated**.
        
    - Coordinate when to trigger learning / synthesis runs based on curriculum or performance.
        

**Direct integrations**

- **M2**: uses CodeModel role (via M2 / llm_stack) to generate skill YAML + impl code.
    
- **M3**: relies on tech_state in episodes to cluster / contextualize skills.
    
- **M4**: virtue metrics from episodes feed into skill evaluation (good vs evil skill).
    
- **M5**: reads Experience episodes and uses SkillRegistry (`skills/registry.py`) to manage versions.
    
- **M9**: emits learning events / candidate status changes for monitoring.
    
- **M11**: curriculum hooks can influence which skills to evolve for a given phase.
    

---

### M11 – Curriculum & Specialization

**What M11 owns**

- Curriculum schema & config:
    
    - `curriculum/schema.py`, `config/curricula/*.yaml`.
        
- Curriculum engine:
    
    - `curriculum/engine.py` (phase selection, next_goal, project unlock).
        
- Curriculum manager & policies:
    
    - `curriculum/manager.py`, `policy.py`, `strategies.py`, `integration_agent_loop.py`.
        
- Long-horizon project modeling:
    
    - `LongHorizonProject` and stage semantics in schema.
        

**What M11 does _not_ own**

- No low-level semantics (M3).
    
- No agent loop logic (M8).
    
- No LLM stack (M2).
    
- No learning implementation (M10), though it coordinates with it.
    

**Capabilities M11 affords**

- The agent can:
    
    - Select the **next goal** based on tech_state, world, and experience summary.
        
    - Track **phase progression** and “complete” phases when conditions met.
        
    - Unlock **long-horizon projects** (like Stargate) when prerequisite phases are complete.
        
    - Expose a **Curriculum View** describing:
        
        - Active phase
            
        - Active goals
            
        - Skill focus (must_have/preferred)
            
        - Unlocked projects.
            

**Direct integrations**

- **M0**: env profile may select which curriculum to load.
    
- **M3**: uses `TechState.active` and `TechState.unlocked` to gate phases / completion.
    
- **M4**: phase configs can set virtue overrides for evaluation.
    
- **M5**: experience summary may inform strategies (which goal next).
    
- **M8**: AgentLoop asks CurriculumEngine for the next AgentGoal and reads ActiveCurriculumView for context.
    
- **M9**: monitoring tools display phase/progress/project state.
    
- **M10**: curriculum and learning coordinator decide when to evolve skills for certain phases/projects (e.g. LV automation vs HV mega-project).
    

---

That’s the current map:

- P0 sets **where & how** you run.
    
- P1 defines **how the agent thinks & knows GTNH**.
    
- P2 defines **how the body moves and senses**.
    
- P3 defines **how episodes are orchestrated and watched**.
    
- P4 defines **how it grows new skills and chooses its long path**.
    



---
## Lab Integration Workflow Design Concepts: How to run this phase without losing your mind

**EXPLICIT PURPOSE OF LAB INTEGRATION: Lab Integration exists to validate the entire architecture, and to make it thoroughly auditable, in a controlled environment, using a real LLM but a fake world, before the system is exposed to irreversible external entropy (Minecraft).**

You’re aiming for **maximum observability with minimal chaos**. To keep continuity and avoid design drift:

**Hard Rules: 

Any problem encountered in live GTNH MUST first be reproduced in Lab Integration.  
However, if it cannot be reproduced in Lab mode, it is not allowed to block development.
In Lab mode, TechState is frozen and deterministic unless the test explicitly changes it.

No LLM harness or test may rely on an implicit prompt.  
All prompts must be versioned.  
All tests must assert the version they expect.

All role I/O schemas must be validated against spec/llm.py on every call.  
Any deviation is treated as a system failure, not a soft warning.

When debugging, the episode JSON is the primary artifact,  
the LLM logs are secondary,  
and the stack traces / monitoring events are tertiary.

---

### 1. Treat Lab Integration as mini-phases: L-Pass A, B, C

Think in **passes**:

# **Lab Pass A: Instrumentation**
    
    - L0 – Config & skill sanity checks
        
    - L1 – Unified logging
        
    - L2 – Episode dumper & viewer
        
    - L3 – AgentLoop debug mode
        
    - L4 – Metrics in replay
        

Goal of Pass A:

> With _fake_ LLMs and _fake_ world, I can see everything and nothing crashes when I inspect it.

---

**Config Sanity Checker (L0 Config)**

- A tool that:
    
    - Validates core YAMLs (`env`, `minecraft`, `models`, `hardware`, curriculum, skills, skill_packs).
        
    - Cross-checks relationships:
        
        - Skills referenced by packs exist in specs.
            
        - Skills referenced by curriculum / skill_focus exist in registry.
            
- Goal: fail early when config is wrong instead of mid-episode.

---

**Unified Logging with `episode_id` + `context_id` (L1 Logging)**

- Standardize logging so that:
    
    - Every significant component logs with at least `episode_id` and `context_id`.
        
    - LLM calls are consistently logged with role + operation.
        
- This especially applies to the LLM stack (planner, plan-code, critic, error, scribe).

---

**Episode Dumper + CLI Viewer (L2 Episodes)**

- A small subsystem that:
    
    - Dumps one JSON per episode (plan, trace, tech_state, virtue, critic, outcome, etc).
        
    - Provides a CLI viewer to pretty-print or filter episode contents (plan, steps, failures, etc).
        
- Unit of inspection is the _episode_, not random log lines.

---

**AgentLoop Debug Mode (L3 Debug)**

- Add a debug toggle in `AgentLoopConfig` that:
    
    - Logs structured LLM responses at DEBUG level (not just raw text).
        
    - Potentially uses stricter caps like lower `max_skill_steps` or other limits in debug context.
        
- This gives you a controlled, high-visibility mode for diagnosing behavior.

---

**Metrics & Counters in Replay (L4 Metrics)**

- Add lightweight metrics:
    
    - Episode counts: total / success / failure.
        
    - Per-skill usage counts if possible.
        
- Occasionally write a `metrics.json` or equivalent, so future M10 learning / evaluation has something structured to read from.

---

- # **Lab Pass B: Control surfaces**
    
    - L5 – Agent Doctor
        
    - L6 – Central prompt templates refactor (so the LLM layer is tweakable)
        

Goal of Pass B:

> I have a **single doctor script** and a **single prompt registry**. I can run one command and get a coherent health report. I can change prompts in one place.

---

**Agent Doctor / System Audit (L5 Doctor)**

- A script / tool that:
    
    - Checks that skills are wired correctly (registry, specs, implementations, deprecated behavior).
        
    - Checks curriculum (active phase, unlocked projects, phase completion semantics).
        
    - Verifies LLM stack wiring (planner / plan-code / critic / error / scribe facades).
        
    - Confirms that the AgentLoop can run a stub episode successfully.
        
- This is your “one-button health check” for the whole architecture.

---

**Central Prompt Templates Module (L6 Prompts)**

- Move scattered prompt text for:
    
    - planner
        
    - plan-code
        
    - critic
        
    - error model
        
    - scribe
        
    - synthesizer
        
- Into a central place (e.g. `llm_stack.prompts` or equivalent), so tuning and versioning prompts is maintainable.

---

# **Lab Pass C: Real LLM integration**
    
    - L7 – Per-role harness tests + Full-role Episode Test for LLM
        
    - L8 – Combined “Lab Integration” system-level test
        

Goal of Pass C:

> With a real LLM plugged in, **one command** does a multi-role end-to-end run, dumps episodes, logs everything, and I can reason about failures.

You don’t have to name them exactly like that, but the idea is:  
**A = instrumentation**, **B = coordination**, **C = real LLM & full stack**.

---
**Real LLM Integration with Per-role Harnesses + Full-role Episode Test (L7 LLM Integration)**

- Wire in real LLMs with per-role harnesses:
    
    - Planner
        
    - Plan-code
        
    - Critic
        
    - Error Model
        
    - Scribe
        
- Then design a “grand” test that:
    
    - Uses a real LLM for those roles.
        
    - Runs a (fake-world) episode that exercises all five roles.
        
    - Checks for:
        
        - Role bleed (wrong behavior from wrong prompt / harness).
            
        - Timing issues and weird interactions between roles.
            
- This test uses **real LLM calls** and **fakes for GTNH world**, to verify the architecture & contracts in a controlled lab environment.

---

**Testing Strategy for the Phase as a Whole (L8)**

- For each new feature/module (Doctor, episode dumper, logging, etc):
    
    - Create tests that verify it in isolation.
        
- Then:
    
    - Integrate each feature at its connection points (e.g., AgentLoop → Doctor, EpisodeResult → dumper, LLM stack → prompts).
        
    - Write integration tests that confirm those connections are working as expected.
        
- Finally:
    
    - Design a “whole subsystem” test that validates the **entire Lab Integration layer**.
        
    - Design a final test that:
        
        - Uses _all_ real variables you can (real configs, real LLM, real curriculum, real skills).
            
        - Substitutes fakes only where absolutely necessary (world, IPC).
            
        - Runs a few episodes and verifies end-to-end behavior.


---

### 2. Use a single “Lab Profile” for repeatability

To avoid config roulette, define a dedicated lab profile in your env:

- `env.yaml` → `profile: lab_integration`
    
- `minecraft.yaml` → fake world specifics
    
- `models.yaml` → one known-good model per role  
    (or even just a single model reused with different prompt templates, but clearly labeled as such)
    

All lab tools & tests should default to **this** profile, so:

- AgentDoctor
    
- episode dumper CLI
    
- grand LLM test
    

all “see” the same world. That’s how you get lossless continuity: no “works here, fails there” because some random env var changed.

---

### 3. Make tests mirror the passes

Don’t just scatter tests randomly. Group them so the test layout itself encodes the workflow:

- `tests/lab/test_L0_config_doctor.py`
    
- `tests/lab/test_L1_logging_and_ids.py`
    
- `tests/lab/test_L2_episode_dump_view.py`
    
- `tests/lab/test_L5_agent_doctor_integration.py`
    
- `tests/lab/test_L8_real_llm_episode.py`
    

Then you can:
```bash
pytest tests/lab -q

```

and know this specifically exercises Lab Integration, not everything.

---

### 4. Episode as “canonical debug artifact”

You’re already leaning this way. Make it official:

- **Rule**: when something misbehaves, first step is:  
    “dump the last episode + relevant LLM logs.”
    

Then arrange the lab tools accordingly:

- `run_lab_episode.py` → runs one lab episode, dumps JSON.
    
- `view_episode.py` → pretty-prints that JSON.
    
- `agent_doctor.py` → checks the system state that produced that episode.
    

That gives you a repeatable workflow:

1. Run lab episode.
    
2. Inspect episode.
    
3. Run doctor if something looks cursed.
    

No flailing around 12 layers of code.

---

### 5. Keep LLM prompt changes versioned / labeled

To avoid subtle regressions later:

- In `llm_stack.prompts` (or wherever), add **prompt version tags**, e.g.:
```python
PLANNER_PROMPT_V1 = ...
PLANNER_PROMPT_V2 = ...

```

And in `models.yaml` or `env.yaml`:
```yaml
llm_prompts:
  planner_version: v1
  critic_version: v1

```

Then the lab tests can assert that **a particular version** of prompts is being used. That’s how you guarantee continuity after tweaks, not via vibes.

---

### 6. Guard rails on real-LLM tests

Real LLM tests are inherently flaky if you let them be. To keep them useful:

- Keep lab prompts **very constrained** and **output format very strict**.
    
- Limit episode complexity:
    
    - 1 goal
        
    - tiny world summary
        
    - 3-5 steps max
        
- Consider labeling them separately:
    
    - `tests/lab/test_L8_real_llm_episode.py` with a marker like `@pytest.mark.real_llm`
        
    - So you can run unit tests without pinging the model constantly.
        

This way, the grandmaster test is a **sanity check**, not a lottery.

---

### 7. Document the lab workflow in one place

Make a small `docs/lab_integration.md` with:

- Phase goals (what Lab Integration is for).
    
- Module list L0–Lx, each with:
    
    - Owner area (config, episodes, logging, LLM).
        
    - Inputs (what it depends on).
        
    - Outputs (what it produces: JSON, logs, metrics).
        
- “Standard debug workflow” section:
    
    1. Run agent doctor.
        
    2. Run a lab episode.
        
    3. View the episode.
        
    4. Inspect LLM logs if needed.
        
    5. If all green, move on to GTNH IPC integration.
        

You want this so that Future-Knoah can drop in cold, read one doc, and reconstruct the lab phase without spelunking your entire brain.

---

Short version:

Before exposing this thing to the horrors of real Minecraft, you’re building a **laboratory harness** around it. Turning it into passes (A/B/C), tying everything to a lab profile, and making episodes & doctor the core artifacts will keep continuity tight and prevent the system from dissolving into “well it kind of works if I don’t breathe on it.”

This phase is about making sure when something breaks later, you don’t have to repeat that level of suffering.

---

# Current State of the Project

# File Structure:
```tree
.                                    # GTNH_Agent repo root: full local AGI-in-a-modpack project
├── bootstrap_structure.py           # Script to initialize / validate the repo layout & scaffolding
├── config                           # All static configuration for env, skills, curriculum, models, etc.
│   ├── curricula                    # Curriculum definitions: tech-phase → goals → projects
│   │   ├── aesthetic_megabase.yaml  # Fancy long-horizon “build pretty megabase” curriculum profile
│   │   ├── default_speedrun.yaml    # Default progression-focused curriculum for main agent runs
│   │   └── eco_factory.yaml         # Curriculum for efficient/eco-focused factory progression
│   ├── env.yaml                     # Global environment profiles (hardware, models, runtime mode)
│   ├── gtnh_blocks.generated.yaml   # Autogenerated GTNH block semantics / metadata (from dumps)
│   ├── gtnh_blocks.yaml             # Hand-edited overrides / curated block semantics
│   ├── gtnh_items.generated.yaml    # Autogenerated GTNH item semantics / metadata
│   ├── gtnh_items.yaml              # Hand-edited item overrides / special cases
│   ├── gtnh_recipes.agent.json      # Slimmed / reshaped recipes view tuned for agent reasoning
│   ├── gtnh_recipes.generated.json  # Raw-ish JSON recipe dump from GTNH tooling
│   ├── gtnh_recipes.json            # Normalized recipes view used by semantics.crafting
│   ├── gtnh_tech_graph.yaml         # Tech-state DAG: tiers, dependencies, unlock conditions
│   ├── hardware.yaml                # Hardware profiles: CPU/GPU/threads for LLM stack & runtime
│   ├── llm_roles.yaml               # Role → model / preset mapping (planner, plan_code, critic, etc.)
│   ├── minecraft.yaml               # Minecraft-side config: IPC endpoints, dimensions, runtimes
│   ├── models.yaml                  # Local LLM model registry: paths, quantization, context limits
│   ├── raw                          # Raw, unprocessed data dumps from GTNH tooling
│   │   ├── block.csv                # Raw CSV of blocks from Tellme/nerd or similar
│   │   ├── item.csv                 # Raw CSV of items
│   │   ├── recipes.json             # Raw recipes dump (big chonky source-of-truth)
│   │   └── recipes_stacks.json      # Variant dump with stack-size / multi-output info
│   ├── skill_packs                  # Skill pack definitions: which skills are enabled for which tier
│   │   ├── lv_core.yaml             # LV core skill pack: baseline automation “kit”
│   │   └── steam_age.yaml           # Steam-age skill pack: pre-LV boiler & coke-oven skills
│   ├── skills                       # Hand-authored SkillSpec YAMLs (one per skill)
│   │   ├── basic_crafting.yaml      # SkillSpec for generic crafting helper skill
│   │   ├── chop_tree.yaml           # SkillSpec: chopping trees & collecting logs
│   │   ├── feed_coke_ovens.yaml     # SkillSpec: feed coke ovens with fuel & wood
│   │   ├── feed_steam_boiler.yaml   # SkillSpec: maintain steam boiler fuel flow
│   │   ├── maintain_coke_ovens.yaml # SkillSpec: keep coke ovens running (output clearing, etc.)
│   │   ├── plant_sapling.yaml       # SkillSpec: replant saplings for renewable wood
│   │   └── refill_water_tanks.yaml  # SkillSpec: keep boiler water tanks topped off
│   ├── skills_candidates            # Slot for synthesized skill YAMLs produced by M10 learning
│   ├── tools                        # Small config-related helper scripts
│   │   ├── print_env.py             # Utility: print resolved env/config profile for debugging
│   │   └── validate_env.py          # Sanity check: validate env/hardware/models YAML coherence
│   └── virtues.yaml                 # Virtue lattice config: weights, labels, overrides
├── data                             # Data artifacts produced by runs (replay, metrics, etc.)
│   └── experiences                  # Experience buffer snapshots / replay logs
│       └── replay.jsonl             # JSONL replay of Experience episodes for learning/debug
├── docs                             # Design docs and high-level architecture references
│   ├── architecture.md              # Core architecture document: modules, flows, contracts
│   ├── ipc_protocol_m6.md           # M6: IPC protocol design for BotCore ↔ Minecraft mod
│   ├── llm_role_boundaries.q1.yaml  # YAML spec defining strict role boundaries for LLMs (Q1)
│   ├── m6_bot_core_1_7_10.md        # Notes on BotCore design specific to 1.7.10 GTNH runtime
│   └── phase1_integration.md        # Phase 1 integration plan & checklist
├── .github                          # CI / automation config
│   └── workflows
│       └── ci.yml                   # CI pipeline: pytest, lint, maybe type checks
├── .gitignore                       # Git ignore rules for local artifacts
├── logs                             # Runtime logs (LLM calls, etc.) – no comments as requested
│   └── llm
│       ├── 20251127T154508_30389_error_model_analyze_failure.json
│       ├── 20251127T154508_30389_plan_code_plan.json
│       ├── 20251127T154508_30389_scribe_summarize_trace.json
│       ├── 20251127T154653_30421_plan_code_plan.json
│       ├── 20251127T154929_30485_error_model_analyze_failure.json
│       ├── 20251127T155054_30542_scribe_summarize_trace.json
│       ├── 20251127T171641_54578_plan_code_plan.json
│       ├── 20251127T172238_54966_plan_code_plan.json
│       ├── 20251127T172404_55071_plan_code_plan.json
│       ├── 20251127T181107_57121_plan_code_plan.json
│       ├── 20251127T181432_57416_plan_code_plan.json
│       ├── 20251127T202123_80866_error_model_analyze_failure.json
│       ├── 20251127T202123_80866_plan_code_plan.json
│       ├── 20251127T202123_80866_scribe_summarize_trace.json
│       ├── 20251127T202334_80953_error_model_analyze_failure.json
│       ├── 20251127T202334_80953_plan_code_plan.json
│       ├── 20251127T202334_80953_scribe_summarize_trace.json
│       ├── 20251127T223951_107281_error_model_analyze_failure.json
│       ├── 20251127T223951_107281_plan_code_plan.json
│       ├── 20251127T223951_107281_scribe_summarize_trace.json
│       ├── 20251128T001446_136742_error_model_analyze_failure.json
│       ├── 20251128T001446_136742_plan_code_plan.json
│       ├── 20251128T001446_136742_scribe_summarize_trace.json
│       ├── 20251128T003700_146815_error_model_analyze_failure.json
│       ├── 20251128T003700_146815_plan_code_plan.json
│       ├── 20251128T003700_146815_scribe_summarize_trace.json
│       ├── 20251128T115106_24456_error_model_analyze_failure.json
│       ├── 20251128T115106_24456_plan_code_plan.json
│       ├── 20251128T115106_24456_scribe_summarize_trace.json
│       ├── 20251128T143640_52852_error_model_analyze_failure.json
│       ├── 20251128T143640_52852_plan_code_plan.json
│       ├── 20251128T143640_52852_scribe_summarize_trace.json
│       ├── 20251128T143816_52937_error_model_analyze_failure.json
│       ├── 20251128T143816_52937_plan_code_plan.json
│       ├── 20251128T143816_52937_scribe_summarize_trace.json
│       ├── 20251128T153451_109564_error_model_analyze_failure.json
│       ├── 20251128T153451_109564_plan_code_plan.json
│       ├── 20251128T153451_109564_scribe_summarize_trace.json
│       ├── 20251128T154112_109913_error_model_analyze_failure.json
│       ├── 20251128T154112_109913_plan_code_plan.json
│       ├── 20251128T154112_109913_scribe_summarize_trace.json
│       ├── 20251128T154622_110074_error_model_analyze_failure.json
│       ├── 20251128T154622_110074_plan_code_plan.json
│       ├── 20251128T154622_110074_scribe_summarize_trace.json
│       ├── 20251128T155834_129265_error_model_analyze_failure.json
│       ├── 20251128T155834_129265_plan_code_plan.json
│       ├── 20251128T155834_129265_scribe_summarize_trace.json
│       ├── 20251128T160155_129475_error_model_analyze_failure.json
│       ├── 20251128T160155_129475_plan_code_plan.json
│       ├── 20251128T160155_129475_scribe_summarize_trace.json
│       ├── 20251128T160633_142703_error_model_analyze_failure.json
│       ├── 20251128T160633_142703_plan_code_plan.json
│       ├── 20251128T160633_142703_scribe_summarize_trace.json
│       ├── 20251128T162316_143085_error_model_analyze_failure.json
│       ├── 20251128T162316_143085_plan_code_plan.json
│       ├── 20251128T162316_143085_scribe_summarize_trace.json
│       ├── 20251128T162956_163507_error_model_analyze_failure.json
│       ├── 20251128T162956_163507_plan_code_plan.json
│       ├── 20251128T162956_163507_scribe_summarize_trace.json
│       ├── 20251128T164527_184688_error_model_analyze_failure.json
│       ├── 20251128T164527_184688_plan_code_plan.json
│       ├── 20251128T164527_184688_scribe_summarize_trace.json
│       ├── 20251128T164725_184780_error_model_analyze_failure.json
│       ├── 20251128T164725_184780_plan_code_plan.json
│       ├── 20251128T164725_184780_scribe_summarize_trace.json
│       ├── 20251128T164846_185052_error_model_analyze_failure.json
│       ├── 20251128T164846_185052_plan_code_plan.json
│       ├── 20251128T164846_185052_scribe_summarize_trace.json
│       ├── 20251128T190228_210275_error_model_analyze_failure.json
│       ├── 20251128T190228_210275_plan_code_plan.json
│       ├── 20251128T190228_210275_scribe_summarize_trace.json
│       ├── 20251128T190457_210363_error_model_analyze_failure.json
│       ├── 20251128T190457_210363_plan_code_plan.json
│       ├── 20251128T190457_210363_scribe_summarize_trace.json
│       ├── 20251128T191608_225560_error_model_analyze_failure.json
│       ├── 20251128T191608_225560_plan_code_plan.json
│       ├── 20251128T191608_225560_scribe_summarize_trace.json
│       ├── 20251128T192046_233385_error_model_analyze_failure.json
│       ├── 20251128T192046_233385_plan_code_plan.json
│       ├── 20251128T192046_233385_scribe_summarize_trace.json
│       ├── 20251128T192251_233478_error_model_analyze_failure.json
│       ├── 20251128T192251_233478_plan_code_plan.json
│       ├── 20251128T192251_233478_scribe_summarize_trace.json
│       ├── 20251128T204447_291001_error_model_analyze_failure.json
│       ├── 20251128T204447_291001_plan_code_plan.json
│       ├── 20251128T204501_291001_scribe_summarize_trace.json
│       ├── 20251128T204732_291164_error_model_analyze_failure.json
│       ├── 20251128T204732_291164_plan_code_plan.json
│       ├── 20251128T204807_291164_scribe_summarize_trace.json
│       ├── 20251128T212201_317178_error_model_analyze_failure.json
│       ├── 20251128T212201_317178_plan_code_plan.json
│       ├── 20251128T212258_317178_scribe_summarize_trace.json
│       ├── 20251128T214908_353050_error_model_analyze_failure.json
│       ├── 20251128T214908_353050_plan_code_plan.json
│       ├── 20251128T215005_353050_scribe_summarize_trace.json
│       ├── 20251128T215508_371511_error_model_analyze_failure.json
│       ├── 20251128T215508_371511_plan_code_plan.json
│       ├── 20251128T215605_371511_scribe_summarize_trace.json
│       ├── 20251128T215741_371600_error_model_analyze_failure.json
│       ├── 20251128T215741_371600_plan_code_plan.json
│       ├── 20251128T215838_371600_scribe_summarize_trace.json
│       ├── 20251129T120138_42538_error_model_analyze_failure.json
│       ├── 20251129T120138_42538_plan_code_plan.json
│       ├── 20251129T120235_42538_scribe_summarize_trace.json
│       ├── 20251129T121735_52787_error_model_analyze_failure.json
│       ├── 20251129T121735_52787_plan_code_plan.json
│       ├── 20251129T121832_52787_scribe_summarize_trace.json
│       ├── 20251129T122136_52946_error_model_analyze_failure.json
│       ├── 20251129T122136_52946_plan_code_plan.json
│       ├── 20251129T122233_52946_scribe_summarize_trace.json
│       ├── 20251129T131642_83734_error_model_analyze_failure.json
│       ├── 20251129T131642_83734_plan_code_plan.json
│       ├── 20251129T131738_83734_scribe_summarize_trace.json
│       ├── 20251129T131934_83828_error_model_analyze_failure.json
│       ├── 20251129T131934_83828_plan_code_plan.json
│       ├── 20251129T132032_83828_scribe_summarize_trace.json
│       ├── 20251129T134009_114623_error_model_analyze_failure.json
│       ├── 20251129T134009_114623_plan_code_plan.json
│       ├── 20251129T134106_114623_scribe_summarize_trace.json
│       ├── 20251129T135607_136708_error_model_analyze_failure.json
│       ├── 20251129T135607_136708_plan_code_plan.json
│       ├── 20251129T135703_136708_scribe_summarize_trace.json
│       ├── 20251129T140350_168767_error_model_analyze_failure.json
│       ├── 20251129T140350_168767_plan_code_plan.json
│       ├── 20251129T140447_168767_scribe_summarize_trace.json
│       ├── 20251129T145028_202536_error_model_analyze_failure.json
│       ├── 20251129T145028_202536_plan_code_plan.json
│       ├── 20251129T145124_202536_scribe_summarize_trace.json
│       ├── 20251129T145252_202626_error_model_analyze_failure.json
│       ├── 20251129T145252_202626_plan_code_plan.json
│       ├── 20251129T145349_202626_scribe_summarize_trace.json
│       ├── 20251129T153928_260438_error_model_analyze_failure.json
│       ├── 20251129T153928_260438_plan_code_plan.json
│       ├── 20251129T154026_260438_scribe_summarize_trace.json
│       ├── 20251129T154319_260642_error_model_analyze_failure.json
│       ├── 20251129T154319_260642_plan_code_plan.json
│       ├── 20251129T154415_260642_scribe_summarize_trace.json
│       ├── 20251129T154536_260720_error_model_analyze_failure.json
│       ├── 20251129T154536_260720_plan_code_plan.json
│       ├── 20251129T154633_260720_scribe_summarize_trace.json
│       ├── 20251129T160320_284771_error_model_analyze_failure.json
│       ├── 20251129T160320_284771_plan_code_plan.json
│       ├── 20251129T160417_284771_scribe_summarize_trace.json
│       ├── 20251129T160525_284855_error_model_analyze_failure.json
│       ├── 20251129T160525_284855_plan_code_plan.json
│       ├── 20251129T160622_284855_scribe_summarize_trace.json
│       ├── 20251129T191136_334068_error_model_analyze_failure.json
│       ├── 20251129T191136_334068_plan_code_plan.json
│       ├── 20251129T191233_334068_scribe_summarize_trace.json
│       ├── 20251129T191456_334202_error_model_analyze_failure.json
│       ├── 20251129T191456_334202_plan_code_plan.json
│       ├── 20251129T191554_334202_scribe_summarize_trace.json
│       ├── 20251130T175204_102295_error_model_analyze_failure.json
│       ├── 20251130T175205_102308_scribe_summarize_trace.json
│       ├── 20251130T180201_102632_error_model_analyze_failure.json
│       ├── 20251130T180258_102632_scribe_summarize_trace.json
│       ├── 20251130T180803_102943_error_model_analyze_failure.json
│       ├── 20251130T180901_102943_scribe_summarize_trace.json
│       ├── 20251130T181104_103176_error_model_analyze_failure.json
│       └── 20251130T181202_103176_scribe_summarize_trace.json
├── pyproject.toml                   # Project metadata, dependencies, pytest config, entry points
├── .pytest_cache                    # Pytest’s own cache; safe to ignore, useful for local runs
│   ├── CACHEDIR.TAG                 # Marker so tools know this is a cache dir
│   ├── .gitignore                   # Prevent cache from being accidentally committed
│   ├── README.md                    # Short explanation of pytest cache usage
│   └── v
│       └── cache
│           ├── lastfailed           # Tracks last failing tests for quick reruns
│           └── nodeids              # Cached test node IDs for faster discovery
├── .python-version                  # Python version pin for tools like pyenv
├── README.md                        # Top-level project README / onboarding info
├── scripts                          # One-off or dev scripts not part of main package
│   ├── compact_recipes_for_agent.py # Script to shrink/reshape recipe dumps for agent use
│   ├── demo_offline_agent_step.py   # Manual demo: run a single offline agent step
│   ├── dev_shell.py                 # Launch a dev REPL with project context loaded
│   ├── ingest_gtnh_semantics.py     # Pipeline: ingest GTNH data into semantics schema
│   ├── ingest_nerd_csv_semantics.py # Convert nerd CSV outputs into internal semantics format
│   ├── ingest_nerd_recipes.py       # Import recipes from nerd/Tellme into JSON/YAML
│   ├── smoke_error_model.py         # Quick sanity check for ErrorModel pipeline
│   ├── smoke_llm_stack.py           # Smoke test for LLM stack wiring / configs
│   └── smoke_scribe_model.py        # Quick check for Scribe summarization behavior
├── src                              # Main source tree (packaged as gtnh_agent)
│   ├── agent                        # High-level “old M6/M7” agent shell abstraction
│   │   ├── bootstrap.py             # Wiring + setup for older agent runtime shell
│   │   ├── controller.py            # Entry point for interactive/legacy agent control
│   │   ├── experience.py            # Legacy agent-side experience utilities
│   │   ├── logging_config.py        # Logging configuration for agent module
│   │   ├── loop.py                  # Legacy agent loop wrapper (pre-agent_loop package)
│   │   ├── runtime_m6_m7.py         # Runtime wiring for M6/M7-style agent environments
│   │   └── state.py                 # Legacy agent state structures
│   ├── agent_loop                   # Newer, spec-aligned M8 AgentLoop implementation
│   │   ├── __init__.py              # Export AgentLoop and related configuration types
│   │   ├── loop.py                  # Core AgentLoop: observe → plan → act → record episode
│   │   ├── schema.py                # Dataclasses for AgentGoal, EpisodeResult, config, etc.
│   │   └── state.py                 # AgentLoop-specific state containers & helpers
│   ├── app                          # App-level runtime shells & integration entry points
│   │   ├── __init__.py              # App package init
│   │   └── runtime.py               # High-level runtime glue (CLI → agent → env)
│   ├── bot_core                     # In-game control core: actions, navigation, IPC, tracking
│   │   ├── actions.py               # Low-level BotCore actions (move, break, place, etc.)
│   │   ├── collision.py             # Collision / hitbox utilities for world navigation
│   │   ├── core.py                  # Main BotCore loop & interfaces
│   │   ├── __init__.py              # BotCore package init
│   │   ├── nav                      # Navigation submodule
│   │   │   ├── grid.py              # Grid representation for pathfinding
│   │   │   ├── __init__.py          # Nav package init
│   │   │   ├── mover.py             # Short-horizon movement controller
│   │   │   └── pathfinder.py        # Pathfinding algorithms (A*, etc.)
│   │   ├── net                      # Networking & IPC between BotCore and Minecraft
│   │   │   ├── client.py            # Client for talking to Forge mod / external process
│   │   │   ├── external_client.py   # Alt client implementation (external bot, etc.)
│   │   │   ├── __init__.py          # Net package init
│   │   │   └── ipc.py               # IPC message schema & send/receive helpers
│   │   ├── runtime.py               # Runtime glue to connect BotCore to an environment
│   │   ├── snapshot.py              # Game-world snapshot structures (for observation)
│   │   ├── testing                  # BotCore-specific test fakes
│   │   │   └── fakes.py             # Fake BotCore implementations for tests
│   │   ├── tracing.py               # BotCore-level tracing hooks
│   │   └── world_tracker.py         # Track world state changes over time
│   ├── cli                          # Command-line entrypoints
│   │   └── phase1_offline.py        # CLI to run Phase 1 offline agent workflows
│   ├── curriculum                   # Curriculum engine & configuration handling
│   │   ├── engine.py                # CurriculumEngine: phase selection, next goal, projects
│   │   ├── example_workflow.py      # Example script showing curriculum usage in code
│   │   ├── __init__.py              # Curriculum package init
│   │   ├── integration_agent_loop.py# Glue: curriculum → AgentLoop goal selection
│   │   ├── loader.py                # Load curriculum yaml → CurriculumConfig
│   │   ├── manager.py               # Higher-level curriculum manager for runtime
│   │   ├── policy.py                # Policies for goal selection / phase transitions
│   │   ├── schema.py                # Dataclasses for curriculum phases, goals, projects
│   │   └── strategies.py            # Strategy implementations for curriculum (explore/exploit)
│   ├── env                          # Environment profiles and config abstraction
│   │   ├── __init__.py              # Env package init
│   │   ├── loader.py                # Load env.yaml and resolve a runtime profile
│   │   └── schema.py                # Dataclasses describing environment configuration
│   ├── gtnh_agent.egg-info          # Packaging metadata for installed distribution
│   │   ├── dependency_links.txt     # Package dependency links
│   │   ├── PKG-INFO                 # Built package metadata
│   │   ├── requires.txt             # Runtime dependency list
│   │   ├── SOURCES.txt              # File list for packaging
│   │   └── top_level.txt            # Top-level package names
│   ├── __init__.py                  # Top-level src package init
│   ├── integration                  # Phase 1 “Lab Integration” & env bridge glue
│   │   ├── adapters
│   │   │   └── m0_env_to_world.py   # Adapter: Phase 0 env config → normalized WorldState
│   │   ├── episode_logging.py       # Helpers to log/dump episodes in a consistent format
│   │   ├── __init__.py              # Integration package init
│   │   ├── phase1_integration.py    # End-to-end integration wiring for Phase 1
│   │   ├── testing                  # Integration-focused fakes
│   │   │   ├── fakes.py             # Fake integration structures (world, runtime, etc.)
│   │   │   └── __init__.py          # Testing package init
│   │   └── validators               # Various guardrails & invariants for integration
│   │       ├── __init__.py          # Validators package init
│   │       ├── planner_guardrails.py# Checks & constraints for planner outputs
│   │       ├── semantics_snapshots.py# Snapshot sanity checks for semantics state
│   │       ├── skill_integrity.py   # Verify skills against specs/packs/curriculum contracts
│   │       └── virtue_snapshots.py  # Virtue config / lattice snapshot verifier
│   ├── learning                     # M10 learning layer (skills + curriculum hooks)
│   │   ├── buffer.py                # ExperienceBuffer: store Experience episodes
│   │   ├── curriculum_hooks.py      # Hooks to feed curriculum state into learning
│   │   ├── evaluator.py             # Skill / policy evaluation utilities
│   │   ├── __init__.py              # Learning package init
│   │   ├── manager.py               # SkillLearningManager: orchestrates skill evolution
│   │   ├── schema.py                # Dataclasses for Experience, SkillCandidate, etc.
│   │   └── synthesizer.py           # SkillSynthesizer: episodes → SkillCandidate
│   ├── llm_stack                    # M2 LLM stack implementation & role harnesses
│   │   ├── backend_llamacpp.py      # Backend client for llama.cpp-style models
│   │   ├── backend.py               # Backend interface + base classes
│   │   ├── codegen.py               # Code-gen helpers (for skills, tools, etc.)
│   │   ├── config.py                # LLM stack configuration / presets wiring
│   │   ├── critic.py                # Critic role implementation (pre-execution evaluation)
│   │   ├── error_model.py           # ErrorModel implementation (post-execution analysis)
│   │   ├── __init__.py              # LLM stack package init
│   │   ├── json_utils.py            # Safe JSON parsing, salvage, and logging helpers
│   │   ├── log_files.py             # File-based logging helpers for LLM calls
│   │   ├── plan_code.py             # Plan-code role impl (Task → skills)
│   │   ├── planner.py               # Planner role impl (Goal → Task plan)
│   │   ├── presets.py               # Role presets (temperature, max_tokens, stop words)
│   │   ├── schema.py                # Local schema: ErrorContext, ErrorAnalysis, TraceSummary*
│   │   ├── scribe.py                # Scribe role impl (trace → summary)
│   │   └── stack.py                 # High-level stack wrapper: build all roles from config
│   ├── monitoring                   # M9 monitoring & tooling
│   │   ├── bus.py                   # Event bus for system-wide monitoring events
│   │   ├── controller.py            # Monitoring controller orchestrating collectors & sinks
│   │   ├── dashboard_tui.py         # Text UI dashboard for live monitoring
│   │   ├── events.py                # Event type definitions for monitoring bus
│   │   ├── __init__.py              # Monitoring package init
│   │   ├── integration.py           # Integration hooks between monitoring and runtime
│   │   ├── llm_logging.py           # Specialized logging for LLM calls / traces
│   │   ├── logger.py                # Central logging helpers / configuration
│   │   └── tools.py                 # Misc monitoring / diagnostic tools
│   ├── observation                  # M7 eyes & ears: observation pipeline
│   │   ├── encoder.py               # Encoders from raw snapshots → model-ready features
│   │   ├── __init__.py              # Observation package init
│   │   ├── pipeline.py              # Full observation pipeline orchestration
│   │   ├── schema.py                # Dataclasses for Observation, WorldState, etc.
│   │   ├── testing.py               # Helpers / fakes for observation tests
│   │   └── trace_schema.py          # PlanTrace and related trace structures
│   ├── planning                     # Thin planning glue between AgentLoop and skills
│   │   ├── adapter.py               # Adapter: LLM planner output → internal Task structures
│   │   └── dispatcher.py            # Dispatch from plan steps → skill invocations
│   ├── runtime                      # High-level runtime orchestration / error handling
│   │   ├── agent_runtime_main.py    # Main entrypoint: spin up env + agent + runtime loops
│   │   ├── bootstrap_phases.py      # Bootstraps phases (P0–P4) based on env config
│   │   ├── curriculum_learning_coordinator.py # Glue for curriculum ↔ learning integration
│   │   ├── curriculum_learning_triggers.py    # When to trigger learning passes based on episodes
│   │   ├── error_handling.py        # Centralized runtime error / failure handling helpers
│   │   ├── failure_mitigation.py    # Mitigation strategies when episodes / skills fail
│   │   ├── __init__.py              # Runtime package init
│   │   └── phase4_curriculum_learning_orchestrator.py # P4 orchestrator tying curriculum + learning
│   ├── semantics                    # M3: world semantics & tech graph
│   │   ├── cache.py                 # Cache for semantics queries (items, recipes, etc.)
│   │   ├── categorize.py            # Item/block categorization logic (groups, tags)
│   │   ├── crafting.py              # Craftability / recipe query engine
│   │   ├── ingest                   # Semantics ingestion utilities
│   │   │   └── __init__.py          # Ingest package init
│   │   ├── __init__.py              # Semantics package init
│   │   ├── loader.py                # Load semantics from config/gtnh_*.yaml/json
│   │   ├── schema.py                # Semantics dataclasses (items, blocks, recipes, tech nodes)
│   │   └── tech_state.py            # TechState logic & inference utilities
│   ├── skills                       # M5: skills runtime
│   │   ├── base                     # Hand-authored base skills implementations
│   │   │   ├── basic_crafting.py    # Implementation of basic_crafting.yaml SkillSpec
│   │   │   ├── chop_tree.py         # Implementation of chop_tree.yaml
│   │   │   ├── feed_coke_ovens.py   # Implementation of feed_coke_ovens.yaml
│   │   │   ├── feed_steam_boiler.py # Implementation of feed_steam_boiler.yaml
│   │   │   ├── __init__.py          # Base skills package init
│   │   │   ├── maintain_coke_ovens.py# Implementation of maintain_coke_ovens.yaml
│   │   │   ├── plant_sapling.py     # Implementation of plant_sapling.yaml
│   │   │   └── refill_water_tanks.py# Implementation of refill_water_tanks.yaml
│   │   ├── __init__.py              # Skills package init
│   │   ├── loader.py                # Load SkillSpec YAMLs → SkillSpec objects
│   │   ├── packs.py                 # Skill pack resolution (per tech_state + profile)
│   │   ├── registry.py              # SkillRegistry & registration decorators
│   │   └── schema.py                # SkillSpec dataclasses & metadata
│   ├── spec                         # Spec-layer contracts (types only, no heavy logic)
│   │   ├── agent_loop.py            # AgentLoop spec: AgentGoal, EpisodeResult, config
│   │   ├── bot_core.py              # Spec for BotCore interface (actions, world snapshots)
│   │   ├── experience.py            # Spec for Experience, Trace, replay format
│   │   ├── __init__.py              # Spec package init exporting core interfaces
│   │   ├── llm.py                   # Role-based LLM interfaces & response schemas
│   │   ├── monitoring.py            # Spec for monitoring bus / events types
│   │   ├── skills.py                # Skill & SkillRegistry Protocols + SkillInvocation
│   │   └── types.py                 # Shared type aliases & struct-like helpers
│   ├── testing                      # Testing helpers used across modules
│   │   └── __init__.py              # Testing package init / shared fixtures
│   ├── virtues                      # M4 virtue lattice system
│   │   ├── evaluator.py             # Evaluate plans/episodes against virtue metrics
│   │   ├── explain.py               # Turn virtue scores into human-readable explanations
│   │   ├── features.py              # Extract features used by virtue evaluators
│   │   ├── __init__.py              # Virtues package init
│   │   ├── integration_overrides.py # Integration-specific virtue overrides / hooks
│   │   ├── lattice.py               # Core virtue lattice, scoring & aggregation
│   │   ├── loader.py                # Load virtues.yaml into runtime objects
│   │   ├── metrics.py               # Aggregate virtue metrics for monitoring/learning
│   │   ├── sanity.py                # Sanity checks over virtue configs & scores
│   │   └── schema.py                # Dataclasses for virtue config & scores
│   └── world                        # High-level world model used by planning/semantics
│       └── world_model.py           # Lightweight predictive world model placeholder
├── tests                            # Full test suite: contracts, integration, properties
│   ├── conftest.py                  # Shared pytest fixtures across tests
│   ├── fakes                        # Fake implementations for tests (LLM, runtime, skills, etc.)
│   │   ├── fake_bot_core.py         # Fake BotCore for tests without real Minecraft
│   │   ├── fake_llm_stack.py        # Fake LLM stack returning deterministic JSON
│   │   ├── fake_runtime.py          # Fake runtime for integration tests
│   │   ├── fake_skills.py           # Fake skill implementations for testing plumbing
│   │   └── __init__.py              # Fakes package init
│   ├── __init__.py                  # Tests package init
│   ├── _skills_candidates           # Test fixture directory for synthesized skill candidates
│   ├── test_actions.py              # Tests for bot_core.actions & Action semantics
│   ├── test_agent_loop_stub.py      # Minimal AgentLoop stub test (no heavy integration)
│   ├── test_agent_loop_v1.py        # M8 AgentLoop integration smoke test
│   ├── test_architecture_integration.py # High-level architecture wiring sanity tests
│   ├── test_bot_core_impl.py        # Tests for BotCore implementation details
│   ├── test_curriculum_engine_basic.py  # Basic CurriculumEngine selection behavior
│   ├── test_curriculum_engine_phase.py  # Phase-specific curriculum behaviors
│   ├── test_curriculum_learning_integration.py # Curriculum ↔ learning integration tests
│   ├── test_curriculum_learning_orchestrator.py # Phase 4 orchestrator tests
│   ├── test_curriculum_learning_properties.py   # Property-style tests for learning behavior
│   ├── test_curriculum_loader.py   # CurriculumConfig loading & schema contract tests
│   ├── test_curriculum_projects.py # Long-horizon project unlock behavior tests
│   ├── test_curriculum_skill_view_policy.py # Skill-view policy interactions with curriculum
│   ├── test_env_loader.py          # Env loader and schema validation tests
│   ├── test_error_model_with_fake_backend.py # ErrorModelImpl JSON parsing and shape tests
│   ├── test_evaluator.py           # Learning evaluator correctness tests
│   ├── test_experience_buffer.py   # ExperienceBuffer persistence / roundtrip tests
│   ├── test_failure_mitigation.py  # Runtime failure mitigation strategies
│   ├── test_full_system_smoke.py   # End-to-end smoke test with fakes
│   ├── test_lab_integration_happy_path.py # Lab integration “happy path” episode test
│   ├── test_lab_integration_skill_view.py # Skill view integration in lab context
│   ├── test_llm_role_boundaries.py # Ensures roles obey llm_role_boundaries spec
│   ├── test_llm_stack_fake_backend.py # LLM stack behavior using FakeBackend
│   ├── test_m6_observe_contract.py # M6 observation contract tests (runtime ↔ observation)
│   ├── test_monitoring_controller.py # Monitoring controller behavior tests
│   ├── test_monitoring_dashboard_tui.py # Dashboard TUI rendering / interactions
│   ├── test_monitoring_event_bus.py # Monitoring event bus publish/subscribe behavior
│   ├── test_monitoring_logger.py   # Monitoring log adapter behavior
│   ├── test_nav_pathfinder.py      # Pathfinding correctness / edge cases
│   ├── test_observation_critic_encoding.py # Observation encoding for critic role
│   ├── test_observation_perf.py    # Perf tests for observation pipeline
│   ├── test_observation_pipeline.py # Observation pipeline functional tests
│   ├── test_observation_planner_encoding.py # Observation encoding for planner role
│   ├── test_observation_worldstate_normalization.py # WorldState normalization tests
│   ├── test_p0_p1_env_bridge.py    # Bridge between P0 env and P1 runtime
│   ├── test_phase012_bootstrap.py  # Bootstrap tests for phases 0–2
│   ├── test_phase0_runtime.py      # Phase 0 runtime smoke tests
│   ├── test_phase1_breakglass_no_plans.py # Breakglass path when planner fails
│   ├── test_phase1_integration_offline.py # Phase 1 offline integration tests
│   ├── test_q1_control_and_experience.py  # Q1 control / experience contract tests
│   ├── test_runtime_integration.py # Runtime integration across modules
│   ├── test_runtime_m6_m7_smoke.py # Legacy M6/M7 runtime smoke tests
│   ├── test_scribe_model_with_fake_backend.py # ScribeModelImpl JSON parsing tests
│   ├── test_semantics_caching_singleton.py   # Semantics cache singleton behavior
│   ├── test_semantics_categorization.py      # Item/block category logic tests
│   ├── test_semantics_craftability.py        # Craftability & recipe resolution tests
│   ├── test_semantics_tech_inference.py      # TechState inference from world / inventory
│   ├── test_semantics_tolerant_fallbacks.py  # Graceful semantics fallbacks
│   ├── test_semantics_with_normalized_worldstate.py # Semantics vs normalized WorldState
│   ├── test_skill_evaluator.py     # Skill-level evaluation logic tests
│   ├── test_skill_learning_manager.py # SkillLearningManager orchestration tests
│   ├── test_skill_learning_view.py # Views over skill learning state
│   ├── test_skill_loader.py        # SkillSpec loader & schema tests
│   ├── test_skill_packs_integrity.py # Cross-checks skill packs vs specs
│   ├── test_skill_packs.py         # Skill packs behavior tests
│   ├── test_skill_registry.py      # SkillRegistry versioning / deprecation tests
│   ├── test_synthesizer.py         # SkillSynthesizer contract tests (episodes → candidate)
│   ├── test_virtue_compare_plans.py# Virtue-based plan comparison behavior
│   ├── test_virtue_config_sanity.py# Virtue config validation tests
│   ├── test_virtue_hard_constraints.py # Hard constraints enforced by virtues
│   ├── test_virtue_lattice_basic.py# Basic virtue lattice math & structure tests
│   ├── test_virtue_overrides.py    # Virtue overrides per phase/curriculum tests
│   └── test_world_tracker.py       # BotCore world_tracker correctness tests
└── tools                            # User-facing scripts & audits (not core src)
    ├── agent_demo.py                # Minimal offline agent demonstration script
    ├── audit                        # LLM role & config audit utilities
    │   ├── check_llm_roles.py       # Check LLM role bindings vs llm_roles.yaml / spec
    │   ├── llm_role_usage_report.py # Summarize how roles are used across codebase
    │   ├── q0_auto_scan_output.json # Output example from q0_auto_scan audit
    │   └── q0_auto_scan.py          # Automated scan of LLM usage vs role boundaries
    ├── inspect_experiences.py       # CLI to inspect Experience replay.jsonl / episode guts
    ├── phase1_demo.py               # Demo runner for Phase 1 integration path
    └── smoke_botcore.py             # Quick smoke test runner for BotCore in isolation

```

---
## Where we are right now

**State of the codebase:**

- All tests pass with fakes & offline logic:
    
    - LLM stack contracts are stable (`llm_stack/*`, `spec/llm.py`, fake backend tests).
        
    - AgentLoop v1 exists and is wired into curriculum, skills, observation, virtues, learning, monitoring.
        
    - Curriculum engine, long-horizon projects, and tech-state gating work and are tested.
        
    - Skills, SkillRegistry, Synthesizer, and learning scaffolding are coherent and version-aware.
        
    - Monitoring / event bus / logging infrastructure exists and is test-covered.
        
    - Phase 0–4 runtime orchestration and integration tests all pass.
        

**High-level conclusion:**  
The architecture is _internally consistent_ in design space with fake worlds and fake LLMs. Now you need a **Lab Integration Phase**: a controlled environment where you hook in real LLMs, exercise the whole system end-to-end, and get serious observability / diagnostics before you touch a Forge mod.

---

## What “Lab Integration” is actually for

Lab Integration is a new phase that sits between “design-only correctness” and “live GTNH IPC hell.”

**Primary goals:**

1. Make it trivial to:
    
    - Run the full agent stack for a few episodes in a fake-but-realistic world.
        
    - Swap LLM backends and prompt presets.
        
    - Inspect _exactly_ what went wrong when something misbehaves.
        
2. Provide **multi-module diagnostics**:
    
    - Check config, skills, packs, curriculum, LLM wiring, and AgentLoop contracts.
        
    - Dump full episodes (observations, plans, actions, errors, virtue scores).
        
    - Trace LLM calls with consistent IDs across logs, traces, and experiences.
        
3. Enable **incremental integration of real LLMs**:
    
    - First per-role harnesses.
        
    - Then a “grandmaster” test that spins a full episode with a real model and fake GTNH.
        

Think of this phase as:

> _“Controlled environment where Future Knoah can ask: ‘What the hell did you just do?’ and actually get an answer.”_

---

## Core concepts & definitions

These are terms you’ll keep reusing in the Lab phase:

- **Episode**  
    Full run of AgentLoop from one `AgentGoal` through observation → planning → acting (fake) → trace → experience → logging.
    
- **Lab Episode**  
    Same as Episode, but with extra instrumentation:
    
    - Every LLM call logged.
        
    - Episode dumped to disk.
        
    - Virtue & curriculum context attached.
        
- **Agent Doctor**  
    A top-level diagnostic script that:
    
    - Validates configs.
        
    - Verifies skills & packs.
        
    - Smokes the LLM stack.
        
    - Runs a short lab episode.
        
    - Prints a structured “health report.”
        
- **Lab Pass**  
    A full sweep of the lab pipeline with a particular configuration:
    
    - Fake models only.
        
    - Real planner only.
        
    - All roles real, etc.
        
- **Grandmaster Test**  
    A single test / CLI workflow that:
    
    - Uses real LLMs for all roles.
        
    - Uses fake BotCore & fake world.
        
    - Runs multiple episodes.
        
    - Emits artifacts you can inspect offline (logs, episodes, metrics, audit summaries).

The Grandmaster Test becomes the **reference behavior** for the entire agent.  
It locks in:

- plan structure
    
- step structure
    
- role isolation
    
- trace format
    
- evaluation pathways

---

## Guardrails & design insights

- **Single source of truth for contracts**  
    Specs live in `src/spec/*`, and Lab tools should _only_ depend on these contracts. Don’t reach directly into random modules.
    
- **Everything gets an ID**  
    Episode, goal, LLM call, world snapshot. Thread them through from AgentLoop into:
    
    - monitoring
        
    - LLM stack
        
    - learning
        
    - episode dumps
        
- **Artifacts over logs**  
    Logs are infinite noise. Lab Phase should prioritize:
    
    - episode JSON
        
    - metrics JSONL
        
    - doctor reports
        
    - structured summaries  
        so you can diff runs and track regressions.
        
- **Tests mirror passes**  
    For each L-module:
    
    - One test for its _local behavior_.
        
    - One _integration test_ where it’s used in a small slice of the system.
        
    - Grandmaster tests cover the whole pipeline, not the internal details.
        
- **Lab ≠ Production**  
    Lab has:
    
    - Debug mode.
        
    - Heavy dumps.
        
    - Slow real-LLM prompts.  
        Production will later trim these down. Keep the knobs separate now (via env.yaml + CLI flags).


---

# LLM Roles 



- Planner
    
- PlanCode
    
- Critic
    
- ErrorModel
    
- Scribe
    


---

## 1. Planner – _Goal → Plan skeleton_

**What Planner _owns_**  
Conceptually: “Given a goal + world summary + available skills, what’s the outline of a plan?”

- **Input domain**
    
    - Encoded goal (`AgentGoal` → text / JSON)
        
    - World summary (normalized observation, tech tier, inventory / machines, etc.)
        
    - Skill _descriptions_ (from `SkillRegistry.describe_for_tech_state`, not implementations)
        
    - Constraints from curriculum / virtue (e.g. “no lava”, “minimize travel”)
        
- **Output contract**
    
    - JSON-ish plan, e.g.:
```json
{
  "goal_id": "...",
  "goal_text": "...",
  "tasks": [...],
  "steps": [...],     // optional preview
  "notes": "...",
  "error": null
}

```

- - Under the hood:
        
        - `PlannerTaskJSON` list for `tasks`
            
        - Fits into `PlanTrace.plan` nicely
            
- **Decisions it’s allowed to make**
    
    - Decompose goal into **tasks** (sub-goals, “what to do first / next”)
        
    - Choose **which skills/categories** are appropriate at a high level (e.g. “use wood-gathering skills”)
        
    - Suggest **ordering** / basic dependency structure
        

**What Planner _does not own_**  
If you let the Planner do these, you’ve lost separation of concerns:

- ❌ Direct skill implementations (no “step: move to (x,y,z), break block” etc.)
    
- ❌ Minecraft IPC / movement / block targeting (that’s BotCore)
    
- ❌ Tech graph truth (it _consumes_ semantics; it doesn’t maintain them)
    
- ❌ Virtue scores / enforcement (it can be _guided_ by them, but not compute them)
    
- ❌ Evaluation / veto (“Is this safe?” is Critic’s job)
    
- ❌ Error classification (“why did this fail?” is ErrorModel)
    

Think of Planner as: **“structured hallucination about how to reach the goal,”** not the law.

---

## 2. PlanCode – _Task → Skill steps_

**What PlanCode _owns_**  
This is the “take this task and spell out the actual skill invocations” layer.

- **Input domain**
    
    - `PlanCodeRequest`:
        
        - `task`: one task from Planner’s output, as JSON
            
        - `world_summary`: same class of world summary as Planner gets
            
    - Skill specs:
        
        - Names, parameter schemas, preconditions/effects (from skills YAML via registry)
            
- **Output contract**
    
    - `PlanCodeResponse` / `PlannerPlanJSON`:
```json
{
  "steps": [
    {"skill": "feed_coke_ovens", "params": {...}, "expected_outcome": "..."},
    ...
  ],
  "notes": "...",
  "raw_text": "...",
  "error": null
}

```

- - These map directly into `SkillInvocation` and then into `PlanTrace.steps` (via dispatcher).
        
- **Decisions it’s allowed to make**
    
    - Choose **which skill** to invoke for a given task
        
    - Choose **parameters** for each skill (numbers, item names, counts, etc.)
        
    - Decide **micro-ordering** within that task (which skill first, second, etc.)
        
    - Optionally annotate expected outcomes for later critique / error analysis
        

**What PlanCode _does not own_**

- ❌ Raw actions (move/break/place) – that’s **skill implementations** & BotCore
    
- ❌ Choosing the _global_ goal – that’s Curriculum + AgentLoop
    
- ❌ Choosing the coarse plan tasks – that’s Planner
    
- ❌ Validating safety / virtue – that’s Critic
    
- ❌ Evaluating execution failures – that’s ErrorModel
    

PlanCode is: **“Given we _are_ doing this task, which skills & params should we fire?”**

---

## 3. Critic – _Pre-execution plan evaluation_

**What Critic _owns_**  
Critic answers: “Should we actually run this plan, and if not, what’s wrong with it?”

- **Input domain**
    
    - `CriticRequest`:
        
        - `trace: PlanTrace` (plan + intended steps, observation, virtue hints if present)
            
        - `context_id: str` (e.g. “steam_age”, curriculum phase, virtue context)
            
    - This is **before** any real execution.
        
- **Output contract**
    
    - `CriticResponse`:
```python
CriticResponse(
    ok: bool,
    summary / critique: str,
    suggested_modifications: Dict[str, Any],
    failure_type: Optional[str],
    severity: Optional[str],
    fix_suggestions: List[str],
    raw_text: str,
)

```
- - Key fields that the rest of the system cares about:
        
        - `ok` (hard gate)
            
        - `failure_type` (shared surface with ErrorModel)
            
        - `severity`
            
        - `fix_suggestions`
            
- **Decisions it’s allowed to make**
    
    - Accept / reject plan (`ok` flag)
        
    - Classify plan issues: “safety risk”, “wasteful”, “infeasible”, etc.
        
    - Suggest **modifications**: “require water buffer first,” “reduce volume,” etc.
        
    - Provide human/LLM-readable explanation for logs / dashboards
        

**What Critic _does not own_**

- ❌ Generating the plan (that’s Planner / PlanCode)
    
- ❌ Executing anything
    
- ❌ Persisting virtue metrics or updating virtue config (Virtue module)
    
- ❌ Retrofitting skills / changing their code (Learning & Skills handle that)
    
- ❌ Episode outcome analysis (that’s ErrorModel)
    

Critic is your **alignment/sanity checkpoint** _before_ the agent does something regrettable.

---

## 4. ErrorModel – _Post-execution outcome analysis_

**What ErrorModel _owns_**  
ErrorModel answers: “Given what actually happened, how and why did this fail?”

Two paths now:

1. **Agent-level outcome evaluation**
    
    - `ErrorModelImpl.evaluate(payload: Dict[str, Any]) -> Dict[str, Any>`
        
    - Payload typically includes:
        
        - `plan`, `observation`, `trace`, `virtue_scores`, `context`
            
    - Returns:
```json
{
  "failure_type": "execution_failure" | ... | null,
  "severity": "low" | "medium" | "high" | null,
  "fix_suggestions": ["...", "..."],
  "notes": "explanation",
  "raw_text": "..."
}

```

1. - This is what Q1 experience / curriculum / failure mitigation will read.
        
2. **Legacy infra-error analysis**
    
    - `analyze_failure(ctx: ErrorContext) -> ErrorAnalysis`
        
    - For “LLM call failed, why?”:
        
        - Classification, summary, suggested_fix, retry_advised.
            

- **Shared surface with Critic**
    
    - `failure_type`, `severity`, `fix_suggestions` are deliberately aligned
        
    - That allows a common reducer to map both into `PlanEvaluation` / `EpisodeOutcome`
        

**What ErrorModel _does not own_**

- ❌ Generating plans
    
- ❌ Approving plans (that’s Critic)
    
- ❌ Raw environment observation encoding (Observation pipeline)
    
- ❌ Direct retry loops / mitigation policy (runtime / failure_mitigation)
    
- ❌ Metrics persistence & curriculum adjustment (Learning / Curriculum, not here)
    

ErrorModel is the **post-mortem brain**: “what broke, how bad, how to fix next time?”

---

## 5. Scribe – _Summarization / compression_

**What Scribe _owns_**  
The Scribe is for turning messy episodes into something compact and searchable.

- **Input domain**
    
    - `TraceSummaryRequest` (llm_stack schema):
```python
TraceSummaryRequest(
    trace: Any,
    purpose: str,  # "context_chunk", "human_summary", etc.
)

```

- - Or upstream helper: `summarize_episode_for_memory(trace)` (non-LLM, deterministic)
        
    - LLM Scribe is built to handle full traces, plans, outcomes, etc.
        
- **Output contract**
    
    - `TraceSummaryResponse` / `ScribeResponse`:
        
        - `summary: str`
            
        - `keywords: List[str]`
            
        - `suggested_tags: List[str]`
            
        - `raw_text: str`
            
- **Decisions it’s allowed to make**
    
    - Pick what details to keep vs drop in the summary
        
    - Choose useful **keywords** / **tags** for replay / curriculum / search
        
    - Format chunk text for feeding into future prompts (episodic memory, context)
        

**What Scribe _does not own_**

- ❌ Deciding which episodes to summarize (Learning / coordinator decides)
    
- ❌ Evaluation / grading of episodes (that’s Critic / ErrorModel)
    
- ❌ Curriculum decisions (which goals next, etc.)
    
- ❌ Skill evolution and promotion (Learning Manager + SkillRegistry)
    
- ❌ Ground truth tech semantics (that’s Semantics)
    

Scribe is the **compression layer**: it shapes history into something the agent can actually reuse.

---

## 6. CodeModel – _Not a sixth role, just a tool_

Just to close the loop on the Synthesizer confusion:

- `CodeModel` (from `spec.llm`) is:
    
    - An interface for: `propose_skill_implementation(skill_spec, examples) -> str`
        
    - Used by **SkillSynthesizer** and maybe dev tools.
        
- It does **not**:
    
    - Take `PlanTrace`
        
    - Participate in the core 5-role loop
        
    - Own any safety / curriculum surfaces
        

Think of it as a **power tool in M10**, not a runtime role.

---

## Wiring consequences for Qwen

When you hook Qwen up, you want:

- **One harness per role**, each with:
    
    - Its own prompt template (tuned for _that_ question)
        
    - Its own temperature / max_tokens / stop sequences
        
    - Its own JSON schema expectations
        
- **Logs tagged with**:
    
    - `episode_id`, `context_id`, and **role** (`planner`, `plan_code`, `critic`, `error_model`, `scribe`)
        
- **No role doing someone else’s job**:
    
    - Planner never does Critic’s veto logic.
        
    - ErrorModel never mutates curriculum.
        
    - Scribe never decides policy.
        

That keeps the system debuggable when something goes sideways:

- “Did we fail because **Planner** hallucinated garbage?”
    
- Or because **PlanCode** bound the wrong skill?
    
- Or because **Critic** let something unsafe pass?
    
- Or because **ErrorModel** mis-classified the failure?
    
- Or because **Scribe** compressed away the one key detail next episode needed?
    

Right now your boundaries are actually pretty clean. The main job as you wire Qwen is:

- Keep these ownership lines intact
    
- Centralize the LLM harness config
    
- Make sure every role’s inputs/outputs are fully JSON-checked before they reach the rest of the stack

---

# Non-Negotiable Constraints

# **1. Canonical Contracts (The Inviolable Interfaces)**

These are the shapes every part of the agent must obey.  
These contracts define the vocabulary the modules use to talk to each other.

## **1.1 LLM Role Contracts**

Roles: **Planner, PlanCode, Critic, ErrorModel, Scribe.**  
Each produces strictly-typed JSON outputs.

### **Planner → PlannerPlanJSON**

- Tasks & subplans
    
- High-level “steps”
    
- No code, no actions
    

### **PlanCode → PlanCodeResponse**

- Bind tasks to skills
    
- Produce `SkillInvocation` sequences
    
- No planning, no goal selection, no critique
    

### **Critic → CriticResponse**

- Evaluate the plan
    
- `ok / failure_type / severity / fix_suggestions`
    
- Cannot modify the plan
    

### **ErrorModel → ErrorAnalysis**

- Interpret runtime failures
    
- Classifies: json errors, execution failures, etc.
    
- Cannot propose tasks, cannot rewrite plan
    

### **Scribe → TraceSummaryResponse**

- Episode summary for compression/memory
    
- Zero influence on control flow
    

---

## **1.2 Episode Contracts**

### **PlanTrace**

- `steps` list
    
- metadata.inc context
    

### **EpisodeResult**

- goal
    
- plan
    
- trace
    
- outcome
    

### **ExperienceEpisode**

- input → EpisodeResult → Experience
    
- stored in replay buffer
    
- consumed by learner (M10) and curriculum (M11)
    

---

## **1.3 Skill Contracts**

### **SkillSpec**

- YAML describing: parameters, preconditions, effects, metadata
    
- Versioned
    
- Status: active/candidate/deprecated
    

### **SkillInvocation**

- Generated by PlanCode
    
- What the agent actually _does_
    

### **SkillRegistry**

- Tracks active specs, candidates, deprecated versions
    
- Binds Python implementations to SkillSpecs
    
- Enforces consistency and prevents deprecated skill use
    

---

## **1.4 Observation Contract**

### **WorldState**

- Normalized snapshot (M6 pipeline)
    
- Includes tech inference features
    
- Carries evidence for semantic loaders and curriculum
    

---

## **1.5 TechState Contract**

- `active` tier
    
- `unlocked` tiers
    
- `evidence` mapping
    
- Curriculum engine uses this directly
    

---

# **2. Runtime Modes Matrix**

The system must behave predictably depending on whether Minecraft + LLM are real or fake.

|Mode|LLM|World|BotCore|Learning|Monitoring|
|---|---|---|---|---|---|
|**offline_fake**|fake|fake|fake|disabled|minimal|
|**lab_integration**|**real**|fake|fake|partial|full|
|**live_gtnh**|real|**actual IPC**|real|full|full|

This is critical for debugging. Every lab test depends on correct mode-switching.

---

# **3. ### **Identity Threading Rules**

Every major artifact carries:

- `episode_id`
    
- `context_id` (curriculum/tech/phase)
    
- `role_id` (planner, plan_code, critic, error, scribe)
    
- `call_id` (unique per LLM invocation)
    
- `trace_id` if relevant
    

Thread these through:

- AgentLoop events
    
- monitoring bus
    
- ExperienceEpisode
    
- PlanTrace
    
- LLM logs
    
- replay buffer
    
- episode dumps
    
- doctor reports
    

Identity threading is what keeps multi-role chaos from collapsing into a formless soup of logs.

---

# **4. Failure Taxonomy & Role Isolation Guarantees**

## **4.1 Three Classes of Failure**

### **A. Planning Failures**

Detected by Critic

- Bad decomposition
    
- Unsafe sequences
    
- Lack of required skills
    
- Missing invariants
    

### **B. Execution Failures**

Detected by ErrorModel

- Skill actions fail
    
- Plan steps mis-executed
    
- World constraints violated
    

### **C. System Failures**

Detected by stack-level guards

- JSON parse errors
    
- Role bleed
    
- Uncaught exceptions
    
- Model-format mismatches
    

Automation and diagnostics depend on treating these as separate phenomena.

---

## **4.2 Role Isolation Guarantees**

Each role has **hard walls**:

- Planner cannot output actions or code.
    
- PlanCode cannot choose goals or invent tasks.
    
- Critic cannot rewrite plan or produce code.
    
- ErrorModel cannot create tasks or skill steps.
    
- Scribe cannot influence planning or execution.
    

This isolation is central to maintain reliability when swapping real LLMs in.

---

# **Unified Episode Flow (The One True Chain)**

The entire system operates in the following sequence:
```scss
AgentLoop.run_episode
  → observe.worldstate (M6)
  → planner.llm_call (M2)
  → plan_code.llm_call (M2)
  → critic.llm_call (M2)
     → retry path if needed
  → dispatcher(skill invocations) (M5/M7)
  → bot_core.exec (fake or real) (M7)
  → trace logging (M8)
  → error_model.llm_call (M2)
  → scribe.llm_call (M2)
  → ExperienceEpisode (M10)
  → ExperienceBuffer.append

```

This chain must never be altered in Lab Integration.

---

# **Versioning Model**

Everything versioned:

- SkillSpecs
    
- Prompt templates
    
- Curriculum phases
    
- Observation formats
    
- Episode dumps
    

You need stable baselines when LLM behavior drifts over time.

---

# **Definitive Guarantees Going Into Lab Integration**

- Codebase is fully test-clean.
    
- Architecture is coherent, modular, and stable.
    
- No new roles are needed; the five roles are final.
    
- All modules are defined with clear ownership.
    
- All future failures will be LLM configuration, not architecture.

---

# **The Lab / Production Separation Principle**


### Lab Mode:

- maximal visibility
    
- heavy logs
    
- debug prompts
    
- strict schema checking
    
- truncated episodes
    
- deterministic fake world
    

### Production Mode:

- minimal logs
    
- stable prompts
    
- no trace dumps
    
- optimized observation pipeline
    
- real Minecraft world

---

# **Canonical Definition of “Fake World”**

### What the fake world MUST guarantee:

- deterministic worldstate
    
- deterministic skill outcomes
    
- deterministic failure modes
    
- deterministic movement & collision
    
- reproducible across tests
    
- minimal but complete geometry / inventory representation
    

### And what it MUST NOT attempt:

- fidelity
    
- navigation realism
    
- generating surprises

---

# **The Three Roles of Episode Dumps**


1. **Debug visibility:** the truth of what actually happened
    
2. **LLM supervision:** material for improving prompts/contracts
    
3. **Learning substrate:** material consumed by M10 Skill Learning

---
# **Explicit Reset Conditions for Lab Episodes**

### **Episode Stop Conditions**

- reached `max_steps`
    
- reached `max_skill_invocations`
    
- planner produced no viable tasks
    
- plan_code failed to bind required skills
    
- critic returned `ok=False` with severity ≥ threshold
    
- execution failure flagged by ErrorModel
    
- catastrophic system failure (parse errors, schema errors)
    
- curriculum says:
    
    - goal invalid
        
    - phase complete

---

### **LLM Call Lifecycle**

1. **Build request**  
    Role harness produces a strict Python `dict`.
    
2. **Validate request**  
    Check it conforms to `spec.llm` before touching a model.
    
3. **Serialize**  
    Convert to text with a controlled prompt template + version tag.
    
4. **Send to backend**  
    Qwen, llama.cpp, LM Studio, whatever.
    
5. **Receive raw text**  
    Don’t trust a single character.
    
6. **Parse**  
    Run through `safe_json_loads` with salvage heuristics.
    
7. **Validate response**  
    Must match the schema for that role or fail _loudly_.
    
8. **Log**  
    Dump final structured result + raw text to `logs/llm/<timestamp>_<episode>_<role>.json`.
    
9. **Return structured object**  
    No strings leaking upward.


---

# **Canonical Definition of Lab Mode vs Production Mode**

You describe the spirit but not the _formal_ difference.

This must be part of the primer.  
Otherwise the next chat thread might treat Lab mode like a slower copy of Production mode, which is wrong.

### **Lab Mode**

- fake world
    
- deterministic outcomes
    
- real LLMs
    
- heavy logs
    
- schema validation hard-on-error
    
- truncated episodes
    
- doctor tools enabled
    
- episode dumps mandatory
    
- no learning promotions
    

### **Production Mode**

- real world (Forge IPC)
    
- non-deterministic outcomes
    
- reduced logging
    
- no forced dumps
    
- mitigations active
    
- full-length episodes
    
- learning promotions allowed

---
## **What Lab Integration Proves**

### Lab Integration exists to prove six non-negotiable invariants:

**1. Role Isolation Invariant**  
Each LLM role obeys its contract, never bleeds across boundaries, and produces only schema-valid JSON.

**2. Control-Loop Stability Invariant**  
The unified episode chain never deadlocks, infinite-loops, or misroutes control signals even when a real model does stupid real-model things.

**3. Observability Invariant**  
Every episode can be reconstructed offline using only:

- episode JSON,
    
- LLM logs,
    
- and metrics.
    

(If this isn’t true, production debugging becomes a full-time career.)

**4. Deterministic-Fake-World Invariant**  
For the same seed, Lab Mode must produce identical results episode to episode.  
If this breaks, you’ll never know whether a bug is in the agent, the model, or the fake world.

**5. Prompt-Versioning Invariant**  
No prompt template can silently change the behavior of a role without updating its version tag and test expectations.

**6. Curriculum-Goal Invariant**  
Curriculum always returns a valid goal or an explicit failure state; never `None` mid-run except when deliberately allowed.

---
# Checklist: What must be true before wiring in real Qwen?

- All schemas validated across spec → llm → loop → experience
    
- Fake-world deterministic
    
- Episode dumper working
    
- CLI viewer working
    
- Unified logging with episode_id/context_id
    
- Prompt templates centralized + versioned
    
- Per-role harness tests pass with fake backend
    
- Doctor script produces zero warnings