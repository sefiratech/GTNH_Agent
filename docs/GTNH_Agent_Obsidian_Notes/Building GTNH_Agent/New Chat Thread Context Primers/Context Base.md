Here is an overview of the entire project for your context:

Phase 0:

M0 - environment_foundation

**Purpose:**

Lock in the actual environment & runtimes.

- Define:

- MC 1.7.10 + Forge 10.13.4.1614 + GTNH 2.8.1 run profile

- Decision: external bot client vs in-process Forge mod with IPC

- Hardware constraints for local LLMs

- **Dependencies:** None

- **Difficulty:** ⭐

- **Scalability notes:**

- Document this in a single config file / README; future changes (new model, new server host) should not touch code.

M1 - agent_architecture_spec

**Purpose:**  
Unify Mineflayer + Voyager insights into a **single architecture spec**.

- Extract from Mineflayer:
    
    - Bot lifecycle
        
    - World model
        
    - Pathfinding
        
    - Action abstraction
        
- Extract from Voyager:
    
    - Planner → Skill library → Execution loop
        
    - Reflection & learning
        
- **Dependencies:** `M0`
    
- **Difficulty:** ⭐⭐
    
- **Scalability notes:**
    
    - Produce one canonical architecture doc: diagrams + interfaces.
        
    - This is the contract everything else conforms to.


Phase 1

M2 - llm_stack_local

**Purpose:**  
Provide reusable interfaces around local models.

- Implement:
    
    - `PlannerModel`: high-level plan generation
        
    - `CodeModel`: skill/code generation
        
    - `CriticModel`: evaluation / refinement
        
- Unified tool schema:
    
    - Input: structured state / goal
        
    - Output: JSON plan / skill spec, no direct MC calls
        
- **Dependencies:** `M1`
    
- **Difficulty:** ⭐⭐–⭐⭐⭐
    
- **Scalability/perf:**
    
    - Centralize model loading & caching.
        
    - Make batch calls possible.
        
    - Log prompts/responses for replay.

M3 - world_semantics_gtnh

**Purpose:**  
Define GTNH tech + world understanding as **data + logic**.

- Data layer (config files):
    
    - Block categories (ores, machines, cables, etc.)
        
    - Item categories (plates, circuits, tools)
        
    - Tech states & prereqs (LV steam, MV, etc.)
        
- Logic layer (Python):
    
    - `infer_tech_state(inventory, machines)`
        
    - `suggest_next_targets(tech_state)`
        
    - `craftable_items(inventory, known_recipes)`
        
- **Dependencies:** `M1`
    
- **Difficulty:** ⭐⭐⭐⭐
    
- **Scalability/perf:**
    
    - Keep recipes & categories in JSON/YAML, not code.
        
    - Cache derived graphs (like tech dependency DAGs).

M4 - virtue_lattice

**Purpose:**  
Encapsulate your Sefirot-based virtues as a reusable scoring layer.

- Define:
    
    - Virtue nodes: Efficiency, Safety, Sustainability, etc.
        
    - Configurable weights per context (e.g., early LV vs late HV)
        
- APIs:
    
    - `score_plan(plan, context) -> dict[virtue -> score]`
        
    - `compare_plans(plans, context) -> best_plan`
        
- **Dependencies:** `M3` (for context & environment semantics)
    
- **Difficulty:** ⭐⭐–⭐⭐⭐
    
- **Scalability/perf:**
    
    - Pure functions, stateless, easy to unit test.
        
    - Configurable weights → you can tune without code changes.

M5 - skill_registry

**Purpose:**  
Central place for skill definitions and metadata.

- Skill spec:
    
    - Name, parameters
        
    - Preconditions (what world/tech state is required)
        
    - Effects (changes in world/tech state)
        
    - Tags (e.g., mining, crafting, building)
        
- LLM interaction:
    
    - Planner only sees skill metadata, not raw code.
        
    - Skill implementations live as Python methods or small scripts.
        
- **Dependencies:** `M1`, `M3`
    
- **Difficulty:** ⭐⭐–⭐⭐⭐
    
- **Scalability/perf:**
    
    - Skills registered via decorators or config files.
        
    - Easy to version and deprecate skills over time.

Phase 2:

M6 - bot_core_1_7_10

**Purpose:**  
Provide a stable, testable “body” that can be used by any controller.

- Capabilities:
    
    - Connect/keepalive
        
    - World tracking (chunks, entities)
        
    - Navigation (A* or similar)
        
    - Actions:
        
        - Move, jump, break block, place block, use item, interact with tile entities
            
- API:
    
    - `observe() -> RawWorldSnapshot`
        
    - `execute_action(Action) -> Result`
        
- **Dependencies:** `M0`, `M1`
    
- **Difficulty:** ⭐⭐⭐⭐
    
- **Scalability/perf:**
    
    - Keep logic modular: pathfinding, inventory, world tracking as submodules.
        
    - Limit unnecessary packet decoding; cache what you can.

M7 - observation_encoding

**Purpose:**  
Map `RawWorldSnapshot` from `M6` into semantic state used by LLMs & planners.

- Functions:
    
    - `encode_for_planner(raw_snapshot, tech_state) -> JSON`
        
    - `encode_for_critic(trace) -> JSON`
        
- Uses:
    
    - `M3` (semantics)
        
    - `M4` (virtues context)
        
- **Dependencies:** `M3`, `M6`
    
- **Difficulty:** ⭐⭐–⭐⭐⭐
    
- **Scalability/perf:**
    
    - Keep encodings compact. Summaries + key entities, not entire chunks.
        
    - Enforce stable schema to avoid breaking old skills.

Phase 3:

M8 - agent_loop_v1

**Purpose:**  
Implement the core loop: observe → plan → choose skills → act → evaluate.

- High-level algorithm:
    
    1. `state = observe()`
        
    2. `tech_state = infer_tech_state(state)`
        
    3. `plan = planner_model.call(state, tech_state, skill_registry, virtues)`
        
    4. Decompose plan into skill invocations
        
    5. Execute via `bot_core_1_7_10`
        
    6. Log result for learning (`M10`)
        
- Strict separation:
    
    - No direct packet calls here.
        
    - No GTNH-hardcoded weirdness here; that lives in `M3` and `M5`.
        
- **Dependencies:**
    
    - `M2` (LLM stack)
        
    - `M3` (world semantics)
        
    - `M4` (virtues)
        
    - `M5` (skills)
        
    - `M6` (bot core)
        
    - `M7` (observation encoding)
        
- **Difficulty:** ⭐⭐⭐⭐
    
- **Scalability/perf:**
    
    - Design as a state machine with clear states (Idle, Planning, Executing, Recovering).
        
    - Rate-limit LLM calls, reuse plans until invalidated.

M9 - monitoring_and_tools

**Purpose:**  
Give you observability and a control surface before the system gaslights you.

- Features:
    
    - Structured logs (JSON)
        
    - Web or TUI dashboard:
        
        - World overview
            
        - Current plan & skills
            
        - Virtue scores
            
        - Tech state
            
    - Manual controls:
        
        - Pause, step, cancel plan, inspect memory
            
- **Dependencies:** `M8`
    
- **Difficulty:** ⭐⭐–⭐⭐⭐
    
- **Scalability/perf:**
    
    - Central logger used by all modules.
        
    - Minimal UI first; upgrade visuals later.


Phase 4:

M10 - skill_learning

**Purpose:**  
Voyager-style learning: derive new skills from experience and refine existing ones.

- Components:
    
    - Experience buffer:
        
        - `{state, goal, plan, actions, outcomes, virtue_scores}`
            
    - LLM-based synthesizer:
        
        - Turn repeated success traces into new skill definitions
            
    - Evaluator:
        
        - Compare new vs existing skills on:
            
            - Success rate
                
            - Cost (time, resources)
                
            - Virtue scores
                
- **Dependencies:** `M8` (loop), `M2` (LLMs), `M5` (skill registry), `M4` (virtue scoring)
    
- **Difficulty:** ⭐⭐⭐⭐⭐
    
- **Scalability/perf:**
    
    - Learning should be offline or scheduled, not constant.
        
    - Skills versioned and can be rolled back if regressions appear.


M11 - gtnh_curriculum_and_specialization

**Purpose:**  
Turn the generic learning agent into a **GTNH-native progression engine**.

- Define:
    
    - Curricula per phase:
        
        - Early LV goals
            
        - Steam infra goals
            
        - MV automation goals
            
    - Long-horizon projects:
        
        - Stargate, high-tier reactors, etc.
            
- The curriculum is:
    
    - A sequence of target tech states
        
    - Each with:
        
        - Reward shaping (virtue weight tweaks)
            
        - Suggested skills to prioritize / learn
            
- **Dependencies:** `M3`, `M5`, `M8`, `M10`
    
- **Difficulty:** ⭐⭐⭐⭐⭐
    
- **Scalability/perf:**
    
    - Curriculum is config, not code.
        
    - Multiple curricula can be swapped (e.g. “eco base”, “speedrun”, “aesthetic build”).


Shortcut View:
# **Phase P0 — Foundations**

|Module|Name|Difficulty|Est. Time|Notes|
|---|---|---|---|---|
|**M0**|environment_foundation|⭐|0.5–2 days|Lock runtime, modpack, IPC choice|
|**M1**|agent_architecture_spec|⭐⭐|2–4 days|Full architecture doc|

### **Phase P0 Total:**

**Difficulty Avg:** ⭐⭐  
**Time:** ~3–6 days

---

# **Phase P1 — Offline Core Pillars (No Minecraft)**

|Module|Name|Difficulty|Est. Time|Notes|
|---|---|---|---|---|
|**M2**|llm_stack_local|⭐⭐–⭐⭐⭐|3–7 days|Local models, prompt tooling|
|**M3**|world_semantics_gtnh|⭐⭐⭐⭐|7–14 days|Tech tree + ontology mapping|
|**M4**|virtue_lattice|⭐⭐–⭐⭐⭐|3–6 days|Scoring/weights system|
|**M5**|skill_registry|⭐⭐–⭐⭐⭐|3–6 days|Skill definitions, metadata|

### **Phase P1 Total:**

**Difficulty Avg:** ⭐⭐⭐  
**Time:** ~2–4 weeks

---

# **Phase P2 — Minecraft Integration Layer**

|Module|Name|Difficulty|Est. Time|Notes|
|---|---|---|---|---|
|**M6**|bot_core_1_7_10|⭐⭐⭐⭐|2–4 weeks|Pathfinding, inventory, world tracking|
|**M7**|observation_encoding|⭐⭐–⭐⭐⭐|3–7 days|Convert raw MC data → semantic state|

### **Phase P2 Total:**

**Difficulty Avg:** ⭐⭐⭐⭐  
**Time:** ~3–5 weeks

---

# **Phase P3 — Agent Orchestration & Tooling**

|Module|Name|Difficulty|Est. Time|Notes|
|---|---|---|---|---|
|**M8**|agent_loop_v1|⭐⭐⭐⭐|1–2 weeks|Full observe → plan → act|
|**M9**|monitoring_and_tools|⭐⭐–⭐⭐⭐|3–7 days|Logs, dashboards, step controls|

### **Phase P3 Total:**

**Difficulty Avg:** ⭐⭐⭐⭐  
**Time:** ~2–3 weeks

---

# **Phase P4 — Learning & Specialization**

|Module|Name|Difficulty|Est. Time|Notes|
|---|---|---|---|---|
|**M10**|skill_learning|⭐⭐⭐⭐⭐|2–4 weeks|Voyager-style skill synthesis|
|**M11**|gtnh_curriculum_and_specialization|⭐⭐⭐⭐⭐|multi-week ongoing|Long-horizon GTNH progression logic|

### **Phase P4 Total:**

**Difficulty Avg:** ⭐⭐⭐⭐⭐  
**Time:** ~4–8+ weeks (ongoing beyond initial build)

---

# **Grand Totals (First-Pass Implementation)**

|Phase|Difficulty Avg|Total Time|
|---|---|---|
|**P0**|⭐⭐|3–6 days|
|**P1**|⭐⭐⭐|2–4 weeks|
|**P2**|⭐⭐⭐⭐|3–5 weeks|
|**P3**|⭐⭐⭐⭐|2–3 weeks|
|**P4**|⭐⭐⭐⭐⭐|4–8+ weeks|

---

# **Overall Estimate**

**Minimum:** ~11 weeks  
**Expected:** ~14–18 weeks  
**Ambitious agent with learning + GTNH specialization:** ~20–30 weeks ongoing refinement


File Structure:

