## How to run this phase without losing your mind

You’re aiming for **maximum observability with minimal chaos**. Good. To keep continuity and avoid design drift, I’d do this:

### 1. Treat Lab Integration as mini-phases: L-Pass A, B, C

Instead of a soup of tasks, think in **passes**:

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

