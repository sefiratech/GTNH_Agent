# GTNH Agent · Architecture Overview (M1)

## 1. Purpose

This document defines the **canonical architecture** and **core interfaces** for the GTNH Agent.

M1 is a **design module** only. It defines:

- Components (bot body, agent loop, skills, LLM stack, learning hooks)
- Data types (world state, observations, actions)
- Interfaces between modules

No Minecraft or LLM implementation lives here. Everything else in the project is expected to obey this spec.

---

## 2. Component View

Top-level component diagram:

```text
+---------------------+
| EnvProfile (M0)     |
+---------------------+
           |
           v
+---------------------+                        +---------------------+
| BotCore (M6)        |                        | LLM Stack (M2)      |
| - get_world_state() |                        | - PlannerModel      |
| - execute_action()  |                        | - CodeModel         |
+----------+----------+                        | - CriticModel       |
           |                                   +----------+----------+
           v                                              |
+---------------------+                                   |
| ObservationEncoder  |                                   |
| (M7)                |                                   |
+----------+----------+                                   |
           |                                              v
           v                                   +--------------------+
      +-----------------------+               | VirtueLattice (M4) |
      | AgentLoop (M8)        |               | WorldSemantics(M3) |
      | - set_goal()          |               +--------------------+
      | - step()              |
      +-----------------------+
                   |
                   v
          +-----------------------+
          | SkillRegistry (M5)    |
          +----------+------------+
                     |
                     v
             +--------------------+
             | ExperienceRecorder |
             | & SkillLearner     |
             |      (M10)         |
             +--------------------+
Once this spec is stable, other modules (M2–M10) can evolve independently...

On CPU-only (Qwen 7B Q4), typical call latency ~45–90s for current prompts. Acceptable for offline planning; M8 should cache & reuse plans.

### Backend Optimization Queue

Backend performance tuning is intentionally deferred until after the Phase 2 runtime is stable.

When the full loop is running, revisit:

- `n_batch` per model (planner, codegen, critic)
- `n_threads` defaults vs explicit pinning
- `n_gpu_layers` for each quant / VRAM configuration
- Context length vs memory usage

Perf measurements will use:

- `scripts/smoke_llm_stack.py --perf` (wall-clock)
- llama-cpp `llama_perf_context_print` output in the logs
