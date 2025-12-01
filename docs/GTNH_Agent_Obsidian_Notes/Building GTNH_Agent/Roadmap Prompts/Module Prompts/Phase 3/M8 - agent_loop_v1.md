Module Prompt:

Purpose: Zoom in on one module for detailed design.

Template: Context: This is part of the GTNH Agent project. Maintain consistency with prior outputs.

Focus on the M8 module from our map.

Explain its architecture and function in detail.

Include Python code snippets with comments on each line and system diagrams if relevant.

Suggest how to test or simulate it locally.

List specific libraries, APIs, or repos worth examining.

At the end, list criteria that would indicate this module is complete or working as intended.

Output: a detailed technical blueprint for one module.


Module:

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