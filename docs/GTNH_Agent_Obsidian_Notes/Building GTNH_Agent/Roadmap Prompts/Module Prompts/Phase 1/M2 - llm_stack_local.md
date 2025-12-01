Module Prompt:

Purpose: Zoom in on one module for detailed design.

Template: Context: This is part of the GTNH Agent project. Maintain consistency with prior outputs.

Focus on the M2 module from our map.

Explain its architecture and function in detail.

Include Python code snippets with comments on each line and system diagrams if relevant.

Suggest how to test or simulate it locally.

List specific libraries, APIs, or repos worth examining.

At the end, list criteria that would indicate this module is complete or working as intended.

Output: a detailed technical blueprint for one module.


Module:

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