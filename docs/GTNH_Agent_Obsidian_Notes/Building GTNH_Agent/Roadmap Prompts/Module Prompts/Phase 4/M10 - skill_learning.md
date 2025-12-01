Module Prompt:

Purpose: Zoom in on one module for detailed design.

Template: Context: This is part of the GTNH Agent project. Maintain consistency with prior outputs.

Focus on the M10 module from our map.

Explain its architecture and function in detail.

Include Python code snippets with comments on each line and system diagrams if relevant.

Suggest how to test or simulate it locally.

List specific libraries, APIs, or repos worth examining.

At the end, list criteria that would indicate this module is complete or working as intended.

Output: a detailed technical blueprint for one module.


Module:

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