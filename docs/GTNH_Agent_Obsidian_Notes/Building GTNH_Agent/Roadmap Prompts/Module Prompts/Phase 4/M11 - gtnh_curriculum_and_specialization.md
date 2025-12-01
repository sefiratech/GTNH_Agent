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