Module Prompt:

Purpose: Zoom in on one module for detailed design.

Template: Context: This is part of the GTNH Agent project. Maintain consistency with prior outputs.

Focus on the M5 module from our map.

Explain its architecture and function in detail.

Include Python code snippets with comments on each line and system diagrams if relevant.

Suggest how to test or simulate it locally.

List specific libraries, APIs, or repos worth examining.

At the end, list criteria that would indicate this module is complete or working as intended.

Output: a detailed technical blueprint for one module.


Module:

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