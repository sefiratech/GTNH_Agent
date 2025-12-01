Module Prompt:

Purpose: Zoom in on one module for detailed design.

Template: Context: This is part of the GTNH Agent project. Maintain consistency with prior outputs.

Focus on the M3 module from our map.

Explain its architecture and function in detail.

Include Python code snippets with comments on each line and system diagrams if relevant.

Suggest how to test or simulate it locally.

List specific libraries, APIs, or repos worth examining.

At the end, list criteria that would indicate this module is complete or working as intended.

Output: a detailed technical blueprint for one module.


Module:

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