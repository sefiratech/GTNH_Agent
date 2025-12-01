Module Prompt:

Purpose: Zoom in on one module for detailed design.

Template: Context: This is part of the GTNH Agent project. Maintain consistency with prior outputs.

Focus on the M6 module from our map.

Explain its architecture and function in detail.

Include Python code snippets with comments on each line and system diagrams if relevant.

Suggest how to test or simulate it locally.

List specific libraries, APIs, or repos worth examining.

At the end, list criteria that would indicate this module is complete or working as intended.

Output: a detailed technical blueprint for one module.


Module:

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