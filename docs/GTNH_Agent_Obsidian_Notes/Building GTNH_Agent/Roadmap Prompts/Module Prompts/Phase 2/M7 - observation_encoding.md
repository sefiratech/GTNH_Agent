Module Prompt:

Purpose: Zoom in on one module for detailed design.

Template: Context: This is part of the GTNH Agent project. Maintain consistency with prior outputs.

Focus on the M7 module from our map.

Explain its architecture and function in detail.

Include Python code snippets with comments on each line and system diagrams if relevant.

Suggest how to test or simulate it locally.

List specific libraries, APIs, or repos worth examining.

At the end, list criteria that would indicate this module is complete or working as intended.

Output: a detailed technical blueprint for one module.


Module: 

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