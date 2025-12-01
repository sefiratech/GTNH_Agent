**Map Prompt:**  
Purpose: Establish the world, goals, roles, and deliverables.

Context: This is part of the Gregtech Agent project. Maintain consistent terminology and module naming.  
Format response using markdown headings and code blocks. Use bullet lists for subpoints.  
Use concise technical language suitable for engineering documentation.

Focus only on high-level architecture and modular design—not full implementation—unless explicitly requested.

Assume constraints and quirks specific to Minecraft 1.7.10, Forge 10.13.4.1614, and GTNH 2.8.1, including lack of modern hooks, outdated protocol structures, and modpack-specific mechanics.

All LLM calls must assume local models (e.g., llama.cpp, vLLM, or transformers pipelines) with no reliance on cloud APIs.

You are a Senior Programming Engineer specializing in Python and Machine Learning. We’re developing a working LLM-based Minecraft agent capable of autonomously learning new skills and aligning to a set of virtues embedded in a virtue lattice.

**Objectives:**

1. Reverse Engineer Mineflayer
    
2. Reverse Engineer Voyager
    
3. Construct a new program with the capabilities of Mineflayer and Voyager, improvements, calls to a set of local LLMs instead of OpenAI API, compatibility with Minecraft version 1.7.10, Forge version 10.13.4.1614, and Gregtech: New Horizons version 2.8.1 instead of Fabric and >MC v1.8.
    

Break the project into sequential phases/modules. For each, define:  
• Purpose  
• Key components / dependencies  
• Skills / knowledge required  
• Resources / documentation  
• Prototype or test goal

End with a suggested order of study or build.

Output: a structured roadmap of modules.