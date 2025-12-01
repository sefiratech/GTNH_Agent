Now it's time for the module prompts. I will cycle through each module with the same prompt template and include the module's current state, after the module prompt, for reference.

Module Prompt:

Purpose: Zoom in on one module for detailed design.

Template: Context: This is part of the GTNH Agent project. Maintain consistency with prior outputs.

Focus on the M0 module from our map.

Explain its architecture and function in detail.

Include Python code snippets with comments on each line and system diagrams if relevant.

Suggest how to test or simulate it locally.

List specific libraries, APIs, or repos worth examining.

At the end, list criteria that would indicate this module is complete or working as intended.

Output: a detailed technical blueprint for one module.



Module:

M0 - environment_foundation

**Purpose:**

Lock in the actual environment & runtimes.

- Define:

- MC 1.7.10 + Forge 10.13.4.1614 + GTNH 2.8.1 run profile

- Decision: external bot client vs in-process Forge mod with IPC

- Hardware constraints for local LLMs

- **Dependencies:** None

- **Difficulty:** ⭐

- **Scalability notes:**

- Document this in a single config file / README; future changes (new model, new server host) should not touch code.