## Bot Mode Selection

The agent can run in two modes, selected via `env.yaml`:

- **forge_mod** — the Minecraft instance exposes APIs through a Forge mod.  
  ✔ Current default  
  ✔ Best for deep GTNH integration  
  ✔ Stable packet access  
  ✔ Full mod-level introspection

- **external_client** — the agent connects as an external bot client via network protocol.  
  ✔ Easier to debug packet flow  
  ✔ Works without modifying the game instance  
  ✘ Limited mod data visibility

The project currently uses **forge_mod** for maximum compatibility with GTNH’s custom systems.

