
**Purpose:**  
Lock in the actual environment & runtimes.

Overview:

- Define:
    
    - MC 1.7.10 + Forge 10.13.4.1614 + GTNH 2.8.1 run profile
        
    - Decision: external bot client vs in-process Forge mod with IPC
        
    - Hardware constraints for local LLMs
        
- **Dependencies:** None
    
- **Difficulty:** ⭐
    
- **Scalability notes:**
    
    - Document this in a single config file / README; future changes (new model, new server host) should not touch code.


Details:

M0, the humble root of all this chaos. Good place to start.

---

# M0 · `environment_foundation`

**Phase:** P0 – Foundations  
**Role:** Single source of truth for runtimes, integration mode, and hardware constraints.
## 1. Responsibilities & Boundaries

### 1.1 What M0 owns

M0 is **purely infrastructural**. It does **not** control the agent or Minecraft logic. It defines and validates:

- **Game stack:**
    
    - Minecraft **1.7.10**
        
    - Forge **10.13.4.1614**
        
    - GTNH **2.8.1** (client & optionally server)
        
- **Integration mode (very important):**
    
    - `bot_mode = "forge_mod"`
        
    - or `bot_mode = "external_client"`
        
- **Local LLM environment:**
    
    - Model files, quantization type, VRAM/CPU constraints
        
    - Backend: `llama.cpp`, `vLLM`, or `transformers`
        
- **Profile metadata:**
    
    - Paths, ports, hostnames
        
    - Profiles: `dev`, `local_sp`, `local_server`, `remote_server`
        

M0 outputs **configuration and validation**, nothing more. Higher modules must treat it as read-only environment description.

### 1.2 What M0 does _not_ do

- No decision-making, planning, or “agent logic.”
    
- No connecting to servers.
    
- No Forge mod code.
    
- No LLM prompts.
    

It’s a **configuration layer + sanity checker**, not a control system.

---

## 2. Target Artifacts & Layout

Suggested directory structure:

```
gtnh_agent/
  config/
    env.yaml            # core environment config
    minecraft.yaml      # GTNH profiles
    models.yaml         # local model definitions
    hardware.yaml       # HW profile and constraints
  tools/
    validate_env.py     # run sanity checks
    print_env.py        # human-friendly summary
  src/
    env/
      __init__.py
      loader.py         # single import point for env
      schema.py         # dataclasses / pydantic models

```

### 2.1 Core config: `config/env.yaml`

Example:

```
# config/env.yaml
profile: dev

bot_mode: forge_mod  # or: external_client

profiles:
  dev:
    minecraft_profile: "local_sp"
    model_profile: "gpu_small"
    hardware_profile: "desktop_4090"
  server_test:
    minecraft_profile: "local_server"
    model_profile: "gpu_medium"
    hardware_profile: "server_2x3090"

```

### 2.2 Minecraft config: `config/minecraft.yaml`

```
minecraft_profiles:
  local_sp:
    install_root: "/home/user/.minecraft/gtnh"
    version: "1.7.10"
    forge_version: "10.13.4.1614"
    gtnh_version: "2.8.1"
    launch_command: "java -Xmx6G -jar MultiMCLauncher.jar"
    world_name: "GrundleGangAgentDev"
    connection:
      mode: "singleplayer"

  local_server:
    install_root: "/opt/gtnh_server"
    version: "1.7.10"
    forge_version: "10.13.4.1614"
    gtnh_version: "2.8.1"
    connection:
      mode: "server"
      host: "127.0.0.1"
      port: 25565

```
### 2.3 Model config: `config/models.yaml`

```
model_profiles:
  gpu_small:
    backend: "llama_cpp"
    models:
      planner:
        path: "/models/planner-q4_K.gguf"
        context_length: 4096
      code:
        path: "/models/code-q4_K.gguf"
        context_length: 4096
      critic:
        path: "/models/critic-q4_K.gguf"
        context_length: 4096

  cpu_only:
    backend: "llama_cpp"
    models:
      planner:
        path: "/models/planner-q2_K.gguf"
        context_length: 2048

```
### 2.4 Hardware config: `config/hardware.yaml`

```
hardware_profiles:
  desktop_5070:
    gpu_count: 1
    gpu_memory_gb: 12
    cpu_cores: 16
    ram_gb: 32
    max_model_vram_gb: 12

```

---

## 3. Architecture Overview

### 3.1 High-level diagram

                +----------------------+
                |   config/env.yaml    |
                | config/minecraft...  |
                |  config/models...    |
                +-----------+----------+
                            |
                            v
                   +-----------------+
                   |  env.loader     |
                   | (M0 module)     |
                   +--------+--------+
                            |
          +-----------------+---------------------+
          |                                       |
          v                                       v
+----------------------+            +-----------------------------+
|  Minecraft-related   |            |    LLM / Hardware-related   |
|  modules (M6, M7)    |            |     modules (M2, etc.)      |
+----------------------+            +-----------------------------+


Everything imports environment via **one path**:

Python:
```
from env.loader import load_environment

```

No other hardcoded paths, ports, or model files _anywhere_ else.

---

## 4. Python Implementation Sketch

This is not full production code, but it’s the correct _shape_.

### 4.1 Schema definitions: `src/env/schema.py`

Python:

```
# src/env/schema.py

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class ConnectionConfig:
    """Network connection config for the Minecraft runtime."""
    mode: str            # "singleplayer" or "server"
    host: Optional[str]  # required if mode == "server"
    port: Optional[int]  # required if mode == "server"


@dataclass
class MinecraftProfile:
    """Describes a GTNH install and how to launch/connect to it."""
    name: str
    install_root: str
    version: str
    forge_version: str
    gtnh_version: str
    launch_command: str
    world_name: Optional[str]
    connection: ConnectionConfig


@dataclass
class ModelConfig:
    """Single model file configuration."""
    path: str
    context_length: int


@dataclass
class ModelProfile:
    """Collects all models used by the agent under a profile name."""
    name: str
    backend: str                   # e.g. "llama_cpp", "vllm"
    models: Dict[str, ModelConfig] # keys: planner, code, critic, etc.


@dataclass
class HardwareProfile:
    """Describes hardware constraints for planning/model decisions."""
    name: str
    gpu_count: int
    gpu_memory_gb: int
    cpu_cores: int
    ram_gb: int
    max_model_vram_gb: int


@dataclass
class EnvProfile:
    """Resolved environment for one active profile."""
    name: str
    bot_mode: str                 # "forge_mod" or "external_client"
    minecraft: MinecraftProfile
    model_profile: ModelProfile
    hardware_profile: HardwareProfile

```


### 4.2 Loader & validator: `src/env/loader.py`

Python:
```
# src/env/loader.py

import os                  # work with file paths and environment variables
from pathlib import Path   # path handling with nicer semantics
from typing import Tuple   # for type hints of multiple return values

import yaml                # parse YAML config files

from .schema import (      # import our dataclasses from schema.py
    ConnectionConfig,
    MinecraftProfile,
    ModelConfig,
    ModelProfile,
    HardwareProfile,
    EnvProfile,
)


CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"
# CONFIG_DIR points to the "config" folder at the project root.


def _load_yaml(name: str) -> dict:
    """Load a YAML file from the config directory."""
    path = CONFIG_DIR / name           # build the full path to the YAML file
    if not path.exists():              # check if the file actually exists
        raise FileNotFoundError(f"Missing config file: {path}")
    with path.open("r", encoding="utf-8") as f:  # open the file safely
        return yaml.safe_load(f)      # parse YAML into a Python dict


def _select_profile(env_cfg: dict) -> Tuple[str, dict]:
    """Resolve which profile is active."""
    # read the "profile" field from env.yaml
    profile_name = env_cfg.get("profile", "dev")
    profiles = env_cfg.get("profiles", {})    # get all profile definitions
    if profile_name not in profiles:          # ensure requested profile exists
        raise ValueError(f"Unknown env profile: {profile_name}")
    return profile_name, profiles[profile_name]


def load_environment() -> EnvProfile:
    """Main entry point: returns a fully resolved EnvProfile."""
    # load all config files as raw dictionaries
    env_cfg = _load_yaml("env.yaml")
    mc_cfg = _load_yaml("minecraft.yaml")
    models_cfg = _load_yaml("models.yaml")
    hw_cfg = _load_yaml("hardware.yaml")

    # determine which env profile is active and get its mapping
    active_profile_name, active_profile = _select_profile(env_cfg)

    # extract profile references from env.yaml
    bot_mode = env_cfg.get("bot_mode", "forge_mod")
    mc_profile_name = active_profile["minecraft_profile"]
    model_profile_name = active_profile["model_profile"]
    hw_profile_name = active_profile["hardware_profile"]

    # resolve Minecraft profile config
    mc_profile_raw = mc_cfg["minecraft_profiles"][mc_profile_name]
    # build ConnectionConfig
    conn = ConnectionConfig(
        mode=mc_profile_raw["connection"]["mode"],
        host=mc_profile_raw["connection"].get("host"),
        port=mc_profile_raw["connection"].get("port"),
    )
    minecraft_profile = MinecraftProfile(
        name=mc_profile_name,
        install_root=mc_profile_raw["install_root"],
        version=mc_profile_raw["version"],
        forge_version=mc_profile_raw["forge_version"],
        gtnh_version=mc_profile_raw["gtnh_version"],
        launch_command=mc_profile_raw["launch_command"],
        world_name=mc_profile_raw.get("world_name"),
        connection=conn,
    )

    # resolve model profile
    model_profile_raw = models_cfg["model_profiles"][model_profile_name]
    model_objects = {
        name: ModelConfig(path=cfg["path"], context_length=cfg["context_length"])
        for name, cfg in model_profile_raw["models"].items()
    }
    model_profile = ModelProfile(
        name=model_profile_name,
        backend=model_profile_raw["backend"],
        models=model_objects,
    )

    # resolve hardware profile
    hw_profile_raw = hw_cfg["hardware_profiles"][hw_profile_name]
    hardware_profile = HardwareProfile(
        name=hw_profile_name,
        gpu_count=hw_profile_raw["gpu_count"],
        gpu_memory_gb=hw_profile_raw["gpu_memory_gb"],
        cpu_cores=hw_profile_raw["cpu_cores"],
        ram_gb=hw_profile_raw["ram_gb"],
        max_model_vram_gb=hw_profile_raw["max_model_vram_gb"],
    )

    # perform basic validation before returning
    _validate_env(bot_mode, minecraft_profile, model_profile, hardware_profile)

    # wrap everything into a single EnvProfile dataclass
    return EnvProfile(
        name=active_profile_name,
        bot_mode=bot_mode,
        minecraft=minecraft_profile,
        model_profile=model_profile,
        hardware_profile=hardware_profile,
    )


def _validate_env(
    bot_mode: str,
    mc: MinecraftProfile,
    models: ModelProfile,
    hw: HardwareProfile,
) -> None:
    """Minimal sanity checks for the environment."""
    # ensure bot_mode is one of the recognized types
    if bot_mode not in ("forge_mod", "external_client"):
        raise ValueError(f"Invalid bot_mode: {bot_mode}")

    # verify MC version/forge/GTNH version fields are correct format strings
    if mc.version != "1.7.10":
        raise ValueError(f"Expected MC 1.7.10, got {mc.version}")
    if mc.forge_version != "10.13.4.1614":
        raise ValueError(f"Expected Forge 10.13.4.1614, got {mc.forge_version}")
    if mc.gtnh_version != "2.8.1":
        raise ValueError(f"Expected GTNH 2.8.1, got {mc.gtnh_version}")

    # basic hardware sanity checks
    if hw.gpu_count == 0 and models.backend != "llama_cpp":
        # gentle nudge: other backends may expect GPU
        raise ValueError("Non-GPU hardware profile with non-CPU backend is suspicious.")

    # confirm model files exist on disk
    for name, cfg in models.models.items():
        model_path = Path(cfg.path)
        if not model_path.exists():
            raise FileNotFoundError(f"Missing model file for {name}: {model_path}")

```

### 4.3 CLI tools in `tools/validate_env.py`
Python:
```
# tools/validate_env.py

import sys           # for exit codes
from pprint import pprint  # for structured printing

# make src discoverable if running as a script
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from env.loader import load_environment  # import our loader


def main() -> None:
    """Load and print the resolved environment, failing fast on errors."""
    try:
        env = load_environment()             # try to resolve the active env profile
    except Exception as e:                   # catch *any* error for debugging
        print("Environment validation FAILED:", file=sys.stderr)
        print(repr(e), file=sys.stderr)
        sys.exit(1)                          # non-zero exit: CI will mark as failed

    print("Environment validation OK.")
    print("\nActive profile:", env.name)
    print("\nBot mode:", env.bot_mode)
    print("\nMinecraft profile:")
    pprint(env.minecraft)
    print("\nModel profile:")
    pprint(env.model_profile)
    print("\nHardware profile:")
    pprint(env.hardware_profile)


if __name__ == "__main__":
    main()  # run main() only when script is executed directly

```

---

## 5. Testing / Simulation Strategy

M0 is extremely testable without running Minecraft or LLMs.

### 5.1 Unit tests

- Test suite: `tests/test_env_loader.py`
    
- Scenarios:
    
    - Valid configs → `load_environment()` returns `EnvProfile`
        
    - Missing configs → raises `FileNotFoundError`
        
    - Invalid versions → raises `ValueError`
        
    - Non-existent model files → raises `FileNotFoundError`
        
    - Invalid `bot_mode` → raises `ValueError`
        

You can use pytest parametrization to test multiple profiles quickly.

### 5.2 Profile simulation

Create fake profiles in `env.yaml`, `minecraft.yaml`, `models.yaml`, `hardware.yaml`:

- `dev`
    
- `cpu_only`
    
- `weak_gpu`
    
- `server_test`
    

Change `profile` in `env.yaml` and run:

bash:
`python tools/validate_env.py`

If this script dies, your env is not valid. No agent should ever run in that state.

### 5.3 CI integration

- Add a CI step:
    
    - `python tools/validate_env.py`
        
- If someone pushes changes to config or env code and breaks it, CI fails.
    

---

## 6. Libraries, APIs, and Repos to Examine

You don’t need to worship them, but they’re useful for patterns:

- **YAML & config management**
    
    - `pyyaml` – already using it
        
    - `pydantic` – if you want stricter validation than dataclasses
        
- **Env/config patterns**
    
    - Any “12-factor app” style config loader repos on GitHub
        
- **LLM backends**
    
    - `llama.cpp` Python bindings (`llama_cpp_python`)
        
    - `vllm` for high-throughput local inference
        
    - `transformers` if you ever decide to run HF models directly
        
- **Minecraft / GTNH references**
    
    - GTNH modpack repo / docs for exact versioning and typical paths
        
    - Forge 1.7.10 setup guides so you know what a valid install layout looks like
        

You don’t have to copy them, just steal the good ideas for config shape and validation.

---

## 7. Completion Criteria for M0

M0 is **done** when the following are true:

1. **Single import source:**
    
    - All other modules get environment/runtime info exclusively via:
        python:
        `from env.loader import load_environment`
        
    - No hardcoded paths or ports in other modules.
        
2. **Config-driven profiles:**
    
    - `config/env.yaml`, `minecraft.yaml`, `models.yaml`, `hardware.yaml` exist and are documented.
        
    - At least one stable profile (`dev`) works end-to-end.
        
3. **Validation works:**
    
    - `python tools/validate_env.py` passes on a correctly configured machine.
        
    - Fails loudly and clearly if:
        
        - Models are missing
            
        - GTNH/Forge/Minecraft versions don’t match the expected ones
            
        - `bot_mode` is invalid
            
4. **Safe evolution:**
    
    - You can:
        
        - Add a new profile (e.g., `server_test`)
            
        - Change hardware profile
            
        - Swap model profile  
            …without touching any code outside this module.
            
5. **Documented choices:**
    
    - The decision between `forge_mod` vs `external_client` is:
        
        - Explicit in `env.yaml`
            
        - Documented in a short README (pros/cons, current choice)
            

Once those criteria are met, M0 stops being interesting (which is exactly what you want) and everything else can safely assume the ground it’s standing on isn’t made of quicksand. You can now move onto M1 of Phase 0.