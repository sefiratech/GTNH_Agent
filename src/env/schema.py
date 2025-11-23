# EnvProfile, MinecraftProfile, ModelProfile dataclasses
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
