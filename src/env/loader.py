from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Any

import yaml


# ---------------------------------------------------------------------------
# Dataclasses for environment representation
# ---------------------------------------------------------------------------


@dataclass
class ConnectionConfig:
    mode: str
    host: str | None = None
    port: int | None = None


@dataclass
class MinecraftProfile:
    name: str
    install_root: str
    version: str
    forge_version: str
    gtnh_version: str
    launch_command: str | None
    world_name: str | None
    connection: ConnectionConfig


@dataclass
class ModelConfig:
    path: str
    context_length: int


@dataclass
class ModelProfile:
    name: str
    backend: str
    models: Dict[str, ModelConfig]


@dataclass
class HardwareProfile:
    name: str
    gpu_count: int
    gpu_memory_gb: int
    cpu_cores: int
    ram_gb: int
    max_model_vram_gb: int


@dataclass
class EnvProfile:
    """Top-level resolved environment profile."""
    name: str
    bot_mode: str
    minecraft: MinecraftProfile
    model_profile: ModelProfile
    hardware_profile: HardwareProfile


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = PROJECT_ROOT / "config"

# GitHub Actions sets CI=true; use that to relax certain checks there.
IS_CI = os.getenv("CI") == "true"


def _load_yaml(name: str) -> Dict[str, Any]:
    """Load a YAML config file from the config/ directory."""
    path = CONFIG_ROOT / name
    if not path.exists():
        raise FileNotFoundError(f"Missing config file: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping at top of {path}, got {type(data)}")
    return data


def _select_profile(env_cfg: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """Return (active_profile_name, active_profile_mapping)."""
    profile_name = env_cfg.get("profile")
    if not profile_name:
        raise ValueError("env.yaml must define a 'profile' key.")
    profiles = env_cfg.get("profiles")
    if not isinstance(profiles, dict):
        raise ValueError("env.yaml must define a 'profiles' mapping.")
    if profile_name not in profiles:
        raise KeyError(f"Profile '{profile_name}' not found in env.yaml profiles.")
    return profile_name, profiles[profile_name]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_environment() -> EnvProfile:
    """Main entry point: returns a fully resolved EnvProfile."""
    # Load all config files as raw dictionaries
    env_cfg = _load_yaml("env.yaml")
    mc_cfg = _load_yaml("minecraft.yaml")
    models_cfg = _load_yaml("models.yaml")
    hw_cfg = _load_yaml("hardware.yaml")

    # Determine which env profile is active and get its mapping
    active_profile_name, active_profile = _select_profile(env_cfg)

    # Extract profile references from env.yaml
    bot_mode = env_cfg.get("bot_mode", "forge_mod")
    mc_profile_name = active_profile["minecraft_profile"]
    model_profile_name = active_profile["model_profile"]
    hw_profile_name = active_profile["hardware_profile"]

    # Resolve Minecraft profile config
    mc_profiles = mc_cfg.get("minecraft_profiles") or {}
    if mc_profile_name not in mc_profiles:
        raise KeyError(f"Minecraft profile '{mc_profile_name}' not found in minecraft.yaml")
    mc_profile_raw = mc_profiles[mc_profile_name]

    conn_raw = mc_profile_raw.get("connection") or {}
    conn = ConnectionConfig(
        mode=conn_raw.get("mode"),
        host=conn_raw.get("host"),
        port=conn_raw.get("port"),
    )
    minecraft_profile = MinecraftProfile(
        name=mc_profile_name,
        install_root=mc_profile_raw["install_root"],
        version=mc_profile_raw["version"],
        forge_version=mc_profile_raw["forge_version"],
        gtnh_version=mc_profile_raw["gtnh_version"],
        launch_command=mc_profile_raw.get("launch_command"),
        world_name=mc_profile_raw.get("world_name"),
        connection=conn,
    )

    # Resolve model profile
    model_profiles = models_cfg.get("model_profiles") or {}
    if model_profile_name not in model_profiles:
        raise KeyError(f"Model profile '{model_profile_name}' not found in models.yaml")
    model_profile_raw = model_profiles[model_profile_name]

    models_dict: Dict[str, ModelConfig] = {
        name: ModelConfig(path=cfg["path"], context_length=cfg["context_length"])
        for name, cfg in model_profile_raw["models"].items()
    }
    model_profile = ModelProfile(
        name=model_profile_name,
        backend=model_profile_raw["backend"],
        models=models_dict,
    )

    # Resolve hardware profile
    hw_profiles = hw_cfg.get("hardware_profiles") or {}
    if hw_profile_name not in hw_profiles:
        raise KeyError(f"Hardware profile '{hw_profile_name}' not found in hardware.yaml")
    hw_profile_raw = hw_profiles[hw_profile_name]
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
            if IS_CI:
                # In CI we don't have your local LM Studio model directory.
                # Treat this as a warning-level issue and let higher layers decide.
                continue
            raise FileNotFoundError(f"Missing model file for {name}: {model_path}")

