# load environment config (env.yaml, minecraft.yaml, models.yaml, hardware.yaml)
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
