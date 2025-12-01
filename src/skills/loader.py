# src/skills/loader.py

from pathlib import Path
from typing import Dict, Any, List, Optional

import yaml

from .schema import (
    ParamSpec,
    SkillPreconditions,
    SkillEffects,
    SkillMetrics,
    SkillSpec,
)

# Default directory for skill YAML files:
# gtnh_agent/config/skills/
CONFIG_SKILLS_DIR = Path(__file__).resolve().parents[2] / "config" / "skills"


def _load_yaml(path: Path) -> Dict[str, Any]:
    """
    Load a YAML file into a dict.

    Returns an empty dict if the file is empty, rather than None.
    """
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def load_skill_spec_from_file(path: Path) -> SkillSpec:
    """
    Parse a single skill spec YAML file into a SkillSpec dataclass.

    This is the main entry point used by tests and by load_all_skill_specs().
    """
    raw = _load_yaml(path)

    # --- Params ---
    params_cfg = raw.get("params", {})
    params: Dict[str, ParamSpec] = {}
    for pname, pdata in params_cfg.items():
        params[pname] = ParamSpec(
            name=pname,
            type=str(pdata.get("type", "any")),
            default=pdata.get("default"),
            description=pdata.get("description", ""),
        )

    # --- Preconditions ---
    pre_cfg = raw.get("preconditions", {})
    preconditions = SkillPreconditions(
        tech_states_any_of=pre_cfg.get("tech_states_any_of", []) or [],
        required_tools=pre_cfg.get("required_tools", []) or [],
        dimension_allowlist=pre_cfg.get("dimension_allowlist", []) or [],
        semantic_tags_any_of=pre_cfg.get("semantic_tags_any_of", []) or [],
        extra={
            k: v
            for k, v in pre_cfg.items()
            if k
            not in (
                "tech_states_any_of",
                "required_tools",
                "dimension_allowlist",
                "semantic_tags_any_of",
            )
        },
    )

    # --- Effects ---
    eff_cfg = raw.get("effects", {})
    effects = SkillEffects(
        inventory_delta=eff_cfg.get("inventory_delta", {}) or {},
        tech_delta=eff_cfg.get("tech_delta", {}) or {},
        tags=eff_cfg.get("tags", []) or [],
        extra={
            k: v
            for k, v in eff_cfg.items()
            if k not in ("inventory_delta", "tech_delta", "tags")
        },
    )

    # --- Metrics ---
    metrics_cfg = raw.get("metrics", {}) or {}
    metrics = SkillMetrics(
        success_rate=metrics_cfg.get("success_rate"),
        avg_cost=metrics_cfg.get("avg_cost"),
    )

    # --- Top-level fields ---
    name = raw["name"]  # must exist
    version = int(raw.get("version", 1))
    status = str(raw.get("status", "active"))
    origin = str(raw.get("origin", "manual"))
    description = raw.get("description", "")
    tags: List[str] = raw.get("tags", []) or []

    # Basic sanity checks (fail fast on obviously bad data)
    if status not in {"active", "candidate", "deprecated"}:
        raise ValueError(f"Invalid status '{status}' for skill '{name}' in {path}")

    return SkillSpec(
        name=name,
        version=version,
        status=status,
        origin=origin,
        description=description,
        params=params,
        preconditions=preconditions,
        effects=effects,
        tags=tags,
        metrics=metrics,
    )


def load_all_skill_specs(skills_dir: Optional[Path] = None) -> Dict[str, SkillSpec]:
    """
    Load all skill specs from the skills config directory.

    Returns:
        Dict[str, SkillSpec]: mapping from skill name -> SkillSpec

    Args:
        skills_dir: override the skills directory (used mainly for tests).
                    Defaults to CONFIG_SKILLS_DIR.
    """
    base_dir = skills_dir or CONFIG_SKILLS_DIR

    if not base_dir.exists():
        raise FileNotFoundError(f"Skills directory does not exist: {base_dir}")

    specs: Dict[str, SkillSpec] = {}
    for path in sorted(base_dir.glob("*.yaml")):
        spec = load_skill_spec_from_file(path)
        if spec.name in specs:
            raise ValueError(
                f"Duplicate skill spec name '{spec.name}' in '{path}'. "
                f"Already defined in another file."
            )
        specs[spec.name] = spec

    return specs

