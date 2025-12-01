# path: src/curriculum/loader.py
# load curriculum YAMLs from config/curricula

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import yaml

from .schema import (
    CurriculumConfig,
    PhaseConfig,
    PhaseTechTargets,
    PhaseGoal,
    PhaseCompletionConditions,
    PhaseSkillFocus,
    LongHorizonProject,
    ProjectStage,
)

# Default curricula directory:
# repo_root / config / curricula
CURRICULA_DIR = Path(__file__).resolve().parents[2] / "config" / "curricula"

__all__ = [
    "CURRICULA_DIR",
    "load_curriculum",
    "load_curriculum_by_id",
    "list_curricula",
]


def _load_yaml(path: Path) -> Dict[str, Any]:
    """
    Load a YAML file into a raw dict.

    Raises:
        FileNotFoundError: if the path does not exist.
        ValueError: if the root YAML node is not a mapping.
    """
    if not path.exists():
        raise FileNotFoundError(f"Curriculum YAML not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Curriculum YAML must be a mapping at root: {path}")
    return raw


def _parse_phase(raw_phase: Dict[str, Any]) -> PhaseConfig:
    """
    Parse a single phase section from raw YAML into a PhaseConfig.

    Expected shape (simplified):

    phases:
      - id: "steam_early"
        name: "Early Steam Infrastructure"
        tech_targets:
          required_active: "steam_age"
          required_unlocked:
            - "steam_machines"
            - "coke_ovens"
        goals:
          - id: "steam_power_core"
            description: "Establish stable steam power..."
        virtue_overrides:
          Efficiency: 1.0
        skill_focus:
          must_have:
            - "feed_coke_ovens"
          preferred:
            - "chunk_mining"
        completion_conditions:
          tech_unlocked:
            - "lv_age"
          machines_present:
            - { type: "large_boiler", min_count: 1 }
    """
    try:
        phase_id = raw_phase["id"]
        name = raw_phase["name"]
        tech_targets_raw = raw_phase["tech_targets"]
    except KeyError as exc:
        raise ValueError(f"Phase is missing required key: {exc!s}") from exc

    tech_targets = PhaseTechTargets(
        required_active=str(tech_targets_raw["required_active"]),
        required_unlocked=list(tech_targets_raw.get("required_unlocked", [])),
    )

    goals_raw = raw_phase.get("goals", [])
    goals: List[PhaseGoal] = []
    for g in goals_raw:
        try:
            gid = g["id"]
            desc = g["description"]
        except KeyError as exc:
            raise ValueError(
                f"Phase '{phase_id}' goal missing required key: {exc!s}"
            ) from exc
        tags = list(g.get("tags", []))
        goals.append(PhaseGoal(id=gid, description=desc, tags=tags))

    virtue_overrides_raw = raw_phase.get("virtue_overrides", {}) or {}
    if not isinstance(virtue_overrides_raw, dict):
        raise ValueError(
            f"Phase '{phase_id}' virtue_overrides must be a mapping, "
            f"got: {type(virtue_overrides_raw).__name__}"
        )
    virtue_overrides: Dict[str, float] = {}
    for vname, weight in virtue_overrides_raw.items():
        virtue_overrides[str(vname)] = float(weight)

    skill_focus_raw = raw_phase.get("skill_focus", {}) or {}
    if not isinstance(skill_focus_raw, dict):
        raise ValueError(
            f"Phase '{phase_id}' skill_focus must be a mapping, "
            f"got: {type(skill_focus_raw).__name__}"
        )
    must_have = list(skill_focus_raw.get("must_have", []))
    preferred = list(skill_focus_raw.get("preferred", []))
    skill_focus = PhaseSkillFocus(must_have=must_have, preferred=preferred)

    completion_raw = raw_phase.get("completion_conditions", {}) or {}
    if not isinstance(completion_raw, dict):
        raise ValueError(
            f"Phase '{phase_id}' completion_conditions must be a mapping, "
            f"got: {type(completion_raw).__name__}"
        )
    completion = PhaseCompletionConditions(
        tech_unlocked=list(completion_raw.get("tech_unlocked", [])),
        machines_present=list(completion_raw.get("machines_present", [])),
    )

    return PhaseConfig(
        id=str(phase_id),
        name=str(name),
        tech_targets=tech_targets,
        goals=goals,
        virtue_overrides=virtue_overrides,
        skill_focus=skill_focus,
        completion_conditions=completion,
    )


def _parse_project(raw_proj: Dict[str, Any]) -> LongHorizonProject:
    """
    Parse a long-horizon project definition.

    Expected shape (simplified):

    long_horizon_projects:
      - id: "stargate_project"
        name: "Stargate Construction"
        description: "Complete the Stargate and related infrastructure."
        stages:
          - id: "preparation"
            description: "Mass automation of high-tier materials."
            depends_on_phases:
              - "mv_automation"
              - "hv_age"
    """
    try:
        proj_id = raw_proj["id"]
        name = raw_proj["name"]
    except KeyError as exc:
        raise ValueError(f"Project missing required key: {exc!s}") from exc

    description = str(raw_proj.get("description", ""))

    stages_raw = raw_proj.get("stages", []) or []
    stages: List[ProjectStage] = []
    for s in stages_raw:
        try:
            sid = s["id"]
            desc = s["description"]
        except KeyError as exc:
            raise ValueError(
                f"Project '{proj_id}' stage missing required key: {exc!s}"
            ) from exc
        depends_on_phases = list(s.get("depends_on_phases", []))
        stages.append(
            ProjectStage(
                id=str(sid),
                description=str(desc),
                depends_on_phases=depends_on_phases,
            )
        )

    return LongHorizonProject(
        id=str(proj_id),
        name=str(name),
        description=description,
        stages=stages,
    )


def load_curriculum(path: Path) -> CurriculumConfig:
    """
    Parse a single curriculum YAML file into a CurriculumConfig.

    This is the main entry for turning a `config/curricula/*.yaml`
    file into a strongly-typed curriculum object.
    """
    raw = _load_yaml(path)

    try:
        cid = raw["id"]
        name = raw["name"]
    except KeyError as exc:
        raise ValueError(f"Curriculum missing required key: {exc!s}") from exc

    description = str(raw.get("description", ""))

    phases_raw = raw.get("phases", []) or []
    if not isinstance(phases_raw, list):
        raise ValueError(
            f"Curriculum '{cid}' phases must be a list, "
            f"got: {type(phases_raw).__name__}"
        )
    phases = [_parse_phase(p) for p in phases_raw]

    projects_raw = raw.get("long_horizon_projects", []) or []
    if not isinstance(projects_raw, list):
        raise ValueError(
            f"Curriculum '{cid}' long_horizon_projects must be a list, "
            f"got: {type(projects_raw).__name__}"
        )
    projects = [_parse_project(p) for p in projects_raw]

    return CurriculumConfig(
        id=str(cid),
        name=str(name),
        description=description,
        phases=phases,
        long_horizon_projects=projects,
    )


def load_curriculum_by_id(curriculum_id: str) -> CurriculumConfig:
    """
    Load a curriculum YAML from the default curricula directory
    by its `id`, assuming the filename is `{id}.yaml`.

    Example:
        cfg = load_curriculum_by_id("default_speedrun")
    """
    path = CURRICULA_DIR / f"{curriculum_id}.yaml"
    return load_curriculum(path)


def list_curricula() -> Dict[str, Path]:
    """
    List all available curricula in the default curricula directory.

    Returns:
        Dict[str, Path]: mapping curriculum_id -> Path

    Notes:
        - Only YAML files with a top-level `id` string are included.
        - If the directory does not exist, returns an empty dict.
    """
    results: Dict[str, Path] = {}
    if not CURRICULA_DIR.exists():
        return results

    for p in CURRICULA_DIR.glob("*.yaml"):
        raw = _load_yaml(p)
        cid = raw.get("id")
        if isinstance(cid, str):
            results[cid] = p

    return results

