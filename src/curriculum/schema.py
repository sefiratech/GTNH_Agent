# src/curriculum/schema.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class PhaseTechTargets:
    """
    Tech requirements for a phase.

    Expected YAML shape:

      tech_targets:
        required_active: "steam_age"
        required_unlocked:
          - "steam_machines"
          - "coke_ovens"

    This is used by the curriculum engine to decide which phase matches
    the current TechState.
    """
    required_active: str                     # expected TechState.active
    required_unlocked: List[str] = field(default_factory=list)


@dataclass
class PhaseGoal:
    """
    Single high-level goal inside a phase.

    Example YAML:

      goals:
        - id: "steam_power_core"
          description: "Establish stable steam power..."
          tags: ["power", "steam", "infrastructure"]
    """
    id: str
    description: str
    tags: List[str] = field(default_factory=list)


@dataclass
class PhaseCompletionConditions:
    """
    Conditions for marking a phase as 'complete'.

    Example YAML:

      completion_conditions:
        tech_unlocked:
          - "lv_age"
        machines_present:
          - { type: "large_boiler", min_count: 1 }

    This is intentionally small and heuristic. You can extend it later
    with items_in_storage, quests_completed, etc.
    """
    tech_unlocked: List[str] = field(default_factory=list)
    machines_present: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class PhaseSkillFocus:
    """
    Skill-focus hints for a phase.

    Example YAML:

      skill_focus:
        must_have:
          - "feed_coke_ovens"
        preferred:
          - "chunk_mining"

    M8/M10 can use this to bias which skills are considered "core" for
    this phase during planning / learning.
    """
    must_have: List[str] = field(default_factory=list)
    preferred: List[str] = field(default_factory=list)


@dataclass
class PhaseConfig:
    """
    Full configuration for a single curriculum phase.

    Fields mirror the M11 docs:

      - id, name
      - tech_targets
      - goals
      - virtue_overrides (per-virtue multipliers)
      - skill_focus (must_have / preferred)
      - completion_conditions

    Phases are ordered from earliest to latest progression in the YAML.
    """
    id: str
    name: str
    tech_targets: PhaseTechTargets
    goals: List[PhaseGoal]
    virtue_overrides: Dict[str, float] = field(default_factory=dict)
    skill_focus: PhaseSkillFocus = field(default_factory=PhaseSkillFocus)
    completion_conditions: PhaseCompletionConditions = field(
        default_factory=PhaseCompletionConditions
    )


@dataclass
class ProjectStage:
    """
    One stage of a long-horizon project.

    Example YAML:

      stages:
        - id: "preparation"
          description: "Mass automation of high-tier materials."
          depends_on_phases:
            - "mv_automation"
            - "hv_age"

    Each stage unlocks when all its phase dependencies are completed.
    """
    id: str
    description: str
    depends_on_phases: List[str] = field(default_factory=list)


@dataclass
class LongHorizonProject:
    """
    High-level, multi-stage long-horizon project.

    Example: Stargate construction track.

    A project is typically considered "unlocked" when at least one
    of its stages has all its phase dependencies satisfied.
    """
    id: str
    name: str
    description: str
    stages: List[ProjectStage] = field(default_factory=list)


@dataclass
class CurriculumConfig:
    """
    Root config object for a curriculum YAML.

    Example top-level:

      id: "default_speedrun"
      name: "Default GTNH Speed-Progression"
      description: "Standard GTNH progression focused on efficient tech climb."
      phases: [...]
      long_horizon_projects: [...]

    The loader is responsible for instantiating this from config/curricula/*.yaml.
    """
    id: str
    name: str
    description: str
    phases: List[PhaseConfig] = field(default_factory=list)
    long_horizon_projects: List[LongHorizonProject] = field(default_factory=list)

