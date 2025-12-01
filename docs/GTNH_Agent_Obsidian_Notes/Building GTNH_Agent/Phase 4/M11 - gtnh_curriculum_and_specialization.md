
# M11 – gtnh_curriculum_and_specialization

## Phase
P4 — Learning & Specialization

## Purpose
Transform the GTNH_Agent from a generic learning system into a **GTNH-native progression engine**.  
This module provides **curricula**, **phase logic**, **goal selection**, **virtue weighting overrides**, **skill-focus guidance**, and **integration with the learning system (M10)**.

M11 decides *what the agent should work on next*, based on tech progression, world state, and long‑horizon projects.

---

## 1. Responsibilities & Boundaries

### 1.1 Responsibilities

M11 owns:

- **Curriculum definitions**
  - Multiple curricula under `config/curricula/*`
  - Each curriculum defines:
    - Phases (Steam → LV → MV → …)
    - Goals for each phase
    - Virtue weighting overrides per phase
    - Skill-focus hints (must‑have / preferred)
    - Phase completion conditions
    - Long-horizon project definitions (e.g., Stargate)

- **Phase resolution**
  - Select active curriculum phase from:
    - Current `TechState`
    - WorldState (machine presence, environment)

- **Goal selection**
  - Provide AgentLoop (M8) with phase‑appropriate goals.

- **Virtue modulation**
  - Provide virtue‑override multipliers for M4 scoring.

- **Skill guidance**
  - Provide skill-focus lists to:
    - Planner (M8) for prompt bias
    - Skill learning (M10) for prioritizing learning cycles

- **Long‑horizon project tracking**
  - Determine when major projects become “unlocked” based on completed phases.

### 1.2 Non‑Responsibilities

M11 does **not**:
- Perform planning (M8 does that)
- Perform skill synthesis/evaluation (M10)
- Perform low‑level world reasoning (M3)
- Perform bot control or observation work
- Hardcode Minecraft logic

---

## 2. Config Layout

Curricula live at:

```
config/curricula/
    aesthetic_megabase.yaml
    default_speedrun.yaml
    eco_factory.yaml
```

Each curriculum file contains:

- Curriculum metadata (id, name, description)
- Phases (steam_early, lv_bootstrap, mv_automation…)
- Goals per phase
- Virtue overrides
- Skill-focus (must_have / preferred)
- Completion conditions (tech unlocks, machine presence)
- Long-horizon projects

All curriculum logic is data-driven.

```python
# path: src/curriculum/schema.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


@dataclass
class PhaseTechTargets:
    """
    Tech requirements for a phase.

    Maps directly from:

    tech_targets:
      required_active: "steam_age"
      required_unlocked:
        - "steam_machines"
        - "coke_ovens"
    """
    required_active: str
    required_unlocked: List[str] = field(default_factory=list)


@dataclass
class PhaseGoal:
    """
    Single goal inside a phase.

    Example YAML:

    goals:
      - id: "steam_power_core"
        description: "Establish stable steam power..."
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
    """
    must_have: List[str] = field(default_factory=list)
    preferred: List[str] = field(default_factory=list)


@dataclass
class PhaseConfig:
    """
    Full configuration for a single curriculum phase.
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
    """
    id: str
    description: str
    depends_on_phases: List[str] = field(default_factory=list)


@dataclass
class LongHorizonProject:
    """
    High-level, multi-stage long-horizon project.

    Example: Stargate construction track.
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
    """
    id: str
    name: str
    description: str
    phases: List[PhaseConfig] = field(default_factory=list)
    long_horizon_projects: List[LongHorizonProject] = field(default_factory=list)

```


```python
# path: src/curriculum/loader.py

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


def _load_yaml(path: Path) -> Dict[str, Any]:
    """
    Load a YAML file into a raw dict.

    Raises FileNotFoundError if the path does not exist.
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
    """
    path = CURRICULA_DIR / f"{curriculum_id}.yaml"
    return load_curriculum(path)


def list_curricula() -> Dict[str, Path]:
    """
    List all available curricula in the default curricula directory.

    Returns a mapping: curriculum_id -> Path.
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

```



---

## 3. Schema (Phase, Curriculum, Projects)

M11 relies on structured dataclasses in:

```
src/curriculum/schema.py
```

These define:

### 3.1 PhaseTechTargets
- Required tech tier (e.g., `"steam_age"`)
- Required unlocks

### 3.2 PhaseGoal
- Goal id
- Description

### 3.3 PhaseCompletionConditions
- Required tech unlocks
- Required machines (type + min_count)

### 3.4 PhaseConfig
- id, name
- tech targets
- goals
- virtue overrides
- skill-focus metadata
- completion conditions

### 3.5 LongHorizonProject / ProjectStage
- Multi‑stage project definitions
- Dependencies on completed phases

### 3.6 CurriculumConfig
- Metadata
- All phases
- All long-horizon projects


```python
# path: src/curriculum/schema.py

from dataclasses import dataclass, field
from typing import Dict, Any, List

# Note: TechState is referenced conceptually in the docs,
# but not required directly in these dataclasses.
# from semantics.schema import TechState  # M3 tech progression (if needed elsewhere)


@dataclass
class PhaseTechTargets:
    """
    Tech requirements for a phase.

    Maps from YAML like:

    tech_targets:
      required_active: "steam_age"
      required_unlocked:
        - "steam_machines"
        - "coke_ovens"
    """
    required_active: str                     # expected TechState.active
    required_unlocked: List[str] = field(default_factory=list)


@dataclass
class PhaseGoal:
    """
    Single goal inside a phase.

    Example YAML:

    goals:
      - id: "steam_power_core"
        description: "Establish stable steam power..."
    """
    id: str
    description: str
    # later: constraints, prerequisites, tags, etc.


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
    """
    tech_unlocked: List[str] = field(default_factory=list)
    machines_present: List[Dict[str, Any]] = field(default_factory=list)
    # could extend with "items_in_storage", "quests_completed", etc.


@dataclass
class PhaseConfig:
    """
    Full configuration for a single curriculum phase.

    Fields mirror the M11 docs:

    - id, name
    - tech_targets
    - goals
    - virtue_overrides (per-virtue multipliers)
    - skill_focus (must_have / preferred skill names)
    - completion_conditions
    """
    id: str
    name: str
    tech_targets: PhaseTechTargets
    goals: List[PhaseGoal]
    virtue_overrides: Dict[str, float] = field(default_factory=dict)   # virtue weight multipliers
    skill_focus: Dict[str, List[str]] = field(                         # must_have / preferred skill names
        default_factory=lambda: {"must_have": [], "preferred": []}
    )
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
    """
    id: str
    description: str
    depends_on_phases: List[str]


@dataclass
class LongHorizonProject:
    """
    High-level, multi-stage project definition.

    Example: Stargate construction track.
    """
    id: str
    name: str
    description: str
    stages: List[ProjectStage]


@dataclass
class CurriculumConfig:
    """
    Root config for a curriculum YAML:

    id: "default_speedrun"
    name: "Default GTNH Speed-Progression"
    description: "Standard GTNH progression focused on efficient tech climb."
    phases: [...]
    long_horizon_projects: [...]
    """
    id: str
    name: str
    description: str
    phases: List[PhaseConfig]
    long_horizon_projects: List[LongHorizonProject]

```




---

## 4. Loader

Implemented in:

```
src/curriculum/loader.py
```

Capabilities:

- Load YAML files into structured `CurriculumConfig`
- Validate structural fields
- Expose:
  ```
  load_curriculum(path)
  load_curriculum_by_id(id)
  list_curricula()
  ```

This enables clean integration with runtime bootstrapping.

```python
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

```

---

## 5. Curriculum Engine

Implemented in:

```
src/curriculum/engine.py
```

### 5.1 Responsibilities

- Resolve **current phase** based on TechState & WorldState
- Determine **phase completion**
- Provide:
  - active goals
  - virtue overrides
  - skill-focus hints
- Determine unlocked long-horizon projects

### 5.2 Key Methods

#### `select_phase(tech_state, world)`
Finds the first phase whose requirements match the current state.

#### `_phase_is_complete(...)`
Evaluates completion conditions.

#### `view(tech_state, world) -> ActiveCurriculumView`
Primary API for M8 and M10.

Returns:

- `phase_view`
  - phase config
  - is_complete flag
  - active_goals
  - virtue_overrides
  - skill_focus
- `unlocked_projects`

### 5.3 ActiveCurriculumView
The main structured output for downstream modules.

```python
# path: src/curriculum/engine.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from semantics.schema import TechState  # M3 tech state model
from spec.types import WorldState       # Shared world-view contract

from .schema import CurriculumConfig, PhaseConfig, LongHorizonProject


@dataclass
class ActivePhaseView:
    """
    What the agent should know about the current curriculum phase.
    """
    phase: PhaseConfig
    is_complete: bool
    active_goals: List[str]
    virtue_overrides: Dict[str, float]
    skill_focus: Dict[str, List[str]]


@dataclass
class ActiveCurriculumView:
    """
    Full view of curriculum context for the agent.

    This is the main structured output consumed by:
      - M8 (AgentLoop) for goals & skill-focus
      - M4 (virtue lattice) for per-phase overrides
      - M10 (learning) for prioritizing skills / goals
    """
    curriculum_id: str
    phase_view: ActivePhaseView
    unlocked_projects: List[LongHorizonProject]


class CurriculumEngine:
    """
    Resolve curriculum phase and specialization hints given TechState + WorldState.

    Responsibilities:
      - Resolve current phase
      - Determine phase completion
      - Provide:
          - active goals
          - virtue overrides
          - skill-focus hints
      - Determine unlocked long-horizon projects
    """

    def __init__(self, config: CurriculumConfig) -> None:
        self._config = config

    # -------------------------------------------------------------------------
    # Phase resolution & completion
    # -------------------------------------------------------------------------

    def _phase_matches(self, phase: PhaseConfig, tech_state: TechState) -> bool:
        """
        Determine if a phase is appropriate for the current TechState.

        A phase "matches" if:
          - tech_state.active == phase.tech_targets.required_active
          - all phase.tech_targets.required_unlocked are present in tech_state.unlocked
        """
        t = phase.tech_targets
        if tech_state.active != t.required_active:
            return False

        required_unlocked = set(t.required_unlocked)
        current_unlocked = set(getattr(tech_state, "unlocked", []))
        if not required_unlocked.issubset(current_unlocked):
            return False

        return True

    def _phase_is_complete(
        self,
        phase: PhaseConfig,
        tech_state: TechState,
        world: WorldState,
    ) -> bool:
        """
        Evaluate whether a phase's completion conditions are satisfied.

        Conditions include:
          - tech_unlocked: all must be in tech_state.unlocked
          - machines_present: each requirement must have at least `min_count`
            matching machines in world.context["machines"]
        """
        cond = phase.completion_conditions

        # Tech unlock conditions
        required_unlocks = set(cond.tech_unlocked)
        current_unlocked = set(getattr(tech_state, "unlocked", []))
        if not required_unlocks.issubset(current_unlocked):
            return False

        # Machine presence conditions
        machines_req = cond.machines_present
        if machines_req:
            machines = list(world.context.get("machines", []))
            for mreq in machines_req:
                mtype = mreq.get("type")
                min_count = int(mreq.get("min_count", 1))
                if mtype is None:
                    continue
                count = sum(1 for m in machines if m.get("type") == mtype)
                if count < min_count:
                    return False

        return True

    def select_phase(self, tech_state: TechState, world: WorldState) -> PhaseConfig:
        """
        Select the most appropriate phase for the current state.

        Strategy (v1):
          - Return the first phase in config.phases for which _phase_matches() is True.
          - If none match, fall back to the last phase in the list.

        This keeps the behavior deterministic and simple while allowing
        phases to be ordered from earliest to latest.
        """
        for phase in self._config.phases:
            if self._phase_matches(phase, tech_state):
                return phase

        if not self._config.phases:
            raise ValueError(
                f"Curriculum '{self._config.id}' has no phases defined."
            )

        # Fallback: last phase (most advanced / catch-all)
        return self._config.phases[-1]

    # -------------------------------------------------------------------------
    # Active view
    # -------------------------------------------------------------------------

    def _completed_phase_ids(
        self,
        tech_state: TechState,
        world: WorldState,
    ) -> List[str]:
        """
        Compute the list of phase IDs whose completion conditions are satisfied.
        """
        completed: List[str] = []
        for phase in self._config.phases:
            if self._phase_is_complete(phase, tech_state, world):
                completed.append(phase.id)
        return completed

    def _unlocked_projects(
        self,
        completed_phase_ids: List[str],
    ) -> List[LongHorizonProject]:
        """
        Determine which long-horizon projects are currently unlocked.

        A project is considered "unlocked" if ANY of its stages has ALL of its
        depends_on_phases satisfied. This is intentionally permissive; later
        versions can introduce stricter semantics.
        """
        completed_set = set(completed_phase_ids)
        unlocked: List[LongHorizonProject] = []

        for proj in self._config.long_horizon_projects:
            stages = proj.stages or []
            for stage in stages:
                deps = set(stage.depends_on_phases or [])
                if deps and deps.issubset(completed_set):
                    unlocked.append(proj)
                    break  # no need to check more stages for this project

        return unlocked

    def view(self, tech_state: TechState, world: WorldState) -> ActiveCurriculumView:
        """
        Build the full ActiveCurriculumView from current TechState and WorldState.

        This is the primary API for downstream modules (M8, M4, M10).

        Returns:
          ActiveCurriculumView with:
            - curriculum_id
            - phase_view:
                - phase (PhaseConfig)
                - is_complete (bool)
                - active_goals (List[str])
                - virtue_overrides (Dict[str, float])
                - skill_focus (Dict[str, List[str]])
            - unlocked_projects (List[LongHorizonProject])
        """
        phase = self.select_phase(tech_state, world)
        is_complete = self._phase_is_complete(phase, tech_state, world)

        # Extract active goals (for planner)
        active_goals = [g.description for g in phase.goals]

        # Virtue overrides (for M4)
        virtue_overrides = dict(phase.virtue_overrides)

        # Skill-focus hints (for M8 + M10)
        # PhaseSkillFocus can be deconstructed into a plain dict here.
        phase_skill_focus = getattr(phase, "skill_focus", None)
        if isinstance(phase_skill_focus, dict):
            skill_focus: Dict[str, List[str]] = {
                "must_have": list(phase_skill_focus.get("must_have", [])),
                "preferred": list(phase_skill_focus.get("preferred", [])),
            }
        else:
            # Assume PhaseSkillFocus-like object with attributes
            must_have = list(getattr(phase_skill_focus, "must_have", []))
            preferred = list(getattr(phase_skill_focus, "preferred", []))
            skill_focus = {"must_have": must_have, "preferred": preferred}

        phase_view = ActivePhaseView(
            phase=phase,
            is_complete=is_complete,
            active_goals=active_goals,
            virtue_overrides=virtue_overrides,
            skill_focus=skill_focus,
        )

        completed_phase_ids = self._completed_phase_ids(tech_state, world)
        unlocked_projects = self._unlocked_projects(completed_phase_ids)

        return ActiveCurriculumView(
            curriculum_id=self._config.id,
            phase_view=phase_view,
            unlocked_projects=unlocked_projects,
        )

```



---

## 6. Integration Points

### 6.1 With M8 (AgentLoop)

AgentLoop requests curriculum guidance:

1. Gather world + tech_state (M3 + M7)
2. Call:
   ```
   curr_view = curriculum_engine.view(tech_state, world)
   ```
3. Choose a planning goal from:
   ```
   curr_view.phase_view.active_goals
   ```

Skill-focus hints may also be injected into the planner prompt.

### 6.2 With M4 (Virtue Lattice)

M11 provides per-phase virtue-overrides.  
These **multiply** the base virtue lattice weights.

This lets different phases emphasize:

- Safety (Steam boilers)
- Efficiency (LV automation)
- Sustainability (Eco factory)
- Exploration, throughput, etc.

### 6.3 With M5 (Skills)

Skill-focus informs:

- Which skills are **mandatory** to load/activate for the phase
- Which skills are **preferred** for planning

### 6.4 With M10 (Skill Learning)

Skill-focus provides learning guidance:

- M10 chooses which skills to refine or synthesize next
- Curriculum may trigger learning cycles after N episodes
- Curriculum determines context_id for learning

M11 thus serves as the **learning scheduler**.

```python
# path: src/curriculum/integration_agent_loop.py

from __future__ import annotations

from typing import Optional, Tuple

from semantics.schema import TechState
from spec.types import WorldState

from .engine import CurriculumEngine, ActiveCurriculumView


def select_goal_from_curriculum(
    engine: CurriculumEngine,
    tech_state: TechState,
    world: WorldState,
    current_goal: Optional[str] = None,
) -> Tuple[str, ActiveCurriculumView]:
    """
    Resolve a planning goal using the curriculum engine.

    This is the recommended entrypoint for M8 (AgentLoop):

      1. Call curriculum_engine.view(tech_state, world)
      2. If the agent already has a non-empty current_goal, keep it.
      3. Otherwise, choose the first curriculum goal as default.
      4. Return (goal, curr_view).

    Parameters
    ----------
    engine:
        CurriculumEngine initialized with the active CurriculumConfig.
    tech_state:
        Current TechState (M3) inferred from semantics.
    world:
        Current WorldState (M6/M7 normalized view).
    current_goal:
        Optional already-set goal string. If non-empty, it is preserved.

    Returns
    -------
    (goal, curr_view):
        goal: str
            Goal string to pass into the planner.
        curr_view: ActiveCurriculumView
            Structured curriculum view for downstream use (M4, M10, UI).
    """
    curr_view = engine.view(tech_state, world)
    phase_view = curr_view.phase_view

    # Preserve an explicit user-defined or previously-set goal
    if isinstance(current_goal, str) and current_goal.strip():
        goal = current_goal.strip()
    else:
        # Fallback: take the first active goal from the phase, or a generic one
        active_goals = phase_view.active_goals or []
        goal = active_goals[0] if active_goals else "advance GTNH tech progression"

    return goal, curr_view

```
How you’d use it inside AgentLoop (conceptually):
```python
# inside AgentLoop._step_planning (pseudo-code)
from curriculum.integration_agent_loop import select_goal_from_curriculum

goal, curr_view = select_goal_from_curriculum(
    self._curriculum_engine,
    tech_state,
    world_state,
    current_goal=self._goal,
)
self._goal = goal
self._curriculum_view = curr_view

```


### 2) Virtue overrides integration

Phase-specific virtue overrides multiply base weights. This helper centralizes that logic for M4.

```python
# path: src/virtues/integration_overrides.py

from __future__ import annotations

from typing import Dict


def merge_virtue_weights(
    base_weights: Dict[str, float],
    overrides: Dict[str, float],
) -> Dict[str, float]:
    """
    Merge base virtue weights with per-phase overrides.

    Overrides are interpreted as *multipliers*:

        merged[v] = base[v] * overrides.get(v, 1.0)

    Any virtue not present in overrides keeps its base weight.

    Parameters
    ----------
    base_weights:
        The default virtue weights loaded from virtues.yaml (M4).
    overrides:
        Per-phase override factors from M11 (e.g. 1.2 for Efficiency).

    Returns
    -------
    Dict[str, float]
        New dict of merged weights.
    """
    merged: Dict[str, float] = {}

    for name, base in base_weights.items():
        factor = overrides.get(name, 1.0)
        try:
            factor_f = float(factor)
        except (TypeError, ValueError):
            factor_f = 1.0
        merged[name] = float(base) * factor_f

    # Include any virtues that only appear in overrides (if you want that behavior)
    for name, factor in overrides.items():
        if name not in merged:
            try:
                merged[name] = float(factor)
            except (TypeError, ValueError):
                continue

    return merged

```
Wiring idea inside M4 scoring (conceptual):
```python
from virtues.integration_overrides import merge_virtue_weights

def score_plan_with_curriculum(plan, metrics, base_weights, phase_overrides):
    weights = merge_virtue_weights(base_weights, phase_overrides)
    # then do your usual scoring with `weights`

```

### 3) Curriculum → Skill learning hooks (M10)

Use skill-focus as a hint for which skills to prioritize in learning cycles.

```python
# path: src/learning/curriculum_hooks.py

from __future__ import annotations

from typing import Iterable, List, Dict, Any, Optional

from semantics.schema import TechState
from spec.types import WorldState

from curriculum.engine import ActiveCurriculumView
from .manager import SkillLearningManager


def select_learning_targets_from_skill_focus(
    skill_focus: Dict[str, List[str]],
    default_targets: Optional[Iterable[str]] = None,
) -> List[str]:
    """
    Compute an ordered list of skills to prioritize for learning,
    based on curriculum skill_focus.

    Priority order:
      1. must_have (in order listed)
      2. preferred (in order listed)
      3. any remaining defaults, if provided

    Parameters
    ----------
    skill_focus:
        Dict with keys:
          - "must_have": List[str]
          - "preferred": List[str]
    default_targets:
        Optional iterable of fallback skill names.

    Returns
    -------
    List[str]
        Ordered list of skill names to focus learning on.
    """
    must_have = list(skill_focus.get("must_have", []))
    preferred = list(skill_focus.get("preferred", []))
    remaining: List[str] = []

    seen = set(must_have) | set(preferred)

    if default_targets is not None:
        for name in default_targets:
            if name not in seen:
                remaining.append(name)

    return must_have + preferred + remaining


def schedule_learning_from_curriculum(
    manager: SkillLearningManager,
    curriculum_view: ActiveCurriculumView,
    tech_state: TechState,
    world: WorldState,
    *,
    context_prefix: str = "",
    min_episodes_per_skill: int = 5,
) -> List[Dict[str, Any]]:
    """
    High-level helper to schedule learning cycles based on the active curriculum.

    This does NOT auto-deploy new skills. It simply:
      - reads phase skill_focus
      - chooses learning targets
      - runs learning cycles via SkillLearningManager
      - returns a list of result dicts for monitoring / UI

    Parameters
    ----------
    manager:
        The SkillLearningManager instance (M10).
    curriculum_view:
        ActiveCurriculumView from CurriculumEngine.view(...).
    tech_state:
        Current TechState (M3).
    world:
        Current WorldState (M6/M7).
    context_prefix:
        Optional prefix for context_id (e.g. "curriculum:default_speedrun").
    min_episodes_per_skill:
        Minimum number of successful episodes to require before attempting
        a learning cycle for that skill.

    Returns
    -------
    List[Dict[str, Any]]
        One entry per attempted learning cycle, each containing:
          - "skill_name"
          - "result" (SkillLearningManager result)
    """
    phase_view = curriculum_view.phase_view
    skill_focus = phase_view.skill_focus

    # Use phase id as context; allow prefix override for nicer separation
    base_context = phase_view.phase.id
    if context_prefix:
        context_id = f"{context_prefix}:{base_context}"
    else:
        context_id = base_context

    # Choose skills to focus on for learning
    targets = select_learning_targets_from_skill_focus(skill_focus)

    results: List[Dict[str, Any]] = []

    for skill_name in targets:
        # Heuristic: derive a loose goal substring from the skill name
        # You can make this more structured later.
        goal_substring = skill_name.replace("_", " ")

        result = manager.run_learning_cycle_for_goal(
            goal_substring=goal_substring,
            target_skill_name=skill_name,
            context_id=context_id,
            tech_tier=tech_state.active,
            success_only=True,
            min_episodes=min_episodes_per_skill,
        )

        results.append(
            {
                "skill_name": skill_name,
                "result": result,
            }
        )

    return results

```
Usage idea in a runtime / supervisor loop:
```python
from learning.curriculum_hooks import schedule_learning_from_curriculum

# after some episodes have been recorded
learning_results = schedule_learning_from_curriculum(
    manager=skill_learning_manager,
    curriculum_view=curriculum_view,
    tech_state=tech_state,
    world=world_state,
    context_prefix=f"curriculum:{curriculum_view.curriculum_id}",
)

```


---

## 7. Long-Horizon Projects

Curriculum can define major project tracks (e.g., Stargate).

Each project includes:

- Multi-stage dependencies
- Requirements in terms of completed phases
- Project unlock state is exposed in `ActiveCurriculumView`

This allows the agent to eventually plan toward extremely long-horizon tasks.

```python
# path: src/curriculum/schema.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, List


@dataclass
class PhaseTechTargets:
    """
    Tech requirements for a phase.

    Maps from YAML like:

    tech_targets:
      required_active: "steam_age"
      required_unlocked:
        - "steam_machines"
        - "coke_ovens"
    """
    required_active: str                     # expected TechState.active
    required_unlocked: List[str] = field(default_factory=list)


@dataclass
class PhaseGoal:
    """
    Single goal inside a phase.

    Example YAML:

    goals:
      - id: "steam_power_core"
        description: "Establish stable steam power..."
    """
    id: str
    description: str
    tags: List[str] = field(default_factory=list)  # optional tagging


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
    """
    tech_unlocked: List[str] = field(default_factory=list)
    machines_present: List[Dict[str, Any]] = field(default_factory=list)
    # later: items_in_storage, quests_completed, etc.


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

    Each stage unlocks when all its depends_on_phases are completed.
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
    """
    id: str
    name: str
    description: str
    phases: List[PhaseConfig] = field(default_factory=list)
    long_horizon_projects: List[LongHorizonProject] = field(default_factory=list)

```


```python
# path: src/curriculum/engine.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from semantics.schema import TechState          # M3 tech state model
from spec.types import WorldState               # Shared world-view contract

from .schema import CurriculumConfig, PhaseConfig, LongHorizonProject


@dataclass
class ActivePhaseView:
    """
    What the agent should know about the current curriculum phase.

    Exposed to:
      - M8 (AgentLoop) for goal selection and skill biasing.
      - M4 (virtues) for per-phase weighting.
      - M10 (learning) for prioritizing refinement targets.
    """
    phase: PhaseConfig
    is_complete: bool
    active_goals: List[str]
    virtue_overrides: Dict[str, float]
    skill_focus: Dict[str, List[str]]


@dataclass
class ActiveCurriculumView:
    """
    Full view of curriculum context for the agent.

    This is the main structured output consumed by downstream modules:

      - M8 (AgentLoop): phase goals & skill-focus.
      - M4 (Virtue lattice): per-phase overrides.
      - M10 (Skill learning): curriculum-aware learning priorities.
      - Monitoring / UI: exposed project unlock state.
    """
    curriculum_id: str
    phase_view: ActivePhaseView
    unlocked_projects: List[LongHorizonProject]


class CurriculumEngine:
    """
    Resolve curriculum phase and specialization hints given TechState + WorldState.

    Responsibilities:
      - Resolve current phase.
      - Determine phase completion.
      - Provide:
          - active goals
          - virtue overrides
          - skill-focus hints
      - Determine unlocked long-horizon projects.
    """

    def __init__(self, config: CurriculumConfig) -> None:
        self._config = config

    # -------------------------------------------------------------------------
    # Phase resolution & completion
    # -------------------------------------------------------------------------

    def _phase_matches(self, phase: PhaseConfig, tech_state: TechState) -> bool:
        """
        Determine if a phase is appropriate for the current TechState.

        A phase "matches" if:
          - tech_state.active == phase.tech_targets.required_active
          - all phase.tech_targets.required_unlocked are present in tech_state.unlocked
        """
        t = phase.tech_targets
        if tech_state.active != t.required_active:
            return False

        required_unlocked = set(t.required_unlocked)
        current_unlocked = set(getattr(tech_state, "unlocked", []))
        if not required_unlocked.issubset(current_unlocked):
            return False

        return True

    def _phase_is_complete(
        self,
        phase: PhaseConfig,
        tech_state: TechState,
        world: WorldState,
    ) -> bool:
        """
        Evaluate whether a phase's completion conditions are satisfied.

        Conditions include:
          - tech_unlocked: all must be in tech_state.unlocked
          - machines_present: each requirement must have at least `min_count`
            matching machines in world.context["machines"].
        """
        cond = phase.completion_conditions

        # Tech unlock conditions
        required_unlocks = set(cond.tech_unlocked)
        current_unlocked = set(getattr(tech_state, "unlocked", []))
        if not required_unlocks.issubset(current_unlocked):
            return False

        # Machine presence conditions
        machines_req = cond.machines_present
        if machines_req:
            machines = list(world.context.get("machines", []))
            for mreq in machines_req:
                mtype = mreq.get("type")
                min_count = int(mreq.get("min_count", 1))
                if not mtype:
                    continue
                count = sum(1 for m in machines if m.get("type") == mtype)
                if count < min_count:
                    return False

        return True

    def select_phase(self, tech_state: TechState, world: WorldState) -> PhaseConfig:
        """
        Select the most appropriate phase for the current state.

        Strategy (v1):
          - Return the first phase in config.phases for which _phase_matches() is True.
          - If none match, fall back to the last phase in the list.

        Phases should be ordered from earliest to latest progression.
        """
        for phase in self._config.phases:
            if self._phase_matches(phase, tech_state):
                return phase

        if not self._config.phases:
            raise ValueError(
                f"Curriculum '{self._config.id}' has no phases defined."
            )

        # Fallback: last phase (most advanced / catch-all)
        return self._config.phases[-1]

    # -------------------------------------------------------------------------
    # Long-horizon project resolution
    # -------------------------------------------------------------------------

    def _completed_phase_ids(
        self,
        tech_state: TechState,
        world: WorldState,
    ) -> List[str]:
        """
        Compute the list of phase IDs whose completion conditions are satisfied.
        """
        completed: List[str] = []
        for phase in self._config.phases:
            if self._phase_is_complete(phase, tech_state, world):
                completed.append(phase.id)
        return completed

    def _unlocked_projects(
        self,
        completed_phase_ids: List[str],
    ) -> List[LongHorizonProject]:
        """
        Determine which long-horizon projects are currently unlocked.

        A project is considered "unlocked" if ANY of its stages has ALL of its
        phase dependencies satisfied:

          stage.depends_on_phases ⊆ completed_phase_ids

        This gives you a simple way to say:
          - "Stargate: preparation stage unlocked once mv_automation + hv_age done"
        """
        completed_set = set(completed_phase_ids)
        unlocked: List[LongHorizonProject] = []

        for proj in self._config.long_horizon_projects:
            stages = proj.stages or []
            for stage in stages:
                deps = set(stage.depends_on_phases or [])
                if deps and deps.issubset(completed_set):
                    unlocked.append(proj)
                    break  # no need to check more stages for this project

        return unlocked

    # -------------------------------------------------------------------------
    # Active view
    # -------------------------------------------------------------------------

    def view(self, tech_state: TechState, world: WorldState) -> ActiveCurriculumView:
        """
        Build the full ActiveCurriculumView from current TechState and WorldState.

        This is the primary API for downstream modules (M8, M4, M10).

        Returns:
          ActiveCurriculumView with:
            - curriculum_id
            - phase_view:
                - phase (PhaseConfig)
                - is_complete (bool)
                - active_goals (List[str])
                - virtue_overrides (Dict[str, float])
                - skill_focus (Dict[str, List[str]])
            - unlocked_projects (List[LongHorizonProject])
        """
        phase = self.select_phase(tech_state, world)
        is_complete = self._phase_is_complete(phase, tech_state, world)

        # Extract active goals (for planner)
        active_goals = [g.description for g in phase.goals]

        # Virtue overrides (for M4)
        virtue_overrides = dict(phase.virtue_overrides)

        # Skill-focus hints (for M8 + M10)
        phase_skill_focus = phase.skill_focus
        must_have = list(getattr(phase_skill_focus, "must_have", []))
        preferred = list(getattr(phase_skill_focus, "preferred", []))
        skill_focus: Dict[str, List[str]] = {
            "must_have": must_have,
            "preferred": preferred,
        }

        phase_view = ActivePhaseView(
            phase=phase,
            is_complete=is_complete,
            active_goals=active_goals,
            virtue_overrides=virtue_overrides,
            skill_focus=skill_focus,
        )

        completed_phase_ids = self._completed_phase_ids(tech_state, world)
        unlocked_projects = self._unlocked_projects(completed_phase_ids)

        return ActiveCurriculumView(
            curriculum_id=self._config.id,
            phase_view=phase_view,
            unlocked_projects=unlocked_projects,
        )

```


---

## 8. Example Workflow

1. M8 observes → infers TechState.
2. M11 resolves phase:
   - `"steam_early"`
3. Outputs:
   - Goals: maintain steam, ore‑processing baseline…
   - Virtue overrides: emphasize Safety
   - Skill-focus: maintain boilers, feed coke ovens
4. M8 uses these to:
   - Bias planner
   - Select goal
5. Agent acts → generates episode (M8)
6. Episode stored (M10 buffer)
7. Curriculum may schedule learning cycle if phase rules say so.


```python
# path: src/curriculum/example_workflow.py

from __future__ import annotations

from typing import Any, Callable, Dict, List, Tuple

from semantics.schema import TechState          # M3: tech inference
from spec.types import WorldState               # M6/M7: normalized world view

from .engine import CurriculumEngine, ActiveCurriculumView


def _select_goal_from_curriculum(
    curriculum_engine: CurriculumEngine,
    tech_state: TechState,
    world_state: WorldState,
    current_goal: str | None = None,
) -> Tuple[str, ActiveCurriculumView]:
    """
    Step 1–4 (curriculum side):

      1. M8 observes → infers TechState           (done outside this helper)
      2. M11 resolves phase                       (curriculum_engine.view)
      3. Outputs goals, virtue overrides, skills
      4. M8 uses them to select a planning goal

    If `current_goal` is non-empty, it is preserved.
    Otherwise the first active curriculum goal is chosen, or a generic fallback.
    """
    curr_view = curriculum_engine.view(tech_state, world_state)
    phase_view = curr_view.phase_view

    if isinstance(current_goal, str) and current_goal.strip():
        goal = current_goal.strip()
    else:
        active_goals = phase_view.active_goals or []
        goal = active_goals[0] if active_goals else "advance GTNH tech progression"

    return goal, curr_view


def _select_learning_targets_from_skill_focus(
    skill_focus: Dict[str, List[str]],
    default_targets: List[str] | None = None,
) -> List[str]:
    """
    Derive an ordered list of skills to train from curriculum skill_focus.

    Priority:
      1. must_have (listed order)
      2. preferred (listed order)
      3. remaining defaults (if provided)
    """
    must_have = list(skill_focus.get("must_have", []))
    preferred = list(skill_focus.get("preferred", []))

    seen = set(must_have) | set(preferred)
    remaining: List[str] = []

    if default_targets is not None:
        for name in default_targets:
            if name not in seen:
                remaining.append(name)

    return must_have + preferred + remaining


def run_curriculum_example_step(
    *,
    curriculum_engine: CurriculumEngine,
    tech_state: TechState,
    world_state: WorldState,
    planner: Callable[[str, WorldState, Dict[str, Any]], Dict[str, Any]],
    episode_buffer: List[Dict[str, Any]],
    skill_learning_manager: Any | None = None,
    current_goal: str | None = None,
    default_learning_targets: List[str] | None = None,
    min_episodes_per_skill: int = 5,
) -> Dict[str, Any]:
    """
    Example end-to-end step showing how M8, M10, and M11 interact.

    This function is intentionally high-level and side-effectful:
      - Chooses a goal using curriculum (M11)
      - Calls a planner (M8-style) with that goal
      - Appends the resulting episode to an episode buffer (M8/M10)
      - Optionally uses skill_focus to schedule learning (M10)

    It corresponds roughly to the doc's “Example Workflow”:

      1. M8 observes → infers TechState.          (tech_state, world_state are inputs)
      2. M11 resolves phase via CurriculumEngine.view(...)
      3. Outputs:
           - goals
           - virtue overrides
           - skill-focus
      4. M8 uses these to:
           - bias planner
           - select goal
      5. Agent acts → generates episode (planner)
      6. Episode stored (episode_buffer)
      7. Curriculum may schedule learning cycle if phase rules say so.

    Parameters
    ----------
    curriculum_engine:
        Initialized CurriculumEngine for the active curriculum.
    tech_state:
        Current TechState (from M3).
    world_state:
        Current WorldState (from observation stack M6/M7).
    planner:
        Callable implementing the planning+acting step:

            episode = planner(goal: str, world_state: WorldState, hints: Dict[str, Any])

        It should return a dict-like episode object (logs, actions, outcome).
    episode_buffer:
        Mutable list that will be appended with the generated episode.
        In a real system this would be your replay buffer / trace store.
    skill_learning_manager:
        Optional learning manager (M10). If provided, this function will
        trigger curriculum-driven learning cycles.
        Expected to expose something like:

            run_learning_cycle_for_goal(
                goal_substring: str,
                target_skill_name: str,
                context_id: str,
                tech_tier: str,
                success_only: bool,
                min_episodes: int,
            ) -> Dict[str, Any]
    current_goal:
        Existing goal string for the agent. If non-empty, it is preserved.
    default_learning_targets:
        Optional list of fallback skill names for learning.
    min_episodes_per_skill:
        Minimum number of episodes to require before triggering
        a learning cycle per skill.

    Returns
    -------
    Dict[str, Any]
        A summary dict with:
          - "goal"
          - "curriculum_view"
          - "episode"
          - "learning_results" (if any)
    """
    # ------------------------------------------------------------------
    # 1–4: Curriculum → goal & hints
    # ------------------------------------------------------------------
    goal, curr_view = _select_goal_from_curriculum(
        curriculum_engine=curriculum_engine,
        tech_state=tech_state,
        world_state=world_state,
        current_goal=current_goal,
    )
    phase_view = curr_view.phase_view

    # Build hints for planner: virtue overrides + skill-focus etc.
    planner_hints: Dict[str, Any] = {
        "curriculum_id": curr_view.curriculum_id,
        "phase_id": phase_view.phase.id,
        "phase_name": phase_view.phase.name,
        "virtue_overrides": phase_view.virtue_overrides,
        "skill_focus": phase_view.skill_focus,
        "unlocked_projects": [
            {"id": p.id, "name": p.name} for p in curr_view.unlocked_projects
        ],
    }

    # ------------------------------------------------------------------
    # 5: Agent acts → generates episode
    # ------------------------------------------------------------------
    episode = planner(goal, world_state, planner_hints)

    # ------------------------------------------------------------------
    # 6: Episode stored
    # ------------------------------------------------------------------
    episode_buffer.append(episode)

    # ------------------------------------------------------------------
    # 7: Curriculum-driven learning (optional)
    # ------------------------------------------------------------------
    learning_results: List[Dict[str, Any]] = []

    if skill_learning_manager is not None:
        # Turn skill_focus into a learning target list
        targets = _select_learning_targets_from_skill_focus(
            skill_focus=phase_view.skill_focus,
            default_targets=default_learning_targets,
        )

        context_id = f"curriculum:{curr_view.curriculum_id}:{phase_view.phase.id}"
        tech_tier = getattr(tech_state, "active", "unknown")

        for skill_name in targets:
            # simple heuristic: use skill_name as goal substring
            goal_substring = skill_name.replace("_", " ")
            result = skill_learning_manager.run_learning_cycle_for_goal(
                goal_substring=goal_substring,
                target_skill_name=skill_name,
                context_id=context_id,
                tech_tier=tech_tier,
                success_only=True,
                min_episodes=min_episodes_per_skill,
            )
            learning_results.append(
                {
                    "skill_name": skill_name,
                    "result": result,
                }
            )

    return {
        "goal": goal,
        "curriculum_view": curr_view,
        "episode": episode,
        "learning_results": learning_results,
    }

```



---

## 9. System Diagram

```
M3 Semantics ───▶ TechState
                     │
                     ▼
           M11 Curriculum Engine
                     │
        ┌────────────┼────────────┐
        ▼             ▼            ▼
   Goals to M8   Virtue overrides   Skill-focus to M10
```

---

## 10. Completion Criteria (for M11)

M11 is complete when:

- [ ] At least one full curriculum loads without errors
- [ ] Engine selects correct phase based on tech state
- [ ] Phase completion conditions trigger correctly
- [ ] Goals and virtue overrides integrate with M8/M4
- [ ] Skill-focus integrates with M8/M10
- [ ] Long-horizon project unlocking works
- [ ] All tests for curriculum loading, selection, and project unlocking pass

---

## 11. Testing

### 11.1 Curriculum Loader Tests
- Verify YAML loads into valid schema.
- Validate missing fields.
- Validate unknown skill names or unlocks.

### 11.2 Phase Resolution Tests
- Correct phase chosen for a given tech_state.
- Phase transitions when completion conditions are met.

### 11.3 Virtue Overrides Tests
- Verify merging logic correctly multiplies base weights.

### 11.4 Project Unlock Tests
- Ensure long-horizon projects unlock properly.

```python
# path: tests/test_curriculum_loader.py

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pytest

from curriculum.loader import load_curriculum, list_curricula, CURRICULA_DIR
from curriculum.schema import CurriculumConfig


def _write_yaml(path: Path, text: str) -> None:
    path.write_text(text.strip() + "\n", encoding="utf-8")


def test_load_curriculum_minimal_valid(tmp_path: Path) -> None:
    """
    Verify YAML loads into a valid CurriculumConfig with minimal fields.
    """
    yaml_text = """
    id: "test_curriculum"
    name: "Test Curriculum"
    description: "Minimal valid curriculum."
    phases:
      - id: "steam_early"
        name: "Steam Early"
        tech_targets:
          required_active: "steam_age"
          required_unlocked: ["steam_machines"]
        goals:
          - id: "g1"
            description: "Make steam."
        virtue_overrides: {}
        skill_focus:
          must_have: []
          preferred: []
        completion_conditions:
          tech_unlocked: []
          machines_present: []
    long_horizon_projects: []
    """
    path = tmp_path / "test_curriculum.yaml"
    _write_yaml(path, yaml_text)

    cfg = load_curriculum(path)
    assert isinstance(cfg, CurriculumConfig)
    assert cfg.id == "test_curriculum"
    assert len(cfg.phases) == 1
    phase = cfg.phases[0]
    assert phase.id == "steam_early"
    assert phase.tech_targets.required_active == "steam_age"
    assert phase.goals[0].description == "Make steam."


def test_load_curriculum_missing_required_fields(tmp_path: Path) -> None:
    """
    Validate that missing required top-level fields cause a ValueError.
    """
    yaml_text = """
    name: "No ID Here"
    description: "This curriculum is missing an id."
    phases: []
    long_horizon_projects: []
    """
    path = tmp_path / "broken_curriculum.yaml"
    _write_yaml(path, yaml_text)

    with pytest.raises(ValueError):
        load_curriculum(path)


def test_load_phase_missing_tech_targets(tmp_path: Path) -> None:
    """
    Validate that missing required phase keys cause a ValueError.
    """
    yaml_text = """
    id: "broken_curriculum"
    name: "Broken"
    description: ""
    phases:
      - id: "steam_early"
        name: "Steam Early"
        # tech_targets missing
        goals: []
        virtue_overrides: {}
        skill_focus:
          must_have: []
          preferred: []
        completion_conditions:
          tech_unlocked: []
          machines_present: []
    long_horizon_projects: []
    """
    path = tmp_path / "broken_curriculum.yaml"
    _write_yaml(path, yaml_text)

    with pytest.raises(ValueError):
        load_curriculum(path)


def test_list_curricula_reads_ids(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """
    Ensure list_curricula discovers YAMLs in CURRICULA_DIR and uses top-level id.
    """
    # Point CURRICULA_DIR at tmp_path
    monkeypatch.setattr("curriculum.loader.CURRICULA_DIR", tmp_path, raising=True)

    yaml_a = """
    id: "cur_a"
    name: "A"
    description: ""
    phases: []
    long_horizon_projects: []
    """
    yaml_b = """
    id: "cur_b"
    name: "B"
    description: ""
    phases: []
    long_horizon_projects: []
    """
    _write_yaml(tmp_path / "a.yaml", yaml_a)
    _write_yaml(tmp_path / "b.yaml", yaml_b)

    found = list_curricula()
    assert "cur_a" in found
    assert "cur_b" in found
    assert all(isinstance(p, Path) for p in found.values())


def test_unknown_skill_names_dont_break_loader(tmp_path: Path) -> None:
    """
    Validate unknown skill names don't crash the loader.

    Actual validation of skill names (against a skill registry) is handled
    elsewhere. Here we just ensure arbitrary strings are accepted structurally.
    """
    yaml_text = """
    id: "skill_weirdness"
    name: "Weird Skills"
    description: "Uses arbitrary skill names."
    phases:
      - id: "phase1"
        name: "Phase 1"
        tech_targets:
          required_active: "steam_age"
          required_unlocked: []
        goals:
          - id: "g1"
            description: "Do something."
        virtue_overrides: {}
        skill_focus:
          must_have:
            - "this_skill_does_not_exist"
          preferred:
            - "another_fake_skill"
        completion_conditions:
          tech_unlocked: []
          machines_present: []
    long_horizon_projects: []
    """
    path = tmp_path / "skill_weirdness.yaml"
    _write_yaml(path, yaml_text)

    cfg = load_curriculum(path)
    phase = cfg.phases[0]
    assert "this_skill_does_not_exist" in phase.skill_focus.must_have
    assert "another_fake_skill" in phase.skill_focus.preferred

```


```python
# path: tests/test_curriculum_engine_phase.py

from __future__ import annotations

from pathlib import Path

from curriculum.loader import load_curriculum
from curriculum.engine import CurriculumEngine
from semantics.schema import TechState
from spec.types import WorldState


def _write_yaml(path: Path, text: str) -> None:
    path.write_text(text.strip() + "\n", encoding="utf-8")


def _world_with_machines(machines: list[dict] | None = None) -> WorldState:
    """
    Construct a minimal WorldState with machine context.
    """
    return WorldState(
        tick=1,
        position={"x": 0, "y": 64, "z": 0},
        dimension="overworld",
        inventory=[],
        nearby_entities=[],
        blocks_of_interest=[],
        tech_state={},
        context={"machines": machines or []},
    )


def test_phase_selection_basic_progression(tmp_path: Path) -> None:
    """
    Correct phase chosen for a given tech_state.
    """
    yaml_text = """
    id: "phase_selection_test"
    name: "Phase Selection Test"
    description: ""
    phases:
      - id: "steam_early"
        name: "Steam Early"
        tech_targets:
          required_active: "steam_age"
          required_unlocked: []
        goals:
          - id: "g1"
            description: "Make steam."
        virtue_overrides: {}
        skill_focus:
          must_have: []
          preferred: []
        completion_conditions:
          tech_unlocked: ["lv_age"]
          machines_present: []
      - id: "lv_bootstrap"
        name: "LV Bootstrap"
        tech_targets:
          required_active: "lv_age"
          required_unlocked: []
        goals:
          - id: "g2"
            description: "Set up LV grid."
        virtue_overrides: {}
        skill_focus:
          must_have: []
          preferred: []
        completion_conditions:
          tech_unlocked: []
          machines_present: []
    long_horizon_projects: []
    """
    path = tmp_path / "phases.yaml"
    _write_yaml(path, yaml_text)

    cfg = load_curriculum(path)
    engine = CurriculumEngine(cfg)

    # Early steam tech_state
    tech_steam = TechState(unlocked=[], active="steam_age", evidence={})
    world = _world_with_machines()

    view_steam = engine.view(tech_steam, world)
    assert view_steam.phase_view.phase.id == "steam_early"
    assert not view_steam.phase_view.is_complete
    assert "Make steam." in view_steam.phase_view.active_goals

    # LV tech_state
    tech_lv = TechState(unlocked=["lv_age"], active="lv_age", evidence={})
    view_lv = engine.view(tech_lv, world)
    assert view_lv.phase_view.phase.id == "lv_bootstrap"
    assert "Set up LV grid." in view_lv.phase_view.active_goals


def test_phase_completion_with_machines(tmp_path: Path) -> None:
    """
    Phase transitions when completion conditions are met:
    tech unlock + machine presence.
    """
    yaml_text = """
    id: "phase_complete_test"
    name: "Phase Completion Test"
    description: ""
    phases:
      - id: "steam_early"
        name: "Steam Early"
        tech_targets:
          required_active: "steam_age"
          required_unlocked: ["steam_machines"]
        goals:
          - id: "g1"
            description: "Make steam."
        virtue_overrides: {}
        skill_focus:
          must_have: []
          preferred: []
        completion_conditions:
          tech_unlocked: ["lv_age"]
          machines_present:
            - { type: "large_boiler", min_count: 1 }
    long_horizon_projects: []
    """
    path = tmp_path / "phases_complete.yaml"
    _write_yaml(path, yaml_text)

    cfg = load_curriculum(path)
    engine = CurriculumEngine(cfg)

    # Not enough tech, no machines
    tech = TechState(unlocked=["steam_machines"], active="steam_age", evidence={})
    world = _world_with_machines()
    view = engine.view(tech, world)
    assert view.phase_view.phase.id == "steam_early"
    assert view.phase_view.is_complete is False

    # Add machine, still missing lv_age
    world2 = _world_with_machines(machines=[{"type": "large_boiler"}])
    view2 = engine.view(tech, world2)
    assert view2.phase_view.is_complete is False

    # Add lv_age unlock → now complete
    tech2 = TechState(
        unlocked=["steam_machines", "lv_age"],
        active="steam_age",
        evidence={},
    )
    view3 = engine.view(tech2, world2)
    assert view3.phase_view.is_complete is True

```


```python
# path: tests/test_virtue_overrides.py

from __future__ import annotations

from virtues.integration_overrides import merge_virtue_weights


def test_virtue_overrides_identity() -> None:
    """
    override = 1.0 → no change.
    """
    base = {"Safety": 1.0, "Efficiency": 0.8}
    overrides = {"Safety": 1.0, "Efficiency": 1.0}
    merged = merge_virtue_weights(base, overrides)

    assert merged["Safety"] == base["Safety"]
    assert merged["Efficiency"] == base["Efficiency"]


def test_virtue_overrides_increase_and_decrease() -> None:
    """
    override > 1.0 increases weight, < 1.0 decreases weight.
    """
    base = {"Safety": 1.0, "Throughput": 1.0}
    overrides = {"Safety": 1.2, "Throughput": 0.5}
    merged = merge_virtue_weights(base, overrides)

    assert merged["Safety"] == pytest.approx(1.2)
    assert merged["Throughput"] == pytest.approx(0.5)


def test_virtue_overrides_new_virtue() -> None:
    """
    Virtue present only in overrides appears in merged map.
    """
    base = {"Safety": 1.0}
    overrides = {"Exploration": 0.3}
    merged = merge_virtue_weights(base, overrides)

    assert "Safety" in merged
    assert "Exploration" in merged
    assert merged["Exploration"] == pytest.approx(0.3)

```

```python
# path: tests/test_curriculum_projects.py

from __future__ import annotations

from pathlib import Path

from curriculum.loader import load_curriculum
from curriculum.engine import CurriculumEngine
from semantics.schema import TechState
from spec.types import WorldState


def _write_yaml(path: Path, text: str) -> None:
    path.write_text(text.strip() + "\n", encoding="utf-8")


def _world() -> WorldState:
    return WorldState(
        tick=1,
        position={"x": 0, "y": 64, "z": 0},
        dimension="overworld",
        inventory=[],
        nearby_entities=[],
        blocks_of_interest=[],
        tech_state={},
        context={"machines": []},
    )


def test_project_unlocks_after_required_phases(tmp_path: Path) -> None:
    """
    Ensure long-horizon projects unlock properly once all required phases
    in a stage's depends_on_phases are completed.
    """
    yaml_text = """
    id: "project_unlock_test"
    name: "Project Unlock Test"
    description: ""
    phases:
      - id: "mv_automation"
        name: "MV Automation"
        tech_targets:
          required_active: "mv_age"
          required_unlocked: []
        goals:
          - id: "g_mv"
            description: "Do MV things."
        virtue_overrides: {}
        skill_focus:
          must_have: []
          preferred: []
        completion_conditions:
          tech_unlocked: ["hv_age"]
          machines_present: []
      - id: "hv_age"
        name: "HV Age"
        tech_targets:
          required_active: "hv_age"
          required_unlocked: []
        goals:
          - id: "g_hv"
            description: "Do HV things."
        virtue_overrides: {}
        skill_focus:
          must_have: []
          preferred: []
        completion_conditions:
          tech_unlocked: []
          machines_present: []
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
    path = tmp_path / "project_unlock.yaml"
    _write_yaml(path, yaml_text)

    cfg = load_curriculum(path)
    engine = CurriculumEngine(cfg)

    # State 1: no phases completed yet → project not unlocked
    tech = TechState(unlocked=[], active="mv_age", evidence={})
    world = _world()
    view1 = engine.view(tech, world)
    assert view1.unlocked_projects == []

    # State 2: complete mv_automation but not hv_age
    tech_mv_done = TechState(unlocked=["hv_age"], active="mv_age", evidence={})
    view2 = engine.view(tech_mv_done, world)
    # mv_automation completion requires hv_age in tech_unlocked; we have that
    # but hv_age phase itself is not complete / maybe not active; still,
    # depends_on_phases requires both mv_automation and hv_age, so not enough yet.
    assert view2.unlocked_projects == []

    # State 3: mark hv_age phase complete too
    # completion conditions for hv_age are empty, so as soon as we set active="hv_age"
    # it's treated as matching and complete.
    tech_hv = TechState(unlocked=["hv_age"], active="hv_age", evidence={})
    view3 = engine.view(tech_hv, world)

    # At this point:
    # - hv_age phase matches and is considered complete (conditions empty)
    # - mv_automation completion conditions are satisfied in previous state
    #   via hv_age unlock; the engine's _completed_phase_ids should now
    #   include both "mv_automation" and "hv_age", which unlocks the project.
    assert any(p.id == "stargate_project" for p in view3.unlocked_projects)

```

---

## 12. Summary

M11 takes all previous modules and turns them into a directed progression engine.  
Instead of reacting blindly, the agent now follows a **curriculum**, gains **specialization**, and supports **long-horizon agenda-setting** such as massive GTNH megaprojects.

M11 is the bridge between:
- Sensory world reasoning (M3, M6–M7)
- Planning (M8)
- Skill evolution (M10)
- Value shaping (M4)

It is the **mindset module** that turns a generic agent into a GTNH-native researcher, technician, builder, and long-term planner.

