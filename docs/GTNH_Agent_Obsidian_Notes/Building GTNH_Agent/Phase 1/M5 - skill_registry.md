# M5 · `skill_registry`

**Phase:** P1 – Offline Core Pillars  
**Role:** Central definition & registry of **skills**: how the agent does things, how they’re described to the planner, and what preconditions/effects they claim.

---

# Purpose

Central place for **skill definitions and metadata**.

Skills define *what the agent can do* in a structured, declarative format.  
The Planner, Critic, and Agent Loop consume this metadata, not the raw Python implementation.

---

# Overview

- **Skill spec** defines:
  - Name
  - Parameters
  - Preconditions (world + tech state required)
  - Effects (world/tech state delta)
  - Tags (mining, crafting, building, progression, etc.)
  - Versioning fields (version, status, origin)
  - Metrics (success_rate, avg_cost)

- **LLM interaction**:
  - Planner only sees skill metadata via `describe_all()`
  - Skill implementations live as Python modules (`src/skills/base/*.py`)
  - Metadata lives in YAML (`config/skills/*.yaml`)

- **Dependencies:** `M1`, `M3`, `M10`, `M11`
- **Difficulty:** ⭐⭐–⭐⭐⭐
- **Scalability/perf:**
  - Skills registered via decorators or config files
  - Easy to version, deprecate, and evolve skills over time
  - Skill Packs allow selective unlocking based on TechState

Skills form the **behavioral vocabulary** of the GTNH Agent.

---

# M5 · `skill_registry`

**Phase:** P1 – Offline Core Pillars  
**Role:** Maintain a structured, versioned skill library accessible to planning & execution modules.

Dependencies:

- **M1:** `Skill`, `SkillRegistry`, `WorldState`, `Action`
- **M3:** `TechState`, world semantics (preconditions/effects)
- **M10:** skill evolution / learning system
- **M11:** curriculum & tech-tier Skill Pack activation

---

## 1. Responsibilities & Boundaries

### 1.1 What M5 Owns

- **Skill specification schema** (YAML):
  - name, description
  - parameters (name, type, default)
  - preconditions:
    - tech states (`TechState` from M3)
    - world conditions (dimension, required tools)
  - effects:
    - expected inventory/tech/world deltas
  - tags
  - version metadata:
    - `version`
    - `status:` active | candidate | deprecated
    - `origin:` manual | synthesized
    - `metrics:` success_rate, avg_cost

- **Skill Packs**
  - LV/Steam/MV/etc bundles
  - Activated by Curriculum (M11)
  - Loaded at runtime to restrict available skills

- **Skill implementation plumbing**
  - Python classes in `src/skills/base`
  - Decorator-based registration
  - Registry implementation with version gating

- **LLM-facing metadata**
  - `describe_all()` → Planner, Critic, Error Model

### 1.2 What M5 Does *Not* Do

- Does not plan
- Does not use LLMs directly
- Does not run IPC with Minecraft
- Does not learn new skills (M10 mutates specs)
- Does not choose skill packs (M11 decides)

M5 is the **API surface layer** for all behavioral capabilities.

---

## 2. File & Module Layout

```
gtnh_agent/
  config/
    skills/
      chop_tree.yaml
      plant_sapling.yaml
      feed_coke_ovens.yaml
    skill_packs/
      lv_core.yaml
      steam_age.yaml
      early_mv.yaml

  src/
    skills/
      __init__.py
      schema.py
      loader.py
      registry.py
      packs.py
      base/
        chop_tree.py
        feed_coke_ovens.py
```

- YAML = metadata  
- Python = implementations  
- Skill Packs = grouping of abilities by tech tier

---

## 3. Data Layer: SkillSpec

### 3.1 Example YAML spec: `config/skills/chop_tree.yaml`

```
name: "chop_tree"
version: 1
status: "active"
origin: "manual"
description: "Locate the nearest tree and chop it down, collecting logs."

params:
  radius:
    type: int
    default: 8
    description: "Search radius in blocks."
  min_logs:
    type: int
    default: 4
    description: "Minimum logs to gather before stopping."

preconditions:
  tech_states_any_of: ["stone_age", "steam_age"]
  required_tools:
    - item_id: "minecraft:iron_axe"
      min_durability: 10
  dimension_allowlist: ["overworld"]
  semantic_tags_any_of: ["tree"]

effects:
  inventory_delta:
    logs:
      min_increase: 4
  tech_delta: {}
  tags: ["gathers_logs"]

tags:
  - "mining"
  - "progression"
  - "low_risk"
  - "resource_gathering"

metrics:
  success_rate: null
  avg_cost: null
```

Planner sees only metadata, not code.

---

## 4. Core Types & Interfaces (`src/skills/schema.py`)

```
@dataclass
class ParamSpec:
    name: str
    type: str
    default: Any
    description: str

@dataclass
class SkillPreconditions:
    tech_states_any_of: List[str]
    required_tools: List[Dict[str, Any]]
    dimension_allowlist: List[str]
    semantic_tags_any_of: List[str]
    extra: Dict[str, Any]

@dataclass
class SkillEffects:
    inventory_delta: Dict[str, Dict[str, Any]]
    tech_delta: Dict[str, Any]
    tags: List[str]
    extra: Dict[str, Any]

@dataclass
class SkillMetrics:
    success_rate: Optional[float]
    avg_cost: Optional[float]

@dataclass
class SkillSpec:
    name: str
    version: int
    status: str
    origin: str
    description: str
    params: Dict[str, ParamSpec]
    preconditions: SkillPreconditions
    effects: SkillEffects
    tags: List[str]
    metrics: SkillMetrics
```

---

## 5. Loading Specs from YAML (`src/skills/loader.py`)

Supports:

- versioning
- status/origin
- semantic tags
- evolution fields for M10

---

## 6. Skill Packs (`config/skill_packs/*.yaml`)

Example:

```
name: "lv_core"
requires_tech_state: "lv"

skills:
  - chop_tree
  - plant_sapling
  - feed_coke_ovens
```

Skill Packs filter skill visibility based on TechState.

## 6.1: Recommended implementation strategy

### Step 1: Define a simple pack schema

Create files like:

`config/skill_packs/lv_core.yaml`  
`config/skill_packs/steam_age.yaml`  
`config/skill_packs/early_mv.yaml`

Schema:
yaml:
```
# config/skill_packs/lv_core.yaml
name: "lv_core"
requires_tech_state: "lv"   # or ">= lv" if you later want ranges
tags:
  - "core"
  - "early_progression"

skills:
  - chop_tree
  - plant_sapling
  - feed_coke_ovens
  - basic_crafting

```

Another example:
yaml:
```
# config/skill_packs/steam_age.yaml
name: "steam_age"
requires_tech_state: "steam_age"
tags:
  - "power"
  - "infrastructure"

skills:
  - feed_steam_boiler
  - maintain_coke_ovens
  - refill_water_tanks

```
You can add more fields later (e.g. `description`, `conflicts_with`, `depends_on_packs`), but this is enough for now.


### 6.2: Add a tiny `SkillPack` schema + loader

`src/skills/packs.py`:
```
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import yaml


PACKS_DIR = Path(__file__).resolve().parents[2] / "config" / "skill_packs"


@dataclass
class SkillPack:
    name: str
    requires_tech_state: str
    tags: List[str]
    skills: List[str]


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def load_skill_pack_from_file(path: Path) -> SkillPack:
    raw = _load_yaml(path)

    name = raw["name"]
    requires_tech_state = raw["requires_tech_state"]
    tags = raw.get("tags", []) or []
    skills = raw.get("skills", []) or []

    if not skills:
        raise ValueError(f"Skill pack '{name}' in {path} has no skills defined.")

    return SkillPack(
        name=name,
        requires_tech_state=requires_tech_state,
        tags=tags,
        skills=skills,
    )


def load_all_skill_packs(packs_dir: Optional[Path] = None) -> Dict[str, SkillPack]:
    base_dir = packs_dir or PACKS_DIR
    if not base_dir.exists():
        raise FileNotFoundError(f"Skill packs directory does not exist: {base_dir}")

    packs: Dict[str, SkillPack] = {}
    for path in sorted(base_dir.glob("*.yaml")):
        pack = load_skill_pack_from_file(path)
        if pack.name in packs:
            raise ValueError(
                f"Duplicate skill pack name '{pack.name}' in '{path}'."
            )
        packs[pack.name] = pack
    return packs

```
Now you have a clean way to say:
python:
```
from skills.packs import load_all_skill_packs

packs = load_all_skill_packs()

```

### Step 3: Tie packs to TechState

You need one helper that answers:

> “Given TechState T and a set of enabled pack names, what skills are available?”

Add to `packs.py`:
python:
```
def get_active_skill_names(
    tech_state: str,
    enabled_pack_names: List[str],
    packs: Optional[Dict[str, SkillPack]] = None,
) -> List[str]:
    """
    Determine which skills are available for a given tech_state and enabled packs.

    For now we treat requires_tech_state as an exact match.
    Later you can generalize (>=, tiers, etc.).
    """
    all_packs = packs or load_all_skill_packs()

    active_skills: List[str] = []

    for name in enabled_pack_names:
        pack = all_packs.get(name)
        if pack is None:
            continue

        if pack.requires_tech_state == tech_state:
            active_skills.extend(pack.skills)

    # deduplicate while preserving order
    seen = set()
    unique_skills: List[str] = []
    for s in active_skills:
        if s in seen:
            continue
        seen.add(s)
        unique_skills.append(s)

    return unique_skills

```

Who provides `enabled_pack_names`?

- That’s **M11’s job** (curriculum):  
    “For this profile, these packs are in play.”

### Step 4: Integrate with `SkillRegistry`

Registry’s `describe_all()` currently just lists all non-deprecated skills.

You make it aware of skill packs by adding a filtered view that M2/M8 will use:

In `src/skills/registry.py`:
python:
```
from typing import Iterable
from .packs import load_all_skill_packs, get_active_skill_names

class SkillRegistry(SkillRegistryProtocol):
    ...

    def describe_for_tech_state(
        self,
        tech_state: str,
        enabled_pack_names: Iterable[str],
    ) -> Dict[str, Dict[str, Any]]:
        """
        Return metadata for skills that are both:
        - registered and non-deprecated, and
        - included in at least one active Skill Pack for this tech_state.
        """
        packs = load_all_skill_packs()
        allowed_names = set(
            get_active_skill_names(tech_state, list(enabled_pack_names), packs=packs)
        )

        descs: Dict[str, Dict[str, Any]] = {}
        for name in self.list_skills():
            if name not in allowed_names:
                continue
            skill = self._skills[name]
            descs[name] = skill.describe()
        return descs

```

Then the Planner does:
python:
```
metadata = registry.describe_for_tech_state(
    tech_state=current_tech_state,
    enabled_pack_names=["lv_core", "steam_age"],  # decided by M11
)

```






---

## 7. Skill Registry (`src/skills/registry.py`)

Registry:

- Loads YAML specs
- Registers Python skill implementations
- Provides active skills (filters out deprecated ones)
- Exposes `describe_all()` for the Planner
- Stores version metadata for M10 to update
- Enforces consistency between YAML + Python
  
src/skills/registry.py:
python:
```
# src/skills/registry.py

from typing import Dict, Any, List, Type, Optional, Iterable

from spec.skills import Skill as SkillProtocol, SkillRegistry as SkillRegistryProtocol
from spec.types import WorldState, Action

from .schema import SkillSpec
from .loader import load_all_skill_specs
from .packs import load_all_skill_packs, get_active_skill_names


class SkillImplBase(SkillProtocol):
    """
    Base class for skill implementations.

    Subclasses must:
    - set `skill_name` to match a SkillSpec.name
    - implement `suggest_actions(world, params)`
    """

    skill_name: str = ""  # override in subclasses

    def __init__(self, spec: SkillSpec) -> None:
        # store the spec for use in describe()
        self._spec = spec

    @property
    def name(self) -> str:
        """Return the skill name defined by the spec."""
        return self._spec.name

    def describe(self) -> Dict[str, Any]:
        """
        Return metadata for planners and critics.

        Includes:
        - name, description
        - version, status, origin
        - parameter specs
        - preconditions
        - effects
        - tags
        - metrics (for M10 / learning)
        """
        return {
            "name": self._spec.name,
            "description": self._spec.description,
            "version": self._spec.version,
            "status": self._spec.status,
            "origin": self._spec.origin,
            "params": {
                pname: {
                    "type": ps.type,
                    "default": ps.default,
                    "description": ps.description,
                }
                for pname, ps in self._spec.params.items()
            },
            "preconditions": {
                "tech_states_any_of": self._spec.preconditions.tech_states_any_of,
                "required_tools": self._spec.preconditions.required_tools,
                "dimension_allowlist": self._spec.preconditions.dimension_allowlist,
                "semantic_tags_any_of": self._spec.preconditions.semantic_tags_any_of,
                "extra": self._spec.preconditions.extra,
            },
            "effects": {
                "inventory_delta": self._spec.effects.inventory_delta,
                "tech_delta": self._spec.effects.tech_delta,
                "tags": self._spec.effects.tags,
                "extra": self._spec.effects.extra,
            },
            "tags": self._spec.tags,
            "metrics": {
                "success_rate": self._spec.metrics.success_rate,
                "avg_cost": self._spec.metrics.avg_cost,
            },
        }

    def suggest_actions(
        self,
        world: WorldState,
        params: Dict[str, Any],
    ) -> List[Action]:
        """
        This method must be implemented by subclasses.

        Implementations convert a high-level skill invocation into a sequence
        of low-level Actions suitable for the agent loop / Minecraft IPC layer.
        """
        raise NotImplementedError("SkillImplBase subclasses must override suggest_actions()")


class SkillRegistry(SkillRegistryProtocol):
    """
    Concrete SkillRegistry implementation backed by YAML specs and Python classes.

    Responsibilities:
    - load SkillSpec metadata from config/skills/*.yaml
    - register concrete Skill implementations
    - provide filtered views (all active skills, or by tech_state + SkillPack)
    """

    def __init__(self) -> None:
        # name -> SkillSpec (from YAML)
        self._specs: Dict[str, SkillSpec] = load_all_skill_specs()
        # name -> SkillImplBase (Python implementation)
        self._skills: Dict[str, SkillImplBase] = {}

    # --- Registration ---

    def register(self, skill: SkillImplBase) -> None:
        """
        Register a single skill implementation instance.

        Enforces:
        - a spec must exist for the skill name
        - no duplicate registrations
        - cannot register explicitly deprecated skills
        """
        name = skill.name

        if name not in self._specs:
            raise KeyError(f"No SkillSpec found for skill: {name}")

        spec = self._specs[name]
        if spec.status == "deprecated":
            raise ValueError(f"Cannot register deprecated skill: {name}")

        if name in self._skills:
            raise KeyError(f"Skill already registered: {name}")

        self._skills[name] = skill

    # --- Query methods ---

    def list_skills(self) -> List[str]:
        """
        Return names of registered skills that are not deprecated.

        This list ignores skills without implementations and those whose
        SkillSpec.status is 'deprecated'.
        """
        return [
            name
            for name, spec in self._specs.items()
            if spec.status != "deprecated" and name in self._skills
        ]

    def get_skill(self, name: str) -> SkillProtocol:
        """Return the skill implementation by name."""
        if name not in self._skills:
            raise KeyError(name)
        return self._skills[name]

    def describe_all(self) -> Dict[str, Dict[str, Any]]:
        """
        Return metadata for all non-deprecated, registered skills.

        This is the generic view used when you don't care about tech_state
        or Skill Packs (e.g. debugging, tooling).
        """
        descs: Dict[str, Dict[str, Any]] = {}
        for name in self.list_skills():
            skill = self._skills[name]
            descs[name] = skill.describe()
        return descs

    def describe_for_tech_state(
        self,
        tech_state: str,
        enabled_pack_names: Iterable[str],
    ) -> Dict[str, Dict[str, Any]]:
        """
        Return metadata for skills that are both:

        - registered and non-deprecated, and
        - included in at least one active Skill Pack for this tech_state.

        This is the primary view used by Planner / Critic / Curriculum systems.
        """
        packs = load_all_skill_packs()
        allowed_names = set(
            get_active_skill_names(
                tech_state=tech_state,
                enabled_pack_names=list(enabled_pack_names),
                packs=packs,
            )
        )

        descs: Dict[str, Dict[str, Any]] = {}
        for name in self.list_skills():
            if name not in allowed_names:
                continue
            skill = self._skills[name]
            descs[name] = skill.describe()
        return descs

    # --- Evolution hook for M10 ---

    def update_spec(self, spec: SkillSpec) -> None:
        """
        Replace or insert a SkillSpec.

        Intended for use by M10 (learning/evolution):
        - updating status (candidate -> active / deprecated)
        - updating metrics (success_rate, avg_cost)
        - updating descriptions / tags over time
        """
        self._specs[spec.name] = spec


# Global registry instance (simple singleton pattern).
_GLOBAL_REGISTRY: Optional[SkillRegistry] = None


def get_global_skill_registry() -> SkillRegistry:
    """
    Lazy-load and return the global SkillRegistry instance.

    This is a convenience for modules that don't use explicit dependency injection.
    """
    global _GLOBAL_REGISTRY
    if _GLOBAL_REGISTRY is None:
        _GLOBAL_REGISTRY = SkillRegistry()
    return _GLOBAL_REGISTRY


def register_skill(cls: Type[SkillImplBase]) -> Type[SkillImplBase]:
    """
    Class decorator to register a skill implementation with the global registry.

    Usage:

        @register_skill
        class ChopTreeSkill(SkillImplBase):
            skill_name = "chop_tree"
            ...

    When the module is imported, this will:
    - look up the corresponding SkillSpec by skill_name
    - instantiate the skill class with that spec
    - register it in the global registry
    """
    registry = get_global_skill_registry()

    skill_name = getattr(cls, "skill_name", "")
    if not skill_name:
        raise ValueError(f"Skill class {cls.__name__} must define skill_name")

    spec = registry._specs.get(skill_name)
    if spec is None:
        raise KeyError(
            f"No SkillSpec found for '{skill_name}' "
            f"(expected YAML in config/skills/)"
        )

    instance = cls(spec)
    registry.register(instance)

    return cls

```

---

## 8. Skill Implementation Example (`src/skills/base/chop_tree.py`)

Skill implementations are thin, concrete adapters between:

- High-level skill intent (`"chop_tree"` with params), and  
- Low-level `Action` objects the agent loop / Minecraft IPC layer will execute (Phase 2).

They **do not** embed game logic like “is this tech unlocked?” – that’s handled by:

- Skill Packs + `TechState` (M3 + M11)
- Preconditions in the `SkillSpec` YAML

A minimal example implementation:

Python:

```python
# src/skills/base/chop_tree.py

from typing import Dict, Any, List

from spec.types import WorldState, Action               # shared agent types (M1)
from skills.registry import SkillImplBase, register_skill  # base + decorator


@register_skill
class ChopTreeSkill(SkillImplBase):
    """
    Implementation of the 'chop_tree' skill.

    High-level behavior:
    - Find a nearby tree/log block from world.blocks_of_interest
    - Move next to it
    - Break the block

    This is intentionally naive; smarter logic (e.g. full tree traversal)
    can be added later without changing the SkillSpec.
    """

    # must match the 'name' field in config/skills/chop_tree.yaml
    skill_name = "chop_tree"

    def suggest_actions(
        self,
        world: WorldState,
        params: Dict[str, Any],
    ) -> List[Action]:
        # read parameters with fallbacks from the YAML defaults
        radius = int(params.get("radius", 8))
        min_logs = int(params.get("min_logs", 4))  # currently unused, but kept for future refinement

        # if world doesn't know about any interesting blocks, no-op
        if not getattr(world, "blocks_of_interest", None):
            return []

        # current position of the agent
        px = world.position.get("x", 0)
        py = world.position.get("y", 0)
        pz = world.position.get("z", 0)

        # pick the nearest block within the radius
        target_block = None
        best_dist2 = None

        for block in world.blocks_of_interest:
            bx = block.get("x")
            by = block.get("y")
            bz = block.get("z")
            if bx is None or by is None or bz is None:
                continue

            dx = bx - px
            dy = by - py
            dz = bz - pz
            dist2 = dx * dx + dy * dy + dz * dz

            if dist2 > radius * radius:
                continue

            if best_dist2 is None or dist2 < best_dist2:
                best_dist2 = dist2
                target_block = block

        # nothing close enough
        if target_block is None:
            return []

        tx = target_block["x"]
        ty = target_block["y"]
        tz = target_block["z"]

        actions: List[Action] = []

        # 1. Move next to the target block
        actions.append(
            Action(
                type="move_to",
                params={
                    "x": tx,
                    "y": ty,
                    "z": tz,
                    "radius": 1,          # stand within 1 block of the target
                },
            )
        )

        # 2. Break the block
        actions.append(
            Action(
                type="break_block",
                params={
                    "x": tx,
                    "y": ty,
                    "z": tz,
                },
            )
        )

        return actions

```
---


## 9. System Diagram


       Skill YAML Specs
            │
            ▼
   SkillSpec Loader (M5)
            │
            ▼
     Skill Registry (M5)
        │             │
        ▼             ▼
   Planner (M2)   AgentLoop (M8)
describe_all()    get_skill().suggest_actions()

---


## 10. Testing

- YAML parsing (SkillSpec correctness)
- Registry:
  - registration
  - deprecation filtering
  - metadata emission
- Skill Pack gating
- Decorator registration

---

## 11. Completion Criteria for M5

M5 is complete when:

1. YAML SkillSpec schema is stable and validated  
2. Skill Packs load and gate skills correctly  
3. Registry exposes correct LLM-facing metadata  
4. Versioning fields (status/origin/metrics) are active  
5. Planner can call `describe_all()` without errors  
6. Agent loop can invoke registered skills  
7. Tests pass for:
   - loader
   - registry
   - decorator
   - skill packs

M5 becomes the **central backbone** of the agent’s behavioral system.

