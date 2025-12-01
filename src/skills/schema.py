# src/skills/schema.py

from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Union


@dataclass
class ParamSpec:
    """Metadata about a single skill parameter."""
    name: str
    type: str
    default: Any
    description: str


@dataclass
class SkillPreconditions:
    """Conditions that must be met before a skill can be used."""
    tech_states_any_of: List[str]
    required_tools: List[Dict[str, Any]]
    dimension_allowlist: List[str]
    semantic_tags_any_of: List[str]
    extra: Dict[str, Any]


@dataclass
class SkillEffects:
    """Expected high-level effects of a skill."""
    inventory_delta: Dict[str, Dict[str, Any]]
    tech_delta: Dict[str, Any]
    tags: List[str]
    extra: Dict[str, Any]


@dataclass
class SkillMetrics:
    """
    Performance/quality metrics for a skill, updated by M10.

    Q1 schema (mandatory in YAML, but allowed to be None in-code while migrating):

    - success_rate: float       (0.0–1.0, rolling)
    - avg_cost: float           (abstract cost units / time / steps)
    - avg_risk: float           (0.0–1.0, estimated failure risk)
    - last_used_at: iso8601 str (timestamp of last successful use)
    """
    success_rate: Optional[float] = None
    avg_cost: Optional[float] = None
    avg_risk: Optional[float] = None
    last_used_at: Optional[str] = None


@dataclass
class SkillMetadata:
    """
    Metadata wrapper for a skill, separated from core behavioral spec.

    Fields (Q1 canonical):

    - version: int | str
    - status: "active" | "deprecated" | "candidate"
    - origin: "hand_authored" | "auto_synthesized"
    - metrics: SkillMetrics
    """
    version: Union[int, str]
    status: str
    origin: str
    metrics: SkillMetrics


@dataclass
class SkillSpec:
    """
    Full declarative specification of a skill.

    This is loaded from YAML and consumed by:
    - SkillRegistry (M5)
    - Planner / Critic (M2) via describe_*()
    - Learning / evolution (M10)

    Q1 requirement: versioning & metrics live under `metadata`.

    The `version`, `status`, `origin`, and `metrics` fields here are
    legacy-compat shims so older loaders that still pass those as
    top-level kwargs don't crash. Canonical versioning lives in
    `metadata.version`, etc.
    """
    name: str
    description: str
    params: Dict[str, ParamSpec]
    preconditions: SkillPreconditions
    effects: SkillEffects
    tags: List[str]

    # Canonical Q1 metadata; may be None when constructed via legacy paths.
    metadata: Optional[SkillMetadata] = None

    # Legacy compatibility: allow SkillSpec(version=..., status=..., origin=..., metrics=...)
    version: Optional[Union[int, str]] = None
    status: Optional[str] = None
    origin: Optional[str] = None
    metrics: Optional[SkillMetrics] = None

    def __post_init__(self) -> None:
        # If metadata is missing, synthesize it from legacy fields.
        if self.metadata is None:
            self.metadata = SkillMetadata(
                version=self.version if self.version is not None else "1",
                status=self.status or "active",
                origin=self.origin or "hand_authored",
                metrics=self.metrics or SkillMetrics(),
            )

