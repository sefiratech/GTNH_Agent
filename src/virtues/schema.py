# path: src/virtues/schema.py

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class VirtueNode:
    id: str
    label: str
    role: str
    tier: str
    pillar: str


@dataclass
class VirtueEdge:
    source: str
    target: str


@dataclass
class VirtueContext:
    id: str
    description: str
    node_weights: Dict[str, float]
    hard_constraints: List[Dict[str, str]]  # simple condition refs for now


@dataclass
class DerivedVirtueSpec:
    id: str
    from_nodes: Dict[str, float]
    description: str


@dataclass
class VirtueConfig:
    """
    Canonical virtue configuration loaded from YAML.

    Fields:
        nodes:
            Mapping from node_id → VirtueNode, representing the base virtue
            lattice (e.g. the 10 sefirot plus any auxiliary nodes).

        edges:
            Directed edges in the lattice, typically used for propagation.

        features:
            Raw mapping from YAML describing how plan features map to nodes.
            Concrete interpretation lives in virtues.metrics.

        contexts:
            Named VirtueContext objects describing different weighting schemes
            (e.g. "factory_safety", "speedrun", "eco_factory").

        derived_virtues:
            Aggregate virtue definitions (e.g. "prudence" from multiple nodes).
    """
    nodes: Dict[str, VirtueNode]
    edges: List[VirtueEdge]
    features: Dict[str, Dict]          # raw mapping from YAML
    contexts: Dict[str, VirtueContext]
    derived_virtues: Dict[str, DerivedVirtueSpec]


@dataclass
class PlanSummary:
    """Compact numeric representation of a plan, post-semantics."""
    id: str
    time_cost: float
    resource_cost: float
    risk_level: float
    pollution_level: float
    infra_reuse_score: float
    infra_impact_score: float
    novelty_score: float
    aesthetic_score: float
    complexity_score: float
    tech_progress_score: float
    stability_score: float
    reversibility_score: float
    # extra context features for constraints / context node, etc.
    context_features: Dict[str, float]


@dataclass
class NodeScore:
    node_id: str
    raw: float
    propagated: float
    rationale: str


@dataclass
class PlanScore:
    """
    Full virtue evaluation for a single plan in a specific context.

    Higher-level modules (AgentLoop, curriculum, learning) usually only
    care about:
        - derived_virtues: dict[str, float]
        - overall_score: float
        - allowed / disallowed_reason
    """
    plan_id: str
    context_id: str
    node_scores: Dict[str, NodeScore]
    derived_virtues: Dict[str, float]
    overall_score: float
    allowed: bool
    disallowed_reason: Optional[str]

