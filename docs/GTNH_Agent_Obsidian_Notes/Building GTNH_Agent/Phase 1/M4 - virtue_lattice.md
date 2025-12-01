**Purpose:**  
Encapsulate your Sefirot-based virtues as a reusable scoring layer.

Overview:

- Define:
    
    - Virtue nodes: Efficiency, Safety, Sustainability, etc.
        
    - Configurable weights per context (e.g., early LV vs late HV)
        
- APIs:
    
    - `score_plan(plan, context) -> dict[virtue -> score]`
        
    - `compare_plans(plans, context) -> best_plan`
        
- **Dependencies:** `M3` (for context & environment semantics)
    
- **Difficulty:** ⭐⭐–⭐⭐⭐
    
- **Scalability/perf:**
    
    - Pure functions, stateless, easy to unit test.
        
    - Configurable weights → you can tune without code changes.


Details:

M4 is where your mystic Kabbalah obsession gets weaponized into a scoring function. Let’s turn the Tree of Life into numbers.

---

# M4 · `virtue_lattice`

**Phase:** P1 – Offline Core Pillars  
**Role:** Score plans according to a Sefirot-inspired virtue system, using GTNH semantics from M3.

- Input:
    
    - A **plan** (sequence of skills + params)
        
    - A **context** (tech state, world snapshot, config profile)
        
- Output:
    
    - `{virtue_name -> score}`
        
    - Optionally: per-metric breakdown and an overall scalar for ranking.
        

Renamed Sefirot:

- Keter = Purpose
    
- Chokhmah = Insight
    
- Binah = Structure
    
- Da’at = Context
    
- Chesed = Generosity
    
- Gevurah = Discipline
    
- Tiferet = Harmony
    
- Netzach = Drive
    
- Hod = Technique
    
- Yesod = Foundation
    
- Malkhut = Manifestation

---

## 1. Responsibilities & Boundaries

### 1.1 What M4 owns

- Definition of the **virtue lattice**:
    
    - Virtue nodes (Sefirot-inspired): e.g. `chesed`, `gevurah`, `tiferet`, mapped to concrete axes like Efficiency, Safety, Sustainability, Harmony, etc.
        
    - Relationships between virtues (lattice edges / balancing pairs).
        
- Context-dependent **weights**:
    
    - e.g. early LV: Efficiency & Safety weighted more; late HV: Sustainability & Stability heavier.
        
- Scoring API:
    
    - `score_plan(plan, context) -> dict[virtue -> score]`
        
    - `compare_plans(plans, context) -> best_plan, scores`
        

### 1.2 What M4 does _not_ do

- No calling LLMs (that’s M2).
    
- No modifying the plan (that’s M8 & M10’s job).
    
- No figuring out world semantics itself (it **uses** M3 outputs).
    

This is a pure scoring layer: “Given metrics, how virtuous is this plan?”

---
# M4 – virtue_lattice

**Phase:** P1 – Offline Core Pillars

**Purpose:**  
Encapsulate your 11-node virtue lattice as a **graph-based scoring engine** that evaluates candidate plans and returns:

- Core virtue scores (the renamed Sefirot nodes)
    
- Derived “lesser” virtues (simple combinations of the core)
    
- A single scalar alignment score for ranking plans
    

This decides **how the agent should act**, not what exists.

---

## High-Level Overview

- Define:
    
    - 11 **core virtues** (renamed Sefirot):  
        `Purpose, Insight, Structure, Context, Generosity, Discipline, Harmony, Drive, Technique, Foundation, Manifestation`
        
    - Their **graph structure** (edges & evaluation order)
        
    - A `PlanSummary` schema (compact, numeric features about a plan)
        
    - A **derived_virtues** layer: named combinations like `patience`, `prudence`, etc.
        
- APIs:
    
    - `score_plan(plan_summary, context_id) -> PlanScore`
        
    - `compare_plans(plan_summaries, context_id) -> (best_plan, [PlanScore])`
        
    - `compute_derived_virtues(node_scores) -> dict[str, float]`
        
    - `explain_plan(plan_score) -> dict | str`
        
- **Dependencies:**
    
    - `M3` (Semantics / TechState / TechGraph)
        
    - `M5` (skill metadata)
        
    - `M11` (for picking context_id later)
        
- **Guaranteed properties:**
    
    - Pure, deterministic functions
        
    - Config-driven (`virtues.yaml`)
        
    - Easy to unit test, log, and introspect
        

---

## 1. Responsibilities & Boundaries

### M4 owns

- The **virtue lattice definition** (nodes, edges, contexts, derived_virtues)
    
- Plan-scoring logic given:
    
    - `PlanSummary`
        
    - `context_id`
        
- Producing:
    
    - `PlanScore` with:
        
        - core node scores
            
        - derived virtue scores
            
        - overall scalar
            
        - allowed / disallowed flag
            

### M4 does _not_ own

- Plan generation (LLMs / M2)
    
- GTNH semantics (M3)
    
- Skill code (M5)
    
- Curriculum sequencing (M11)
    

M4 is a **judge**, not an architect or executor.

---

## 2. Files & Layout
```
GTNH_Agent/
  config/
    virtues.yaml              # core nodes, edges, features, derived_virtues, contexts

  src/
    virtues/
      __init__.py
      schema.py               # dataclasses: VirtueNode, VirtueContext, PlanSummary, NodeScore, PlanScore
      loader.py               # load virtues.yaml into VirtueConfig
      features.py             # map (plan + semantics + skills) -> PlanSummary
      lattice.py              # core scoring: score_plan, compare_plans, compute_derived_virtues
      explain.py              # human-readable explanation + structured debug output
      sanity.py               # config & scoring sanity checks

```

## 3. Config: `config/virtues.yaml`

### 3.1 Core structure (simplified template)
yaml:
```
version: 1

sefirot_nodes:
  purpose:
    label: "Purpose"
    role: "Overarching mission alignment"
    tier: "supernal"
    pillar: "center"

  insight:
    label: "Insight"
    role: "Creative deviation when appropriate"
    tier: "supernal"
    pillar: "right"

  structure:
    label: "Structure"
    role: "Decomposition and clarity"
    tier: "supernal"
    pillar: "left"

  context:
    label: "Context"
    role: "Situational coherence and mode selection"
    tier: "hidden"
    pillar: "center"

  generosity:
    label: "Generosity"
    role: "Redundancy / margin"
    tier: "ethical"
    pillar: "right"

  discipline:
    label: "Discipline"
    role: "Constraints / safety / frugality"
    tier: "ethical"
    pillar: "left"

  harmony:
    label: "Harmony"
    role: "Balance and aesthetic coherence"
    tier: "ethical"
    pillar: "center"

  drive:
    label: "Drive"
    role: "Progress toward long-horizon goals"
    tier: "operational"
    pillar: "right"

  technique:
    label: "Technique"
    role: "Execution quality and correctness"
    tier: "operational"
    pillar: "left"

  foundation:
    label: "Foundation"
    role: "Infrastructure strength and stability"
    tier: "operational"
    pillar: "center"

  manifestation:
    label: "Manifestation"
    role: "Realized world state and tech gains"
    tier: "kingdom"
    pillar: "center"

edges:
  - { from: purpose,      to: insight }
  - { from: purpose,      to: structure }
  - { from: insight,      to: generosity }
  - { from: insight,      to: drive }
  - { from: structure,    to: discipline }
  - { from: structure,    to: technique }
  - { from: generosity,   to: harmony }
  - { from: discipline,   to: harmony }
  - { from: harmony,      to: drive }
  - { from: harmony,      to: technique }
  - { from: drive,        to: foundation }
  - { from: technique,    to: foundation }
  - { from: foundation,   to: manifestation }
  - { from: context,      to: harmony }
  - { from: context,      to: discipline }
  - { from: context,      to: foundation }

# Features: how PlanSummary fields feed node-local raw scores
features:
  time_cost:
    affects:
      discipline:
        direction: "lower_better"
        weight: 0.7
      drive:
        direction: "lower_better"
        weight: 0.3

  resource_cost:
    affects:
      discipline:
        direction: "lower_better"
        weight: 1.0

  risk_level:
    affects:
      discipline:
        direction: "lower_better"
        weight: 0.7
      foundation:
        direction: "lower_better"
        weight: 0.3

  infra_reuse_score:
    affects:
      foundation:
        direction: "higher_better"
        weight: 0.7
      technique:
        direction: "higher_better"
        weight: 0.3

  novelty_score:
    affects:
      insight:
        direction: "higher_better"
        weight: 0.8
      technique:
        direction: "higher_better"
        weight: 0.2

  aesthetic_score:
    affects:
      harmony:
        direction: "higher_better"
        weight: 1.0

  tech_progress_score:
    affects:
      drive:
        direction: "higher_better"
        weight: 0.6
      manifestation:
        direction: "higher_better"
        weight: 0.4

  stability_score:
    affects:
      foundation:
        direction: "higher_better"
        weight: 1.0

# NEW: derived_virtues layer
derived_virtues:
  patience:
    from_nodes:
      drive: 0.4
      discipline: 0.4
      harmony: 0.2
    description: "Steady pursuit of goals without reckless shortcuts."

  prudence:
    from_nodes:
      structure: 0.4
      context: 0.4
      discipline: 0.2
    description: "Acting with foresight and situational awareness."

  industriousness:
    from_nodes:
      drive: 0.5
      technique: 0.3
      foundation: 0.2
    description: "Consistent productive effort that builds lasting systems."

  restraint:
    from_nodes:
      discipline: 0.5
      harmony: 0.3
      generosity: 0.2
    description: "Holding back when excess would harm stability or goals."

contexts:
  default:
    description: "Fallback virtue weighting"
    node_weights:
      purpose:        1.0
      insight:        0.6
      structure:      0.8
      context:        1.0
      generosity:     0.6
      discipline:     0.8
      harmony:        1.0
      drive:          0.7
      technique:      0.7
      foundation:     0.9
      manifestation:  1.0
    hard_constraints:
      - id: "no_base_self_destruction"
        description: "Reject any plan that destroys critical base infra without replacement."
        condition_ref: "base_integrity_min"

  lv_bootstrap:
    description: "Early LV / steam grind"
    node_weights:
      purpose:        1.0
      insight:        0.4
      structure:      0.9
      context:        1.0
      generosity:     0.4
      discipline:     0.9
      harmony:        0.7
      drive:          0.9
      technique:      0.7
      foundation:     0.9
      manifestation:  1.0

  stargate_megaproject:
    description: "Late-game megastructure / aesthetic engineering"
    node_weights:
      purpose:        1.0
      insight:        0.9
      structure:      0.9
      context:        1.0
      generosity:     0.7
      discipline:     0.7
      harmony:        1.0
      drive:          0.9
      technique:      0.8
      foundation:     1.0
      manifestation:  1.0

```

That’s your config spine. Everything else in M4 reads this.

---

## 4. Core Types (`src/virtues/schema.py`)

Skeleton only, but complete enough to wire:
python:
```
# src/virtues/schema.py

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
    plan_id: str
    context_id: str
    node_scores: Dict[str, NodeScore]
    derived_virtues: Dict[str, float]
    overall_score: float
    allowed: bool
    disallowed_reason: Optional[str]

```

## 5. Loader (`src/virtues/loader.py`)

Stub with all pieces:
python:
```
# src/virtues/loader.py

from pathlib import Path
from typing import Dict, Any

import yaml

from .schema import (
    VirtueNode,
    VirtueEdge,
    VirtueContext,
    DerivedVirtueSpec,
    VirtueConfig,
)

CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"
VIRTUES_PATH = CONFIG_DIR / "virtues.yaml"


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_virtue_config(path: Path | None = None) -> VirtueConfig:
    raw = _load_yaml(path or VIRTUES_PATH)

    # nodes
    nodes: Dict[str, VirtueNode] = {}
    for node_id, data in raw["sefirot_nodes"].items():
        nodes[node_id] = VirtueNode(
            id=node_id,
            label=data.get("label", node_id),
            role=data.get("role", ""),
            tier=data.get("tier", ""),
            pillar=data.get("pillar", ""),
        )

    # edges
    edges = [
        VirtueEdge(source=e["from"], target=e["to"])
        for e in raw.get("edges", [])
    ]

    # contexts
    contexts: Dict[str, VirtueContext] = {}
    for ctx_id, ctx_data in raw.get("contexts", {}).items():
        contexts[ctx_id] = VirtueContext(
            id=ctx_id,
            description=ctx_data.get("description", ""),
            node_weights=ctx_data.get("node_weights", {}),
            hard_constraints=ctx_data.get("hard_constraints", []),
        )

    # derived virtues
    derived: Dict[str, DerivedVirtueSpec] = {}
    for dv_id, dv_data in raw.get("derived_virtues", {}).items():
        derived[dv_id] = DerivedVirtueSpec(
            id=dv_id,
            from_nodes=dv_data.get("from_nodes", {}),
            description=dv_data.get("description", ""),
        )

    features = raw.get("features", {})

    return VirtueConfig(
        nodes=nodes,
        edges=edges,
        features=features,
        contexts=contexts,
        derived_virtues=derived,
    )

```

## 6. Feature Extraction (`src/virtues/features.py`)

This is where plan + semantics + skill metadata become `PlanSummary`.

Minimal template:
python:
```
# src/virtues/features.py

from typing import Dict, Any

from .schema import PlanSummary
from spec.types import WorldState  # from M1/M3
from semantics.schema import TechState
from semantics.loader import SemanticsDB


def summarize_plan(
    plan: Dict[str, Any],
    world: WorldState,
    tech_state: TechState,
    db: SemanticsDB,
    skill_metadata: Dict[str, Dict[str, Any]],
) -> PlanSummary:
    """
    Turn a raw plan + semantics into a PlanSummary with numeric features.
    """

    steps = plan.get("steps", [])

    # Placeholder heuristics; you’ll tune later.
    time_cost = float(len(steps))
    resource_cost = 0.0
    risk_level = 0.0
    pollution_level = 0.0
    infra_reuse_score = 0.0
    infra_impact_score = 0.0
    novelty_score = 0.0
    aesthetic_score = plan.get("aesthetic_hint", 0.0)
    complexity_score = min(1.0, len(steps) / 20.0)
    tech_progress_score = 0.0
    stability_score = 0.5
    reversibility_score = 0.5

    for step in steps:
        skill = step.get("skill", "")
        meta = skill_metadata.get(skill, {})
        tags = meta.get("tags", [])

        resource_cost += float(meta.get("estimated_resource_cost", 0.0))
        risk_level += float(meta.get("risk_delta", 0.0))
        pollution_level += float(meta.get("pollution_delta", 0.0))
        infra_reuse_score += float(meta.get("infra_reuse", 0.0))
        infra_impact_score += float(meta.get("infra_impact", 0.0))
        novelty_score += float(meta.get("novelty_delta", 0.0))
        tech_progress_score += float(meta.get("progress_value", 0.0))
        stability_score += float(meta.get("stability_delta", 0.0))
        reversibility_score += float(meta.get("reversibility_delta", 0.0))

    # Clamp / normalize as needed
    risk_level = max(0.0, min(1.0, risk_level))
    pollution_level = max(0.0, min(1.0, pollution_level))
    tech_progress_score = max(0.0, min(1.0, tech_progress_score))

    context_features: Dict[str, float] = {
        "tech_tier": float(tech_state.tier.value) if hasattr(tech_state, "tier") else 0.0,
    }

    return PlanSummary(
        id=plan.get("id", "plan"),
        time_cost=time_cost,
        resource_cost=resource_cost,
        risk_level=risk_level,
        pollution_level=pollution_level,
        infra_reuse_score=infra_reuse_score,
        infra_impact_score=infra_impact_score,
        novelty_score=novelty_score,
        aesthetic_score=aesthetic_score,
        complexity_score=complexity_score,
        tech_progress_score=tech_progress_score,
        stability_score=stability_score,
        reversibility_score=reversibility_score,
        context_features=context_features,
    )
```

## 7. Lattice & Derived Virtues (`src/virtues/lattice.py`)

This is the meat.
python:
```
# src/virtues/lattice.py

from typing import Dict, List, Tuple

from .schema import (
    VirtueConfig,
    PlanSummary,
    NodeScore,
    PlanScore,
)
from .loader import load_virtue_config


def _normalize_feature(value: float, direction: str) -> float:
    """
    Map a raw feature value to [0, 1] where 1.0 is 'more virtuous'.
    Very simple for now; can be tuned later.
    """
    if direction == "higher_better":
        # Assume value in [0, inf); use 1 / (1 + x) inverse for cost-like,
        # but here we just clamp [0,1] assuming upstream normalized.
        return max(0.0, min(1.0, value))
    elif direction == "lower_better":
        # Assume input normalized [0,1] where 1 is worst
        return max(0.0, min(1.0, 1.0 - value))
    else:
        return 0.0


def compute_raw_node_scores(
    plan: PlanSummary,
    config: VirtueConfig,
) -> Dict[str, NodeScore]:
    """
    Use config.features to turn PlanSummary into raw node-local scores.
    """
    # Start everything at neutral 0.5
    scores: Dict[str, NodeScore] = {
        node_id: NodeScore(node_id=node_id, raw=0.5, propagated=0.5, rationale="")
        for node_id in config.nodes.keys()
    }

    feature_map = config.features

    # Map PlanSummary attributes by name
    plan_dict = {
      "time_cost": plan.time_cost,
      "resource_cost": plan.resource_cost,
      "risk_level": plan.risk_level,
      "pollution_level": plan.pollution_level,
      "infra_reuse_score": plan.infra_reuse_score,
      "infra_impact_score": plan.infra_impact_score,
      "novelty_score": plan.novelty_score,
      "aesthetic_score": plan.aesthetic_score,
      "complexity_score": plan.complexity_score,
      "tech_progress_score": plan.tech_progress_score,
      "stability_score": plan.stability_score,
      "reversibility_score": plan.reversibility_score,
    }

    for feature_name, spec in feature_map.items():
        if feature_name not in plan_dict:
            continue
        value = float(plan_dict[feature_name])
        for node_id, mapping in spec.get("affects", {}).items():
            direction = mapping.get("direction", "higher_better")
            weight = float(mapping.get("weight", 0.0))
            contrib = _normalize_feature(value, direction) * weight
            ns = scores[node_id]
            ns.raw += contrib
            scores[node_id] = ns

    # Simple clamp to [0,1]
    for node_id, ns in scores.items():
        ns.raw = max(0.0, min(1.0, ns.raw))
        ns.propagated = ns.raw

    return scores


def propagate_node_scores(
    raw_scores: Dict[str, NodeScore],
    config: VirtueConfig,
) -> Dict[str, NodeScore]:
    """
    Very simple propagation: parent nodes get a small contribution
    from their children, according to the edges.
    """
    scores = {k: NodeScore(**vars(v)) for k, v in raw_scores.items()}

    # For now: one forward pass over edges, no fancy ordering.
    for edge in config.edges:
        if edge.source not in scores or edge.target not in scores:
            continue
        src = scores[edge.source]
        tgt = scores[edge.target]
        # Let source pull up/down a bit based on target
        influence = 0.2  # small coupling
        src.propagated = max(
            0.0,
            min(1.0, (1 - influence) * src.propagated + influence * tgt.raw),
        )
        scores[edge.source] = src

    return scores


def compute_derived_virtues(
    node_scores: Dict[str, NodeScore],
    config: VirtueConfig,
) -> Dict[str, float]:
    """
    Combine core node scores into a small set of derived virtues
    using config.derived_virtues.
    """
    result: Dict[str, float] = {}
    for dv_id, spec in config.derived_virtues.items():
        num = 0.0
        den = 0.0
        for node_id, weight in spec.from_nodes.items():
            if node_id in node_scores:
                num += node_scores[node_id].propagated * weight
                den += abs(weight)
        result[dv_id] = num / den if den > 0 else 0.0
    return result


def score_plan(
    plan: PlanSummary,
    context_id: str,
    config: VirtueConfig | None = None,
) -> PlanScore:
    """
    Full scoring: raw → propagated → derived → overall scalar.
    """
    if config is None:
        config = load_virtue_config()

    context = config.contexts.get(context_id, config.contexts["default"])

    raw = compute_raw_node_scores(plan, config)
    propagated = propagate_node_scores(raw, config)

    # Apply hard constraints (very simple for now: just allowed/disallowed flag)
    allowed = True
    reason = None
    # TODO: actually use context.hard_constraints & plan.context_features

    # Overall scalar: weighted sum of propagated node scores
    overall = 0.0
    for node_id, ns in propagated.items():
        weight = context.node_weights.get(node_id, 0.0)
        overall += ns.propagated * weight

    # Derived virtues
    derived = compute_derived_virtues(propagated, config)

    return PlanScore(
        plan_id=plan.id,
        context_id=context_id,
        node_scores=propagated,
        derived_virtues=derived,
        overall_score=overall,
        allowed=allowed,
        disallowed_reason=reason,
    )


def compare_plans(
    plans: List[PlanSummary],
    context_id: str,
    config: VirtueConfig | None = None,
) -> Tuple[PlanSummary, List[PlanScore]]:
    """
    Score all plans and return (best_plan, all_scores).
    """
    if config is None:
        config = load_virtue_config()

    scores: List[PlanScore] = []
    best_idx = 0
    best_score = float("-inf")

    for idx, plan in enumerate(plans):
        ps = score_plan(plan, context_id, config)
        scores.append(ps)
        if not ps.allowed:
            continue
        if ps.overall_score > best_score:
            best_score = ps.overall_score
            best_idx = idx

    return plans[best_idx], scores

```

## 8. Explanation Layer (`src/virtues/explain.py`)

Template, so you can plug into monitoring:
python:
```
# src/virtues/explain.py

from .schema import PlanScore


def explain_plan(plan_score: PlanScore) -> dict:
    """
    Turn a PlanScore into a structured explanation for logs/UI.
    """
    core = {
        node_id: {
            "raw": ns.raw,
            "propagated": ns.propagated,
            "rationale": ns.rationale,
        }
        for node_id, ns in plan_score.node_scores.items()
    }

    return {
        "plan_id": plan_score.plan_id,
        "context_id": plan_score.context_id,
        "allowed": plan_score.allowed,
        "disallowed_reason": plan_score.disallowed_reason,
        "overall_score": plan_score.overall_score,
        "core_virtues": core,
        "derived_virtues": plan_score.derived_virtues,
    }

```

## 9. Sanity Checks (`src/virtues/sanity.py`)

Bare minimum structure:
python:
```
# src/virtues/sanity.py

from .loader import load_virtue_config


def validate_virtue_config() -> None:
    """
    Run basic sanity checks on virtues.yaml.
    Raise AssertionError or custom exceptions on failures.
    """
    cfg = load_virtue_config()

    assert cfg.nodes, "No virtue nodes defined"
    assert cfg.contexts, "No contexts defined"

    node_ids = set(cfg.nodes.keys())

    # edges reference valid nodes
    for e in cfg.edges:
        assert e.source in node_ids, f"Edge source {e.source} not a node"
        assert e.target in node_ids, f"Edge target {e.target} not a node"

    # contexts have reasonable weights
    for ctx_id, ctx in cfg.contexts.items():
        total_weight = sum(ctx.node_weights.values())
        assert total_weight > 0.0, f"Context {ctx_id} has zero total weight"

    # derived virtues reference valid nodes
    for dv_id, dv in cfg.derived_virtues.items():
        for node_id in dv.from_nodes.keys():
            assert node_id in node_ids, f"Derived virtue {dv_id} references unknown node {node_id}"

```


---

## 10. System Diagram

High-level flow with other modules:

```
               +---------------------------+
               |   Planner (M2 / LLM)     |
               |  raw candidate plans     |
               +-------------+-------------+
                             |
                             v
                 +------------------------+
                 |   M3: Semantics        |
                 | - TechState            |
                 | - SemanticsDB          |
                 +-----------+------------+
                             |
                             v
                 +------------------------+
                 |   M5: Skill Metadata   |
                 | - skill tags/costs     |
                 +-----------+------------+
                             |
                             v
                 +------------------------+
                 |  M4: Feature Layer     |
                 |  features.summarize_plan
                 |  -> PlanSummary        |
                 +-----------+------------+
                             |
                             v
                 +------------------------+
                 |  M4: Virtue Lattice    |
                 |  lattice.score_plan    |
                 |  - core node scores    |
                 |  - derived_virtues     |
                 |  - overall_score       |
                 +-----------+------------+
                             |
                             v
                 +------------------------+
                 | M4: Plan Selection     |
                 | compare_plans          |
                 +------------------------+


```

And around that:

- **M11 (Curriculum)** chooses the active `context_id` for M4 (e.g. `lv_bootstrap`, `stargate_megaproject`).
    
- **M8 (AgentLoop)**:
    
    - collects candidate plans from the planner
        
    - runs them through the M3+M4 pipeline
        
    - uses `compare_plans(...)` to pick the plan it actually executes.
        
- **M9 (Monitoring)** uses `explain_plan(...)` to show:
    
    - core virtue scores
        
    - derived virtues (patience, prudence, etc.)
        
    - chosen vs rejected plans.

---

## 11. Local Testing & Simulation

This module is _less_ painless now, but it’s also the brain, so you don’t get to be lazy.

You still don’t need GTNH running to test 90% of it. Most tests work with:

- fake `PlanSummary` objects
    
- the real `virtues.yaml`
    
- pure `score_plan` / `compare_plans` calls
    

### 11.1 Config & wiring sanity

First: make sure your `virtues.yaml` isn’t cursed.
python:
```
# tests/test_virtue_config_sanity.py

from virtues.sanity import validate_virtue_config


def test_virtue_config_is_sane():
    # This will raise if something is structurally broken:
    # - edges referencing non-existent nodes
    # - contexts with zero total weight
    # - derived virtues pointing at unknown nodes
    validate_virtue_config()

```

If this fails, everything else will give you nonsense, so fix it first.

---

### 11.2 Unit tests with fake PlanSummary (core scoring)

Here you just feed the lattice a couple of toy plans and make sure the scoring behaves like a non-deranged judge.
python:
```
# tests/test_virtue_lattice_basic.py

from virtues.schema import PlanSummary
from virtues.lattice import score_plan
from virtues.loader import load_virtue_config


def _make_plan_summary(
    *,
    plan_id: str,
    time_cost: float,
    resource_cost: float,
    risk_level: float,
    pollution_level: float,
    tech_progress_score: float,
) -> PlanSummary:
    # Fill non-essential fields with neutral-ish defaults
    return PlanSummary(
        id=plan_id,
        time_cost=time_cost,
        resource_cost=resource_cost,
        risk_level=risk_level,
        pollution_level=pollution_level,
        infra_reuse_score=0.5,
        infra_impact_score=0.5,
        novelty_score=0.5,
        aesthetic_score=0.5,
        complexity_score=0.3,
        tech_progress_score=tech_progress_score,
        stability_score=0.5,
        reversibility_score=0.5,
        context_features={},
    )


def test_basic_scoring_low_risk_high_progress_is_better():
    config = load_virtue_config()

    # Plan A: slower, more expensive, higher risk, low progress
    bad_plan = _make_plan_summary(
        plan_id="bad",
        time_cost=10.0,
        resource_cost=5.0,
        risk_level=0.8,           # high risk
        pollution_level=0.6,
        tech_progress_score=0.2,  # meh progress
    )

    # Plan B: cheaper, safer, better progress
    good_plan = _make_plan_summary(
        plan_id="good",
        time_cost=6.0,
        resource_cost=2.0,
        risk_level=0.1,           # low risk
        pollution_level=0.2,
        tech_progress_score=0.7,  # solid progress
    )

    bad_score = score_plan(bad_plan, context_id="lv_bootstrap", config=config)
    good_score = score_plan(good_plan, context_id="lv_bootstrap", config=config)

    assert bad_score.allowed
    assert good_score.allowed

    # sanity: scalar alignment should prefer the good plan
    assert good_score.overall_score > bad_score.overall_score

    # safety / “discipline” node should be better for the lower-risk plan
    good_disc = good_score.node_scores["discipline"].propagated
    bad_disc = bad_score.node_scores["discipline"].propagated
    assert good_disc > bad_disc

    # tech progress should show up in “drive” / “manifestation”
    assert good_score.node_scores["drive"].propagated >= bad_score.node_scores["drive"].propagated
    assert good_score.node_scores["manifestation"].propagated >= bad_score.node_scores["manifestation"].propagated

```

### 11.3 Plan comparison & derived virtues

Now test the full compare flow and make sure derived virtues are computed sanely.
python:
```
# tests/test_virtue_compare_plans.py

from virtues.schema import PlanSummary
from virtues.lattice import compare_plans
from virtues.loader import load_virtue_config


def _plan(
    pid: str,
    *,
    risk_level: float,
    resource_cost: float,
    tech_progress_score: float,
    time_cost: float = 5.0,
) -> PlanSummary:
    return PlanSummary(
        id=pid,
        time_cost=time_cost,
        resource_cost=resource_cost,
        risk_level=risk_level,
        pollution_level=0.2,
        infra_reuse_score=0.5,
        infra_impact_score=0.5,
        novelty_score=0.5,
        aesthetic_score=0.5,
        complexity_score=0.3,
        tech_progress_score=tech_progress_score,
        stability_score=0.5,
        reversibility_score=0.5,
        context_features={},
    )


def test_compare_plans_prefers_safer_progress_plan():
    config = load_virtue_config()

    # Plan 0: high risk, slightly more progress
    risky = _plan(
        "risky",
        risk_level=0.9,
        resource_cost=2.0,
        tech_progress_score=0.8,
    )

    # Plan 1: much safer, slightly less progress
    safe = _plan(
        "safe",
        risk_level=0.1,
        resource_cost=2.0,
        tech_progress_score=0.7,
    )

    best, scores = compare_plans(
        plans=[risky, safe],
        context_id="lv_bootstrap",
        config=config,
    )

    # Should pick “safe” in LV context, where Discipline & Foundation are heavy
    assert best.id == "safe"

    # Derived virtues should reflect that too (e.g. patience, prudence)
    risky_score = next(s for s in scores if s.plan_id == "risky")
    safe_score = next(s for s in scores if s.plan_id == "safe")

    patience_safe = safe_score.derived_virtues.get("patience", 0.0)
    patience_risky = risky_score.derived_virtues.get("patience", 0.0)
    assert patience_safe > patience_risky

```

You’re checking:

- `compare_plans` actually uses the lattice, not random nonsense
    
- derived virtues move in the intuitive direction (safer plan = more “patience”)
    

---

### 11.4 Hard constraint behavior (optional but important)

You’ll eventually wire real constraints (e.g. “don’t blow up main power”). When you do, test that too.
python:
```
# tests/test_virtue_hard_constraints.py

from virtues.schema import PlanSummary
from virtues.lattice import score_plan
from virtues.loader import load_virtue_config


def test_plan_can_be_disallowed_by_constraints():
    config = load_virtue_config()

    # Fake plan that clearly violates some hypothetical constraint
    # For now, just stuff context_features with something you'd check later.
    evil_plan = PlanSummary(
        id="delete_base",
        time_cost=1.0,
        resource_cost=0.1,
        risk_level=1.0,
        pollution_level=1.0,
        infra_reuse_score=0.0,
        infra_impact_score=-1.0,   # negative = destroys infra, for example
        novelty_score=0.5,
        aesthetic_score=0.0,
        complexity_score=0.1,
        tech_progress_score=0.0,
        stability_score=0.0,
        reversibility_score=0.0,
        context_features={"base_integrity_delta": -1.0},
    )

    score = score_plan(evil_plan, context_id="default", config=config)

    # Once hard-constraint logic is implemented, this should flip:
    # assert not score.allowed
    # For now we just assert the test runs; you’ll tighten this when the constraint handler is written.
    assert score.plan_id == "delete_base"

```

You’ll tighten that once you hook `hard_constraints` into `score_plan`.

---

### 11.5 Integration with semantics (later, but straightforward)

Once M3 is ready, add a higher-level integration test:

- Build a fake `WorldState`, `TechState`, `SemanticsDB`, and `skill_metadata`.
    
- Build a fake planner output: `plan = {"id": "lv_boiler", "steps": [...skills...]}`.
    
- Run:
python:
```
from virtues.features import summarize_plan
from virtues.lattice import score_plan, compare_plans

def test_end_to_end_lv_scenario(fake_world, fake_tech_state, fake_semantics_db, fake_skill_metadata):
    plan_a = {...}  # e.g. build coke ovens badly
    plan_b = {...}  # e.g. build them in a sane layout

    summary_a = summarize_plan(plan_a, fake_world, fake_tech_state, fake_semantics_db, fake_skill_metadata)
    summary_b = summarize_plan(plan_b, fake_world, fake_tech_state, fake_semantics_db, fake_skill_metadata)

    best, scores = compare_plans(
        plans=[summary_a, summary_b],
        context_id="lv_bootstrap",
    )

    # Should pick the one that actually improves infra, reduces risk, etc.
    assert best.id in ("sane_layout", "whatever_you_named_it")

```

This is where you confirm the _whole_ stack behaves: semantics + features + lattice + comparison.

---

That’s the testing section updated. No more fake `PlanMetrics`, no orphaned helpers, and the derived_virtues layer is actually in the loop instead of being theoretical décor.


## 12. Completion Criteria for M4 (v1)

You’re “shipped” on this module when:

- `config/virtues.yaml`:
    
    - defines 11 core virtues
        
    - defines edges
        
    - defines `features`
        
    - defines a small set of `derived_virtues`
        
    - defines at least `default`, `lv_bootstrap`, `stargate_megaproject` contexts
        
- `schema.py`, `loader.py`, `features.py`, `lattice.py`, `explain.py`, `sanity.py` all exist and import cleanly
    
- You can:
    
    - create a couple of fake `PlanSummary` objects
        
    - run `compare_plans` for some context
        
    - see core + derived virtue scores and a clear “best” plan
        
- `validate_virtue_config()` passes with your actual YAML
    

That’s your M4 template, upgraded and fully wired with **derived_virtues** but still sane enough to actually implement without a theology degree.
