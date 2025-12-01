# **M7 – observation_encoding**

_Module Workflow Document_

---

## **Module Summary**

**Purpose:**  
Convert `RawWorldSnapshot` from **M6** into compact, stable, LLM-friendly JSON encodings used by the **planner** and **critic** in M2.  
This module is the agent’s perceptual bottleneck: it decides what the brain sees and how it sees it.

**Role in architecture:**  
M7 sits between raw Minecraft packets and higher-level reasoning.  
It shapes data from M6 (raw) and M3 (semantics) into concise, structured observations that M2 can reliably consume.

---

## **Status**

- **Phase:** P2 – Minecraft Integration
    
- **Difficulty:** ⭐⭐–⭐⭐⭐
    
- **Dependencies:**
    
    - **M6:** Raw world snapshots
        
    - **M3:** SemanticsDB, TechState, craftability
        
    - **M1:** Observation type
        
    - **M4:** Only to load context IDs (not scoring)
        
- **Stability goal:**  
    This module must maintain a stable JSON schema to prevent breaking old skills or planner assumptions.
    

---

## **Responsibilities & Boundaries**

### **What M7 _owns_**

- The observation schema for the planner:
    
    - `encode_for_planner(...) -> dict`
        
    - `make_planner_observation(...) -> Observation`
        
- The observation schema for the critic:
    
    - `encode_for_critic(trace) -> dict`
        
- All summarization logic:
    
    - Spatial summary
        
    - Inventory categorization
        
    - Machine detection
        
    - Craftability snapshot
        
    - Entity filtering
        
    - Environmental context
        
- Generation of short NL summaries for LLM prompting.
    
- Enforcing compactness and caps (entities, craftables, etc.)
    

### **What M7 does _not_ own**

- No world inference (chunks, positions, entities). That’s M6.
    
- No semantics inference or classification. That’s M3.
    
- No scoring or moral evaluation. That’s M4.
    
- No planning or critique logic. That’s M2.
    

---

## **High-Level Flow**
```
M6 Raw Snapshot
      |
      v
M3 Semantics + TechState
      |
      v
M7 Observation Encoder
      |
      +-----→ PlannerModel (M2)
      |
      +-----→ CriticModel (M2)

```

---

## **Data Types Used**

### **Inputs**

- `RawWorldSnapshot` (M6)
    
- `TechState` (M3)
    
- `SemanticsDB` (M3)
    
- `PlanTrace`, `TraceStep` (internal)
    
- `context_id` from M4 config
    

### **Outputs**

#### **PlannerEncoding**

- `tech_state`
    
- `agent`
    
- `inventory_summary`
    
- `machines_summary`
    
- `nearby_entities`
    
- `env_summary`
    
- `craftable_summary`
    
- `context_id`
    
- `text_summary`
    

#### **CriticEncoding**

- `tech_state`
    
- `context_id`
    
- `plan`
    
- `steps`
    
- `planner_observation`
    
- `virtue_scores`
    
- `text_summary`
    

Full Python dataclasses are preserved exactly as defined in the context primer.

```
# src/observation/schema.py
"""
Schema types for M7: observation encodings.

These are the JSON-like encodings consumed by:
- PlannerModel (M2)
- CriticModel (M2)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

# M1: LLM-facing Observation (used in make_planner_observation)
from spec.types import Observation


@dataclass
class PlannerEncoding:
    """
    Structured JSON-style encoding for the planner.

    This is *not* the Observation type itself; it's the payload that will be
    attached to Observation.json_payload.
    """

    tech_state: Dict[str, Any]
    agent: Dict[str, Any]
    inventory_summary: Dict[str, Any]
    machines_summary: Dict[str, Any]
    nearby_entities: List[Dict[str, Any]]
    env_summary: Dict[str, Any]
    craftable_summary: Dict[str, Any]
    context_id: str
    text_summary: str


@dataclass
class CriticEncoding:
    """
    Structured JSON-style encoding for the critic.

    Built from a PlanTrace (see trace_schema.py) and used to drive the
    CriticModel's evaluation / scoring.
    """

    tech_state: Dict[str, Any]
    context_id: str
    plan: Dict[str, Any]
    steps: List[Dict[str, Any]]
    planner_observation: Dict[str, Any]
    virtue_scores: Optional[Dict[str, Any]]
    text_summary: str


__all__ = [
    "PlannerEncoding",
    "CriticEncoding",
]

```

```
# src/observation/trace_schema.py
"""
Trace schema for critic encoding (M7).

Defines the internal types that represent an executed plan:
- TraceStep: single action + world before/after + metadata
- PlanTrace: full execution trace for a plan
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

# Core agent types from M1
from spec.types import WorldState, Action, ActionResult

# Tech state from M3
from semantics.schema import TechState


@dataclass
class TraceStep:
    """
    One step in the execution trace.

    world_before/world_after should be *reasonable* views of the world, not full
    raw snapshots; use whatever M1 WorldState gives you.
    """

    world_before: WorldState
    action: Action
    result: ActionResult
    world_after: WorldState
    meta: Dict[str, Any]


@dataclass
class PlanTrace:
    """
    Execution trace for a plan.

    planner_payload is usually the PlannerEncoding dict that was given
    to the planner when this plan was generated.
    """

    plan: Dict[str, Any]
    steps: List[TraceStep]
    tech_state: TechState
    planner_payload: Dict[str, Any]
    context_id: str
    virtue_scores: Dict[str, Any]


__all__ = [
    "TraceStep",
    "PlanTrace",
]

```

```
# src/observation/encoder.py
"""
M7 - Observation Encoder

Turns:
    - RawWorldSnapshot (M6)
    - TechState + SemanticsDB (M3)
into:
    - PlannerEncoding (for PlannerModel, M2)
    - CriticEncoding (for CriticModel, M2)

This is the bottleneck where we choose what the LLM actually sees.
"""

from __future__ import annotations

from dataclasses import asdict
from math import sqrt
from typing import Any, Dict, List

# M6: raw world snapshot types
from bot_core.snapshot import RawWorldSnapshot, RawEntity

# M3: semantics + tech state
from semantics.loader import SemanticsDB
from semantics.schema import TechState
from semantics.crafting import craftable_items

# M1: core types
from spec.types import Observation, WorldState

# M4: virtue context ids live in config; we just pass strings around here

# Local schema types
from .schema import PlannerEncoding, CriticEncoding
from .trace_schema import PlanTrace, TraceStep


# ---------------------------------------------------------------------------
# Tech state serialization
# ---------------------------------------------------------------------------


def _serialize_tech_state(tech_state: TechState) -> Dict[str, Any]:
    """
    Turn TechState dataclass into JSON-like dict.

    Assumes TechState has:
        - active: str
        - unlocked: Collection[str]
        - evidence: optional and omitted/compressed here
    """
    # If TechState is a dataclass, asdict will work; otherwise fall back.
    try:
        data = asdict(tech_state)
    except TypeError:
        # Manual minimal serialization
        data = {
            "active": getattr(tech_state, "active", None),
            "unlocked": list(getattr(tech_state, "unlocked", [])),
        }

    # Be explicit about the keys we care about.
    return {
        "active": data.get("active"),
        "unlocked": list(data.get("unlocked", [])),
    }


# ---------------------------------------------------------------------------
# Planner helpers
# ---------------------------------------------------------------------------


def _summarize_agent(raw: RawWorldSnapshot) -> Dict[str, Any]:
    """Summarize agent position and basic status."""
    pos = raw.player_pos
    return {
        "position": pos,
        "dimension": raw.dimension,
        "on_ground": raw.on_ground,
        "yaw": raw.player_yaw,
        "pitch": raw.player_pitch,
    }


def _summarize_inventory(
    raw: RawWorldSnapshot,
    db: SemanticsDB,
) -> Dict[str, Any]:
    """
    Compress inventory into counts by:
        - item
        - category (from SemanticsDB)
        - material (from SemanticsDB)

    Expects inventory entries like:
        {"item_id": "modid:item", "variant": "...", "count": 12}
    and SemanticsDB.get_item_info(item_id, variant) returning an object with
        - category: str
        - material: Optional[str]
    """
    summary: Dict[str, Any] = {
        "by_item": {},
        "by_category": {},
        "by_material": {},
    }

    for stack in raw.inventory:
        item_id = stack.get("item_id")
        variant = stack.get("variant")
        count = int(stack.get("count", 0))

        if not item_id or count <= 0:
            continue

        try:
            info = db.get_item_info(item_id, variant)
        except Exception:
            # If semantics is missing, just bucket under "unknown".
            info = type("TmpInfo", (), {"category": "unknown", "material": "unknown"})()

        key_item = f"{item_id}:{variant}" if variant else item_id
        summary["by_item"].setdefault(key_item, 0)
        summary["by_item"][key_item] += count

        cat = getattr(info, "category", "unknown") or "unknown"
        summary["by_category"].setdefault(cat, 0)
        summary["by_category"][cat] += count

        mat = getattr(info, "material", "unknown") or "unknown"
        summary["by_material"].setdefault(mat, 0)
        summary["by_material"][mat] += count

    return summary


def _summarize_machines(raw: RawWorldSnapshot) -> Dict[str, Any]:
    """
    Summarize machines & power sources from raw.context["machines"].

    Expects M6/M3 to populate raw.context["machines"] with entries like:
        {
            "type": "gregtech:steam_bronze_boiler",
            "tier": "steam" | "lv" | "mv" | ...
        }
    """
    machines: List[Dict[str, Any]] = raw.context.get("machines", []) or []

    summary: Dict[str, Any] = {
        "count": len(machines),
        "by_type": {},
        "by_tier": {},
        "has_steam": False,
        "has_lv": False,
        "has_mv": False,
    }

    for m in machines:
        mtype = m.get("type", "unknown")
        tier = m.get("tier", "unknown")

        summary["by_type"].setdefault(mtype, 0)
        summary["by_type"][mtype] += 1

        summary["by_tier"].setdefault(tier, 0)
        summary["by_tier"][tier] += 1

        if "steam" in (mtype or "").lower() or tier == "steam":
            summary["has_steam"] = True
        if tier == "lv":
            summary["has_lv"] = True
        if tier == "mv":
            summary["has_mv"] = True

    return summary


def _summarize_nearby_entities(
    raw: RawWorldSnapshot,
    radius: float = 16.0,
    max_entities: int = 16,
) -> List[Dict[str, Any]]:
    """
    Filter entities around player within a limited radius, sorted by distance.

    Expects RawEntity:
        entity_id, type, x, y, z, data: Dict[str, Any]
    """
    px = raw.player_pos.get("x", 0.0)
    py = raw.player_pos.get("y", 0.0)
    pz = raw.player_pos.get("z", 0.0)

    result: List[Dict[str, Any]] = []

    for e in raw.entities:
        if not isinstance(e, RawEntity):
            # If someone passes dicts for tests, handle that.
            ex = getattr(e, "x", e.get("x", 0.0))
            ey = getattr(e, "y", e.get("y", 0.0))
            ez = getattr(e, "z", e.get("z", 0.0))
            etype = getattr(e, "type", e.get("type", "unknown"))
            edata = getattr(e, "data", e.get("data", {}))
        else:
            ex, ey, ez = e.x, e.y, e.z
            etype = e.type
            edata = e.data or {}

        dx = ex - px
        dy = ey - py
        dz = ez - pz
        dist = sqrt(dx * dx + dy * dy + dz * dz)
        if dist > radius:
            continue

        result.append(
            {
                "type": etype,
                "x": ex,
                "y": ey,
                "z": ez,
                "distance": dist,
                "is_hostile": bool(edata.get("hostile", False)),
                "is_item": etype == "item",
            }
        )

    result.sort(key=lambda ent: ent["distance"])
    return result[:max_entities]


def _summarize_env(raw: RawWorldSnapshot) -> Dict[str, Any]:
    """
    Misc environmental summary. Assumes raw.context contains arbitrary metadata
    like warnings/notes.
    """
    return {
        "tick": raw.tick,
        "dimension": raw.dimension,
        "warnings": raw.context.get("warnings", []),
        "notes": raw.context.get("notes", ""),
    }


def _summarize_craftables(
    raw: RawWorldSnapshot,
    db: SemanticsDB,
    max_outputs: int = 10,
) -> Dict[str, Any]:
    """
    Summarize what we can craft *right now* into a small list.

    Builds a minimal WorldState wrapper for craftable_items.
    """
    world = WorldState(
        tick=raw.tick,
        position=raw.player_pos,
        dimension=raw.dimension,
        inventory=raw.inventory,
        nearby_entities=[],
        blocks_of_interest=[],
        tech_state={},  # TechState logic is separate; this is for craftability.
        context=raw.context,
    )

    options = craftable_items(world, db)
    summary: Dict[str, Any] = {"top_outputs": []}

    for opt in options[:max_outputs]:
        summary["top_outputs"].append(
            {
                "output_item": getattr(opt, "output_item", None),
                "output_count": getattr(opt, "output_count", None),
                "recipe_id": getattr(opt, "recipe_id", None),
                "notes": getattr(opt, "notes", ""),
            }
        )

    return summary


def _make_text_summary(
    raw: RawWorldSnapshot,
    tech_state: TechState,
    machines_summary: Dict[str, Any],
    inventory_summary: Dict[str, Any],
) -> str:
    """
    Build a compact one-paragraph summary as prompt context.

    This is intentionally short; the bulk of reasoning should use the structured
    JSON, not this string.
    """
    dim = raw.dimension
    active_tech = getattr(tech_state, "active", None)
    inv_cats = inventory_summary.get("by_category", {})
    machines_count = machines_summary.get("count", 0)

    top_cats = sorted(inv_cats.items(), key=lambda kv: kv[1], reverse=True)[:3]
    if top_cats:
        top_cats_str = ", ".join(f"{cat} x{amt}" for cat, amt in top_cats)
    else:
        top_cats_str = "no notable categories"

    return (
        f"You are in dimension '{dim}' at position {raw.player_pos}, "
        f"current tech state is '{active_tech}'. "
        f"There are {machines_count} known machines. "
        f"Key inventory categories: {top_cats_str}."
    )


# ---------------------------------------------------------------------------
# Public planner API
# ---------------------------------------------------------------------------


def encode_for_planner(
    raw_snapshot: RawWorldSnapshot,
    tech_state: TechState,
    db: SemanticsDB,
    context_id: str,
) -> Dict[str, Any]:
    """
    Build a JSON-serializable dict suitable as Observation.json_payload
    for the planner.

    Returns a dict matching PlannerEncoding fields.
    """
    agent = _summarize_agent(raw_snapshot)
    inv_summary = _summarize_inventory(raw_snapshot, db)
    machines_summary = _summarize_machines(raw_snapshot)
    entities = _summarize_nearby_entities(raw_snapshot)
    env_summary = _summarize_env(raw_snapshot)
    craftable_summary = _summarize_craftables(raw_snapshot, db)
    tech_dict = _serialize_tech_state(tech_state)

    text_summary = _make_text_summary(
        raw_snapshot,
        tech_state,
        machines_summary,
        inv_summary,
    )

    enc = PlannerEncoding(
        tech_state=tech_dict,
        agent=agent,
        inventory_summary=inv_summary,
        machines_summary=machines_summary,
        nearby_entities=entities,
        env_summary=env_summary,
        craftable_summary=craftable_summary,
        context_id=context_id,
        text_summary=text_summary,
    )

    # Explicit dict instead of asdict(enc) to keep schema obvious/stable.
    return {
        "tech_state": enc.tech_state,
        "agent": enc.agent,
        "inventory_summary": enc.inventory_summary,
        "machines_summary": enc.machines_summary,
        "nearby_entities": enc.nearby_entities,
        "env_summary": enc.env_summary,
        "craftable_summary": enc.craftable_summary,
        "context_id": enc.context_id,
        "text_summary": enc.text_summary,
    }


def make_planner_observation(
    raw_snapshot: RawWorldSnapshot,
    tech_state: TechState,
    db: SemanticsDB,
    context_id: str,
) -> Observation:
    """
    Convenience function to create an M1 Observation for the planner LLM.
    """
    payload = encode_for_planner(raw_snapshot, tech_state, db, context_id)
    return Observation(
        json_payload=payload,
        text_summary=payload["text_summary"],
    )


# ---------------------------------------------------------------------------
# Critic encoding
# ---------------------------------------------------------------------------


def _summarize_step(step: TraceStep) -> Dict[str, Any]:
    """
    Summarize a single TraceStep into critic-friendly JSON.

    Exposes:
        - action type/params
        - result success/error
        - meta
        - before/after positions
    """
    world_before_pos = getattr(step.world_before, "position", None)
    world_after_pos = getattr(step.world_after, "position", None)

    return {
        "action": {
            "type": getattr(step.action, "type", None),
            "params": getattr(step.action, "params", {}),
        },
        "result": {
            "success": getattr(step.result, "success", False),
            "error": getattr(step.result, "error", None),
        },
        "meta": step.meta,
        "world_before_pos": world_before_pos,
        "world_after_pos": world_after_pos,
    }


def _make_critic_text_summary(trace: PlanTrace) -> str:
    """
    Brief natural-language summary for the critic context.
    """
    plan_steps = len(trace.plan.get("steps", []))
    exec_steps = len(trace.steps)
    active_tech = getattr(trace.tech_state, "active", None)
    ctx = trace.context_id

    return (
        f"Evaluating execution of plan with {plan_steps} steps "
        f"over {exec_steps} execution steps at tech state '{active_tech}' "
        f"in context '{ctx}'."
    )


def encode_for_critic(trace: PlanTrace) -> Dict[str, Any]:
    """
    Build JSON encoding for the critic model, based on a PlanTrace instance.

    Returns a dict matching CriticEncoding fields.
    """
    tech_dict = _serialize_tech_state(trace.tech_state)
    step_json = [_summarize_step(s) for s in trace.steps]
    text_summary = _make_critic_text_summary(trace)

    enc = CriticEncoding(
        tech_state=tech_dict,
        context_id=trace.context_id,
        plan=trace.plan,
        steps=step_json,
        planner_observation=trace.planner_payload,
        virtue_scores=trace.virtue_scores or {},
        text_summary=text_summary,
    )

    return {
        "tech_state": enc.tech_state,
        "context_id": enc.context_id,
        "plan": enc.plan,
        "steps": enc.steps,
        "planner_observation": enc.planner_observation,
        "virtue_scores": enc.virtue_scores,
        "text_summary": enc.text_summary,
    }


__all__ = [
    "encode_for_planner",
    "make_planner_observation",
    "encode_for_critic",
]

```

```
# src/observation/encoder.py
"""
M7 - Observation Encoder

Turns:
    - RawWorldSnapshot (M6)
    - TechState + SemanticsDB (M3)
into:
    - PlannerEncoding (for PlannerModel, M2)
    - CriticEncoding (for CriticModel, M2)

This is the bottleneck where we choose what the LLM actually sees.
"""

from __future__ import annotations

from dataclasses import asdict
from math import sqrt
from typing import Any, Dict, List

# M6: raw world snapshot types
from bot_core.snapshot import RawWorldSnapshot, RawEntity

# M3: semantics + tech state
from semantics.loader import SemanticsDB
from semantics.schema import TechState
from semantics.crafting import craftable_items

# M1: core types
from spec.types import Observation, WorldState

# M4: virtue context ids live in config; we just pass strings around here

# Local schema types
from .schema import PlannerEncoding, CriticEncoding
from .trace_schema import PlanTrace, TraceStep


# ---------------------------------------------------------------------------
# Tech state serialization
# ---------------------------------------------------------------------------


def _serialize_tech_state(tech_state: TechState) -> Dict[str, Any]:
    """
    Turn TechState dataclass into JSON-like dict.

    Assumes TechState has:
        - active: str
        - unlocked: Collection[str]
        - evidence: optional and omitted/compressed here
    """
    # If TechState is a dataclass, asdict will work; otherwise fall back.
    try:
        data = asdict(tech_state)
    except TypeError:
        # Manual minimal serialization
        data = {
            "active": getattr(tech_state, "active", None),
            "unlocked": list(getattr(tech_state, "unlocked", [])),
        }

    # Be explicit about the keys we care about.
    return {
        "active": data.get("active"),
        "unlocked": list(data.get("unlocked", [])),
    }


# ---------------------------------------------------------------------------
# Planner helpers
# ---------------------------------------------------------------------------


def _summarize_agent(raw: RawWorldSnapshot) -> Dict[str, Any]:
    """Summarize agent position and basic status."""
    pos = raw.player_pos
    return {
        "position": pos,
        "dimension": raw.dimension,
        "on_ground": raw.on_ground,
        "yaw": raw.player_yaw,
        "pitch": raw.player_pitch,
    }


def _summarize_inventory(
    raw: RawWorldSnapshot,
    db: SemanticsDB,
) -> Dict[str, Any]:
    """
    Compress inventory into counts by:
        - item
        - category (from SemanticsDB)
        - material (from SemanticsDB)

    Expects inventory entries like:
        {"item_id": "modid:item", "variant": "...", "count": 12}
    and SemanticsDB.get_item_info(item_id, variant) returning an object with
        - category: str
        - material: Optional[str]
    """
    summary: Dict[str, Any] = {
        "by_item": {},
        "by_category": {},
        "by_material": {},
    }

    for stack in raw.inventory:
        item_id = stack.get("item_id")
        variant = stack.get("variant")
        count = int(stack.get("count", 0))

        if not item_id or count <= 0:
            continue

        try:
            info = db.get_item_info(item_id, variant)
        except Exception:
            # If semantics is missing, just bucket under "unknown".
            info = type("TmpInfo", (), {"category": "unknown", "material": "unknown"})()

        key_item = f"{item_id}:{variant}" if variant else item_id
        summary["by_item"].setdefault(key_item, 0)
        summary["by_item"][key_item] += count

        cat = getattr(info, "category", "unknown") or "unknown"
        summary["by_category"].setdefault(cat, 0)
        summary["by_category"][cat] += count

        mat = getattr(info, "material", "unknown") or "unknown"
        summary["by_material"].setdefault(mat, 0)
        summary["by_material"][mat] += count

    return summary


def _summarize_machines(raw: RawWorldSnapshot) -> Dict[str, Any]:
    """
    Summarize machines & power sources from raw.context["machines"].

    Expects M6/M3 to populate raw.context["machines"] with entries like:
        {
            "type": "gregtech:steam_bronze_boiler",
            "tier": "steam" | "lv" | "mv" | ...
        }
    """
    machines: List[Dict[str, Any]] = raw.context.get("machines", []) or []

    summary: Dict[str, Any] = {
        "count": len(machines),
        "by_type": {},
        "by_tier": {},
        "has_steam": False,
        "has_lv": False,
        "has_mv": False,
    }

    for m in machines:
        mtype = m.get("type", "unknown")
        tier = m.get("tier", "unknown")

        summary["by_type"].setdefault(mtype, 0)
        summary["by_type"][mtype] += 1

        summary["by_tier"].setdefault(tier, 0)
        summary["by_tier"][tier] += 1

        if "steam" in (mtype or "").lower() or tier == "steam":
            summary["has_steam"] = True
        if tier == "lv":
            summary["has_lv"] = True
        if tier == "mv":
            summary["has_mv"] = True

    return summary


def _summarize_nearby_entities(
    raw: RawWorldSnapshot,
    radius: float = 16.0,
    max_entities: int = 16,
) -> List[Dict[str, Any]]:
    """
    Filter entities around player within a limited radius, sorted by distance.

    Expects RawEntity:
        entity_id, type, x, y, z, data: Dict[str, Any]
    """
    px = raw.player_pos.get("x", 0.0)
    py = raw.player_pos.get("y", 0.0)
    pz = raw.player_pos.get("z", 0.0)

    result: List[Dict[str, Any]] = []

    for e in raw.entities:
        if not isinstance(e, RawEntity):
            # If someone passes dicts for tests, handle that.
            ex = getattr(e, "x", e.get("x", 0.0))
            ey = getattr(e, "y", e.get("y", 0.0))
            ez = getattr(e, "z", e.get("z", 0.0))
            etype = getattr(e, "type", e.get("type", "unknown"))
            edata = getattr(e, "data", e.get("data", {}))
        else:
            ex, ey, ez = e.x, e.y, e.z
            etype = e.type
            edata = e.data or {}

        dx = ex - px
        dy = ey - py
        dz = ez - pz
        dist = sqrt(dx * dx + dy * dy + dz * dz)
        if dist > radius:
            continue

        result.append(
            {
                "type": etype,
                "x": ex,
                "y": ey,
                "z": ez,
                "distance": dist,
                "is_hostile": bool(edata.get("hostile", False)),
                "is_item": etype == "item",
            }
        )

    result.sort(key=lambda ent: ent["distance"])
    return result[:max_entities]


def _summarize_env(raw: RawWorldSnapshot) -> Dict[str, Any]:
    """
    Misc environmental summary. Assumes raw.context contains arbitrary metadata
    like warnings/notes.
    """
    return {
        "tick": raw.tick,
        "dimension": raw.dimension,
        "warnings": raw.context.get("warnings", []),
        "notes": raw.context.get("notes", ""),
    }


def _summarize_craftables(
    raw: RawWorldSnapshot,
    db: SemanticsDB,
    max_outputs: int = 10,
) -> Dict[str, Any]:
    """
    Summarize what we can craft *right now* into a small list.

    Builds a minimal WorldState wrapper for craftable_items.
    """
    world = WorldState(
        tick=raw.tick,
        position=raw.player_pos,
        dimension=raw.dimension,
        inventory=raw.inventory,
        nearby_entities=[],
        blocks_of_interest=[],
        tech_state={},  # TechState logic is separate; this is for craftability.
        context=raw.context,
    )

    options = craftable_items(world, db)
    summary: Dict[str, Any] = {"top_outputs": []}

    for opt in options[:max_outputs]:
        summary["top_outputs"].append(
            {
                "output_item": getattr(opt, "output_item", None),
                "output_count": getattr(opt, "output_count", None),
                "recipe_id": getattr(opt, "recipe_id", None),
                "notes": getattr(opt, "notes", ""),
            }
        )

    return summary


def _make_text_summary(
    raw: RawWorldSnapshot,
    tech_state: TechState,
    machines_summary: Dict[str, Any],
    inventory_summary: Dict[str, Any],
) -> str:
    """
    Build a compact one-paragraph summary as prompt context.

    This is intentionally short; the bulk of reasoning should use the structured
    JSON, not this string.
    """
    dim = raw.dimension
    active_tech = getattr(tech_state, "active", None)
    inv_cats = inventory_summary.get("by_category", {})
    machines_count = machines_summary.get("count", 0)

    top_cats = sorted(inv_cats.items(), key=lambda kv: kv[1], reverse=True)[:3]
    if top_cats:
        top_cats_str = ", ".join(f"{cat} x{amt}" for cat, amt in top_cats)
    else:
        top_cats_str = "no notable categories"

    return (
        f"You are in dimension '{dim}' at position {raw.player_pos}, "
        f"current tech state is '{active_tech}'. "
        f"There are {machines_count} known machines. "
        f"Key inventory categories: {top_cats_str}."
    )


# ---------------------------------------------------------------------------
# Public planner API
# ---------------------------------------------------------------------------


def encode_for_planner(
    raw_snapshot: RawWorldSnapshot,
    tech_state: TechState,
    db: SemanticsDB,
    context_id: str,
) -> Dict[str, Any]:
    """
    Build a JSON-serializable dict suitable as Observation.json_payload
    for the planner.

    Returns a dict matching PlannerEncoding fields.
    """
    agent = _summarize_agent(raw_snapshot)
    inv_summary = _summarize_inventory(raw_snapshot, db)
    machines_summary = _summarize_machines(raw_snapshot)
    entities = _summarize_nearby_entities(raw_snapshot)
    env_summary = _summarize_env(raw_snapshot)
    craftable_summary = _summarize_craftables(raw_snapshot, db)
    tech_dict = _serialize_tech_state(tech_state)

    text_summary = _make_text_summary(
        raw_snapshot,
        tech_state,
        machines_summary,
        inv_summary,
    )

    enc = PlannerEncoding(
        tech_state=tech_dict,
        agent=agent,
        inventory_summary=inv_summary,
        machines_summary=machines_summary,
        nearby_entities=entities,
        env_summary=env_summary,
        craftable_summary=craftable_summary,
        context_id=context_id,
        text_summary=text_summary,
    )

    # Explicit dict instead of asdict(enc) to keep schema obvious/stable.
    return {
        "tech_state": enc.tech_state,
        "agent": enc.agent,
        "inventory_summary": enc.inventory_summary,
        "machines_summary": enc.machines_summary,
        "nearby_entities": enc.nearby_entities,
        "env_summary": enc.env_summary,
        "craftable_summary": enc.craftable_summary,
        "context_id": enc.context_id,
        "text_summary": enc.text_summary,
    }


def make_planner_observation(
    raw_snapshot: RawWorldSnapshot,
    tech_state: TechState,
    db: SemanticsDB,
    context_id: str,
) -> Observation:
    """
    Convenience function to create an M1 Observation for the planner LLM.
    """
    payload = encode_for_planner(raw_snapshot, tech_state, db, context_id)
    return Observation(
        json_payload=payload,
        text_summary=payload["text_summary"],
    )


# ---------------------------------------------------------------------------
# Critic encoding
# ---------------------------------------------------------------------------


def _summarize_step(step: TraceStep) -> Dict[str, Any]:
    """
    Summarize a single TraceStep into critic-friendly JSON.

    Exposes:
        - action type/params
        - result success/error
        - meta
        - before/after positions
    """
    world_before_pos = getattr(step.world_before, "position", None)
    world_after_pos = getattr(step.world_after, "position", None)

    return {
        "action": {
            "type": getattr(step.action, "type", None),
            "params": getattr(step.action, "params", {}),
        },
        "result": {
            "success": getattr(step.result, "success", False),
            "error": getattr(step.result, "error", None),
        },
        "meta": step.meta,
        "world_before_pos": world_before_pos,
        "world_after_pos": world_after_pos,
    }


def _make_critic_text_summary(trace: PlanTrace) -> str:
    """
    Brief natural-language summary for the critic context.
    """
    plan_steps = len(trace.plan.get("steps", []))
    exec_steps = len(trace.steps)
    active_tech = getattr(trace.tech_state, "active", None)
    ctx = trace.context_id

    return (
        f"Evaluating execution of plan with {plan_steps} steps "
        f"over {exec_steps} execution steps at tech state '{active_tech}' "
        f"in context '{ctx}'."
    )


def encode_for_critic(trace: PlanTrace) -> Dict[str, Any]:
    """
    Build JSON encoding for the critic model, based on a PlanTrace instance.

    Returns a dict matching CriticEncoding fields.
    """
    tech_dict = _serialize_tech_state(trace.tech_state)
    step_json = [_summarize_step(s) for s in trace.steps]
    text_summary = _make_critic_text_summary(trace)

    enc = CriticEncoding(
        tech_state=tech_dict,
        context_id=trace.context_id,
        plan=trace.plan,
        steps=step_json,
        planner_observation=trace.planner_payload,
        virtue_scores=trace.virtue_scores or {},
        text_summary=text_summary,
    )

    return {
        "tech_state": enc.tech_state,
        "context_id": enc.context_id,
        "plan": enc.plan,
        "steps": enc.steps,
        "planner_observation": enc.planner_observation,
        "virtue_scores": enc.virtue_scores,
        "text_summary": enc.text_summary,
    }


__all__ = [
    "encode_for_planner",
    "make_planner_observation",
    "encode_for_critic",
]

```




---

## **Planner Encoding Architecture**

### **Entry Points**
scss:
```
encode_for_critic(trace: PlanTrace) -> dict

```

### **Trace summarization**

For each `TraceStep`:

- Record action type & params
    
- Record result success/error
    
- Include metadata (skill, timestamps)
    
- Include world_before/world_after position diffs
    
- No large structures, only the essentials
    

### **Critic summary**

Short NL context summarizing:

- Number of plan steps
    
- Number of executed steps
    
- Active tech tier
    
- Context ID
    
- Purpose of evaluation

```
# src/observation/trace_schema.py
"""
Trace schema for critic encoding (M7).

Defines the internal types representing an executed plan:
- TraceStep: one action + world_before/world_after + meta
- PlanTrace: full execution trace for a plan
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from spec.types import WorldState, Action, ActionResult
from semantics.schema import TechState


@dataclass
class TraceStep:
    """
    One step in the execution trace.

    world_before/world_after should be reasonably small WorldState views,
    not raw snapshots.
    """

    world_before: WorldState
    action: Action
    result: ActionResult
    world_after: WorldState
    meta: Dict[str, Any]


@dataclass
class PlanTrace:
    """
    Execution trace for a plan.

    planner_payload is usually the PlannerEncoding dict that was given to the
    planner when this plan was generated.
    """

    plan: Dict[str, Any]
    steps: List[TraceStep]
    tech_state: TechState
    planner_payload: Dict[str, Any]
    context_id: str
    virtue_scores: Dict[str, Any]


__all__ = [
    "TraceStep",
    "PlanTrace",
]

```


```
# src/observation/encoder.py
"""
M7 - Observation Encoder / WorldState Normalization

Responsibilities:

1. WorldState normalization (for M8 and friends):
   - Take raw BotCore snapshots (arbitrary dict-ish shapes)
   - Normalize them into a WorldState that downstream modules (M3 semantics,
     planner, virtues, etc.) can rely on.

   Pipeline:
     BotCore snapshot (raw dict)
       -> build_world_state(raw_snapshot: dict) -> WorldState (normalized)

2. Planner / Critic encodings (for M2):
   - Take RawWorldSnapshot (M6) + TechState + SemanticsDB (M3)
   - Produce compact JSON encodings for the planner and critic LLMs.
"""

from __future__ import annotations

from dataclasses import asdict
from math import sqrt
from typing import Any, Dict, List

# M1: core types
from spec.types import WorldState, Observation

# M6: raw world snapshot types
from bot_core.snapshot import RawWorldSnapshot, RawEntity

# M3: semantics + tech state
from semantics.loader import SemanticsDB
from semantics.schema import TechState
from semantics.crafting import craftable_items

# Local schema types
from .schema import PlannerEncoding, CriticEncoding
from .trace_schema import PlanTrace, TraceStep


# ---------------------------------------------------------------------------
# WorldState normalization (raw BotCore snapshot -> WorldState)
# ---------------------------------------------------------------------------


def _normalize_inventory(raw_inventory: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Normalize raw inventory entries into the InventoryStack shape:

      {
        "item_id": "modid:item",
        "variant": "some.variant.key" or None,
        "count": int >= 0
      }

    Tolerant of various incoming shapes, and skips malformed entries.
    """
    normalized: List[Dict[str, Any]] = []

    for stack in raw_inventory:
        if not isinstance(stack, dict):
            continue

        item_id = (
            stack.get("item_id")
            or stack.get("item")
            or stack.get("id")
        )
        if not isinstance(item_id, str):
            continue

        variant = stack.get("variant")
        count = int(stack.get("count", 0))
        if count <= 0:
            continue

        normalized.append(
            {
                "item_id": item_id,
                "variant": variant,
                "count": count,
            }
        )

    return normalized


def _normalize_machines(raw_machines: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Normalize raw machine descriptors into the MachineEntry shape:

      {
        "id": "gregtech:steam_macerator",
        "type": "gregtech:steam_macerator",  # free-form / legacy type
        "tier": "steam" or "lv" or ...,
        "extra": {...}                       # all other fields preserved
      }

    Tolerant of:
      - "machine_id"
      - "id"
      - "type"
    """
    normalized: List[Dict[str, Any]] = []

    for m in raw_machines:
        if not isinstance(m, dict):
            continue

        machine_id = (
            m.get("machine_id")
            or m.get("id")
            or m.get("type")
        )
        mtype = m.get("type") or machine_id
        tier = m.get("tier")

        extra = {
            k: v
            for k, v in m.items()
            if k not in ("machine_id", "id", "type", "tier")
        }

        normalized.append(
            {
                "id": machine_id,
                "type": mtype,
                "tier": tier,
                "extra": extra,
            }
        )

    return normalized


def build_world_state(raw_snapshot: Dict[str, Any]) -> WorldState:
    """
    Entry point M8 should call with a BotCore snapshot.

    raw_snapshot is whatever BotCore/core emits; we normalize it here
    into the shape M3 / planner / critic expect, matching spec.types.WorldState.
    """
    tick = int(raw_snapshot.get("tick", 0))

    raw_pos = raw_snapshot.get("position", {}) or {}
    position = {
        "x": float(raw_pos.get("x", 0.0)),
        "y": float(raw_pos.get("y", 0.0)),
        "z": float(raw_pos.get("z", 0.0)),
    }

    dimension = str(raw_snapshot.get("dimension", "overworld"))

    raw_inventory = raw_snapshot.get("inventory", []) or []
    raw_nearby_entities = raw_snapshot.get("nearby_entities", []) or []
    raw_blocks_of_interest = raw_snapshot.get("blocks_of_interest", []) or []
    raw_tech_state = raw_snapshot.get("tech_state", {}) or {}
    raw_context = raw_snapshot.get("context", {}) or {}

    norm_inventory = _normalize_inventory(raw_inventory)
    norm_machines = _normalize_machines(raw_context.get("machines", []) or [])

    context: Dict[str, Any] = dict(raw_context)
    context["machines"] = norm_machines

    return WorldState(
        tick=tick,
        position=position,
        dimension=dimension,
        inventory=norm_inventory,
        nearby_entities=list(raw_nearby_entities),
        blocks_of_interest=list(raw_blocks_of_interest),
        tech_state=dict(raw_tech_state),
        context=context,
    )


# ---------------------------------------------------------------------------
# Tech state serialization (for encodings)
# ---------------------------------------------------------------------------


def _serialize_tech_state(tech_state: TechState) -> Dict[str, Any]:
    """
    Turn TechState dataclass into JSON-like dict.

    Assumes TechState has:
        - active: str
        - unlocked: Collection[str]
        - evidence: optional and omitted/compressed here
    """
    try:
        data = asdict(tech_state)
    except TypeError:
        data = {
            "active": getattr(tech_state, "active", None),
            "unlocked": list(getattr(tech_state, "unlocked", [])),
        }

    return {
        "active": data.get("active"),
        "unlocked": list(data.get("unlocked", [])),
    }


# ---------------------------------------------------------------------------
# Planner helpers (RawWorldSnapshot -> PlannerEncoding payload)
# ---------------------------------------------------------------------------


def _summarize_agent(raw: RawWorldSnapshot) -> Dict[str, Any]:
    """Summarize agent position and basic status."""
    pos = raw.player_pos
    return {
        "position": pos,
        "dimension": raw.dimension,
        "on_ground": raw.on_ground,
        "yaw": raw.player_yaw,
        "pitch": raw.player_pitch,
    }


def _summarize_inventory(
    raw: RawWorldSnapshot,
    db: SemanticsDB,
) -> Dict[str, Any]:
    """
    Compress inventory into counts by:
        - item
        - category (from SemanticsDB)
        - material (from SemanticsDB)
    """
    summary: Dict[str, Any] = {
        "by_item": {},
        "by_category": {},
        "by_material": {},
    }

    for stack in raw.inventory:
        item_id = stack.get("item_id")
        variant = stack.get("variant")
        count = int(stack.get("count", 0))

        if not item_id or count <= 0:
            continue

        try:
            info = db.get_item_info(item_id, variant)
        except Exception:
            info = type("TmpInfo", (), {"category": "unknown", "material": "unknown"})()

        key_item = f"{item_id}:{variant}" if variant else item_id
        summary["by_item"].setdefault(key_item, 0)
        summary["by_item"][key_item] += count

        cat = getattr(info, "category", "unknown") or "unknown"
        summary["by_category"].setdefault(cat, 0)
        summary["by_category"][cat] += count

        mat = getattr(info, "material", "unknown") or "unknown"
        summary["by_material"].setdefault(mat, 0)
        summary["by_material"][mat] += count

    return summary


def _summarize_machines(raw: RawWorldSnapshot) -> Dict[str, Any]:
    """
    Summarize machines & power sources from raw.context["machines"].
    """
    machines: List[Dict[str, Any]] = raw.context.get("machines", []) or []

    summary: Dict[str, Any] = {
        "count": len(machines),
        "by_type": {},
        "by_tier": {},
        "has_steam": False,
        "has_lv": False,
        "has_mv": False,
    }

    for m in machines:
        mtype = m.get("type", "unknown")
        tier = m.get("tier", "unknown")

        summary["by_type"].setdefault(mtype, 0)
        summary["by_type"][mtype] += 1

        summary["by_tier"].setdefault(tier, 0)
        summary["by_tier"][tier] += 1

        if "steam" in (mtype or "").lower() or tier == "steam":
            summary["has_steam"] = True
        if tier == "lv":
            summary["has_lv"] = True
        if tier == "mv":
            summary["has_mv"] = True

    return summary


def _summarize_nearby_entities(
    raw: RawWorldSnapshot,
    radius: float = 16.0,
    max_entities: int = 16,
) -> List[Dict[str, Any]]:
    """
    Filter entities around player within a limited radius, sorted by distance.
    """
    px = raw.player_pos.get("x", 0.0)
    py = raw.player_pos.get("y", 0.0)
    pz = raw.player_pos.get("z", 0.0)

    result: List[Dict[str, Any]] = []

    for e in raw.entities:
        if not isinstance(e, RawEntity):
            ex = getattr(e, "x", e.get("x", 0.0))
            ey = getattr(e, "y", e.get("y", 0.0))
            ez = getattr(e, "z", e.get("z", 0.0))
            etype = getattr(e, "type", e.get("type", "unknown"))
            edata = getattr(e, "data", e.get("data", {}))
        else:
            ex, ey, ez = e.x, e.y, e.z
            etype = e.type
            edata = e.data or {}

        dx = ex - px
        dy = ey - py
        dz = ez - pz
        dist = sqrt(dx * dx + dy * dy + dz * dz)
        if dist > radius:
            continue

        result.append(
            {
                "type": etype,
                "x": ex,
                "y": ey,
                "z": ez,
                "distance": dist,
                "is_hostile": bool(edata.get("hostile", False)),
                "is_item": etype == "item",
            }
        )

    result.sort(key=lambda ent: ent["distance"])
    return result[:max_entities]


def _summarize_env(raw: RawWorldSnapshot) -> Dict[str, Any]:
    """Misc environmental summary, including warnings/notes from context."""
    return {
        "tick": raw.tick,
        "dimension": raw.dimension,
        "warnings": raw.context.get("warnings", []),
        "notes": raw.context.get("notes", ""),
    }


def _summarize_craftables(
    raw: RawWorldSnapshot,
    db: SemanticsDB,
    max_outputs: int = 10,
) -> Dict[str, Any]:
    """
    Summarize what we can craft *right now* into a small list.
    """
    world = WorldState(
        tick=raw.tick,
        position=raw.player_pos,
        dimension=raw.dimension,
        inventory=raw.inventory,
        nearby_entities=[],
        blocks_of_interest=[],
        tech_state={},
        context=raw.context,
    )

    options = craftable_items(world, db)
    summary: Dict[str, Any] = {"top_outputs": []}

    for opt in options[:max_outputs]:
        summary["top_outputs"].append(
            {
                "output_item": getattr(opt, "output_item", None),
                "output_count": getattr(opt, "output_count", None),
                "recipe_id": getattr(opt, "recipe_id", None),
                "notes": getattr(opt, "notes", ""),
            }
        )

    return summary


def _make_text_summary(
    raw: RawWorldSnapshot,
    tech_state: TechState,
    machines_summary: Dict[str, Any],
    inventory_summary: Dict[str, Any],
) -> str:
    """
    Build a compact one-paragraph summary as prompt context.
    """
    dim = raw.dimension
    active_tech = getattr(tech_state, "active", None)
    inv_cats = inventory_summary.get("by_category", {})
    machines_count = machines_summary.get("count", 0)

    top_cats = sorted(inv_cats.items(), key=lambda kv: kv[1], reverse=True)[:3]
    if top_cats:
        top_cats_str = ", ".join(f"{cat} x{amt}" for cat, amt in top_cats)
    else:
        top_cats_str = "no notable categories"

    return (
        f"You are in dimension '{dim}' at position {raw.player_pos}, "
        f"current tech state is '{active_tech}'. "
        f"There are {machines_count} known machines. "
        f"Key inventory categories: {top_cats_str}."
    )


def encode_for_planner(
    raw_snapshot: RawWorldSnapshot,
    tech_state: TechState,
    db: SemanticsDB,
    context_id: str,
) -> Dict[str, Any]:
    """
    Build a JSON-serializable dict suitable as Observation.json_payload
    for the planner. Returns a dict matching PlannerEncoding fields.
    """
    agent = _summarize_agent(raw_snapshot)
    inv_summary = _summarize_inventory(raw_snapshot, db)
    machines_summary = _summarize_machines(raw_snapshot)
    entities = _summarize_nearby_entities(raw_snapshot)
    env_summary = _summarize_env(raw_snapshot)
    craftable_summary = _summarize_craftables(raw_snapshot, db)
    tech_dict = _serialize_tech_state(tech_state)

    text_summary = _make_text_summary(
        raw_snapshot,
        tech_state,
        machines_summary,
        inv_summary,
    )

    enc = PlannerEncoding(
        tech_state=tech_dict,
        agent=agent,
        inventory_summary=inv_summary,
        machines_summary=machines_summary,
        nearby_entities=entities,
        env_summary=env_summary,
        craftable_summary=craftable_summary,
        context_id=context_id,
        text_summary=text_summary,
    )

    return {
        "tech_state": enc.tech_state,
        "agent": enc.agent,
        "inventory_summary": enc.inventory_summary,
        "machines_summary": enc.machines_summary,
        "nearby_entities": enc.nearby_entities,
        "env_summary": enc.env_summary,
        "craftable_summary": enc.craftable_summary,
        "context_id": enc.context_id,
        "text_summary": enc.text_summary,
    }


def make_planner_observation(
    raw_snapshot: RawWorldSnapshot,
    tech_state: TechState,
    db: SemanticsDB,
    context_id: str,
) -> Observation:
    """
    Convenience function to create an M1 Observation for the planner LLM.
    """
    payload = encode_for_planner(raw_snapshot, tech_state, db, context_id)
    return Observation(
        json_payload=payload,
        text_summary=payload["text_summary"],
    )


# ---------------------------------------------------------------------------
# Critic encoding (PlanTrace -> CriticEncoding dict)
# ---------------------------------------------------------------------------


def _summarize_step(step: TraceStep) -> Dict[str, Any]:
    """
    Summarize a single TraceStep into critic-friendly JSON.

    For each step:
      - record action type & params
      - record result success/error
      - include metadata (skill, timestamps, etc.)
      - include world_before / world_after positions
      - avoid large nested structures
    """
    world_before_pos = getattr(step.world_before, "position", None)
    world_after_pos = getattr(step.world_after, "position", None)

    return {
        "action": {
            "type": getattr(step.action, "type", None),
            "params": getattr(step.action, "params", {}),
        },
        "result": {
            "success": getattr(step.result, "success", False),
            "error": getattr(step.result, "error", None),
        },
        "meta": step.meta,
        "world_before_pos": world_before_pos,
        "world_after_pos": world_after_pos,
    }


def _make_critic_text_summary(trace: PlanTrace) -> str:
    """
    Build a short NL context summarizing:
      - number of plan steps
      - number of executed steps
      - active tech tier
      - context ID
      - purpose of evaluation
    """
    plan_steps = len(trace.plan.get("steps", []))
    exec_steps = len(trace.steps)
    active_tech = getattr(trace.tech_state, "active", None)
    ctx = trace.context_id

    return (
        f"Evaluating execution of plan with {plan_steps} steps "
        f"over {exec_steps} execution steps at tech state '{active_tech}' "
        f"in context '{ctx}'."
    )


def encode_for_critic(trace: PlanTrace) -> Dict[str, Any]:
    """
    Entry point for critic encoding.

    encode_for_critic(trace: PlanTrace) -> dict

    Produces a JSON-serializable dict matching CriticEncoding:
      - tech_state
      - context_id
      - plan
      - steps
      - planner_observation
      - virtue_scores
      - text_summary
    """
    tech_dict = _serialize_tech_state(trace.tech_state)
    step_json = [_summarize_step(s) for s in trace.steps]
    text_summary = _make_critic_text_summary(trace)

    enc = CriticEncoding(
        tech_state=tech_dict,
        context_id=trace.context_id,
        plan=trace.plan,
        steps=step_json,
        planner_observation=trace.planner_payload,
        virtue_scores=trace.virtue_scores or {},
        text_summary=text_summary,
    )

    return {
        "tech_state": enc.tech_state,
        "context_id": enc.context_id,
        "plan": enc.plan,
        "steps": enc.steps,
        "planner_observation": enc.planner_observation,
        "virtue_scores": enc.virtue_scores,
        "text_summary": enc.text_summary,
    }


__all__ = [
    "build_world_state",
    "encode_for_planner",
    "make_planner_observation",
    "encode_for_critic",
]

```


```
# tests/test_observation_critic_encoding.py
"""
Basic tests for encode_for_critic in M7.

These are sanity checks:
  - structure of returned dict
  - basic propagation of fields
  - text summary presence
"""

from observation.encoder import encode_for_critic
from observation.trace_schema import TraceStep, PlanTrace
from semantics.schema import TechState
from spec.types import WorldState, Action, ActionResult


def _make_world_state() -> WorldState:
    return WorldState(
        tick=1,
        position={"x": 0, "y": 64, "z": 0},
        dimension="overworld",
        inventory=[],
        nearby_entities=[],
        blocks_of_interest=[],
        tech_state={},
        context={},
    )


def test_encode_for_critic_basic():
    world = _make_world_state()

    action = Action(
        type="move_to",
        params={"x": 1, "y": 64, "z": 0},
    )
    result = ActionResult(
        success=True,
        error=None,
        details={},
    )

    step = TraceStep(
        world_before=world,
        action=action,
        result=result,
        world_after=world,
        meta={"skill": "test_skill", "timestamp": 123},
    )

    tech = TechState(
        unlocked=["stone_age"],
        active="stone_age",
        evidence={},
    )

    trace = PlanTrace(
        plan={"steps": [{"skill": "test_skill", "params": {}}]},
        steps=[step],
        tech_state=tech,
        planner_payload={"some": "payload"},
        context_id="lv_early_factory",
        virtue_scores={"overall": 0.5},
    )

    enc = encode_for_critic(trace)

    # Core structural checks
    assert enc["tech_state"]["active"] == "stone_age"
    assert enc["context_id"] == "lv_early_factory"
    assert "plan" in enc
    assert "steps" in enc
    assert "planner_observation" in enc
    assert "virtue_scores" in enc
    assert isinstance(enc["text_summary"], str)
    assert len(enc["steps"]) == 1

    step0 = enc["steps"][0]
    assert step0["action"]["type"] == "move_to"
    assert step0["result"]["success"] is True
    assert step0["meta"]["skill"] == "test_skill"

```


---

## **System Position**
csharp:
```
M6 observe()
 → RawWorldSnapshot
M3 semantics
 → TechState, SemanticsDB
M7 encode_for_planner / encode_for_critic
 → PlannerModel, CriticModel (M2)

```

```
# src/observation/pipeline.py
"""
M7 - System Position / Integration Helpers

This module encodes the *position* of M7 in the overall system:

    M6: BotCore.observe()          -> RawWorldSnapshot
    M3: Semantics (SemanticsDB, TechState)
    M7: encode_for_planner / encode_for_critic
    M2: PlannerModel / CriticModel

It provides thin orchestration helpers that:
  - Pull a RawWorldSnapshot from BotCore (M6)
  - Combine it with SemanticsDB + TechState (M3)
  - Use M7 encoders to build planner/critic payloads
  - Call abstract PlannerModel / CriticModel interfaces (M2)

This keeps wiring logic out of M8 while making the dataflow explicit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Dict, Any

from semantics.loader import SemanticsDB
from semantics.schema import TechState
from spec.types import Observation

from .encoder import (
    encode_for_planner,
    make_planner_observation,
    encode_for_critic,
)
from .trace_schema import PlanTrace


# ---------------------------------------------------------------------------
# Abstract interfaces for neighboring modules
# ---------------------------------------------------------------------------


class BotCore(Protocol):
    """
    Minimal BotCore interface for M7.

    M6 should provide a concrete implementation with:
        def observe(self) -> RawWorldSnapshot: ...
    """

    def observe(self) -> Any:  # RawWorldSnapshot, but we avoid hard import to keep coupling minimal
        ...


class PlannerModel(Protocol):
    """
    Minimal planner interface in M2.

    The planner consumes an Observation and returns a plan dict
    (the exact structure lives in M2).
    """

    def plan(self, observation: Observation) -> Dict[str, Any]:
        ...


class CriticModel(Protocol):
    """
    Minimal critic interface in M2.

    The critic consumes a critic payload (encoded PlanTrace) and returns
    an evaluation / scoring dict.
    """

    def evaluate(self, critic_payload: Dict[str, Any]) -> Dict[str, Any]:
        ...


# ---------------------------------------------------------------------------
# Coordinator types
# ---------------------------------------------------------------------------


@dataclass
class PlannerContext:
    """
    Bundle of dependencies needed to produce planner observations and plans.

    This is mainly for ergonomics, so M8-style loops can just hold one
    PlannerContext and call step_planner(...) repeatedly.
    """

    bot_core: BotCore
    semantics_db: SemanticsDB
    planner_model: PlannerModel


@dataclass
class CriticContext:
    """
    Bundle of dependencies for critic evaluation.
    """

    critic_model: CriticModel


# ---------------------------------------------------------------------------
# Planner path: M6 -> M3 -> M7 -> M2
# ---------------------------------------------------------------------------


def build_planner_observation(
    bot_core: BotCore,
    semantics_db: SemanticsDB,
    tech_state: TechState,
    context_id: str,
) -> Observation:
    """
    High-level helper:

        M6 BotCore.observe()          -> RawWorldSnapshot
        + M3 TechState + SemanticsDB  -> M7 encode_for_planner
        -> Observation (M1)

    This is the recommended single call for "give me something I can hand
    to the planner LLM."
    """
    raw_snapshot = bot_core.observe()
    return make_planner_observation(
        raw_snapshot=raw_snapshot,
        tech_state=tech_state,
        db=semantics_db,
        context_id=context_id,
    )


def planner_step(
    ctx: PlannerContext,
    tech_state: TechState,
    context_id: str,
) -> Dict[str, Any]:
    """
    One full planner step:

      1. M6: call bot_core.observe() -> RawWorldSnapshot
      2. M7: build Observation (encode_for_planner + make_planner_observation)
      3. M2: planner_model.plan(observation) -> plan dict

    Returns:
        plan: arbitrary dict defined by M2's PlannerModel.
    """
    observation = build_planner_observation(
        bot_core=ctx.bot_core,
        semantics_db=ctx.semantics_db,
        tech_state=tech_state,
        context_id=context_id,
    )
    plan = ctx.planner_model.plan(observation)
    return plan


# ---------------------------------------------------------------------------
# Critic path: M8 PlanTrace -> M7 -> M2
# ---------------------------------------------------------------------------


def critic_step(
    ctx: CriticContext,
    trace: PlanTrace,
) -> Dict[str, Any]:
    """
    One full critic step:

      1. M8: build PlanTrace (plan + steps + tech_state + planner_payload)
      2. M7: encode_for_critic(trace) -> critic_payload dict
      3. M2: critic_model.evaluate(critic_payload) -> evaluation dict

    Returns:
        evaluation: arbitrary dict defined by M2's CriticModel.
    """
    critic_payload = encode_for_critic(trace)
    evaluation = ctx.critic_model.evaluate(critic_payload)
    return evaluation


__all__ = [
    "BotCore",
    "PlannerModel",
    "CriticModel",
    "PlannerContext",
    "CriticContext",
    "build_planner_observation",
    "planner_step",
    "critic_step",
]

```







## **File Structure**

_(This section mirrors your repo layout and ensures alignment with implementation paths.)_
```
src/
├── observation/
│   ├── encoder.py             # main planner + critic encoders
│   ├── schema.py              # PlannerEncoding, CriticEncoding
│   ├── trace_schema.py        # TraceStep, PlanTrace
│   └── __init__.py

```
Test files:
```
tests/
├── test_observation_planner_encoding.py
└── test_observation_critic_encoding.py

```

## **Testing & Simulation**

### **Planner tests**

- Minimal RawWorldSnapshot
    
- Minimal SemanticsDB mock
    
- Check fields exist and format is correct
    
- Validate text summary is nonempty string
    

### **Critic tests**

- Synthetic PlanTrace
    
- Check encoding structure
    
- Confirm steps serialized correctly
    

### **Performance tests**

- Stress with:
    
    - 100+ entities
        
    - 100+ inventory stacks
        
- Ensure:
    
    - Runtime is small and stable
        
    - Output size remains controlled
        

```
# src/observation/testing.py
"""
Testing helpers for M7 - observation_encoding.

Provides:
  - minimal RawWorldSnapshot factories
  - heavy RawWorldSnapshot factories (for perf / scale tests)
  - dummy SemanticsDB implementation
"""

from __future__ import annotations

from typing import Any, Dict, List

from bot_core.snapshot import RawWorldSnapshot, RawEntity
from semantics.loader import SemanticsDB
from semantics.schema import TechState


class DummySemanticsDB(SemanticsDB):
    """
    Minimal SemanticsDB stub for tests.

    Only implements get_item_info in a way that matches M7 encoder expectations:
      - returns an object with .category and .material attributes.
    """

    def get_item_info(self, item_id: str, variant: Any = None) -> Any:
        # Cheap heuristic: treat logs/wood as "raw_material"/"wood", everything else generic.
        category = "raw_material" if "log" in item_id or "wood" in item_id else "misc"
        material = "wood" if "log" in item_id or "wood" in item_id else "unknown"
        return type("Info", (), {"category": category, "material": material})()


def make_minimal_tech_state() -> TechState:
    """
    Tiny TechState instance suitable for basic tests.
    """
    return TechState(
        unlocked=["stone_age"],
        active="stone_age",
        evidence={},
    )


def make_minimal_snapshot() -> RawWorldSnapshot:
    """
    Minimal RawWorldSnapshot used in planner encoding tests.

    Keeps everything tiny but structurally valid.
    """
    return RawWorldSnapshot(
        tick=123,
        dimension="overworld",
        player_pos={"x": 0.0, "y": 64.0, "z": 0.0},
        player_yaw=0.0,
        player_pitch=0.0,
        on_ground=True,
        chunks={},
        entities=[
            RawEntity(
                entity_id=1,
                type="item",
                x=2.0,
                y=64.0,
                z=2.0,
                data={"hostile": False},
            )
        ],
        inventory=[
            {"item_id": "minecraft:log", "variant": None, "count": 16},
        ],
        context={"machines": []},
    )


def make_heavy_snapshot(
    entity_count: int = 120,
    inventory_count: int = 130,
) -> RawWorldSnapshot:
    """
    Heavier RawWorldSnapshot for perf / scale tests.

    Builds:
      - ~entity_count entities in a sphere around player
      - ~inventory_count inventory stacks with simple variation
    """
    entities: List[RawEntity] = []
    for i in range(entity_count):
        # Spread entities in a ring-ish pattern around the origin.
        x = float(i % 32)
        z = float((i // 32) % 32)
        y = 64.0
        entities.append(
            RawEntity(
                entity_id=i + 1,
                type="item" if i % 3 == 0 else "mob",
                x=x,
                y=y,
                z=z,
                data={"hostile": i % 5 == 0},
            )
        )

    inventory: List[Dict[str, Any]] = []
    for i in range(inventory_count):
        item_id = "minecraft:log" if i % 2 == 0 else "minecraft:cobblestone"
        inventory.append(
            {
                "item_id": item_id,
                "variant": None,
                "count": 32,
            }
        )

    machines: List[Dict[str, Any]] = []
    for i in range(8):
        machines.append(
            {
                "type": "gregtech:steam_bronze_boiler" if i < 4 else "gregtech:lv_macerator",
                "tier": "steam" if i < 4 else "lv",
            }
        )

    return RawWorldSnapshot(
        tick=9999,
        dimension="overworld",
        player_pos={"x": 0.0, "y": 64.0, "z": 0.0},
        player_yaw=90.0,
        player_pitch=0.0,
        on_ground=True,
        chunks={},
        entities=entities,
        inventory=inventory,
        context={"machines": machines},
    )


__all__ = [
    "DummySemanticsDB",
    "make_minimal_tech_state",
    "make_minimal_snapshot",
    "make_heavy_snapshot",
]

```

```
# tests/test_observation_planner_encoding.py
"""
Planner tests for M7 - observation_encoding.

Covers:
  - minimal RawWorldSnapshot
  - DummySemanticsDB
  - required fields in planner encoding
  - non-empty text_summary
"""

from observation.encoder import encode_for_planner, make_planner_observation
from observation.testing import (
    DummySemanticsDB,
    make_minimal_snapshot,
    make_minimal_tech_state,
)
from semantics.schema import TechState
from spec.types import Observation


def test_encode_for_planner_basic_structure():
    raw = make_minimal_snapshot()
    db = DummySemanticsDB()
    tech = make_minimal_tech_state()

    enc = encode_for_planner(
        raw_snapshot=raw,
        tech_state=tech,
        db=db,
        context_id="lv_early_factory",
    )

    # Field existence
    assert "tech_state" in enc
    assert "agent" in enc
    assert "inventory_summary" in enc
    assert "machines_summary" in enc
    assert "nearby_entities" in enc
    assert "env_summary" in enc
    assert "craftable_summary" in enc
    assert "context_id" in enc
    assert "text_summary" in enc

    # Basic correctness
    assert enc["tech_state"]["active"] == "stone_age"
    assert enc["agent"]["dimension"] == "overworld"
    assert isinstance(enc["nearby_entities"], list)
    assert isinstance(enc["inventory_summary"], dict)
    assert isinstance(enc["text_summary"], str)
    assert len(enc["text_summary"]) > 0


def test_make_planner_observation_returns_observation():
    raw = make_minimal_snapshot()
    db = DummySemanticsDB()
    tech = make_minimal_tech_state()

    obs = make_planner_observation(
        raw_snapshot=raw,
        tech_state=tech,
        db=db,
        context_id="lv_early_factory",
    )

    assert isinstance(obs, Observation)
    assert isinstance(obs.json_payload, dict)
    assert "text_summary" in obs.json_payload
    assert obs.text_summary == obs.json_payload["text_summary"]

```

```
# tests/test_observation_perf.py
"""
Performance / scale sanity tests for M7 - observation_encoding.

Goal is not microbenchmarking, but:
  - ensure encode_for_planner handles larger snapshots without exploding
  - confirm caps on entities/craftables keep output size controlled
"""

from observation.encoder import encode_for_planner
from observation.testing import (
    DummySemanticsDB,
    make_heavy_snapshot,
    make_minimal_tech_state,
)


def test_encode_for_planner_heavy_snapshot_caps_and_structure():
    """
    Use a heavier snapshot with many entities + inventory stacks and check:

      - function completes without error
      - entity list is capped (<= 16)
      - craftables list is capped (<= 10)
      - encoding is still reasonably sized
    """
    raw = make_heavy_snapshot(entity_count=150, inventory_count=150)
    db = DummySemanticsDB()
    tech = make_minimal_tech_state()

    enc = encode_for_planner(
        raw_snapshot=raw,
        tech_state=tech,
        db=db,
        context_id="lv_early_factory",
    )

    # Entity list cap
    assert "nearby_entities" in enc
    assert len(enc["nearby_entities"]) <= 16

    # Craftables cap (top_outputs from semantics.crafting)
    craft = enc.get("craftable_summary", {})
    top_outputs = craft.get("top_outputs", [])
    assert len(top_outputs) <= 10

    # Encoding size sanity: avoid mega-blobs
    # Rough upper bound on top-level keys and list lengths.
    assert isinstance(enc["inventory_summary"], dict)
    assert len(enc["inventory_summary"].get("by_item", {})) <= 200

    # Text summary should still exist and be short-ish.
    summary = enc.get("text_summary", "")
    assert isinstance(summary, str)
    assert len(summary) > 0
    assert len(summary) < 1024  # arbitrary "not insane" limit

```





---

## **Implementation Checklist**

You can call M7 _complete_ when:

### **Schema stability**

- PlannerEncoding and CriticEncoding are frozen
    
- Caps enforced (entities, craftables)
    

### **Planner integration**

- `make_planner_observation` returns valid M1 Observation
    
- PlannerModel accepts M7 payload without schema hacks
    

### **Semantics integration**

- Inventory summaries use categories/materials from M3
    
- TechState serialization round-trips cleanly
    

### **Trace pipeline**

- Critic receives:
    
    - full plan
        
    - execution steps
        
    - tech state
        
    - virtue scores
        

### **Tests pass**

- Planner tests
    
- Critic tests
    
- Optional perf tests
    

When all this is satisfied, M7 becomes a stable perceptual API for the rest of the agent.

---

## **Next Steps**

With M7 operational, the next modules (M8+):

- Add planning loop
    
- Add self-evaluation retry logic
    
- Add memory (experience traces → experience)
    
- Strengthen world model (lightweight predictive)