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

