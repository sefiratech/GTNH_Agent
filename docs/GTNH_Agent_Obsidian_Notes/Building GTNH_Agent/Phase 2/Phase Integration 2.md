## 1. Architecture Overview: M6 ↔ M7

**Modules in play:**

- **M6 – observation_capture / BotCore**
    
    - Owns Minecraft IPC / socket / whatever to the Forge side
        
    - Exposes `BotCore.observe() -> RawWorldSnapshot`
        
    - Deals with packets, chunk decoding, entity lists, raw inventory, etc.
        
- **M7 – observation_encoding**
    
    - Owns _interpretation_ of that raw data for LLMs
        
    - Key functions:
        
        - `build_world_state(raw_snapshot: dict) -> WorldState` (for general use)
            
        - `encode_for_planner(raw_snapshot, tech_state, db, context_id) -> dict`
            
        - `make_planner_observation(...) -> Observation` (M1)
            
        - `encode_for_critic(trace: PlanTrace) -> dict`
            
    - Glue functions in `observation.pipeline`:
        
        - `build_planner_observation(...) -> Observation`
            
        - `planner_step(ctx: PlannerContext, tech_state, context_id) -> plan`
            
        - `critic_step(ctx: CriticContext, trace: PlanTrace) -> evaluation`
            

**Data flow (Phase 2 Integration focus):**

`Forge / Minecraft   ↓   (events, packets) M6 BotCore.observe()   ↓   RawWorldSnapshot M7 encoder.encode_for_planner()   ↓   PlannerEncoding dict M7 make_planner_observation()   ↓   Observation (M1) M2 PlannerModel.plan()   ↓   Plan dict M8 (later): executes plan, builds PlanTrace   ↓ M7 encode_for_critic()   ↓   CriticEncoding dict M2 CriticModel.evaluate()`

**Event handling semantics:**

- “Event” here is basically **“tick where we decide to think”**:
    
    - M6: “new snapshot available”
        
    - M7: “encode snapshot + tech state + semantics into planner input”
        

No fancy async; just a clean deterministic pipeline per tick.

---

## 2. Function Interactions

### Observation capture (M6 → M7)

1. `bot_core.observe()`
    
    - Runs whatever IO / IPC with Forge
        
    - Returns a `RawWorldSnapshot` object with:
        
        - `tick`, `dimension`, `player_pos`, `entities`, `inventory`, `context`, etc.
            
2. M7:
    
    - `make_planner_observation(raw_snapshot, tech_state, db, context_id)`
        
    - Internally:
        
        - `_summarize_agent(raw_snapshot)`
            
        - `_summarize_inventory(raw_snapshot, db)`
            
        - `_summarize_machines(raw_snapshot)`
            
        - `_summarize_nearby_entities(raw_snapshot)`
            
        - `_summarize_env(raw_snapshot)`
            
        - `_summarize_craftables(raw_snapshot, db)`
            
        - `_serialize_tech_state(tech_state)`
            
        - `_make_text_summary(...)`
            
3. Result is an `Observation` that M2’s planner consumes.
    

### Critic path (M8 → M7)

1. M8 builds `PlanTrace`:
    
    - `plan: dict`
        
    - `steps: list[TraceStep]`
        
    - `tech_state: TechState`
        
    - `planner_payload: dict` (original planner encoding)
        
    - `context_id`, `virtue_scores`
        
2. M7:
    
    - `encode_for_critic(trace: PlanTrace) -> dict`
        
    - `_summarize_step(step: TraceStep)` per step
        
    - `_make_critic_text_summary(trace)`
        
3. M2’s critic model receives clean, compressed JSON instead of raw chaos.
    

---

## 3. Unified Integration Code (M6 + M7)

Here’s a coherent “Phase 2 Integration” runtime slice that wires everything together.  
File name suggestion: **`src/agent/runtime_m6_m7.py`**

Every line is commented so future-you doesn’t have to reverse-engineer it at 3 AM.
```
# src/agent/runtime_m6_m7.py
"""
Minimal integration of M6 (observation capture) and M7 (observation encoding).

This file shows:
  - How BotCore (M6) provides RawWorldSnapshot
  - How M7 turns that into planner / critic-ready encodings
  - How M2 planner/critic models would be invoked from this layer
"""

from __future__ import annotations  # enable postponed type hints for forward refs

from dataclasses import dataclass  # used to group runtime dependencies
from typing import Dict, Any       # basic typing for JSON-like dicts

# M6: BotCore interface lives logically in observation.pipeline as a Protocol
from observation.pipeline import (   # reuse the shared interfaces / helpers
    BotCore,                         # Protocol describing M6's observe()
    PlannerModel,                    # Protocol for planner (M2)
    CriticModel,                     # Protocol for critic (M2)
    PlannerContext,                  # Bundles bot_core + semantics + planner
    CriticContext,                   # Bundles critic only
    planner_step,                    # M6 -> M7 -> M2 planner call
    critic_step,                     # M8 -> M7 -> M2 critic call
)

# M7: For direct calls if needed (not strictly required if pipeline is enough)
from observation.encoder import (
    build_world_state,               # raw dict -> WorldState (if you use it)
    encode_for_planner,              # RawWorldSnapshot -> planner dict
    make_planner_observation,        # RawWorldSnapshot -> Observation
    encode_for_critic,               # PlanTrace -> critic dict
)

# M7: types for traces
from observation.trace_schema import PlanTrace  # critic path trace type

# M3: semantics + tech-state
from semantics.loader import SemanticsDB       # semantic database interface
from semantics.schema import TechState         # tech progression state

# M1: Observation type
from spec.types import Observation             # planner / LLM observation type


@dataclass
class AgentRuntimeConfig:
    """
    Configuration / static context for a running agent.

    This includes:
      - context_id: virtue / scenario id from M4 config
      - initial_tech_state: starting inferred tech state from M3
    """
    context_id: str                  # identifies scenario / virtue context
    initial_tech_state: TechState    # snapshot of current tech progression


@dataclass
class AgentRuntime:
    """
    Integrates M6 and M7 around the planner and critic models.

    This is the "runtime shell" that M8 will likely call into.
    """

    bot_core: BotCore                # M6: provides observe() -> RawWorldSnapshot
    semantics_db: SemanticsDB        # M3: semantic info + item categories
    planner_model: PlannerModel      # M2: planner LLM wrapper
    critic_model: CriticModel        # M2: critic LLM wrapper
    config: AgentRuntimeConfig       # static config (context_id, initial tech)

    def __post_init__(self) -> None:
        """
        Build planner and critic contexts after initialization.

        This keeps M7's pipeline helpers as the single source of truth
        for how these pieces are wired.
        """
        # Build a M7 planner context with M6 + M3 + M2 planner
        self._planner_ctx = PlannerContext(
            bot_core=self.bot_core,
            semantics_db=self.semantics_db,
            planner_model=self.planner_model,
        )
        # Build a M7 critic context with M2 critic
        self._critic_ctx = CriticContext(
            critic_model=self.critic_model,
        )
        # Track current tech_state; may evolve over time as M3 updates it
        self._tech_state = self.config.initial_tech_state

    # ------------------------------------------------------------------ #
    # Planner path: M6 → M7 → M2                                         #
    # ------------------------------------------------------------------ #

    def planner_tick(self) -> Dict[str, Any]:
        """
        Perform one planner cycle:

          1. M6: bot_core.observe() -> RawWorldSnapshot
          2. M7: planner_step(...) -> Observation -> PlannerEncoding payload
          3. M2: planner_model.plan(observation) -> plan dict

        Returns:
            plan: arbitrary JSON-like dict defined by M2.
        """
        # Use the shared pipeline helper so we don't duplicate wiring logic
        plan = planner_step(
            ctx=self._planner_ctx,
            tech_state=self._tech_state,
            context_id=self.config.context_id,
        )
        # Future: you might update tech_state here based on the plan if needed
        return plan

    def get_latest_planner_observation(self) -> Observation:
        """
        If you want to inspect the planner observation without actually
        calling the planner model, you can call this.

        It only uses M6+M7, not the planner LLM.
        """
        # Grab a snapshot directly from M6
        raw_snapshot = self.bot_core.observe()
        # Build the Observation via M7
        obs = make_planner_observation(
            raw_snapshot=raw_snapshot,
            tech_state=self._tech_state,
            db=self.semantics_db,
            context_id=self.config.context_id,
        )
        return obs

    # ------------------------------------------------------------------ #
    # Critic path: M8 → M7 → M2                                          #
    # ------------------------------------------------------------------ #

    def evaluate_trace(self, trace: PlanTrace) -> Dict[str, Any]:
        """
        Evaluate a completed PlanTrace using CriticModel via M7:

          4. M8: builds PlanTrace with plan + steps + tech_state + payload
          5. M7: critic_step(...) -> CriticEncoding payload dict
          6. M2: critic_model.evaluate(payload) -> evaluation dict

        Returns:
            evaluation: JSON-like dict with score / issues / advice.
        """
        # Use pipeline helper to ensure encoding is consistent
        evaluation = critic_step(
            ctx=self._critic_ctx,
            trace=trace,
        )
        return evaluation

    # ------------------------------------------------------------------ #
    # Optional: world-state utilities                                   #
    # ------------------------------------------------------------------ #

    def get_normalized_world_state(self) -> Any:
        """
        Demonstrate how to use M7's build_world_state() with M6 output.

        This is mostly for debugging, logging, or non-LLM subsystems.
        """
        # Get raw snapshot from BotCore
        raw_snapshot = self.bot_core.observe()
        # Raw snapshot is usually a custom object; convert to dict shape
        raw_dict = {
            "tick": raw_snapshot.tick,
            "position": raw_snapshot.player_pos,
            "dimension": raw_snapshot.dimension,
            "inventory": raw_snapshot.inventory,
            "nearby_entities": raw_snapshot.entities,
            "blocks_of_interest": raw_snapshot.context.get(
                "blocks_of_interest", []
            ),
            "tech_state": {},  # WorldState tech_state is often filled elsewhere
            "context": raw_snapshot.context,
        }
        # Normalize via M7's helper
        world_state = build_world_state(raw_dict)
        return world_state

    # ------------------------------------------------------------------ #
    # Hooks for M8 / outer loop                                          #
    # ------------------------------------------------------------------ #

    def update_tech_state(self, new_state: TechState) -> None:
        """
        Allow outer systems (e.g., semantics / progression) to update TechState.

        Keeps tech_state consistent across planner and critic calls.
        """
        self._tech_state = new_state

```

This is your **M6–M7 integration spine**:

- M6 only needs to implement a `BotCore` with `observe()`.
    
- M7 stays in charge of all encoding details.
    
- M2 plugs in via `PlannerModel` / `CriticModel` without knowing the raw world format.
    
- M8 will just call `planner_tick()` and `evaluate_trace()`.
    

---

## 4. Testing Notes

You already have:

- **Unit tests on M7:**
    
    - `test_observation_planner_encoding.py`
        
    - `test_observation_critic_encoding.py`
        
    - `test_observation_perf.py`
        
    - `test_observation_pipeline.py`
        

For integration:

1. **Runtime smoke test**
    
    - Create dummy implementations:
        
        - `DummyBotCore` (returns fixed RawWorldSnapshot)
            
        - `DummyPlannerModel` (returns trivial plan)
            
        - `DummyCriticModel` (returns constant score)
            
    - Build `AgentRuntime` with those.
        
    - Call `planner_tick()` and `get_latest_planner_observation()` and assert:
        
        - planner returns a dict with expected keys
            
        - observation has all expected fields
            
    - Build a small `PlanTrace` and call `evaluate_trace()`.
        
2. **Contract tests for M6**
    
    - Test that the _real_ M6 `BotCore.observe()` returns a `RawWorldSnapshot`  
        whose attributes match what M7 expects:
        
        - `player_pos` is a dict with `x,y,z`
            
        - `entities` is a list of dicts or RawEntity objects with `x,y,z,type,data`
            
        - `inventory` is a list of dicts with at least `item_id` & `count`
            

---

## 5. Logging & Failure Points

If something is going to explode, it’s here:

1. **Shape mismatch from M6 → M7**
    
    - If `RawWorldSnapshot` changes (field renamed, entity structure tweaked):
        
        - `_summarize_*` helpers may KeyError or silently degrade.
            
    - Recommended:
        
        - Add lightweight runtime validation / logging in M7:
            
            - log unknown entity shapes
                
            - log missing inventory fields
                
2. **SemanticsDB errors**
    
    - `db.get_item_info(item_id, variant)` can raise or return weird stuff.
        
    - Current code already catches exceptions and falls back to `"unknown"`.
        
    - For debugging, log:
        
        - item_id / variant pairs that fail lookup frequently
            
3. **TechState drift**
    
    - If TechState isn’t kept in sync with actual progression, planner / critic  
        will reason with the wrong tier.
        
    - Add logs when `update_tech_state` is called:
        
        - old active → new active
            
4. **Critic not wired yet**
    
    - Until the critic model exists, `AgentRuntime.evaluate_trace()` is dead weight.
        
    - Either:
        
        - Wire in a stub CriticModel that just returns `"score": 0.0`, or
            
        - Guard calls with a clear error if critic is not configured.
            
5. **Performance regressions**
    
    - If M6 starts returning massive snapshots:
        
        - the perf test already checks caps, but you may want timing asserts too.
            
    - Drop simple logging around `encode_for_planner` runtime in debug mode.


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

Logging / diagnostics (failure points):

  - Shape mismatches in entities / inventory are logged at DEBUG level.
  - Semantics lookup failures are logged at DEBUG level.
  - encode_for_planner runtime is logged at DEBUG level (perf_counter).
"""

from __future__ import annotations

from dataclasses import asdict          # for TechState serialization
from math import sqrt                   # for distance calculations
from time import perf_counter           # for simple perf timing
from typing import Any, Dict, List      # JSON-like typing

import logging                          # for diagnostics

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


# Logger for this module
logger = logging.getLogger(__name__)


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

    Tolerant behavior:
      - accepts several common item id keys ("item_id", "item", "id")
      - skips non-dict / malformed entries
      - logs unusual shapes at DEBUG level
    """
    normalized: List[Dict[str, Any]] = []

    for stack in raw_inventory:
        if not isinstance(stack, dict):
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("Skipping non-dict inventory entry: %r", stack)
            continue

        item_id = (
            stack.get("item_id")
            or stack.get("item")
            or stack.get("id")
        )
        if not isinstance(item_id, str):
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "Skipping inventory entry missing item id: %r", stack
                )
            continue

        variant = stack.get("variant")  # M3 semantics will interpret this via gtnh_items
        try:
            count = int(stack.get("count", 0))
        except Exception:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "Inventory entry has non-int count; defaulting to 0: %r",
                    stack,
                )
            count = 0

        if count <= 0:
            # Negative/zero counts are filtered out silently.
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
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("Skipping non-dict machine entry: %r", m)
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

    If keys are missing, defaults are applied, and unusual shapes are logged
    at DEBUG level.
    """
    tick = int(raw_snapshot.get("tick", 0))

    raw_pos = raw_snapshot.get("position", {}) or {}
    if logger.isEnabledFor(logging.DEBUG):
        if not isinstance(raw_pos, dict):
            logger.debug("Position is not a dict; got %r", raw_pos)

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

    Logs SemanticsDB failures at DEBUG level, but does not crash.
    """
    summary: Dict[str, Any] = {
        "by_item": {},
        "by_category": {},
        "by_material": {},
    }

    for stack in raw.inventory:
        item_id = stack.get("item_id") or stack.get("item") or stack.get("id")
        variant = stack.get("variant")
        try:
            count = int(stack.get("count", 0))
        except Exception:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "Inventory stack has non-int count; defaulting to 0: %r",
                    stack,
                )
            count = 0

        if not item_id or count <= 0:
            continue

        try:
            info = db.get_item_info(item_id, variant)
        except Exception as exc:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "SemanticsDB.get_item_info failed for item_id=%r "
                    "variant=%r: %r",
                    item_id,
                    variant,
                    exc,
                )
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

    Logs unusual entity shapes at DEBUG level, but does not crash.
    """
    px = raw.player_pos.get("x", 0.0)
    py = raw.player_pos.get("y", 0.0)
    pz = raw.player_pos.get("z", 0.0)

    result: List[Dict[str, Any]] = []

    for e in raw.entities:
        if not isinstance(e, RawEntity):
            # If someone passes dicts for tests, handle that.
            if isinstance(e, dict) and logger.isEnabledFor(logging.DEBUG):
                missing = [k for k in ("x", "y", "z", "type") if k not in e]
                if missing:
                    logger.debug(
                        "Entity dict missing keys %s: %r",
                        missing,
                        e,
                    )

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

    Caps outputs at max_outputs to avoid bloating the encoding.
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

    Logs runtime and snapshot size at DEBUG level.
    """
    debug_timing = logger.isEnabledFor(logging.DEBUG)
    if debug_timing:
        t0 = perf_counter()

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

    payload = {
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

    if debug_timing:
        dt_ms = (perf_counter() - t0) * 1000.0
        try:
            entity_count = len(raw_snapshot.entities)
            inv_count = len(raw_snapshot.inventory)
        except Exception:
            entity_count = -1
            inv_count = -1

        logger.debug(
            "encode_for_planner runtime=%.2f ms entities=%s inventory=%s",
            dt_ms,
            entity_count,
            inv_count,
        )

    return payload


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
# src/agent/runtime_m6_m7.py
"""
Minimal integration of M6 (observation capture) and M7 (observation encoding).

Shows:
  - How BotCore (M6) provides RawWorldSnapshot
  - How M7 turns that into planner / critic-ready encodings
  - How M2 planner/critic models are invoked from this layer

Logging / failure handling:

  - Logs initial context_id + active tech.
  - Logs TechState changes on update_tech_state().
  - Warns if CriticModel is not configured.
  - Raises RuntimeError if evaluate_trace() is called without a critic.
"""

from __future__ import annotations

from dataclasses import dataclass               # to group runtime dependencies
from typing import Dict, Any, Optional          # JSON-like typing and Optional

import logging                                  # for runtime logging

# M7: pipeline helpers and Protocols
from observation.pipeline import (
    BotCore,                                    # Protocol describing M6's observe()
    PlannerModel,                               # Protocol for planner (M2)
    CriticModel,                                # Protocol for critic (M2)
    PlannerContext,                             # Bundles bot_core + semantics + planner
    CriticContext,                              # Bundles critic only
    planner_step,                               # M6 -> M7 -> M2 planner call
    critic_step,                                # M8 -> M7 -> M2 critic call
)

# M7: direct encoder usage (for optional helper)
from observation.encoder import (
    build_world_state,                          # raw dict -> WorldState (if you use it)
    make_planner_observation,                   # RawWorldSnapshot -> Observation
)

# M7: types for traces
from observation.trace_schema import PlanTrace   # critic path trace type

# M3: semantics + tech-state
from semantics.loader import SemanticsDB        # semantic database interface
from semantics.schema import TechState          # tech progression state

# M1: Observation type
from spec.types import Observation              # planner / LLM observation type


# Logger for this module
logger = logging.getLogger(__name__)


@dataclass
class AgentRuntimeConfig:
    """
    Configuration / static context for a running agent.

    Includes:
      - context_id: virtue / scenario id from M4 config
      - initial_tech_state: starting inferred tech state from M3
    """
    context_id: str
    initial_tech_state: TechState


@dataclass
class AgentRuntime:
    """
    Integrates M6 and M7 around the planner and critic models.

    This is the "runtime shell" that M8 will likely call into.
    """

    bot_core: BotCore                      # M6: provides observe() -> RawWorldSnapshot
    semantics_db: SemanticsDB              # M3: semantic info + item categories
    planner_model: PlannerModel            # M2: planner LLM wrapper
    critic_model: Optional[CriticModel]    # M2: critic LLM wrapper (optional)
    config: AgentRuntimeConfig             # static config (context_id, initial tech)

    def __post_init__(self) -> None:
        """
        Build planner and critic contexts after initialization.

        Also logs initial context and tech state.
        """
        # Build M7 planner context (M6 + M3 + M2 planner).
        self._planner_ctx = PlannerContext(
            bot_core=self.bot_core,
            semantics_db=self.semantics_db,
            planner_model=self.planner_model,
        )

        # Build M7 critic context if critic_model is provided.
        if self.critic_model is not None:
            self._critic_ctx: Optional[CriticContext] = CriticContext(
                critic_model=self.critic_model,
            )
        else:
            self._critic_ctx = None
            logger.warning(
                "AgentRuntime initialized without CriticModel; "
                "evaluate_trace() will raise until a critic is configured."
            )

        # Track current tech_state; may evolve over time as M3 updates it.
        self._tech_state = self.config.initial_tech_state

        # Log initial context and active tech.
        logger.debug(
            "AgentRuntime initialized with context_id=%s active_tech=%r",
            self.config.context_id,
            getattr(self._tech_state, "active", None),
        )

    # ------------------------------------------------------------------ #
    # Planner path: M6 → M7 → M2                                         #
    # ------------------------------------------------------------------ #

    def planner_tick(self) -> Dict[str, Any]:
        """
        Perform one planner cycle:

          1. M6: bot_core.observe() -> RawWorldSnapshot
          2. M7: planner_step(...) -> Observation -> PlannerEncoding payload
          3. M2: planner_model.plan(observation) -> plan dict

        Returns:
            plan: arbitrary JSON-like dict defined by M2.
        """
        plan = planner_step(
            ctx=self._planner_ctx,
            tech_state=self._tech_state,
            context_id=self.config.context_id,
        )
        return plan

    def get_latest_planner_observation(self) -> Observation:
        """
        Inspect the planner observation without actually calling the planner.

        Uses only M6 + M7, not the planner LLM.
        """
        raw_snapshot = self.bot_core.observe()
        obs = make_planner_observation(
            raw_snapshot=raw_snapshot,
            tech_state=self._tech_state,
            db=self.semantics_db,
            context_id=self.config.context_id,
        )
        return obs

    # ------------------------------------------------------------------ #
    # Critic path: M8 → M7 → M2                                          #
    # ------------------------------------------------------------------ #

    def evaluate_trace(self, trace: PlanTrace) -> Dict[str, Any]:
        """
        Evaluate a completed PlanTrace using CriticModel via M7:

          4. M8: builds PlanTrace with plan + steps + tech_state + payload
          5. M7: critic_step(...) -> CriticEncoding payload dict
          6. M2: critic_model.evaluate(payload) -> evaluation dict

        Raises:
            RuntimeError if no CriticModel is configured.
        """
        if self._critic_ctx is None:
            raise RuntimeError(
                "CriticModel is not configured on AgentRuntime; "
                "cannot evaluate trace."
            )

        evaluation = critic_step(
            ctx=self._critic_ctx,
            trace=trace,
        )
        return evaluation

    # ------------------------------------------------------------------ #
    # Optional: world-state utilities                                   #
    # ------------------------------------------------------------------ #

    def get_normalized_world_state(self) -> Any:
        """
        Demonstrate how to use M7's build_world_state() with M6 output.

        This is mostly for debugging, logging, or non-LLM subsystems.
        """
        raw_snapshot = self.bot_core.observe()
        raw_dict = {
            "tick": raw_snapshot.tick,
            "position": raw_snapshot.player_pos,
            "dimension": raw_snapshot.dimension,
            "inventory": raw_snapshot.inventory,
            "nearby_entities": raw_snapshot.entities,
            "blocks_of_interest": raw_snapshot.context.get(
                "blocks_of_interest", []
            ),
            "tech_state": {},
            "context": raw_snapshot.context,
        }
        world_state = build_world_state(raw_dict)
        return world_state

    # ------------------------------------------------------------------ #
    # Hooks for M8 / outer loop                                          #
    # ------------------------------------------------------------------ #

    def update_tech_state(self, new_state: TechState) -> None:
        """
        Allow outer systems (e.g., semantics / progression) to update TechState.

        Logs old_active -> new_active when the tier changes.
        """
        old_active = getattr(self._tech_state, "active", None)
        new_active = getattr(new_state, "active", None)

        if old_active != new_active:
            logger.info(
                "TechState updated: active %r -> %r",
                old_active,
                new_active,
            )

        self._tech_state = new_state

```


```
# src/agent/logging_config.py
"""
Central logging configuration for GTNH_Agent runtime.

Call configure_logging() from your main entrypoint once, for example:

    from agent.logging_config import configure_logging
    configure_logging()

After that, M6/M7 logs (including diagnostics in observation.encoder and
agent.runtime_m6_m7) will be visible on stdout.
"""

from __future__ import annotations

import logging
import sys


def configure_logging(level: int = logging.INFO) -> None:
    """
    Configure root logging if no handlers are attached yet.

    Args:
        level: default logging level (e.g., logging.INFO, logging.DEBUG)
    """
    root = logging.getLogger()

    # Don't duplicate handlers if someone already configured logging.
    if root.handlers:
        return

    handler = logging.StreamHandler(stream=sys.stdout)
    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    handler.setFormatter(formatter)
    root.addHandler(handler)
    root.setLevel(level)

```


### How this matches your checklist

1. **Shape mismatch from M6 → M7**
    
    - `_normalize_inventory`, `_normalize_machines`, `_summarize_nearby_entities`, and `build_world_state` log weird shapes at DEBUG.
        
2. **SemanticsDB errors**
    
    - `_summarize_inventory` catches `db.get_item_info` exceptions and logs `item_id` / `variant` / exception at DEBUG.
        
3. **TechState drift**
    
    - `AgentRuntime.update_tech_state` logs `active` tier changes as INFO.
        
    - `AgentRuntime.__post_init__` logs initial context + active tech at DEBUG.
        
4. **Critic not wired yet**
    
    - `AgentRuntime` now allows `critic_model=None`.
        
    - Logs a WARNING if critic is absent.
        
    - `evaluate_trace` raises `RuntimeError` if called with no critic configured.
        
5. **Performance regressions**
    
    - `encode_for_planner` measures runtime with `perf_counter()` when DEBUG logging is enabled and logs ms + entity/inventory counts.
        

Enable DEBUG logging with:
python:
```
from agent.logging_config import configure_logging
import logging

configure_logging(level=logging.DEBUG)

```

Then when something explodes, at least you’ll have breadcrumbs instead of vibes.


---

So, by the time you plug this into Phase 2 Integration, M6 and M7 are no longer “two separate modules” but **one coherent perception stack**:

- M6 owns **raw sensory access**
    
- M7 owns **what the brain actually sees**
    

And now you’ve got a runtime shell that makes the whole thing callable without spelunking through five layers of abstraction every time.