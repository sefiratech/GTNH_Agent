# src/semantics/tech_state.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List, Set, Optional
from pathlib import Path

import yaml
import networkx as nx

from spec.types import WorldState          # from M1
from .schema import TechState, TechTarget  # our semantic types


CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"


# ---------------------------------------------------------------------------
# Internal representation of tech graph nodes
# ---------------------------------------------------------------------------

@dataclass
class TechNode:
    """
    Metadata for a single tech node, as loaded from gtnh_tech_graph.yaml.

    - id: tech state id ("stone_age", "steam_age", "lv_electric", ...)
    - description: human-readable description
    - tags: semantic tags for this state ("early-game", "steam", "lv", ...)
    - prerequisites: list of prerequisite tech state ids
    - unlocks: list of tech states this one directly unlocks
    - requirements: structured requirements block:
        {
          "min_items": [...],
          "min_machines": [...],
          "flags": [...]
        }
    - recommended_goals: list of human-readable goals used for targets/curriculum
    - virtue_profile: optional profile name linking into virtues.yaml
    """
    id: str
    description: str
    tags: List[str]
    prerequisites: List[str]
    unlocks: List[str]
    requirements: Dict[str, Any]
    recommended_goals: List[str]
    virtue_profile: Optional[str]


# ---------------------------------------------------------------------------
# Tech graph
# ---------------------------------------------------------------------------

class TechGraph:
    """
    DAG representation of GTNH tech progression.

    Primary consumer:
      - infer_tech_state_from_world(...) to determine active/unlocked tech.
      - WorldModel.simulate_tech_progress(...) for "how many steps to X?" and
        "which tech nodes are still gating this goal?" style queries.
    """

    def __init__(self) -> None:
        raw = self._load_graph_cfg()
        states_cfg: Dict[str, Dict[str, Any]] = raw.get("tech_states", {})

        self._nodes: Dict[str, TechNode] = {}
        self._graph = nx.DiGraph()

        # Populate nodes
        for tech_id, data in states_cfg.items():
            node = TechNode(
                id=tech_id,
                description=data.get("description", ""),
                tags=list(data.get("tags", [])),
                prerequisites=list(data.get("prerequisites", []) or []),
                unlocks=list(data.get("unlocks", []) or []),
                requirements=dict(data.get("requirements", {}) or {}),
                recommended_goals=list(data.get("recommended_goals", []) or []),
                virtue_profile=data.get("virtue_profile"),
            )
            self._nodes[tech_id] = node
            self._graph.add_node(tech_id)

        # Add edges prereq -> tech
        for tech_id, node in self._nodes.items():
            for prereq in node.prerequisites:
                self._graph.add_edge(prereq, tech_id)

        # Ensure graph is a DAG
        if not nx.is_directed_acyclic_graph(self._graph):
            raise ValueError("Tech graph must be a DAG")

    def _load_graph_cfg(self) -> Dict[str, Any]:
        """Load tech graph YAML from config."""
        path = CONFIG_DIR / "gtnh_tech_graph.yaml"
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if data is None:
            return {}
        if not isinstance(data, dict):
            raise ValueError(f"Tech graph config {path} must be a mapping at top level.")
        return data

    # ---------------------- basic query helpers -------------------------

    def all_nodes(self) -> List[str]:
        """Return all tech ids."""
        return list(self._nodes.keys())

    def get_node(self, tech_id: str) -> TechNode:
        """Return TechNode by id."""
        return self._nodes[tech_id]

    def prerequisites_of(self, tech_id: str) -> List[str]:
        """Return direct prerequisites for a tech id."""
        return self._nodes[tech_id].prerequisites

    def successors_of(self, tech_id: str) -> List[str]:
        """Return tech states unlocked by this one."""
        return self._nodes[tech_id].unlocks

    def topological_order(self) -> List[str]:
        """Return a topologically sorted list of tech ids."""
        return list(nx.topological_sort(self._graph))


# ---------------------------------------------------------------------------
# Tech state inference
# ---------------------------------------------------------------------------

def _extract_item_ids_from_inventory(inventory: Any) -> Set[str]:
    """
    Best-effort extraction of item ids from WorldState.inventory.

    This is intentionally loose: inventory representation may evolve.
    We try common patterns:
      - list of { "item_id": "modid:item", ... }
      - list of { "item": "modid:item", ... }
      - list of { "id": "modid:item", ... }
    """
    item_ids: Set[str] = set()
    if isinstance(inventory, list):
        for slot in inventory:
            if not isinstance(slot, dict):
                continue
            item_id = (
                slot.get("item_id")
                or slot.get("item")
                or slot.get("id")
            )
            if isinstance(item_id, str):
                item_ids.add(item_id)
    return item_ids


def _extract_machine_ids_from_context(context: Dict[str, Any]) -> Set[str]:
    """
    Best-effort extraction of machine ids from WorldState.context["machines"].

    Expected patterns:
      - { "id": "gregtech:lv_macerator", ... }
      - { "machine_id": "gregtech:steam_macerator", ... }
      - { "type": "gregtech:steam_macerator", ... }   # legacy / loose
    """
    machines_list = context.get("machines", [])
    machine_ids: Set[str] = set()
    if isinstance(machines_list, list):
        for m in machines_list:
            if not isinstance(m, dict):
                continue
            mid = (
                m.get("machine_id")
                or m.get("id")
                or m.get("type")
            )
            if isinstance(mid, str):
                machine_ids.add(mid)
    return machine_ids


def _derive_flags(items: Set[str], machines: Set[str]) -> Set[str]:
    """
    Derive coarse flags from items/machines.

    These are heuristics that can be matched by requirements.flags in the graph:
      - "has_machines"
      - "has_steam_power"
      - "has_lv_power"
      - "has_mv_power"
    """
    flags: Set[str] = set()

    if machines:
        flags.add("has_machines")

    # Very rough tier detection; can be refined later
    for mid in machines:
        lower = mid.lower()
        if "steam" in lower:
            flags.add("has_steam_power")
        if "lv_" in lower or ":lv" in lower:
            flags.add("has_lv_power")
        if "mv_" in lower or ":mv" in lower:
            flags.add("has_mv_power")

    # Item-based hints can be added later (e.g. circuits, blast furnaces)
    return flags


def _requirements_satisfied(
    node: TechNode,
    items_present: Set[str],
    machines_present: Set[str],
    flags_present: Set[str],
) -> bool:
    """
    Check if a node's requirements are satisfied given current world evidence.

    requirements:
      min_items:     all must be present in items_present (if non-empty)
      min_machines:  all must be present in machines_present (if non-empty)
      flags:         all must be present in flags_present (if non-empty)
    """
    req = node.requirements or {}
    min_items = set(req.get("min_items", []) or [])
    min_machines = set(req.get("min_machines", []) or [])
    req_flags = set(req.get("flags", []) or [])

    if min_items and not min_items.issubset(items_present):
        return False
    if min_machines and not min_machines.issubset(machines_present):
        return False
    if req_flags and not req_flags.issubset(flags_present):
        return False
    return True


def infer_tech_state_from_world(
    world: WorldState,
    graph: Optional[TechGraph] = None,
) -> TechState:
    """
    Infer the tech state from current machines & inventory.

    Primary expectation:
      - world.inventory is a list of normalized InventoryStack dicts
        with keys: "item_id", "variant", "count"
      - world.context["machines"] is a list of normalized MachineEntry dicts
        with keys: "id", "type", "tier", "extra"

    The function is tolerant of some legacy shapes (e.g. "item", "id",
    "machine_id", "type"), but callers SHOULD go through
    observation.encoder.build_world_state.

    This uses:
      - gtnh_tech_graph.yaml: tech_states[*].requirements
      - WorldState.inventory          → item ids
      - WorldState.context["machines"]→ machine ids

    Strategy:
      1. Extract item_ids, machine_ids, and derived flags.
      2. Traverse tech graph in topological order.
      3. For each node:
           - if all prerequisites are already unlocked AND
             its requirements are satisfied → mark as unlocked.
      4. The "active" state is the *last* unlocked node in topo order,
         or "stone_age" / first node if nothing else unlocked.
    """
    if graph is None:
        graph = TechGraph()

    # Extract evidence from world
    context = getattr(world, "context", {}) or {}
    inventory = getattr(world, "inventory", []) or []

    item_ids = _extract_item_ids_from_inventory(inventory)
    machine_ids = _extract_machine_ids_from_context(context)
    flags = _derive_flags(item_ids, machine_ids)

    unlocked: Set[str] = set()

    # Traverse in topo order so prereqs are always considered first
    for tech_id in graph.topological_order():
        node = graph.get_node(tech_id)
        prereqs = set(node.prerequisites)

        # Require all prereqs to be unlocked first
        if not prereqs.issubset(unlocked):
            continue

        # Check requirements against current evidence
        if _requirements_satisfied(node, item_ids, machine_ids, flags):
            unlocked.add(tech_id)

    # Determine active state: the "highest" unlocked node in topo order
    active: Optional[str] = None
    for tech_id in graph.topological_order():
        if tech_id in unlocked:
            active = tech_id
    if active is None and graph.all_nodes():
        # Nothing satisfied; fall back to the first node as baseline
        active = graph.topological_order()[0]
    elif active is None:
        # Truly empty graph, just fake something
        active = "unknown"

    # Aggregate tags & virtue profile from the active node
    active_node = graph.get_node(active) if active in graph.all_nodes() else None
    tags: List[str] = list(active_node.tags) if active_node else []
    virtue_profile: Optional[str] = active_node.virtue_profile if active_node else None

    evidence: Dict[str, Any] = {
        "items": sorted(item_ids),
        "machines": sorted(machine_ids),
        "flags": sorted(flags),
    }

    return TechState(
        active=active,
        unlocked=sorted(unlocked),
        tags=tags,
        virtue_profile=virtue_profile,
        evidence=evidence,
    )


# ---------------------------------------------------------------------------
# Next-target suggestion
# ---------------------------------------------------------------------------

def suggest_next_targets(tech_state: TechState, graph: TechGraph) -> List[TechTarget]:
    """
    Given current tech_state, suggest next reachable tech targets.

    For each tech node not yet unlocked:
      - compute missing prerequisites
      - estimate difficulty as len(missing_prereqs) (cheap but effective)
      - use recommended_goals as expected_benefits

    This is exactly the kind of “what’s gating me?” summary the WorldModel
    and curriculum can lean on without doing full-blown simulation.
    """
    unlocked = set(tech_state.unlocked)
    suggestions: List[TechTarget] = []

    topo = graph.topological_order()
    position = {tid: idx for idx, tid in enumerate(topo)}

    for tech_id in topo:
        if tech_id in unlocked:
            continue

        node = graph.get_node(tech_id)
        prereqs = set(node.prerequisites)
        missing = sorted(prereqs - unlocked)

        if missing:
            reason = f"Requires: {', '.join(missing)}"
            difficulty = float(len(missing))
        else:
            reason = "All prerequisites satisfied."
            difficulty = 1.0

        suggestions.append(
            TechTarget(
                id=tech_id,
                reason=reason,
                prerequisites_missing=missing,
                difficulty_score=difficulty,
                expected_benefits=list(node.recommended_goals),
            )
        )

    # Sort: easiest first, then by graph order
    suggestions.sort(
        key=lambda t: (t.difficulty_score, position.get(t.id, 1e9))
    )
    return suggestions


__all__ = [
    "TechNode",
    "TechGraph",
    "infer_tech_state_from_world",
    "suggest_next_targets",
]

