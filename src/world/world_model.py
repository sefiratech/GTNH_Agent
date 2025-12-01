# path: src/world/world_model.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import logging

from semantics.schema import TechState  # M3 tech state model

try:
    # Expected M3 tech graph abstraction
    from semantics.tech_graph import TechGraph
except Exception:  # pragma: no cover - optional import for now
    TechGraph = Any  # type: ignore[assignment]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prediction / delta schemas
# ---------------------------------------------------------------------------

@dataclass
class TechProgressPrediction:
    """
    Cheap, heuristic prediction for progressing toward a tech goal.

    Fields
    ------
    goal_id:
        Identifier for the candidate goal / tech target.

    reachable:
        Whether the goal appears reachable from the current TechState,
        according to TechGraph (if present).

    estimated_steps:
        Rough number of tech steps / unlocks required to reach the goal.

    blocking_prereqs:
        Names/IDs of tech nodes that are missing and must be unlocked first.

    path:
        Best-effort ordered path of tech nodes from current to target.

    notes:
        Free-form explanation / debugging trail for humans & LLMs.
    """
    goal_id: str
    reachable: bool
    estimated_steps: int
    blocking_prereqs: List[str] = field(default_factory=list)
    path: List[str] = field(default_factory=list)
    notes: str = ""


@dataclass
class InfraDelta:
    """
    Heuristic estimate of infrastructure impact.

    This is deliberately coarse. The point is to give:
      - a throughput multiplier
      - a list of likely new bottlenecks
      - some side-effect metadata

    Fields
    ------
    description:
        Human-readable summary.

    throughput_multiplier:
        Approximate multiplicative effect on overall factory throughput
        for the targeted subsystem (1.0 = no change).

    bottlenecks:
        Labels for inferred or suspected bottlenecks after the change.

    side_effects:
        Arbitrary metadata, e.g.,
          {
            "machines_added": 2,
            "machines_removed": 0,
            "power_delta_eu_per_tick": 96,
          }
    """
    description: str
    throughput_multiplier: float
    bottlenecks: List[str] = field(default_factory=list)
    side_effects: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ResourceForecast:
    """
    Simple linear resource forecast.

    Fields
    ------
    horizon_ticks:
        Number of ticks / time units in the forecast horizon.

    projected:
        Mapping from resource name -> projected amount at horizon end.

    shortages:
        Resources that are expected to hit zero or below within the horizon.

    surplus:
        Resources that are expected to increase or remain comfortably positive.

    notes:
        Free-form explanation or debug info.
    """
    horizon_ticks: int
    projected: Dict[str, float]
    shortages: List[str] = field(default_factory=list)
    surplus: List[str] = field(default_factory=list)
    notes: str = ""


# ---------------------------------------------------------------------------
# WorldModel core
# ---------------------------------------------------------------------------

class WorldModel:
    """
    Lightweight predictive world-model.

    This is intentionally NOT a full simulator. It is a cheap heuristic layer
    that gives the rest of the system a rough sense of:

      - how far a tech goal is from the current TechState
      - what infra changes are likely to do to throughput
      - whether resources are trending toward shortage or comfort

    Responsibilities
    ----------------
    - Wrap TechGraph (M3) for simple tech-progress predictions.
    - Use basic arithmetic / metadata for infra & resource forecasts.
    - Stay pure and side-effect free (no I/O, no model calls).

    Design notes
    ------------
    - All methods are written to degrade gracefully if optional inputs
      (TechGraph, rich factory_layout semantics) are not fully wired yet.
    - Callers can replace this class with a richer implementation later
      without changing the public method signatures.
    """

    def __init__(
        self,
        tech_graph: Optional[TechGraph] = None,
        *,
        semantics: Optional[Any] = None,
    ) -> None:
        """
        Parameters
        ----------
        tech_graph:
            Optional TechGraph instance from M3. If provided, it is used to
            estimate paths / distances in tech space.

        semantics:
            Optional handle to item/block semantics. For v1, this is just
            stored and passed around; implementations may introspect it later.
        """
        self._tech_graph = tech_graph
        self._semantics = semantics

    # -----------------------------------------------------------------------
    # Tech progress prediction
    # -----------------------------------------------------------------------

    def simulate_tech_progress(
        self,
        current_tech_state: TechState,
        candidate_goal: Any,
    ) -> TechProgressPrediction:
        """
        Predict how hard it is to reach a candidate tech goal.

        `candidate_goal` is intentionally flexible:
          - str  -> interpreted as a tech node ID or label
          - AgentGoal -> we try goal.id or goal.text as a tech node hint
          - dict -> check keys ["tech_target", "target_node", "id"]

        Heuristics:
        - If TechGraph is available and can supply a path from
          current_tech_state.active -> target_node, we:
            * use the path length as estimated_steps
            * anything in path not in current_tech_state.unlocked
              becomes a "blocking prereq".
        - If TechGraph is missing or the target is unknown, we:
            * mark reachable=False
            * estimated_steps=0
            * leave notes explaining the situation.
        """
        target_id = self._extract_goal_target_id(candidate_goal)
        goal_id = target_id or getattr(candidate_goal, "id", "unknown_goal")

        if self._tech_graph is None:
            notes = "No TechGraph available; cannot simulate path. Treating as unreachable for now."
            return TechProgressPrediction(
                goal_id=str(goal_id),
                reachable=False,
                estimated_steps=0,
                blocking_prereqs=[],
                path=[],
                notes=notes,
            )

        active = getattr(current_tech_state, "active", None)
        unlocked = set(getattr(current_tech_state, "unlocked", []) or [])

        if not active or not target_id:
            notes = f"Missing active tech or target_id (active={active}, target={target_id}); cannot compute path."
            return TechProgressPrediction(
                goal_id=str(goal_id),
                reachable=False,
                estimated_steps=0,
                blocking_prereqs=[],
                path=[],
                notes=notes,
            )

        path: List[str] = []
        try:
            # Duck-typed TechGraph API.
            # Common patterns:
            #   - tech_graph.shortest_path(src, dst) -> List[str]
            #   - tech_graph.path(src, dst) -> List[str]
            if hasattr(self._tech_graph, "shortest_path"):
                path = self._tech_graph.shortest_path(active, target_id)  # type: ignore[call-arg]
            elif hasattr(self._tech_graph, "path"):
                path = self._tech_graph.path(active, target_id)  # type: ignore[call-arg]
            else:
                raise AttributeError("TechGraph has no shortest_path/path API")
        except Exception as exc:
            logger.warning("WorldModel.simulate_tech_progress: failed to compute path: %r", exc)
            return TechProgressPrediction(
                goal_id=str(goal_id),
                reachable=False,
                estimated_steps=0,
                blocking_prereqs=[],
                path=[],
                notes=f"TechGraph error: {exc!r}",
            )

        if not path:
            return TechProgressPrediction(
                goal_id=str(goal_id),
                reachable=False,
                estimated_steps=0,
                blocking_prereqs=[],
                path=[],
                notes="No path returned from TechGraph.",
            )

        # Path includes current active node; "steps" are edges.
        estimated_steps = max(0, len(path) - 1)

        # Any node in path that is not unlocked is considered a blocking prereq.
        blocking_prereqs = [node for node in path if node not in unlocked]

        return TechProgressPrediction(
            goal_id=str(goal_id),
            reachable=True,
            estimated_steps=estimated_steps,
            blocking_prereqs=blocking_prereqs,
            path=path,
            notes="Heuristic estimate from TechGraph shortest path.",
        )

    def _extract_goal_target_id(self, candidate_goal: Any) -> Optional[str]:
        """
        Best-effort extraction of a tech target identifier from goal-like objects.
        """
        # Case 1: already a string
        if isinstance(candidate_goal, str):
            return candidate_goal

        # Case 2: AgentGoal-like: maybe has .tech_target or just .id
        tech_target = getattr(candidate_goal, "tech_target", None)
        if isinstance(tech_target, str) and tech_target:
            return tech_target

        goal_id = getattr(candidate_goal, "id", None)
        if isinstance(goal_id, str) and goal_id:
            return goal_id

        # Case 3: dict-like payload from curriculum / planner
        if isinstance(candidate_goal, dict):
            for key in ("tech_target", "target_node", "id", "goal_id"):
                value = candidate_goal.get(key)
                if isinstance(value, str) and value:
                    return value

        return None

    # -----------------------------------------------------------------------
    # Infrastructure effects
    # -----------------------------------------------------------------------

    def estimate_infra_effect(
        self,
        factory_layout: Dict[str, Any],
        change: Dict[str, Any],
    ) -> InfraDelta:
        """
        Estimate impact of an infrastructure change.

        Inputs are deliberately generic:
          - factory_layout:
              {
                "machines": [
                  {"id": "...", "type": "coke_oven", "throughput": 1.0, ...},
                  ...
                ],
                "power_capacity_eu_per_tick": 512,
                ...
              }

          - change:
              {
                "op": "add" | "remove" | "modify",
                "machine": {...},           # for add
                "machine_id": "...",        # for remove/modify
                "throughput_multiplier": 2  # optional hint
              }

        Heuristics:
          - Sum baseline machine "throughput" across all machines.
          - Apply change (add/remove/modify) to get new total throughput.
          - throughput_multiplier = new / old (defaulting old to >0).
          - Power impact / bottlenecks are inferred from simple rules of thumb.
        """
        machines = list(factory_layout.get("machines", []) or [])
        baseline_throughput = self._total_throughput(machines)

        op = change.get("op") or change.get("type") or "modify"
        op = str(op).lower()

        # Clone the list so we don't mutate caller state
        new_machines = [dict(m) for m in machines]

        if op == "add":
            m = change.get("machine") or {}
            if isinstance(m, dict):
                new_machines.append(dict(m))
        elif op == "remove":
            mid = str(change.get("machine_id", ""))
            new_machines = [m for m in new_machines if str(m.get("id", "")) != mid]
        elif op == "modify":
            mid = str(change.get("machine_id", ""))
            patch = change.get("machine") or {}
            if isinstance(patch, dict) and mid:
                for m in new_machines:
                    if str(m.get("id", "")) == mid:
                        m.update(patch)
                        break

        new_throughput = self._total_throughput(new_machines)

        if baseline_throughput <= 0:
            throughput_multiplier = 1.0 if new_throughput <= 0 else 2.0
        else:
            throughput_multiplier = float(new_throughput) / float(baseline_throughput)

        # Power estimates: assume each machine exposes power_draw_eu_per_tick
        old_power = sum(float(m.get("power_draw_eu_per_tick", 0.0)) for m in machines)
        new_power = sum(float(m.get("power_draw_eu_per_tick", 0.0)) for m in new_machines)
        power_delta = new_power - old_power

        power_capacity = float(factory_layout.get("power_capacity_eu_per_tick", 0.0))
        bottlenecks: List[str] = []

        if power_capacity > 0 and new_power > power_capacity:
            bottlenecks.append("power_capacity")
        if throughput_multiplier > 1.5:
            bottlenecks.append("downstream_io")
        if throughput_multiplier < 0.75:
            bottlenecks.append("underutilization")

        side_effects: Dict[str, Any] = {
            "machines_before": len(machines),
            "machines_after": len(new_machines),
            "throughput_before": baseline_throughput,
            "throughput_after": new_throughput,
            "power_draw_delta_eu_per_tick": power_delta,
        }

        desc = f"Inferred throughput multiplier ≈ {throughput_multiplier:.2f} for op={op!r}."

        return InfraDelta(
            description=desc,
            throughput_multiplier=throughput_multiplier,
            bottlenecks=bottlenecks,
            side_effects=side_effects,
        )

    def _total_throughput(self, machines: List[Dict[str, Any]]) -> float:
        """
        Sum machine 'throughput' with a few fallbacks.

        If no explicit throughput is present, count machines as 1.0 each.
        """
        total = 0.0
        for m in machines:
            if not isinstance(m, dict):
                continue
            if "throughput" in m:
                try:
                    total += float(m["throughput"])
                except Exception:
                    total += 1.0
            else:
                total += 1.0
        return total

    # -----------------------------------------------------------------------
    # Resource trajectory
    # -----------------------------------------------------------------------

    def estimate_resource_trajectory(
        self,
        inventory: Dict[str, float],
        consumption_rates: Dict[str, float],
        horizon: int,
    ) -> ResourceForecast:
        """
        Predict resource levels after a fixed horizon under simple
        linear consumption/production.

        Parameters
        ----------
        inventory:
            Mapping of resource name -> current amount.

        consumption_rates:
            Mapping of resource name -> net rate per tick.
            Convention:
              - positive = net consumption (amount decreases)
              - negative = net production (amount increases)

        horizon:
            Number of ticks / time units to project forward.

        Heuristics
        ----------
        projected[r] = inventory[r] - consumption_rates[r] * horizon

        - Any resource with projected <= 0 is marked as a shortage.
        - Any resource with projected > inventory[r] is considered surplus.
        """
        if horizon <= 0:
            return ResourceForecast(
                horizon_ticks=0,
                projected=dict(inventory),
                shortages=[],
                surplus=[],
                notes="Zero or negative horizon; returning current inventory.",
            )

        projected: Dict[str, float] = {}
        shortages: List[str] = []
        surplus: List[str] = []

        all_keys = set(inventory.keys()) | set(consumption_rates.keys())

        for key in sorted(all_keys):
            current = float(inventory.get(key, 0.0))
            rate = float(consumption_rates.get(key, 0.0))

            future = current - rate * horizon
            projected[key] = future

            if future <= 0.0 and rate > 0.0:
                shortages.append(key)
            elif future > current:
                surplus.append(key)

        notes = (
            "Linear projection; ignores stochasticity, crafting graphs, "
            "and storage caps. Upgrade later if you start caring."
        )

        return ResourceForecast(
            horizon_ticks=horizon,
            projected=projected,
            shortages=shortages,
            surplus=surplus,
            notes=notes,
        )


__all__ = [
    "WorldModel",
    "TechProgressPrediction",
    "InfraDelta",
    "ResourceForecast",
]

