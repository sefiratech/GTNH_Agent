# src/agent/bootstrap.py
"""
Inter-Phase Integration 0,1,2 bootstrap.

Purpose:
    Provide a single entrypoint that wires together:

      - Phase 0: M0 (env profiles), M1 (core types)
      - Phase 1: M2 (LLM stack interfaces), M3 (semantics), 
                 M4 (virtue lattice), M5 (skills)
      - Phase 2: M6 (BotCore spec), M7 (observation encoder)

    In practice, this file currently:
      - uses local dummy implementations for:
          * BotCore (M6 runtime)
          * SemanticsDB (M3 data source)
          * PlannerModel & CriticModel (M2)
      - relies on real:
          * TechState (M3 schema)
          * RawWorldSnapshot (M6 snapshot types)
          * M7 encoder / AgentRuntime integration layer

    This gives you a working Phase 0–2 wiring that can be
    swapped to real implementations later (Forge/IPC BotCore,
    real SemanticsDB, real planner/critic backends).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import logging

from agent.logging_config import configure_logging
from agent.runtime_m6_m7 import AgentRuntime, AgentRuntimeConfig

# M6: snapshot types (used by tests and observation.testing)
from bot_core.snapshot import RawWorldSnapshot

# M7: test helpers – we reuse the same minimal snapshot as tests
from observation.testing import make_minimal_snapshot

# M3: tech-state schema
from semantics.schema import TechState

# M1: Observation type (for planner model interface)
from spec.types import Observation


# ---------------------------------------------------------------------------
# Dummy implementations for integration harness
# ---------------------------------------------------------------------------


class DummySemanticsDB:
    """
    Minimal SemanticsDB stub.

    Must satisfy:

        - db.get_item_info(item_id, variant) -> object with:
            * category: str
            * material: str | None
        - db.recipes: iterable (used by semantics.crafting.craftable_items)

    For now:
        - category/material always "unknown"
        - recipes is an empty list so no craftables are produced
    """

    class _Info:
        __slots__ = ("category", "material")

        def __init__(self, category: str = "unknown", material: str | None = "unknown"):
            self.category = category
            self.material = material

    def __init__(self) -> None:
        # semantics.crafting.craftable_items iterates over this.
        # Leaving it empty means _summarize_craftables() just returns
        # an empty "top_outputs" list, which is fine for Phase 0–2 wiring.
        self.recipes: list[Any] = []

    def get_item_info(self, item_id: str, variant: str | None) -> "DummySemanticsDB._Info":
        # You can add smarter behavior later if you want.
        return self._Info()


class DummyBotCore:
    """
    Minimal BotCore implementation for the Phase 0–2 integration demo.

    observe() returns a RawWorldSnapshot using the same helper as M7 tests,
    so the shape is guaranteed to match the encoder expectations.
    """

    def observe(self) -> RawWorldSnapshot:
        return make_minimal_snapshot()


class DummyPlannerModel:
    """
    Minimal planner model stub implementing the PlannerModel protocol.

    plan(observation: Observation) -> dict

    Returns a trivial static plan that still exercises the M7 encoding.
    """

    def plan(self, observation: Observation) -> Dict[str, Any]:
        return {
            "steps": [],
            "meta": {
                "note": "dummy planner output",
                "tech_state": observation.json_payload.get("tech_state"),
                "context_id": observation.json_payload.get("context_id"),
            },
        }


class DummyCriticModel:
    """
    Minimal critic model stub implementing the CriticModel protocol.

    evaluate(payload: dict) -> dict

    You can replace this with a real critic later.
    """

    def evaluate(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "score": 0.0,
            "verdict": "stub",
            "notes": "DummyCriticModel: no real evaluation performed.",
        }


# ---------------------------------------------------------------------------
# Bootstrap API
# ---------------------------------------------------------------------------


@dataclass
class Phase012BootstrapConfig:
    """
    Aggregate config for Phase 0–2 bootstrap.

    For now this is intentionally tiny; later you can thread in:
      - env profile name
      - logging level
      - flags for real vs dummy semantics / models
    """
    context_id: str = "dev_dummy"
    logging_level: int = logging.INFO


def build_agent_runtime(
    profile: str = "dev_local",
    use_dummy_semantics: bool = True,
    bootstrap_config: Phase012BootstrapConfig | None = None,
) -> AgentRuntime:
    """
    Inter-Phase 0,1,2: construct an AgentRuntime using dummy M2/M3/M6
    implementations and real M7 encoding.

    Args:
        profile:
            Reserved for future env-profile integration (M0).
            Currently unused; kept for forward compatibility.

        use_dummy_semantics:
            Kept for future toggle when a real SemanticsDB loader exists.
            Currently, only dummy semantics are used.

        bootstrap_config:
            Optional advanced config. If omitted, defaults are used.

    Returns:
        AgentRuntime configured with:
          - DummyBotCore
          - DummySemanticsDB
          - DummyPlannerModel
          - DummyCriticModel
          - TechState(active="stone_age")
    """
    if bootstrap_config is None:
        bootstrap_config = Phase012BootstrapConfig()

    # Configure logging once.
    configure_logging(level=bootstrap_config.logging_level)

    # --- M3: TechState + SemanticsDB (dummy for now) ---
    semantics_db = DummySemanticsDB()
    tech_state = TechState(
        unlocked=["stone_age"],
        active="stone_age",
        evidence={},
    )

    # --- M6: BotCore (dummy) ---
    bot_core = DummyBotCore()

    # --- M2: Planner / Critic (dummy) ---
    planner_model = DummyPlannerModel()
    critic_model = DummyCriticModel()

    # --- Runtime config (context_id from bootstrap config) ---
    runtime_config = AgentRuntimeConfig(
        context_id=bootstrap_config.context_id,
        initial_tech_state=tech_state,
    )

    runtime = AgentRuntime(
        bot_core=bot_core,
        semantics_db=semantics_db,
        planner_model=planner_model,
        critic_model=critic_model,
        config=runtime_config,
    )

    return runtime

