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
