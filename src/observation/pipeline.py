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

