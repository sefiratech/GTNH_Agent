# tests/fakes/fake_runtime.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from observation.trace_schema import PlanTrace, TraceStep
from spec.types import WorldState, Action, ActionResult


@dataclass
class _FakeObservation:
    """Minimal observation wrapper providing a json_payload field."""
    json_payload: Dict[str, Any]


@dataclass
class _FakeRuntimeConfig:
    """Minimal config object to satisfy AgentRuntime expectations."""
    context_id: str = "test_context"
    profile_name: str = "test_profile"


class FakeAgentRuntime:
    """
    Dummy AgentRuntime for M8 tests.

    Responsibilities:
      - Provide planner_tick() returning a tiny plan
      - Provide get_latest_planner_observation() for PlanTrace
      - Provide get_tech_state() and get_context_id()
      - Provide execute_plan_step() → list[TraceStep]
      - Omit critic_model so critic remains optional
    """

    def __init__(self) -> None:
        self.config = _FakeRuntimeConfig()
        self.planner_name = "fake_planner_v1"
        self._tick: int = 0
        self._position = {"x": 0.0, "y": 64.0, "z": 0.0}

    # ------------------------------------------------
    # Planner-facing API
    # ------------------------------------------------

    def planner_tick(self) -> Dict[str, Any]:
        """
        Return a trivial plan with a single step that moves the bot.
        """
        return {
            "steps": [
                {
                    "skill": "test_move",
                    "params": {"x": 1, "y": 64, "z": 0},
                }
            ]
        }

    def get_latest_planner_observation(self) -> _FakeObservation:
        """
        Minimal observation object with a json_payload dict.
        """
        self._tick += 1
        payload: Dict[str, Any] = {
            "tick": self._tick,
            "position": dict(self._position),
            "context_id": self.config.context_id,
        }
        return _FakeObservation(json_payload=payload)

    def get_tech_state(self) -> Dict[str, Any]:
        """
        Simple tech state stub for PlanTrace.tech_state.
        """
        return {"stage": "test_stage", "tier": "lv"}

    def get_context_id(self) -> str:
        """
        Context ID used by virtue scoring.
        """
        return self.config.context_id

    # ------------------------------------------------
    # Execution-facing API
    # ------------------------------------------------

    def execute_plan_step(
        self,
        step_spec: Dict[str, Any],
        step_idx: int,
    ) -> List[TraceStep]:
        """
        Execute a single plan step as one or more low-level actions.

        In this fake implementation, we:
          - build a single Action of type 'move_to'
          - update the internal position
          - wrap the result in a TraceStep
        """
        params = step_spec.get("params", {})
        target_x = float(params.get("x", self._position["x"]))
        target_y = float(params.get("y", self._position["y"]))
        target_z = float(params.get("z", self._position["z"]))

        world_before = self._make_world_state()
        action = Action(
            type="move_to",
            params={"x": target_x, "y": target_y, "z": target_z},
        )

        # Commit the move
        self._position["x"] = target_x
        self._position["y"] = target_y
        self._position["z"] = target_z

        result = ActionResult(
            success=True,
            error=None,
            details={"step_idx": step_idx},
        )
        world_after = self._make_world_state()

        trace_step = TraceStep(
            world_before=world_before,
            action=action,
            result=result,
            world_after=world_after,
            meta={
                "skill": step_spec.get("skill", "test_move"),
                "plan_step_idx": step_idx,
                "context_id": self.config.context_id,
            },
        )
        return [trace_step]

    # ------------------------------------------------
    # Optional critic API (not used in basic test)
    # ------------------------------------------------

    @property
    def critic_model(self) -> None:
        """
        No critic in the basic fake runtime; AgentLoop should handle None.
        """
        return None

    # ------------------------------------------------
    # Helpers
    # ------------------------------------------------

    def _make_world_state(self) -> WorldState:
        """
        Construct a minimal WorldState that satisfies TraceStep usage.
        """
        self._tick += 1
        return WorldState(
            tick=self._tick,
            position=dict(self._position),
            dimension="overworld",
            inventory=[],
            nearby_entities=[],
            blocks_of_interest=[],
            tech_state={},
            context={},
        )

