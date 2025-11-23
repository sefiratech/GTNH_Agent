# high-level integration tests placeholder
# tests/test_architecture_integration.py

from __future__ import annotations

from typing import Any, Dict, List

from spec.types import WorldState, Action, ActionResult, Observation
from spec.bot_core import BotCore
from spec.skills import Skill, SkillRegistry
from spec.llm import PlannerModel, CriticModel
from spec.agent_loop import AgentLoop


class FakeBotCore(BotCore):
    def __init__(self) -> None:
        self._tick = 0
        self._pos: Dict[str, float] = {"x": 0.0, "y": 64.0, "z": 0.0}

    def connect(self) -> None:
        pass

    def disconnect(self) -> None:
        pass

    def get_world_state(self) -> WorldState:
        return WorldState(
            tick=self._tick,
            position=dict(self._pos),
            dimension="overworld",
            inventory=[],
            nearby_entities=[],
            blocks_of_interest=[],
            tech_state={},
            context={},
        )

    def execute_action(self, action: Action) -> ActionResult:
        if action.type == "move":
            dx = action.params.get("dx", 0)
            dz = action.params.get("dz", 0)
            self._pos["x"] += dx
            self._pos["z"] += dz
            self._tick += 1
            return ActionResult(
                success=True,
                error=None,
                details={"position": dict(self._pos)},
            )
        return ActionResult(
            success=False,
            error="unsupported_action",
            details={},
        )

    def tick(self) -> None:
        self._tick += 1


class FakeSkill(Skill):
    @property
    def name(self) -> str:
        return "move_forward"

    def describe(self) -> Dict[str, Any]:
        return {
            "description": "Move the agent forward by one block.",
            "params": {},
            "preconditions": [],
            "effects": ["position changes"],
        }

    def suggest_actions(self, world: WorldState, params: Dict[str, Any]) -> List[Action]:
        return [Action(type="move", params={"dx": 1, "dz": 0})]


class FakeSkillRegistry(SkillRegistry):
    def list_skills(self) -> List[str]:
        return ["move_forward"]

    def get_skill(self, name: str) -> Skill:
        if name != "move_forward":
            raise KeyError(name)
        return FakeSkill()

    def describe_all(self) -> Dict[str, Dict[str, Any]]:
        s = FakeSkill()
        return {s.name: s.describe()}


class FakePlanner(PlannerModel):
    def plan(
        self,
        observation: Observation,
        goal: str,
        skill_descriptions: Dict[str, Dict[str, Any]],
        constraints: Dict[str, Any],
    ) -> Dict[str, Any]:
        return {"steps": [{"skill": "move_forward", "params": {}}]}


class FakeCritic(CriticModel):
    def evaluate_plan(
        self,
        observation: Observation,
        plan: Dict[str, Any],
        virtue_scores: Dict[str, float],
    ) -> Dict[str, Any]:
        return {"ok": True, "score": 1.0}


class SimpleAgentLoop(AgentLoop):
    def __init__(
        self,
        bot: BotCore,
        skills: SkillRegistry,
        planner: PlannerModel,
        critic: CriticModel,
    ) -> None:
        self.bot = bot
        self.skills = skills
        self.planner = planner
        self.critic = critic
        self._goal = "test"
        self._last_plan: Dict[str, Any] = {}
        self._last_critique: Dict[str, Any] = {}

    def set_goal(self, goal: str, context: Dict[str, Any]) -> None:
        self._goal = goal

    def get_status(self) -> Dict[str, Any]:
        return {
            "goal": self._goal,
            "plan": self._last_plan,
            "critique": self._last_critique,
        }

    def step(self) -> None:
        world = self.bot.get_world_state()
        obs = Observation(json_payload={"tick": world.tick}, text_summary="test")

        plan = self.planner.plan(
            obs,
            self._goal,
            self.skills.describe_all(),
            constraints={},
        )
        self._last_plan = plan

        # Let the critic look at the plan (even if we ignore the result for now).
        self._last_critique = self.critic.evaluate_plan(
            obs,
            plan,
            virtue_scores={},
        )

        for step in plan["steps"]:
            skill = self.skills.get_skill(step["skill"])
            actions = skill.suggest_actions(world, step.get("params", {}))
            for action in actions:
                self.bot.execute_action(action)


def test_fake_architecture_round_trip():
    bot = FakeBotCore()
    skills = FakeSkillRegistry()
    planner = FakePlanner()
    critic = FakeCritic()
    agent = SimpleAgentLoop(bot, skills, planner, critic)

    agent.set_goal("go_somewhere", {})
    agent.step()
    status = agent.get_status()

    assert status["plan"]["steps"][0]["skill"] == "move_forward"
    assert status["critique"]["ok"] is True

