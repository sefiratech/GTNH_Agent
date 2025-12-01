# path: tests/test_curriculum_engine_basic.py

from curriculum.engine import CurriculumEngine
from curriculum.schema import (
    CurriculumConfig,
    PhaseConfig,
    PhaseTechTargets,
    PhaseGoal,
)
from spec.agent_loop import AgentGoal


class DummyTechState:
    """
    Minimal TechState stub for testing.

    CurriculumEngine only needs:
      - .active : str
      - .unlocked : list[str]
    """

    def __init__(self, active: str, unlocked: list[str]) -> None:
        self.active = active
        self.unlocked = unlocked


class DummyWorld:
    """
    Minimal WorldState stub for testing.

    CurriculumEngine only ever does:
      getattr(world, "context", {})["machines"/"inventory"/"virtue_scores"]

    So a single .context dict is enough.
    """

    def __init__(self) -> None:
        self.context = {}


def test_curriculum_engine_selects_goal_by_tech_state():
    # Build fake minimal curriculum
    phase1 = PhaseConfig(
        id="p1",
        name="Stone Age",
        tech_targets=PhaseTechTargets(required_active="stone_age"),
        goals=[PhaseGoal(id="g1", description="Do stone stuff")],
    )

    phase2 = PhaseConfig(
        id="p2",
        name="Steam Age",
        tech_targets=PhaseTechTargets(required_active="steam_age"),
        goals=[PhaseGoal(id="g2", description="Do steam stuff")],
    )

    cfg = CurriculumConfig(
        id="demo",
        name="demo",
        description="demo",
    )
    cfg.phases = [phase1, phase2]

    engine = CurriculumEngine(cfg)

    # Fake TechState + fake WorldState
    tech = DummyTechState(active="stone_age", unlocked=[])
    world = DummyWorld()

    # Call next_goal
    goal = engine.next_goal(
        tech_state=tech,
        world=world,
        experience_summary=None,
    )

    # Assertions (the important part)
    assert goal is not None
    assert isinstance(goal, AgentGoal)
    # ID is derived in engine as f"{phase.id}:{goal.id}"
    assert goal.id == "p1:g1"
    assert goal.text == "Do stone stuff"
    assert goal.phase == "p1"
    assert goal.source == "curriculum"

