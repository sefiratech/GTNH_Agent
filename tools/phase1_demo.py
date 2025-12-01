# tools/phase1_demo.py

from spec.types import WorldState
from integration.phase1_integration import run_phase1_planning_episode


def main() -> None:
    world = WorldState(label="demo_world")  # however your fake/real world is built
    goal = "Design optimal LV coke oven and boiler layout"
    virtue_context_id = "lv_coke_ovens"

    best_plan, scores = run_phase1_planning_episode(
        world=world,
        goal=goal,
        virtue_context_id=virtue_context_id,
    )

    print("Best plan:", best_plan)
    print("Virtue scores:", scores)


if __name__ == "__main__":
    main()
