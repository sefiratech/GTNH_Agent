# tests/test_p0_p1_env_bridge.py

from integration.adapters.m0_env_to_world import world_from_env_profile
from integration.phase1_integration import run_phase1_planning_episode


class DummyEnvProfile:
    """
    Minimal EnvProfile-like object for Phase 0 → Phase 1 bridge test.

    We deliberately keep this tiny so the test does NOT depend
    on the concrete EnvProfile implementation.
    """

    def __init__(self) -> None:
        self.name = "test_env"
        self.modpack_id = "gtnh-2.8.1"
        self.gtnh_version = "2.8.1"
        self.world_seed = 123456789
        self.source_path = "/fake/path/env.yaml"


def test_p0_env_to_p1_planning_bridge():
    profile = DummyEnvProfile()

    # Adapter: Phase 0-like profile -> WorldState
    world = world_from_env_profile(profile)

    # If WorldState doesn't have to_summary_dict yet, patch a minimal one
    if not hasattr(world, "to_summary_dict"):

        def _summary():
            return {
                "env_name": "test_env",
                "modpack": "gtnh-2.8.1",
            }

        setattr(world, "to_summary_dict", _summary)

    # Phase 1: run planning episode with fake backends
    best_plan, scores = run_phase1_planning_episode(
        world=world,
        goal="sanity check planning goal",
        virtue_context_id="lv_resources",
    )

    assert isinstance(best_plan, dict)
    assert isinstance(scores, dict)
    assert "id" in best_plan  # fake planner uses plan ids

