# tests/test_phase012_bootstrap.py
from agent.bootstrap import build_agent_runtime
from spec.types import Observation

def test_phase012_bootstrap_planner_flow():
    runtime = build_agent_runtime()
    plan = runtime.planner_tick()
    obs = runtime.get_latest_planner_observation()

    assert isinstance(plan, dict)
    assert isinstance(obs, Observation)
    keys = obs.json_payload.keys()
    for k in [
        "tech_state",
        "agent",
        "inventory_summary",
        "machines_summary",
        "nearby_entities",
        "env_summary",
        "craftable_summary",
        "context_id",
        "text_summary",
    ]:
        assert k in keys
