# tests/test_agent_loop_stub.py

from agent.bootstrap import build_agent_runtime
from agent.loop import AgentLoop, AgentLoopConfig
from agent.experience import ExperienceBuffer, Experience
from observation.trace_schema import PlanTrace


def test_agent_loop_run_episode_minimal():
    runtime = build_agent_runtime()
    buffer = ExperienceBuffer()
    loop = AgentLoop(
        runtime=runtime,
        experience_buffer=buffer,
        config=AgentLoopConfig(
            enable_critic=True,
            store_experiences=True,
            max_planner_calls=1,
        ),
    )

    result = loop.run_episode(episode_id=1)

    # Basic structural checks
    assert isinstance(result.plan, dict)
    assert isinstance(result.trace, PlanTrace)
    # Critic may be a dummy; just assert it's either dict or None
    assert result.critic_result is None or isinstance(result.critic_result, dict)

    # Experience logging should have recorded exactly one entry
    assert len(buffer) == 1
    exp = buffer.last()
    assert isinstance(exp, Experience)
    assert exp.episode_id == 1
    assert isinstance(exp.trace, PlanTrace)

