# tests for AgentLoopV1 with fakes
# tests/test_agent_loop_v1.py

from __future__ import annotations

from agent.loop import AgentLoop, AgentLoopConfig
from agent.experience import ExperienceBuffer
from tests.fakes.fake_runtime import FakeAgentRuntime


def test_agent_loop_episode_runs_end_to_end() -> None:
    """
    Basic integration test for M8:

      - Uses FakeAgentRuntime (no real Minecraft, LLM, or skills)
      - Verifies that:
          * planner was called
          * at least one TraceStep was produced
          * an Experience was stored in the buffer
    """
    runtime = FakeAgentRuntime()
    buffer = ExperienceBuffer()
    loop = AgentLoop(
        runtime=runtime,
        config=AgentLoopConfig(),
        experience_buffer=buffer,
    )

    result = loop.run_episode(episode_id=1)

    # Planner produced a plan
    assert isinstance(result.plan, dict)
    assert "steps" in result.plan
    assert len(result.plan["steps"]) > 0

    # Execution produced at least one trace step
    assert len(result.trace.steps) > 0

    # Experience was stored in the buffer
    assert len(buffer) == 1

    xp = buffer.last()
    assert xp is not None
    assert xp.episode_id == 1
    # Sanity check some metadata
    assert xp.env_profile_name == runtime.config.profile_name
    assert xp.context_id == result.trace.context_id

