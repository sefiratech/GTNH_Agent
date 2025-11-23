# tests/test_phase0_runtime.py

from env.loader import load_environment
from app.runtime import create_phase0_agent


def test_phase0_bootstrap_runs():
    env = load_environment()
    bot, agent = create_phase0_agent(env)

    bot.connect()
    agent.set_goal("test_goal", context={})
    bot.tick()
    agent.step()
    status = agent.get_status()
    bot.disconnect()

    assert status["goal"] == "test_goal"
    assert "plan" in status

