# tests for llm_stack with fake backend
# tests/test_llm_stack_fake_backend.py

from typing import List, Optional

import pytest

try:
    import llama_cpp  # type: ignore
except ImportError:
    pytest.skip("llama_cpp not installed; skipping LLM backend tests in CI", allow_module_level=True)

from llm_stack.backend import LLMBackend
from llm_stack.presets import RolePreset
from llm_stack.plan_code import PlanCodeModelImpl
from spec.types import Observation


class FakeBackend(LLMBackend):
    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int,
        temperature: float,
        stop: Optional[List[str]] = None,
        system_prompt: Optional[str] = None,
    ) -> str:
        # Return a fixed JSON plan regardless of input
        return '{"steps": [{"skill": "test_skill", "params": {"x": 1}}], "notes": "ok"}'


def test_plan_code_planner_with_fake_backend():
    backend = FakeBackend()
    preset = RolePreset(
        name="plan_code",
        temperature=0.2,
        max_tokens=512,
    )
    model = PlanCodeModelImpl(backend, preset)

    obs = Observation(
        json_payload={"foo": "bar"},
        text_summary="test observation",
    )

    plan = model.plan(
        observation=obs,
        goal="do something testy",
        skill_descriptions={"test_skill": {"description": "dummy"}},
        constraints={},
    )

    assert plan["steps"][0]["skill"] == "test_skill"
    assert plan["steps"][0]["params"]["x"] == 1

