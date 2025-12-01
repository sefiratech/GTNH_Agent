# tests/test_error_model_with_fake_backend.py

from __future__ import annotations

from typing import List, Optional

import pytest

try:
    import llama_cpp  # type: ignore
except ImportError:
    pytest.skip("llama_cpp not installed; skipping LLM backend tests in CI", allow_module_level=True)

from llm_stack.backend import LLMBackend
from llm_stack.presets import RolePreset
from llm_stack.error_model import ErrorModelImpl
from llm_stack.schema import ErrorContext


class FakeBackend(LLMBackend):
    """Backend that always returns a fixed JSON error analysis."""

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int,
        temperature: float,
        stop: Optional[List[str]] = None,
        system_prompt: Optional[str] = None,
    ) -> str:
        # Ignore all inputs and return a well-formed JSON blob
        return """
{
  "classification": "json_syntax_error",
  "summary": "The JSON response was truncated and missing a closing brace.",
  "suggested_fix": {
    "action": "retry",
    "note": "Ask the model to respond with shorter JSON or reduce context."
  },
  "retry_advised": true
}
""".strip()


def test_error_model_parses_fixed_json() -> None:
    backend = FakeBackend()
    preset = RolePreset(
        name="error",
        temperature=0.0,
        max_tokens=256,
    )
    model = ErrorModelImpl(backend, preset)

    ctx = ErrorContext(
        role="plan_code",
        operation="plan",
        prompt="dummy prompt",
        raw_response="dummy raw",
        error_type="json_decode_error",
        metadata={"test": True},
    )

    analysis = model.analyze_failure(ctx)

    # Basic type/field sanity
    assert analysis.classification == "json_syntax_error"
    assert isinstance(analysis.summary, str)
    assert "truncated" in analysis.summary
    assert isinstance(analysis.suggested_fix, dict)
    assert analysis.suggested_fix.get("action") == "retry"
    assert analysis.retry_advised is True
    # raw_text should not be empty
    assert isinstance(analysis.raw_text, str)
    assert analysis.raw_text.strip().startswith("{")

