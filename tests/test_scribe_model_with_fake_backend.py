# tests/test_scribe_model_with_fake_backend.py

from __future__ import annotations

from typing import List, Optional

import pytest

try:
    import llama_cpp  # type: ignore
except ImportError:
    pytest.skip("llama_cpp not installed; skipping LLM backend tests in CI", allow_module_level=True)


from llm_stack.backend import LLMBackend
from llm_stack.presets import RolePreset
from llm_stack.scribe import ScribeModelImpl
from llm_stack.schema import TraceSummaryRequest


class FakeBackend(LLMBackend):
    """Backend that always returns a fixed JSON trace summary."""

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int,
        temperature: float,
        stop: Optional[List[str]] = None,
        system_prompt: Optional[str] = None,
    ) -> str:
        # Ignore the prompt, just return stable JSON
        return """
{
  "summary": "Maintained continuous charcoal and creosote production using coke ovens and a storage tank.",
  "keywords": ["charcoal", "creosote", "coke_ovens", "storage_tank"],
  "suggested_tags": ["automation", "production", "GTNH"]
}
""".strip()


def test_scribe_model_parses_fixed_json() -> None:
    backend = FakeBackend()
    preset = RolePreset(
        name="scribe",
        temperature=0.3,
        max_tokens=256,
    )
    model = ScribeModelImpl(backend, preset)

    trace = {
        "goal": "dummy",
        "plan": {},
        "actions": [],
        "outcome": {},
    }
    req = TraceSummaryRequest(
        trace=trace,
        purpose="context_chunk",
    )

    resp = model.summarize_trace(req)

    # Field sanity
    assert isinstance(resp.summary, str)
    assert "charcoal" in resp.summary

    assert isinstance(resp.keywords, list)
    assert "charcoal" in resp.keywords
    assert "creosote" in resp.keywords

    assert isinstance(resp.suggested_tags, list)
    assert "automation" in resp.suggested_tags

    # raw_text should mirror the JSON
    assert isinstance(resp.raw_text, str)
    assert resp.raw_text.strip().startswith("{")

