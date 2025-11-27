# src/llm_stack/scribe.py

import json
import logging

from spec.llm import ScribeModel as ScribeModelInterface
from .schema import TraceSummaryRequest, TraceSummaryResponse
from .backend import LLMBackend
from .presets import RolePreset


logger = logging.getLogger(__name__)


class ScribeModelImpl(ScribeModelInterface):
    """ScribeModel for summarizing traces and logs."""

    def __init__(self, backend: LLMBackend, preset: RolePreset) -> None:
        self._backend = backend
        self._preset = preset

    def _build_prompt(self, req: TraceSummaryRequest) -> str:
        trace_json = json.dumps(req.trace, indent=2)

        prompt = f"""
You are a scribe for a Minecraft GTNH automation agent.

Summarize the following trace for purpose: "{req.purpose}"

Trace:
{trace_json}

Produce a concise summary plus keywords and tags.

Respond ONLY with valid JSON:
{{
  "summary": "text",
  "keywords": ["...", "..."],
  "suggested_tags": ["...", "..."]
}}
"""
        return prompt.strip()

    def summarize_trace(self, req: TraceSummaryRequest) -> TraceSummaryResponse:
        prompt = self._build_prompt(req)
        logger.debug("ScribeModel prompt: %s", prompt)

        raw = self._backend.generate(
            prompt,
            max_tokens=self._preset.max_tokens,
            temperature=self._preset.temperature,
            stop=self._preset.stop,
            system_prompt=self._preset.system_prompt,
        )
        logger.debug("ScribeModel raw output: %s", raw)

        data = json.loads(raw)

        return TraceSummaryResponse(
            summary=data.get("summary", ""),
            keywords=data.get("keywords", []),
            suggested_tags=data.get("suggested_tags", []),
            raw_text=raw,
        )

