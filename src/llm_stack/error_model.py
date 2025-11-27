# src/llm_stack/error_model.py

import json
import logging

from spec.llm import ErrorModel as ErrorModelInterface
from .schema import ErrorContext, ErrorAnalysis
from .backend import LLMBackend
from .presets import RolePreset


logger = logging.getLogger(__name__)


class ErrorModelImpl(ErrorModelInterface):
    """ErrorModel backed by the same local LLM, low temperature."""

    def __init__(self, backend: LLMBackend, preset: RolePreset) -> None:
        self._backend = backend
        self._preset = preset

    def _build_prompt(self, ctx: ErrorContext) -> str:
        ctx_json = json.dumps(
            {
                "role": ctx.role,
                "operation": ctx.operation,
                "prompt": ctx.prompt,
                "raw_response": ctx.raw_response,
                "error_type": ctx.error_type,
                "metadata": ctx.metadata,
            },
            indent=2,
        )

        prompt = f"""
You are an error analyst for an LLM-based GTNH agent.

Here is the context of a failure:
{ctx_json}

Explain:
1. What likely went wrong (classification).
2. A brief human-readable summary.
3. A JSON object suggesting how to fix or retry.

Respond ONLY with valid JSON:
{{
  "classification": "short_label",
  "summary": "short explanation",
  "suggested_fix": {{ ... }},
  "retry_advised": true or false
}}
"""
        return prompt.strip()

    def analyze_failure(self, ctx: ErrorContext) -> ErrorAnalysis:
        prompt = self._build_prompt(ctx)
        logger.debug("ErrorModel prompt: %s", prompt)

        raw = self._backend.generate(
            prompt,
            max_tokens=self._preset.max_tokens,
            temperature=self._preset.temperature,
            stop=self._preset.stop,
            system_prompt=self._preset.system_prompt,
        )
        logger.debug("ErrorModel raw output: %s", raw)

        data = json.loads(raw)

        return ErrorAnalysis(
            classification=data.get("classification", "unknown"),
            summary=data.get("summary", ""),
            suggested_fix=data.get("suggested_fix", {}),
            retry_advised=bool(data.get("retry_advised", False)),
            raw_text=raw,
        )

