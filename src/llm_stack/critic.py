# CriticModel wrapper logic
# path: src/llm_stack/critic.py

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from spec.llm import CriticModel
from .backend import LLMBackend
from .presets import RolePreset
from .json_utils import load_json_or_none
from .log_files import log_llm_call

logger = logging.getLogger(__name__)


class CriticModelImpl(CriticModel):
    """CriticModel backed by a single local LLM with role presets.

    Responsibilities:
        - Pre-execution critique of candidate plans.
        - Post-execution diagnosis of failed or suboptimal episodes
          (same schema, different context).

    Contract:
        - `evaluate(payload)` is the generic entrypoint used by M8:
            payload should contain:
                {
                    "plan": {...},             # structured plan dict
                    "observation": {...},      # world view / trace summary
                    "virtue_scores": {...},    # optional, may be empty
                    "context": {...},          # optional metadata
                }

          It returns a dict:
                {
                    "failure_type": Optional[str],
                    "severity": Optional[str],
                    "fix_suggestions": List[str],
                    "notes": str,
                }
          plus optional fields:
                "raw_text", "error"

        - `call_critic(plan, world_summary, virtues_hint)` is a convenience
          wrapper for higher-level callers that don't want to assemble the
          payload by hand.
    """

    def __init__(self, backend: LLMBackend, preset: RolePreset) -> None:
        self._backend = backend
        self._preset = preset

    # -------------------------------------------------------------------------
    # High-level helper
    # -------------------------------------------------------------------------

    def call_critic(
        self,
        plan: Dict[str, Any],
        world_summary: Dict[str, Any],
        virtues_hint: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """Convenience wrapper to critique a plan with world / virtue context.

        Args:
            plan:
                Structured plan dict, typically:
                    {"steps": [...], "notes": "...", ...}

            world_summary:
                Dict with keys like:
                    "observation": {...},  # text + json summary
                    "context": {...},      # optional metadata

            virtues_hint:
                Optional virtue score dict that can help the critic reason
                about tradeoffs, e.g. {"prudence": 0.7, "courage": 0.4, ...}.

        Returns:
            Critic response dict as described in the class docstring.
        """
        payload: Dict[str, Any] = {
            "plan": plan,
            "observation": world_summary.get("observation", {}),
            "context": world_summary.get("context", {}),
        }
        if virtues_hint is not None:
            payload["virtue_scores"] = virtues_hint

        return self.evaluate(payload)

    # -------------------------------------------------------------------------
    # Core API used by AgentLoop / encode_for_critic
    # -------------------------------------------------------------------------

    def _build_critic_prompt(self, payload: Dict[str, Any]) -> str:
        plan = payload.get("plan", {}) or {}
        observation = payload.get("observation", {}) or {}
        virtue_scores = payload.get("virtue_scores", {}) or {}
        context = payload.get("context", {}) or {}

        plan_json = json.dumps(plan, indent=2)
        obs_json = json.dumps(observation, indent=2)
        virtues_json = json.dumps(virtue_scores, indent=2)
        ctx_json = json.dumps(context, indent=2)

        prompt = f"""
You are a safety and quality critic for a Minecraft GTNH automation agent.

Your ONLY job now is to evaluate a candidate plan and provide structured
feedback on risks, failure modes, and possible improvements.

You MUST return a single JSON object and nothing else.
The FIRST non-whitespace character in your reply MUST be '{{'.
The LAST non-whitespace character in your reply MUST be '}}'.

The JSON MUST have this structure:

{{
  "failure_type": "str or null",
  "severity": "low" | "medium" | "high" | null,
  "fix_suggestions": ["list", "of", "concrete", "improvements"],
  "notes": "free-form explanation"
}}

Use these semantics:
- failure_type:
    - null if the plan looks acceptable
    - "plan_quality" if steps are unclear, missing, or contradictory
    - "missing_prereq" if prerequisites are absent
    - "resource_risk" if it likely fails due to resources
    - "safety_risk" if it risks killing the player or corrupting the world
- severity:
    - "low": minor issues, plan is probably fine
    - "medium": significant risk or inefficiency
    - "high": likely to fail or cause serious problems
- fix_suggestions:
    - 1–4 specific, actionable improvements (or empty list if none)
- notes:
    - brief explanation of your reasoning

Plan under review:
{plan_json}

Observation / world summary:
{obs_json}

Virtue scores (higher is better):
{virtues_json}

Context:
{ctx_json}
"""
        return prompt.strip()

    def _extract_json_object(self, raw: str) -> str:
        """
        Try to salvage a JSON object from a noisy LLM reply.

        Strategy:
        - Strip whitespace
        - Find first '{' and last '}' and take that slice
        - If that fails, just return the original raw text
        """
        if not raw:
            return raw

        trimmed = raw.strip()
        first = trimmed.find("{")
        last = trimmed.rfind("}")

        if first == -1 or last == -1 or last <= first:
            return trimmed

        candidate = trimmed[first : last + 1]
        logger.debug("CriticModel extracted JSON candidate: %s", candidate)
        return candidate

    def evaluate(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a plan + context payload and return structured feedback.

        This is the entrypoint expected by AgentLoop._maybe_run_critic:

            critic_payload = encode_for_critic(trace)
            result = critic_model.evaluate(critic_payload)

        Args:
            payload:
                Dict containing at least:
                    "plan": {...},
                    "observation": {...},
                and optionally:
                    "virtue_scores": {...},
                    "context": {...}.

        Returns:
            Dict with keys:
                - "failure_type": Optional[str]
                - "severity": Optional[str]
                - "fix_suggestions": List[str]
                - "notes": str
                - "raw_text": raw LLM output
              plus optional:
                - "error": str, if parsing failed
        """
        prompt = self._build_critic_prompt(payload)
        logger.debug("CriticModel evaluate prompt: %s", prompt)

        # Be slightly conservative on temperature for critique.
        max_tokens = self._preset.max_tokens
        temperature = min(self._preset.temperature, 0.4)

        raw = self._backend.generate(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=self._preset.stop,
            system_prompt=self._preset.system_prompt,
        )
        logger.debug("CriticModel evaluate raw output: %s", raw)

        # File-level log for replay / forensics
        log_llm_call(
            role="critic",
            operation="evaluate",
            prompt=prompt,
            raw_response=raw,
            extra={
                "has_plan": "plan" in payload,
                "has_observation": "observation" in payload,
            },
        )

        candidate = self._extract_json_object(raw)
        data, err = load_json_or_none(candidate, context="CriticModel.evaluate")

        if data is None:
            logger.error("CriticModel JSON decode error: %s", err)
            # Return a best-effort fallback result so the caller can
            # still treat this like a CriticResponse.
            return {
                "failure_type": None,
                "severity": None,
                "fix_suggestions": [],
                "notes": "json_parse_error",
                "raw_text": raw,
                "error": err,
            }

        # Normalize fields and fill in defaults
        failure_type = data.get("failure_type")
        severity = data.get("severity")
        fix_suggestions = data.get("fix_suggestions", [])
        notes = data.get("notes", "")

        if not isinstance(fix_suggestions, list):
            fix_suggestions = [str(fix_suggestions)]

        result: Dict[str, Any] = {
            "failure_type": failure_type,
            "severity": severity,
            "fix_suggestions": [str(s) for s in fix_suggestions],
            "notes": str(notes),
            "raw_text": raw,
        }
        return result

