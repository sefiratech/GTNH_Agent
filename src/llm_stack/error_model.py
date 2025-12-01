# path: src/llm_stack/error_model.py

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from spec.llm import ErrorModel as ErrorModelInterface
from .schema import ErrorContext, ErrorAnalysis
from .backend import LLMBackend
from .presets import RolePreset
from .json_utils import load_json_or_none
from .log_files import log_llm_call


logger = logging.getLogger(__name__)


class ErrorModelImpl(ErrorModelInterface):
    """ErrorModel backed by the same local LLM, low temperature.

    Responsibilities (Q1-aligned):

      1. Post-execution outcome analysis for plans (agent-level):
         - `evaluate(payload)`:
             payload typically contains:
                 {
                     "plan": {...},
                     "observation": {...},
                     "virtue_scores": {...},   # optional
                     "trace": {...},           # optional episode summary
                     "context": {...},         # optional metadata
                 }
             returns a dict with:
                 {
                     "failure_type": Optional[str],
                     "severity": Optional[str],
                     "fix_suggestions": List[str],
                     "notes": str,
                     "raw_text": str,
                     "error": Optional[str],   # if parsing failed
                 }

         - `call_error_model(episode_trace, plan)`:
             convenience wrapper that assembles a payload and calls `evaluate`.

      2. LLM-call error analysis (legacy / infra-level):
         - `analyze_failure(ctx: ErrorContext) -> ErrorAnalysis`
             unchanged in spirit:
                 {
                   "classification": "short_label",
                   "summary": "short explanation",
                   "suggested_fix": {...},
                   "retry_advised": bool,
                   "raw_text": str,
                 }

    The two paths are deliberately separate:
      - AgentLoop uses `evaluate` / `call_error_model`.
      - Lower-level LLM infra may still use `analyze_failure`.
    """

    def __init__(self, backend: LLMBackend, preset: RolePreset) -> None:
        self._backend = backend
        self._preset = preset

    # -------------------------------------------------------------------------
    # Q1: Agent-level outcome evaluation
    # -------------------------------------------------------------------------

    def call_error_model(
        self,
        episode_trace: Any,
        plan: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Convenience wrapper for agent-level outcome evaluation.

        Args:
            episode_trace:
                Typically a PlanTrace or a dict-like representation of an
                episode. We only make best-effort use of it here; the caller
                can pre-encode a more detailed payload if desired.

            plan:
                Structured plan dict (same schema as CriticModel sees).

        Returns:
            ErrorModel-style result dict with keys:
                - failure_type
                - severity
                - fix_suggestions
                - notes
                - raw_text
                - error (optional)
        """
        # Best-effort extraction of observation / trace info.
        if isinstance(episode_trace, dict):
            observation = episode_trace.get("observation", {})
            trace_info = episode_trace
        else:
            # Try some common attributes; if they don't exist, fall back.
            observation = getattr(episode_trace, "observation", None)
            if observation is None and hasattr(episode_trace, "planner_payload"):
                # Observations often encoded as json_payload in trace.
                observation = {
                    "json_payload": getattr(episode_trace, "planner_payload", None),
                    "text_summary": getattr(episode_trace, "text_summary", ""),
                }
            trace_info = getattr(episode_trace, "to_serializable", None)
            if callable(trace_info):
                trace_info = trace_info()
            elif not isinstance(trace_info, dict):
                trace_info = {"raw": str(episode_trace)}

        payload: Dict[str, Any] = {
            "plan": plan,
            "observation": observation or {},
            "trace": trace_info,
            "context": {"source": "agent_outcome"},
        }
        return self.evaluate(payload)

    def _build_outcome_prompt(self, payload: Dict[str, Any]) -> str:
        plan = payload.get("plan", {}) or {}
        observation = payload.get("observation", {}) or {}
        trace = payload.get("trace", {}) or {}
        virtue_scores = payload.get("virtue_scores", {}) or {}
        context = payload.get("context", {}) or {}

        plan_json = json.dumps(plan, indent=2)
        obs_json = json.dumps(observation, indent=2)
        trace_json = json.dumps(trace, indent=2)
        virtues_json = json.dumps(virtue_scores, indent=2)
        ctx_json = json.dumps(context, indent=2)

        prompt = f"""
You are a failure analyst for a Minecraft GTNH automation agent.

The plan has ALREADY BEEN EXECUTED. Your job is to analyze what went wrong
(or what nearly went wrong) and provide structured guidance for improving
future plans and skills.

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
    - null if execution was successful enough
    - "execution_error" if steps failed at runtime (blocked, exceptions, etc.)
    - "missing_prereq" if runtime lacked required resources/tech
    - "resource_exhaustion" if resources ran out or would soon
    - "safety_risk" if it caused or risked serious harm (deaths, explosions)
    - "plan_mismatch" if execution diverged from plan in a problematic way
- severity:
    - "low": minor issues, easy to patch
    - "medium": noticeable failures, some impact
    - "high": serious or repeated failure; plan/skill needs revision
- fix_suggestions:
    - 1–4 specific, actionable changes (e.g. new prerequisites, checks,
      or alternative strategies)
- notes:
    - brief explanation of what happened and why

Plan that was executed:
{plan_json}

Observation / world summary at planning time:
{obs_json}

Execution trace (summarized):
{trace_json}

Virtue scores (if any, higher is better):
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
        logger.debug("ErrorModel extracted JSON candidate: %s", candidate)
        return candidate

    def evaluate(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Agent-level outcome evaluation entrypoint.

        This mirrors CriticModel.evaluate's contract where possible, but is
        intended for post-execution analysis.

        Args:
            payload:
                Dict containing at least:
                    "plan": {...},
                    "observation": {...},
                and optionally:
                    "trace": {...},
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
        prompt = self._build_outcome_prompt(payload)
        logger.debug("ErrorModel evaluate prompt: %s", prompt)

        max_tokens = self._preset.max_tokens
        temperature = min(self._preset.temperature, 0.3)

        raw = self._backend.generate(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=self._preset.stop,
            system_prompt=self._preset.system_prompt,
        )
        logger.debug("ErrorModel evaluate raw output: %s", raw)

        log_llm_call(
            role="error_model",
            operation="evaluate_outcome",
            prompt=prompt,
            raw_response=raw,
            extra={
                "has_plan": "plan" in payload,
                "has_observation": "observation" in payload,
            },
        )

        candidate = self._extract_json_object(raw)
        data, err = load_json_or_none(candidate, context="ErrorModel.evaluate")

        if data is None:
            logger.error("ErrorModel JSON decode error: %s", err)
            return {
                "failure_type": None,
                "severity": None,
                "fix_suggestions": [],
                "notes": "json_parse_error",
                "raw_text": raw,
                "error": err,
            }

        failure_type = data.get("failure_type")
        severity = data.get("severity")
        fix_suggestions = data.get("fix_suggestions", [])
        notes = data.get("notes", "")

        if not isinstance(fix_suggestions, list):
            fix_suggestions = [str(fix_suggestions)]

        return {
            "failure_type": failure_type,
            "severity": severity,
            "fix_suggestions": [str(s) for s in fix_suggestions],
            "notes": str(notes),
            "raw_text": raw,
        }

    # -------------------------------------------------------------------------
    # Legacy LLM-call failure analysis (infra-level)
    # -------------------------------------------------------------------------

    def _build_prompt(self, ctx: ErrorContext) -> str:
        """Legacy prompt for analyzing LLM call failures.

        This is kept separate from the agent-level outcome evaluation.
        """
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
        """Legacy API: analyze a single LLM call failure.

        This is kept for compatibility with tooling that inspects backend
        errors and wants a richer ErrorAnalysis object.
        """
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

        # File-level log for replay / forensics
        log_llm_call(
            role="error_model",
            operation="analyze_failure",
            prompt=prompt,
            raw_response=raw,
            extra={
                "error_type": ctx.error_type,
                "source_role": ctx.role,
                "source_operation": ctx.operation,
            },
        )

        candidate = self._extract_json_object(raw)
        data, err = load_json_or_none(candidate, context="ErrorModel.analyze_failure")
        if data is None:
            # If the error model itself returns garbage, we still give callers
            # a structured ErrorAnalysis describing that situation.
            logger.error("ErrorModel JSON decode error: %s", err)
            return ErrorAnalysis(
                classification="json_parse_error",
                summary=f"Failed to parse ErrorModel JSON: {err}",
                suggested_fix={},
                retry_advised=False,
                raw_text=raw,
            )

        return ErrorAnalysis(
            classification=data.get("classification", "unknown"),
            summary=data.get("summary", ""),
            suggested_fix=data.get("suggested_fix", {}),
            retry_advised=bool(data.get("retry_advised", False)),
            raw_text=raw,
        )

