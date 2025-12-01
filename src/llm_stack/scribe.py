from __future__ import annotations

import json
import logging
from typing import Any, Dict

from spec.llm import (
    ScribeModel as ScribeModelInterface,
    ScribeRequest,
    ScribeResponse,
)
from .schema import TraceSummaryRequest, TraceSummaryResponse
from .backend import LLMBackend
from .presets import RolePreset
from .json_utils import load_json_or_none
from .log_files import log_llm_call


logger = logging.getLogger(__name__)


class ScribeModelImpl(ScribeModelInterface):
    """ScribeModel for summarizing traces and logs."""

    def __init__(self, backend: LLMBackend, preset: RolePreset) -> None:
        self._backend = backend
        self._preset = preset

    # ------------------------------------------------------------------
    # Protocol entrypoint (spec.llm.ScribeModel)
    # ------------------------------------------------------------------

    def summarize(self, req: ScribeRequest) -> ScribeResponse:
        """
        Protocol-compliant wrapper around the lower-level trace API.

        This keeps the tests (which use TraceSummaryRequest/Response)
        happy while giving higher-level code a stable ScribeRequest API.
        """
        ts_req = TraceSummaryRequest(trace=req.trace, purpose=req.purpose)
        ts_resp = self.summarize_trace(ts_req)

        return ScribeResponse(
            summary=ts_resp.summary,
            keywords=getattr(ts_resp, "keywords", []),
            suggested_tags=getattr(ts_resp, "suggested_tags", []),
            raw_text=getattr(ts_resp, "raw_text", ""),
        )

    # ------------------------------------------------------------------
    # Trace-level API used by tests
    # ------------------------------------------------------------------

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

        # File-level log for replay / forensics
        log_llm_call(
            role="scribe",
            operation="summarize_trace",
            prompt=prompt,
            raw_response=raw,
            extra={
                "purpose": req.purpose,
            },
        )

        data, err = load_json_or_none(raw, context="ScribeModel.summarize_trace")
        if data is None:
            # Again, don't hard-crash; give a structured, degraded response.
            logger.error("ScribeModel JSON decode error: %s", err)
            return TraceSummaryResponse(
                summary="json_parse_error",
                keywords=[],
                suggested_tags=[],
                raw_text=raw,
            )

        return TraceSummaryResponse(
            summary=data.get("summary", ""),
            keywords=data.get("keywords", []),
            suggested_tags=data.get("suggested_tags", []),
            raw_text=raw,
        )


# ---------------------------------------------------------------------------
# Scribe helper for M10 experience memory
# ---------------------------------------------------------------------------

def _get_attr_or_key(obj: Any, name: str, default: Any = None) -> Any:
    """
    Best-effort accessor for either attribute or dict key.

    This keeps the helper usable with both PlanTrace-like objects and
    plain dicts.
    """
    if obj is None:
        return default
    if hasattr(obj, name):
        return getattr(obj, name, default)
    if isinstance(obj, dict):
        return obj.get(name, default)
    return default


def summarize_episode_for_memory(trace: Any) -> Dict[str, Any]:
    """
    Build a compact, memory-oriented summary for an episode.

    Input:
        trace:
            A PlanTrace-like object or a dict with at least:
                - plan: {goal_id, goal_text, ...}
                - steps: list[...]
                - tech_state: object or dict (optional)

    Output dict:
        {
          "problem_signature": {...},
          "lessons": "short text",
          "tags": ["...", ...]   # optional, may be empty
        }

    This helper is intentionally lightweight and deterministic so it can
    be used by the Experience builder (Q1.4) without depending on an LLM
    at call-site. Higher-level code can still enrich / override this with
    ScribeModelImpl if desired.
    """
    # Normalize plan
    plan = _get_attr_or_key(trace, "plan", {}) or {}
    if not isinstance(plan, dict):
        plan = {}

    goal_id = plan.get("goal_id") or plan.get("id") or ""
    goal_text = plan.get("goal_text") or plan.get("goal") or ""

    # Normalize tech_state (object or dict)
    tech_state = _get_attr_or_key(trace, "tech_state", None)
    tech_active = None
    tech_stage = None

    if tech_state is not None:
        # Dataclass / object with attributes
        tech_active = _get_attr_or_key(tech_state, "active", None)
        tech_stage = _get_attr_or_key(tech_state, "stage", None)
        if isinstance(tech_state, dict):
            tech_active = tech_state.get("active", tech_active)
            tech_stage = tech_state.get("stage", tech_stage)

    # Step count + coarse success heuristic
    steps = _get_attr_or_key(trace, "steps", []) or []
    num_steps = len(steps)

    success = True
    if steps:
        # Try to look at last step.result.success if present
        last = steps[-1]
        result_obj = _get_attr_or_key(last, "result", None)
        if result_obj is None and isinstance(last, dict):
            result_obj = last.get("result")
        if result_obj is not None:
            success = bool(_get_attr_or_key(result_obj, "success", True))

    # Problem signature: stable-ish key for similarity search
    problem_signature: Dict[str, Any] = {
        "goal_id": goal_id,
        "goal_text": goal_text,
        "tech_active": tech_active,
        "tech_stage": tech_stage,
        "num_steps": num_steps,
        "success": success,
    }

    # Lessons: short, human/LLM-facing description
    if goal_text:
        base_goal = goal_text
    elif goal_id:
        base_goal = f"goal_id={goal_id}"
    else:
        base_goal = "unspecified goal"

    status_word = "succeeded" if success else "failed"
    lessons = f"Episode {status_word} while pursuing {base_goal} at tech_state={tech_active or tech_stage or 'unknown'}."

    # Tags: compact filters for replay / curriculum
    tags: List[str] = []
    if tech_active:
        tags.append(f"tier:{tech_active}")
    if tech_stage:
        tags.append(f"stage:{tech_stage}")
    if goal_text:
        tags.append("has_goal_text")
    if not success:
        tags.append("failure")

    return {
        "problem_signature": problem_signature,
        "lessons": lessons,
        "tags": tags,
    }

