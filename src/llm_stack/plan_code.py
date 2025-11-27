# src/llm_stack/plan_code.py

import json
import logging
from typing import Dict, Any, List

from spec.llm import PlanCodeModel
from spec.types import Observation
from .schema import (
    PlanRequest,
    PlanResponse,
    PlanStep,
    SkillImplRequest,
    SkillImplResponse,
)
from .backend import LLMBackend
from .presets import RolePreset


logger = logging.getLogger(__name__)


class PlanCodeModelImpl(PlanCodeModel):
    """PlanCodeModel backed by a single local LLM with role presets."""

    def __init__(self, backend: LLMBackend, preset: RolePreset) -> None:
        self._backend = backend
        self._preset = preset

    # --- Planning ---

    def _build_plan_prompt(self, req: PlanRequest) -> str:
        skills_json = json.dumps(req.skills, indent=2)
        obs_json = json.dumps(req.observation, indent=2)
        constraints_json = json.dumps(req.constraints, indent=2)

        prompt = f"""
You are the planning and coding brain for a Minecraft GTNH automation agent.

Your ONLY job now is to produce a JSON object describing a plan.
Do NOT write explanations, commentary, Markdown, or any text before or after the JSON.
The FIRST non-whitespace character in your reply MUST be '{{'.
The LAST non-whitespace character in your reply MUST be '}}'.

Keep the plan concise. Prefer 1–4 steps unless constraints require more.

Observation:
{obs_json}

Goal:
{req.goal}

Available skills (with descriptions and parameters):
{skills_json}

Constraints:
{constraints_json}

Return JSON exactly in this structure:
{{
  "steps": [
    {{"skill": "str", "params": {{ ... }} }},
    ...
  ],
  "notes": "optional free-form explanation"
}}
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
            # No obvious JSON object; give the caller the original
            return trimmed

        candidate = trimmed[first : last + 1]
        logger.debug("PlanCodeModel extracted JSON candidate: %s", candidate)
        return candidate

    def plan(
        self,
        observation: Observation,
        goal: str,
        skill_descriptions: Dict[str, Dict[str, Any]],
        constraints: Dict[str, Any],
    ) -> Dict[str, Any]:
        obs_dict = {
            "json_payload": observation.json_payload,
            "text_summary": observation.text_summary,
        }

        req = PlanRequest(
            observation=obs_dict,
            goal=goal,
            skills=skill_descriptions,
            constraints=constraints,
        )
        prompt = self._build_plan_prompt(req)
        logger.debug("PlanCodeModel plan prompt: %s", prompt)

        # IMPORTANT: respect the preset; no hidden clamp here.
        max_tokens = self._preset.max_tokens

        raw = self._backend.generate(
            prompt,
            max_tokens=max_tokens,
            temperature=self._preset.temperature,
            stop=self._preset.stop,
            system_prompt=self._preset.system_prompt,
        )
        logger.debug("PlanCodeModel plan raw output: %s", raw)

        # Try to clean up any extra junk around the JSON
        candidate = self._extract_json_object(raw)

        try:
            data = json.loads(candidate)
        except json.JSONDecodeError as e:
            # Don't hard-crash here; surface the raw text so higher layers
            # (or ErrorModel) can decide what to do.
            logger.error("PlanCodeModel JSON decode error: %s", e)
            return {
                "steps": [],
                "notes": "json_parse_error",
                "raw_text": raw,
            }

        steps = [
            PlanStep(skill=step["skill"], params=step.get("params", {}))
            for step in data.get("steps", [])
        ]
        resp = PlanResponse(
            steps=steps,
            notes=data.get("notes", ""),
            raw_text=raw,
        )
        return {
            "steps": [s.__dict__ for s in resp.steps],
            "notes": resp.notes,
            "raw_text": resp.raw_text,
        }

    # --- Codegen ---

    def _build_code_prompt(self, req: SkillImplRequest) -> str:
        name = req.skill_spec.get("name", "unknown_skill")
        description = req.skill_spec.get("description", "")
        params = req.skill_spec.get("params", {})

        examples_text = ""
        for ex in req.examples:
            examples_text += f"\nExample trace:\n{json.dumps(ex, indent=2)}\n"

        prompt = f"""
You are generating Python code for a skill in a Minecraft GTNH agent.

Skill name: {name}
Description: {description}
Parameters: {json.dumps(params, indent=2)}
{examples_text}

Write a Python function body that, given a WorldState and params dict,
returns a list of Action objects to achieve the skill.

Return ONLY Python code, without backticks or explanations.
"""
        return prompt.strip()

    def propose_skill_implementation(
        self,
        skill_spec: Dict[str, Any],
        examples: List[Dict[str, Any]],
    ) -> str:
        req = SkillImplRequest(skill_spec=skill_spec, examples=examples)
        prompt = self._build_code_prompt(req)
        logger.debug("PlanCodeModel codegen prompt: %s", prompt)

        raw = self._backend.generate(
            prompt,
            max_tokens=self._preset.max_tokens,
            temperature=min(self._preset.temperature, 0.2),
            stop=None,
            system_prompt=self._preset.system_prompt,
        )
        logger.debug("PlanCodeModel codegen raw output: %s", raw)

        resp = SkillImplResponse(code=raw, notes="", raw_text=raw)
        return resp.code

