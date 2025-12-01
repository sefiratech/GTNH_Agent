# src/llm_stack/plan_code.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from spec.llm import PlanCodeModel, PlannerPlanJSON, PlannerStepJSON
from spec.types import Observation

from .backend_llamacpp import LlamaCppBackend
from .presets import RolePreset


@dataclass
class PlanCodeModelImpl(PlanCodeModel):
    """
    Implementation of the unified planning + codegen role.

    This class talks to a single backend (e.g. llama.cpp) using a
    role-specific preset. It is intentionally oblivious to error-analysis
    or logging; those belong to other roles.
    """
    backend: LlamaCppBackend
    preset: RolePreset

    # ------------------------------------------------------------------
    # Planning
    # ------------------------------------------------------------------

    def plan(
        self,
        observation: Observation,
        goal: str,
        skill_descriptions: Dict[str, Dict[str, Any]],
        constraints: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Produce a structured plan dict compatible with PlannerPlanJSON.

        The implementation is deliberately simple and conservative; Q1.3+
        callers can wrap the returned dict into PlannerPlanJSON.
        """
        prompt = self._build_plan_prompt(
            observation=observation,
            goal=goal,
            skills=skill_descriptions,
            constraints=constraints,
        )

        raw_text = self.backend.generate(
            prompt=prompt,
            temperature=self.preset.temperature,
            max_tokens=self.preset.max_tokens,
            stop=self.preset.stop,
            system_prompt=self.preset.system_prompt,
        )

        plan_json = self._parse_plan_response(raw_text)
        plan_json.raw_text = raw_text

        return {
            "steps": [
                {
                    "skill": step.skill,
                    "params": dict(step.params),
                    "expected_outcome": step.expected_outcome,
                }
                for step in plan_json.steps
            ],
            "notes": plan_json.notes,
            "raw_text": plan_json.raw_text,
            "error": plan_json.error,
        }

    def _build_plan_prompt(
        self,
        observation: Observation,
        goal: str,
        skills: Dict[str, Dict[str, Any]],
        constraints: Dict[str, Any],
    ) -> str:
        """
        Construct the text prompt for the planning call.

        This stays intentionally loose; the important contract is that
        the *output* parses into PlannerPlanJSON.
        """
        skills_lines: List[str] = []
        for name, meta in skills.items():
            desc = meta.get("description", "")
            skills_lines.append(f"- {name}: {desc}")

        constraint_lines: List[str] = []
        for k, v in (constraints or {}).items():
            constraint_lines.append(f"- {k}: {v}")

        return (
            "You are the planning brain of a Minecraft GTNH automation agent.\n\n"
            "Goal:\n"
            f"{goal}\n\n"
            "Observation (JSON-ish):\n"
            f"{observation}\n\n"
            "Available skills:\n"
            f"{chr(10).join(skills_lines)}\n\n"
            "Constraints:\n"
            f"{chr(10).join(constraint_lines)}\n\n"
            "Respond with STRICT JSON using the shape:\n"
            "{\n"
            '  "steps": [\n'
            '    {"skill": "name", "params": {...}, "expected_outcome": "optional"},\n'
            "    ...\n"
            "  ],\n"
            '  "notes": "short explanation"\n'
            "}\n"
        )

    def _parse_plan_response(self, raw_text: str) -> PlannerPlanJSON:
        """
        Best-effort parsing of raw LLM output into PlannerPlanJSON.

        This keeps parsing deliberately tolerant; on failure, we return a
        PlannerPlanJSON with error set and no steps.
        """
        import json

        try:
            data = json.loads(raw_text)
        except Exception:
            # Completely failed to parse; return empty with error flag.
            return PlannerPlanJSON(
                steps=[],
                notes="",
                raw_text=raw_text,
                error="json_parse_error",
            )

        steps_raw = data.get("steps") or []
        steps: List[PlannerStepJSON] = []
        for s in steps_raw:
            if not isinstance(s, dict):
                continue
            skill = s.get("skill")
            if not isinstance(skill, str) or not skill:
                continue
            params = s.get("params") or {}
            if not isinstance(params, dict):
                params = {}
            expected = s.get("expected_outcome")
            if expected is not None:
                expected = str(expected)
            steps.append(
                PlannerStepJSON(
                    skill=skill,
                    params=params,
                    expected_outcome=expected,
                )
            )

        notes = data.get("notes") or ""
        if not isinstance(notes, str):
            notes = str(notes)

        return PlannerPlanJSON(
            steps=steps,
            notes=notes,
            raw_text=raw_text,
            error=None,
        )

    # ------------------------------------------------------------------
    # Codegen
    # ------------------------------------------------------------------

    def propose_skill_implementation(
        self,
        skill_spec: Dict[str, Any],
        examples: List[Dict[str, Any]],
    ) -> str:
        """
        Produce Python code implementing the given skill.

        This is a plain text-generation call; any safety / sandboxing is
        handled by callers and tooling.
        """
        prompt = self._build_codegen_prompt(skill_spec, examples)
        code = self.backend.generate(
            prompt=prompt,
            temperature=self.preset.temperature,
            max_tokens=self.preset.max_tokens,
            stop=self.preset.stop,
            system_prompt=self.preset.system_prompt,
        )
        return code

    def _build_codegen_prompt(
        self,
        skill_spec: Dict[str, Any],
        examples: List[Dict[str, Any]],
    ) -> str:
        lines: List[str] = [
            "You write Python skills for a GTNH Minecraft automation agent.",
            "Implement the following skill as a single Python function.",
            "",
            "Skill spec (JSON-ish):",
            repr(skill_spec),
        ]
        if examples:
            lines.append("")
            lines.append("Example traces:")
            for ex in examples:
                lines.append(repr(ex))
        lines.append("")
        lines.append("Return only valid Python code.")
        return "\n".join(lines)

