"""
M10 Skill Synthesizer

Uses the M2 CodeModel to turn clusters of successful episodes into
SkillCandidate objects (YAML SkillSpec + Python impl + rationale).

This module is intentionally narrow in scope:

- INPUT:
    - Cluster of ExperienceEpisode objects
    - Optional target skill name (for refinement)
    - Candidate ID

- OUTPUT:
    - SkillCandidate dataclass instance

It does NOT:
- Decide which episodes to use (that's the manager's job).
- Persist candidates to disk (also the manager's job).
- Evaluate performance (handled by evaluator.py).

Q1.2:
- Synthesizer is responsible for producing a fully-formed SkillCandidate.
  The SkillLearningManager will then:
    * persist artifacts
    * register the spec as a candidate in the SkillRegistry
    * optionally promote, based on evaluation.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List, Optional

from .schema import ExperienceEpisode, SkillCandidate, SkillName, CandidateId
from semantics.schema import TechState
from observation.trace_schema import PlanTrace

# We assume spec.llm defines a CodeModel-like interface that can return JSON.
# If your actual interface differs, you can adapt the call site in `generate_skill_json`.
from spec.llm import CodeModel  # type: ignore[import]


class SkillSynthesizer:
    """
    Uses CodeModel (M2) to derive new skill definitions from experience episodes.

    Typical usage:
        synthesizer = SkillSynthesizer(code_model)
        candidate = synthesizer.propose_from_episodes(
            episodes=cluster,
            target_skill_name="feed_coke_ovens",
            candidate_id="feed_coke_ovens_v2_001",
        )
    """

    def __init__(self, code_model: CodeModel) -> None:
        """
        Parameters
        ----------
        code_model:
            A CodeModel instance from M2 capable of structured JSON generation.
        """
        self._model = code_model

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def propose_from_episodes(
        self,
        episodes: List[ExperienceEpisode],
        target_skill_name: Optional[SkillName],
        candidate_id: CandidateId,
        *,
        context_hint: Optional[str] = None,
    ) -> SkillCandidate:
        """
        Build a structured prompt from a cluster of episodes and ask the
        CodeModel to synthesize a SkillCandidate.

        Parameters
        ----------
        episodes:
            Cluster of successful ExperienceEpisode objects that represent
            similar behavior the agent should encapsulate as a skill.
        target_skill_name:
            If refining an existing skill, pass its name. Otherwise None
            for a brand-new skill.
        candidate_id:
            Identifier for the candidate (used for filenames, logging, etc).
        context_hint:
            Optional free-form context string (e.g. "steam_age", "LV_core")
            that can be used to steer the model.

        Returns
        -------
        SkillCandidate
            A candidate with spec_yaml, impl_code, rationale, and initial status.

        Q1.2 detail:
        - status is initialized to "proposed" so that tests and higher-level
          code can treat this as a newly proposed skill ready for review.
        """
        if not episodes:
            raise ValueError("SkillSynthesizer.propose_from_episodes called with empty episodes list.")

        prompt = self._build_prompt_payload(
            episodes=episodes,
            target_skill_name=target_skill_name,
            candidate_id=candidate_id,
            context_hint=context_hint,
        )

        response = self._generate_skill_json(prompt)

        spec_yaml = response.get("spec_yaml", "").strip()
        impl_code = response.get("impl_code", "").strip()
        rationale = response.get("rationale", "").strip()

        candidate = SkillCandidate(
            id=candidate_id,
            base_skill_name=target_skill_name,
            spec_yaml=spec_yaml,
            impl_code=impl_code,
            rationale=rationale,
            # Test contract: candidate.status must be "proposed"
            status="proposed",
            metrics_before=None,
            metrics_after=None,
            created_at=None,
            updated_at=None,
            tags=[],
            extra={
                "synthesizer_prompt": {
                    "target_skill_name": target_skill_name,
                    "context_hint": context_hint,
                    "episode_ids": [ep.id for ep in episodes],
                }
            },
        )
        return candidate

    # ------------------------------------------------------------------
    # Core prompt construction
    # ------------------------------------------------------------------

    def _build_prompt_payload(
        self,
        episodes: List[ExperienceEpisode],
        target_skill_name: Optional[SkillName],
        candidate_id: CandidateId,
        context_hint: Optional[str],
    ) -> Dict[str, Any]:
        """
        Convert episodes into a compact, LLM-friendly payload.

        The idea is to summarize each episode in terms of:
        - goal text
        - tech_state.active
        - virtue scores
        - condensed step list (skill, action_type, params, success)
        """
        # Convert episodes into simplified trace summaries
        trace_summaries: List[Dict[str, Any]] = []
        for ep in episodes:
            trace_summaries.append(self._summarize_episode(ep))

        # High-level instruction for the CodeModel
        instruction = (
            "You are helping design a reusable Minecraft GregTech New Horizons skill.\n"
            "Given several examples of successful behavior (episodes), extract the common\n"
            "pattern and propose a single high-level skill definition that:\n"
            "- is safe, robust, and efficient\n"
            "- matches the repeated behavior across episodes\n"
            "- respects the tech tier and environment\n\n"
            "Return your answer as JSON with fields: spec_yaml, impl_code, rationale."
        )

        if target_skill_name:
            instruction += (
                "\n\nThe skill MAY refine an existing skill named "
                f"'{target_skill_name}'. If so, improve its behavior."
            )

        if context_hint:
            instruction += (
                "\n\nContext hint from the curriculum / environment: "
                f"{context_hint}"
            )

        output_format_schema = {
            "spec_yaml": (
                "YAML string describing the skill in the existing SkillSpec format. "
                "Include name, version, description, parameters, preconditions, "
                "effects, tags, and Q1 metadata fields: version, status, origin, "
                "metrics.success_rate/avg_cost/avg_risk/last_used_at (use nulls "
                "for unknown metric values)."
            ),
            "impl_code": (
                "Python code implementing the skill in the existing skill runtime "
                "style (a class or function skeleton), with clear TODOs where needed."
            ),
            "rationale": (
                "Natural language explanation (1–3 paragraphs) explaining the design, "
                "assumptions, and why this encapsulation is useful."
            ),
        }

        payload: Dict[str, Any] = {
            "instruction": instruction,
            "candidate_id": candidate_id,
            "target_skill_name": target_skill_name,
            "context_hint": context_hint,
            "episodes": trace_summaries,
            "output_format": output_format_schema,
        }
        return payload

    def _summarize_episode(self, episode: ExperienceEpisode) -> Dict[str, Any]:
        """
        Convert a single ExperienceEpisode into a compact, JSON-friendly summary.

        This intentionally avoids dumping the whole PlanTrace/TechState structure;
        instead it takes only fields that are most useful for synthesis.
        """
        tech: TechState = episode.tech_state
        trace: PlanTrace = episode.trace

        # Basic tech info
        tech_summary: Dict[str, Any] = {
            "active_tier": getattr(tech, "active", None),
            "unlocked": list(getattr(tech, "unlocked", []) or []),
        }

        # Condense plan + steps into something LLM can digest
        steps_summary: List[Dict[str, Any]] = []
        for step in getattr(trace, "steps", []) or []:
            # We don't rely on full type structure here; we just probe common attributes.
            action = getattr(step, "action", None)
            result = getattr(step, "result", None)
            meta = getattr(step, "meta", {}) or {}

            action_type = getattr(action, "type", None) if action is not None else None
            params = getattr(action, "params", {}) if action is not None else {}
            success = getattr(result, "success", None) if result is not None else None

            steps_summary.append(
                {
                    "skill": meta.get("skill"),
                    "action_type": action_type,
                    "params": params,
                    "success": success,
                    "meta": meta,
                }
            )

        plan_summary: Dict[str, Any] = {}
        raw_plan = getattr(trace, "plan", None)
        if isinstance(raw_plan, dict):
            plan_summary = raw_plan
        elif raw_plan is not None:
            # Last resort: reflect as dict if possible
            try:
                plan_summary = asdict(raw_plan)  # type: ignore[arg-type]
            except Exception:
                plan_summary = {"repr": repr(raw_plan)}

        return {
            "episode_id": episode.id,
            "goal": episode.goal,
            "tech_state": tech_summary,
            "virtue_scores": dict(episode.virtue_scores),
            "success": episode.success,
            "metadata": episode.metadata.to_dict(),
            "plan": plan_summary,
            "steps": steps_summary,
        }

    # ------------------------------------------------------------------
    # CodeModel integration
    # ------------------------------------------------------------------

    def _generate_skill_json(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Call the underlying CodeModel to get a structured JSON response.

        This method centralizes the coupling to M2 so that if the interface
        changes, only this function needs to be updated.

        Expected behavior:
            self._model.generate_json(prompt=payload) -> Dict[str, Any]

        If your CodeModel uses a different method name or signature, adapt it here.
        """
        # Adapt this to your actual CodeModel API.
        response: Dict[str, Any] = self._model.generate_json(prompt=payload)  # type: ignore[attr-defined]

        if not isinstance(response, Dict):
            raise TypeError(
                f"CodeModel.generate_json returned non-dict response: {type(response)}"
            )

        return response

