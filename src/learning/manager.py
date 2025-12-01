# path: src/learning/manager.py

"""
M10 SkillLearningManager

High-level orchestrator for skill learning:

Responsibilities
----------------
- Read experiences from ExperienceBuffer (Q1.4 Experience)
- Filter/cluster by goal substring or skill pattern
- Invoke SkillSynthesizer for candidate creation
- Run SkillEvaluator for baseline metrics & recommendation
- Persist candidate artifacts under config/skills_candidates/
- Expose a simple "learning cycle" API that M11 or tools can call

Q1 context:
- Experience carries plan, final_outcome, virtue_scores, etc.
  This manager can use those for aggregation and learning policies.

Q1.2 additions:
- Provide a registry-backed metrics update API:
    update_skill_metrics(skill_name, success, cost, risk, decay)
  that M8/M10 can call after each episode.
- When a new candidate skill is synthesized and evaluated:
    * Persist candidate artifacts
    * Optionally register as a candidate in the SkillRegistry
    * Optionally auto-promote if the evaluator recommends promotion and
      success_rate improves by a threshold.

Q1.2/Q1.4 aggregation:
- Compute aggregated SkillPerformanceStats from Experience data.
- Provide SkillView for curriculum:
    * active_skills: stable / safe
    * candidate_skills: allowed if exploration enabled
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from .buffer import ExperienceBuffer
from .schema import Experience, SkillCandidate, SkillPerformanceStats, SkillName
from .synthesizer import SkillSynthesizer
from .evaluator import SkillEvaluator

from skills.registry import SkillRegistry           # M5 skill registry
from skills.schema import (                         # M5 skill spec dataclasses
    ParamSpec as _ParamSpec,
    SkillEffects as _SkillEffects,
    SkillMetadata as _SkillMetadata,
    SkillMetrics as _SkillMetrics,
    SkillPreconditions as _SkillPreconditions,
    SkillSpec as _SkillSpec,
)


@dataclass
class SkillView:
    """
    Lightweight view for curriculum:

    - active_skills:
        Skills considered "stable" and safe to use in default policies.
        In Pass C, this is derived purely from SkillRegistry metadata status.
    - candidate_skills:
        Skills that are experimental / exploratory (optional).
    """
    active_skills: List[SkillName]
    candidate_skills: List[SkillName]


class SkillLearningManager:
    """
    High-level orchestrator for M10 skill learning.

    Typical usage (offline or scheduled):

        manager = SkillLearningManager(
            buffer=experience_buffer,
            synthesizer=skill_synthesizer,
            evaluator=skill_evaluator,
            skills=skill_registry,
            candidates_dir=Path("config/skills_candidates"),
        )

        result = manager.run_learning_cycle_for_goal(
            goal_substring="maintain coke ovens",
            target_skill_name="maintain_coke_ovens",
            context_id="steam_age",
            tech_tier="steam_age",
        )

    Q1.2:
    - `update_skill_metrics(...)` can be called after each episode to keep
      SkillSpec.metadata.metrics updated as a decayed moving average.
    - Auto-registration / promotion of synthesized candidates is handled via
      the SkillRegistry versioning API.

    Q1.4:
    - Aggregates skill performance from Experience objects and exposes
      SkillPerformanceStats + SkillView for curriculum policies.
    """

    def __init__(
        self,
        buffer: ExperienceBuffer,
        synthesizer: SkillSynthesizer,
        evaluator: SkillEvaluator,
        skills: SkillRegistry,
        candidates_dir: Path,
        *,
        semantics_db: Any | None = None,
    ) -> None:
        """
        Parameters
        ----------
        buffer:
            ExperienceBuffer instance for reading Q1.4 Experience objects.
        synthesizer:
            SkillSynthesizer instance for generating SkillCandidate objects.
        evaluator:
            SkillEvaluator instance for computing performance stats.
        skills:
            SkillRegistry from M5 (used for metadata, baseline skills, and
            version-aware candidate registration).
        candidates_dir:
            Directory where candidate YAML/code/metadata files are written.
        semantics_db:
            Optional semantics DB / cache to pass into evaluator metrics.
        """
        self._buffer = buffer
        self._synthesizer = synthesizer
        self._evaluator = evaluator
        self._skills = skills
        self._candidates_dir = candidates_dir
        self._candidates_dir.mkdir(parents=True, exist_ok=True)
        self._semantics_db = semantics_db

    # ------------------------------------------------------------------
    # Public API – Learning cycle
    # ------------------------------------------------------------------

    def run_learning_cycle_for_goal(
        self,
        goal_substring: str,
        target_skill_name: Optional[SkillName],
        context_id: str,
        *,
        tech_tier: Optional[str] = None,
        success_only: bool = True,
        min_episodes: int = 5,
    ) -> Optional[Dict[str, Any]]:
        """
        Execute a full learning cycle for a given goal substring.

        Steps:
        1. Gather relevant experiences from ExperienceBuffer.
        2. Cluster/filter.
        3. Synthesize candidate skill via SkillSynthesizer.
        4. Evaluate vs baseline (if target_skill_name is provided).
        5. Persist candidate artifacts to candidates_dir.
        6. Optionally register / promote candidate via SkillRegistry.
        7. Return a structured result dict.
        """
        episodes = self._select_experiences_for_goal(
            goal_substring=goal_substring,
            skill_name=target_skill_name,
            tech_tier=tech_tier,
            success_only=success_only,
        )

        if len(episodes) < min_episodes:
            # Not enough signal to do anything useful.
            return None

        # 3. Synthesize candidate skill
        candidate_id = self._make_candidate_id(goal_substring, target_skill_name, len(episodes))
        candidate = self._synthesizer.propose_from_episodes(
            episodes=episodes,
            target_skill_name=target_skill_name,
            candidate_id=candidate_id,
            context_hint=context_id,
        )

        # 4. Evaluation
        baseline_stats: Optional[SkillPerformanceStats] = None
        candidate_stats: Optional[SkillPerformanceStats] = None
        evaluation_result: Optional[Dict[str, Any]] = None

        skill_metadata: Dict[str, Dict[str, Any]] = self._skills.describe_all()

        # When refining an existing skill, compute baseline
        if target_skill_name:
            baseline_stats = self._evaluator.aggregate_skill_stats(
                episodes=episodes,
                skill_name=target_skill_name,
                context_id=context_id,
                skill_metadata=skill_metadata,
                semantics_db=self._semantics_db,
            )
            candidate.metrics_before = baseline_stats

        # For v1, we don't have A/B experiments yet, so candidate_stats
        # == baseline_stats or is left None. Future M10+M8 integration can
        # update candidate.metrics_after and re-run compare_stats().
        if baseline_stats is not None:
            candidate_stats = baseline_stats

            evaluation_result = self._evaluator.compare_stats(
                baseline=baseline_stats,
                candidate=candidate_stats,
            )

        # 5. Persist candidate artifacts
        self._save_candidate(candidate, evaluation_result)

        # 6. Optionally register / promote candidate into the SkillRegistry
        self._maybe_register_and_promote_candidate(
            candidate=candidate,
            baseline_stats=baseline_stats,
            candidate_stats=candidate_stats,
            evaluation=evaluation_result,
        )

        return {
            "candidate": candidate,
            "baseline_stats": baseline_stats,
            "candidate_stats": candidate_stats,
            "evaluation": evaluation_result,
            # No explicit episode IDs on Experience; expose goal IDs as a proxy.
            "episodes_used": [getattr(getattr(ep, "goal", None), "id", "") for ep in episodes],
        }

    # ------------------------------------------------------------------
    # Aggregation – Skill performance from Experience (Q1.4)
    # ------------------------------------------------------------------

    def compute_skill_stats(self) -> Dict[SkillName, SkillPerformanceStats]:
        """
        Scan all Experiences and aggregate per-skill performance.

        For each Experience:
        - For each plan.skill_invocation:
            * count uses
            * track success/failure via final_outcome["success"] (default True)
            * accumulate duration_sec / resource_cost if present
            * accumulate virtue scores (averaged later)
        """
        aggregates: Dict[SkillName, Dict[str, Any]] = {}

        for exp in self._buffer.load_all():
            final = getattr(exp, "final_outcome", None) or {}
            success_flag = bool(final.get("success", True))
            # Best-effort numeric extraction; zero if missing.
            duration = float(final.get("duration_sec", 0.0) or 0.0)
            resource_cost = float(final.get("resource_cost", 0.0) or 0.0)
            virtues = getattr(exp, "virtue_scores", None) or {}

            plan_obj = getattr(exp, "plan", None)
            invocations = getattr(plan_obj, "skill_invocations", []) or []
            for inv in invocations:
                name: Optional[str] = getattr(inv, "skill_name", None)
                if name is None and isinstance(inv, dict):
                    name = str(inv.get("skill_name") or inv.get("skill") or "")
                if not name:
                    continue

                agg = aggregates.setdefault(
                    name,
                    {
                        "uses": 0,
                        "successes": 0,
                        "total_time": 0.0,
                        "total_cost": 0.0,
                        "virtue_sums": {},  # virtue_name -> sum
                    },
                )
                agg["uses"] += 1
                if success_flag:
                    agg["successes"] += 1
                agg["total_time"] += duration
                agg["total_cost"] += resource_cost

                virtue_sums: Dict[str, float] = agg["virtue_sums"]
                for v_name, v_val in virtues.items():
                    try:
                        v_num = float(v_val)
                    except (TypeError, ValueError):
                        continue
                    virtue_sums[v_name] = virtue_sums.get(v_name, 0.0) + v_num

        stats: Dict[SkillName, SkillPerformanceStats] = {}
        for skill_name, agg in aggregates.items():
            uses = agg["uses"] or 0
            if uses <= 0:
                stats[skill_name] = SkillPerformanceStats.zero(skill_name)
                continue

            successes = agg["successes"]
            total_time = agg["total_time"]
            total_cost = agg["total_cost"]
            virtue_sums = agg["virtue_sums"]

            success_rate = float(successes) / float(uses)
            avg_time = float(total_time) / float(uses)
            avg_cost = float(total_cost) / float(uses)
            avg_virtues = {
                v_name: v_sum / float(uses)
                for v_name, v_sum in virtue_sums.items()
            }

            stats[skill_name] = SkillPerformanceStats(
                skill_name=skill_name,
                uses=uses,
                success_rate=success_rate,
                avg_time=avg_time,
                avg_resource_cost=avg_cost,
                avg_virtue_scores=avg_virtues,
            )

        return stats

    def get_skill_stats(self, skill_name: SkillName) -> SkillPerformanceStats:
        """
        Convenience view: return stats for a single skill.

        If no data has been recorded yet, returns SkillPerformanceStats.zero(...).
        """
        all_stats = self.compute_skill_stats()
        return all_stats.get(skill_name, SkillPerformanceStats.zero(skill_name))

    def get_all_skill_stats(self) -> Dict[SkillName, SkillPerformanceStats]:
        """
        Convenience view: return stats for all skills seen in Experience data.
        """
        return self.compute_skill_stats()

    def build_skill_view(
        self,
        *,
        include_candidates: bool,
    ) -> SkillView:
        """
        Build a SkillView for curriculum.

        Pass C semantics (simple, structural):

        - active_skills:
            All skills whose registry metadata.status is one of:
                "active", "accepted", "stable"
        - candidate_skills:
            All skills whose status is "candidate"
            Returned only when include_candidates=True.

        No thresholding on success_rate or usage yet; those live in
        SkillPerformanceStats for future policy layers.
        """
        meta = self._skills.describe_all()

        active: List[SkillName] = []
        candidates: List[SkillName] = []

        for name, payload in meta.items():
            # Metadata may be nested or flat; be tolerant.
            md = payload.get("metadata", payload)
            status = str(md.get("status", "active")).lower()

            if status in {"active", "accepted", "stable"}:
                active.append(name)
            elif status == "candidate":
                if include_candidates:
                    candidates.append(name)
            else:
                # Other statuses (retired, deprecated, etc.) are ignored here.
                continue

        return SkillView(
            active_skills=sorted(active),
            candidate_skills=sorted(candidates),
        )

    # ------------------------------------------------------------------
    # Episode-level metrics integration (Q1.2)
    # ------------------------------------------------------------------

    def update_skill_metrics(
        self,
        skill_name: SkillName,
        *,
        success: float,
        cost: float,
        risk: float,
        decay: float = 0.2,
        now: Optional[datetime] = None,
    ) -> None:
        """
        Update a skill's registry-backed metrics with a decayed moving average.
        """
        handle = self._skills.get_latest_skill(skill_name)
        spec = handle.spec
        metrics = spec.metadata.metrics

        def _update(old: Optional[float], new: float) -> float:
            if old is None:
                return float(new)
            return float((1.0 - decay) * old + decay * new)

        metrics.success_rate = _update(metrics.success_rate, success)
        metrics.avg_cost = _update(metrics.avg_cost, cost)
        metrics.avg_risk = _update(metrics.avg_risk, risk)
        metrics.last_used_at = (now or datetime.now(timezone.utc)).isoformat()

        # Persist back into the registry (in-place + index update).
        self._skills.update_spec(spec)

    def update_skill_metrics_from_stats(
        self,
        stats: SkillPerformanceStats,
        *,
        decay: float = 0.2,
    ) -> None:
        """
        Convenience wrapper: update registry metrics from aggregate stats.

        Risk is approximated as (1 - success_rate) for now.
        """
        risk = max(0.0, min(1.0, 1.0 - stats.success_rate))
        self.update_skill_metrics(
            skill_name=stats.skill_name,
            success=stats.success_rate,
            cost=stats.avg_resource_cost,
            risk=risk,
            decay=decay,
        )

    # ------------------------------------------------------------------
    # Experience selection / clustering (Q1.4-powered, but legacy-tolerant)
    # ------------------------------------------------------------------

    def _select_experiences_for_goal(
        self,
        goal_substring: str,
        *,
        skill_name: Optional[SkillName],
        tech_tier: Optional[str],
        success_only: bool,
    ) -> List[Any]:
        """
        Filter experiences based on:
        - goal substring
        - optional success flag
        - optional skill usage
        - optional tech tier

        Compatible with:
        - Q1.4 Experience objects
        - Legacy episode-like objects (ExperienceEpisode / LegacyEpisode)
        """
        episodes: List[Any] = []

        substring_lower = goal_substring.lower()
        for exp in self._buffer.load_all():
            # Goal text filter
            raw_goal = getattr(exp, "goal", None)
            if raw_goal is None:
                goal_text = ""
            else:
                goal_text = getattr(raw_goal, "text", str(raw_goal))

            if substring_lower not in goal_text.lower():
                continue

            # Success filter (best-effort)
            if success_only:
                # Prefer Experience-style final_outcome["success"]
                fo = getattr(exp, "final_outcome", None) or {}
                success_flag = fo.get("success")
                if success_flag is None:
                    # Fall back to legacy exp.success if present
                    if hasattr(exp, "success"):
                        success_flag = bool(getattr(exp, "success"))
                    else:
                        success_flag = True
                else:
                    success_flag = bool(success_flag)

                if not success_flag:
                    continue

            # Tech tier filter:
            # 1) Experience.problem_signature["tech_tier"]
            # 2) legacy tech_state.active
            if tech_tier is not None:
                sig_tier: Optional[str] = None

                ps = getattr(exp, "problem_signature", None)
                if isinstance(ps, dict):
                    sig_tier = ps.get("tech_tier")

                if sig_tier is None:
                    ts = getattr(exp, "tech_state", None)
                    if ts is not None:
                        sig_tier = getattr(ts, "active", None)

                if sig_tier is not None and sig_tier != tech_tier:
                    continue

            # Skill usage filter via ExperiencePlan.skill_invocations or legacy trace
            if skill_name is not None and not self._experience_uses_skill(exp, skill_name):
                continue

            episodes.append(exp)

        return episodes

    def _experience_uses_skill(
        self,
        exp: Any,
        skill_name: SkillName,
    ) -> bool:
        """
        Return True if the given skill appears in the episode.

        Compatible with:
        - Q1.4 Experience (plan.skill_invocations)
        - Legacy episodes (trace.steps[].meta["skill"])
        """
        # Q1.4 Experience path: plan.skill_invocations
        plan_obj = getattr(exp, "plan", None)
        invocations = getattr(plan_obj, "skill_invocations", None)
        if invocations is not None:
            inv_list = invocations or []
            for inv in inv_list:
                name = getattr(inv, "skill_name", None)
                if name is None and isinstance(inv, dict):
                    name = inv.get("skill_name") or inv.get("skill")
                if name == skill_name:
                    return True

        # Legacy episode path: trace.steps[].meta["skill"]
        trace = getattr(exp, "trace", None)
        steps = getattr(trace, "steps", None)
        if steps is not None:
            for step in steps or []:
                meta = getattr(step, "meta", None)
                if isinstance(meta, dict) and meta.get("skill") == skill_name:
                    return True

        return False

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _make_candidate_id(
        self,
        goal_substring: str,
        target_skill_name: Optional[SkillName],
        episode_count: int,
    ) -> str:
        """
        Construct a simple, readable candidate ID from inputs.
        """
        base = target_skill_name or "auto_skill"
        safe_goal = goal_substring.strip().replace(" ", "_")[:32]
        return f"{base}_{safe_goal}_{episode_count}"

    def _save_candidate(
        self,
        candidate: SkillCandidate,
        evaluation_result: Optional[Dict[str, Any]],
    ) -> None:
        """
        Write candidate spec, implementation, and metadata to disk.
        """
        spec_path = self._candidates_dir / f"{candidate.id}.yaml"
        code_path = self._candidates_dir / f"{candidate.id}.py"
        meta_path = self._candidates_dir / f"{candidate.id}.meta.json"

        spec_path.write_text(candidate.spec_yaml, encoding="utf-8")
        code_path.write_text(candidate.impl_code, encoding="utf-8")

        meta_payload: Dict[str, Any] = {
            "candidate": candidate.to_dict(),
            "evaluation": evaluation_result,
        }
        meta_path.write_text(
            json.dumps(meta_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # ------------------------------------------------------------------
    # Candidate → SkillRegistry integration
    # ------------------------------------------------------------------

    def _maybe_register_and_promote_candidate(
        self,
        *,
        candidate: SkillCandidate,
        baseline_stats: Optional[SkillPerformanceStats],
        candidate_stats: Optional[SkillPerformanceStats],
        evaluation: Optional[Dict[str, Any]],
        success_improvement_threshold: float = 0.05,
    ) -> None:
        """
        Integrate a synthesized candidate with the SkillRegistry.
        """
        # Parse YAML into SkillSpec first.
        try:
            spec = self._parse_skill_spec_from_yaml(candidate.spec_yaml)
        except Exception:
            # If the YAML is garbage, we still keep artifacts on disk and move on.
            return

        # Register as a candidate version in the registry.
        try:
            handle = self._skills.register_skill_candidate(spec)
        except Exception:
            # Don't explode the learning loop if registration fails.
            return

        # No baseline / stats / evaluation? Just keep as candidate, no auto-promotion.
        if baseline_stats is None or candidate_stats is None or evaluation is None:
            return

        # The evaluator is the first gatekeeper.
        if evaluation.get("recommendation") != "promote_candidate":
            return

        # Check success-rate improvement against active baseline.
        if candidate_stats.success_rate <= baseline_stats.success_rate * (1.0 + success_improvement_threshold):
            return

        # At this point, policy allows promotion.
        try:
            self._skills.promote_skill_candidate(handle)
        except Exception:
            # If promotion fails, we leave the candidate registered but non-active.
            return

    def _parse_skill_spec_from_yaml(self, yaml_text: str) -> _SkillSpec:
        """
        Parse a YAML skill definition into a SkillSpec dataclass.
        """
        data = yaml.safe_load(yaml_text) or {}
        if not isinstance(data, dict):
            raise TypeError("Skill spec YAML must decode to a mapping.")

        name = data["name"]
        description = data.get("description", "")

        # Params
        params: Dict[str, _ParamSpec] = {}
        for pname, pval in (data.get("params") or {}).items():
            if not isinstance(pval, dict):
                pval = {}
            params[pname] = _ParamSpec(
                name=pname,
                type=str(pval.get("type", "Any")),
                default=pval.get("default"),
                description=str(pval.get("description", "") or ""),
            )

        # Preconditions
        pre_raw = data.get("preconditions") or []
        if isinstance(pre_raw, dict):
            pre_raw = [pre_raw]
        tech_states_any_of: List[str] = []
        required_tools: List[str] = []
        dimension_allowlist: List[str] = []
        semantic_tags_any_of: List[str] = []
        pre_extra: Dict[str, Any] = {}
        for pre in pre_raw:
            if not isinstance(pre, dict):
                continue
            tech_states_any_of.extend(pre.get("tech_states_any_of") or [])
            required_tools.extend(pre.get("required_tools") or [])
            dimension_allowlist.extend(pre.get("dimension_allowlist") or [])
            semantic_tags_any_of.extend(pre.get("semantic_tags_any_of") or [])
            for k, v in pre.items():
                if k not in {
                    "tech_states_any_of",
                    "required_tools",
                    "dimension_allowlist",
                    "semantic_tags_any_of",
                }:
                    pre_extra[k] = v

        preconditions = _SkillPreconditions(
            tech_states_any_of=tech_states_any_of,
            required_tools=required_tools,
            dimension_allowlist=dimension_allowlist,
            semantic_tags_any_of=semantic_tags_any_of,
            extra=pre_extra,
        )

        # Effects
        eff_raw = data.get("effects") or {}
        inventory_delta = dict(eff_raw.get("inventory_delta") or {})
        tech_delta = dict(eff_raw.get("tech_delta") or {})
        effect_tags = list(eff_raw.get("tags") or [])
        eff_extra = {
            k: v
            for k, v in eff_raw.items()
            if k not in {"inventory_delta", "tech_delta", "tags"}
        }
        effects = _SkillEffects(
            inventory_delta=inventory_delta,
            tech_delta=tech_delta,
            tags=effect_tags,
            extra=eff_extra,
        )

        # Metadata & metrics
        metrics_raw = data.get("metrics") or {}
        metrics = _SkillMetrics(
            success_rate=metrics_raw.get("success_rate"),
            avg_cost=metrics_raw.get("avg_cost"),
            avg_risk=metrics_raw.get("avg_risk"),
            last_used_at=metrics_raw.get("last_used_at"),
        )

        metadata = _SkillMetadata(
            version=data.get("version", "1"),
            status=data.get("status", "candidate"),
            origin=data.get("origin", "auto_synthesized"),
            metrics=metrics,
        )

        tags = list(data.get("tags") or [])

        spec = _SkillSpec(
            name=name,
            description=description,
            params=params,
            preconditions=preconditions,
            effects=effects,
            tags=tags,
            metadata=metadata,
        )
        return spec

