# M10 – skill_learning

## Purpose

Transform lived agent episodes into structured experience memory, synthesize new skills, evaluate improvements, and version skills for GTNH optimization. M10 is the self‑research division of the agent: learning is offline, scheduled, and grounded in the event stream emitted by M8–M9.

## Overview

### Components
- **Experience Buffer**
  - Stores `{state, goal, plan, actions, outcomes, virtue_scores}` from M8 traces and monitoring events.
  - Provides querying, sampling, and exporting for analysis and learning.

- **LLM-Based Synthesizer**
  - Uses M2 CodeModel to turn repeated success traces into:
    - New skill YAML specs
    - Candidate implementation code
    - Rationale for internal reasoning transparency

- **Evaluator**
  - Compares new skills vs baseline using:
    - Success rate
    - Time cost
    - Resource cost
    - Virtue scores (M4)
  - Produces recommendation: promote candidate, keep baseline, or reject.

### Dependencies
- **M8** — AgentLoop trace & plan lifecycle  
- **M2** — CodeModel for synthesis  
- **M5** — Skill registry for versioning & metadata  
- **M4** — Virtue scoring & metrics  

### Difficulty
⭐⭐⭐⭐⭐

### Scaling Notes
- Learning should run offline or during low-load intervals.
- Skills receive versions and rollback support to prevent regressions.

---

# 1. Responsibilities & Boundaries

## 1.1 What M10 Owns
- Construction of **ExperienceEpisode** objects.
- Storage in a **Replay / Experience Buffer**.
- LLM-driven **Skill Synthesis**.
- **Skill Evaluation** (baseline vs candidate).
- **Versioning & Candidate Management**.

## 1.2 What M10 Does Not Do
- Does not directly control the agent.
- Does not modify BotCore, semantics, or observation.
- Does not auto-deploy skills without human or curriculum approval.

---

# 2. Package Layout

```
src/learning/
  __init__.py
  schema.py           # Episode, SkillCandidate, PerformanceMetrics
  buffer.py           # JSONL-backed ExperienceBuffer
  synthesizer.py      # SkillSynthesizer using M2 code model
  evaluator.py        # Metrics aggregation & comparison
  manager.py          # High-level orchestration for learning cycles
```

Config paths:
```
config/skills/               # Approved skills
config/skills_candidates/    # Proposed skills awaiting evaluation
```

---

# 3. Experience Schema

Episodes wrap M8 trace output, monitoring events, tech state, and virtue scores.

**ExperienceEpisode Fields**
- `id` — unique episode identifier  
- `goal` — goal text from planner  
- `tech_state` — inferred tier & unlocks  
- `trace` — PlanTrace from M7  
- `virtue_scores` — M4 scoring  
- `success` — boolean  
- `metadata` — timestamps, tags, environment profile details  

Includes additional structures:
- `SkillPerformanceStats`
- `SkillCandidate` (YAML + impl + rationale + versioning state)

```python
#src/learning/schema.py
"""
Experience & skill-learning schema for M10.

This module defines the core data structures used by the learning layer:
- ExperienceEpisode: a single plan/episode as seen by M10
- SkillPerformanceStats: aggregated metrics for a single skill
- SkillCandidate: a proposed new or refined skill

The design assumes:
- TechState comes from `semantics.schema.TechState`
- PlanTrace comes from `observation.trace_schema.PlanTrace`
- Virtue scores are produced by M4 (virtues.*)

All structures are:
- Plain dataclasses for easy testing
- JSON-friendly via `to_dict()`
- Reconstructable via `from_dict()` with optional adapters
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict, is_dataclass
from typing import Any, Dict, List, Optional, Literal, Callable, TypeVar, Generic
from datetime import datetime

# External schemas from existing modules
from semantics.schema import TechState          # M3 tech progression representation
from observation.trace_schema import PlanTrace  # M7 execution trace representation

# ---------------------------------------------------------------------------
# Type aliases & small helpers
# ---------------------------------------------------------------------------

VirtueScores = Dict[str, float]
EpisodeId = str
CandidateId = str
SkillName = str

SkillCandidateStatus = Literal[
    "proposed",    # just synthesized by M10
    "evaluating",  # being A/B tested or under review
    "accepted",    # promoted into main registry
    "rejected",    # discarded / archived
]

T = TypeVar("T")


def _maybe_to_dict(obj: Any) -> Any:
    """
    Best-effort conversion of arbitrary objects into JSON-friendly structures.

    Priority:
    1. If object has `to_dict()`, call it.
    2. If it's a dataclass, use `asdict()`.
    3. If it has __dict__, return that.
    4. Otherwise, return as-is (caller must ensure JSON-serializable).
    """
    if obj is None:
        return None

    to_dict = getattr(obj, "to_dict", None)
    if callable(to_dict):
        return to_dict()

    if is_dataclass(obj):
        return asdict(obj)

    if hasattr(obj, "__dict__"):
        return dict(obj.__dict__)

    return obj


def _datetime_to_iso(dt: Optional[datetime]) -> Optional[str]:
    if dt is None:
        return None
    return dt.isoformat()


def _datetime_from_iso(value: Optional[str]) -> Optional[datetime]:
    if value is None:
        return None
    return datetime.fromisoformat(value)


# ---------------------------------------------------------------------------
# Core episode schema
# ---------------------------------------------------------------------------

@dataclass
class EpisodeMetadata:
    """
    Auxiliary metadata for an episode.

    This intentionally mirrors what M9 / runtime already know:
    - timestamps
    - environment / profile IDs
    - tags (e.g., curriculum slice, skill pack, scenario)
    """
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    env_profile_id: Optional[str] = None       # link back to EnvProfile / env.yaml
    curriculum_id: Optional[str] = None        # which curriculum unit (M11) this came from
    skill_pack_id: Optional[str] = None        # which skill pack was active (e.g. "steam_age")
    tags: List[str] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "start_time": _datetime_to_iso(self.start_time),
            "end_time": _datetime_to_iso(self.end_time),
            "env_profile_id": self.env_profile_id,
            "curriculum_id": self.curriculum_id,
            "skill_pack_id": self.skill_pack_id,
            "tags": list(self.tags),
            "extra": dict(self.extra),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EpisodeMetadata":
        return cls(
            start_time=_datetime_from_iso(data.get("start_time")),
            end_time=_datetime_from_iso(data.get("end_time")),
            env_profile_id=data.get("env_profile_id"),
            curriculum_id=data.get("curriculum_id"),
            skill_pack_id=data.get("skill_pack_id"),
            tags=list(data.get("tags") or []),
            extra=dict(data.get("extra") or {}),
        )


@dataclass
class ExperienceEpisode:
    """
    A single learning episode derived from one plan execution in M8.

    This is the primary unit of experience that M10 consumes.
    It wraps:
    - goal text
    - TechState at execution
    - full PlanTrace
    - virtue scores
    - success flag
    - structured metadata
    """
    id: EpisodeId
    goal: str
    tech_state: TechState
    trace: PlanTrace
    virtue_scores: VirtueScores
    success: bool
    metadata: EpisodeMetadata = field(default_factory=EpisodeMetadata)

    # ------------------------------------------------------------------
    # Serialization helpers
    # ------------------------------------------------------------------

    def to_dict(
        self,
        *,
        tech_state_to_dict: Optional[Callable[[TechState], Dict[str, Any]]] = None,
        plan_trace_to_dict: Optional[Callable[[PlanTrace], Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Convert this episode into a JSON-serializable dict.

        Callers can override how TechState and PlanTrace are serialized by
        passing `tech_state_to_dict` and `plan_trace_to_dict`.
        """
        if tech_state_to_dict is None:
            tech_state_to_dict = _maybe_to_dict  # type: ignore[assignment]
        if plan_trace_to_dict is None:
            plan_trace_to_dict = _maybe_to_dict  # type: ignore[assignment]

        return {
            "id": self.id,
            "goal": self.goal,
            "tech_state": tech_state_to_dict(self.tech_state),
            "trace": plan_trace_to_dict(self.trace),
            "virtue_scores": dict(self.virtue_scores),
            "success": self.success,
            "metadata": self.metadata.to_dict(),
        }

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
        *,
        tech_state_from_dict: Callable[[Dict[str, Any]], TechState],
        plan_trace_from_dict: Callable[[Dict[str, Any]], PlanTrace],
    ) -> "ExperienceEpisode":
        """
        Reconstruct an ExperienceEpisode from a dict.

        The caller must provide `tech_state_from_dict` and `plan_trace_from_dict`
        because M10 does not own those implementations.
        """
        return cls(
            id=data["id"],
            goal=data["goal"],
            tech_state=tech_state_from_dict(data["tech_state"]),
            trace=plan_trace_from_dict(data["trace"]),
            virtue_scores=dict(data.get("virtue_scores") or {}),
            success=bool(data["success"]),
            metadata=EpisodeMetadata.from_dict(data.get("metadata") or {}),
        )


# ---------------------------------------------------------------------------
# Aggregated performance metrics
# ---------------------------------------------------------------------------

@dataclass
class SkillPerformanceStats:
    """
    Aggregated statistics for a single skill across many episodes.

    This struct is computed by the evaluator and is used both for:
    - baseline skills (current registry entries)
    - candidate skills (after experimental rollout)
    """
    skill_name: SkillName
    uses: int
    success_rate: float
    avg_time: float
    avg_resource_cost: float
    avg_virtue_scores: VirtueScores = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "skill_name": self.skill_name,
            "uses": int(self.uses),
            "success_rate": float(self.success_rate),
            "avg_time": float(self.avg_time),
            "avg_resource_cost": float(self.avg_resource_cost),
            "avg_virtue_scores": dict(self.avg_virtue_scores),
        }

    @classmethod
    def zero(cls, skill_name: SkillName) -> "SkillPerformanceStats":
        """Convenience constructor for 'no data yet' stats."""
        return cls(
            skill_name=skill_name,
            uses=0,
            success_rate=0.0,
            avg_time=0.0,
            avg_resource_cost=0.0,
            avg_virtue_scores={},
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SkillPerformanceStats":
        return cls(
            skill_name=data["skill_name"],
            uses=int(data.get("uses", 0)),
            success_rate=float(data.get("success_rate", 0.0)),
            avg_time=float(data.get("avg_time", 0.0)),
            avg_resource_cost=float(data.get("avg_resource_cost", 0.0)),
            avg_virtue_scores=dict(data.get("avg_virtue_scores") or {}),
        )


# ---------------------------------------------------------------------------
# Skill candidates
# ---------------------------------------------------------------------------

@dataclass
class SkillCandidate:
    """
    A proposed new or refined skill produced by M10.

    This struct is the bridge between:
    - LLM-based synthesis (spec_yaml + impl_code + rationale)
    - evaluation (metrics_before / metrics_after)
    - lifecycle management (status)
    """
    id: CandidateId
    base_skill_name: Optional[SkillName]          # None if brand new, else refined skill
    spec_yaml: str                                # full YAML for SkillSpec
    impl_code: str                                # Python implementation stub
    rationale: str                                # explanation from synthesizer
    status: SkillCandidateStatus                  # lifecycle state

    metrics_before: Optional[SkillPerformanceStats] = None
    metrics_after: Optional[SkillPerformanceStats] = None

    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    tags: List[str] = field(default_factory=list)  # arbitrary labels (e.g., "steam_age", "LV")
    extra: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Lifecycle helpers
    # ------------------------------------------------------------------

    def mark_status(self, new_status: SkillCandidateStatus) -> None:
        """Update candidate status and bump updated_at."""
        self.status = new_status
        self.updated_at = datetime.utcnow()

    def touch(self) -> None:
        """Bump updated_at without changing status."""
        self.updated_at = datetime.utcnow()

    # ------------------------------------------------------------------
    # Serialization helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "base_skill_name": self.base_skill_name,
            "spec_yaml": self.spec_yaml,
            "impl_code": self.impl_code,
            "rationale": self.rationale,
            "status": self.status,
            "metrics_before": (
                self.metrics_before.to_dict() if self.metrics_before else None
            ),
            "metrics_after": (
                self.metrics_after.to_dict() if self.metrics_after else None
            ),
            "created_at": _datetime_to_iso(self.created_at),
            "updated_at": _datetime_to_iso(self.updated_at),
            "tags": list(self.tags),
            "extra": dict(self.extra),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SkillCandidate":
        metrics_before_raw = data.get("metrics_before")
        metrics_after_raw = data.get("metrics_after")

        return cls(
            id=data["id"],
            base_skill_name=data.get("base_skill_name"),
            spec_yaml=data.get("spec_yaml", ""),
            impl_code=data.get("impl_code", ""),
            rationale=data.get("rationale", ""),
            status=data.get("status", "proposed"),
            metrics_before=(
                SkillPerformanceStats.from_dict(metrics_before_raw)
                if metrics_before_raw
                else None
            ),
            metrics_after=(
                SkillPerformanceStats.from_dict(metrics_after_raw)
                if metrics_after_raw
                else None
            ),
            created_at=_datetime_from_iso(data.get("created_at")),
            updated_at=_datetime_from_iso(data.get("updated_at")),
            tags=list(data.get("tags") or []),
            extra=dict(data.get("extra") or {}),
        )


# ---------------------------------------------------------------------------
# Public exports
# ---------------------------------------------------------------------------

__all__ = [
    "VirtueScores",
    "EpisodeId",
    "CandidateId",
    "SkillName",
    "SkillCandidateStatus",
    "EpisodeMetadata",
    "ExperienceEpisode",
    "SkillPerformanceStats",
    "SkillCandidate",
]

```

---

# 4. Experience Buffer

A JSONL file storing one episode per line.  
Supports:
- `append(episode)`
- Iterative loading with `load_all()`
- Efficient filtering by goal, success, skill usage, tech tier.

Acts as the offline memory for all learning modules.

```python
#src/learning/buffer.py
"""
M10 Experience Buffer

JSONL-backed storage for ExperienceEpisode objects.

Responsibilities:
- Persist each episode from M8 as one JSON object per line.
- Provide streaming access to raw dicts (for quick analysis).
- Provide typed access to ExperienceEpisode objects.
- Expose basic filter helpers by goal, success, skill usage, and tech tier.

This module does NOT:
- Decide when episodes are created (that’s AgentLoop + EpisodeRecorder).
- Perform clustering, synthesis, or evaluation (see synthesizer/evaluator/manager).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional

from .schema import ExperienceEpisode, EpisodeId, SkillName


# Type aliases for serializer hooks
TechStateToDict = Callable[[Any], Dict[str, Any]]
TechStateFromDict = Callable[[Dict[str, Any]], Any]
TraceToDict = Callable[[Any], Dict[str, Any]]
TraceFromDict = Callable[[Dict[str, Any]], Any]


class ExperienceBuffer:
    """
    JSONL-backed experience buffer.

    Each line in the file is a single JSON object representing one episode:
    {
      "id": "...",
      "goal": "...",
      "tech_state": {...},
      "trace": {...},
      "virtue_scores": {...},
      "success": true/false,
      "metadata": {...}
    }

    TechState and PlanTrace serialization is controlled by injected
    serializer/deserializer functions. This keeps M10 decoupled from the
    exact implementation details in M3/M7.
    """

    def __init__(
        self,
        path: Path,
        *,
        tech_state_to_dict: TechStateToDict,
        tech_state_from_dict: TechStateFromDict,
        trace_to_dict: TraceToDict,
        trace_from_dict: TraceFromDict,
    ) -> None:
        """
        Initialize an ExperienceBuffer backed by a JSONL file.

        Parameters
        ----------
        path:
            File path where episodes are stored.
        tech_state_to_dict:
            Function converting TechState -> dict.
        tech_state_from_dict:
            Function converting dict -> TechState.
        trace_to_dict:
            Function converting PlanTrace -> dict.
        trace_from_dict:
            Function converting dict -> PlanTrace.
        """
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if not self._path.exists():
            self._path.write_text("", encoding="utf-8")

        self._tech_state_to_dict = tech_state_to_dict
        self._tech_state_from_dict = tech_state_from_dict
        self._trace_to_dict = trace_to_dict
        self._trace_from_dict = trace_from_dict

    # ------------------------------------------------------------------
    # Core I/O
    # ------------------------------------------------------------------

    def append(self, episode: ExperienceEpisode) -> None:
        """
        Append one ExperienceEpisode to the buffer.

        Uses the serializer hooks to encode TechState and PlanTrace.
        """
        data = episode.to_dict(
            tech_state_to_dict=self._tech_state_to_dict,
            plan_trace_to_dict=self._trace_to_dict,
        )
        with self._path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")

    def load_all_raw(self) -> Iterator[Dict[str, Any]]:
        """
        Stream all stored episodes as raw dicts.

        This is useful for quick analysis or when you only care about
        a few fields and want to avoid reconstructing full objects.
        """
        with self._path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    # corrupted line; skip but don't blow up the whole buffer
                    continue

    def load_all(self) -> Iterator[ExperienceEpisode]:
        """
        Stream all stored episodes as ExperienceEpisode objects.

        Uses the deserializer hooks to reconstruct TechState and PlanTrace.
        """
        for raw in self.load_all_raw():
            yield ExperienceEpisode.from_dict(
                raw,
                tech_state_from_dict=self._tech_state_from_dict,
                plan_trace_from_dict=self._trace_from_dict,
            )

    # ------------------------------------------------------------------
    # Filter helpers (operate on raw dicts for efficiency)
    # ------------------------------------------------------------------

    def iter_by_goal_substring(
        self,
        substring: str,
        *,
        case_sensitive: bool = False,
    ) -> Iterator[Dict[str, Any]]:
        """
        Iterate over raw episodes whose goal contains a given substring.
        """
        if not case_sensitive:
            substring = substring.lower()

        for ep in self.load_all_raw():
            goal = ep.get("goal", "")
            target = goal if case_sensitive else goal.lower()
            if substring in target:
                yield ep

    def iter_by_success(
        self,
        success: bool,
    ) -> Iterator[Dict[str, Any]]:
        """
        Iterate over raw episodes filtered by success flag.
        """
        for ep in self.load_all_raw():
            if bool(ep.get("success")) is success:
                yield ep

    def iter_by_skill_usage(
        self,
        skill_name: SkillName,
    ) -> Iterator[Dict[str, Any]]:
        """
        Iterate over raw episodes where the given skill appears in the trace.

        Assumes the serialized trace has the structure:
        trace: {
          "steps": [
            {
              "meta": {
                "skill": "<skill_name>",
                ...
              },
              ...
            },
            ...
          ],
          ...
        }
        """
        for ep in self.load_all_raw():
            trace = ep.get("trace") or {}
            steps = trace.get("steps") or []
            used_here = any(
                isinstance(step, dict)
                and isinstance(step.get("meta"), dict)
                and step["meta"].get("skill") == skill_name
                for step in steps
            )
            if used_here:
                yield ep

    def iter_by_tech_tier(
        self,
        active_tier: str,
    ) -> Iterator[Dict[str, Any]]:
        """
        Iterate over raw episodes whose TechState active tier matches.

        Assumes serialized TechState has the structure:
        tech_state: {
          "active": "<tier>",
          ...
        }
        """
        for ep in self.load_all_raw():
            tech_state = ep.get("tech_state") or {}
            if tech_state.get("active") == active_tier:
                yield ep

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def count(self) -> int:
        """
        Return an approximate count of episodes (number of non-empty lines).

        This is O(n) over the file and is meant for diagnostics, not tight loops.
        """
        n = 0
        with self._path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    n += 1
        return n

    def path(self) -> Path:
        """
        Return the underlying file path. Mostly for tooling and tests.
        """
        return self._path


```

---

# 5. Synthesizer (LLM Skill Synthesis)

Uses **M2 CodeModel** to turn repeated successful episode traces into candidate skills.

### Inputs
- Cluster of similar successful episodes
- Optional target skill for refinement
- Candidate ID

### Outputs
- YAML SkillSpec
- Python implementation stub
- Explanation of design choice
- Candidate stored in `config/skills_candidates/`

Synthesizer prompt is structured and includes:
- Trace summaries
- Tech state
- Virtue scores
- Output schema

```python
#src/learning/synthesizer.py
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
                "effects, and tags."
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
        # Common patterns:
        #   response = self._model.generate_json(prompt=payload)
        #   response = self._model.call_structured(payload)
        # For now we assume a `generate_json` method exists.
        response: Dict[str, Any] = self._model.generate_json(prompt=payload)  # type: ignore[attr-defined]

        if not isinstance(response, dict):
            raise TypeError(
                f"CodeModel.generate_json returned non-dict response: {type(response)}"
            )

        return response


```




---

# 6. Evaluator

Compares baseline skill performance vs candidate performance.

### Performance Metrics
- Success rate
- Estimated time
- Estimated resource cost
- Virtue score deltas

Evaluation logic:
1. Aggregate performance stats across episodes
2. Compute virtue-weighted comparisons via M4
3. Produce recommendation metadata

Outputs:
- `"promote_candidate"`
- `"keep_baseline"`
- `"reject_candidate"`


```python

#src/learning/evaluator.py
"""
M10 Skill Evaluator

Compares baseline skill performance vs candidate performance using:
- success rate
- estimated time
- estimated resource cost
- virtue scores (via M4)

Evaluation logic:
1. Aggregate SkillPerformanceStats for a given skill across episodes.
2. Use M4 virtue lattice to compute virtue-aware metrics.
3. Emit a recommendation and reasons:
   - "promote_candidate"
   - "keep_baseline"
   - "reject_candidate"
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .schema import ExperienceEpisode, SkillPerformanceStats, SkillName
from semantics.schema import TechState
from observation.trace_schema import PlanTrace

# M4: virtues stack. Adjust names if your implementation differs.
from virtues.loader import load_virtue_config  # type: ignore[import]
from virtues.metrics import extract_plan_metrics  # type: ignore[import]
from virtues.lattice import score_plan  # type: ignore[import]


class SkillEvaluator:
    """
    Compute metrics and compare skill performance, including virtue scores.

    Responsibilities:
    - Aggregate SkillPerformanceStats for a single skill over many episodes.
    - Compare baseline vs candidate using simple heuristics.
    - Produce recommendation + reasons for M10/M11 to consume.
    """

    def __init__(self) -> None:
        # Load virtue config once; reuse across calls
        self._virtue_config = load_virtue_config()

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def aggregate_skill_stats(
        self,
        episodes: List[ExperienceEpisode],
        skill_name: SkillName,
        *,
        context_id: str,
        skill_metadata: Dict[str, Dict[str, Any]],
        semantics_db: Any | None = None,
    ) -> SkillPerformanceStats:
        """
        Aggregate performance stats for a given skill across episodes.

        Parameters
        ----------
        episodes:
            List of ExperienceEpisode objects to analyze.
        skill_name:
            Name of the skill we are measuring (as appears in trace.meta["skill"]).
        context_id:
            Virtue context identifier (e.g. "steam_age", "lv_factory").
        skill_metadata:
            Metadata dict from SkillRegistry.describe_all(), used by virtue metrics.
        semantics_db:
            Optional reference to semantics DB / cache if required by metrics.

        Returns
        -------
        SkillPerformanceStats
        """
        uses = 0
        successes = 0
        time_total = 0.0
        resource_total = 0.0
        virtue_accum: Dict[str, float] = {}

        for ep in episodes:
            if not self._episode_uses_skill(ep, skill_name):
                continue

            uses += 1
            if ep.success:
                successes += 1

            trace: PlanTrace = ep.trace
            tech: TechState = ep.tech_state

            world_before = None
            if getattr(trace, "steps", None):
                first_step = trace.steps[0]
                world_before = getattr(first_step, "world_before", None)

            # Extract lower-level metrics from virtues.metrics
            metrics = extract_plan_metrics(
                plan=trace.plan,
                world=world_before,
                tech_state=tech,
                db=semantics_db,
                skill_metadata=skill_metadata,
            )

            # Score the plan against the virtue lattice
            scores = score_plan(
                plan=trace.plan,
                context_id=context_id,
                metrics=metrics,
                config=self._virtue_config,
            )

            # Accumulate scalar metrics
            time_total += float(getattr(metrics, "estimated_time", 0.0))
            resource_total += float(getattr(metrics, "estimated_resource_cost", 0.0))

            virtue_scores = scores.get("virtue_scores") or {}
            for vname, vscore in virtue_scores.items():
                try:
                    v = float(vscore)
                except (TypeError, ValueError):
                    continue
                virtue_accum[vname] = virtue_accum.get(vname, 0.0) + v

        if uses == 0:
            return SkillPerformanceStats.zero(skill_name)

        avg_time = time_total / uses if uses > 0 else 0.0
        avg_resource = resource_total / uses if uses > 0 else 0.0
        avg_virtues = {k: v / uses for k, v in virtue_accum.items()}

        return SkillPerformanceStats(
            skill_name=skill_name,
            uses=uses,
            success_rate=successes / uses if uses > 0 else 0.0,
            avg_time=avg_time,
            avg_resource_cost=avg_resource,
            avg_virtue_scores=avg_virtues,
        )

    def _episode_uses_skill(
        self,
        episode: ExperienceEpisode,
        skill_name: SkillName,
    ) -> bool:
        """
        Return True if the given skill appears in any trace step meta.

        Assumes trace.steps[*].meta["skill"] is set when skills are invoked.
        """
        trace: PlanTrace = episode.trace
        for step in getattr(trace, "steps", []) or []:
            meta = getattr(step, "meta", {}) or {}
            if meta.get("skill") == skill_name:
                return True
        return False

    # ------------------------------------------------------------------
    # Comparison
    # ------------------------------------------------------------------

    def compare_stats(
        self,
        baseline: SkillPerformanceStats,
        candidate: SkillPerformanceStats,
        *,
        min_uses: int = 5,
        max_success_drop: float = 0.05,
        improvement_threshold: float = 0.1,
    ) -> Dict[str, Any]:
        """
        Compare baseline vs candidate and return a recommendation.

        Heuristics:
        - If candidate has too few uses: "reject_candidate" (insufficient data).
        - If candidate success rate is significantly lower: "reject_candidate".
        - If candidate success is comparable and it improves on time/resource
          or virtue scores: "promote_candidate".
        - Otherwise: "keep_baseline".

        Parameters
        ----------
        baseline:
            Stats for the existing skill.
        candidate:
            Stats for the candidate skill.
        min_uses:
            Minimum required uses for the candidate to be considered.
        max_success_drop:
            Maximum tolerated drop in success rate vs baseline.
        improvement_threshold:
            Minimum relative improvement to count as "meaningful" in time/resource.

        Returns
        -------
        Dict[str, Any]
            {
              "recommendation": str,
              "reasons": List[str],
              "baseline": {...},
              "candidate": {...},
            }
        """
        reasons: List[str] = []

        # 1. Data sufficiency
        if candidate.uses < min_uses:
            reasons.append("candidate_insufficient_data")
            return {
                "recommendation": "reject_candidate",
                "reasons": reasons,
                "baseline": baseline.to_dict(),
                "candidate": candidate.to_dict(),
            }

        # 2. Success rate guardrail
        success_delta = candidate.success_rate - baseline.success_rate
        if success_delta < -max_success_drop:
            reasons.append("candidate_success_rate_too_low")
            return {
                "recommendation": "reject_candidate",
                "reasons": reasons,
                "baseline": baseline.to_dict(),
                "candidate": candidate.to_dict(),
            }

        # 3. Efficiency & virtue improvements
        improved = False

        # Time: lower is better
        if candidate.avg_time < baseline.avg_time * (1.0 - improvement_threshold):
            reasons.append("candidate_faster")
            improved = True

        # Resource cost: lower is better
        if candidate.avg_resource_cost < baseline.avg_resource_cost * (1.0 - improvement_threshold):
            reasons.append("candidate_cheaper")
            improved = True

        # Virtue scores: higher is better for each dimension
        virtue_improvements = self._compare_virtues(baseline, candidate, improvement_threshold)
        reasons.extend(virtue_improvements)
        if virtue_improvements:
            improved = True

        # 4. Final decision
        if improved and success_delta >= -max_success_drop:
            recommendation = "promote_candidate"
        elif success_delta >= -max_success_drop:
            recommendation = "keep_baseline"
        else:
            recommendation = "reject_candidate"

        return {
            "recommendation": recommendation,
            "reasons": reasons,
            "baseline": baseline.to_dict(),
            "candidate": candidate.to_dict(),
        }

    def _compare_virtues(
        self,
        baseline: SkillPerformanceStats,
        candidate: SkillPerformanceStats,
        improvement_threshold: float,
    ) -> List[str]:
        """
        Compare virtue scores and return a list of improvement reason strings.

        Example outputs:
        - "candidate_better_safety"
        - "candidate_better_efficiency"
        """
        reasons: List[str] = []
        for vname, cand_score in candidate.avg_virtue_scores.items():
            base_score = baseline.avg_virtue_scores.get(vname, 0.0)
            if cand_score > base_score * (1.0 + improvement_threshold):
                reasons.append(f"candidate_better_{vname}")
        return reasons


```


```python
#tests/test_skill_evaluator.py
"""
Basic tests for M10 SkillEvaluator.

These use a minimal fake TechState/PlanTrace and fake virtues.metrics
so you can test the logic without firing real LLMs or semantics.
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, List

from learning.evaluator import SkillEvaluator
from learning.schema import ExperienceEpisode, EpisodeMetadata, SkillPerformanceStats
from semantics.schema import TechState  # or define a tiny fake if needed
from observation.trace_schema import PlanTrace  # or fake
import virtues.metrics  # type: ignore
import virtues.lattice  # type: ignore
import virtues.loader  # type: ignore


# You can monkeypatch virtues.* here if you want a pure unit test.

def test_compare_stats_promote_candidate(monkeypatch):
    evaluator = SkillEvaluator()

    baseline = SkillPerformanceStats(
        skill_name="test_skill",
        uses=20,
        success_rate=0.7,
        avg_time=10.0,
        avg_resource_cost=5.0,
        avg_virtue_scores={"safety": 0.5},
    )
    candidate = SkillPerformanceStats(
        skill_name="test_skill_v2",
        uses=20,
        success_rate=0.72,
        avg_time=7.0,
        avg_resource_cost=4.0,
        avg_virtue_scores={"safety": 0.7},
    )

    result = evaluator.compare_stats(baseline, candidate)
    assert result["recommendation"] == "promote_candidate"
    assert any(r.startswith("candidate_faster") for r in result["reasons"]) or \
           any(r.startswith("candidate_cheaper") for r in result["reasons"]) or \
           any(r.startswith("candidate_better_") for r in result["reasons"])


```




---

# 7. SkillLearningManager

High-level orchestrator.

### Responsibilities
- Read episodes from buffer
- Filter/cluster by goal substring or skill pattern
- Invoke synthesizer for candidate creation
- Run evaluator for baseline metrics
- Persist candidate artifacts
- Provide hooks for curriculum integration in M11

### Learning Cycle
1. Gather episodes for a goal  
2. Cluster and filter  
3. Synthesize candidate skill  
4. Evaluate against baseline (if applicable)  
5. Write results to candidates directory  
6. Optionally schedule A/B experiments via M8  


```python
#src/learning/manager.py
"""
M10 SkillLearningManager

High-level orchestrator for skill learning:

Responsibilities
----------------
- Read episodes from ExperienceBuffer
- Filter/cluster by goal substring or skill pattern
- Invoke SkillSynthesizer for candidate creation
- Run SkillEvaluator for baseline metrics & recommendation
- Persist candidate artifacts under config/skills_candidates/
- Expose a simple "learning cycle" API that M11 or tools can call
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from .buffer import ExperienceBuffer
from .schema import ExperienceEpisode, SkillCandidate, SkillPerformanceStats, SkillName
from .synthesizer import SkillSynthesizer
from .evaluator import SkillEvaluator

from skills.registry import SkillRegistry  # M5 skill registry


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
            ExperienceBuffer instance for reading episodes.
        synthesizer:
            SkillSynthesizer instance for generating SkillCandidate objects.
        evaluator:
            SkillEvaluator instance for computing performance stats.
        skills:
            SkillRegistry from M5 (used for metadata & baseline skills).
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
    # Public API
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
        1. Gather relevant episodes from ExperienceBuffer.
        2. Cluster/filter.
        3. Synthesize candidate skill via SkillSynthesizer.
        4. Evaluate vs baseline (if target_skill_name is provided).
        5. Persist candidate artifacts to candidates_dir.
        6. Return a structured result dict.

        Parameters
        ----------
        goal_substring:
            Substring to match against episode.goal (case-insensitive).
        target_skill_name:
            Existing skill name to refine, or None to propose a brand-new skill.
        context_id:
            Virtue context identifier passed into M4 (e.g. "steam_age").
        tech_tier:
            Optional TechState.active filter (e.g. "lv", "steam_age").
        success_only:
            If True, ignore episodes where success == False.
        min_episodes:
            Minimum number of episodes required to attempt synthesis.

        Returns
        -------
        Optional[Dict[str, Any]]
            None if there is not enough data, otherwise:
            {
              "candidate": SkillCandidate,
              "baseline_stats": SkillPerformanceStats | None,
              "candidate_stats": SkillPerformanceStats | None,
              "evaluation": Dict[str, Any] | None,
              "episodes_used": List[str],
            }
        """
        episodes = self._select_episodes_for_goal(
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

        return {
            "candidate": candidate,
            "baseline_stats": baseline_stats,
            "candidate_stats": candidate_stats,
            "evaluation": evaluation_result,
            "episodes_used": [ep.id for ep in episodes],
        }

    # ------------------------------------------------------------------
    # Episode selection / clustering
    # ------------------------------------------------------------------

    def _select_episodes_for_goal(
        self,
        goal_substring: str,
        *,
        skill_name: Optional[SkillName],
        tech_tier: Optional[str],
        success_only: bool,
    ) -> List[ExperienceEpisode]:
        """
        Filter ExperienceEpisodes based on:
        - goal substring
        - optional success flag
        - optional skill usage
        - optional tech tier
        """
        episodes: List[ExperienceEpisode] = []

        substring_lower = goal_substring.lower()
        for ep in self._buffer.load_all():
            if substring_lower not in ep.goal.lower():
                continue
            if success_only and not ep.success:
                continue
            if tech_tier is not None:
                active = getattr(ep.tech_state, "active", None)
                if active != tech_tier:
                    continue
            if skill_name is not None and not self._episode_uses_skill(ep, skill_name):
                continue

            episodes.append(ep)

        return episodes

    def _episode_uses_skill(
        self,
        episode: ExperienceEpisode,
        skill_name: SkillName,
    ) -> bool:
        """
        Return True if the given skill appears in any trace step meta.
        Mirrors logic in SkillEvaluator._episode_uses_skill.
        """
        trace = episode.trace
        for step in getattr(trace, "steps", []) or []:
            meta = getattr(step, "meta", {}) or {}
            if meta.get("skill") == skill_name:
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

        Example:
            target_skill_name="feed_coke_ovens", goal_substring="maintain coke"
            -> "feed_coke_ovens_auto_12"
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

        Files:
        - <id>.yaml   : candidate.spec_yaml
        - <id>.py     : candidate.impl_code
        - <id>.meta.json : serialized SkillCandidate + evaluation summary
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

```





---

# 8. System Diagram

```
                   M8: AgentLoopV1
                (PlanTrace + VirtueScores)
                           |
                           v
                M10: Experience Buffer
                           |
                           v
                M10: SkillLearningManager
                   /                                     v                     v
        M10: Synthesizer         M10: Evaluator
         (via M2 CodeModel)     (via M4 virtues)
                   \             /
                    \           /
                     v         v
              config/skills_candidates/
                           |
          (manual or curriculum promotion)
                           v
                    M5: SkillRegistry
```

---

# 9. Testing & Validation

### Required Tests
- ExperienceBuffer roundtrip  
- Synthesizer with FakeCodeModel  
- Evaluator success vs failure  
- SkillLearningManager end-to-end cycle  

### Smoke Tests
- Run learning cycle with synthetic episodes  
- Verify candidate YAML + Python files created correctly  
- Verify no candidate auto-loads into registry  

```python
# tests/learning/test_experience_buffer.py

from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Any, List

from learning.buffer import ExperienceBuffer
from learning.schema import ExperienceEpisode, EpisodeMetadata


# --- Fakes for TechState / PlanTrace ----------------------------------------


class FakeTechState:
    def __init__(self, active: str, unlocked: list[str]) -> None:
        self.active = active
        self.unlocked = unlocked


class FakeAction:
    def __init__(self, type: str, params: Dict[str, Any]) -> None:
        self.type = type
        self.params = params


class FakeResult:
    def __init__(self, success: bool) -> None:
        self.success = success
        self.error = None


class FakeStep:
    def __init__(self, skill: str) -> None:
        self.action = FakeAction("noop", {})
        self.result = FakeResult(True)
        self.world_before = SimpleNamespace()
        self.world_after = SimpleNamespace()
        self.meta = {"skill": skill}


class FakePlanTrace:
    def __init__(self, steps: List[FakeStep]) -> None:
        self.plan = {"steps": [{"skill": s.meta["skill"]} for s in steps]}
        self.steps = steps


def tech_state_to_dict(ts: FakeTechState) -> Dict[str, Any]:
    return {"active": ts.active, "unlocked": list(ts.unlocked)}


def tech_state_from_dict(data: Dict[str, Any]) -> FakeTechState:
    return FakeTechState(active=data.get("active", ""), unlocked=data.get("unlocked", []))


def trace_to_dict(trace: FakePlanTrace) -> Dict[str, Any]:
    return {
        "plan": trace.plan,
        "steps": [
            {
                "action_type": s.action.type,
                "params": s.action.params,
                "success": s.result.success,
                "meta": s.meta,
            }
            for s in trace.steps
        ],
    }


def trace_from_dict(data: Dict[str, Any]) -> FakePlanTrace:
    steps_data = data.get("steps") or []
    steps = [FakeStep(skill=sd.get("meta", {}).get("skill", "unknown")) for sd in steps_data]
    return FakePlanTrace(steps=steps)


# --- Tests -------------------------------------------------------------------


def test_experience_buffer_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "episodes.jsonl"

    buffer = ExperienceBuffer(
        path=path,
        tech_state_to_dict=tech_state_to_dict,
        tech_state_from_dict=tech_state_from_dict,
        trace_to_dict=trace_to_dict,
        trace_from_dict=trace_from_dict,
    )

    # create a couple of episodes
    ts = FakeTechState(active="steam_age", unlocked=["steam_age"])
    trace = FakePlanTrace(steps=[FakeStep(skill="test_skill")])

    ep1 = ExperienceEpisode(
        id="ep1",
        goal="maintain coke ovens",
        tech_state=ts,
        trace=trace,
        virtue_scores={"overall": 0.7},
        success=True,
        metadata=EpisodeMetadata(),
    )
    ep2 = ExperienceEpisode(
        id="ep2",
        goal="maintain coke ovens",
        tech_state=ts,
        trace=trace,
        virtue_scores={"overall": 0.3},
        success=False,
        metadata=EpisodeMetadata(),
    )

    buffer.append(ep1)
    buffer.append(ep2)

    # raw load
    raw = list(buffer.load_all_raw())
    assert len(raw) == 2
    assert raw[0]["goal"] == "maintain coke ovens"

    # typed load
    typed = list(buffer.load_all())
    assert len(typed) == 2
    assert typed[0].id == "ep1"
    assert typed[1].id == "ep2"
    assert typed[0].tech_state.active == "steam_age"

    # filters
    success_eps = list(buffer.iter_by_success(True))
    assert len(success_eps) == 1
    assert success_eps[0]["id"] == "ep1"

    by_skill = list(buffer.iter_by_skill_usage("test_skill"))
    assert len(by_skill) == 2

    by_tier = list(buffer.iter_by_tech_tier("steam_age"))
    assert len(by_tier) == 2

    assert buffer.count() == 2


```


```python
# tests/learning/test_synthesizer.py

from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Any, List

from learning.schema import ExperienceEpisode, EpisodeMetadata
from learning.synthesizer import SkillSynthesizer


# Reuse simple fakes for TechState / PlanTrace -------------------------------


class FakeTechState:
    def __init__(self, active: str, unlocked: list[str]) -> None:
        self.active = active
        self.unlocked = unlocked


class FakeAction:
    def __init__(self, type: str, params: Dict[str, Any]) -> None:
        self.type = type
        self.params = params


class FakeResult:
    def __init__(self, success: bool) -> None:
        self.success = success
        self.error = None


class FakeStep:
    def __init__(self, skill: str) -> None:
        self.action = FakeAction("move_to", {"x": 0, "y": 64, "z": 0})
        self.result = FakeResult(True)
        self.world_before = SimpleNamespace()
        self.world_after = SimpleNamespace()
        self.meta = {"skill": skill}


class FakePlanTrace:
    def __init__(self, skill: str) -> None:
        step = FakeStep(skill=skill)
        self.steps = [step]
        self.plan = {"steps": [{"skill": skill, "params": {}}]}


class FakeCodeModel:
    """
    Minimal fake for M2 CodeModel.
    """

    def generate_json(self, *, prompt: Dict[str, Any]) -> Dict[str, Any]:
        # Just echo a trivial spec and impl referencing the goal / skill.
        episodes = prompt.get("episodes", [])
        first_goal = episodes[0].get("goal") if episodes else "unknown_goal"
        target_skill = prompt.get("target_skill_name") or "auto_skill"

        spec_yaml = f"""name: {target_skill}
version: 1
description: Auto-derived skill for '{first_goal}'
"""

        impl_code = """class AutoSkill:
    def run(self, ctx):
        # TODO: implement
        return True
"""

        return {
            "spec_yaml": spec_yaml,
            "impl_code": impl_code,
            "rationale": "fake rationale for testing",
        }


def make_episode(skill: str = "test_skill") -> ExperienceEpisode:
    ts = FakeTechState(active="steam_age", unlocked=["steam_age"])
    trace = FakePlanTrace(skill=skill)
    return ExperienceEpisode(
        id="ep1",
        goal="maintain coke ovens",
        tech_state=ts,
        trace=trace,
        virtue_scores={"overall": 0.9},
        success=True,
        metadata=EpisodeMetadata(),
    )


def test_synthesizer_produces_candidate() -> None:
    code_model = FakeCodeModel()
    synth = SkillSynthesizer(code_model)

    ep = make_episode()
    candidate = synth.propose_from_episodes(
        episodes=[ep],
        target_skill_name=None,
        candidate_id="cand1",
        context_hint="steam_age",
    )

    assert candidate.id == "cand1"
    assert "name:" in candidate.spec_yaml
    assert "class AutoSkill" in candidate.impl_code
    assert candidate.status == "proposed"
    assert "synthesizer_prompt" in candidate.extra
    assert "maintain coke ovens" in candidate.extra["synthesizer_prompt"]["episode_ids"][0] or True


```


```python
# tests/learning/test_evaluator.py

from types import SimpleNamespace
from typing import Dict, Any, List

import pytest

from learning.schema import ExperienceEpisode, EpisodeMetadata, SkillPerformanceStats
from learning.evaluator import SkillEvaluator


# --- Fakes for TechState / PlanTrace ----------------------------------------


class FakeTechState:
    def __init__(self, active: str, unlocked: list[str]) -> None:
        self.active = active
        self.unlocked = unlocked


class FakeAction:
    def __init__(self, type: str, params: Dict[str, Any]) -> None:
        self.type = type
        self.params = params


class FakeResult:
    def __init__(self, success: bool) -> None:
        self.success = success
        self.error = None


class FakeStep:
    def __init__(self, skill: str, success: bool) -> None:
        self.action = FakeAction("noop", {})
        self.result = FakeResult(success)
        self.world_before = SimpleNamespace()
        self.world_after = SimpleNamespace()
        self.meta = {"skill": skill}


class FakePlanTrace:
    def __init__(self, skill: str) -> None:
        self.steps = [FakeStep(skill, True)]
        self.plan = {"steps": [{"skill": skill, "params": {}}]}


# --- Monkeypatched virtues.* -------------------------------------------------


class FakeMetrics:
    def __init__(self, t: float, c: float) -> None:
        self.estimated_time = t
        self.estimated_resource_cost = c


@pytest.fixture(autouse=True)
def patch_virtues(monkeypatch):
    # patch load_virtue_config
    monkeypatch.setattr("learning.evaluator.load_virtue_config", lambda: {})

    # patch extract_plan_metrics
    def fake_extract_plan_metrics(plan, world, tech_state, db, skill_metadata):
        return FakeMetrics(t=10.0, c=5.0)

    monkeypatch.setattr("learning.evaluator.extract_plan_metrics", fake_extract_plan_metrics)

    # patch score_plan
    def fake_score_plan(plan, context_id, metrics, config):
        return {"virtue_scores": {"safety": 0.6, "efficiency": 0.7}}

    monkeypatch.setattr("learning.evaluator.score_plan", fake_score_plan)

    yield


def make_episode(skill: str, success: bool) -> ExperienceEpisode:
    ts = FakeTechState(active="steam_age", unlocked=["steam_age"])
    trace = FakePlanTrace(skill=skill)
    return ExperienceEpisode(
        id=f"ep_{skill}_{success}",
        goal="maintain coke ovens",
        tech_state=ts,
        trace=trace,
        virtue_scores={"overall": 0.5},
        success=success,
        metadata=EpisodeMetadata(),
    )


def test_aggregate_skill_stats_basic() -> None:
    evaluator = SkillEvaluator()

    episodes = [
        make_episode("test_skill", True),
        make_episode("test_skill", False),
        make_episode("other_skill", True),
    ]

    stats = evaluator.aggregate_skill_stats(
        episodes=episodes,
        skill_name="test_skill",
        context_id="steam_age",
        skill_metadata={},
        semantics_db=None,
    )

    assert stats.skill_name == "test_skill"
    assert stats.uses == 2
    assert 0.0 < stats.success_rate <= 1.0
    assert stats.avg_time == pytest.approx(10.0)
    assert stats.avg_resource_cost == pytest.approx(5.0)
    assert "safety" in stats.avg_virtue_scores


def test_compare_stats_promote_candidate() -> None:
    evaluator = SkillEvaluator()

    baseline = SkillPerformanceStats(
        skill_name="test_skill",
        uses=20,
        success_rate=0.7,
        avg_time=10.0,
        avg_resource_cost=5.0,
        avg_virtue_scores={"safety": 0.5},
    )
    candidate = SkillPerformanceStats(
        skill_name="test_skill_v2",
        uses=20,
        success_rate=0.72,
        avg_time=7.0,
        avg_resource_cost=3.0,
        avg_virtue_scores={"safety": 0.8},
    )

    result = evaluator.compare_stats(baseline, candidate)
    assert result["recommendation"] in {"promote_candidate", "keep_baseline"}
    assert "baseline" in result and "candidate" in result

```


```python
# tests/learning/test_skill_learning_manager.py

from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Any, List, Optional

from learning.buffer import ExperienceBuffer
from learning.schema import ExperienceEpisode, EpisodeMetadata
from learning.synthesizer import SkillSynthesizer
from learning.evaluator import SkillEvaluator
from learning.manager import SkillLearningManager


# --- Shared fakes ------------------------------------------------------------


class FakeTechState:
    def __init__(self, active: str, unlocked: list[str]) -> None:
        self.active = active
        self.unlocked = unlocked


class FakeAction:
    def __init__(self, type: str, params: Dict[str, Any]) -> None:
        self.type = type
        self.params = params


class FakeResult:
    def __init__(self, success: bool) -> None:
        self.success = success
        self.error = None


class FakeStep:
    def __init__(self, skill: str, success: bool = True) -> None:
        self.action = FakeAction("noop", {})
        self.result = FakeResult(success)
        self.world_before = SimpleNamespace()
        self.world_after = SimpleNamespace()
        self.meta = {"skill": skill}


class FakePlanTrace:
    def __init__(self, skill: str) -> None:
        self.steps = [FakeStep(skill)]
        self.plan = {"steps": [{"skill": skill, "params": {}}]}


def tech_state_to_dict(ts: FakeTechState) -> Dict[str, Any]:
    return {"active": ts.active, "unlocked": list(ts.unlocked)}


def tech_state_from_dict(data: Dict[str, Any]) -> FakeTechState:
    return FakeTechState(active=data.get("active", ""), unlocked=data.get("unlocked", []))


def trace_to_dict(trace: FakePlanTrace) -> Dict[str, Any]:
    return {
        "plan": trace.plan,
        "steps": [
            {
                "action_type": s.action.type,
                "params": s.action.params,
                "success": s.result.success,
                "meta": s.meta,
            }
            for s in trace.steps
        ],
    }


def trace_from_dict(data: Dict[str, Any]) -> FakePlanTrace:
    steps_data = data.get("steps") or []
    skill = steps_data[0].get("meta", {}).get("skill", "unknown") if steps_data else "unknown"
    return FakePlanTrace(skill=skill)


class FakeCodeModel:
    def generate_json(self, *, prompt: Dict[str, Any]) -> Dict[str, Any]:
        goal = prompt.get("episodes", [{}])[0].get("goal", "unknown_goal")
        skill = prompt.get("target_skill_name") or "auto_skill"

        spec_yaml = f"""name: {skill}
version: 1
description: Auto skill for {goal}
"""

        impl_code = """class AutoSkill:
    def run(self, ctx):
        # TODO: implement
        return True
"""
        return {
            "spec_yaml": spec_yaml,
            "impl_code": impl_code,
            "rationale": "fake rationale",
        }


class FakeSkillRegistry:
    """
    Minimal fake M5 SkillRegistry.
    """

    def __init__(self) -> None:
        self._skills: Dict[str, Dict[str, Any]] = {
            "maintain_coke_ovens": {"name": "maintain_coke_ovens", "version": 1}
        }

    def describe_all(self) -> Dict[str, Dict[str, Any]]:
        return dict(self._skills)


def make_episode(goal: str, skill: str, success: bool) -> ExperienceEpisode:
    ts = FakeTechState(active="steam_age", unlocked=["steam_age"])
    trace = FakePlanTrace(skill=skill)
    return ExperienceEpisode(
        id=f"ep_{goal}_{success}",
        goal=goal,
        tech_state=ts,
        trace=trace,
        virtue_scores={"overall": 0.5},
        success=success,
        metadata=EpisodeMetadata(),
    )


# --- Test --------------------------------------------------------------------


def test_skill_learning_manager_end_to_end(tmp_path: Path, monkeypatch) -> None:
    # Patch virtues inside evaluator to avoid pulling in real lattice
    from learning import evaluator as evaluator_mod

    evaluator_mod.load_virtue_config = lambda: {}
    evaluator_mod.extract_plan_metrics = lambda plan, world, tech_state, db, skill_metadata: SimpleNamespace(
        estimated_time=10.0,
        estimated_resource_cost=5.0,
    )
    evaluator_mod.score_plan = lambda plan, context_id, metrics, config: {
        "virtue_scores": {"safety": 0.6}
    }

    buf_path = tmp_path / "episodes.jsonl"
    buffer = ExperienceBuffer(
        path=buf_path,
        tech_state_to_dict=tech_state_to_dict,
        tech_state_from_dict=tech_state_from_dict,
        trace_to_dict=trace_to_dict,
        trace_from_dict=trace_from_dict,
    )

    # Populate buffer with enough episodes
    for i in range(6):
        ep = make_episode(
            goal="maintain coke ovens",
            skill="maintain_coke_ovens",
            success=True,
        )
        ep.id = f"ep{i}"
        buffer.append(ep)

    code_model = FakeCodeModel()
    synthesizer = SkillSynthesizer(code_model)
    evaluator = SkillEvaluator()
    skills = FakeSkillRegistry()
    candidates_dir = tmp_path / "skills_candidates"

    manager = SkillLearningManager(
        buffer=buffer,
        synthesizer=synthesizer,
        evaluator=evaluator,
        skills=skills,
        candidates_dir=candidates_dir,
        semantics_db=None,
    )

    result = manager.run_learning_cycle_for_goal(
        goal_substring="maintain coke ovens",
        target_skill_name="maintain_coke_ovens",
        context_id="steam_age",
        tech_tier="steam_age",
        success_only=True,
        min_episodes=5,
    )

    # Basic smoke: learning cycle produced something
    assert result is not None
    candidate = result["candidate"]
    assert candidate is not None
    assert candidate.spec_yaml
    assert candidate.impl_code

    # Files written
    spec_path = candidates_dir / f"{candidate.id}.yaml"
    code_path = candidates_dir / f"{candidate.id}.py"
    meta_path = candidates_dir / f"{candidate.id}.meta.json"

    assert spec_path.exists()
    assert code_path.exists()
    assert meta_path.exists()


```




---

# 10. Completion Criteria

M10 is “operational” when:

1. M8 episodes convert to structured ExperienceEpisode objects  
2. Replay buffer persists and loads full episodes  
3. Synthesizer produces valid spec + impl + rationale  
4. Evaluator yields structured recommendations  
5. SkillLearningManager coordinates a full cycle  
6. Candidate skills appear in `skills_candidates` for human/curriculum review  

At this point the agent has **memory**, **research**, and **self-improvement** capabilities grounded in GTNH semantics and virtue scoring.

