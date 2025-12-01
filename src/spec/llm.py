from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Dict, Any, List, Optional

from .types import Observation  # kept for backwards-compatibility / future use
from observation.trace_schema import PlanTrace


# ======================================================================
# Q1.6 – ROLE-SPECIFIC API DEFINITIONS
# ======================================================================

# NOTE:
# - We keep these role-specific Protocols as the *current* design.
# - A minimal legacy `CodeModel` is also provided so older imports
#   (e.g. spec.__init__) keep working without touching the rest of
#   the codebase.


# ---------------------------------------------------------
# PLANNER ROLE (Goal → TaskPlan)
# ---------------------------------------------------------

@dataclass
class PlannerRequest:
    """
    Request payload for the Planner role.

    goal:
        AgentGoal-like object (spec.agent_loop.AgentGoal); referenced
        as a forward annotation to avoid circular imports.
    world_summary:
        Dict containing observation, skills, constraints, etc.
        (See AgentLoop._build_world_summary for shape.)
    """
    goal: "AgentGoal"
    world_summary: Dict[str, Any]


@dataclass
class PlannerTaskJSON:
    """
    Canonical JSON-safe representation of a single task in a goal plan.

    Intended to be convertible into spec.agent_loop.Task.
    """
    id: str
    description: str


@dataclass
class PlannerResponse:
    """
    Planner LLM response:

        - tasks: list of tasks (required)
        - notes: optional explanation / ordering hints
        - raw_text: raw model output (for debugging)
        - error: optional parse / contract error description
    """
    tasks: List[PlannerTaskJSON]
    notes: str = ""
    raw_text: str = ""
    error: Optional[str] = None


class PlannerModel(Protocol):
    """LLM role for Goal → Task decomposition."""

    def plan(  # legacy signature used in some Phase 0/1 code
        self,
        observation: Observation,
        goal: str,
        skill_descriptions: Dict[str, Dict[str, Any]],
        constraints: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Return a JSON-ish plan dict. Newer code should prefer
        PlannerRequest/PlannerResponse, but many callsites still expect
        this simpler signature.
        """
        ...

    # Future-facing shape (not enforced yet):
    # def plan_goal(self, req: PlannerRequest) -> PlannerResponse: ...


# ---------------------------------------------------------
# PLAN-CODE ROLE (Task → SkillInvocations)
# ---------------------------------------------------------

@dataclass
class PlanCodeRequest:
    """
    Request payload for PlanCodeModel (Task → Skills).

    task:
        JSON-safe representation of a Task (id, description, etc.).
    world_summary:
        Same world summary structure as for PlannerRequest.
    """
    task: Dict[str, Any]
    world_summary: Dict[str, Any]


@dataclass
class PlanCodeStepJSON:
    """
    Canonical JSON-safe representation of a single skill step.

    This should map cleanly into spec.skills.SkillInvocation.
    """
    skill: str
    params: Dict[str, Any] = field(default_factory=dict)
    expected_outcome: Optional[str] = None


# Backwards-compat aliases for older plan-code stack
# llm_stack.plan_code imports PlannerPlanJSON / PlannerStepJSON.
PlannerStepJSON = PlanCodeStepJSON


@dataclass
class PlannerPlanJSON:
    """
    Canonical JSON-safe representation of a *plan* used by the plan-code stack.

    This is what PlanCodeModelImpl._parse_plan_response() returns and then
    re-flattens into the dict shape the rest of the system uses:

        {
          "steps": [
            {"skill": "name", "params": {...}, "expected_outcome": "..."},
            ...
          ],
          "notes": "short explanation",
          "raw_text": "...",
          "error": "optional_error_code" | None
        }
    """
    steps: List[PlannerStepJSON]
    notes: str = ""
    raw_text: str = ""
    error: Optional[str] = None


@dataclass
class PlanCodeResponse:
    """
    Response from PlanCodeModel:

        - steps: ordered list of skill calls
        - notes: optional commentary / rationale
        - raw_text: raw LLM output
        - error: optional error string on protocol violations
    """
    steps: List[PlanCodeStepJSON]
    notes: str = ""
    raw_text: str = ""
    error: Optional[str] = None


class PlanCodeModel(Protocol):
    """LLM role for Task → Skills decomposition."""

    # Newer shape (not yet used everywhere):
    def plan_task(self, req: PlanCodeRequest) -> PlanCodeResponse:  # pragma: no cover - protocol
        ...

    # Older shape used by PlanCodeModelImpl:
    def propose_skill_implementation(
        self,
        skill_spec: Dict[str, Any],
        examples: List[Dict[str, Any]],
    ) -> str:  # pragma: no cover - protocol
        ...


# ----------------------------------------------------------------------
# Legacy CodeModel (M1-style) – kept for backwards compatibility
# ----------------------------------------------------------------------

class CodeModel(Protocol):
    """
    Legacy code-generation interface.

    Old code may import CodeModel from spec.llm or spec.__init__.
    New work should prefer PlanCodeModel and role-specific facades,
    but this keeps imports and type hints from exploding.
    """

    def propose_skill_implementation(
        self,
        skill_spec: Dict[str, Any],
        examples: List[Dict[str, Any]],
    ) -> str:
        """
        Given a skill specification and optional example traces,
        return Python code implementing the skill.
        """
        ...


# ---------------------------------------------------------
# CRITIC ROLE (pre-execution evaluation)
# ---------------------------------------------------------

@dataclass
class CriticRequest:
    """
    Request for CriticModel.

    trace:
        PlanTrace representing the planned episode before execution,
        typically built from planner output + current observation.
    context_id:
        Virtue / curriculum context identifier (e.g. "steam_age").
    """
    trace: PlanTrace
    context_id: str


@dataclass
class CriticResponse:
    """
    Canonical critic result shape.

    Tests construct it like:

        CriticResponse(
            ok=False,
            critique="too risky",
            suggested_modifications={"lower_risk": True},
            failure_type="risk_violation",
            severity="high",
            fix_suggestions=["reduce risk", "use safer setup"],
            raw_text="...",
        )

    Fields:
        ok:
            Whether the plan is acceptable as-is.
        summary / critique:
            Human-readable critique / justification. Both names are
            kept for compatibility and kept in sync.
        suggested_modifications:
            Optional machine-friendly patch / metadata.
        failure_type:
            Optional failure classification (e.g. "safety", "wasteful").
        severity:
            Optional severity label ("low", "medium", "high", etc.).
        fix_suggestions:
            Optional list of concrete fix suggestions.
        raw_text:
            Raw LLM output (for debugging / replay).
    """
    ok: bool
    # Keep both names so old and new call sites work.
    summary: str = ""
    critique: str = ""
    suggested_modifications: Dict[str, Any] = field(default_factory=dict)
    failure_type: Optional[str] = None
    severity: Optional[str] = None
    fix_suggestions: List[str] = field(default_factory=list)
    raw_text: str = ""

    def __post_init__(self) -> None:
        # Mirror summary/critique for compatibility.
        if not self.summary and self.critique:
            self.summary = self.critique
        elif self.summary and not self.critique:
            self.critique = self.summary


class CriticModel(Protocol):
    """LLM role for evaluating a plan before execution."""

    def evaluate(self, req: CriticRequest) -> CriticResponse:  # pragma: no cover - protocol
        ...


# ---------------------------------------------------------
# ERROR MODEL ROLE (post-execution evaluation)
# ---------------------------------------------------------

@dataclass
class ErrorContext:
    """
    Context passed into the ErrorModel.

    Tests construct it like:

        ErrorContext(
            role="plan_code",
            operation="plan",
            prompt="dummy prompt",
            raw_response="dummy raw",
            error_type="json_decode_error",
            metadata={"test": True},
        )
    """
    role: str
    operation: str
    prompt: str
    raw_response: str
    error_type: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    # Optional extras for future-proofing
    model: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ErrorContext":
        return cls(
            role=data.get("role", ""),
            operation=data.get("operation", ""),
            prompt=data.get("prompt", ""),
            raw_response=data.get("raw_response", ""),
            error_type=data.get("error_type", ""),
            metadata=data.get("metadata", {}) or {},
            model=data.get("model", ""),
            extra={
                k: v
                for k, v in data.items()
                if k
                not in {
                    "role",
                    "operation",
                    "prompt",
                    "raw_response",
                    "error_type",
                    "metadata",
                    "model",
                }
            },
        )


@dataclass
class ErrorModelRequest:
    """
    Request for ErrorModel.

    trace:
        PlanTrace with execution results (including failures).
    context_id:
        Same context identifier used for virtues / curriculum.
    """
    trace: PlanTrace
    context_id: str


@dataclass
class ErrorModelResponse:
    """
    Canonical error-model result shape.

    Tests construct it like:

        ErrorModelResponse(
            classification="execution_failure",
            summary="bot fell into lava",
            suggested_fix={"hint": "avoid open lava path"},
            retry_advised=False,
            failure_type="execution_failure",
            severity="high",
            fix_suggestions=["adjust pathfinding", "build guard rails"],
            raw_text="...",
        )

    Fields align with CriticResponse so both can plug into the
    same evaluation / logging surfaces:

      - failure_type
      - severity
      - fix_suggestions

    Extra legacy/infra fields:
      - classification, suggested_fix, retry_advised, ok, message, metadata
    """
    # Infra / classification-level view
    classification: str = ""
    summary: str = ""
    suggested_fix: Dict[str, Any] = field(default_factory=dict)
    retry_advised: bool = False

    # Shared evaluation surface with CriticResponse
    failure_type: str = ""
    severity: str = ""
    fix_suggestions: List[str] = field(default_factory=list)
    raw_text: str = ""

    # Optional extras
    ok: bool = False
    message: str = ""
    context: Optional[ErrorContext] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ErrorModelResponse":
        ctx_data = data.get("context")
        ctx: Optional[ErrorContext]
        if isinstance(ctx_data, dict):
            ctx = ErrorContext.from_dict(ctx_data)
        else:
            ctx = None

        return cls(
            classification=data.get("classification", ""),
            summary=data.get("summary", data.get("message", "")) or "",
            suggested_fix=data.get("suggested_fix", {}) or {},
            retry_advised=bool(data.get("retry_advised", False)),
            failure_type=data.get("failure_type", data.get("error_type", "")) or "",
            severity=data.get("severity", "") or "",
            fix_suggestions=list(data.get("fix_suggestions", []) or []),
            raw_text=data.get("raw_text", data.get("raw", "")) or "",
            ok=bool(data.get("ok", False)),
            message=data.get("message", "") or "",
            context=ctx,
            metadata=data.get("metadata", {}) or {},
        )


class ErrorModel(Protocol):
    """LLM role for post-execution error analysis."""

    def evaluate(self, req: ErrorModelRequest) -> ErrorModelResponse:  # pragma: no cover - protocol
        ...


# ---------------------------------------------------------
# SCRIBE ROLE (episode summarization)
# ---------------------------------------------------------

@dataclass
class TraceSummaryRequest:
    """
    Request type used by the scribe / trace summarizer.

    Tests construct it like:

        TraceSummaryRequest(
            trace=trace,
            purpose="context_chunk",
        )
    """
    trace: Dict[str, Any]
    purpose: str


@dataclass
class TraceSummaryResponse:
    """
    Shape for what the scribe model returns.

    This is mainly used inside llm_stack; tests interact via
    summary / keywords / suggested_tags / raw_text.
    """
    summary: str
    keywords: List[str]
    suggested_tags: List[str]
    raw_text: str


@dataclass
class ScribeRequest:
    """
    Request for ScribeModel.

    trace:
        JSON-safe episode / experience representation.
    purpose:
        Short hint for style / focus (e.g. "human_summary",
        "replay_buffer", "debug_trace").
    """
    trace: Dict[str, Any]
    purpose: str


@dataclass
class ScribeResponse:
    """
    Scribe summarization output.

    summary:
        Compressed natural-language description.
    keywords:
        Key terms extracted from the trace.
    suggested_tags:
        High-level tags useful for indexing / search.
    raw_text:
        Raw LLM output.
    """
    summary: str
    keywords: List[str]
    suggested_tags: List[str]
    raw_text: str


class ScribeModel(Protocol):
    """LLM role for compressing/explaining episodes."""

    def summarize(self, req: ScribeRequest) -> ScribeResponse:  # pragma: no cover - protocol
        ...

