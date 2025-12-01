# src/llm_stack/schema.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


# ============================================================================
# Error model schemas
# ============================================================================

@dataclass
class ErrorContext:
    """
    Context for a single LLM call failure.

    Tests construct it like:

        ErrorContext(
            role="plan_code",
            operation="plan",
            prompt="dummy prompt",
            raw_response="dummy raw",
            error_type="json_decode_error",
            metadata={"test": True},
        )

    This is kept small and focused so it lines up with both the tests and
    ErrorModelImpl._build_prompt().
    """
    role: str
    operation: str
    prompt: str
    raw_response: str
    error_type: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ErrorAnalysis:
    """
    Canonical structured error-analysis response.

    This is what ErrorModelImpl.analyze_failure() returns on the
    legacy / infra-level path:

        ErrorAnalysis(
            classification="execution_failure",
            summary="bot fell into lava",
            suggested_fix={"hint": "avoid open lava path"},
            retry_advised=False,
            raw_text="...",
        )

    Extra Q1 fields (failure_type / severity / fix_suggestions) are
    available for callers that want to harmonize shapes with
    CriticResponse / ErrorModelResponse, but tests only rely on the
    primary five fields.
    """
    classification: str
    summary: str
    suggested_fix: Dict[str, Any] = field(default_factory=dict)
    retry_advised: bool = False
    raw_text: str = ""

    # Optional Q1 failure-surface harmonization fields
    failure_type: str | None = None
    severity: str | None = None
    fix_suggestions: List[str] = field(default_factory=list)


# ============================================================================
# Scribe / trace-summary schemas
# ============================================================================

@dataclass
class TraceSummaryRequest:
    """
    Request payload for the Scribe / trace summarizer.

    Tests construct it like:

        TraceSummaryRequest(
            trace=trace,
            purpose="context_chunk",
        )

    Fields:
        trace:
            JSON-safe representation of the episode / trace.
        purpose:
            Short hint for summarization style / focus.
        metadata:
            Optional extra context for higher-level routing.
    """
    trace: Any
    purpose: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TraceSummaryResponse:
    """
    Response payload for the Scribe role.

    Intended usage:
      - summary: compressed natural-language description
      - keywords: short keyword list for indexing / search
      - suggested_tags: tags for curriculum / monitoring
      - raw_text: raw LLM output
    """
    summary: str
    keywords: List[str] = field(default_factory=list)
    suggested_tags: List[str] = field(default_factory=list)
    raw_text: str = ""

