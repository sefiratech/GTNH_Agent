# request/response types for planner, codegen, critic
# src/llm_stack/schema.py

from dataclasses import dataclass
from typing import Dict, Any, List


# --- Planning & Codegen (PlanCodeModel) ---

@dataclass
class PlanRequest:
    """Input to the planning side of PlanCodeModel."""
    observation: Dict[str, Any]             # pre-encoded state (from M7)
    goal: str                               # top-level goal description
    skills: Dict[str, Dict[str, Any]]       # skill metadata (from SkillRegistry)
    constraints: Dict[str, Any]             # time/resources/other limits


@dataclass
class PlanStep:
    """One step in a plan, typically a skill invocation."""
    skill: str
    params: Dict[str, Any]


@dataclass
class PlanResponse:
    """Output from the planning side of PlanCodeModel."""
    steps: List[PlanStep]                   # ordered sequence of steps
    notes: str                              # optional textual notes/comments
    raw_text: str                           # raw LLM output for debugging


@dataclass
class SkillImplRequest:
    """Input for skill/code generation."""
    skill_spec: Dict[str, Any]              # name, description, parameters, etc.
    examples: List[Dict[str, Any]]          # traces, snippets, example usages


@dataclass
class SkillImplResponse:
    """Output of skill implementation generation."""
    code: str                               # generated or updated code
    notes: str                              # comments or rationale
    raw_text: str                           # full LLM output for forensics


# --- ErrorModel ---

@dataclass
class ErrorContext:
    """Context about a failed or suspicious LLM interaction."""
    role: str                               # e.g. "plan_code", "scribe"
    operation: str                          # e.g. "plan", "codegen", "summarize"
    prompt: str                             # prompt sent to the model
    raw_response: str                       # response received (if any)
    error_type: str                         # e.g. "json_decode_error", "timeout"
    metadata: Dict[str, Any]                # call params, timing, etc.


@dataclass
class ErrorAnalysis:
    """Structured result from ErrorModel."""
    classification: str                     # e.g. "parsing_error", "hallucination"
    summary: str                            # human-readable explanation
    suggested_fix: Dict[str, Any]           # e.g. new stop tokens, shorter input
    retry_advised: bool                     # whether a retry is recommended
    raw_text: str                           # raw LLM output


# --- ScribeModel ---

@dataclass
class TraceSummaryRequest:
    """Input for log/trace summarization."""
    trace: Dict[str, Any]                   # plan + actions + outcomes, etc.
    purpose: str                            # "human_doc", "context_chunk", "debug"


@dataclass
class TraceSummaryResponse:
    """Output from ScribeModel."""
    summary: str                            # compressed narrative summary
    keywords: List[str]                     # keywords for indexing
    suggested_tags: List[str]               # tags/categories (e.g. "LV", "boilers")
    raw_text: str                           # raw LLM output

