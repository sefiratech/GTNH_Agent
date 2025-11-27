# src/spec/llm.py

from __future__ import annotations

from typing import Protocol, Dict, Any, List

from .types import Observation


# ---------------------------------------------------------------------------
# Legacy / M1 interfaces (Planner / Code / Critic)
# ---------------------------------------------------------------------------

class PlannerModel(Protocol):
    """High-level planner interface (M1 legacy)."""

    def plan(
        self,
        observation: Observation,
        goal: str,
        skill_descriptions: Dict[str, Dict[str, Any]],
        constraints: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Given an observation and goal, return a structured plan.

        The exact structure is left to the implementation, but typically:
        {
          "steps": [...],
          "raw_text": "...",
          ...
        }
        """
        ...


class CodeModel(Protocol):
    """Code / skill implementation model interface (M1 legacy)."""

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


class CriticModel(Protocol):
    """Plan / behavior critic interface (M1 legacy)."""

    def evaluate_plan(
        self,
        observation: Observation,
        plan: Dict[str, Any],
        virtue_scores: Dict[str, float],
    ) -> Dict[str, Any]:
        """
        Given an observation, plan, and virtue scores, return a critique.

        Typical output:
        {
          "ok": bool,
          "critique": "text",
          "suggested_modifications": {...},
          "raw_text": "..."
        }
        """
        ...


# ---------------------------------------------------------------------------
# M2 interfaces (single-model multi-role: PlanCode / Error / Scribe)
# ---------------------------------------------------------------------------

class PlanCodeModel(Protocol):
    """
    Unified planning + codegen interface over a single underlying model.

    Implemented by llm_stack.plan_code.PlanCodeModelImpl.
    """

    def plan(
        self,
        observation: Observation,
        goal: str,
        skill_descriptions: Dict[str, Dict[str, Any]],
        constraints: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Produce a structured plan dict.

        M2 convention (PlanCodeModelImpl):
        {
          "steps": [
            {"skill": "name", "params": {...}},
            ...
          ],
          "notes": "text",
          "raw_text": "raw LLM output"
        }
        """
        ...

    def propose_skill_implementation(
        self,
        skill_spec: Dict[str, Any],
        examples: List[Dict[str, Any]],
    ) -> str:
        """
        Produce Python code implementing the given skill.

        Implementations are free to return additional metadata elsewhere,
        but this interface returns just the code string.
        """
        ...


class ErrorModel(Protocol):
    """
    Error analysis / recovery interface.

    To avoid circular imports, we type the argument and return value as Any.
    Concrete dataclasses live in llm_stack.schema:
      - ErrorContext
      - ErrorAnalysis
    """

    def analyze_failure(self, ctx: Any) -> Any:
        """
        Analyze a failure context and return structured analysis describing:

        - classification (short label)
        - summary (human-readable)
        - suggested_fix (machine-usable hints)
        - retry_advised (bool)
        """
        ...


class ScribeModel(Protocol):
    """
    Log/trace summarization interface.

    Concrete request/response types live in llm_stack.schema:
      - TraceSummaryRequest
      - TraceSummaryResponse
    """

    def summarize_trace(self, req: Any) -> Any:
        """
        Summarize a trace/log into a compressed representation suitable for:

        - human documentation
        - long-term context storage
        - debugging

        Implementations typically return:
          TraceSummaryResponse(summary, keywords, suggested_tags, raw_text)
        """
        ...

