# src/spec/__init__.py

from __future__ import annotations

"""
Public spec surface for core GTNH Agent types.

This module re-exports *interfaces and data types* used across the codebase:
  - AgentLoop domain types (AgentGoal, Task, TaskPlan, VirtueEngine, PlanEvaluation, RetryPolicy)
  - LLM role protocols (PlannerModel, PlanCodeModel, CriticModel, ErrorModel, ScribeModel)
  - Observation / WorldState primitives

Deliberately does NOT export a concrete AgentLoop implementation to avoid
circular imports and keep runtime wiring in src/agent/.
"""

# Agent loop domain types
from .agent_loop import (
    AgentGoal,
    Task,
    TaskPlan,
    VirtueEngine,
    PlanEvaluation,
    RetryPolicy,
)

# LLM role protocols (Q1.6 roles)
from .llm import (
    PlannerModel,
    PlanCodeModel,
    CriticModel,
    ErrorModel,
    ScribeModel,
)

# Core world / observation types
from .types import (
    Observation,
    WorldState,
)

__all__ = [
    # AgentLoop domain
    "AgentGoal",
    "Task",
    "TaskPlan",
    "VirtueEngine",
    "PlanEvaluation",
    "RetryPolicy",
    # LLM roles
    "PlannerModel",
    "PlanCodeModel",
    "CriticModel",
    "ErrorModel",
    "ScribeModel",
    # World / observation
    "Observation",
    "WorldState",
]

