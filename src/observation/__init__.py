# src/observation/__init__.py
"""
M7 - observation_encoding

Encodes normalized world state and execution traces into compact JSON payloads
for planner and critic models in M2.
"""

from .schema import PlannerEncoding, CriticEncoding
from .trace_schema import TraceStep, PlanTrace
from .encoder import (
    build_world_state,
    encode_for_planner,
    make_planner_observation,
    encode_for_critic,
)

__all__ = [
    "PlannerEncoding",
    "CriticEncoding",
    "TraceStep",
    "PlanTrace",
    "build_world_state",
    "encode_for_planner",
    "make_planner_observation",
    "encode_for_critic",
]

