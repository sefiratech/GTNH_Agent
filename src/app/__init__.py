# src/app/__init__.py
"""
Application entrypoints for the GTNH Agent.

Phase 0 exposes:
- create_phase0_agent: wiring M0 EnvProfile + M1 spec into a stub runtime
- main: simple CLI entry that runs a few agent steps for sanity checks
"""

from __future__ import annotations

from .runtime import create_phase0_agent, main

__all__ = [
    "create_phase0_agent",
    "main",
]

