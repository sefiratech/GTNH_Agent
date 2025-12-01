# bot_core package
# src/bot_core/__init__.py
"""
bot_core_1_7_10 package.

Exports:
    - BotCoreImpl: main body/controller implementation for MC 1.7.10 GTNH
    - BotCoreError: domain-level error type for non-action failures
"""

from __future__ import annotations

from .core import BotCoreImpl, BotCoreError

__all__ = [
    "BotCoreImpl",
    "BotCoreError",
]

