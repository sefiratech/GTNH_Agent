# src/agent/logging_config.py
"""
Central logging configuration for GTNH_Agent runtime.

Call configure_logging() from your main entrypoint once, for example:

    from agent.logging_config import configure_logging
    configure_logging()

After that, M6/M7 logs (including diagnostics in observation.encoder and
agent.runtime_m6_m7) will be visible on stdout.
"""

from __future__ import annotations

import logging
import sys


def configure_logging(level: int = logging.INFO) -> None:
    """
    Configure root logging if no handlers are attached yet.

    Args:
        level: default logging level (e.g., logging.INFO, logging.DEBUG)
    """
    root = logging.getLogger()

    # Don't duplicate handlers if someone already configured logging.
    if root.handlers:
        return

    handler = logging.StreamHandler(stream=sys.stdout)
    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    handler.setFormatter(formatter)
    root.addHandler(handler)
    root.setLevel(level)
