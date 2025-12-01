# path: src/monitoring/llm_logging.py

"""
LLM-specific logging utilities for GTNH_Agent (M2 + M9 bridge).

Structured monitoring events still use:
    monitoring.logger.log_event -> JsonFileLogger -> logs/monitoring/events.log

This module is specifically for:
    - Per-call LLM logs (planner / critic / scribe / error_model / etc.)
    - Written as individual JSON files under logs/llm/
    - Consumed by monitoring.tools.iter_llm_logs and human tools (jq, etc.)

Schema (one JSON per file):

{
  "ts": <float unix timestamp>,
  "role": "planner" | "critic" | "scribe" | "error_model" | ...,
  "model": "<model identifier>",
  "episode_id": "<episode-id or null>",
  "context_id": "<context-id or null>",
  "prompt": "<raw text prompt or short summary>",
  "response": "<raw text response or short summary>",
  "tokens_prompt": <int or null>,
  "tokens_completion": <int or null>,
  "meta": {
    ... arbitrary extra fields, e.g.:
    "temperature": 0.2,
    "top_p": 0.95,
    "latency_ms": 123.4,
    "error": null
  }
}

Notes:
- This is intentionally simple and file-based to stay fully offline.
- Filenames are unique based on timestamp + a random suffix.
"""

from __future__ import annotations  # allow forward references in type hints

import json                         # for writing JSON payloads
import time                         # for timestamps
import uuid                         # for unique filenames
from dataclasses import dataclass, asdict  # for LLMCallLog
from pathlib import Path            # for filesystem paths
from typing import Any, Dict, Optional     # for type hints


JsonDict = Dict[str, Any]


# ------------------------------------------------------------------------------
# Data structure representing a single LLM call
# ------------------------------------------------------------------------------

@dataclass
class LLMCallLog:
    """
    Structured record for one LLM interaction.

    This is the canonical schema for logs under logs/llm/*.json.
    """
    ts: float                        # unix timestamp when call completed
    role: str                        # "planner", "critic", "scribe", "error_model", etc.
    model: str                       # model identifier (e.g. "qwen2.5-coder-7b-instruct")
    episode_id: Optional[str]        # logical episode id, if available
    context_id: Optional[str]        # context / env id, if available
    prompt: str                      # text passed into the LLM (raw or summarized)
    response: str                    # text returned by the LLM (raw or summarized)
    tokens_prompt: Optional[int]     # number of prompt tokens, if known
    tokens_completion: Optional[int] # number of completion tokens, if known
    meta: JsonDict                   # free-form metadata (latency, error flags, etc.)

    def to_dict(self) -> JsonDict:
        """
        Convert the dataclass to a JSON-serializable dict.
        """
        return asdict(self)


# ------------------------------------------------------------------------------
# LLM log writer
# ------------------------------------------------------------------------------

class LLMLogWriter:
    """
    File-based LLM log writer.

    Responsibilities:
    - Ensure logs/llm directory exists.
    - Write one JSON file per LLM call using the LLMCallLog schema.
    - Keep the interface dead simple so any LLM client can call it.

    Typical usage:

        writer = LLMLogWriter()
        log = LLMCallLog(
            ts=time.time(),
            role="planner",
            model="qwen2.5-coder-7b-instruct",
            episode_id=episode_id,
            context_id=context_id,
            prompt=prompt_text,
            response=response_text,
            tokens_prompt=prompt_tokens,
            tokens_completion=completion_tokens,
            meta={"temperature": 0.2, "latency_ms": 150.0},
        )
        writer.write(log)
    """

    def __init__(self, log_dir: Path | None = None) -> None:
        # Default directory for LLM logs if not specified
        if log_dir is None:
            log_dir = Path("logs") / "llm"
        self._log_dir = log_dir
        # Ensure the directory exists
        self._log_dir.mkdir(parents=True, exist_ok=True)

    @property
    def log_dir(self) -> Path:
        """
        Return the directory where LLM logs are written.
        """
        return self._log_dir

    def write(self, call_log: LLMCallLog) -> Path:
        """
        Persist a single LLM call log as a JSON file.

        Returns:
            The full path to the written JSON file.
        """
        # Use timestamp + random suffix to avoid collisions
        ts_part = f"{call_log.ts:.6f}"
        role_part = call_log.role or "unknown"
        rand_part = uuid.uuid4().hex[:8]
        filename = f"{ts_part}_{role_part}_{rand_part}.json"
        path = self._log_dir / filename

        # Serialize dataclass to a dict then to JSON
        data = call_log.to_dict()

        # Write as pretty-printed JSON (small files, human-readable)
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, sort_keys=True)

        return path


# ------------------------------------------------------------------------------
# Convenience function for quick logging
# ------------------------------------------------------------------------------

def log_llm_call(
    role: str,
    model: str,
    prompt: str,
    response: str,
    episode_id: Optional[str] = None,
    context_id: Optional[str] = None,
    tokens_prompt: Optional[int] = None,
    tokens_completion: Optional[int] = None,
    meta: Optional[JsonDict] = None,
    log_dir: Path | None = None,
) -> Path:
    """
    One-shot helper to log an LLM call without manually constructing LLMCallLog.

    Arguments:
        role:
            Logical role of this LLM call, e.g. "planner", "critic", "scribe".
        model:
            Model identifier string.
        prompt:
            Prompt text (raw or summarized).
        response:
            Response text (raw or summarized).
        episode_id:
            Episode correlation id, if available.
        context_id:
            Context/environment id, if available.
        tokens_prompt:
            Number of prompt tokens, if known.
        tokens_completion:
            Number of completion tokens, if known.
        meta:
            Additional metadata such as latency, temperature, errors, etc.
        log_dir:
            Optional override for the logs directory (defaults to logs/llm).

    Returns:
        Path to the written JSON log file.
    """
    # Create an LLMCallLog instance with current timestamp and provided details
    call_log = LLMCallLog(
        ts=time.time(),
        role=role,
        model=model,
        episode_id=episode_id,
        context_id=context_id,
        prompt=prompt,
        response=response,
        tokens_prompt=tokens_prompt,
        tokens_completion=tokens_completion,
        meta=meta or {},
    )

    # Create a writer for the given directory and persist the log
    writer = LLMLogWriter(log_dir=log_dir)
    return writer.write(call_log)

