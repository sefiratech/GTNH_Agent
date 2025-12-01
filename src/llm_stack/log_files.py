# src/llm_stack/log_files.py

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = PROJECT_ROOT / "logs" / "llm"


def _ensure_log_dir() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def log_llm_call(
    *,
    role: str,
    operation: str,
    prompt: str,
    raw_response: str,
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    """Persist a single LLM interaction to logs/llm/ as JSON.

    Returns the path of the written file.
    """
    _ensure_log_dir()
    ts = time.strftime("%Y%m%dT%H%M%S")
    pid = os.getpid()
    filename = f"{ts}_{pid}_{role}_{operation}.json"
    path = LOG_DIR / filename

    payload = {
        "timestamp": ts,
        "pid": pid,
        "role": role,
        "operation": operation,
        "prompt": prompt,
        "raw_response": raw_response,
        "extra": extra or {},
    }

    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return path
