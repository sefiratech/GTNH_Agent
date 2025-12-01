# src/llm_stack/json_utils.py

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


def load_json_or_none(
    raw: str,
    *,
    context: str = "unknown",
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Best-effort JSON loader.

    Returns (data, error_message). If parsing fails, data is None and
    error_message describes the failure.
    """
    try:
        data = json.loads(raw)
        return data, None
    except json.JSONDecodeError as e:
        msg = f"{context}: JSONDecodeError at pos {e.pos}: {e.msg}"
        logger.debug("load_json_or_none failed: %s; raw=%r", msg, raw)
        return None, msg

