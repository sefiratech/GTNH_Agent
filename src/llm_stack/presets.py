# src/llm_stack/presets.py

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class RolePreset:
    """Configuration for a logical LLM role."""
    name: str
    temperature: float
    max_tokens: int
    system_prompt: Optional[str] = None
    stop: Optional[List[str]] = None

