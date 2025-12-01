# src/llm_stack/config.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class ModelConfig:
    """Minimal model config used by LLMBackend / LlamaCppBackend."""

    model_path: str

    # generation parameters
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 40
    repeat_penalty: float = 1.1

    # context / performance knobs
    n_ctx: int = 4096
    n_gpu_layers: int = 48
    n_threads: Optional[int] = None
    n_batch: Optional[int] = None

    # prompt shaping
    stop: Optional[List[str]] = None
    system_prompt: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelConfig":
        """Convenience constructor from a plain dict (e.g. YAML)."""
        return cls(
            model_path=data["model_path"],
            max_tokens=data.get("max_tokens", 512),
            temperature=data.get("temperature", 0.7),
            top_p=data.get("top_p", 0.95),
            top_k=data.get("top_k", 40),
            repeat_penalty=data.get("repeat_penalty", 1.1),
            n_ctx=data.get("n_ctx", 4096),
            n_gpu_layers=data.get("n_gpu_layers", 48),
            n_threads=data.get("n_threads"),
            stop=data.get("stop"),
            system_prompt=data.get("system_prompt"),
        )

