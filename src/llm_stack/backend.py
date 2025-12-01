# src/llm_stack/backend.py
"""
Backend interface and concrete implementation for local LLM engines.
Currently backed by llama_cpp for GGUF models.
"""

from __future__ import annotations

from typing import Protocol, List, Optional

from llama_cpp import Llama

# Adjust this import if ModelConfig lives somewhere else
from .config import ModelConfig


class LLMBackend(Protocol):
    """Simple interface around a local text generation backend."""

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int,
        temperature: float,
        stop: Optional[List[str]] = None,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Generate a text completion for the given prompt."""
        ...


class LlamaCppBackend:
    """
    Concrete LLM backend using llama_cpp and a local GGUF model.

    This is intentionally minimal: config in, text out.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        kwargs = {
            "model_path": cfg.model_path,
            "n_ctx": cfg.n_ctx,
        }

        # Optional CUDA-related config
        if getattr(cfg, "n_gpu_layers", None) is not None:
            kwargs["n_gpu_layers"] = cfg.n_gpu_layers
        if getattr(cfg, "main_gpu", None) is not None:
            kwargs["main_gpu"] = cfg.main_gpu
        if getattr(cfg, "tensor_split", None) is not None:
            kwargs["tensor_split"] = cfg.tensor_split

        self._llm = Llama(**kwargs)

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int,
        temperature: float,
        stop: Optional[List[str]] = None,
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Generate a completion for the given prompt using the underlying Llama model.
        """
        if system_prompt:
            full_prompt = f"{system_prompt.rstrip()}\n\n{prompt.lstrip()}"
        else:
            full_prompt = prompt

        result = self._llm(
            full_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop,
        )

        # llama_cpp returns an OpenAI-style response dict
        text = result["choices"][0]["text"]
        return text.strip()

