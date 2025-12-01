# src/llm_stack/backend_llamacpp.py

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import os
from llama_cpp import Llama

from .backend import LLMBackend
from .config import ModelConfig


class LlamaCppBackend(LLMBackend):
    """LLMBackend implementation using llama.cpp local inference.

    Supports two construction styles:

    1) New (preferred):
        backend = LlamaCppBackend(config=model_config)

       where `model_config` is a ModelConfig from models.yaml, including:
         - path
         - context_length
         - gpu_layers (optional)
         - n_threads (optional)
         - n_batch (optional)

    2) Legacy (backward compatible):
        backend = LlamaCppBackend(
            model_path="path/to/model.gguf",
            context_length=4096,
            n_gpu_layers=48,
        )
    """

    def __init__(
        self,
        config: Optional[ModelConfig] = None,
        *,
        # legacy kwargs for backward compatibility with existing code
        model_path: Optional[str] = None,
        context_length: Optional[int] = None,
        n_gpu_layers: Optional[int] = None,
        n_threads: Optional[int] = None,
        n_batch: Optional[int] = None,
    ) -> None:
        # Decide which input style we're using
        if config is None:
            if model_path is None:
                raise ValueError(
                    "LlamaCppBackend requires either `config` or `model_path`."
                )

            # Legacy path: build effective values from kwargs
            path = Path(model_path)
            ctx_len = context_length or 4096

            # If caller doesn't specify n_gpu_layers, default to "all layers"
            # and let llama.cpp offload as many as VRAM allows.
            if n_gpu_layers is None:
                gpu_layers = 9999
            else:
                gpu_layers = n_gpu_layers

            cfg_n_threads = n_threads
            cfg_n_batch = n_batch
        else:
            # New path: pull everything from ModelConfig
            path = Path(config.path)
            ctx_len = config.context_length

            # Same default: if gpu_layers not set in config, offload all layers
            if getattr(config, "gpu_layers", None) is None:
                gpu_layers = 9999
            else:
                gpu_layers = config.gpu_layers  # type: ignore[assignment]

            cfg_n_threads = getattr(config, "n_threads", None)
            cfg_n_batch = getattr(config, "n_batch", None)

        if not path.exists():
            raise FileNotFoundError(path)

        # Sensible default: use all but 1 CPU core if not specified
        default_threads = max(1, (os.cpu_count() or 1) - 1)

        effective_n_threads = cfg_n_threads or default_threads
        effective_n_batch = cfg_n_batch or 512

        # Let llama.cpp use the model's native chat template (Qwen-style),
        # since we're going to call create_chat_completion below.
        self._llm = Llama(
            model_path=str(path),
            n_ctx=ctx_len,
            n_gpu_layers=gpu_layers,
            n_threads=effective_n_threads,
            n_batch=effective_n_batch,
        )

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
        Generate text using chat-completion style calls.

        We map:
        - system_prompt -> system message (if provided)
        - prompt        -> user message
        and return the assistant's message content as a plain string.
        """
        messages = []

        if system_prompt:
            messages.append(
                {
                    "role": "system",
                    "content": system_prompt,
                }
            )

        messages.append(
            {
                "role": "user",
                "content": prompt,
            }
        )

        out = self._llm.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop or [],
        )

        # Standard OpenAI-style chat completion shape:
        # choices[0]["message"]["content"] is the assistant text.
        text = out["choices"][0]["message"]["content"]
        return text.strip()

