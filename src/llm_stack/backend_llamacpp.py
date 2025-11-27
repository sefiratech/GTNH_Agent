# src/llm_stack/backend_llamacpp.py

from typing import List, Optional
from pathlib import Path

from llama_cpp import Llama

from .backend import LLMBackend


class LlamaCppBackend(LLMBackend):
    """LLMBackend implementation using llama.cpp local inference."""

    def __init__(
        self,
        model_path: str,
        context_length: int = 4096,
        n_gpu_layers: int = 0,
    ) -> None:
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(path)

        # Let llama.cpp use the model's native chat template (Qwen-style),
        # since we're going to call create_chat_completion below.
        self._llm = Llama(
            model_path=str(path),
            n_ctx=context_length,
            n_gpu_layers=n_gpu_layers,
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

