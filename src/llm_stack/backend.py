# abstract backend interface for local LLM engines
# src/llm_stack/backend.py

from typing import Protocol, List, Optional


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

