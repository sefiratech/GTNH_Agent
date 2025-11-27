# src/llm_stack/stack.py

from env.loader import load_environment

from spec.llm import PlanCodeModel, ErrorModel, ScribeModel

from .backend_llamacpp import LlamaCppBackend
from .presets import RolePreset
from .plan_code import PlanCodeModelImpl
from .error_model import ErrorModelImpl
from .scribe import ScribeModelImpl


class LLMStack:
    """Convenience holder for all role-specific facades over a single backend."""

    def __init__(self) -> None:
        # Load environment profile (M0)
        env = load_environment()
        mp = env.model_profile  # ModelProfile(name, backend, models)

        if mp.backend != "llama_cpp":
            raise NotImplementedError(f"Unsupported backend: {mp.backend}")

        # ---- Choose primary model config ----
        # Prefer a "primary" entry if present; otherwise just take the first.
        if "primary" in mp.models:
            primary_cfg = mp.models["primary"]
        else:
            # fallback: first entry in the models dict
            primary_cfg = next(iter(mp.models.values()))

        # Shared backend for all roles
        self._backend = LlamaCppBackend(
            model_path=primary_cfg.path,
            context_length=primary_cfg.context_length,
        )

        # ---- Role presets (hardcoded for now; can be moved to YAML later) ----

        self._plan_code_preset = RolePreset(
            name="plan_code",
            temperature=0.2,
            max_tokens=768,
            system_prompt=(
                "You are the planning and coding brain for a Minecraft GTNH "
                "automation agent. You strictly follow the requested JSON or "
                "code-only output format."
            ),
            stop=None,
        )

        self._error_preset = RolePreset(
            name="error",
            temperature=0.0,
            max_tokens=512,
            system_prompt=(
                "You are a precise error analyst for an LLM stack. You only "
                "respond with valid JSON describing the error and suggested fix."
            ),
            stop=None,
        )

        self._scribe_preset = RolePreset(
            name="scribe",
            temperature=0.3,
            max_tokens=1024,
            system_prompt=(
                "You are a concise technical scribe for a Minecraft GTNH agent. "
                "You summarize traces and produce JSON with summary, keywords, and tags."
            ),
            stop=None,
        )

        # ---- Instantiate facades ----
        self._plan_code = PlanCodeModelImpl(self._backend, self._plan_code_preset)
        self._error = ErrorModelImpl(self._backend, self._error_preset)
        self._scribe = ScribeModelImpl(self._backend, self._scribe_preset)

    # ---- Public accessors ----

    @property
    def plan_code(self) -> PlanCodeModel:
        return self._plan_code

    @property
    def error(self) -> ErrorModel:
        return self._error

    @property
    def scribe(self) -> ScribeModel:
        return self._scribe

