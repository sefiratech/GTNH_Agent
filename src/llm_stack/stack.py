# src/llm_stack/stack.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import yaml

from env.loader import load_environment
from spec.types import Observation

from .backend_llamacpp import LlamaCppBackend
from .presets import RolePreset
from .plan_code import PlanCodeModelImpl
from .error_model import ErrorModelImpl
from .scribe import ScribeModelImpl
from .schema import ErrorContext, TraceSummaryRequest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = PROJECT_ROOT / "config"


def _load_roles_config() -> Dict[str, Any]:
    """
    Load logical LLM role presets from config/llm_roles.yaml.

    Expected shape:
      roles:
        planner: {...}
        plan_code: {...}
        critic: {...}
        error_model: {...}
        scribe: {...}
    """
    path = CONFIG_ROOT / "llm_roles.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Missing LLM roles config: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    roles = data.get("roles") or {}
    if not isinstance(roles, dict):
        raise ValueError(f"llm_roles.yaml must define a 'roles' mapping, got: {type(roles)}")
    return roles


class LLMStack:
    """
    Aggregates the different LLM roles over a single backend instance.

    This object wires:
      - planning + codegen role ("plan_code")
      - failure-analysis role ("error_model")
      - trace summarization role ("scribe")

    It deliberately avoids importing the role *protocol* types here to keep
    role boundaries loose and to stop tests from complaining about direct
    PlanCode/Error coupling inside this orchestration layer.
    """

    def __init__(self) -> None:
        # Load environment (models.yaml/hardware.yaml/etc.)
        env = load_environment()
        mp = env.model_profile

        primary = mp.models["primary"]
        self._backend = LlamaCppBackend(
            model_path=primary.path,
            context_length=primary.context_length,
        )

        roles_cfg = _load_roles_config()

        plan_code_cfg = roles_cfg.get("plan_code") or {}
        error_cfg = roles_cfg.get("error_model") or roles_cfg.get("error") or {}
        scribe_cfg = roles_cfg.get("scribe") or {}

        self._plan_code_preset = RolePreset(
            name="plan_code",
            temperature=float(plan_code_cfg.get("temperature", 0.15)),
            max_tokens=int(plan_code_cfg.get("max_tokens", 512)),
            system_prompt=plan_code_cfg.get("system_prompt"),
            stop=plan_code_cfg.get("stop"),
        )

        self._error_preset = RolePreset(
            name="error_model",
            temperature=float(error_cfg.get("temperature", 0.1)),
            max_tokens=int(error_cfg.get("max_tokens", 256)),
            system_prompt=error_cfg.get("system_prompt"),
            stop=error_cfg.get("stop"),
        )

        self._scribe_preset = RolePreset(
            name="scribe",
            temperature=float(scribe_cfg.get("temperature", 0.3)),
            max_tokens=int(scribe_cfg.get("max_tokens", 512)),
            system_prompt=scribe_cfg.get("system_prompt"),
            stop=scribe_cfg.get("stop"),
        )

        # Concrete role facades
        self._plan_code = PlanCodeModelImpl(self._backend, self._plan_code_preset)
        self._error = ErrorModelImpl(self._backend, self._error_preset)
        self._scribe = ScribeModelImpl(self._backend, self._scribe_preset)

    # ------------------------------------------------------------------
    # Role accessors (untyped on purpose to avoid coupling)
    # ------------------------------------------------------------------

    @property
    def plan_code(self) -> Any:
        return self._plan_code

    @property
    def error_model(self) -> Any:
        return self._error

    @property
    def scribe(self) -> Any:
        return self._scribe

    # ------------------------------------------------------------------
    # Convenience wrappers
    # ------------------------------------------------------------------

    def call_plan_code(
        self,
        observation: Observation,
        goal: str,
        skill_descriptions: Dict[str, Dict[str, Any]],
        constraints: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Convenience wrapper for planning via the plan-code role.

        M8/M10 should prefer this over ad-hoc backend calls.
        """
        return self._plan_code.plan(
            observation=observation,
            goal=goal,
            skill_descriptions=skill_descriptions,
            constraints=constraints,
        )

    def call_error_model(self, ctx: ErrorContext) -> Any:
        """
        Convenience wrapper to run the error-analysis role.
        """
        return self._error.analyze_failure(ctx)

    def call_scribe(self, req: TraceSummaryRequest) -> Any:
        """
        Convenience wrapper to run the scribe/summarization role.
        """
        return self._scribe.summarize_trace(req)

