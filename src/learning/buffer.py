# path: src/learning/buffer.py

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, Optional

from .schema import Experience, ExperienceEpisode, EpisodeMetadata


@dataclass
class ExperienceBuffer:
    """
    JSONL-backed experience store.

    Dual-use design:

    - Q1.4+ (M10 path):
        Stores `learning.schema.Experience` objects, using Experience.to_dict /
        Experience.from_dict for round-tripping.

    - Q1 "control + experience" tests:
        Stores older `ExperienceEpisode`-style objects that:
          * expose `.to_dict(tech_state_to_dict=..., plan_trace_to_dict=...)`
          * expect tech_state/trace to be reconstructed via hook functions.

      For those, we:
        - call their to_dict(...) when appending
        - reconstruct a lightweight episode-like wrapper when reading,
          using the provided hook functions.

    The rules:

    - append_experience(x):
        * if x is an ExperienceEpisode → call to_dict with tech/trace hooks
        * else if it has .to_dict → try Q1 hook signature, then fall back
        * else if it's an Experience → use Experience.to_dict()
        * otherwise raise TypeError

    - load_all():
        * first tries Experience.from_dict(raw)
        * if that explodes, falls back to a "legacy episode" wrapper that
          behaves like ExperienceEpisode for the Q1 tests.
    """

    path: Path
    tech_state_to_dict: Optional[Callable[[Any], Dict[str, Any]]] = None
    tech_state_from_dict: Optional[Callable[[Dict[str, Any]], Any]] = None
    trace_to_dict: Optional[Callable[[Any], Dict[str, Any]]] = None
    trace_from_dict: Optional[Callable[[Dict[str, Any]], Any]] = None

    def __post_init__(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Internal aliases so we don't accidentally shadow attributes later
        self._tech_state_to_dict = self.tech_state_to_dict
        self._tech_state_from_dict = self.tech_state_from_dict
        self._trace_to_dict = self.trace_to_dict
        self._trace_from_dict = self.trace_from_dict

    # ------------------------------------------------------------------ #
    # Append / raw IO
    # ------------------------------------------------------------------ #

    def append_experience(self, exp: Any) -> None:
        """
        Append a single experience-like object as one JSONL line.

        Supported inputs:
        - learning.schema.Experience
        - learning.schema.ExperienceEpisode
        - Q1-style objects with .to_dict(...)
        """
        payload: Dict[str, Any]

        # 1) Explicit path for ExperienceEpisode so we can pass the *right* hook name.
        if isinstance(exp, ExperienceEpisode):
            payload = exp.to_dict(
                tech_state_to_dict=self._tech_state_to_dict,
                plan_trace_to_dict=self._trace_to_dict,
            )

        # 2) New-style Experience object
        elif isinstance(exp, Experience):
            payload = exp.to_dict()

        # 3) Generic objects with a to_dict method (DummyExperience, older tests, etc.)
        elif hasattr(exp, "to_dict"):
            to_dict = getattr(exp, "to_dict")

            # If hooks are available, try to use the classic Q1 signature first.
            if self._tech_state_to_dict is not None or self._trace_to_dict is not None:
                # 3a) Try keyword arguments (tech_state_to_dict / trace_to_dict)
                try:
                    payload = to_dict(
                        tech_state_to_dict=self._tech_state_to_dict,
                        trace_to_dict=self._trace_to_dict,
                    )
                except TypeError:
                    # 3b) Try positional arguments
                    try:
                        payload = to_dict(
                            self._tech_state_to_dict,
                            self._trace_to_dict,
                        )
                    except TypeError:
                        # 3c) Fall back to plain to_dict()
                        payload = to_dict()  # type: ignore[call-arg]
            else:
                # No hooks configured, just call to_dict() directly.
                payload = to_dict()

        else:
            raise TypeError(
                f"append_experience expected an ExperienceEpisode, Experience, "
                f"or an object with to_dict(); got {type(exp)!r}"
            )

        line = json.dumps(payload, ensure_ascii=False)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def append(self, exp: Any) -> None:
        """
        Backwards-compatible alias used by some M10/Q1 tests.

        Functionally identical to append_experience(...).
        """
        self.append_experience(exp)

    def load_all_raw(self) -> Iterator[Dict[str, Any]]:
        """
        Yield each stored experience as a raw dict (no typing).
        """
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(obj, dict):
                    continue
                yield obj

    def count(self) -> int:
        """
        Count non-empty lines in the JSONL file.
        """
        if not self.path.exists():
            return 0
        n = 0
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    n += 1
        return n

    # ------------------------------------------------------------------ #
    # Typed load API
    # ------------------------------------------------------------------ #

    def load_all(self) -> Iterator[Any]:
        """
        Yield typed experiences.

        Strategy:
          1. First try `Experience.from_dict(raw)` (new M10 format).
          2. If ANY exception is raised, fall back to a legacy
             ExperienceEpisode-style wrapper that:

              - exposes attributes:
                  id, goal, plan, pre_eval, post_eval, final_outcome,
                  tech_state, trace, virtue_scores, success,
                  failure_type, severity, metadata
              - uses the provided tech_state_from_dict / trace_from_dict
                hooks when reconstructing those fields.

        This keeps M10 code happy *and* lets the Q1 tests use their
        older ExperienceEpisode schema without changing the tests.
        """
        for raw in self.load_all_raw():
            # New-style Experience path
            try:
                yield Experience.from_dict(raw)
                continue
            except Exception:
                # Fall back to legacy episode wrapper
                yield self._legacy_episode_from_dict(raw)

    # ------------------------------------------------------------------ #
    # Legacy Q1 episode compatibility
    # ------------------------------------------------------------------ #

    def _legacy_episode_from_dict(self, raw: Dict[str, Any]) -> Any:
        """
        Build a lightweight object that behaves like the old
        ExperienceEpisode used in Q1 tests.

        It is deliberately dumb: just attaches attributes from the dict,
        plus tech_state/trace reconstruction via hooks if available.
        """

        tech_state_from = self._tech_state_from_dict
        trace_from = self._trace_from_dict

        class LegacyEpisode:
            __slots__ = (
                "id",
                "goal",
                "plan",
                "pre_eval",
                "post_eval",
                "final_outcome",
                "tech_state",
                "trace",
                "virtue_scores",
                "success",
                "failure_type",
                "severity",
                "metadata",
            )

            def __init__(self, data: Dict[str, Any]) -> None:
                self.id = data.get("id")
                # Q1 tests often use `goal` as a plain string
                self.goal = data.get("goal")

                self.plan = data.get("plan") or {}

                self.pre_eval = data.get("pre_eval") or {}
                self.post_eval = data.get("post_eval") or {}
                self.final_outcome = data.get("final_outcome") or {}

                # Virtue info
                self.virtue_scores = data.get("virtue_scores") or {}

                # Success / failure annotations
                self.success = data.get("success")
                self.failure_type = data.get("failure_type")
                self.severity = data.get("severity")

                # Metadata: reconstruct EpisodeMetadata if present, else empty.
                meta_raw = data.get("metadata") or {}
                try:
                    if isinstance(meta_raw, dict):
                        self.metadata = EpisodeMetadata.from_dict(meta_raw)
                    elif isinstance(meta_raw, EpisodeMetadata):
                        self.metadata = meta_raw
                    else:
                        self.metadata = EpisodeMetadata()
                except Exception:
                    self.metadata = EpisodeMetadata()

                # Tech state roundtrip (best effort)
                ts_raw = data.get("tech_state")
                if ts_raw is not None and callable(tech_state_from):
                    try:
                        self.tech_state = tech_state_from(ts_raw)  # type: ignore[arg-type]
                    except Exception:
                        self.tech_state = ts_raw
                else:
                    self.tech_state = ts_raw

                # Trace roundtrip (best effort)
                tr_raw = data.get("trace")
                if tr_raw is not None and callable(trace_from):
                    try:
                        self.trace = trace_from(tr_raw)  # type: ignore[arg-type]
                    except Exception:
                        self.trace = tr_raw
                else:
                    self.trace = tr_raw

            def to_dict(self, **_: Any) -> Dict[str, Any]:
                """
                Provide a to_dict so this object can be re-serialized
                if anyone ever tries to append it again.
                We just return the original raw mapping.
                """
                # Note: we don't try to re-run hooks here; tests don't need it.
                return dict(raw)

        return LegacyEpisode(raw)

