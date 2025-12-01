# src/agent/experience.py
"""
Phase 4 preparation: experience structures.

This module defines small, generic containers that can be used by
the agent loop (M8) and any future experience / memory system:

  - Experience: one episode's plan/trace/critic bundle, plus context
  - ExperienceBuffer: an in-memory collection with a simple API

Nothing here assumes a specific storage backend. You can later:
  - bolt on filesystem / DB logging
  - add sampling logic for training
  - add indexing for retrieval
without changing the M8 loop signature.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import time

from observation.trace_schema import PlanTrace
from spec.experience import CriticResult


@dataclass
class Experience:
    """
    Episode-level experience bundle suitable for M10 learning.

    Fields:
        episode_id:
            Optional id for grouping experiences by logical episode.

        trace:
            PlanTrace containing:
              - plan emitted by the planner
              - execution steps (may be empty at first)
              - tech_state, context_id, planner_payload, virtue_scores

        critic_result:
            Output from the critic model, if enabled. Typed via CriticResult
            so different critic backends can still share a common shape.

        env_profile_name:
            Name/identifier of the active environment profile
            (e.g. from env.yaml via AgentRuntime).

        context_id:
            High-level virtue / scenario context id associated with the
            episode (e.g. "lv_early_factory").

        tech_state_snapshot:
            Serialized snapshot of tech_state at the end of the episode,
            suitable for logging and learning.

        meta:
            Free-form metadata (profile name, seed, environment tags, duration,
            curriculum goal, etc.).

        timestamp:
            Seconds since epoch when the experience was recorded.
    """
    trace: PlanTrace
    critic_result: Optional[CriticResult]
    episode_id: Optional[int] = None
    env_profile_name: str = ""
    context_id: str = ""
    tech_state_snapshot: Dict[str, Any] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=lambda: time.time())


@dataclass
class ExperienceBuffer:
    """
    Simple append-only buffer for episodes.

    This is intentionally simple for Phase 3/4 prep:
      - append-only list
      - trivial retrieval API

    Later you can:
      - add persistence hooks (write to disk, DB, etc.)
      - add sampling strategies (for training)
      - add filters / indexes
    """
    _items: List[Experience] = field(default_factory=list)

    def add(self, experience: Experience) -> None:
        """Append a new experience to the buffer."""
        self._items.append(experience)

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self._items)

    def last(self) -> Optional[Experience]:
        """Return the most recent experience, or None if buffer is empty."""
        return self._items[-1] if self._items else None

    @property
    def experiences(self) -> List[Experience]:
        """
        Backwards-compatible view of the internal list.

        Older tests or code may still access .experiences directly.
        """
        return self._items

