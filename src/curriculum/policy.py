# src/curriculum/policy.py

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum


class SkillUsageMode(str, Enum):
    """How aggressively the agent can use skills."""
    STABLE_ONLY = "stable_only"
    ALLOW_CANDIDATES = "allow_candidates"


@dataclass
class SkillPolicy:
    """
    Curriculum-level skill usage policy.

    Controls whether candidate/experimental skills are visible
    to planners and dispatchers.
    """
    usage_mode: SkillUsageMode = SkillUsageMode.STABLE_ONLY

    @property
    def include_candidates(self) -> bool:
        """Whether this policy allows experimental skills."""
        return self.usage_mode == SkillUsageMode.ALLOW_CANDIDATES

