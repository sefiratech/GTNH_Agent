#tests/test_skill_evaluator.py
"""
Basic tests for M10 SkillEvaluator.

These use a minimal fake TechState/PlanTrace and fake virtues.metrics
so you can test the logic without firing real LLMs or semantics.
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, List

from learning.evaluator import SkillEvaluator
from learning.schema import ExperienceEpisode, EpisodeMetadata, SkillPerformanceStats
from semantics.schema import TechState  # or define a tiny fake if needed
from observation.trace_schema import PlanTrace  # or fake
import virtues.metrics  # type: ignore
import virtues.lattice  # type: ignore
import virtues.loader  # type: ignore


# You can monkeypatch virtues.* here if you want a pure unit test.

def test_compare_stats_promote_candidate(monkeypatch):
    evaluator = SkillEvaluator()

    baseline = SkillPerformanceStats(
        skill_name="test_skill",
        uses=20,
        success_rate=0.7,
        avg_time=10.0,
        avg_resource_cost=5.0,
        avg_virtue_scores={"safety": 0.5},
    )
    candidate = SkillPerformanceStats(
        skill_name="test_skill_v2",
        uses=20,
        success_rate=0.72,
        avg_time=7.0,
        avg_resource_cost=4.0,
        avg_virtue_scores={"safety": 0.7},
    )

    result = evaluator.compare_stats(baseline, candidate)
    assert result["recommendation"] == "promote_candidate"
    assert any(r.startswith("candidate_faster") for r in result["reasons"]) or \
           any(r.startswith("candidate_cheaper") for r in result["reasons"]) or \
           any(r.startswith("candidate_better_") for r in result["reasons"])

