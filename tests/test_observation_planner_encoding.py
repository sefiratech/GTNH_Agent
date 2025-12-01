# tests/test_observation_planner_encoding.py
"""
Planner tests for M7 - observation_encoding.

Covers:
  - minimal RawWorldSnapshot
  - DummySemanticsDB
  - required fields in planner encoding
  - non-empty text_summary
"""

from observation.encoder import encode_for_planner, make_planner_observation
from observation.testing import (
    DummySemanticsDB,
    make_minimal_snapshot,
    make_minimal_tech_state,
)
from spec.types import Observation


def test_encode_for_planner_basic_structure():
    raw = make_minimal_snapshot()
    db = DummySemanticsDB()
    tech = make_minimal_tech_state()

    enc = encode_for_planner(
        raw_snapshot=raw,
        tech_state=tech,
        db=db,
        context_id="lv_early_factory",
    )

    # Field existence
    assert "tech_state" in enc
    assert "agent" in enc
    assert "inventory_summary" in enc
    assert "machines_summary" in enc
    assert "nearby_entities" in enc
    assert "env_summary" in enc
    assert "craftable_summary" in enc
    assert "context_id" in enc
    assert "text_summary" in enc

    # Basic correctness
    assert enc["tech_state"]["active"] == "stone_age"
    assert enc["agent"]["dimension"] == "overworld"
    assert isinstance(enc["nearby_entities"], list)
    assert isinstance(enc["inventory_summary"], dict)
    assert isinstance(enc["text_summary"], str)
    assert len(enc["text_summary"]) > 0


def test_make_planner_observation_returns_observation():
    raw = make_minimal_snapshot()
    db = DummySemanticsDB()
    tech = make_minimal_tech_state()

    obs = make_planner_observation(
        raw_snapshot=raw,
        tech_state=tech,
        db=db,
        context_id="lv_early_factory",
    )

    assert isinstance(obs, Observation)
    assert isinstance(obs.json_payload, dict)
    assert "text_summary" in obs.json_payload
    assert obs.text_summary == obs.json_payload["text_summary"]

