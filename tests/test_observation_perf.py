# tests/test_observation_perf.py
"""
Performance / scale sanity tests for M7 - observation_encoding.

Goal is not microbenchmarking, but:
  - ensure encode_for_planner handles larger snapshots without exploding
  - confirm caps on entities/craftables keep output size controlled
"""

from observation.encoder import encode_for_planner
from observation.testing import (
    DummySemanticsDB,
    make_heavy_snapshot,
    make_minimal_tech_state,
)


def test_encode_for_planner_heavy_snapshot_caps_and_structure():
    """
    Use a heavier snapshot with many entities + inventory stacks and check:

      - function completes without error
      - entity list is capped (<= 16)
      - craftables list is capped (<= 10)
      - encoding is still reasonably sized
    """
    raw = make_heavy_snapshot(entity_count=150, inventory_count=150)
    db = DummySemanticsDB()
    tech = make_minimal_tech_state()

    enc = encode_for_planner(
        raw_snapshot=raw,
        tech_state=tech,
        db=db,
        context_id="lv_early_factory",
    )

    # Entity list cap
    assert "nearby_entities" in enc
    assert len(enc["nearby_entities"]) <= 16

    # Craftables cap (top_outputs from semantics.crafting)
    craft = enc.get("craftable_summary", {})
    top_outputs = craft.get("top_outputs", [])
    assert len(top_outputs) <= 10

    # Encoding size sanity: avoid mega-blobs
    # Rough upper bound on top-level keys and list lengths.
    assert isinstance(enc["inventory_summary"], dict)
    assert len(enc["inventory_summary"].get("by_item", {})) <= 200

    # Text summary should still exist and be short-ish.
    summary = enc.get("text_summary", "")
    assert isinstance(summary, str)
    assert len(summary) > 0
    assert len(summary) < 1024  # arbitrary "not insane" limit

