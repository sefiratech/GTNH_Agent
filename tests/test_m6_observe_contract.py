# tests/test_m6_observe_contract.py
"""
Contract tests for M6's BotCore.observe() output.

Goal:
  - Ensure the *real* BotCore.observe() returns a RawWorldSnapshot whose shape
    is compatible with M7's expectations.

This test is written to be future-proof:
  - If bot_core or get_bot_core() don't exist yet, tests are skipped.
  - Once you add a concrete M6 implementation and a get_bot_core() helper,
    these tests will start enforcing the contract automatically.

Expected contract for RawWorldSnapshot:
  - snapshot.player_pos: dict with x, y, z
  - snapshot.entities: list of dicts or objects with x, y, z, type, data
  - snapshot.inventory: list of dicts with at least an id and count-like field
"""

from typing import Any, Dict

import pytest


def _get_rawworldsnapshot_cls():
    """
    Try to import RawWorldSnapshot from bot_core.snapshot.

    If bot_core.snapshot doesn't exist yet, skip the tests gracefully.
    """
    snap_mod = pytest.importorskip("bot_core.snapshot", reason="M6 bot_core not implemented yet")
    RawWorldSnapshot = getattr(snap_mod, "RawWorldSnapshot", None)
    if RawWorldSnapshot is None:
        pytest.skip("bot_core.snapshot.RawWorldSnapshot missing")
    return RawWorldSnapshot


def _get_real_botcore():
    """
    Try to get a real BotCore instance.

    Convention:
      - Provide a factory function in `bot_core.runtime`:

            def get_bot_core() -> Any:
                ...

      that returns the concrete BotCore implementation.
    """
    try:
        runtime_mod = __import__("bot_core.runtime", fromlist=["get_bot_core"])
    except ImportError:
        pytest.skip("bot_core.runtime.get_bot_core() not available yet")

    get_bot_core = getattr(runtime_mod, "get_bot_core", None)
    if get_bot_core is None:
        pytest.skip("get_bot_core() not defined in bot_core.runtime")

    return get_bot_core()


def _assert_player_pos(snapshot: Any) -> None:
    pos = getattr(snapshot, "player_pos", None)
    assert isinstance(pos, dict), "player_pos must be a dict"
    for k in ("x", "y", "z"):
        assert k in pos, f"player_pos missing key '{k}'"


def _assert_entities(snapshot: Any) -> None:
    entities = getattr(snapshot, "entities", None)
    assert isinstance(entities, (list, tuple)), "entities must be a list/tuple"

    if not entities:
        # Empty is allowed, no further checks needed.
        return

    sample = entities[0]

    # M7 handles two shapes:
    #   - dict with keys
    #   - object with attributes
    def get_field(obj: Any, key: str):
        if isinstance(obj, dict):
            return obj.get(key, None)
        return getattr(obj, key, None)

    for key in ("x", "y", "z", "type"):
        value = get_field(sample, key)
        assert value is not None, f"entity missing '{key}' field or attribute"

    data = get_field(sample, "data")
    assert isinstance(data, (dict, type(None))), "entity.data should be dict or None"


def _assert_inventory(snapshot: Any) -> None:
    inventory = getattr(snapshot, "inventory", None)
    assert isinstance(inventory, (list, tuple)), "inventory must be a list/tuple"

    if not inventory:
        # Empty inventory is allowed.
        return

    for stack in inventory:
        assert isinstance(stack, dict), "inventory entries must be dicts"

        # Some kind of id field must exist for M7's summarizers / normalizers.
        has_id = any(k in stack for k in ("item_id", "item", "id"))
        assert has_id, "inventory stack missing item identifier (item_id/item/id)"

        # Count must be convertible to int for M7.
        assert "count" in stack, "inventory stack missing 'count' field"


def test_botcore_observe_contract_shape():
    """
    Contract test for M6 BotCore.observe().

    Steps:
      1. Get RawWorldSnapshot class (via bot_core.snapshot).
      2. Get real BotCore instance via bot_core.runtime.get_bot_core().
      3. Call observe() once.
      4. Assert structural contract required by M7.
    """
    RawWorldSnapshot = _get_rawworldsnapshot_cls()
    bot_core = _get_real_botcore()

    snapshot = bot_core.observe()

    # Type check
    assert isinstance(snapshot, RawWorldSnapshot), "observe() must return RawWorldSnapshot"

    # Structural checks matching M7 expectations.
    _assert_player_pos(snapshot)
    _assert_entities(snapshot)
    _assert_inventory(snapshot)

