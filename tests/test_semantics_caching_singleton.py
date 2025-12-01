# tests/test_semantics_caching_singleton.py
"""
Tests for semantics.cache singleton behavior.

Goal:
  - Ensure SemanticsDB and TechGraph are constructed at most once per process.
  - Ensure repeated calls return the exact same instance.
"""

from semantics import cache as semantics_cache


def test_get_semantics_db_is_singleton(monkeypatch):
    calls = {"count": 0}

    class DummyDB:
        def __init__(self) -> None:
            calls["count"] += 1

    # Reset caches to a known clean state
    semantics_cache._reset_caches_for_tests()

    # Patch SemanticsDB constructor used inside cache
    monkeypatch.setattr(semantics_cache, "SemanticsDB", DummyDB, raising=True)

    # First call should construct, second should reuse
    db1 = semantics_cache.get_semantics_db()
    db2 = semantics_cache.get_semantics_db()

    assert db1 is db2, "get_semantics_db must return the same instance"
    assert calls["count"] == 1, "SemanticsDB constructor should only be called once"


def test_get_tech_graph_is_singleton(monkeypatch):
    calls = {"count": 0}

    class DummyGraph:
        def __init__(self) -> None:
            calls["count"] += 1

    # Reset caches again
    semantics_cache._reset_caches_for_tests()

    # Patch TechGraph constructor used inside cache
    monkeypatch.setattr(semantics_cache, "TechGraph", DummyGraph, raising=True)

    # First call should construct, second should reuse
    g1 = semantics_cache.get_tech_graph()
    g2 = semantics_cache.get_tech_graph()

    assert g1 is g2, "get_tech_graph must return the same instance"
    assert calls["count"] == 1, "TechGraph constructor should only be called once"

