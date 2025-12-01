# src/semantics/cache.py
"""
Lightweight caching for M3 runtime objects.

Responsibility:
  - Provide process-local singletons for:
      * SemanticsDB  (blocks/items/recipes)
      * TechGraph    (tech progression DAG)
  - Avoid repeated disk I/O and graph construction in hot paths.

Usage:

    from semantics.cache import get_semantics_db, get_tech_graph

    db = get_semantics_db()
    graph = get_tech_graph()

These accessors are intentionally simple so they are easy to reason about and
easy to patch in tests.
"""

from typing import Optional

from .loader import SemanticsDB
from .tech_state import TechGraph


_semantics_db: Optional[SemanticsDB] = None
_tech_graph: Optional[TechGraph] = None


def get_semantics_db() -> SemanticsDB:
    """
    Return a process-local SemanticsDB singleton.

    First call constructs the DB (reading YAML/JSON configs), subsequent calls
    return the same instance.

    Tests can monkeypatch semantics.cache.SemanticsDB to verify call counts.
    """
    global _semantics_db
    if _semantics_db is None:
        _semantics_db = SemanticsDB()
    return _semantics_db


def get_tech_graph() -> TechGraph:
    """
    Return a process-local TechGraph singleton.

    First call constructs the graph from gtnh_tech_graph.yaml, subsequent calls
    return the same instance.

    Tests can monkeypatch semantics.cache.TechGraph to verify call counts.
    """
    global _tech_graph
    if _tech_graph is None:
        _tech_graph = TechGraph()
    return _tech_graph


def _reset_caches_for_tests() -> None:
    """
    Internal helper used by tests to hard-reset the singletons.

    Do not use this in normal code; it's only meant for test isolation.
    """
    global _semantics_db, _tech_graph
    _semantics_db = None
    _tech_graph = None

