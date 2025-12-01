# path: tests/test_virtue_overrides.py

from __future__ import annotations

import pytest

from virtues.integration_overrides import merge_virtue_weights


def test_virtue_overrides_identity() -> None:
    """
    override = 1.0 → no change.
    """
    base = {"Safety": 1.0, "Efficiency": 0.8}
    overrides = {"Safety": 1.0, "Efficiency": 1.0}
    merged = merge_virtue_weights(base, overrides)

    assert merged["Safety"] == pytest.approx(base["Safety"])
    assert merged["Efficiency"] == pytest.approx(base["Efficiency"])


def test_virtue_overrides_increase_and_decrease() -> None:
    """
    override > 1.0 increases weight, < 1.0 decreases weight.
    """
    base = {"Safety": 1.0, "Throughput": 1.0}
    overrides = {"Safety": 1.2, "Throughput": 0.5}
    merged = merge_virtue_weights(base, overrides)

    assert merged["Safety"] == pytest.approx(1.2)
    assert merged["Throughput"] == pytest.approx(0.5)


def test_virtue_overrides_new_virtue() -> None:
    """
    Virtue present only in overrides appears in merged map.
    """
    base = {"Safety": 1.0}
    overrides = {"Exploration": 0.3}
    merged = merge_virtue_weights(base, overrides)

    assert "Safety" in merged
    assert "Exploration" in merged
    assert merged["Exploration"] == pytest.approx(0.3)

