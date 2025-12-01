#tests/test_monitoring_logger.py
"""
Tests for monitoring.logger.JsonFileLogger and log_event.

Covers:
- JSON structure validity
- Correct field encoding
- Flush behavior (file actually gets data)
"""

from __future__ import annotations

import json
from pathlib import Path

from monitoring.bus import EventBus
from monitoring.logger import JsonFileLogger, log_event
from monitoring.events import EventType


def test_json_file_logger_writes_valid_json(tmp_path: Path):
    bus = EventBus()
    log_path = tmp_path / "events.log"

    logger = JsonFileLogger(log_path, bus)

    # Emit one event
    log_event(
        bus=bus,
        module="test.module",
        event_type=EventType.LOG,
        message="hello world",
        payload={"a": 1, "b": "x"},
        correlation_id="episode-123",
    )

    # Explicit close to ensure file handle is flushed
    logger.close()

    content = log_path.read_text(encoding="utf-8").strip()
    lines = content.splitlines()
    assert len(lines) == 1

    data = json.loads(lines[0])

    assert data["module"] == "test.module"
    assert data["event_type"] == "LOG"
    assert data["message"] == "hello world"
    assert data["payload"]["a"] == 1
    assert data["payload"]["b"] == "x"
    assert data["correlation_id"] == "episode-123"
    assert isinstance(data["ts"], (int, float))


def test_logger_parent_dir_created(tmp_path: Path):
    # Create nested path that doesn't exist initially
    log_dir = tmp_path / "nested" / "logs"
    log_path = log_dir / "events.log"

    bus = EventBus()
    logger = JsonFileLogger(log_path, bus)

    # Emit something
    log_event(
        bus=bus,
        module="test.module",
        event_type=EventType.LOG,
        message="hello",
        payload={},
    )
    logger.close()

    assert log_path.exists()
    content = log_path.read_text(encoding="utf-8").strip()
    assert content  # not empty
