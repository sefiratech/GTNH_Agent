#tests/test_monitoring_event_bus.py
"""
Tests for monitoring.bus.EventBus

Covers:
- Publish/subscribe behavior
- Unsubscribe behavior
- Ordering guarantees
- Basic thread-safety smoke check
"""

from __future__ import annotations

import threading
import time
from typing import List

from monitoring.bus import EventBus
from monitoring.events import MonitoringEvent, EventType


def make_event(ts: float, module: str = "test", msg: str = "msg") -> MonitoringEvent:
    return MonitoringEvent(
        ts=ts,
        module=module,
        event_type=EventType.LOG,
        message=msg,
        payload={},
        correlation_id=None,
    )


def test_event_bus_publish_subscribe_basic():
    bus = EventBus()
    received: List[MonitoringEvent] = []

    def subscriber(evt: MonitoringEvent) -> None:
        received.append(evt)

    bus.subscribe(subscriber)

    e1 = make_event(1.0, msg="first")
    e2 = make_event(2.0, msg="second")

    bus.publish(e1)
    bus.publish(e2)

    assert len(received) == 2
    assert received[0].message == "first"
    assert received[1].message == "second"


def test_event_bus_unsubscribe():
    bus = EventBus()
    received: List[MonitoringEvent] = []

    def subscriber(evt: MonitoringEvent) -> None:
        received.append(evt)

    bus.subscribe(subscriber)
    bus.unsubscribe(subscriber)

    bus.publish(make_event(1.0))

    # Should not receive anything after unsubscribe
    assert received == []


def test_event_bus_ordering_guarantee():
    bus = EventBus()
    seen: List[int] = []

    def subscriber(evt: MonitoringEvent) -> None:
        seen.append(int(evt.ts))

    bus.subscribe(subscriber)

    # Publish in ascending order
    for ts in [1, 2, 3, 4, 5]:
        bus.publish(make_event(float(ts)))

    assert seen == [1, 2, 3, 4, 5]


def test_event_bus_thread_safety_smoke():
    """
    Smoke test: multiple threads publishing simultaneously should not crash
    and subscribers should receive the correct number of events.
    """
    bus = EventBus()
    count = 100

    received: List[MonitoringEvent] = []
    lock = threading.Lock()

    def subscriber(evt: MonitoringEvent) -> None:
        with lock:
            received.append(evt)

    bus.subscribe(subscriber)

    def publisher_thread(start: int) -> None:
        for i in range(start, start + count):
            bus.publish(make_event(float(i)))

    threads = [
        threading.Thread(target=publisher_thread, args=(0,)),
        threading.Thread(target=publisher_thread, args=(1000,)),
    ]

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # We expect 2 * count events
    assert len(received) == 2 * count

