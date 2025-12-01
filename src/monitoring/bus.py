# EventBus for monitoring events and control commands
"""
Event bus for M9 – monitoring_and_tools.

Provides a minimal, thread-safe, in-process pub/sub mechanism:

- Subscribers receive MonitoringEvent objects.
- Command handlers receive ControlCommand objects.
- Used by:
    - TUI dashboard
    - File-based logger
    - AgentLoop (M8) instrumentation
    - Dev tools / scripts
"""

from __future__ import annotations

from threading import Lock
from typing import Callable, List

from .events import MonitoringEvent, ControlCommand


# ============================================================
# Type aliases
# ============================================================

SubscriberFn = Callable[[MonitoringEvent], None]
CommandHandlerFn = Callable[[ControlCommand], None]


# ============================================================
# Event Bus
# ============================================================

class EventBus:
    """
    Simple in-process event bus for monitoring events and control commands.

    Design goals:
    - Minimal: no external dependencies or IPC.
    - Thread-safe: subscribers list protected by a Lock.
    - Non-blocking-ish: each publish iterates over a snapshot of subscribers.
    """

    def __init__(self) -> None:
        # Registered monitoring event subscribers
        self._subscribers: List[SubscriberFn] = []
        # Registered control command handlers
        self._cmd_handlers: List[CommandHandlerFn] = []
        # Single lock guarding both lists
        self._lock = Lock()

    # --------------------------------------------------------
    # Subscription API: Monitoring Events
    # --------------------------------------------------------

    def subscribe(self, fn: SubscriberFn) -> None:
        """
        Register a subscriber to receive MonitoringEvent instances.

        Subscribers MUST NOT throw exceptions; if they do, it's on them.
        """
        with self._lock:
            self._subscribers.append(fn)

    def unsubscribe(self, fn: SubscriberFn) -> None:
        """
        Remove a previously registered MonitoringEvent subscriber.

        Safe to call even if `fn` is not present.
        """
        with self._lock:
            if fn in self._subscribers:
                self._subscribers.remove(fn)

    # --------------------------------------------------------
    # Subscription API: Control Commands
    # --------------------------------------------------------

    def subscribe_commands(self, fn: CommandHandlerFn) -> None:
        """
        Register a handler to receive ControlCommand instances.
        """
        with self._lock:
            self._cmd_handlers.append(fn)

    def unsubscribe_commands(self, fn: CommandHandlerFn) -> None:
        """
        Remove a previously registered ControlCommand handler.

        Safe to call even if `fn` is not present.
        """
        with self._lock:
            if fn in self._cmd_handlers:
                self._cmd_handlers.remove(fn)

    # --------------------------------------------------------
    # Publish API: Monitoring Events
    # --------------------------------------------------------

    def publish(self, event: MonitoringEvent) -> None:
        """
        Publish a MonitoringEvent to all subscribers.

        Takes a snapshot of subscribers under the lock, then iterates without
        holding the lock to avoid deadlocks if subscribers call back into the bus.
        """
        with self._lock:
            subscribers = list(self._subscribers)

        for fn in subscribers:
            try:
                fn(event)
            except Exception:
                # Deliberately swallow exceptions to avoid one bad subscriber
                # killing the event stream. If you want to log this, hook a
                # dedicated error subscriber.
                pass

    # --------------------------------------------------------
    # Publish API: Control Commands
    # --------------------------------------------------------

    def publish_command(self, cmd: ControlCommand) -> None:
        """
        Publish a ControlCommand to all registered command handlers.
        """
        with self._lock:
            handlers = list(self._cmd_handlers)

        for fn in handlers:
            try:
                fn(cmd)
            except Exception:
                # Same rationale as for events: bad handlers shouldn't break others.
                pass

    # --------------------------------------------------------
    # Utility
    # --------------------------------------------------------

    def clear(self) -> None:
        """
        Clear all subscribers and handlers.

        Mostly useful for tests; probably not what you want in production.
        """
        with self._lock:
            self._subscribers.clear()
            self._cmd_handlers.clear()


# ============================================================
# Optional: shared bus instance
# ============================================================

# Many runtimes will prefer to use a single, process-wide bus.
# You can import `default_bus` where DI is annoying, or ignore this
# and manage EventBus instances explicitly.
default_bus = EventBus()

