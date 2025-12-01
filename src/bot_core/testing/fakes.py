# src/bot_core/testing/fakes.py
"""
Test helpers for bot_core_1_7_10.

Provides:
- FakePacketClient: in-memory PacketClient implementation for unit tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping

from ..net import PacketClient, PacketHandler


@dataclass
class SentPacket:
    """Record of a packet sent through FakePacketClient."""

    packet_type: str
    data: Dict[str, Any]


class FakePacketClient(PacketClient):
    """
    In-memory PacketClient used for unit and integration tests.

    Features:
    - Records all packets sent via send_packet().
    - Allows manual emission of incoming packets to registered handlers.
    - No real network or IPC.
    """

    def __init__(self) -> None:
        self.connected: bool = False
        self.sent_packets: List[SentPacket] = []
        self._handlers: Dict[str, PacketHandler] = {}

    # ------------------------------------------------------------------
    # PacketClient protocol
    # ------------------------------------------------------------------

    def connect(self) -> None:
        self.connected = True

    def disconnect(self) -> None:
        self.connected = False

    def tick(self) -> None:
        """
        No-op for FakePacketClient.

        In more advanced tests you could schedule "queued" messages here.
        """
        return

    def send_packet(self, packet_type: str, data: Mapping[str, Any]) -> None:
        self.sent_packets.append(
            SentPacket(packet_type=packet_type, data=dict(data))
        )

    def on_packet(self, packet_type: str, handler: PacketHandler) -> None:
        self._handlers[packet_type] = handler

    # ------------------------------------------------------------------
    # Test-only helpers
    # ------------------------------------------------------------------

    def emit(self, packet_type: str, payload: Mapping[str, Any]) -> None:
        """
        Manually trigger a packet event for tests.

        This calls the registered handler (if any) with the given payload.
        """
        handler = self._handlers.get(packet_type)
        if handler is not None:
            handler(payload)

