# src/bot_core/net/external_client.py
"""
External protocol client for bot_core_1_7_10.

This client is responsible for speaking the actual 1.7.10 protocol
to a Minecraft/GTNH server. It implements PacketClient but leaves
protocol details as TODOs for later implementation.

Right now this is a skeleton with clear extension points.
"""

from __future__ import annotations

import logging
import socket
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Dict, Mapping, MutableMapping

from .client import PacketClient, PacketHandler

log = logging.getLogger(__name__)


@dataclass
class ExternalClientConfig:
    """
    Connection parameters for the external protocol client.

    This is expected to be derived from EnvProfile.minecraft.* fields.
    """

    host: str
    port: int
    # Future extension: username, auth token, protocol version, etc.


class ExternalClient(PacketClient):
    """
    External Minecraft protocol client for 1.7.10.

    NOTE:
    - This is intentionally minimal and NOT a full implementation.
    - It defines structure, handler registration, and a basic TCP loop.
    - Actual protocol encoding/decoding must be added later.
    """

    def __init__(self, env_profile: Any) -> None:
        """
        Build the client from an EnvProfile.

        We assume env_profile.minecraft.connection.host/port or similar.
        """
        mc = getattr(env_profile, "minecraft", None)
        if mc is None:
            raise ValueError("EnvProfile is missing 'minecraft' section")

        # Try both attribute and dict access, because schemas evolve.
        host = getattr(mc, "host", None) or getattr(
            getattr(mc, "connection", None) or mc, "host", None
        )
        port = getattr(mc, "port", None) or getattr(
            getattr(mc, "connection", None) or mc, "port", None
        )

        if host is None or port is None:
            raise ValueError("EnvProfile.minecraft must provide host and port")

        self._config = ExternalClientConfig(host=str(host), port=int(port))
        self._sock: socket.socket | None = None

        self._handlers: Dict[str, PacketHandler] = {}
        self._lock = Lock()
        self._connected: bool = False

    # ------------------------------------------------------------------
    # PacketClient protocol
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """Open a TCP connection to the Minecraft server."""
        if self._connected:
            return

        log.info(
            "ExternalClient connecting to %s:%d",
            self._config.host,
            self._config.port,
        )
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((self._config.host, self._config.port))

        # TODO: send 1.7.10 handshake and login sequence here.
        # For now we just mark as connected.
        self._sock = sock
        self._connected = True

    def disconnect(self) -> None:
        """Close the TCP connection."""
        with self._lock:
            if not self._connected:
                return
            log.info("ExternalClient disconnecting")
            try:
                if self._sock is not None:
                    self._sock.close()
            finally:
                self._sock = None
                self._connected = False

    def tick(self) -> None:
        """
        Pump the network socket and dispatch any received data.

        In a real implementation this would:
          - read framed packets from the socket
          - decode them into {type, payload}
          - call registered handlers
        """
        if not self._connected or self._sock is None:
            return

        # This is a placeholder to keep the loop safe.
        # You will eventually implement nonblocking reads + packet parsing here.
        return

    def send_packet(self, packet_type: str, data: Mapping[str, Any]) -> None:
        """
        Encode a high-level packet into the 1.7.10 wire format and send it.

        Right now this just logs, to avoid pretending we have a protocol layer.
        """
        if not self._connected or self._sock is None:
            raise RuntimeError("ExternalClient is not connected")

        log.debug("send_packet(type=%s, data=%r)", packet_type, dict(data))

        # TODO: map packet_type+data to real 1.7.10 packet bytes.
        # For now we do not write anything to the socket.

    def on_packet(self, packet_type: str, handler: PacketHandler) -> None:
        """Register a handler for decoded incoming packets."""
        self._handlers[packet_type] = handler

    # ------------------------------------------------------------------
    # Internal helpers (to be filled in when protocol is implemented)
    # ------------------------------------------------------------------

    def _dispatch_packet(self, packet_type: str, payload: Mapping[str, Any]) -> None:
        """Call the handler registered for a given packet_type, if any."""
        handler = self._handlers.get(packet_type)
        if handler is not None:
            try:
                handler(payload)
            except Exception:
                log.exception("Error in packet handler for %s", packet_type)

