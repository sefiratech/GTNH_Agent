# IPC bridge to in-process Forge mod (if used)
# src/bot_core/net/ipc.py
"""
IPC-based client for Forge mod mode.

This client talks to a Forge mod (inside the 1.7.10 JVM) via a simple
message protocol (e.g. JSON over TCP or Unix socket). The Forge side is
responsible for translating these messages into real Minecraft actions
and snapshots.

The goal is:
- keep Python-side logic simple
- keep a clean PacketClient interface
- leave room for richer message schemas later
"""

from __future__ import annotations

import json
import logging
import socket
from dataclasses import dataclass
from threading import Lock
from typing import Any, Dict, Mapping

from .client import PacketClient, PacketHandler

log = logging.getLogger(__name__)


@dataclass
class IpcConfig:
    """
    Configuration for IPC client.

    Assumes the Forge mod listens on a host/port pair specified in the
    EnvProfile. You can later extend this to support Unix sockets, etc.
    """

    host: str
    port: int


class IpcClient(PacketClient):
    """
    IPC client for communicating with a Forge mod.

    Message format (version 1, suggested):
      - Each message is a single line of UTF-8 JSON.
      - JSON object:
          {
            "type": "<packet_type>",
            "payload": { ... }
          }
    The Forge mod is expected to send and receive messages in this format.
    """

    def __init__(self, env_profile: Any) -> None:
        mc = getattr(env_profile, "minecraft", None)
        if mc is None:
            raise ValueError("EnvProfile is missing 'minecraft' section")

        # Try both attribute and dict access for flexibility.
        ipc_cfg = getattr(mc, "ipc", None) or getattr(mc, "forge_ipc", None) or mc

        host = getattr(ipc_cfg, "host", None) or getattr(ipc_cfg, "ipc_host", None)
        port = getattr(ipc_cfg, "port", None) or getattr(ipc_cfg, "ipc_port", None)

        if host is None or port is None:
            raise ValueError(
                "EnvProfile.minecraft.ipc must provide host and port "
                "(or ipc_host/ipc_port)"
            )

        self._config = IpcConfig(host=str(host), port=int(port))
        self._sock: socket.socket | None = None
        self._handlers: Dict[str, PacketHandler] = {}
        self._lock = Lock()
        self._connected = False

        # Buffer for partial lines
        self._recv_buffer = b""

    # ------------------------------------------------------------------
    # PacketClient protocol
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """Open IPC connection to Forge mod."""
        if self._connected:
            return

        log.info("IpcClient connecting to %s:%d", self._config.host, self._config.port)
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((self._config.host, self._config.port))
        sock.setblocking(False)

        self._sock = sock
        self._connected = True

    def disconnect(self) -> None:
        """Close IPC connection."""
        with self._lock:
            if not self._connected:
                return
            log.info("IpcClient disconnecting")
            try:
                if self._sock is not None:
                    self._sock.close()
            finally:
                self._sock = None
                self._connected = False

    def tick(self) -> None:
        """
        Pump IPC socket and dispatch any complete messages.

        This reads available bytes, splits on newline, decodes JSON, and
        dispatches based on "type" field.
        """
        if not self._connected or self._sock is None:
            return

        try:
            chunk = self._sock.recv(4096)
        except BlockingIOError:
            # No data available this tick.
            return
        except OSError:
            log.exception("IpcClient socket error, disconnecting")
            self.disconnect()
            return

        if not chunk:
            # EOF
            log.info("IpcClient received EOF; disconnecting")
            self.disconnect()
            return

        self._recv_buffer += chunk

        while b"\n" in self._recv_buffer:
            line, self._recv_buffer = self._recv_buffer.split(b"\n", 1)
            line = line.strip()
            if not line:
                continue
            self._handle_raw_line(line)

    def send_packet(self, packet_type: str, data: Mapping[str, Any]) -> None:
        """
        Send a JSON message to the Forge mod.

        Format:
          {"type": "<packet_type>", "payload": { ... }}
        """
        if not self._connected or self._sock is None:
            raise RuntimeError("IpcClient is not connected")

        msg = {"type": packet_type, "payload": dict(data)}
        encoded = json.dumps(msg, separators=(",", ":")).encode("utf-8") + b"\n"

        with self._lock:
            self._sock.sendall(encoded)

    def on_packet(self, packet_type: str, handler: PacketHandler) -> None:
        """Register a handler for incoming IPC messages of a given type."""
        self._handlers[packet_type] = handler

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _handle_raw_line(self, line: bytes) -> None:
        """Decode a JSON line and dispatch to the appropriate handler."""
        try:
            obj = json.loads(line.decode("utf-8"))
        except Exception:
            log.exception("IpcClient failed to decode JSON line: %r", line)
            return

        packet_type = obj.get("type")
        payload = obj.get("payload", {})

        if not isinstance(packet_type, str):
            log.warning("IpcClient received message without valid type: %r", obj)
            return
        if not isinstance(payload, dict):
            log.warning("IpcClient received message with non-dict payload: %r", obj)
            return

        handler = self._handlers.get(packet_type)
        if handler is None:
            log.debug("IpcClient no handler for packet_type=%s", packet_type)
            return

        try:
            handler(payload)
        except Exception:
            log.exception("Error in IPC handler for %s", packet_type)

