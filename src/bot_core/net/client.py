# external protocol client for MC 1.7.10
# src/bot_core/net/client.py
"""
Client abstraction for bot_core_1_7_10.

Defines the PacketClient protocol used by bot_core, plus a factory
for constructing a concrete client based on the Phase 0 environment
configuration (env.yaml / EnvProfile).
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, MutableMapping, Protocol

from env.loader import load_environment  # Phase 0: EnvProfile loader

# Type alias for packet handlers.
PacketHandler = Callable[[Mapping[str, Any]], None]


class PacketClient(Protocol):
    """
    Abstract interface for a packet-level Minecraft client.

    Implementations:
    - ExternalClient (full 1.7.10 protocol client)
    - IpcClient (Forge mod IPC bridge)
    """

    def connect(self) -> None:
        """Establish connection and complete handshake."""
        ...

    def disconnect(self) -> None:
        """Cleanly disconnect from the server or IPC endpoint."""
        ...

    def tick(self) -> None:
        """
        Pump network/IPC events, calling registered handlers and updating
        internal state. Should be called regularly from the main loop.
        """
        ...

    def send_packet(self, packet_type: str, data: Mapping[str, Any]) -> None:
        """
        Send a high-level packet representation.

        The exact mapping of packet_type → wire format is implementation
        specific. This function is the only way bot_core should emit data.
        """
        ...

    def on_packet(self, packet_type: str, handler: PacketHandler) -> None:
        """
        Register a handler for packets/messages of a given type.

        Handlers receive a decoded mapping representation of the packet
        payload. Implementations are responsible for parsing wire data
        into this format.
        """
        ...


def create_packet_client_for_env() -> PacketClient:
    """
    Construct an appropriate PacketClient based on the current EnvProfile.

    This uses Phase 0 config (env.yaml) and assumes EnvProfile contains:
      - bot_mode: Literal["external_client", "forge_mod"]
      - minecraft: object or mapping with connection details

    The concrete implementations live in:
      - external_client.ExternalClient
      - ipc.IpcClient
    """
    env = load_environment()
    bot_mode = getattr(env, "bot_mode", None)

    # Lazy imports to avoid cycles.
    if bot_mode == "external_client":
        from .external_client import ExternalClient

        return ExternalClient(env)
    elif bot_mode == "forge_mod":
        from .ipc import IpcClient

        return IpcClient(env)
    else:
        raise ValueError(f"Unknown bot_mode in EnvProfile: {bot_mode!r}")

