# bot_core.net package
# src/bot_core/net/__init__.py
"""
Network layer for bot_core_1_7_10.

This package provides:
- PacketClient protocol (common interface)
- Concrete client implementations for:
    - external_client: full protocol client
    - forge_mod: IPC-based client
- Factory helpers wired to the Phase 0 environment config.
"""

from __future__ import annotations

from .client import (
    PacketClient,
    PacketHandler,
    create_packet_client_for_env,
)

__all__ = [
    "PacketClient",
    "PacketHandler",
    "create_packet_client_for_env",
]

