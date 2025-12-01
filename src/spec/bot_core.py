# BotCore interface definition
# src/spec/bot_core.py

from __future__ import annotations

from typing import Protocol

from .types import WorldState, Action, ActionResult


class BotCore(Protocol):
    """Abstract interface for a controllable Minecraft agent body.

    This is the Mineflayer-adjacent "body" layer:
    - maintains connection to the world (SP or server)
    - tracks semantic world state
    - maps high-level Actions to concrete navigation / interaction
    """

    def connect(self) -> None:
        """Connect to the Minecraft world (SP or server) and start receiving updates."""
        ...

    def disconnect(self) -> None:
        """Cleanly disconnect from the world."""
        ...

    def get_world_state(self) -> WorldState:
        """Return the latest known semantic world state."""
        ...

    def execute_action(self, action: Action) -> ActionResult:
        """
        Execute a single high-level action and return the result.

        Pathfinding, movement, and low-level interaction happen under the hood.
        From AgentLoop's perspective this is a single opaque step.
        """
        ...

    def tick(self) -> None:
        """
        Advance any internal event loops for the bot core.

        Intended to be called regularly by the runtime (e.g. once per main loop
        iteration) to process incoming events, keep state fresh, etc.
        """
        ...

