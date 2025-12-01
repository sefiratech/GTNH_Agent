# Module M6 — bot_core_1_7_10

## Purpose
Provide a stable, testable Minecraft 1.7.10 “body” that any controller (M8+) can use. This module hides protocol details, tracks world state, exposes a clean Action→WorldState interface, and ensures future modules (M7–M11) can attach without refactoring.

## Overview
Capabilities:
- Connect / disconnect / tick loop
- Packet or IPC-based world ingestion
- World tracking (chunks, entities, inventory, position)
- Navigation (grid, pathfinding, mover)
- Action execution (move, break, place, interact)

API:
- observe() → RawWorldSnapshot  
- get_world_state() → WorldState  
- execute_action(Action) → ActionResult  

Dependencies: M0, M1  
Difficulty: ⭐⭐⭐⭐  
Scalability: modular subsystems; cached chunk/entity diffs; stable action/state schema

## 1. Responsibilities & Boundaries

### 1.1 What M6 owns
- Network/IPC connection lifecycle
- World tracking via packet ingestion
- Navigation: grid, walkability, A* pathfinding, mover→actions
- High-level actions mapped to MC 1.7.10 packets

### 1.2 What M6 never owns
- No planning, no virtues, no skills
- No semantics or tech inference
- No curriculum logic
- No LLM calls



## Important Design Choices:

1 Connection mode - B Forge mod IPC 
2 Action granularity - A high-level atomic-ish actions 
3 Nav ownership - A Python NavGrid/A* 
4 Pacing - A one execute_action per whole operation 
5 Failure semantics - A explicit, structured failures 
6 Semantic boundary - A M6 stays thin (no tech semantics, no virtues)

Notes:

## Where this setup can hurt you if you’re sloppy

Now the fun part: systemic failure modes.

### 3.1 “Atomic action” turning into a black hole

You chose:

- High-level actions
    
- One `execute_action` per whole operation
    

Design risk:

- If `execute_action` behaves like:
    
    - “Block the caller while I internally loop, send 500 IPC messages, and then eventually return success/failure,”
        
- Then M8, monitoring, and learning see almost nothing but:
    
    - “Try action A → 2 seconds of silence → final Result.”
        

**Why this is bad systemically:**

- You can’t **re-plan mid-action** when something changes (mob appears, server lag, chunk unloads).
    
- Learning (M10) only sees very coarse transitions:
    
    - State₀ → [mystery fog] → State₁  
        instead of a usable micro-trace.
        
- Monitoring (M9) can’t show meaningful progress: the dashboard just hangs on “Executing…”
    

**Mitigation requirement:**

Even with “one `execute_action` per operation,” M6 must internally:

- Run a **tick-level state machine** for each action.
    
- Emit regular **events**:
    
    - `action_progress`, `nav_step`, `blocked_path`, `took_damage`, etc.
        
- Provide a way for M8 / monitoring to:
    
    - **Cancel** or **preempt** the running action.
        

So from M8’s _API_ perspective, it’s atomic-ish.  
From the system’s _internals_, it’s a multi-step process with observable structure.

### Python nav + IPC: partial knowledge problems

You chose Python NavGrid + A* while using IPC. Design stress point:

- If Forge mod sends **only summaries** (“here’s a player pos, some entities, good luck”), then NavGrid has trash data.
    
- With incomplete/low-resolution chunk info, A* becomes:
    
    - Overoptimistic (“path goes through unloaded void”)
        
    - Overconservative (“no path” when there actually is one)
        

System effect:

- Planner thinks “bot is stuck” when it’s not.
    
- Virtue lattice starts preferring dumb but “safe” plans.
    
- Curriculum gets noisy signals about “base layout” performance.
    

**Mitigation requirement:**

Your IPC schema must expose enough map data for nav to be sane:

- At minimum:
    
    - Block collision for a radius around the player.
        
    - Vertical walkable columns.
        
- Ideally:
    
    - A bounded, cached chunk window around the bot.
        

As long as the Forge side sends a **consistent collision view**, Python nav is fine.


### Reactivity vs. long-running actions

With high-level actions and full-operation pacing, you risk:

- Bot commits to a long `move_to` or `break_block` while:
    
    - Mob walks up
        
    - Lava flows near feet
        
    - Inventory fills
        
    - Tool breaks
        

If M6 does not support **preemption**:

- M8 can’t say “abort current action, emergency retreat.”
    
- Monitoring can’t “manual override” the bot quickly.
    
- Safety-related virtues (M4) become theoretical.
    

**Mitigation requirement:**

The combo only stays good if you design:

- A **cancel/abort mechanism**:
    
    - `execute_action` must watch for an abort flag / higher-level command.
        
- A separate, lightweight:
    
    - Event stream for `danger` / `anomaly` events feeding M8/M9.


## Design Choice Interpretations
I’d only refine the _interpretation_ of two:

1. **Pacing (A)**:  
    Treat “one `execute_action` per whole operation” as:
    
    - “One _logical_ operation per call,”  
        **not** “one monolithic blocking blob with no observability.”
        
2. **Action granularity (A)**:  
    Keep actions high-level, but define a **small, orthogonal set**:
    
    - `move_to`, `look_at`, `break_block`, `place_block`, `use_item_on`, maybe `wait_until(condition, timeout)`.  
        Don’t let M6 explode into 50 weird special actions.
        

If you adhere to that, you **do not need an alternate configuration**. The system is structurally sound and future-proof enough for what you’re trying to build.

---

## 2. Architectural Layout

```
src/bot_core/
  core.py           # BotCoreImpl
  actions.py        # ActionExecutor
  snapshot.py       # RawWorldSnapshot + conversions
  world_tracker.py  # chunk/entity tracking
  net/
    client.py       # PacketClient protocol
    ipc.py          # Forge mod IPC adapter
  nav/
    grid.py         # NavGrid abstraction
    pathfinder.py   # A*
    mover.py        # path → actions
```

BotCoreImpl must obey the spec in `src/spec/bot_core.py`, powering M8 without leaking low-level details.

---

## 3. Core Data Structures

### RawWorldSnapshot
- tick  
- dimension  
- player_pos {x,y,z}, yaw, pitch  
- on_ground  
- chunks {(cx,cz) → RawChunk}  
- entities [RawEntity]  
- inventory list  
- context map  

### WorldState (from M1)
Built from RawWorldSnapshot using `snapshot_to_world_state`. Must remain stable.

Here’s a **complete `src/bot_core/snapshot.py`** you can drop in and use to back M6’s `RawWorldSnapshot` → `WorldState` bridge.
python:
```
# src/bot_core/snapshot.py
"""
Snapshot structures for bot_core_1_7_10.

This module defines the raw world snapshot types used internally by M6 and
provides the adapter that converts a RawWorldSnapshot into the canonical
WorldState type defined in `spec.types`.

Design goals:
- Keep Raw* structures close to the data we ingest from the network / IPC.
- Keep WorldState stable and M1-owned; this module only adapts into it.
- Do not embed GTNH semantics or tech inference here. That is M3’s job.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, MutableMapping, Tuple


# ---------------------------------------------------------------------------
# Raw world types (internal to bot_core)
# ---------------------------------------------------------------------------


@dataclass
class RawChunk:
    """
    Raw representation of a chunk as seen by the bot_core.

    This is intentionally opaque with respect to block storage layout.
    Higher-level semantics and block categorization are handled by M3.
    """

    x: int
    z: int
    blocks: Any  # implementation-defined representation (e.g. 16x256x16 array)
    biome_data: Any | None = None


@dataclass
class RawEntity:
    """
    Raw entity data as captured from packets or IPC messages.

    `data` should contain the original payload or a thin normalized view
    (e.g. type-specific fields, NBT-ish data, health, motion).
    """

    entity_id: int
    kind: str  # "player", "mob", "item", etc.
    x: float
    y: float
    z: float
    data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RawWorldSnapshot:
    """
    Full raw snapshot of the world as tracked by M6.

    This is the canonical "raw" view for bot_core. All other subsystems
    (observation encoder, semantics, planner) should work with WorldState,
    not this type.
    """

    tick: int
    dimension: str

    player_pos: Dict[str, float]  # {"x": float, "y": float, "z": float}
    player_yaw: float
    player_pitch: float
    on_ground: bool

    chunks: Dict[Tuple[int, int], RawChunk]
    entities: List[RawEntity]
    inventory: List[Dict[str, Any]]

    # Misc runtime context, used to carry extra signals such as:
    # - current env profile metadata
    # - high-level tags (near machines, hazards, etc.)
    # - bridge info from M0/M1
    context: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# WorldState adapter (RawWorldSnapshot -> spec.types.WorldState)
# ---------------------------------------------------------------------------


def _entity_to_summary(entity: RawEntity) -> Dict[str, Any]:
    """
    Convert a RawEntity to the generic "nearby_entities" dict format
    expected by WorldState.

    The exact structure is intentionally simple and stable; richer per-entity
    semantics belong in M3 / semantics.
    """
    return {
        "id": entity.entity_id,
        "kind": entity.kind,
        "x": entity.x,
        "y": entity.y,
        "z": entity.z,
        "data": dict(entity.data),
    }


def _compute_blocks_of_interest(snapshot: RawWorldSnapshot) -> List[Dict[str, Any]]:
    """
    Extract a shallow 'blocks_of_interest' list from the snapshot.

    M6 does not know GTNH tech semantics, so this is intentionally minimal.
    You can later extend this to surface:
      - blocks directly under / around the player
      - known "front-of-player" interaction targets
      - cached points of interest maintained by world_tracker

    For now, this returns an empty list; semantics modules can derive
    more specific views directly from RawWorldSnapshot + chunk data if needed.
    """
    # Placeholder: no opinion about which blocks are "interesting" at M6.
    # This keeps M6 decoupled from GTNH semantics (M3).
    return []


def snapshot_to_world_state(snapshot: RawWorldSnapshot) -> "WorldState":
    """
    Adapt a RawWorldSnapshot into the canonical WorldState.

    This is the ONLY place bot_core should construct a WorldState. All other
    code should either:
      - operate on RawWorldSnapshot, or
      - consume WorldState produced here.

    Expected WorldState fields (as defined in spec.types):
      - tick: int
      - position: Dict[str, float]
      - dimension: str
      - inventory: List[Dict[str, Any]]
      - nearby_entities: List[Dict[str, Any]]
      - blocks_of_interest: List[Dict[str, Any]]
      - tech_state: Mapping[str, Any] or similar (filled by M3)
      - context: Dict[str, Any]
    """
    # Import here to avoid circular imports at module import time.
    from spec.types import WorldState  # type: ignore

    # Shallow copy to avoid surprising mutations from outside.
    position = {
        "x": float(snapshot.player_pos.get("x", 0.0)),
        "y": float(snapshot.player_pos.get("y", 0.0)),
        "z": float(snapshot.player_pos.get("z", 0.0)),
    }

    nearby_entities: List[Dict[str, Any]] = [
        _entity_to_summary(e) for e in snapshot.entities
    ]

    blocks_of_interest = _compute_blocks_of_interest(snapshot)

    # tech_state is intentionally empty at this layer.
    tech_state: Dict[str, Any] = {}

    world_state = WorldState(
        tick=int(snapshot.tick),
        position=position,
        dimension=str(snapshot.dimension),
        inventory=[dict(stack) for stack in snapshot.inventory],
        nearby_entities=nearby_entities,
        blocks_of_interest=blocks_of_interest,
        tech_state=tech_state,
        context=dict(snapshot.context),
    )

    return world_state


__all__ = [
    "RawChunk",
    "RawEntity",
    "RawWorldSnapshot",
    "snapshot_to_world_state",
]

```




---

## 4. Network Layer

Two modes (controlled by env.yaml / M0):
1. external_client: full protocol client  
2. forge_mod: IPC via sockets/ZeroMQ

PacketClient protocol:
- connect()/disconnect()
- tick() — pump events
- send_packet(type,data)
- on_packet(type,handler)

```
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

```

```
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

```

```
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

```

```
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

```





---

## 5. World Tracker

Consumes packet events:
- time_update → tick
- position_update → player pos/rotation
- chunk_data → update chunk map
- spawn_entity/destroy_entities
- inventory set_slot/window_items

Produces RawWorldSnapshot via build_snapshot().

Tracker rules:
- Never assemble WorldState itself  
- Never embed semantics  
- Store raw, minimal structures  
- Update incrementally  


```
# src/bot_core/world_tracker.py
"""
World tracker for bot_core_1_7_10.

Consumes normalized packet / IPC events from a PacketClient and maintains
a raw, incrementally updated view of the world. This module is the ONLY
owner of RawWorldSnapshot assembly.

Rules:
- Never construct WorldState here (that lives in bot_core.snapshot).
- Never embed GTNH semantics or tech-state inference (M3 owns that).
- Keep storage minimal and "raw"; avoid bloated derived structures.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Tuple

from .snapshot import RawWorldSnapshot, RawChunk, RawEntity
from .net import PacketClient, PacketHandler


@dataclass
class _PlayerState:
    """Minimal tracked state for the local player."""

    pos: Dict[str, float] = field(
        default_factory=lambda: {"x": 0.0, "y": 64.0, "z": 0.0}
    )
    yaw: float = 0.0
    pitch: float = 0.0
    on_ground: bool = True
    dimension: str = "overworld"


class WorldTracker:
    """
    Maintains an incrementally updated RawWorldSnapshot.

    PacketClient implementations must normalize wire data into logically
    named event types:

        - "time_update"        → tick/time updates
        - "position_update"    → player position/rotation
        - "chunk_data"         → chunk or region data
        - "spawn_entity"       → entity created
        - "destroy_entities"   → entities destroyed
        - "set_slot"           → single inventory slot changed
        - "window_items"       → full inventory snapshot
        - (optionally) "dimension_change" → dimension switch

    This tracker does NOT:
      - Interpret block types or GTNH tech semantics
      - Compute tech_state
      - Talk to LLMs

    It only stores enough to build a coherent RawWorldSnapshot.
    """

    def __init__(self, client: PacketClient) -> None:
        self._client = client

        # Basic scalar world state
        self._tick: int = 0
        self._player: _PlayerState = _PlayerState()

        # Chunks keyed by (chunk_x, chunk_z)
        self._chunks: Dict[Tuple[int, int], RawChunk] = {}

        # Entities keyed by numeric ID
        self._entities: Dict[int, RawEntity] = {}

        # Flat inventory representation (list of slot dicts)
        self._inventory: List[Dict[str, Any]] = []

        # Context map for misc metadata (env profile, hazards, etc.)
        self._context: Dict[str, Any] = {}

        # Wire client → handlers
        self._register_handlers()

    # ------------------------------------------------------------------
    # Packet wiring
    # ------------------------------------------------------------------

    def _register_handlers(self) -> None:
        """
        Register handlers for the normalized event types we care about.

        PacketClient is responsible for translating real wire protocol /
        IPC messages into these logical events.
        """
        self._client.on_packet("time_update", self._handle_time_update)
        self._client.on_packet("position_update", self._handle_position_update)
        self._client.on_packet("chunk_data", self._handle_chunk_data)
        self._client.on_packet("spawn_entity", self._handle_spawn_entity)
        self._client.on_packet("destroy_entities", self._handle_destroy_entities)
        self._client.on_packet("set_slot", self._handle_set_slot)
        self._client.on_packet("window_items", self._handle_window_items)
        # Optional, but useful if IPC exposes it:
        self._client.on_packet("dimension_change", self._handle_dimension_change)

    # ------------------------------------------------------------------
    # Packet handlers
    # ------------------------------------------------------------------

    def _handle_time_update(self, pkt: Mapping[str, Any]) -> None:
        """
        Update world tick/time from server.

        Expected fields (normalized by client / IPC layer):
            - "tick" or "time": int
        """
        value = pkt.get("tick", pkt.get("time", self._tick))
        try:
            self._tick = int(value)
        except (TypeError, ValueError):
            # Leave tick unchanged on garbage input.
            return

    def _handle_position_update(self, pkt: Mapping[str, Any]) -> None:
        """
        Update player position/rotation.

        Expected fields:
            - "x", "y", "z": float
            - "yaw", "pitch": float
            - "on_ground": bool (optional)
        """
        x = pkt.get("x", self._player.pos["x"])
        y = pkt.get("y", self._player.pos["y"])
        z = pkt.get("z", self._player.pos["z"])

        try:
            self._player.pos = {
                "x": float(x),
                "y": float(y),
                "z": float(z),
            }
        except (TypeError, ValueError):
            # Ignore malformed position updates.
            pass

        yaw = pkt.get("yaw")
        pitch = pkt.get("pitch")
        if yaw is not None:
            try:
                self._player.yaw = float(yaw)
            except (TypeError, ValueError):
                pass
        if pitch is not None:
            try:
                self._player.pitch = float(pitch)
            except (TypeError, ValueError):
                pass

        if "on_ground" in pkt:
            self._player.on_ground = bool(pkt["on_ground"])

    def _handle_chunk_data(self, pkt: Mapping[str, Any]) -> None:
        """
        Store or update chunk data.

        Expected fields:
            - "chunk_x", "chunk_z": ints
            - "blocks": opaque block storage
            - "biomes": optional biome data
        """
        try:
            cx = int(pkt["chunk_x"])
            cz = int(pkt["chunk_z"])
        except Exception:
            return

        blocks = pkt.get("blocks")
        biomes = pkt.get("biomes")

        if blocks is None:
            # No block data, nothing to store.
            return

        self._chunks[(cx, cz)] = RawChunk(
            x=cx,
            z=cz,
            blocks=blocks,
            biome_data=biomes,
        )

    def _handle_spawn_entity(self, pkt: Mapping[str, Any]) -> None:
        """
        Track newly spawned entities.

        Expected fields:
            - "entity_id": int
            - "kind" or "type": str
            - "x", "y", "z": float
            - plus any additional data (kept in 'data')
        """
        try:
            entity_id = int(pkt["entity_id"])
        except Exception:
            return

        kind = pkt.get("kind", pkt.get("type", "unknown"))

        try:
            x = float(pkt.get("x", 0.0))
            y = float(pkt.get("y", 0.0))
            z = float(pkt.get("z", 0.0))
        except (TypeError, ValueError):
            x = y = z = 0.0

        # Keep original payload as data for now.
        data = dict(pkt)

        self._entities[entity_id] = RawEntity(
            entity_id=entity_id,
            kind=str(kind),
            x=x,
            y=y,
            z=z,
            data=data,
        )

    def _handle_destroy_entities(self, pkt: Mapping[str, Any]) -> None:
        """
        Remove entities that the server reports as destroyed.

        Expected fields:
            - "entity_ids": iterable of ints
        """
        ids = pkt.get("entity_ids")
        if not ids:
            return

        for raw_id in ids:
            try:
                eid = int(raw_id)
            except Exception:
                continue
            self._entities.pop(eid, None)

    def _handle_set_slot(self, pkt: Mapping[str, Any]) -> None:
        """
        Update a single inventory slot.

        Expected fields (normalized):
            - "slot": int
            - "item": mapping (stack data) or None
        """
        slot = pkt.get("slot")
        item = pkt.get("item")

        try:
            idx = int(slot)
        except Exception:
            return

        # Ensure list is large enough.
        while len(self._inventory) <= idx:
            self._inventory.append({})

        # Represent empty slot as {} for simplicity.
        self._inventory[idx] = dict(item) if isinstance(item, Mapping) else {}

    def _handle_window_items(self, pkt: Mapping[str, Any]) -> None:
        """
        Replace the entire inventory representation.

        Expected fields:
            - "items": list[Mapping[str, Any]] (one per slot)
        """
        items = pkt.get("items")
        if not isinstance(items, list):
            return

        new_inv: List[Dict[str, Any]] = []
        for entry in items:
            if isinstance(entry, Mapping):
                new_inv.append(dict(entry))
            else:
                new_inv.append({})
        self._inventory = new_inv

    def _handle_dimension_change(self, pkt: Mapping[str, Any]) -> None:
        """
        Handle dimension change events.

        Expected fields:
            - "dimension": str or int
        """
        dim = pkt.get("dimension")
        if dim is None:
            return
        self._player.dimension = str(dim)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def tick(self) -> int:
        """Current world tick as tracked by the server."""
        return self._tick

    @property
    def dimension(self) -> str:
        """Current dimension ID/name."""
        return self._player.dimension

    def set_context(self, key: str, value: Any) -> None:
        """
        Attach arbitrary metadata to the tracker context.

        Useful for storing:
          - env profile labels
          - world seed (if known)
          - runtime tags (e.g. "phase": "m6_online")
        """
        self._context[key] = value

    def update_context(self, data: Mapping[str, Any]) -> None:
        """Bulk update context with a mapping."""
        for k, v in data.items():
            self._context[k] = v

    def build_snapshot(self) -> RawWorldSnapshot:
        """
        Build a RawWorldSnapshot from the current tracked state.

        This is the ONLY way to get a snapshot out of the tracker and the
        ONLY place in M6 that assembles a RawWorldSnapshot.
        """
        return RawWorldSnapshot(
            tick=self._tick,
            dimension=self._player.dimension,
            player_pos=dict(self._player.pos),
            player_yaw=self._player.yaw,
            player_pitch=self._player.pitch,
            on_ground=self._player.on_ground,
            chunks=dict(self._chunks),
            entities=list(self._entities.values()),
            inventory=[dict(stack) for stack in self._inventory],
            context=dict(self._context),
        )


__all__ = ["WorldTracker"]

```

---

## 6. Navigation Subsystem

### NavGrid
Constructed from RawWorldSnapshot. Must expose:
- is_walkable(x,y,z)
- accessible neighbors

Walkability rules derived from block solidity; final implementation delegated to M3 block data later.

### Pathfinding
A* search:
- Manhattan heuristic  
- 4-direction neighbors  
- max_steps guard  

### Mover
Convert path → sequence of move_to Actions:
- move_to {x,y,z,radius}
- Higher layers decide how many steps; M6 keeps low-level control.

```
# src/bot_core/nav/__init__.py
"""
Navigation subsystem for bot_core_1_7_10.

Provides:
- NavGrid: walkability queries over a RawWorldSnapshot
- A* pathfinding: find_path
- Mover: path_to_actions to convert paths into move_to Actions
"""

from __future__ import annotations

from .grid import NavGrid, BlockSolidFn
from .pathfinder import Coord, find_path
from .mover import path_to_actions

__all__ = [
    "NavGrid",
    "BlockSolidFn",
    "Coord",
    "find_path",
    "path_to_actions",
]

```



```
# src/bot_core/nav/grid.py
"""
NavGrid: navigation grid abstraction over RawWorldSnapshot.

This module does not know GTNH semantics. It only:
- Exposes walkability queries.
- Uses a pluggable is_solid callback to decide collisions.

Actual "what blocks are solid" logic belongs to M3 (semantics).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Tuple

from ..snapshot import RawWorldSnapshot, RawChunk

# (x, y, z) integer coordinates
Coord = Tuple[int, int, int]

# Signature for a block-solid callback:
#   is_solid(x, y, z, snapshot) -> bool
BlockSolidFn = Callable[[int, int, int, RawWorldSnapshot], bool]


@dataclass
class NavGrid:
    """
    Navigation grid built on top of a RawWorldSnapshot.

    Responsibilities:
    - Provide walkability tests (is_walkable).
    - Provide neighbor coordinates for pathfinding.

    It does NOT:
    - Interpret block types.
    - Perform semantics or tech reasoning.
    """

    snapshot: RawWorldSnapshot
    is_solid_block: BlockSolidFn

    # Optional constraints; these can be tuned later.
    max_fall_height: int = 4  # how far the bot is allowed to drop
    max_step_height: int = 1  # how high the bot can step up

    # ------------------------------------------------------------------
    # Core queries
    # ------------------------------------------------------------------

    def is_walkable(self, x: int, y: int, z: int) -> bool:
        """
        Determine if the bot can "stand" at (x, y, z).

        Simple rule:
        - Block at (x, y - 1, z) is solid (floor).
        - Blocks at (x, y, z) and (x, y + 1, z) are non-solid (no collision).
        """
        floor_y = y - 1

        if not self._is_coord_supported(x, floor_y, z):
            return False

        # Space must be free for body + head.
        if self._is_block_solid(x, y, z):
            return False
        if self._is_block_solid(x, y + 1, z):
            return False

        return True

    def neighbors_4dir(self, coord: Coord) -> list[Coord]:
        """
        Return potential 4-directional neighbors on the x-z plane,
        with basic vertical adjustment.

        We allow up/down steps within max_step_height and controlled falls.
        """
        x, y, z = coord
        candidates: list[Coord] = []

        # Offsets in x-z plane
        for dx, dz in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx = x + dx
            nz = z + dz

            # Try to find a suitable y at (nx, nz) near current y
            # First: small upward/downward steps
            for dy in range(-self.max_step_height, self.max_step_height + 1):
                ny = y + dy
                if self.is_walkable(nx, ny, nz):
                    candidates.append((nx, ny, nz))
                    break
            else:
                # If no small step works, see if we can safely fall
                fall_target = self._find_fall_target(nx, y, nz)
                if fall_target is not None:
                    candidates.append(fall_target)

        return candidates

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _is_block_solid(self, x: int, y: int, z: int) -> bool:
        """
        Delegate to is_solid_block callback.

        This is the only place RawWorldSnapshot → collision decision happens.
        """
        return bool(self.is_solid_block(x, y, z, self.snapshot))

    def _is_coord_supported(self, x: int, y: int, z: int) -> bool:
        """
        Check whether a block at (x, y, z) exists and is solid enough
        to act as a floor.
        """
        return self._is_block_solid(x, y, z)

    def _find_fall_target(self, x: int, start_y: int, z: int) -> Coord | None:
        """
        Try to find a valid landing spot when walking off an edge.

        We search downward from start_y to start_y - max_fall_height.
        """
        lowest_y = start_y - self.max_fall_height
        y = start_y

        # Already in air? Step down until floor or limit.
        while y >= lowest_y:
            # We want (x, y, z) to be walkable; if we find one, we accept it.
            if self.is_walkable(x, y, z):
                return (x, y, z)
            y -= 1

        return None

```
Note: by design, `is_solid_block` is injected. For now you can use a dumb placeholder; later M3 can provide a proper implementation using semantics.


```
# src/bot_core/nav/pathfinder.py
"""
A* pathfinding over NavGrid.

- Uses Manhattan distance heuristic.
- 4-directional neighbors (x-z plane) with limited vertical adjustment.
- max_steps guard to avoid infinite loops / huge searches.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .grid import NavGrid, Coord


@dataclass
class PathfindingResult:
    """Structured result for a pathfinding attempt."""

    path: List[Coord]
    success: bool
    reason: str | None = None


def _heuristic(a: Coord, b: Coord) -> float:
    """Manhattan distance heuristic for A*."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1]) + abs(a[2] - b[2])


def find_path(
    grid: NavGrid,
    start: Coord,
    goal: Coord,
    max_steps: int = 1024,
) -> PathfindingResult:
    """
    A* search for a path from start to goal on NavGrid.

    Returns a PathfindingResult with:
      - path: possibly empty list of coordinates including start and goal
      - success: bool
      - reason: if not success, a human-readable explanation

    This function does not send any packets or mutate world state.
    """
    if start == goal:
        return PathfindingResult(path=[start], success=True)

    open_heap: List[tuple[float, Coord]] = []
    heapq.heappush(open_heap, (0.0, start))

    came_from: Dict[Coord, Coord] = {}
    g_score: Dict[Coord, float] = {start: 0.0}

    steps_remaining = max_steps

    while open_heap and steps_remaining > 0:
        _, current = heapq.heappop(open_heap)
        steps_remaining -= 1

        if current == goal:
            # reconstruct path
            return PathfindingResult(
                path=_reconstruct_path(came_from, current),
                success=True,
            )

        for nxt in grid.neighbors_4dir(current):
            tentative_g = g_score[current] + 1.0

            if tentative_g < g_score.get(nxt, float("inf")):
                came_from[nxt] = current
                g_score[nxt] = tentative_g
                f_score = tentative_g + _heuristic(nxt, goal)
                heapq.heappush(open_heap, (f_score, nxt))

    # No path found or max_steps exhausted
    reason = (
        "max_steps_exhausted"
        if steps_remaining <= 0
        else "no_path_found"
    )
    return PathfindingResult(path=[], success=False, reason=reason)


def _reconstruct_path(
    came_from: Dict[Coord, Coord],
    current: Coord,
) -> List[Coord]:
    """Reconstruct full path from came_from map."""
    path: List[Coord] = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path

```


```
# src/bot_core/nav/mover.py
"""
Mover: convert paths into high-level move_to Actions.

M6 only owns:
- path → sequence of move_to Actions
- basic parameter shaping (radius, etc.)

It does NOT send packets or talk to the world; that's ActionExecutor's job.
"""

from __future__ import annotations

from typing import List

from spec.types import Action  # Action is defined in M1 (spec.types)
from ..snapshot import RawWorldSnapshot
from .pathfinder import PathfindingResult, Coord


def path_to_actions(
    path_result: PathfindingResult,
    snapshot: RawWorldSnapshot,
    *,
    radius: float = 0.5,
) -> List[Action]:
    """
    Convert a PathfindingResult into a list of move_to Actions.

    Each step becomes a single move_to with:
      - x, y, z: integer block coordinates
      - radius: how close is "good enough" for arrival

    Higher layers (skills, planner) can choose whether to:
      - compress steps into fewer waypoints
      - run only a prefix of the path
    """
    if not path_result.success or not path_result.path:
        return []

    actions: List[Action] = []
    for (x, y, z) in path_result.path:
        actions.append(
            Action(
                type="move_to",
                params={
                    "x": int(x),
                    "y": int(y),
                    "z": int(z),
                    "radius": float(radius),
                },
            )
        )

    return actions


def current_coord_from_snapshot(snapshot: RawWorldSnapshot) -> Coord:
    """
    Helper: derive integer Coord from the current player position.

    Floor the floating position to an int grid; this keeps NavGrid
    and real position in sync enough for path planning.
    """
    px = snapshot.player_pos.get("x", 0.0)
    py = snapshot.player_pos.get("y", 0.0)
    pz = snapshot.player_pos.get("z", 0.0)

    return int(px), int(py), int(pz)

```

### How this hangs together

- **NavGrid**
    
    - Built as `NavGrid(snapshot, is_solid_block=your_collision_fn)`
        
    - `neighbors_4dir` uses `is_walkable`, which calls your `is_solid_block`.
        
    - `is_solid_block` is where M3 will eventually plug in “this is stone / this is air / this is fluid.”
        
- **Pathfinding**
    
    - `find_path(grid, start, goal)` returns a `PathfindingResult` with `success` and `reason`.
        
    - This lets M6 report structured nav failures up to `ActionExecutor` → `ActionResult`.
        
- **Mover**
    
    - Converts paths into `move_to` `Action`s.
        
    - Higher layers (M8/M5) choose how many steps they actually use.
        

No semantics, no tech_state, no LLM. Just geometry and suffering.

---

## 7. Action Execution

### ActionExecutor
Implements:
- move_to → build NavGrid → find_path → send incremental movement packets
- break_block → send dig/start + dig/stop
- place_block → send block_place
- use_item / interact TBD (packet names vary across implementations)

ActionResult:
- success  
- error or None  
- details {…}  
- Never mutate internal world state; only network I/O.

Error modes:
- No path found  
- Unsupported action  
- IPC timeout  
- Packet error  

```
# src/bot_core/actions.py
"""
Action execution for bot_core_1_7_10.

This module translates high-level Actions into protocol- or IPC-level
messages via a PacketClient.

Design constraints:
- High-level, atomic-ish actions:
    - move_to
    - break_block
    - place_block
    - use_item
    - interact
- One execute_action(...) call per logical operation.
- Explicit, structured failures:
    - no path
    - unsupported action
    - IO/IPC error
- No mutation of world state; this layer only sends packets/messages.
- No semantics, no tech_state, no virtues.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping

from spec.types import Action, ActionResult  # from M1
from .snapshot import RawWorldSnapshot
from .net import PacketClient
from .nav import (
    NavGrid,
    BlockSolidFn,
    find_path,
    path_to_actions,
    current_coord_from_snapshot,
)


# ---------------------------------------------------------------------------
# Defaults / helpers
# ---------------------------------------------------------------------------


def _default_is_solid_block(
    x: int,
    y: int,
    z: int,
    snapshot: RawWorldSnapshot,
) -> bool:
    """
    Extremely conservative placeholder for block solidity.

    For now, pretend *nothing* is solid except the void boundary.

    This is intentionally dumb; real collision logic belongs to M3 and
    should be injected via a proper BlockSolidFn.
    """
    # You can choose to treat y < 0 as "solid floor" to avoid falling
    # into the void in tests, but by default we report no solid blocks
    # so that pathfinding is explicitly told "no path".
    return False


@dataclass
class ActionExecutorConfig:
    """
    Configuration knobs for ActionExecutor.

    These should be environment- and modpack-agnostic. GTNH-specific
    tuning belongs in higher layers or injected strategies.
    """

    # Limit on nav search complexity
    max_nav_steps: int = 2048

    # Default arrival radius for move_to
    default_move_radius: float = 0.5


class ActionExecutor:
    """
    Translate high-level Actions into PacketClient messages.

    Public contract:
      execute(action, snapshot) -> ActionResult

    Responsibilities:
    - Build NavGrid from RawWorldSnapshot.
    - Run A* pathfinding for move_to.
    - Emit movement / dig / place / use / interact packets.
    - Surface explicit, structured errors.

    Non-responsibilities:
    - WorldState construction.
    - Tech semantics.
    - Planning, skill selection, or virtue scoring.
    """

    def __init__(
        self,
        client: PacketClient,
        *,
        is_solid_block: BlockSolidFn | None = None,
        config: ActionExecutorConfig | None = None,
    ) -> None:
        self._client = client
        self._is_solid_block: BlockSolidFn = (
            is_solid_block if is_solid_block is not None else _default_is_solid_block
        )
        self._cfg = config if config is not None else ActionExecutorConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def execute(self, action: Action, snapshot: RawWorldSnapshot) -> ActionResult:
        """
        Execute a single high-level Action against the given snapshot.

        The snapshot is treated as read-only and must not be mutated here.
        """
        atype = getattr(action, "type", None)
        params = getattr(action, "params", {}) or {}

        if not isinstance(params, Mapping):
            return ActionResult(
                success=False,
                error="invalid_params",
                details={"reason": "params_not_mapping", "action_type": atype},
            )

        try:
            if atype == "move_to":
                return self._execute_move_to(params, snapshot)
            elif atype == "break_block":
                return self._execute_break_block(params, snapshot)
            elif atype == "place_block":
                return self._execute_place_block(params, snapshot)
            elif atype == "use_item":
                return self._execute_use_item(params, snapshot)
            elif atype == "interact":
                return self._execute_interact(params, snapshot)
            else:
                return ActionResult(
                    success=False,
                    error="unsupported_action",
                    details={"action_type": atype},
                )
        except Exception as exc:
            # Catch-all for transport / unexpected errors.
            return ActionResult(
                success=False,
                error="execution_exception",
                details={
                    "action_type": atype,
                    "exception": repr(exc),
                },
            )

    # ------------------------------------------------------------------
    # Action handlers
    # ------------------------------------------------------------------

    def _execute_move_to(
        self,
        params: Mapping[str, Any],
        snapshot: RawWorldSnapshot,
    ) -> ActionResult:
        """
        Move the player to a target coordinate using NavGrid + A*.

        Expected params:
            - "x", "y", "z": numeric target position (block coords)
            - "radius": optional float radius (default from config)

        Behavior:
            - Build NavGrid with injected is_solid_block.
            - Run A* from current_coord → target_coord.
            - If success: convert path to move_to Actions and emit
              movement packets for each step.
            - If failure: return ActionResult with reason.
        """
        try:
            tx = int(params["x"])
            ty = int(params.get("y", snapshot.player_pos.get("y", 0.0)))
            tz = int(params["z"])
        except Exception:
            return ActionResult(
                success=False,
                error="invalid_params",
                details={"reason": "missing_or_non_numeric_xyz", "params": dict(params)},
            )

        radius = float(params.get("radius", self._cfg.default_move_radius))

        # Build NavGrid from snapshot
        grid = NavGrid(snapshot=snapshot, is_solid_block=self._is_solid_block)

        # Derive start/goal coords
        start = current_coord_from_snapshot(snapshot)
        goal = (tx, ty, tz)

        # Run pathfinding
        pf_result = find_path(
            grid,
            start=start,
            goal=goal,
            max_steps=self._cfg.max_nav_steps,
        )

        if not pf_result.success or not pf_result.path:
            return ActionResult(
                success=False,
                error="nav_failure",
                details={
                    "from": start,
                    "to": goal,
                    "reason": pf_result.reason,
                },
            )

        # Convert path to move_to actions
        move_actions = path_to_actions(
            pf_result,
            snapshot,
            radius=radius,
        )

        # Emit movement packets for each step.
        # The PacketClient / Forge mod decides how to interpret this.
        steps_sent = 0
        for step in move_actions:
            try:
                self._send_move_step(step.params)
                steps_sent += 1
            except Exception as exc:
                # On IO error, stop and report partial success.
                return ActionResult(
                    success=False,
                    error="io_error",
                    details={
                        "stage": "move_to",
                        "from": start,
                        "to": goal,
                        "steps_attempted": steps_sent,
                        "exception": repr(exc),
                    },
                )

        return ActionResult(
            success=True,
            error=None,
            details={
                "from": start,
                "to": goal,
                "steps": steps_sent,
            },
        )

    def _execute_break_block(
        self,
        params: Mapping[str, Any],
        snapshot: RawWorldSnapshot,  # unused but kept for future context
    ) -> ActionResult:
        """
        Break a block at the given coordinates.

        Expected params:
            - "x", "y", "z": block coordinates
            - "face": optional face index or direction (depends on IPC schema)

        Behavior:
            - Send "block_dig" start + stop messages.
        """
        try:
            x = int(params["x"])
            y = int(params["y"])
            z = int(params["z"])
        except Exception:
            return ActionResult(
                success=False,
                error="invalid_params",
                details={"reason": "missing_or_non_numeric_xyz", "params": dict(params)},
            )

        face = params.get("face", "auto")

        try:
            # Start digging
            self._client.send_packet(
                "block_dig",
                {
                    "status": "start",
                    "x": x,
                    "y": y,
                    "z": z,
                    "face": face,
                },
            )
            # Stop / finish digging
            self._client.send_packet(
                "block_dig",
                {
                    "status": "stop",
                    "x": x,
                    "y": y,
                    "z": z,
                    "face": face,
                },
            )
        except Exception as exc:
            return ActionResult(
                success=False,
                error="io_error",
                details={
                    "stage": "break_block",
                    "x": x,
                    "y": y,
                    "z": z,
                    "exception": repr(exc),
                },
            )

        return ActionResult(
            success=True,
            error=None,
            details={"x": x, "y": y, "z": z},
        )

    def _execute_place_block(
        self,
        params: Mapping[str, Any],
        snapshot: RawWorldSnapshot,  # unused but reserved for future checks
    ) -> ActionResult:
        """
        Place a block using the currently held item at the target coordinates.

        Expected params:
            - "x", "y", "z": target block position or adjacent position
            - "face": optional face direction
        """
        try:
            x = int(params["x"])
            y = int(params["y"])
            z = int(params["z"])
        except Exception:
            return ActionResult(
                success=False,
                error="invalid_params",
                details={"reason": "missing_or_non_numeric_xyz", "params": dict(params)},
            )

        face = params.get("face", "auto")

        try:
            self._client.send_packet(
                "block_place",
                {
                    "x": x,
                    "y": y,
                    "z": z,
                    "face": face,
                },
            )
        except Exception as exc:
            return ActionResult(
                success=False,
                error="io_error",
                details={
                    "stage": "place_block",
                    "x": x,
                    "y": y,
                    "z": z,
                    "exception": repr(exc),
                },
            )

        return ActionResult(
            success=True,
            error=None,
            details={"x": x, "y": y, "z": z},
        )

    def _execute_use_item(
        self,
        params: Mapping[str, Any],
        snapshot: RawWorldSnapshot,  # reserved for later (line of sight, etc.)
    ) -> ActionResult:
        """
        Use the currently held item (e.g., right-click in air or on a block).

        Expected params (schema is flexible and IPC-defined):
            - "target": one of {"self", "block", "entity", "air"} (optional)
            - "x", "y", "z": optional coordinates (for block/entity)
            - any other keys are passed through to IPC layer.
        """
        payload: Dict[str, Any] = {"target": params.get("target", "air")}
        # Pass through positional info if present.
        for key in ("x", "y", "z", "face", "entity_id"):
            if key in params:
                payload[key] = params[key]

        # Forward any extra params untouched.
        for k, v in params.items():
            if k not in payload:
                payload[k] = v

        try:
            self._client.send_packet("use_item", payload)
        except Exception as exc:
            return ActionResult(
                success=False,
                error="io_error",
                details={
                    "stage": "use_item",
                    "payload": payload,
                    "exception": repr(exc),
                },
            )

        return ActionResult(
            success=True,
            error=None,
            details=payload,
        )

    def _execute_interact(
        self,
        params: Mapping[str, Any],
        snapshot: RawWorldSnapshot,  # reserved (e.g., for range checks)
    ) -> ActionResult:
        """
        Interact with an entity or block.

        Expected params:
            - "kind": "entity" or "block"
            - "entity_id": int if kind == "entity"
            - "x", "y", "z": coords if kind == "block"
        """
        kind = params.get("kind")

        if kind not in ("entity", "block"):
            return ActionResult(
                success=False,
                error="invalid_params",
                details={
                    "reason": "kind_must_be_entity_or_block",
                    "params": dict(params),
                },
            )

        payload: Dict[str, Any] = {"kind": kind}

        if kind == "entity":
            try:
                payload["entity_id"] = int(params["entity_id"])
            except Exception:
                return ActionResult(
                    success=False,
                    error="invalid_params",
                    details={
                        "reason": "missing_or_non_numeric_entity_id",
                        "params": dict(params),
                    },
                )
        else:  # kind == "block"
            try:
                payload["x"] = int(params["x"])
                payload["y"] = int(params["y"])
                payload["z"] = int(params["z"])
            except Exception:
                return ActionResult(
                    success=False,
                    error="invalid_params",
                    details={
                        "reason": "missing_or_non_numeric_xyz",
                        "params": dict(params),
                    },
                )

        # Pass through any extra parameters
        for k, v in params.items():
            if k not in payload:
                payload[k] = v

        try:
            self._client.send_packet("interact", payload)
        except Exception as exc:
            return ActionResult(
                success=False,
                error="io_error",
                details={
                    "stage": "interact",
                    "payload": payload,
                    "exception": repr(exc),
                },
            )

        return ActionResult(
            success=True,
            error=None,
            details=payload,
        )

    # ------------------------------------------------------------------
    # Low-level emitters
    # ------------------------------------------------------------------

    def _send_move_step(self, step_params: Mapping[str, Any]) -> None:
        """
        Emit a single movement step.

        For IPC mode, you can interpret this generically:
          - type: "move_step"
          - payload: {x, y, z, radius}

        Forge mod / external client can translate that into concrete
        packets (position updates, relative moves, etc.).
        """
        x = step_params.get("x")
        y = step_params.get("y")
        z = step_params.get("z")
        radius = step_params.get("radius")

        payload = {
            "x": x,
            "y": y,
            "z": z,
            "radius": radius,
        }

        # Use a generic high-level move_step packet type; the concrete
        # implementation (IPC or protocol) decides how to realize it.
        self._client.send_packet("move_step", payload)

```

This ticks all your boxes:

- **High-level actions** only
    
- Uses **NavGrid + A*** for `move_to`
    
- Explicit **error codes** (`nav_failure`, `unsupported_action`, `io_error`, `invalid_params`, `execution_exception`)
    
- Never mutates the world, just talks to `PacketClient`
    
- Keeps semantics free so M3 can do all the GTNH wizardry later
---

## 8. BotCoreImpl

Fully unifies:
- PacketClient / IPC
- WorldTracker
- ActionExecutor

Exposes:
- connect()
- disconnect()
- tick()
- observe() → RawWorldSnapshot
- get_world_state() → WorldState
- execute_action(Action) → ActionResult

Must not leak:
- packets  
- raw snapshot internals  
- nav exceptions  

All failures surface via ActionResult or raised domain errors.

```
# src/bot_core/core.py
"""
Concrete BotCore implementation for Minecraft 1.7.10 (GTNH).

This module wires together:
- PacketClient / IPC transport (M6 network layer)
- WorldTracker (incremental raw world state)
- ActionExecutor (high-level action execution)
- Snapshot adapter (RawWorldSnapshot -> WorldState)

Public surface (for higher layers like M8):
    class BotCoreImpl(BotCore):
        connect() -> None
        disconnect() -> None
        tick() -> None
        observe() -> RawWorldSnapshot
        get_world_state() -> WorldState
        execute_action(Action) -> ActionResult

Design constraints:
- No packet or protocol details leak to callers.
- No navigation internals or exceptions leak to callers.
- All action-related failures return ActionResult with explicit error codes.
- Non-action failures (e.g., connection problems) raise domain errors.
- No semantics, virtues, or tech-state logic here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from env.loader import load_environment  # Phase 0: EnvProfile loader
from spec.types import Action, ActionResult, WorldState  # M1 shared types

from spec.bot_core import BotCore  # BotCore interface definition (M1)
from .net import PacketClient, create_packet_client_for_env
from .world_tracker import WorldTracker
from .snapshot import RawWorldSnapshot, snapshot_to_world_state
from .actions import ActionExecutor, ActionExecutorConfig


# ---------------------------------------------------------------------------
# Domain errors
# ---------------------------------------------------------------------------


@dataclass
class BotCoreError(RuntimeError):
    """
    Domain-level error raised by BotCoreImpl for non-action failures.

    Examples:
        - failed to connect or disconnect cleanly
        - tick loop I/O failures
        - configuration errors

    Action execution errors should generally NOT raise this; instead,
    ActionExecutor must return an ActionResult with error set.
    """

    code: str
    details: dict[str, Any]

    def __str__(self) -> str:
        return f"BotCoreError(code={self.code!r}, details={self.details!r})"


# ---------------------------------------------------------------------------
# Implementation
# ---------------------------------------------------------------------------


class BotCoreImpl(BotCore):
    """
    Concrete BotCore for MC 1.7.10 GTNH.

    Orchestrates:
        - PacketClient / IPC (transport)
        - WorldTracker (RawWorldSnapshot)
        - ActionExecutor (high-level actions)

    Consumers (M8, etc.) see:
        - connect / disconnect / tick
        - observe -> RawWorldSnapshot
        - get_world_state -> WorldState
        - execute_action(Action) -> ActionResult
    """

    def __init__(
        self,
        client: Optional[PacketClient] = None,
        *,
        action_config: Optional[ActionExecutorConfig] = None,
        # Placeholder for future injection of collision semantics
        is_solid_block=None,
    ) -> None:
        """
        Build a BotCoreImpl.

        If `client` is None, it is constructed from the Phase 0 environment
        via create_packet_client_for_env(), which uses env.yaml / EnvProfile.

        Parameters:
            client: Optional PacketClient (useful for tests)
            action_config: optional ActionExecutorConfig
            is_solid_block: optional BlockSolidFn to plug in collision logic
                            (M3 semantics can inject this later)
        """
        # EnvProfile mainly used for logging/context right now; creation of
        # the client is delegated to create_packet_client_for_env().
        self._env = load_environment()

        # Transport layer
        self._client: PacketClient = client or create_packet_client_for_env()

        # World tracking
        self._tracker = WorldTracker(self._client)

        # Action execution
        self._executor = ActionExecutor(
            self._client,
            is_solid_block=is_solid_block,
            config=action_config,
        )

        self._connected: bool = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """
        Establish connection to the Minecraft world.

        Raises:
            BotCoreError if the underlying client fails to connect.
        """
        if self._connected:
            return

        try:
            self._client.connect()
        except Exception as exc:
            raise BotCoreError(
                code="connect_failed",
                details={"exception": repr(exc)},
            ) from exc

        self._connected = True

        # Seed tracker context with basic env profile metadata (optional).
        self._tracker.set_context(
            "env_profile_name",
            getattr(self._env, "name", "default"),
        )
        self._tracker.set_context("bot_mode", getattr(self._env, "bot_mode", "unknown"))

    def disconnect(self) -> None:
        """
        Disconnect from the Minecraft world.

        Raises:
            BotCoreError if the underlying client fails to disconnect.
        """
        if not self._connected:
            return

        try:
            self._client.disconnect()
        except Exception as exc:
            # We still consider ourselves disconnected, but bubble the error.
            self._connected = False
            raise BotCoreError(
                code="disconnect_failed",
                details={"exception": repr(exc)},
            ) from exc

        self._connected = False

    def tick(self) -> None:
        """
        Pump the transport layer and feed events into the WorldTracker.

        This must be called regularly (e.g., once per iteration of the main
        agent loop). It does NOT itself execute any actions.
        """
        if not self._connected:
            # It's acceptable to call tick() before connect, but it's a no-op.
            return

        try:
            self._client.tick()
        except Exception as exc:
            # Transport-level issues are domain errors, not ActionResult issues.
            raise BotCoreError(
                code="tick_failed",
                details={"exception": repr(exc)},
            ) from exc

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def observe(self) -> RawWorldSnapshot:
        """
        Return a raw snapshot of the current world state.

        This is intended for internal consumers (e.g., debugging, advanced
        semantics modules). Most of the time higher layers should use
        get_world_state() instead.

        Returns:
            RawWorldSnapshot constructed by WorldTracker.
        """
        # WorldTracker.build_snapshot() already returns a fresh dataclass
        # with shallow copies of its internal structures, so we can forward
        # it directly without exposing mutable internals.
        return self._tracker.build_snapshot()

    def get_world_state(self) -> WorldState:
        """
        Return a semantic WorldState for agent consumption.

        This is the primary observation API for the planner and skills.

        Returns:
            WorldState created via snapshot_to_world_state(observe()).
        """
        raw = self._tracker.build_snapshot()
        # snapshot_to_world_state is the only place in M6 that constructs
        # WorldState; BotCoreImpl simply delegates to it.
        return snapshot_to_world_state(raw)

    # ------------------------------------------------------------------
    # Action execution
    # ------------------------------------------------------------------

    def execute_action(self, action: Action) -> ActionResult:
        """
        Execute a single high-level Action.

        All navigation / pathfinding / I/O details are internal. Nav and
        transport exceptions are caught and surfaced as ActionResult errors.

        Returns:
            ActionResult with:
                - success: bool
                - error: str | None
                - details: dict[str, Any]
        """
        # Capture a snapshot at the moment we decide to act. We do NOT rely
        # on the caller to provide this; BotCore owns timing.
        raw = self._tracker.build_snapshot()

        try:
            result = self._executor.execute(action, raw)
        except Exception as exc:
            # As a last resort, ensure no low-level exception leaks upstream.
            return ActionResult(
                success=False,
                error="botcore_execute_exception",
                details={
                    "exception": repr(exc),
                    "action_type": getattr(action, "type", None),
                },
            )

        # Ensure we always return an ActionResult instance.
        if not isinstance(result, ActionResult):
            return ActionResult(
                success=False,
                error="invalid_executor_result",
                details={"result_repr": repr(result)},
            )

        return result

```


```
# src/bot_core/__init__.py
"""
bot_core_1_7_10 package.

Exports:
    - BotCoreImpl: main body/controller implementation for MC 1.7.10 GTNH
    - BotCoreError: domain-level error type for non-action failures
"""

from __future__ import annotations

from .core import BotCoreImpl, BotCoreError

__all__ = [
    "BotCoreImpl",
    "BotCoreError",
]

```

This gives you a clean, testable, non-leaky body:

- M8+ talks to **BotCoreImpl** and **never** sees packets, nav internals, or IPC weirdness.
    
- All transport / tick failures raise `BotCoreError`.
    
- All action-level failures come back as `ActionResult` with structured details.
    
- M3, M5, M8, M10 can sit on top of this without having to rip it out later.

---

## 9. Testing Strategy

### Unit Tests
- Fake PacketClient to record send_packet calls
- Test all tracker packet handlers
- Pathfinding around obstacles with synthetic chunks
- ActionExecutor move_to → ensures path→packets
- break_block/ place_block send correct packets

### Integration Tests
- BotCoreImpl(fake) end-to-end:  
  world_tracker.build_snapshot → execute_action → packet log

### Future Live Tests
- Connect to real GTNH environment  
- smoke: move+look  
- inventory update  
- break block near player  

```
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

```


```
# tests/test_world_tracker.py
"""
Unit tests for WorldTracker.

Covers:
- time_update
- position_update
- chunk_data
- spawn_entity / destroy_entities
- set_slot / window_items
"""

from __future__ import annotations

from typing import Any, Dict

from bot_core.world_tracker import WorldTracker
from bot_core.snapshot import RawWorldSnapshot
from bot_core.testing.fakes import FakePacketClient


def test_time_update_updates_tick() -> None:
    client = FakePacketClient()
    tracker = WorldTracker(client)

    client.emit("time_update", {"tick": 42})
    snap = tracker.build_snapshot()

    assert isinstance(snap, RawWorldSnapshot)
    assert snap.tick == 42


def test_position_update_updates_player_state() -> None:
    client = FakePacketClient()
    tracker = WorldTracker(client)

    client.emit(
        "position_update",
        {
            "x": 10.5,
            "y": 65.0,
            "z": -3.25,
            "yaw": 90.0,
            "pitch": 10.0,
            "on_ground": False,
        },
    )

    snap = tracker.build_snapshot()
    assert snap.player_pos["x"] == 10.5
    assert snap.player_pos["y"] == 65.0
    assert snap.player_pos["z"] == -3.25
    assert snap.player_yaw == 90.0
    assert snap.player_pitch == 10.0
    assert snap.on_ground is False


def test_chunk_data_stores_chunk() -> None:
    client = FakePacketClient()
    tracker = WorldTracker(client)

    client.emit(
        "chunk_data",
        {
            "chunk_x": 1,
            "chunk_z": -2,
            "blocks": "opaque_blocks_representation",
            "biomes": "biome_data_stub",
        },
    )

    snap = tracker.build_snapshot()
    assert (1, -2) in snap.chunks
    chunk = snap.chunks[(1, -2)]
    assert chunk.blocks == "opaque_blocks_representation"
    assert chunk.biome_data == "biome_data_stub"


def test_spawn_and_destroy_entities() -> None:
    client = FakePacketClient()
    tracker = WorldTracker(client)

    client.emit(
        "spawn_entity",
        {
            "entity_id": 123,
            "kind": "mob",
            "x": 1.0,
            "y": 64.0,
            "z": 2.0,
            "extra": "data",
        },
    )

    snap = tracker.build_snapshot()
    assert any(e.entity_id == 123 for e in snap.entities)

    client.emit("destroy_entities", {"entity_ids": [123]})
    snap2 = tracker.build_snapshot()
    assert all(e.entity_id != 123 for e in snap2.entities)


def test_inventory_set_slot_and_window_items() -> None:
    client = FakePacketClient()
    tracker = WorldTracker(client)

    # full inventory snapshot
    client.emit(
        "window_items",
        {
            "items": [
                {"id": "minecraft:stone", "count": 1},
                {"id": "minecraft:dirt", "count": 2},
            ]
        },
    )

    snap = tracker.build_snapshot()
    assert len(snap.inventory) == 2
    assert snap.inventory[0]["id"] == "minecraft:stone"

    # single slot update
    client.emit(
        "set_slot",
        {
            "slot": 1,
            "item": {"id": "minecraft:planks", "count": 4},
        },
    )

    snap2 = tracker.build_snapshot()
    assert snap2.inventory[1]["id"] == "minecraft:planks"
    assert snap2.inventory[1]["count"] == 4

```



```
# tests/test_nav_pathfinder.py
"""
Unit tests for NavGrid + A* pathfinder.

We build synthetic "worlds" by faking is_solid_block behavior,
so no real chunk data is required.
"""

from __future__ import annotations

from typing import Any

from bot_core.snapshot import RawWorldSnapshot, RawChunk, RawEntity
from bot_core.nav import NavGrid, find_path, Coord


def make_empty_snapshot() -> RawWorldSnapshot:
    return RawWorldSnapshot(
        tick=0,
        dimension="overworld",
        player_pos={"x": 0.0, "y": 64.0, "z": 0.0},
        player_yaw=0.0,
        player_pitch=0.0,
        on_ground=True,
        chunks={},
        entities=[],
        inventory=[],
        context={},
    )


def flat_floor_is_solid(x: int, y: int, z: int, snapshot: RawWorldSnapshot) -> bool:
    """
    Simple solidity function:
    - floor at y=63 is solid
    - everything else is non-solid
    """
    return y == 63


def wall_is_solid(x: int, y: int, z: int, snapshot: RawWorldSnapshot) -> bool:
    """
    Like flat_floor_is_solid, but with a wall at x == 2, for z in [0..4].
    """
    if y == 63:
        if x == 2 and 0 <= z <= 4:
            return True  # wall on the floor
        return True
    return False


def test_find_path_in_open_space() -> None:
    snap = make_empty_snapshot()
    grid = NavGrid(snapshot=snap, is_solid_block=flat_floor_is_solid)

    start: Coord = (0, 64, 0)
    goal: Coord = (4, 64, 0)

    result = find_path(grid, start=start, goal=goal, max_steps=100)

    assert result.success
    assert result.path[0] == start
    assert result.path[-1] == goal
    # path length should be >= 5 (0..4 inclusive)
    assert len(result.path) >= 5


def test_find_path_around_wall() -> None:
    snap = make_empty_snapshot()
    grid = NavGrid(snapshot=snap, is_solid_block=wall_is_solid)

    start: Coord = (0, 64, 0)
    goal: Coord = (4, 64, 0)

    result = find_path(grid, start=start, goal=goal, max_steps=200)

    assert result.success
    assert result.path[0] == start
    assert result.path[-1] == goal

    # Ensure at least one step avoids x == 2, z in [0..4] at y=64
    for (x, y, z) in result.path:
        # we only care about positions directly over the floor (y=64)
        if y == 64 and x == 2 and 0 <= z <= 4:
            raise AssertionError("Path incorrectly goes through the wall")


def test_find_path_max_steps_exhaustion() -> None:
    snap = make_empty_snapshot()
    # treat EVERYTHING as solid so no path is possible
    grid = NavGrid(
        snapshot=snap,
        is_solid_block=lambda x, y, z, s: True,  # type: ignore[arg-type]
    )

    start: Coord = (0, 64, 0)
    goal: Coord = (10, 64, 10)

    result = find_path(grid, start=start, goal=goal, max_steps=10)

    assert not result.success
    assert result.reason in ("no_path_found", "max_steps_exhausted")
    assert result.path == []

```



```
# tests/test_actions.py
"""
Unit tests for ActionExecutor.

Focus:
- move_to calls pathfinder and emits move_step packets.
- break_block emits block_dig start/stop.
- place_block emits block_place.
"""

from __future__ import annotations

from typing import Any, Dict

from spec.types import Action  # M1 type
from bot_core.snapshot import RawWorldSnapshot
from bot_core.actions import ActionExecutor, ActionExecutorConfig
from bot_core.testing.fakes import FakePacketClient
from bot_core.nav import NavGrid, BlockSolidFn


def make_flat_snapshot() -> RawWorldSnapshot:
    return RawWorldSnapshot(
        tick=0,
        dimension="overworld",
        player_pos={"x": 0.0, "y": 64.0, "z": 0.0},
        player_yaw=0.0,
        player_pitch=0.0,
        on_ground=True,
        chunks={},
        entities=[],
        inventory=[],
        context={},
    )


def floor_solid(x: int, y: int, z: int, snapshot: RawWorldSnapshot) -> bool:
    return y == 63


def test_move_to_emits_move_step_packets() -> None:
    client = FakePacketClient()
    executor = ActionExecutor(
        client,
        is_solid_block=floor_solid,
        config=ActionExecutorConfig(max_nav_steps=256, default_move_radius=0.5),
    )

    snapshot = make_flat_snapshot()

    action = Action(
        type="move_to",
        params={"x": 3, "y": 64, "z": 0, "radius": 0.5},
    )

    result = executor.execute(action, snapshot)

    assert result.success
    assert result.error is None
    assert result.details["steps"] > 0

    move_packets = [
        p for p in client.sent_packets if p.packet_type == "move_step"
    ]
    assert len(move_packets) == result.details["steps"]
    # Last step should be near the target
    last = move_packets[-1].data
    assert last["x"] == 3
    assert last["z"] == 0


def test_move_to_nav_failure_returns_error() -> None:
    client = FakePacketClient()
    # everything non-solid -> no path
    executor = ActionExecutor(
        client,
        is_solid_block=lambda x, y, z, s: False,  # type: ignore[arg-type]
        config=ActionExecutorConfig(max_nav_steps=64),
    )

    snapshot = make_flat_snapshot()

    action = Action(
        type="move_to",
        params={"x": 5, "y": 64, "z": 5},
    )

    result = executor.execute(action, snapshot)

    assert not result.success
    assert result.error == "nav_failure"
    assert "reason" in result.details


def test_break_block_sends_block_dig_packets() -> None:
    client = FakePacketClient()
    executor = ActionExecutor(client)

    snapshot = make_flat_snapshot()
    action = Action(
        type="break_block",
        params={"x": 1, "y": 64, "z": 2},
    )

    result = executor.execute(action, snapshot)

    assert result.success
    types = [p.packet_type for p in client.sent_packets]
    assert types.count("block_dig") == 2

    payloads = [p.data for p in client.sent_packets]
    statuses = {p["status"] for p in payloads}
    assert {"start", "stop"} <= statuses


def test_place_block_sends_block_place_packet() -> None:
    client = FakePacketClient()
    executor = ActionExecutor(client)

    snapshot = make_flat_snapshot()
    action = Action(
        type="place_block",
        params={"x": 2, "y": 64, "z": 2},
    )

    result = executor.execute(action, snapshot)

    assert result.success
    types = [p.packet_type for p in client.sent_packets]
    assert "block_place" in types

```


```
# tests/test_bot_core_impl.py
"""
Integration tests for BotCoreImpl using FakePacketClient.

Covers:
- Wiring between BotCoreImpl, WorldTracker, ActionExecutor.
- observe() and get_world_state() path.
- execute_action() sending packets via FakePacketClient.
"""

from __future__ import annotations

from typing import Any

import types

from spec.types import Action, WorldState  # M1 types
from bot_core.core import BotCoreImpl
from bot_core.testing.fakes import FakePacketClient
from bot_core.snapshot import RawWorldSnapshot
from bot_core.actions import ActionExecutorConfig


class DummyEnv:
    """Minimal stand-in for EnvProfile for tests."""

    def __init__(self) -> None:
        self.name = "test_env"
        self.bot_mode = "forge_mod"


def _patch_load_environment(monkeypatch: Any) -> None:
    import env.loader as loader

    monkeypatch.setattr(loader, "load_environment", lambda: DummyEnv())


def test_botcore_observe_and_world_state(monkeypatch: Any) -> None:
    _patch_load_environment(monkeypatch)

    client = FakePacketClient()
    bot = BotCoreImpl(client=client)

    bot.connect()
    client.emit(
        "position_update",
        {"x": 1.0, "y": 64.0, "z": 2.0, "yaw": 45.0, "pitch": 0.0},
    )
    client.emit("time_update", {"tick": 99})

    bot.tick()  # no-op for FakePacketClient but keeps the API consistent

    raw = bot.observe()
    assert isinstance(raw, RawWorldSnapshot)
    assert raw.tick == 99
    assert raw.player_pos["x"] == 1.0

    world = bot.get_world_state()
    assert isinstance(world, WorldState)
    assert world.tick == 99
    assert world.position["x"] == 1.0


def test_botcore_execute_action_routes_to_executor(monkeypatch: Any) -> None:
    _patch_load_environment(monkeypatch)

    client = FakePacketClient()
    bot = BotCoreImpl(client=client, action_config=ActionExecutorConfig())

    bot.connect()

    # Set initial position in tracker via FakePacketClient
    client.emit(
        "position_update",
        {"x": 0.0, "y": 64.0, "z": 0.0},
    )

    action = Action(
        type="break_block",
        params={"x": 1, "y": 64, "z": 2},
    )

    result = bot.execute_action(action)

    assert result.success
    types = [p.packet_type for p in client.sent_packets]
    assert "block_dig" in types

```


## 6. Live tests (manual, future)

You don’t need code for this yet, but the plan you wrote is still the right one:

1. Start GTNH with your Forge IPC mod.
    
2. Configure `env.yaml` with:
    
    - `bot_mode: forge_mod`
        
    - correct IPC host/port
        
3. Run a tiny harness like:
python:
```
from bot_core import BotCoreImpl
from spec.types import Action

bot = BotCoreImpl()
bot.connect()

while True:
    bot.tick()
    ws = bot.get_world_state()
    # print basic info

    # try a one-off move or break_block
    result = bot.execute_action(Action(type="move_to", params={"x": 3, "y": 64, "z": 3}))
    print(result)
    break

```


---

## 10. Completion Criteria

M6 is complete when:
- BotCoreImpl implements the full interface
- Snapshot→WorldState conversion stable
- Navigation works in offline tests
- Packets generated correctly
- Configurable via env.yaml bot_mode
- Fully isolated from planning/virtues/skills
- All tests pass

This establishes the physical agent used by M7–M11.



## Module Complete Notes

## 1. IPC / Forge mod contract (the big missing piece)

Right now:

- Python assumes normalized events:
    
    - `time_update`, `position_update`, `chunk_data`, `spawn_entity`, `destroy_entities`, `set_slot`, `window_items`, `dimension_change`
        
- And action packet types:
    
    - `move_step`, `block_dig`, `block_place`, `use_item`, `interact`
        

What’s missing is the **actual spec** for the Forge mod:

- Define a small JSON schema:
    
    - Incoming → `{ "type": "chunk_data", "payload": { ... } }`
        
    - Outgoing → `{ "type": "move_step", "payload": { "x":..., "y":..., "z":..., "radius":... } }`
        
- Write this down in a doc:
    
    - `docs/ipc_protocol_m6.md` or similar
        
- Make sure:
    
    - Dimension IDs, entity IDs, inventory windows, etc. are all nailed down.
        

Until that exists, bot_core is a very well-shaped brainstem with no spinal cord.

## 2. `is_solid_block` is still dumb

Right now the default `is_solid_block` is:
python:
```
return False

```

So:

- Nav works only in tests where you pass a fake solidity function.
    
- Real-world pathfinding will return `nav_failure` for basically everything.
    

What’s missing:

- A **real collision callback** that can:
    
    - Look at `snapshot.chunks`
        
    - Decode block IDs / meta → solid vs non-solid
        
- That logic properly belongs in **M3**, but you need:
    
    - A temporary implementation now
        
    - A clean hook later
        

Minimal interim hack:

- Add a `BlockCollisionProfile` that:
    
    - Treats “air” as non-solid
        
    - Treats everything else as solid
        
- Wire it into `ActionExecutor` via `is_solid_block` at construction time.

```
# src/bot_core/collision.py
"""
Block collision helpers for bot_core_1_7_10.

This is a minimal interim layer that defines how the navigation system
decides whether a block is "solid" for movement purposes.

Long-term:
    - A richer implementation in M3 should provide block IDs / materials
      from RawWorldSnapshot.chunks and drive a more accurate collision
      model.

Short-term (minimal hack):
    - Treat "air-like" blocks as non-solid.
    - When we don't know the actual blocks, optionally fall back to a
      simple floor model (e.g., fixed Y-level).

This module is intentionally lightweight and does NOT:
    - Interpret GTNH tech semantics
    - Reason about hazards (lava, fire, etc.)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

from .snapshot import RawWorldSnapshot

# Signature for a "block lookup" function, if/when you implement one:
#   block_at(snapshot, x, y, z) -> Any
BlockAtFn = Callable[[RawWorldSnapshot, int, int, int], Any]


def _is_air_like(block: Any) -> bool:
    """
    Heuristic to decide if a block is "air-like".

    This is deliberately forgiving, since we don't yet know the real
    chunk/ID layout. It handles a few common patterns:

        - None or {} or []  → air
        - numeric 0         → air (old-school ID)
        - mapping with id in {"minecraft:air", "air"} → air

    Anything else is treated as non-air (i.e., potentially solid).
    """
    if block is None:
        return True

    # Empty containers are treated as "no block"
    if block == {} or block == []:
        return True

    # Classic numeric ID for air
    if isinstance(block, (int, float)) and block == 0:
        return True

    # Dict-like with an ID field
    if isinstance(block, dict):
        bid = block.get("id") or block.get("name")
        if isinstance(bid, str) and bid.lower() in ("minecraft:air", "air"):
            return True

    return False


@dataclass
class BlockCollisionProfile:
    """
    Encapsulates a collision policy for navigation.

    Parameters:
        block_at:
            Optional function for retrieving block data from a snapshot.

        default_floor_y:
            Optional Y-level that is treated as "solid floor everywhere"
            when block_at is not available. This is a crude fallback used
            for initial smoke tests and simple worlds.

    The main entrypoint is `is_solid_block(x, y, z, snapshot)`.
    """

    block_at: Optional[BlockAtFn] = None
    default_floor_y: Optional[int] = None

    def is_solid_block(
        self,
        x: int,
        y: int,
        z: int,
        snapshot: RawWorldSnapshot,
    ) -> bool:
        """
        Decide if the block at (x, y, z) should be treated as solid.

        Behavior:
            - If block_at is provided:
                * call it and treat any non-air-like value as solid.
            - Else if default_floor_y is set:
                * y <= default_floor_y → solid
                * y >  default_floor_y → non-solid
            - Else:
                * everything is non-solid (nav will generally fail).
        """
        if self.block_at is not None:
            block = self.block_at(snapshot, x, y, z)
            return not _is_air_like(block)

        if self.default_floor_y is not None:
            # Very crude: treat everything at or below this Y as solid.
            return y <= self.default_floor_y

        # No information: treat everything as non-solid.
        return False


# ---------------------------------------------------------------------------
# Default profile & convenience entry point
# ---------------------------------------------------------------------------

# For now, assume a flat solid floor at y=63 (common "ground" level in tests).
_DEFAULT_PROFILE = BlockCollisionProfile(
    block_at=None,
    default_floor_y=63,
)


def default_block_collision_profile() -> BlockCollisionProfile:
    """
    Return the default BlockCollisionProfile used by BotCoreImpl.

    This can later be replaced or decorated by M3 when a richer
    block-at lookup is available.
    """
    return _DEFAULT_PROFILE


def default_is_solid_block(
    x: int,
    y: int,
    z: int,
    snapshot: RawWorldSnapshot,
) -> bool:
    """
    Convenience function for use as a BlockSolidFn in nav/grid.

    Uses the module-level default profile.
    """
    return _DEFAULT_PROFILE.is_solid_block(x, y, z, snapshot)

```
This gives you a real `is_solid_block` that:

- Works in tests
    
- Works in a very basic flat world
    
- Can later be overridden by M3 without touching the rest of M6
Then when M3 lands, you swap that in for something actually GTNH-aware.

## 2. Updated `src/bot_core/actions.py`

This version drops the old `_default_is_solid_block` and uses `default_is_solid_block` from the new collision module. It’s the whole file, ready to overwrite.
```
# src/bot_core/actions.py
"""
Action execution for bot_core_1_7_10.

This module translates high-level Actions into protocol- or IPC-level
messages via a PacketClient.

Design constraints:
- High-level, atomic-ish actions:
    - move_to
    - break_block
    - place_block
    - use_item
    - interact
- One execute_action(...) call per logical operation.
- Explicit, structured failures:
    - no path
    - unsupported action
    - IO/IPC error
- No mutation of world state; this layer only sends packets/messages.
- No semantics, no tech_state, no virtues.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping

from spec.types import Action, ActionResult  # from M1
from .snapshot import RawWorldSnapshot
from .net import PacketClient
from .nav import (
    NavGrid,
    BlockSolidFn,
    find_path,
    path_to_actions,
    current_coord_from_snapshot,
)
from .collision import (
    BlockCollisionProfile,
    default_is_solid_block,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class ActionExecutorConfig:
    """
    Configuration knobs for ActionExecutor.

    These should be environment- and modpack-agnostic. GTNH-specific
    tuning belongs in higher layers or injected strategies.
    """

    # Limit on nav search complexity
    max_nav_steps: int = 2048

    # Default arrival radius for move_to
    default_move_radius: float = 0.5


class ActionExecutor:
    """
    Translate high-level Actions into PacketClient messages.

    Public contract:
      execute(action, snapshot) -> ActionResult

    Responsibilities:
    - Build NavGrid from RawWorldSnapshot.
    - Run A* pathfinding for move_to.
    - Emit movement / dig / place / use / interact packets.
    - Surface explicit, structured errors.

    Non-responsibilities:
    - WorldState construction.
    - Tech semantics.
    - Planning, skill selection, or virtue scoring.
    """

    def __init__(
        self,
        client: PacketClient,
        *,
        is_solid_block: BlockSolidFn | None = None,
        config: ActionExecutorConfig | None = None,
    ) -> None:
        self._client = client

        # Use injected collision logic if provided; otherwise fall back
        # to the module-level default profile.
        self._is_solid_block: BlockSolidFn = (
            is_solid_block if is_solid_block is not None else default_is_solid_block
        )

        self._cfg = config if config is not None else ActionExecutorConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def execute(self, action: Action, snapshot: RawWorldSnapshot) -> ActionResult:
        """
        Execute a single high-level Action against the given snapshot.

        The snapshot is treated as read-only and must not be mutated here.
        """
        atype = getattr(action, "type", None)
        params = getattr(action, "params", {}) or {}

        if not isinstance(params, Mapping):
            return ActionResult(
                success=False,
                error="invalid_params",
                details={"reason": "params_not_mapping", "action_type": atype},
            )

        try:
            if atype == "move_to":
                return self._execute_move_to(params, snapshot)
            elif atype == "break_block":
                return self._execute_break_block(params, snapshot)
            elif atype == "place_block":
                return self._execute_place_block(params, snapshot)
            elif atype == "use_item":
                return self._execute_use_item(params, snapshot)
            elif atype == "interact":
                return self._execute_interact(params, snapshot)
            else:
                return ActionResult(
                    success=False,
                    error="unsupported_action",
                    details={"action_type": atype},
                )
        except Exception as exc:
            # Catch-all for transport / unexpected errors.
            return ActionResult(
                success=False,
                error="execution_exception",
                details={
                    "action_type": atype,
                    "exception": repr(exc),
                },
            )

    # ------------------------------------------------------------------
    # Action handlers
    # ------------------------------------------------------------------

    def _execute_move_to(
        self,
        params: Mapping[str, Any],
        snapshot: RawWorldSnapshot,
    ) -> ActionResult:
        """
        Move the player to a target coordinate using NavGrid + A*.

        Expected params:
            - "x", "y", "z": numeric target position (block coords)
            - "radius": optional float radius (default from config)

        Behavior:
            - Build NavGrid with injected is_solid_block.
            - Run A* from current_coord → target_coord.
            - If success: convert path to move_to Actions and emit
              movement packets for each step.
            - If failure: return ActionResult with reason.
        """
        try:
            tx = int(params["x"])
            ty = int(params.get("y", snapshot.player_pos.get("y", 0.0)))
            tz = int(params["z"])
        except Exception:
            return ActionResult(
                success=False,
                error="invalid_params",
                details={"reason": "missing_or_non_numeric_xyz", "params": dict(params)},
            )

        radius = float(params.get("radius", self._cfg.default_move_radius))

        # Build NavGrid from snapshot
        grid = NavGrid(snapshot=snapshot, is_solid_block=self._is_solid_block)

        # Derive start/goal coords
        start = current_coord_from_snapshot(snapshot)
        goal = (tx, ty, tz)

        # Run pathfinding
        pf_result = find_path(
            grid,
            start=start,
            goal=goal,
            max_steps=self._cfg.max_nav_steps,
        )

        if not pf_result.success or not pf_result.path:
            return ActionResult(
                success=False,
                error="nav_failure",
                details={
                    "from": start,
                    "to": goal,
                    "reason": pf_result.reason,
                },
            )

        # Convert path to move_to actions
        move_actions = path_to_actions(
            pf_result,
            snapshot,
            radius=radius,
        )

        # Emit movement packets for each step.
        steps_sent = 0
        for step in move_actions:
            try:
                self._send_move_step(step.params)
                steps_sent += 1
            except Exception as exc:
                # On IO error, stop and report partial success.
                return ActionResult(
                    success=False,
                    error="io_error",
                    details={
                        "stage": "move_to",
                        "from": start,
                        "to": goal,
                        "steps_attempted": steps_sent,
                        "exception": repr(exc),
                    },
                )

        return ActionResult(
            success=True,
            error=None,
            details={
                "from": start,
                "to": goal,
                "steps": steps_sent,
            },
        )

    def _execute_break_block(
        self,
        params: Mapping[str, Any],
        snapshot: RawWorldSnapshot,  # unused but kept for future context
    ) -> ActionResult:
        """
        Break a block at the given coordinates.

        Expected params:
            - "x", "y", "z": block coordinates
            - "face": optional face index or direction (depends on IPC schema)
        """
        try:
            x = int(params["x"])
            y = int(params["y"])
            z = int(params["z"])
        except Exception:
            return ActionResult(
                success=False,
                error="invalid_params",
                details={"reason": "missing_or_non_numeric_xyz", "params": dict(params)},
            )

        face = params.get("face", "auto")

        try:
            # Start digging
            self._client.send_packet(
                "block_dig",
                {
                    "status": "start",
                    "x": x,
                    "y": y,
                    "z": z,
                    "face": face,
                },
            )
            # Stop / finish digging
            self._client.send_packet(
                "block_dig",
                {
                    "status": "stop",
                    "x": x,
                    "y": y,
                    "z": z,
                    "face": face,
                },
            )
        except Exception as exc:
            return ActionResult(
                success=False,
                error="io_error",
                details={
                    "stage": "break_block",
                    "x": x,
                    "y": y,
                    "z": z,
                    "exception": repr(exc),
                },
            )

        return ActionResult(
            success=True,
            error=None,
            details={"x": x, "y": y, "z": z},
        )

    def _execute_place_block(
        self,
        params: Mapping[str, Any],
        snapshot: RawWorldSnapshot,  # unused but reserved for future checks
    ) -> ActionResult:
        """
        Place a block using the currently held item at the target coordinates.

        Expected params:
            - "x", "y", "z": target block position or adjacent position
            - "face": optional face direction
        """
        try:
            x = int(params["x"])
            y = int(params["y"])
            z = int(params["z"])
        except Exception:
            return ActionResult(
                success=False,
                error="invalid_params",
                details={"reason": "missing_or_non_numeric_xyz", "params": dict(params)},
            )

        face = params.get("face", "auto")

        try:
            self._client.send_packet(
                "block_place",
                {
                    "x": x,
                    "y": y,
                    "z": z,
                    "face": face,
                },
            )
        except Exception as exc:
            return ActionResult(
                success=False,
                error="io_error",
                details={
                    "stage": "place_block",
                    "x": x,
                    "y": y,
                    "z": z,
                    "exception": repr(exc),
                },
            )

        return ActionResult(
            success=True,
            error=None,
            details={"x": x, "y": y, "z": z},
        )

    def _execute_use_item(
        self,
        params: Mapping[str, Any],
        snapshot: RawWorldSnapshot,  # reserved for later (line of sight, etc.)
    ) -> ActionResult:
        """
        Use the currently held item (e.g., right-click in air or on a block).

        Expected params (schema is flexible and IPC-defined):
            - "target": one of {"self", "block", "entity", "air"} (optional)
            - "x", "y", "z": optional coordinates (for block/entity)
            - any other keys are passed through to IPC layer.
        """
        payload: Dict[str, Any] = {"target": params.get("target", "air")}
        # Pass through positional info if present.
        for key in ("x", "y", "z", "face", "entity_id"):
            if key in params:
                payload[key] = params[key]

        # Forward any extra params untouched.
        for k, v in params.items():
            if k not in payload:
                payload[k] = v

        try:
            self._client.send_packet("use_item", payload)
        except Exception as exc:
            return ActionResult(
                success=False,
                error="io_error",
                details={
                    "stage": "use_item",
                    "payload": payload,
                    "exception": repr(exc),
                },
            )

        return ActionResult(
            success=True,
            error=None,
            details=payload,
        )

    def _execute_interact(
        self,
        params: Mapping[str, Any],
        snapshot: RawWorldSnapshot,  # reserved (e.g., for range checks)
    ) -> ActionResult:
        """
        Interact with an entity or block.

        Expected params:
            - "kind": "entity" or "block"
            - "entity_id": int if kind == "entity"
            - "x", "y", "z": coords if kind == "block"
        """
        kind = params.get("kind")

        if kind not in ("entity", "block"):
            return ActionResult(
                success=False,
                error="invalid_params",
                details={
                    "reason": "kind_must_be_entity_or_block",
                    "params": dict(params),
                },
            )

        payload: Dict[str, Any] = {"kind": kind}

        if kind == "entity":
            try:
                payload["entity_id"] = int(params["entity_id"])
            except Exception:
                return ActionResult(
                    success=False,
                    error="invalid_params",
                    details={
                        "reason": "missing_or_non_numeric_entity_id",
                        "params": dict(params),
                    },
                )
        else:  # kind == "block"
            try:
                payload["x"] = int(params["x"])
                payload["y"] = int(params["y"])
                payload["z"] = int(params["z"])
            except Exception:
                return ActionResult(
                    success=False,
                    error="invalid_params",
                    details={
                        "reason": "missing_or_non_numeric_xyz",
                        "params": dict(params),
                    },
                )

        # Pass through any extra parameters
        for k, v in params.items():
            if k not in payload:
                payload[k] = v

        try:
            self._client.send_packet("interact", payload)
        except Exception as exc:
            return ActionResult(
                success=False,
                error="io_error",
                details={
                    "stage": "interact",
                    "payload": payload,
                    "exception": repr(exc),
                },
            )

        return ActionResult(
            success=True,
            error=None,
            details=payload,
        )

    # ------------------------------------------------------------------
    # Low-level emitters
    # ------------------------------------------------------------------

    def _send_move_step(self, step_params: Mapping[str, Any]) -> None:
        """
        Emit a single movement step.

        For IPC mode, you can interpret this generically:
          - type: "move_step"
          - payload: {x, y, z, radius}

        Forge mod / external client can translate that into concrete
        packets (position updates, relative moves, etc.).
        """
        x = step_params.get("x")
        y = step_params.get("y")
        z = step_params.get("z")
        radius = step_params.get("radius")

        payload = {
            "x": x,
            "y": y,
            "z": z,
            "radius": radius,
        }

        # Use a generic high-level move_step packet type; the concrete
        # implementation (IPC or protocol) decides how to realize it.
        self._client.send_packet("move_step", payload)

```

Tests that pass in their own `is_solid_block` still work exactly as before.

---

## 3. Updated `src/bot_core/core.py` (optional but cleaner)

You already allow injection of `is_solid_block` into `BotCoreImpl`. Now we make sure that, if nothing is passed, it uses the default profile.

Full updated file:
```
# src/bot_core/core.py
"""
Concrete BotCore implementation for Minecraft 1.7.10 (GTNH).

This module wires together:
- PacketClient / IPC transport (M6 network layer)
- WorldTracker (incremental raw world state)
- ActionExecutor (high-level action execution)
- Snapshot adapter (RawWorldSnapshot -> WorldState)

Public surface (for higher layers like M8):
    class BotCoreImpl(BotCore):
        connect() -> None
        disconnect() -> None
        tick() -> None
        observe() -> RawWorldSnapshot
        get_world_state() -> WorldState
        execute_action(Action) -> ActionResult

Design constraints:
- No packet or protocol details leak to callers.
- No navigation internals or exceptions leak to callers.
- All action-related failures return ActionResult with explicit error codes.
- Non-action failures (e.g., connection problems) raise domain errors.
- No semantics, virtues, or tech-state logic here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from env.loader import load_environment  # Phase 0: EnvProfile loader
from spec.types import Action, ActionResult, WorldState  # M1 shared types

from spec.bot_core import BotCore  # BotCore interface definition (M1)
from .net import PacketClient, create_packet_client_for_env
from .world_tracker import WorldTracker
from .snapshot import RawWorldSnapshot, snapshot_to_world_state
from .actions import ActionExecutor, ActionExecutorConfig
from .collision import default_is_solid_block


# ---------------------------------------------------------------------------
# Domain errors
# ---------------------------------------------------------------------------


@dataclass
class BotCoreError(RuntimeError):
    """
    Domain-level error raised by BotCoreImpl for non-action failures.

    Examples:
        - failed to connect or disconnect cleanly
        - tick loop I/O failures
        - configuration errors

    Action execution errors should generally NOT raise this; instead,
    ActionExecutor must return an ActionResult with error set.
    """

    code: str
    details: dict[str, Any]

    def __str__(self) -> str:
        return f"BotCoreError(code={self.code!r}, details={self.details!r})"


# ---------------------------------------------------------------------------
# Implementation
# ---------------------------------------------------------------------------


class BotCoreImpl(BotCore):
    """
    Concrete BotCore for MC 1.7.10 GTNH.

    Orchestrates:
        - PacketClient / IPC (transport)
        - WorldTracker (RawWorldSnapshot)
        - ActionExecutor (high-level actions)

    Consumers (M8, etc.) see:
        - connect / disconnect / tick
        - observe -> RawWorldSnapshot
        - get_world_state -> WorldState
        - execute_action(Action) -> ActionResult
    """

    def __init__(
        self,
        client: Optional[PacketClient] = None,
        *,
        action_config: Optional[ActionExecutorConfig] = None,
        # Optional hook for supplying custom collision semantics.
        is_solid_block=None,
    ) -> None:
        """
        Build a BotCoreImpl.

        If `client` is None, it is constructed from the Phase 0 environment
        via create_packet_client_for_env(), which uses env.yaml / EnvProfile.

        Parameters:
            client: Optional PacketClient (useful for tests)
            action_config: optional ActionExecutorConfig
            is_solid_block: optional BlockSolidFn to plug in collision logic
                            (M3 semantics can inject this later)
        """
        # EnvProfile mainly used for logging/context right now; creation of
        # the client is delegated to create_packet_client_for_env().
        self._env = load_environment()

        # Transport layer
        self._client: PacketClient = client or create_packet_client_for_env()

        # World tracking
        self._tracker = WorldTracker(self._client)

        # Choose collision function: injected or default profile.
        solid_fn = is_solid_block if is_solid_block is not None else default_is_solid_block

        # Action execution
        self._executor = ActionExecutor(
            self._client,
            is_solid_block=solid_fn,
            config=action_config,
        )

        self._connected: bool = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """
        Establish connection to the Minecraft world.

        Raises:
            BotCoreError if the underlying client fails to connect.
        """
        if self._connected:
            return

        try:
            self._client.connect()
        except Exception as exc:
            raise BotCoreError(
                code="connect_failed",
                details={"exception": repr(exc)},
            ) from exc

        self._connected = True

        # Seed tracker context with basic env profile metadata (optional).
        self._tracker.set_context(
            "env_profile_name",
            getattr(self._env, "name", "default"),
        )
        self._tracker.set_context("bot_mode", getattr(self._env, "bot_mode", "unknown"))

    def disconnect(self) -> None:
        """
        Disconnect from the Minecraft world.

        Raises:
            BotCoreError if the underlying client fails to disconnect.
        """
        if not self._connected:
            return

        try:
            self._client.disconnect()
        except Exception as exc:
            # We still consider ourselves disconnected, but bubble the error.
            self._connected = False
            raise BotCoreError(
                code="disconnect_failed",
                details={"exception": repr(exc)},
            ) from exc

        self._connected = False

    def tick(self) -> None:
        """
        Pump the transport layer and feed events into the WorldTracker.

        This must be called regularly (e.g., once per iteration of the main
        agent loop). It does NOT itself execute any actions.
        """
        if not self._connected:
            # It's acceptable to call tick() before connect, but it's a no-op.
            return

        try:
            self._client.tick()
        except Exception as exc:
            # Transport-level issues are domain errors, not ActionResult issues.
            raise BotCoreError(
                code="tick_failed",
                details={"exception": repr(exc)},
            ) from exc

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def observe(self) -> RawWorldSnapshot:
        """
        Return a raw snapshot of the current world state.

        This is intended for internal consumers (e.g., debugging, advanced
        semantics modules). Most of the time higher layers should use
        get_world_state() instead.

        Returns:
            RawWorldSnapshot constructed by WorldTracker.
        """
        return self._tracker.build_snapshot()

    def get_world_state(self) -> WorldState:
        """
        Return a semantic WorldState for agent consumption.

        This is the primary observation API for the planner and skills.

        Returns:
            WorldState created via snapshot_to_world_state(observe()).
        """
        raw = self._tracker.build_snapshot()
        return snapshot_to_world_state(raw)

    # ------------------------------------------------------------------
    # Action execution
    # ------------------------------------------------------------------

    def execute_action(self, action: Action) -> ActionResult:
        """
        Execute a single high-level Action.

        All navigation / pathfinding / I/O details are internal. Nav and
        transport exceptions are caught and surfaced as ActionResult errors.

        Returns:
            ActionResult with:
                - success: bool
                - error: str | None
                - details: dict[str, Any]
        """
        raw = self._tracker.build_snapshot()

        try:
            result = self._executor.execute(action, raw)
        except Exception as exc:
            # As a last resort, ensure no low-level exception leaks upstream.
            return ActionResult(
                success=False,
                error="botcore_execute_exception",
                details={
                    "exception": repr(exc),
                    "action_type": getattr(action, "type", None),
                },
            )

        if not isinstance(result, ActionResult):
            return ActionResult(
                success=False,
                error="invalid_executor_result",
                details={"result_repr": repr(result)},
            )

        return result

```



## 3. Logging, metrics, and traces

Right now:

- Errors come back as `ActionResult.error` or `BotCoreError`.
    
- But you’re not capturing structured logs anywhere.
    

You’ll want:

- A **thin logging wrapper** around:
    
    - ActionExecutor.execute
        
    - BotCoreImpl.execute_action
        
- So you can log:
    
    - Action type, params (redacted if needed)
        
    - Duration
        
    - Result (success/error code)
        
- And eventually feed that into:
    
    - M9 monitoring
        
    - M10 experience / replay buffer
        

Doing it now saves you from dumb archaeology later.

## 1. Add a tracing module

Create: `src/bot_core/tracing.py`

This gives you:

- `ActionTraceRecord` – structured trace entry
    
- `ActionTracer` – in-memory ring buffer + logging hook

```
# src/bot_core/tracing.py
"""
Tracing and metrics for bot_core_1_7_10.

This module provides a thin, structured logging layer around action
execution so that higher modules (M9 monitoring, M10 experience) can
consume consistent traces.

It does NOT:
- Call LLMs
- Infer semantics
- Make control decisions
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, List, Optional

from spec.types import Action, ActionResult
from .snapshot import RawWorldSnapshot


@dataclass
class ActionTraceRecord:
    """
    Structured record of a single action execution.

    Fields are intentionally generic and stable so that M9/M10 can build
    on top of this without depending on internal details of bot_core.
    """

    timestamp: float           # wall-clock time (time.time())
    duration_s: float          # execution duration in seconds

    action_type: Optional[str]
    params: dict[str, Any]

    success: bool
    error: Optional[str]

    # Minimal world context at decision time
    tick: Optional[int]
    dimension: Optional[str]
    position: dict[str, float]


class ActionTracer:
    """
    In-memory action tracer with optional logging.

    Responsibilities:
    - Keep a rolling buffer of recent ActionTraceRecord entries.
    - Emit a single structured log line per action (info level).

    This is deliberately minimal; M9/M10 can later:
      - consume tracer records
      - or replace this with a more complex sink.
    """

    def __init__(
        self,
        *,
        logger: Optional[logging.Logger] = None,
        max_records: int = 10_000,
    ) -> None:
        self._logger = logger or logging.getLogger("bot_core.action")
        self._records: Deque[ActionTraceRecord] = deque(maxlen=max_records)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record(
        self,
        *,
        action: Action,
        snapshot: RawWorldSnapshot,
        result: ActionResult,
        duration_s: float,
    ) -> None:
        """
        Record a trace for a completed action.

        This should be called even on failures; `success` and `error`
        capture outcome.
        """
        try:
            record = self._build_record(action, snapshot, result, duration_s)
        except Exception:
            # Tracing must never crash the caller.
            self._logger.exception("Failed to build ActionTraceRecord")
            return

        self._records.append(record)

        # Emit a structured log line. Keep it compact but informative.
        self._logger.info(
            "action_exec type=%s success=%s error=%s duration=%.4fs tick=%s "
            "dim=%s pos=(%.2f,%.2f,%.2f)",
            record.action_type,
            record.success,
            record.error,
            record.duration_s,
            record.tick,
            record.dimension,
            record.position.get("x", 0.0),
            record.position.get("y", 0.0),
            record.position.get("z", 0.0),
        )

    def get_records(self) -> List[ActionTraceRecord]:
        """
        Return a snapshot of all currently buffered records.

        Intended for debugging / M9 / M10; not performance-critical.
        """
        return list(self._records)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_record(
        self,
        action: Action,
        snapshot: RawWorldSnapshot,
        result: ActionResult,
        duration_s: float,
    ) -> ActionTraceRecord:
        # Extract minimal context
        tick = snapshot.tick
        dimension = snapshot.dimension
        pos = {
            "x": float(snapshot.player_pos.get("x", 0.0)),
            "y": float(snapshot.player_pos.get("y", 0.0)),
            "z": float(snapshot.player_pos.get("z", 0.0)),
        }

        # Params can be large; use a shallow copy to avoid surprises.
        params = {}
        raw_params = getattr(action, "params", {}) or {}
        if isinstance(raw_params, dict):
            params = dict(raw_params)

        return ActionTraceRecord(
            timestamp=time.time(),
            duration_s=duration_s,
            action_type=getattr(action, "type", None),
            params=params,
            success=bool(getattr(result, "success", False)),
            error=getattr(result, "error", None),
            tick=tick,
            dimension=dimension,
            position=pos,
        )

```

## 2. Add logging wrapper inside ActionExecutor

Now we give `ActionExecutor` its own logger & debug output. The _metrics_ live at BotCore level; this is just lightweight internal logging.

Replace your existing `src/bot_core/actions.py` with this full updated version (includes collision + logging):
```
# src/bot_core/actions.py
"""
Action execution for bot_core_1_7_10.

This module translates high-level Actions into protocol- or IPC-level
messages via a PacketClient.

Design constraints:
- High-level, atomic-ish actions:
    - move_to
    - break_block
    - place_block
    - use_item
    - interact
- One execute_action(...) call per logical operation.
- Explicit, structured failures:
    - no path
    - unsupported action
    - IO/IPC error
- No mutation of world state; this layer only sends packets/messages.
- No semantics, no tech_state, no virtues.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Mapping

from spec.types import Action, ActionResult  # from M1
from .snapshot import RawWorldSnapshot
from .net import PacketClient
from .nav import (
    NavGrid,
    BlockSolidFn,
    find_path,
    path_to_actions,
    current_coord_from_snapshot,
)
from .collision import (
    BlockCollisionProfile,
    default_is_solid_block,
)


log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class ActionExecutorConfig:
    """
    Configuration knobs for ActionExecutor.

    These should be environment- and modpack-agnostic. GTNH-specific
    tuning belongs in higher layers or injected strategies.
    """

    # Limit on nav search complexity
    max_nav_steps: int = 2048

    # Default arrival radius for move_to
    default_move_radius: float = 0.5


class ActionExecutor:
    """
    Translate high-level Actions into PacketClient messages.

    Public contract:
      execute(action, snapshot) -> ActionResult

    Responsibilities:
    - Build NavGrid from RawWorldSnapshot.
    - Run A* pathfinding for move_to.
    - Emit movement / dig / place / use / interact packets.
    - Surface explicit, structured errors.

    Non-responsibilities:
    - WorldState construction.
    - Tech semantics.
    - Planning, skill selection, or virtue scoring.
    """

    def __init__(
        self,
        client: PacketClient,
        *,
        is_solid_block: BlockSolidFn | None = None,
        config: ActionExecutorConfig | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self._client = client

        # Use injected collision logic if provided; otherwise fall back
        # to the module-level default profile.
        self._is_solid_block: BlockSolidFn = (
            is_solid_block if is_solid_block is not None else default_is_solid_block
        )

        self._cfg = config if config is not None else ActionExecutorConfig()
        self._log = logger or log

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def execute(self, action: Action, snapshot: RawWorldSnapshot) -> ActionResult:
        """
        Execute a single high-level Action against the given snapshot.

        The snapshot is treated as read-only and must not be mutated here.
        """
        atype = getattr(action, "type", None)
        params = getattr(action, "params", {}) or {}

        self._log.debug("ActionExecutor.execute start type=%s params=%r", atype, params)

        if not isinstance(params, Mapping):
            self._log.warning(
                "ActionExecutor.execute invalid params (not mapping): %r", params
            )
            return ActionResult(
                success=False,
                error="invalid_params",
                details={"reason": "params_not_mapping", "action_type": atype},
            )

        try:
            if atype == "move_to":
                result = self._execute_move_to(params, snapshot)
            elif atype == "break_block":
                result = self._execute_break_block(params, snapshot)
            elif atype == "place_block":
                result = self._execute_place_block(params, snapshot)
            elif atype == "use_item":
                result = self._execute_use_item(params, snapshot)
            elif atype == "interact":
                result = self._execute_interact(params, snapshot)
            else:
                result = ActionResult(
                    success=False,
                    error="unsupported_action",
                    details={"action_type": atype},
                )
        except Exception as exc:
            # Catch-all for transport / unexpected errors.
            self._log.exception(
                "ActionExecutor.execute raised unexpectedly for type=%s", atype
            )
            return ActionResult(
                success=False,
                error="execution_exception",
                details={
                    "action_type": atype,
                    "exception": repr(exc),
                },
            )

        self._log.debug(
            "ActionExecutor.execute end type=%s success=%s error=%s",
            atype,
            getattr(result, "success", None),
            getattr(result, "error", None),
        )
        return result

    # ------------------------------------------------------------------
    # Action handlers
    # ------------------------------------------------------------------

    def _execute_move_to(
        self,
        params: Mapping[str, Any],
        snapshot: RawWorldSnapshot,
    ) -> ActionResult:
        """
        Move the player to a target coordinate using NavGrid + A*.

        Expected params:
            - "x", "y", "z": numeric target position (block coords)
            - "radius": optional float radius (default from config)
        """
        try:
            tx = int(params["x"])
            ty = int(params.get("y", snapshot.player_pos.get("y", 0.0)))
            tz = int(params["z"])
        except Exception:
            return ActionResult(
                success=False,
                error="invalid_params",
                details={"reason": "missing_or_non_numeric_xyz", "params": dict(params)},
            )

        radius = float(params.get("radius", self._cfg.default_move_radius))

        grid = NavGrid(snapshot=snapshot, is_solid_block=self._is_solid_block)

        start = current_coord_from_snapshot(snapshot)
        goal = (tx, ty, tz)

        pf_result = find_path(
            grid,
            start=start,
            goal=goal,
            max_steps=self._cfg.max_nav_steps,
        )

        if not pf_result.success or not pf_result.path:
            return ActionResult(
                success=False,
                error="nav_failure",
                details={
                    "from": start,
                    "to": goal,
                    "reason": pf_result.reason,
                },
            )

        move_actions = path_to_actions(
            pf_result,
            snapshot,
            radius=radius,
        )

        steps_sent = 0
        for step in move_actions:
            try:
                self._send_move_step(step.params)
                steps_sent += 1
            except Exception as exc:
                return ActionResult(
                    success=False,
                    error="io_error",
                    details={
                        "stage": "move_to",
                        "from": start,
                        "to": goal,
                        "steps_attempted": steps_sent,
                        "exception": repr(exc),
                    },
                )

        return ActionResult(
            success=True,
            error=None,
            details={
                "from": start,
                "to": goal,
                "steps": steps_sent,
            },
        )

    def _execute_break_block(
        self,
        params: Mapping[str, Any],
        snapshot: RawWorldSnapshot,  # unused but kept for future context
    ) -> ActionResult:
        """
        Break a block at the given coordinates.

        Expected params:
            - "x", "y", "z": block coordinates
            - "face": optional face index or direction (depends on IPC schema)
        """
        try:
            x = int(params["x"])
            y = int(params["y"])
            z = int(params["z"])
        except Exception:
            return ActionResult(
                success=False,
                error="invalid_params",
                details={"reason": "missing_or_non_numeric_xyz", "params": dict(params)},
            )

        face = params.get("face", "auto")

        try:
            self._client.send_packet(
                "block_dig",
                {
                    "status": "start",
                    "x": x,
                    "y": y,
                    "z": z,
                    "face": face,
                },
            )
            self._client.send_packet(
                "block_dig",
                {
                    "status": "stop",
                    "x": x,
                    "y": y,
                    "z": z,
                    "face": face,
                },
            )
        except Exception as exc:
            return ActionResult(
                success=False,
                error="io_error",
                details={
                    "stage": "break_block",
                    "x": x,
                    "y": y,
                    "z": z,
                    "exception": repr(exc),
                },
            )

        return ActionResult(
            success=True,
            error=None,
            details={"x": x, "y": y, "z": z},
        )

    def _execute_place_block(
        self,
        params: Mapping[str, Any],
        snapshot: RawWorldSnapshot,  # unused but reserved for future checks
    ) -> ActionResult:
        """
        Place a block using the currently held item at the target coordinates.

        Expected params:
            - "x", "y", "z": target block position or adjacent position
            - "face": optional face direction
        """
        try:
            x = int(params["x"])
            y = int(params["y"])
            z = int(params["z"])
        except Exception:
            return ActionResult(
                success=False,
                error="invalid_params",
                details={"reason": "missing_or_non_numeric_xyz", "params": dict(params)},
            )

        face = params.get("face", "auto")

        try:
            self._client.send_packet(
                "block_place",
                {
                    "x": x,
                    "y": y,
                    "z": z,
                    "face": face,
                },
            )
        except Exception as exc:
            return ActionResult(
                success=False,
                error="io_error",
                details={
                    "stage": "place_block",
                    "x": x,
                    "y": y,
                    "z": z,
                    "exception": repr(exc),
                },
            )

        return ActionResult(
            success=True,
            error=None,
            details={"x": x, "y": y, "z": z},
        )

    def _execute_use_item(
        self,
        params: Mapping[str, Any],
        snapshot: RawWorldSnapshot,  # reserved for later (line of sight, etc.)
    ) -> ActionResult:
        """
        Use the currently held item (e.g., right-click in air or on a block).

        Expected params (schema is flexible and IPC-defined):
            - "target": one of {"self", "block", "entity", "air"} (optional)
            - "x", "y", "z": optional coordinates (for block/entity)
            - any other keys are passed through to IPC layer.
        """
        payload: Dict[str, Any] = {"target": params.get("target", "air")}
        for key in ("x", "y", "z", "face", "entity_id"):
            if key in params:
                payload[key] = params[key]

        for k, v in params.items():
            if k not in payload:
                payload[k] = v

        try:
            self._client.send_packet("use_item", payload)
        except Exception as exc:
            return ActionResult(
                success=False,
                error="io_error",
                details={
                    "stage": "use_item",
                    "payload": payload,
                    "exception": repr(exc),
                },
            )

        return ActionResult(
            success=True,
            error=None,
            details=payload,
        )

    def _execute_interact(
        self,
        params: Mapping[str, Any],
        snapshot: RawWorldSnapshot,  # reserved (e.g., for range checks)
    ) -> ActionResult:
        """
        Interact with an entity or block.

        Expected params:
            - "kind": "entity" or "block"
            - "entity_id": int if kind == "entity"
            - "x", "y", "z": coords if kind == "block"
        """
        kind = params.get("kind")

        if kind not in ("entity", "block"):
            return ActionResult(
                success=False,
                error="invalid_params",
                details={
                    "reason": "kind_must_be_entity_or_block",
                    "params": dict(params),
                },
            )

        payload: Dict[str, Any] = {"kind": kind}

        if kind == "entity":
            try:
                payload["entity_id"] = int(params["entity_id"])
            except Exception:
                return ActionResult(
                    success=False,
                    error="invalid_params",
                    details={
                        "reason": "missing_or_non_numeric_entity_id",
                        "params": dict(params),
                    },
                )
        else:  # kind == "block"
            try:
                payload["x"] = int(params["x"])
                payload["y"] = int(params["y"])
                payload["z"] = int(params["z"])
            except Exception:
                return ActionResult(
                    success=False,
                    error="invalid_params",
                    details={
                        "reason": "missing_or_non_numeric_xyz",
                        "params": dict(params),
                    },
                )

        for k, v in params.items():
            if k not in payload:
                payload[k] = v

        try:
            self._client.send_packet("interact", payload)
        except Exception as exc:
            return ActionResult(
                success=False,
                error="io_error",
                details={
                    "stage": "interact",
                    "payload": payload,
                    "exception": repr(exc),
                },
            )

        return ActionResult(
            success=True,
            error=None,
            details=payload,
        )

    # ------------------------------------------------------------------
    # Low-level emitters
    # ------------------------------------------------------------------

    def _send_move_step(self, step_params: Mapping[str, Any]) -> None:
        """
        Emit a single movement step.

        For IPC mode, you can interpret this generically:
          - type: "move_step"
          - payload: {x, y, z, radius}
        """
        x = step_params.get("x")
        y = step_params.get("y")
        z = step_params.get("z")
        radius = step_params.get("radius")

        payload = {
            "x": x,
            "y": y,
            "z": z,
            "radius": radius,
        }

        self._client.send_packet("move_step", payload)

```

Tests will still pass: they don’t care about logging.

---

## 3. Wrap BotCoreImpl.execute_action with timing + tracing

Now we add the actual trace/metrics wrapper at the level that M8+ will call.

Replace your existing `src/bot_core/core.py` with this updated version:
```
# src/bot_core/core.py
"""
Concrete BotCore implementation for Minecraft 1.7.10 (GTNH).

This module wires together:
- PacketClient / IPC transport (M6 network layer)
- WorldTracker (incremental raw world state)
- ActionExecutor (high-level action execution)
- Snapshot adapter (RawWorldSnapshot -> WorldState)
- ActionTracer (logging / metrics for actions)

Public surface (for higher layers like M8):
    class BotCoreImpl(BotCore):
        connect() -> None
        disconnect() -> None
        tick() -> None
        observe() -> RawWorldSnapshot
        get_world_state() -> WorldState
        execute_action(Action) -> ActionResult

Design constraints:
- No packet or protocol details leak to callers.
- No navigation internals or exceptions leak to callers.
- All action-related failures return ActionResult with explicit error codes.
- Non-action failures (e.g., connection problems) raise domain errors.
- No semantics, virtues, or tech-state logic here.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Optional

from env.loader import load_environment  # Phase 0: EnvProfile loader
from spec.types import Action, ActionResult, WorldState  # M1 shared types

from spec.bot_core import BotCore  # BotCore interface definition (M1)
from .net import PacketClient, create_packet_client_for_env
from .world_tracker import WorldTracker
from .snapshot import RawWorldSnapshot, snapshot_to_world_state
from .actions import ActionExecutor, ActionExecutorConfig
from .collision import default_is_solid_block
from .tracing import ActionTracer


log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Domain errors
# ---------------------------------------------------------------------------


@dataclass
class BotCoreError(RuntimeError):
    """
    Domain-level error raised by BotCoreImpl for non-action failures.

    Examples:
        - failed to connect or disconnect cleanly
        - tick loop I/O failures
        - configuration errors

    Action execution errors should generally NOT raise this; instead,
    ActionExecutor must return an ActionResult with error set.
    """

    code: str
    details: dict[str, Any]

    def __str__(self) -> str:
        return f"BotCoreError(code={self.code!r}, details={self.details!r})"


# ---------------------------------------------------------------------------
# Implementation
# ---------------------------------------------------------------------------


class BotCoreImpl(BotCore):
    """
    Concrete BotCore for MC 1.7.10 GTNH.

    Orchestrates:
        - PacketClient / IPC (transport)
        - WorldTracker (RawWorldSnapshot)
        - ActionExecutor (high-level actions)
        - ActionTracer (metrics)

    Consumers (M8, etc.) see:
        - connect / disconnect / tick
        - observe -> RawWorldSnapshot
        - get_world_state -> WorldState
        - execute_action(Action) -> ActionResult
    """

    def __init__(
        self,
        client: Optional[PacketClient] = None,
        *,
        action_config: Optional[ActionExecutorConfig] = None,
        # Optional hook for supplying custom collision semantics.
        is_solid_block=None,
        tracer: Optional[ActionTracer] = None,
    ) -> None:
        """
        Build a BotCoreImpl.

        If `client` is None, it is constructed from the Phase 0 environment
        via create_packet_client_for_env(), which uses env.yaml / EnvProfile.
        """
        self._env = load_environment()

        # Transport layer
        self._client: PacketClient = client or create_packet_client_for_env()

        # World tracking
        self._tracker = WorldTracker(self._client)

        # Collision function: injected or default profile.
        solid_fn = is_solid_block if is_solid_block is not None else default_is_solid_block

        # Action execution
        self._executor = ActionExecutor(
            self._client,
            is_solid_block=solid_fn,
            config=action_config,
        )

        # Tracing / metrics
        self._tracer: ActionTracer = tracer or ActionTracer()

        self._connected: bool = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """
        Establish connection to the Minecraft world.

        Raises:
            BotCoreError if the underlying client fails to connect.
        """
        if self._connected:
            return

        try:
            self._client.connect()
        except Exception as exc:
            raise BotCoreError(
                code="connect_failed",
                details={"exception": repr(exc)},
            ) from exc

        self._connected = True

        # Seed tracker context with basic env profile metadata (optional).
        self._tracker.set_context(
            "env_profile_name",
            getattr(self._env, "name", "default"),
        )
        self._tracker.set_context("bot_mode", getattr(self._env, "bot_mode", "unknown"))

    def disconnect(self) -> None:
        """
        Disconnect from the Minecraft world.

        Raises:
            BotCoreError if the underlying client fails to disconnect.
        """
        if not self._connected:
            return

        try:
            self._client.disconnect()
        except Exception as exc:
            self._connected = False
            raise BotCoreError(
                code="disconnect_failed",
                details={"exception": repr(exc)},
            ) from exc

        self._connected = False

    def tick(self) -> None:
        """
        Pump the transport layer and feed events into the WorldTracker.

        This must be called regularly (e.g., once per iteration of the main
        agent loop). It does NOT itself execute any actions.
        """
        if not self._connected:
            return

        try:
            self._client.tick()
        except Exception as exc:
            raise BotCoreError(
                code="tick_failed",
                details={"exception": repr(exc)},
            ) from exc

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def observe(self) -> RawWorldSnapshot:
        """
        Return a raw snapshot of the current world state.

        This is intended for internal consumers (e.g., debugging, advanced
        semantics modules). Most of the time higher layers should use
        get_world_state() instead.
        """
        return self._tracker.build_snapshot()

    def get_world_state(self) -> WorldState:
        """
        Return a semantic WorldState for agent consumption.
        """
        raw = self._tracker.build_snapshot()
        return snapshot_to_world_state(raw)

    # ------------------------------------------------------------------
    # Action execution
    # ------------------------------------------------------------------

    def execute_action(self, action: Action) -> ActionResult:
        """
        Execute a single high-level Action.

        All navigation / pathfinding / I/O details are internal. Nav and
        transport exceptions are caught and surfaced as ActionResult errors.
        """
        raw = self._tracker.build_snapshot()

        start = perf_counter()
        try:
            result = self._executor.execute(action, raw)
        except Exception as exc:
            duration = perf_counter() - start
            log.exception("BotCoreImpl.execute_action unexpected exception")
            # Even if executor exploded, we still try to trace the failure.
            fallback = ActionResult(
                success=False,
                error="botcore_execute_exception",
                details={
                    "exception": repr(exc),
                    "action_type": getattr(action, "type", None),
                },
            )
            try:
                self._tracer.record(
                    action=action,
                    snapshot=raw,
                    result=fallback,
                    duration_s=duration,
                )
            except Exception:
                # Tracing must never break the caller.
                log.exception("Action tracing failed after executor exception")
            return fallback

        duration = perf_counter() - start

        # Ensure we always have a proper ActionResult object.
        if not isinstance(result, ActionResult):
            result = ActionResult(
                success=False,
                error="invalid_executor_result",
                details={"result_repr": repr(result)},
            )

        # Record trace; errors here must be non-fatal.
        try:
            self._tracer.record(
                action=action,
                snapshot=raw,
                result=result,
                duration_s=duration,
            )
        except Exception:
            log.exception("Action tracing failed")

        return result

    # ------------------------------------------------------------------
    # Tracing access (optional helpers)
    # ------------------------------------------------------------------

    def get_action_traces(self) -> list[Any]:
        """
        Return a snapshot of recorded action traces.

        Primarily for debugging / monitoring tooling. The exact record type
        is ActionTraceRecord, but we keep the signature loose to avoid
        coupling callers to this module.
        """
        return self._tracer.get_records()

```




## 4. Timeouts & guards for “atomic-ish” actions

Design-wise you decided:

- One `execute_action` covers the whole operation (e.g. full `move_to`).
    

We have:

- `max_nav_steps` in `ActionExecutorConfig` for pathfinding.
    
- But **no runtime timeout** for long-running actions on the IPC side.
    

You probably want at least:

- A soft guard on **steps sent**:
    
    - `max_move_steps` beyond which you error with `"nav_too_long"` or similar.
        
- Config entries like:
    
    - `max_move_duration_ms`
        
    - `max_dig_duration_ms` (for later when you have async feedback).
        

This stops a bad nav situation or a laggy server from making `execute_action` feel hung.


```
# src/bot_core/actions.py
"""
Action execution for bot_core_1_7_10.

This module translates high-level Actions into protocol- or IPC-level
messages via a PacketClient.

Design constraints:
- High-level, atomic-ish actions:
    - move_to
    - break_block
    - place_block
    - use_item
    - interact
- One execute_action(...) call per logical operation.
- Explicit, structured failures:
    - no path
    - unsupported action
    - IO/IPC error
    - guards: nav_too_long, move_timeout
- No mutation of world state; this layer only sends packets/messages.
- No semantics, no tech_state, no virtues.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Dict, Mapping

from spec.types import Action, ActionResult  # from M1
from .snapshot import RawWorldSnapshot
from .net import PacketClient
from .nav import (
    NavGrid,
    BlockSolidFn,
    find_path,
    path_to_actions,
    current_coord_from_snapshot,
)
from .collision import (
    default_is_solid_block,
)


log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class ActionExecutorConfig:
    """
    Configuration knobs for ActionExecutor.

    These should be environment- and modpack-agnostic. GTNH-specific
    tuning belongs in higher layers or injected strategies.
    """

    # Limit on nav search complexity (A* internal steps)
    max_nav_steps: int = 2048

    # Default arrival radius for move_to
    default_move_radius: float = 0.5

    # Hard guard: maximum number of move steps we are willing to send
    # in a single move_to action. None disables this guard.
    max_move_steps: int | None = 10_000

    # Soft guard: maximum wall-clock time (seconds) we allow for sending
    # movement steps in a single execute_action("move_to", ...).
    # None disables this guard.
    max_move_duration_s: float | None = 30.0

    # Reserved for future use when we have async dig feedback.
    # For now, break_block is a fire-and-forget start+stop pair.
    max_dig_duration_s: float | None = 15.0


class ActionExecutor:
    """
    Translate high-level Actions into PacketClient messages.

    Public contract:
      execute(action, snapshot) -> ActionResult

    Responsibilities:
    - Build NavGrid from RawWorldSnapshot.
    - Run A* pathfinding for move_to.
    - Emit movement / dig / place / use / interact packets.
    - Surface explicit, structured errors.

    Non-responsibilities:
    - WorldState construction.
    - Tech semantics.
    - Planning, skill selection, or virtue scoring.
    """

    def __init__(
        self,
        client: PacketClient,
        *,
        is_solid_block: BlockSolidFn | None = None,
        config: ActionExecutorConfig | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self._client = client

        # Use injected collision logic if provided; otherwise fall back
        # to the module-level default profile.
        self._is_solid_block: BlockSolidFn = (
            is_solid_block if is_solid_block is not None else default_is_solid_block
        )

        self._cfg = config if config is not None else ActionExecutorConfig()
        self._log = logger or log

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def execute(self, action: Action, snapshot: RawWorldSnapshot) -> ActionResult:
        """
        Execute a single high-level Action against the given snapshot.

        The snapshot is treated as read-only and must not be mutated here.
        """
        atype = getattr(action, "type", None)
        params = getattr(action, "params", {}) or {}

        self._log.debug("ActionExecutor.execute start type=%s params=%r", atype, params)

        if not isinstance(params, Mapping):
            self._log.warning(
                "ActionExecutor.execute invalid params (not mapping): %r", params
            )
            return ActionResult(
                success=False,
                error="invalid_params",
                details={"reason": "params_not_mapping", "action_type": atype},
            )

        try:
            if atype == "move_to":
                result = self._execute_move_to(params, snapshot)
            elif atype == "break_block":
                result = self._execute_break_block(params, snapshot)
            elif atype == "place_block":
                result = self._execute_place_block(params, snapshot)
            elif atype == "use_item":
                result = self._execute_use_item(params, snapshot)
            elif atype == "interact":
                result = self._execute_interact(params, snapshot)
            else:
                result = ActionResult(
                    success=False,
                    error="unsupported_action",
                    details={"action_type": atype},
                )
        except Exception as exc:
            # Catch-all for transport / unexpected errors.
            self._log.exception(
                "ActionExecutor.execute raised unexpectedly for type=%s", atype
            )
            return ActionResult(
                success=False,
                error="execution_exception",
                details={
                    "action_type": atype,
                    "exception": repr(exc),
                },
            )

        self._log.debug(
            "ActionExecutor.execute end type=%s success=%s error=%s",
            atype,
            getattr(result, "success", None),
            getattr(result, "error", None),
        )
        return result

    # ------------------------------------------------------------------
    # Action handlers
    # ------------------------------------------------------------------

    def _execute_move_to(
        self,
        params: Mapping[str, Any],
        snapshot: RawWorldSnapshot,
    ) -> ActionResult:
        """
        Move the player to a target coordinate using NavGrid + A*.

        Expected params:
            - "x", "y", "z": numeric target position (block coords)
            - "radius": optional float radius (default from config)

        Guards:
            - max_nav_steps: bounds A* search complexity
            - max_move_steps: bounds number of steps we are willing to send
            - max_move_duration_s: bounds wall-clock sending duration
        """
        try:
            tx = int(params["x"])
            ty = int(params.get("y", snapshot.player_pos.get("y", 0.0)))
            tz = int(params["z"])
        except Exception:
            return ActionResult(
                success=False,
                error="invalid_params",
                details={"reason": "missing_or_non_numeric_xyz", "params": dict(params)},
            )

        radius = float(params.get("radius", self._cfg.default_move_radius))

        grid = NavGrid(snapshot=snapshot, is_solid_block=self._is_solid_block)

        start = current_coord_from_snapshot(snapshot)
        goal = (tx, ty, tz)

        pf_result = find_path(
            grid,
            start=start,
            goal=goal,
            max_steps=self._cfg.max_nav_steps,
        )

        if not pf_result.success or not pf_result.path:
            return ActionResult(
                success=False,
                error="nav_failure",
                details={
                    "from": start,
                    "to": goal,
                    "reason": pf_result.reason,
                },
            )

        move_actions = path_to_actions(
            pf_result,
            snapshot,
            radius=radius,
        )

        steps_planned = len(move_actions)
        max_steps_guard = self._cfg.max_move_steps

        if max_steps_guard is not None and steps_planned > max_steps_guard:
            # Path is conceptually valid but too long for one atomic action.
            return ActionResult(
                success=False,
                error="nav_too_long",
                details={
                    "from": start,
                    "to": goal,
                    "planned_steps": steps_planned,
                    "max_move_steps": max_steps_guard,
                },
            )

        # Emit movement packets for each step with a runtime guard.
        steps_sent = 0
        start_time = perf_counter()
        max_duration = self._cfg.max_move_duration_s

        for step in move_actions:
            # Check runtime timeout before sending the next step.
            if max_duration is not None:
                elapsed = perf_counter() - start_time
                if elapsed > max_duration:
                    return ActionResult(
                        success=False,
                        error="move_timeout",
                        details={
                            "from": start,
                            "to": goal,
                            "steps_attempted": steps_sent,
                            "planned_steps": steps_planned,
                            "max_move_duration_s": max_duration,
                            "elapsed_s": elapsed,
                        },
                    )

            try:
                self._send_move_step(step.params)
                steps_sent += 1
            except Exception as exc:
                return ActionResult(
                    success=False,
                    error="io_error",
                    details={
                        "stage": "move_to",
                        "from": start,
                        "to": goal,
                        "steps_attempted": steps_sent,
                        "planned_steps": steps_planned,
                        "exception": repr(exc),
                    },
                )

        return ActionResult(
            success=True,
            error=None,
            details={
                "from": start,
                "to": goal,
                "steps": steps_sent,
            },
        )

    def _execute_break_block(
        self,
        params: Mapping[str, Any],
        snapshot: RawWorldSnapshot,  # unused but kept for future context
    ) -> ActionResult:
        """
        Break a block at the given coordinates.

        Expected params:
            - "x", "y", "z": block coordinates
            - "face": optional face index or direction (depends on IPC schema)

        Note:
            For now this is synchronous fire-and-forget (start+stop). The
            max_dig_duration_s guard will become relevant once we have async
            feedback from the server / IPC layer.
        """
        try:
            x = int(params["x"])
            y = int(params["y"])
            z = int(params["z"])
        except Exception:
            return ActionResult(
                success=False,
                error="invalid_params",
                details={"reason": "missing_or_non_numeric_xyz", "params": dict(params)},
            )

        face = params.get("face", "auto")

        try:
            self._client.send_packet(
                "block_dig",
                {
                    "status": "start",
                    "x": x,
                    "y": y,
                    "z": z,
                    "face": face,
                },
            )
            self._client.send_packet(
                "block_dig",
                {
                    "status": "stop",
                    "x": x,
                    "y": y,
                    "z": z,
                    "face": face,
                },
            )
        except Exception as exc:
            return ActionResult(
                success=False,
                error="io_error",
                details={
                    "stage": "break_block",
                    "x": x,
                    "y": y,
                    "z": z,
                    "exception": repr(exc),
                },
            )

        return ActionResult(
            success=True,
            error=None,
            details={"x": x, "y": y, "z": z},
        )

    def _execute_place_block(
        self,
        params: Mapping[str, Any],
        snapshot: RawWorldSnapshot,  # unused but reserved for future checks
    ) -> ActionResult:
        """
        Place a block using the currently held item at the target coordinates.

        Expected params:
            - "x", "y", "z": target block position or adjacent position
            - "face": optional face direction
        """
        try:
            x = int(params["x"])
            y = int(params["y"])
            z = int(params["z"])
        except Exception:
            return ActionResult(
                success=False,
                error="invalid_params",
                details={"reason": "missing_or_non_numeric_xyz", "params": dict(params)},
            )

        face = params.get("face", "auto")

        try:
            self._client.send_packet(
                "block_place",
                {
                    "x": x,
                    "y": y,
                    "z": z,
                    "face": face,
                },
            )
        except Exception as exc:
            return ActionResult(
                success=False,
                error="io_error",
                details={
                    "stage": "place_block",
                    "x": x,
                    "y": y,
                    "z": z,
                    "exception": repr(exc),
                },
            )

        return ActionResult(
            success=True,
            error=None,
            details={"x": x, "y": y, "z": z},
        )

    def _execute_use_item(
        self,
        params: Mapping[str, Any],
        snapshot: RawWorldSnapshot,  # reserved for later (line of sight, etc.)
    ) -> ActionResult:
        """
        Use the currently held item (e.g., right-click in air or on a block).

        Expected params (schema is flexible and IPC-defined):
            - "target": one of {"self", "block", "entity", "air"} (optional)
            - "x", "y", "z": optional coordinates (for block/entity)
            - any other keys are passed through to IPC layer.
        """
        payload: Dict[str, Any] = {"target": params.get("target", "air")}
        for key in ("x", "y", "z", "face", "entity_id"):
            if key in params:
                payload[key] = params[key]

        for k, v in params.items():
            if k not in payload:
                payload[k] = v

        try:
            self._client.send_packet("use_item", payload)
        except Exception as exc:
            return ActionResult(
                success=False,
                error="io_error",
                details={
                    "stage": "use_item",
                    "payload": payload,
                    "exception": repr(exc),
                },
            )

        return ActionResult(
            success=True,
            error=None,
            details=payload,
        )

    def _execute_interact(
        self,
        params: Mapping[str, Any],
        snapshot: RawWorldSnapshot,  # reserved (e.g., for range checks)
    ) -> ActionResult:
        """
        Interact with an entity or block.

        Expected params:
            - "kind": "entity" or "block"
            - "entity_id": int if kind == "entity"
            - "x", "y", "z": coords if kind == "block"
        """
        kind = params.get("kind")

        if kind not in ("entity", "block"):
            return ActionResult(
                success=False,
                error="invalid_params",
                details={
                    "reason": "kind_must_be_entity_or_block",
                    "params": dict(params),
                },
            )

        payload: Dict[str, Any] = {"kind": kind}

        if kind == "entity":
            try:
                payload["entity_id"] = int(params["entity_id"])
            except Exception:
                return ActionResult(
                    success=False,
                    error="invalid_params",
                    details={
                        "reason": "missing_or_non_numeric_entity_id",
                        "params": dict(params),
                    },
                )
        else:  # kind == "block"
            try:
                payload["x"] = int(params["x"])
                payload["y"] = int(params["y"])
                payload["z"] = int(params["z"])
            except Exception:
                return ActionResult(
                    success=False,
                    error="invalid_params",
                    details={
                        "reason": "missing_or_non_numeric_xyz",
                        "params": dict(params),
                    },
                )

        for k, v in params.items():
            if k not in payload:
                payload[k] = v

        try:
            self._client.send_packet("interact", payload)
        except Exception as exc:
            return ActionResult(
                success=False,
                error="io_error",
                details={
                    "stage": "interact",
                    "payload": payload,
                    "exception": repr(exc),
                },
            )

        return ActionResult(
            success=True,
            error=None,
            details=payload,
        )

    # ------------------------------------------------------------------
    # Low-level emitters
    # ------------------------------------------------------------------

    def _send_move_step(self, step_params: Mapping[str, Any]) -> None:
        """
        Emit a single movement step.

        For IPC mode, you can interpret this generically:
          - type: "move_step"
          - payload: {x, y, z, radius}
        """
        x = step_params.get("x")
        y = step_params.get("y")
        z = step_params.get("z")
        radius = step_params.get("radius")

        payload = {
            "x": x,
            "y": y,
            "z": z,
            "radius": radius,
        }

        self._client.send_packet("move_step", payload)

```


## 5. A minimal harness script

You have tests. Great. You still want a quick “does it breathe” harness:

Something like `tools/smoke_botcore.py`:

- Use `FakePacketClient` at first:
    
    - Emit a few packets
        
    - Call `bot.observe()`, `get_world_state()`, `execute_action()`
        
    - Pretty-print `WorldState` and `ActionResult`
        
- Later, flip to real IPC client in `env.yaml`:
    
    - Point at running Forge mod
        
    - Try:
        
        - one `move_to` nearby
            
        - one `break_block` near feet
            

This gives you a **manual smoke test** without touching higher modules.

You want a “does this thing even have a pulse” script. Good. Sanity checks are how you avoid 3-hour debugging sessions later.

Here’s a **minimal harness** at `tools/smoke_botcore.py` that:

- Uses `FakePacketClient` by default
    
- Emits a couple of fake packets (`time_update`, `position_update`)
    
- Calls:
    
    - `bot.observe()`
        
    - `bot.get_world_state()`
        
    - `bot.execute_action(move_to)`
        
    - `bot.execute_action(break_block)`
        
- Pretty-prints `WorldState`, `ActionResult`, and the captured packets
    

It also has a `--mode real` flag stubbed for when you hook up the actual Forge IPC.

```
#!/usr/bin/env python3
"""
tools/smoke_botcore.py

Minimal harness to sanity-check BotCoreImpl wiring.

Default mode:
    - Uses FakePacketClient (no real network)
    - Emits a few synthetic packets
    - Calls:
        - observe()
        - get_world_state()
        - execute_action(move_to)
        - execute_action(break_block)
    - Prints WorldState, ActionResult, and sent packets

Later:
    - Use --mode real to use the real PacketClient from env.yaml
      (once the Forge IPC side exists).
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Ensure src/ is on sys.path when running as a script
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# ---------------------------------------------------------------------------
# Imports from the project
# ---------------------------------------------------------------------------

from spec.types import Action, WorldState, ActionResult  # type: ignore[import]
from bot_core import BotCoreImpl  # type: ignore[import]
from bot_core.snapshot import RawWorldSnapshot  # type: ignore[import]
from bot_core.testing.fakes import FakePacketClient  # type: ignore[import]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_dict(obj: Any) -> Any:
    """Best-effort conversion of dataclasses to plain dicts for printing."""
    if is_dataclass(obj):
        return asdict(obj)
    return obj


def _print_header(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def run_fake_mode() -> None:
    """
    Run a pure in-memory smoke test with FakePacketClient.

    This does NOT require a running Minecraft or Forge mod.
    """
    _print_header("Fake mode: initializing BotCoreImpl with FakePacketClient")

    client = FakePacketClient()
    bot = BotCoreImpl(client=client)

    # Connect (FakePacketClient.connect is a no-op except for state flag)
    bot.connect()

    # Emit some basic world packets into the WorldTracker via FakePacketClient
    client.emit("time_update", {"tick": 42})
    client.emit(
        "position_update",
        {
            "x": 1.5,
            "y": 64.0,
            "z": -2.0,
            "yaw": 90.0,
            "pitch": 10.0,
            "on_ground": True,
        },
    )

    # One tick() call to match the usual flow; FakePacketClient.tick is a no-op.
    bot.tick()

    # ------------------------------------------------------------------
    # Observe raw + semantic world state
    # ------------------------------------------------------------------
    _print_header("RawWorldSnapshot (observe)")
    raw: RawWorldSnapshot = bot.observe()
    print(_to_dict(raw))

    _print_header("WorldState (get_world_state)")
    world: WorldState = bot.get_world_state()
    print(_to_dict(world))

    # ------------------------------------------------------------------
    # Execute a move_to action
    # ------------------------------------------------------------------
    _print_header("execute_action: move_to {x=3,y=64,z=-2}")

    move_action = Action(
        type="move_to",
        params={"x": 3, "y": 64, "z": -2, "radius": 0.5},
    )

    move_result: ActionResult = bot.execute_action(move_action)
    print("ActionResult:", _to_dict(move_result))

    print("\nSent packets (after move_to):")
    for p in client.sent_packets:
        print(f"  - {p.packet_type}: {p.data}")

    # Clear sent packets so we can see only the break_block effect next.
    client.sent_packets.clear()

    # ------------------------------------------------------------------
    # Execute a break_block action
    # ------------------------------------------------------------------
    _print_header("execute_action: break_block {x=1,y=64,z=-2}")

    break_action = Action(
        type="break_block",
        params={"x": 1, "y": 64, "z": -2},
    )

    break_result: ActionResult = bot.execute_action(break_action)
    print("ActionResult:", _to_dict(break_result))

    print("\nSent packets (after break_block):")
    for p in client.sent_packets:
        print(f"  - {p.packet_type}: {p.data}")

    _print_header("Fake mode completed")


def run_real_mode() -> None:
    """
    Placeholder for future real IPC smoke test.

    This expects:
        - env.yaml to be configured with bot_mode: forge_mod or similar
        - create_packet_client_for_env() to construct a working transport
        - A running Forge mod IPC server

    For now this just wires BotCoreImpl with its default client and
    attempts a minimal call sequence.
    """
    _print_header("Real mode: initializing BotCoreImpl with real PacketClient")

    bot = BotCoreImpl()  # uses real client from env.yaml

    try:
        bot.connect()
    except Exception as exc:  # BotCoreError or lower-level error
        print("connect() failed:", repr(exc))
        return

    try:
        # Give the transport a few ticks to populate basic state.
        for _ in range(3):
            bot.tick()

        _print_header("WorldState (get_world_state)")
        world = bot.get_world_state()
        print(_to_dict(world))

        # Try a small move near the current position.
        pos = world.position  # type: ignore[attr-defined]
        move_action = Action(
            type="move_to",
            params={
                "x": int(pos.get("x", 0)) + 1,
                "y": int(pos.get("y", 64)),
                "z": int(pos.get("z", 0)),
                "radius": 0.5,
            },
        )

        _print_header("execute_action (real): move_to one block over")
        result = bot.execute_action(move_action)
        print("ActionResult:", _to_dict(result))

    finally:
        try:
            bot.disconnect()
        except Exception as exc:
            print("disconnect() failed:", repr(exc))

    _print_header("Real mode completed")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Smoke test harness for bot_core BotCoreImpl",
    )
    parser.add_argument(
        "--mode",
        choices=["fake", "real"],
        default="fake",
        help="Run in 'fake' (no network) or 'real' (env.yaml PacketClient) mode",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR)",
    )

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    if args.mode == "fake":
        run_fake_mode()
    else:
        run_real_mode()


if __name__ == "__main__":
    main()

```

### How to use it

From repo root:
bash:
```
python3 tools/smoke_botcore.py --mode fake --log-level DEBUG

```

You should see:

- A printed `RawWorldSnapshot`
    
- A printed `WorldState`
    
- An `ActionResult` for `move_to`
    
- The `move_step` packets logged from `FakePacketClient`
    
- An `ActionResult` for `break_block`
    
- The `block_dig` packets
    

Once the Forge mod IPC exists and `env.yaml` is wired:
bash:
```
python3 tools/smoke_botcore.py --mode real

```
At that point, if something breaks, it’s not because “bot_core doesn’t breathe,” it’s because the server side or IPC contract is wrong. Which is exactly the kind of isolation you want.

---

## 6. Documentation for future-you

You don’t need a book, but you do need a **single source of truth**:

`docs/m6_bot_core_1_7_10.md` (or update your existing module doc) should include:

- Public API:
    
    - `BotCoreImpl` methods & types
        
- What’s guaranteed:
    
    - `ActionResult` shape
        
    - Error codes (`nav_failure`, `io_error`, `unsupported_action`, etc.)
        
- Integration hooks:
    
    - Where to inject `is_solid_block`
        
    - Where IPC events/packets are normalized
        

This prevents M8/M10 from guessing what M6 does based on vibes.

---

## 7. When is M6 “done enough” to move on?

Given where you are:

- ✅ Code structure complete
    
- ✅ Unit & integration tests passing
    
- ✅ Clear separation: no semantics, no LLM, no planner leakage
    

M6 is “done enough” when you add:

1. **IPC protocol doc** + agreement on message shapes
    
2. **Non-trivial `is_solid_block`** wired somewhere (even if crude)
    
3. **Basic logging around actions**
    
4. **One tiny harness script** that lets you poke the body from a REPL
    

After that, the next actual progress comes from **M3 + Forge mod IPC**, not more polishing in M6.

So: nice work. The body’s here. Now you have to actually animate it.


## Fake botcore implementation (Current progress - M8 complete)

## Concrete plan (without writing code yet)

What we’ll do _when you’re ready to wire it_:

1. Add `src/bot_core/runtime.py` with:
    
    - `get_bot_core(env_profile_name: Optional[str] = None) -> BotCore`
        
2. Implement a **FakeBotCore** there that:
    
    - Implements `spec.bot_core.BotCore`
        
    - Returns a `WorldState` in `get_world_state()`
        
    - Maybe no-op or record-only in `execute_action()`
        
3. Make sure `test_m6_observe_contract.py` runs and passes against this fake.
    

That keeps everything consistent:

- M6 is genuinely “functionally complete.”
    
- M8 has a real downstream of `AgentRuntime → BotCore`, even if it’s fake.
    
- You still don’t have to spin up Minecraft.