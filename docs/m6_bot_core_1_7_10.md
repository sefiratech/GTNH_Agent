# M6 — bot_core_1_7_10

**Module role:**  
Provide a clean, testable “body controller” for Minecraft 1.7.10 (GTNH):

- Connects to the game via a transport (`PacketClient` / Forge IPC)
- Tracks raw world state (`WorldTracker` → `RawWorldSnapshot`)
- Adapts to semantic state (`WorldState`)
- Executes high-level actions (`Action`) via `ActionExecutor`
- Emits structured traces for monitoring and experience (`ActionTracer`)

This module is **intentionally thin**:

- No GTNH tech semantics
- No virtues / curriculum
- No LLM logic
- No planning

It is a stable substrate M8+ can rely on.

---

## 1. High-level Architecture

Core components:

- `BotCoreImpl`
- `PacketClient` (transport, injected or built from `env.yaml`)
- `WorldTracker` (incremental raw world state)
- `RawWorldSnapshot` / `snapshot_to_world_state`
- `ActionExecutor`
- `ActionTracer`
- `BlockCollisionProfile` / `is_solid_block` hook

```text
env.yaml → EnvProfile ┐
                       │
                 create_packet_client_for_env()
                       │
                  PacketClient  <── Forge IPC / protocol client
                       │
         ┌─────────────┴─────────────┐
         │                           │
   WorldTracker                 ActionExecutor
         │                           │
  RawWorldSnapshot            send_packet(...)
         │
snapshot_to_world_state
         │
     WorldState

BotCoreImpl orchestrates all of this and exposes the only public API higher layers should touch.

2. Public API (BotCoreImpl)

Location: src/bot_core/core.py
Interface: implements spec.bot_core.BotCore

2.1 Construction
from bot_core import BotCoreImpl
from bot_core.actions import ActionExecutorConfig
from bot_core.collision import default_is_solid_block
from bot_core.tracing import ActionTracer

bot = BotCoreImpl(
    client=None,                       # optional PacketClient
    action_config=ActionExecutorConfig(),
    is_solid_block=None,              # optional BlockSolidFn
    tracer=None,                      # optional ActionTracer
)


client

If None: constructed via create_packet_client_for_env() using env.yaml (M0/M1).

action_config

Tuning knobs for navigation / execution guards.

is_solid_block

Optional collision override (BlockSolidFn). If not provided, uses default_is_solid_block from collision.py.

tracer

Optional ActionTracer for metrics; default instance used if omitted.

2.2 Lifecycle
bot.connect()      # establishes connection to world
bot.tick()         # pumps PacketClient, updates WorldTracker
bot.disconnect()   # closes connection


Errors:

Non-action failures raise BotCoreError(code, details):

"connect_failed"

"disconnect_failed"

"tick_failed"

2.3 Observation
from bot_core.snapshot import RawWorldSnapshot
from spec.types import WorldState

raw: RawWorldSnapshot = bot.observe()
ws: WorldState = bot.get_world_state()


Guarantees:

observe() returns a fresh RawWorldSnapshot built by WorldTracker.

get_world_state() calls snapshot_to_world_state(raw) and returns a semantic WorldState:

tick: int

position: {x, y, z}

dimension: str

inventory: [...]

nearby_entities: [...]

blocks_of_interest: [...]

tech_state: dict

context: dict

WorldState is defined in spec.types and owned by M1.

2.4 Action execution
from spec.types import Action, ActionResult

action = Action(type="move_to", params={"x": 3, "y": 64, "z": -2})
result: ActionResult = bot.execute_action(action)


Contract:

execute_action:

Captures a RawWorldSnapshot internally

Delegates to ActionExecutor.execute(action, snapshot)

Records a trace via ActionTracer

Returns a valid ActionResult in all cases (no exceptions leak out from navigation or I/O)

3. ActionResult Shape & Error Codes

Type: spec.types.ActionResult

Typical shape:

ActionResult(
    success: bool,
    error: str | None,
    details: dict[str, Any],
)

3.1 Common error codes

These are produced by ActionExecutor and surfaced through BotCoreImpl:

"invalid_params"

Params missing or wrong type (e.g. missing x/y/z).

"unsupported_action"

Action.type not recognized by ActionExecutor.

"nav_failure"

Pathfinding (find_path) returned failure:

No path found

max_nav_steps exhausted

"nav_too_long"

Path is conceptually valid but exceeds max_move_steps.

"move_timeout"

Runtime guard: sending move_step packets exceeded max_move_duration_s.

"io_error"

PacketClient.send_packet raised during an action.

"execution_exception"

Unexpected exception inside ActionExecutor.execute.

"botcore_execute_exception"

Unexpected exception in BotCoreImpl.execute_action wrapper.

"invalid_executor_result"

ActionExecutor.execute returned a non-ActionResult object (should not happen, but guarded).

Action-level errors do not raise; they are always encoded in ActionResult.

3.2 Success case

On success:

ActionResult(
    success=True,
    error=None,
    details={...},
)


Current details conventions:

move_to:

details = {
    "from": (sx, sy, sz),
    "to":   (tx, ty, tz),
    "steps": int,  # number of move_step packets sent
}


break_block:

details = {"x": int, "y": int, "z": int}


place_block:

details = {"x": int, "y": int, "z": int}


use_item / interact:

details = payload_sent_to_packet_client

4. ActionExecutor Semantics

Location: src/bot_core/actions.py
Constructor:

from bot_core.actions import ActionExecutor, ActionExecutorConfig
from bot_core.collision import default_is_solid_block

executor = ActionExecutor(
    client=packet_client,
    is_solid_block=None,      # or custom BlockSolidFn
    config=ActionExecutorConfig(...),
)

4.1 Config
@dataclass
class ActionExecutorConfig:
    max_nav_steps: int = 2048
    default_move_radius: float = 0.5
    max_move_steps: int | None = 10_000
    max_move_duration_s: float | None = 30.0
    max_dig_duration_s: float | None = 15.0  # reserved for future async dig
)


max_nav_steps

Bounds A* search complexity inside find_path.

default_move_radius

Default arrival radius if move_to.params["radius"] is missing.

max_move_steps

Upper bound on the number of move_step packets allowed for one move_to.

If exceeded, returns error="nav_too_long".

max_move_duration_s

Max wall-clock time allowed for sending movement steps.

If exceeded, returns error="move_timeout".

max_dig_duration_s

Placeholder for future async dig logic; currently not enforced (break_block is synchronous start+stop).

4.2 Supported actions

Supported Action.type values:

"move_to"

"break_block"

"place_block"

"use_item"

"interact"

Anything else → error="unsupported_action".

move_to

Builds NavGrid(snapshot, is_solid_block)

start = current_coord_from_snapshot(snapshot)

goal = (x, y, z) from params

Calls find_path(grid, start, goal, max_steps=max_nav_steps)

On success:

Converts path to low-level steps via path_to_actions

Sends one "move_step" packet per step:

payload = {"x": x, "y": y, "z": z, "radius": radius}


Guards:

max_nav_steps → handled inside pathfinding

max_move_steps → "nav_too_long"

max_move_duration_s → "move_timeout"

break_block

Sends:

"block_dig" with {status: "start", x, y, z, face}

"block_dig" with {status: "stop", x, y, z, face}

place_block

Sends:

"block_place" with {x, y, z, face}

use_item

Sends:

"use_item" with a flexible payload:

"target" ("air" by default)

Optional x, y, z, face, entity_id

Any extra fields passed through.

interact

Sends:

"interact" with:

For kind="entity": {kind: "entity", entity_id: int, ...}

For kind="block": {kind: "block", x, y, z, ...}

5. Collision & Navigation Integration

Location: src/bot_core/collision.py, src/bot_core/nav/…

5.1 BlockSolidFn

Signature:

BlockSolidFn = Callable[[int, int, int, RawWorldSnapshot], bool]


Used by:

NavGrid(snapshot, is_solid_block)

ActionExecutor / BotCoreImpl via injection

5.2 Default collision behavior

collision.py provides:

from bot_core.collision import (
    BlockCollisionProfile,
    default_block_collision_profile,
    default_is_solid_block,
)


Current default:

BlockCollisionProfile with:

block_at=None

default_floor_y=63

default_is_solid_block(x, y, z, snapshot):

Treats everything at y <= 63 as solid (flat floor)

Everything above as non-solid

This is a temporary stand-in until M3 provides a real block_at(snapshot, x, y, z) backed by GTNH block data.

5.3 Injecting real collision semantics

Options:

At BotCoreImpl construction time:

def gtnh_is_solid_block(x, y, z, snapshot: RawWorldSnapshot) -> bool:
    # Look up chunk, block_id, material, etc.
    ...

bot = BotCoreImpl(is_solid_block=gtnh_is_solid_block)


By replacing the default profile:

Implement BlockCollisionProfile(block_at=...)

Call that from default_is_solid_block (internal evolution later)

M6 itself does not know GTNH semantics; it just exposes the hook.

6. IPC / Packet Normalization

Transport interface: PacketClient (in src/bot_core/net.py)

class PacketClient(Protocol):
    def connect(self) -> None: ...
    def disconnect(self) -> None: ...
    def tick(self) -> None: ...
    def send_packet(self, packet_type: str, data: Mapping[str, Any]) -> None: ...
    def on_packet(self, packet_type: str, handler: PacketHandler) -> None: ...

6.1 Expected incoming events

The Forge mod / external client must normalize raw protocol packets into these event types for WorldTracker:

"time_update" → {"tick": int}

"position_update" → {"x", "y", "z", "yaw", "pitch", "on_ground"}

"chunk_data" → {"chunk_x", "chunk_z", "blocks", "biomes"}

"spawn_entity" → {"entity_id", "kind", "x", "y", "z", ...}

"destroy_entities" → {"entity_ids": [int, ...]}

"window_items" → {"items": [...]} (full inventory snapshot)

"set_slot" → {"slot": int, "item": {...}}

(Plus optional events like "dimension_change" if implemented)

These are wired via:

client.on_packet("time_update", tracker.handle_time_update)
...

6.2 Expected outgoing actions

ActionExecutor sends high-level packet types via PacketClient.send_packet:

"move_step" → {x, y, z, radius}

"block_dig" → {status: "start" | "stop", x, y, z, face}

"block_place" → {x, y, z, face}

"use_item" → {target, x?, y?, z?, face?, entity_id?, ...}

"interact" → {kind: "entity" | "block", entity_id? | x/y/z, ...}

The Forge mod / protocol client is responsible for translating these into concrete Minecraft 1.7.10 packets.

7. Tracing & Metrics

Location: src/bot_core/tracing.py

7.1 ActionTracer
from bot_core.tracing import ActionTracer

tracer = ActionTracer(max_records=10_000)
bot = BotCoreImpl(tracer=tracer)


Each execute_action:

Measures duration (perf_counter)

Builds an ActionTraceRecord:

@dataclass
class ActionTraceRecord:
    timestamp: float
    duration_s: float

    action_type: str | None
    params: dict[str, Any]

    success: bool
    error: str | None

    tick: int | None
    dimension: str | None
    position: dict[str, float]  # {x, y, z}


Appends to an in-memory ring buffer (deque)

Logs a single line via logger "bot_core.action":

Example:

INFO bot_core.action action_exec type=move_to success=True error=None \
duration=0.0002s tick=42 dim=overworld pos=(1.50,64.00,-2.00)


Access:

records = bot.get_action_traces()  # list[ActionTraceRecord]


M9 and M10 can later consume this for monitoring / experience replay.

8. Testing & Smoke Harness
8.1 Unit & integration tests

Key test files:

tests/test_world_tracker.py

Validates packet handlers → RawWorldSnapshot

tests/test_nav_pathfinder.py

A* behavior and obstacle avoidance with synthetic solidity functions

tests/test_actions.py

Ensures move_to emits correct move_step packets

Ensures break_block / place_block send correct packets

tests/test_bot_core_impl.py

End-to-end with FakePacketClient

observe() / get_world_state()

execute_action() → packet log

Test helper:

src/bot_core/testing/fakes.py → FakePacketClient

8.2 Smoke harness

Location: tools/smoke_botcore.py

Usage (fake mode, no Minecraft required):

python3 tools/smoke_botcore.py --mode fake --log-level DEBUG


Does:

Initialize BotCoreImpl with FakePacketClient

Emit time_update + position_update

Print:

RawWorldSnapshot

WorldState

ActionResult for:

move_to

break_block

Captured packets

Real mode stub (for future IPC):

python3 tools/smoke_botcore.py --mode real


Assumes:

env.yaml configured for real IPC

Forge mod / external client running

9. What M6 Guarantees to Higher Modules

For M8, M9, M10, M11, etc., you can treat M6 as guaranteeing:

Stable APIs

BotCoreImpl methods:

connect(), disconnect(), tick()

observe() -> RawWorldSnapshot

get_world_state() -> WorldState

execute_action(Action) -> ActionResult

No protocol leakage

No raw packets, no protocol IDs, no IPC details visible.

No uncaught nav / I/O exceptions

Action-level failures always encoded in ActionResult.

Transport failures as BotCoreError.

Configurable collision hook

GTNH semantics can be injected without rewriting M6.

Traceability

Every action has:

Type

Params

Result

Duration

Minimal world context (tick, dim, position)

This is enough for:

Planners to reason safely about ActionResult.error

Monitoring to inspect recent actions

Experience / replay systems to reconstruct behavior without spelunking into packets.

M6 is the “body” layer. Everything smarter builds on this, not around it.
