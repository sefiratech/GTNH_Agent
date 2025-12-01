## 0. Big Picture: How the phases talk

### Phase 0 – Foundations (M0, M1)

- **M0: environment_foundation**
    
    - Loads & validates:
        
        - `env.yaml`, `hardware.yaml`, `minecraft.yaml`, `models.yaml`
            
    - Produces:
        
        - `EnvProfile` (runtime profile: paths, ports, model choices, timeouts, etc.)
            
- **M1: tooling & validation**
    
    - CLI / scripts like `tools/validate_env.py`
        
    - Early tests to ensure configs + filesystem are sane before anything fancy.
        

> Output of Phase 0: a **validated environment profile** & config objects everything else leans on.

---

### Phase 1 – Cognition & Knowledge (M2–M5)

- **M2: LLM Stack**
    
    - Builds:
        
        - LLM clients (planner, critic, scribe, error-model, etc.)
            
        - Role presets from `llm_roles.yaml` & `models.yaml`
            
    - Consumes:
        
        - `EnvProfile` (model paths, n_ctx, hardware limits) from M0
            
    - Emits:
        
        - LLM-span logs → M9 (M2 → M9 path via JSON logs / log_event)
            
- **M3: World Semantics (GTNH)**
    
    - Loads:
        
        - `gtnh_items.yaml / .generated`
            
        - `gtnh_blocks.yaml / .generated`
            
        - `gtnh_recipes*.json`
            
        - `gtnh_tech_graph.yaml`
            
    - Provides:
        
        - `TechState` inference (from player/base info)
            
        - Semantic lookup helpers (`describe_item`, `missing_prereqs_for(tech)`, etc.)
            
- **M4: Virtue Lattice**
    
    - Defines:
        
        - Virtue graph / weights (Kabbalah-esque virtue lattice)
            
        - Scoring rules over traces / plans
            
    - Consumes:
        
        - Trace / plan summaries from M8
            
    - Emits:
        
        - `VIRTUE_SCORES` events via M9
            
- **M5: Skills & Curriculum**
    
    - Loads:
        
        - `config/curricula/*.yaml`
            
        - Skill definitions & goal templates
            
    - Provides:
        
        - Skill registry for planner
            
        - Curriculum slices (e.g., “early LV progression,” “eco factory,” etc.)
            

> Output of Phase 1: **thinking stack**  
> LLMs + GTNH semantics + ethics + skills / curricula.

---

### Phase 2 – Body & Perception (M6, M7)

- **M6: BotCore**
    
    - Talks to Minecraft:
        
        - Observations → `RawWorldSnapshot` / `WorldState`
            
        - Actions → movement, interactions, inventory, etc.
            
    - Consumes:
        
        - `EnvProfile.minecraft` (host, port, protocol) from M0
            
    - Emits:
        
        - Low-level action results & observation metadata → M9 via `emit_action_executed_from_botcore`, `emit_observation_metadata`.
            
- **M7: Observation Encoding**
    
    - Consumes:
        
        - `WorldState` from M6
            
        - Semantics (M3)
            
        - TechState (M3)
            
    - Produces:
        
        - Planner-ready JSON payloads
            
        - Critic & virtue payloads
            
    - Emits:
        
        - `SNAPSHOT` events for planner payloads / trace summaries via `emit_planner_observation_snapshot`, `emit_trace_structure_snapshot`.
            

> Output of Phase 2: **senses & body**  
> A structured view of the world and a way to act on it.

---

### Phase 3 – Agency & Monitoring (M8, M9)

- **M8: AgentLoop**
    
    - Orchestrates:
        
        - `observe → plan → act → critique → log experience`
            
    - Consumes:
        
        - LLM stack (M2)
            
        - Semantics & TechState (M3)
            
        - Virtue engine (M4)
            
        - Skills & curricula (M5)
            
        - BotCore (M6)
            
        - Observation encoder (M7)
            
        - Monitoring bus (M9)
            
    - Emits:
        
        - Phase changes, plan lifecycle, critic results, virtue scores, experience markers → via `monitoring.integration` helpers.
            
- **M9: monitoring_and_tools**
    
    - `EventBus` & `MonitoringEvent` schema
        
    - `JsonFileLogger` → `logs/monitoring/events.log`
        
    - `AgentController` → pause/resume/step/goal/cancel/dump-state
        
    - `TuiDashboard` → HUD
        
    - `tools.py` → CLI, episode inspector, LLM log viewer
        

> Output of Phase 3: **agent with a HUD and remote control**.

---

## 1. Data Flow Across Phases

In a single high-level loop:

1. **Phase 0**: Load & validate configs → produce `EnvProfile`.
    
2. **Phase 1**:
    
    - Build LLM clients / presets from env+models.
        
    - Build semantics (items / tech graph), virtues, and skills.
        
3. **Phase 2**:
    
    - Spin up BotCore (connected to Minecraft).
        
    - Build observation encoder referencing semantics & TechState.
        
4. **Phase 3**:
    
    - Build monitoring stack (bus + logger + controller + TUI).
        
    - Build AgentLoop wired to:
        
        - M2 (planner/critic/etc)
            
        - M3 (semantics/TechState)
            
        - M4 (virtue engine)
            
        - M5 (skills/curriculum)
            
        - M6 (BotCore)
            
        - M7 (observation encoder)
            
        - M9 (event bus + logging + HUD)
            
    - Main loop: `safe_step_with_logging(controller, bus, episode_id, context_id)`.
        

---

## 2. Unified Bootstrap Script (Phases 0–3)

This is a **conceptual integration script** that shows the wiring across all phases. It’s not supposed to import your exact module paths perfectly; it’s the “how the pieces should talk” reference.

You can adapt this into `src/runtime/bootstrap_phases.py` or fold pieces into your existing runtime.

```python
# path: src/runtime/bootstrap_phases.py

"""
Bootstrap the full GTNH_Agent stack across Phases 0–3.

Phases:
- Phase 0: M0 (environment), M1 (validation/tools)
- Phase 1: M2 (LLM stack), M3 (semantics), M4 (virtues), M5 (skills)
- Phase 2: M6 (BotCore), M7 (observation encoding)
- Phase 3: M8 (AgentLoop), M9 (monitoring & tools)
"""

from __future__ import annotations  # forward references in type hints

import threading                    # for TUI thread
import time                         # for main loop pacing
from pathlib import Path            # for config & log paths
from typing import Tuple            # for clearer return annotations

# ---- Phase 0 imports (environment + validation) ------------------------------

from env_loader.core import load_env_profile         # M0: load + validate env.yaml/hardware.yaml/etc.
from tools.validate_env import validate_environment  # M1: sanity checks on paths / versions

# ---- Phase 1 imports (LLM stack, semantics, virtues, skills) -----------------

from llm_stack.builder import build_llm_stack        # M2: planner/critic/scribe/error-model clients
from semantics.tech_state import (                   # M3: semantics + tech graph
    load_semantic_models,
    TechStateEngine,
)
from virtues.engine import VirtueEngine              # M4: virtue lattice + scoring rules
from skills.registry import SkillRegistry            # M5: skill definitions + curriculum link

# ---- Phase 2 imports (BotCore + observation encoding) ------------------------

from bot_core.core import BotCore                    # M6: Minecraft IO & action engine
from observation.encoder import ObservationEncoder   # M7: planner/critic payload builder

# ---- Phase 3 imports (AgentLoop + monitoring) --------------------------------

from monitoring.bus import EventBus, default_bus     # M9: event bus
from monitoring.logger import JsonFileLogger         # M9: JSONL logger
from monitoring.controller import AgentController    # M9: control surface
from monitoring.dashboard_tui import TuiDashboard    # M9: HUD
from monitoring import integration as mon_int        # M9: integration helpers (emit_*)

from agent_loop.loop import AgentLoopV1              # M8: core agent loop

from runtime.error_handling import safe_step_with_logging  # runtime helper for step exceptions


# ------------------------------------------------------------------------------
# Phase 0 – environment & validation
# ------------------------------------------------------------------------------

def bootstrap_phase0(config_root: Path) -> "EnvProfile":
    """
    Phase 0:
    - Load environment configs (env.yaml, hardware.yaml, minecraft.yaml, models.yaml).
    - Perform validation to ensure everything is runnable before continuing.
    """
    env_profile = load_env_profile(config_root=config_root)
    validate_environment(env_profile)
    return env_profile


# ------------------------------------------------------------------------------
# Phase 1 – LLM stack, semantics, virtues, skills
# ------------------------------------------------------------------------------

def bootstrap_phase1(
    env_profile: "EnvProfile",
) -> Tuple["LlmStack", TechStateEngine, VirtueEngine, SkillRegistry]:
    """
    Phase 1:
    - Build LLM stack (planner, critic, scribe, etc.) using env/model config.
    - Load GTNH semantics & tech graph.
    - Initialize virtue engine with lattice + scoring rules.
    - Initialize skills registry + curricula.
    """
    # Build the LLM stack (M2) based on env + models configuration
    llm_stack = build_llm_stack(env_profile=env_profile)

    # Load semantics (M3): items, blocks, recipes, tech graph, etc.
    semantics_models = load_semantic_models(config_root=env_profile.config_root)

    # Create a TechStateEngine which can infer current tech tier / missing unlocks
    tech_state_engine = TechStateEngine(semantics_models=semantics_models)

    # Initialize virtue engine (M4) with virtue lattice & rules
    virtue_engine = VirtueEngine(config_root=env_profile.config_root)

    # Build skills registry (M5) from curricula + skill definitions
    skill_registry = SkillRegistry(config_root=env_profile.config_root)

    return llm_stack, tech_state_engine, virtue_engine, skill_registry


# ------------------------------------------------------------------------------
# Phase 2 – BotCore + observation encoding
# ------------------------------------------------------------------------------

def bootstrap_phase2(
    env_profile: "EnvProfile",
    tech_state_engine: TechStateEngine,
) -> Tuple[BotCore, ObservationEncoder]:
    """
    Phase 2:
    - Create BotCore (M6) wired to the Minecraft instance.
    - Create an ObservationEncoder (M7) that converts world state into planner/critic payloads.
    """
    bot_core = BotCore(
        mc_host=env_profile.minecraft.host,
        mc_port=env_profile.minecraft.port,
        profile=env_profile.minecraft,
    )

    obs_encoder = ObservationEncoder(
        tech_state_engine=tech_state_engine,
        semantics_models=tech_state_engine.semantics_models,
    )

    return bot_core, obs_encoder


# ------------------------------------------------------------------------------
# Phase 3 – monitoring stack (M9) and AgentLoop (M8)
# ------------------------------------------------------------------------------

def build_monitoring_stack(
    use_default_bus: bool = True,
    log_path: Path | None = None,
) -> Tuple[EventBus, JsonFileLogger, TuiDashboard]:
    """
    Phase 3 (M9):
    - Build EventBus and JsonFileLogger.
    - Optionally reuse default_bus so external tools can attach.
    - Start TUI dashboard in background.
    """
    if use_default_bus:
        bus = default_bus
    else:
        bus = EventBus()

    if log_path is None:
        log_path = Path("logs") / "monitoring" / "events.log"

    logger = JsonFileLogger(path=log_path, bus=bus)

    dashboard = TuiDashboard(bus)
    t = threading.Thread(
        target=lambda: dashboard.run(refresh_per_second=4.0),
        name="TuiDashboardThread",
    )
    t.daemon = True
    t.start()

    return bus, logger, dashboard


def bootstrap_phase3(
    env_profile: "EnvProfile",
    llm_stack: "LlmStack",
    tech_state_engine: TechStateEngine,
    virtue_engine: VirtueEngine,
    skill_registry: SkillRegistry,
    bot_core: BotCore,
    obs_encoder: ObservationEncoder,
    bus: EventBus,
) -> Tuple[AgentLoopV1, AgentController]:
    """
    Phase 3:
    - Build the AgentLoop (M8) with all its dependencies.
    - Wrap it in an AgentController (M9) to handle control commands.
    """
    agent_loop = AgentLoopV1(
        env_profile=env_profile,              # Phase 0: environment config
        llm_stack=llm_stack,                  # Phase 1: LLM clients
        tech_state_engine=tech_state_engine,  # Phase 1: semantics/tech inference
        virtue_engine=virtue_engine,          # Phase 1: virtue scoring
        skill_registry=skill_registry,        # Phase 1: skills/curricula
        bot_core=bot_core,                    # Phase 2: body
        obs_encoder=obs_encoder,              # Phase 2: perception / planner payloads
        event_bus=bus,                        # Phase 3: monitoring bus
        mon_integration=mon_int,              # Phase 3: emit_* helpers
    )

    controller = AgentController(agent=agent_loop, bus=bus)
    return agent_loop, controller


# ------------------------------------------------------------------------------
# Unified runtime main
# ------------------------------------------------------------------------------

def run_full_system(config_root: Path) -> None:
    """
    Bring up the entire GTNH_Agent system across Phases 0–3 and run the main loop.

    High-level:
    - Phase 0: load & validate environment.
    - Phase 1: build thinking stack (LLM, semantics, virtues, skills).
    - Phase 2: build body + perception (BotCore + ObservationEncoder).
    - Phase 3: build monitoring + AgentLoop + controller + TUI.
    """
    # Phase 0: environment & validation
    env_profile = bootstrap_phase0(config_root=config_root)

    # Phase 1: LLM stack, semantics, virtues, skills
    llm_stack, tech_state_engine, virtue_engine, skill_registry = bootstrap_phase1(env_profile)

    # Phase 2: BotCore + observation encoding
    bot_core, obs_encoder = bootstrap_phase2(env_profile, tech_state_engine)

    # Phase 3: monitoring stack (EventBus, logger, TUI)
    bus, logger, _dashboard = build_monitoring_stack(
        use_default_bus=True,  # share bus with monitoring CLI tools
        log_path=None,         # default logs/monitoring/events.log
    )

    # Phase 3: AgentLoop + AgentController
    agent_loop, controller = bootstrap_phase3(
        env_profile=env_profile,
        llm_stack=llm_stack,
        tech_state_engine=tech_state_engine,
        virtue_engine=virtue_engine,
        skill_registry=skill_registry,
        bot_core=bot_core,
        obs_encoder=obs_encoder,
        bus=bus,
    )

    # Main runtime loop
    try:
        while True:
            safe_step_with_logging(
                controller=controller,
                bus=bus,
                episode_id=None,   # TODO: thread real episode_id from agent_loop
                context_id=None,   # TODO: thread context_id from env/profile
            )
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Shutting down GTNH_Agent full system...")
    finally:
        logger.close()


if __name__ == "__main__":
    run_full_system(config_root=Path("config"))


```


## 3. Testing, logging, and failure points (across phases)

### Testing hooks

- **Phase 0**
    
    - `tests/test_env_loader.py`: env loader & validation.
        
    - Suggested integration test: load full `EnvProfile`, assert all paths exist / model files discoverable.
        
- **Phase 1**
    
    - LLM stack: test that each role can be instantiated with dummy backend or mocked client.
        
    - Semantics: test TechState inference with toy items / tech graph.
        
    - Virtues: test scoring on synthetic traces.
        
    - Skills: test registry resolves skills & curriculum slices.
        
- **Phase 2**
    
    - BotCore: fake connection mode / mock IO, assert action → result mapping.
        
    - Observation encoder: world-state fixtures → deterministic planner payloads & tech-state.
        
- **Phase 3**
    
    - Already covered: EventBus, logger, controller, TUI, integration test with fake AgentLoop.
        
- **Full system integration**
    
    - Use a `dummy` env profile:
        
        - Local-only LLM mocks
            
        - No real Minecraft
            
    - Smoke test: `run_full_system` for a few iterations, ensure:
        
        - Logs produced
            
        - No immediate blow-up
            
        - M9 sees phase transitions and sample events.


```python
# path: tests/test_full_system_smoke.py

"""
Full system–style smoke test for GTNH_Agent Phases 0–3.

This test does NOT talk to:
- Real Minecraft
- Real LLM backends
- Real semantics/virtues/skills

Instead, it:
- Builds an EventBus + JsonFileLogger (M9).
- Wraps a FakeAgentLoop in AgentController (M9, M8 interface).
- Uses safe_step_with_logging (runtime error handling).
- Simulates a small "episode" by emitting:
    - AGENT_PHASE_CHANGE
    - PLAN_CREATED
    - PLAN_STEP_EXECUTED
- Asserts that:
    - Logs are written.
    - Event types and correlation_id form a coherent episode.

Conceptually, this stands in for a full stack:
- Phase 0: DummyEnvProfile (not explicitly exercised here).
- Phase 1: LLM / semantics / virtues / skills represented by no-op attributes.
- Phase 2: BotCore + ObservationEncoder represented by no-op attributes.
- Phase 3: AgentLoop + monitoring & tools fully exercised on the logging side.
"""

from __future__ import annotations

import json                           # to parse JSONL file content
from pathlib import Path              # for temp log path
from typing import Any, Dict, List    # for type hints

from monitoring.bus import EventBus              # core event bus (M9)
from monitoring.logger import JsonFileLogger, log_event  # JSONL logger + helper (M9)
from monitoring.controller import AgentController        # control surface (M9)
from monitoring.events import (
    EventType,
)                                              # event type enum (M9 schema)
from runtime.error_handling import safe_step_with_logging  # runtime helper


class FakeAgentLoop:
    """
    Minimal fake AgentLoop implementing AgentLoopControl protocol.

    This stands in for the fully-wired AgentLoopV1 (M8) with Phase 0–2
    dependencies hidden behind simple attributes.

    Behavior:
    - `step()` increments a counter.
    - On the first step, emits AGENT_PHASE_CHANGE + PLAN_CREATED.
    - On subsequent steps, emits PLAN_STEP_EXECUTED events.
    - `cancel_current_plan()` toggles a flag.
    - `set_goal()` records the last goal.
    - `debug_state()` returns a JSON-safe state snapshot used by DUMP_STATE.
    """

    def __init__(self, bus: EventBus) -> None:
        # Monitoring bus used to emit events
        self._bus = bus
        # Count how many steps have been executed
        self.step_count: int = 0
        # Track whether the current plan has been cancelled
        self.cancelled: bool = False
        # Track goals that have been set
        self.goals: List[str] = []
        # Fixed IDs to mimic an episode
        self.episode_id: str = "ep-full-1"
        self.context_id: str = "test-context-full"

    # --- AgentLoopControl protocol methods -----------------------------------

    def step(self) -> None:
        """
        Simulate one iteration of the agent loop.

        On the first step:
        - Emits AGENT_PHASE_CHANGE and PLAN_CREATED.

        On later steps:
        - Emits PLAN_STEP_EXECUTED for the next step index.
        """
        # Increment internal step counter
        self.step_count += 1

        # On first step, mark planning and create a new plan
        if self.step_count == 1:
            # Phase change: entering PLANNING
            log_event(
                bus=self._bus,
                module="M8.agent_loop",
                event_type=EventType.AGENT_PHASE_CHANGE,
                message="Agent entering PLANNING (fake loop)",
                payload={
                    "phase": "PLANNING",
                    "episode_id": self.episode_id,
                    "context_id": self.context_id,
                },
                correlation_id=self.episode_id,
            )

            # Plan creation (fake plan with 3 steps)
            fake_plan = {"id": "plan-full-1", "steps": [0, 1, 2]}
            log_event(
                bus=self._bus,
                module="M8.agent_loop",
                event_type=EventType.PLAN_CREATED,
                message="Fake plan created",
                payload={
                    "plan": fake_plan,
                    "goal": "Fake goal: bootstrap full system",
                    "step_count": 3,
                    "episode_id": self.episode_id,
                    "context_id": self.context_id,
                },
                correlation_id=self.episode_id,
            )
        else:
            # For subsequent steps, emit PLAN_STEP_EXECUTED
            step_index = self.step_count - 2  # 0-based index for executed steps
            log_event(
                bus=self._bus,
                module="M8.agent_loop",
                event_type=EventType.PLAN_STEP_EXECUTED,
                message=f"Fake plan step executed (index={step_index})",
                payload={
                    "episode_id": self.episode_id,
                    "context_id": self.context_id,
                    "step_index": step_index,
                    "step_spec": {"idx": step_index},
                    "trace_step": {"status": "ok"},
                },
                correlation_id=self.episode_id,
            )

    def cancel_current_plan(self) -> None:
        """
        Mark the current plan as cancelled.
        """
        self.cancelled = True

    def set_goal(self, goal: str) -> None:
        """
        Record a new high-level goal.
        """
        self.goals.append(goal)

    def debug_state(self) -> Dict[str, Any]:
        """
        Return a JSON-safe snapshot of internal state.

        Used by AgentController when handling DUMP_STATE to emit a SNAPSHOT event.
        """
        return {
            "phase": "EXECUTING" if self.step_count > 0 else "IDLE",
            "current_plan": {
                "id": "plan-full-1",
                "steps": [0, 1, 2],
            },
            "step_count": self.step_count,
            "tech_state": {"tier": "LV"},
            "episode_id": self.episode_id,
            "context_id": self.context_id,
        }


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """
    Helper: read a JSONL file and return a list of JSON dicts.
    """
    if not path.exists():
        return []
    items: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            items.append(json.loads(line))
        except json.JSONDecodeError:
            # Skip malformed lines; shouldn't normally happen under JsonFileLogger
            continue
    return items


def test_full_system_style_smoke(tmp_path: Path) -> None:
    """
    Full system–style smoke test:

    Steps:
    - Create an EventBus + JsonFileLogger.
    - Create a FakeAgentLoop bound to the bus.
    - Wrap it in AgentController.
    - Call safe_step_with_logging() a few times to simulate the main loop.
    - Verify that:
        - The log file exists and is non-empty.
        - There is a coherent episode with:
            - AGENT_PHASE_CHANGE
            - PLAN_CREATED
            - PLAN_STEP_EXECUTED
        - correlation_id == episode_id across those events.
    """
    # Arrange: create a fresh EventBus
    bus = EventBus()

    # Arrange: pick a log file under pytest's tmp_path
    log_path = tmp_path / "events_full_system.log"

    # Arrange: create a JsonFileLogger subscribed to the bus
    logger = JsonFileLogger(path=log_path, bus=bus)

    # Arrange: create a fake agent loop and wrap it with AgentController
    agent_loop = FakeAgentLoop(bus=bus)
    controller = AgentController(agent=agent_loop, bus=bus)

    # Act: simulate a few iterations of the main runtime loop
    # We don't run forever; just enough to:
    # - emit AGENT_PHASE_CHANGE + PLAN_CREATED on first step
    # - emit a couple of PLAN_STEP_EXECUTED events
    for _ in range(4):
        safe_step_with_logging(
            controller=controller,
            bus=bus,
            episode_id=agent_loop.episode_id,
            context_id=agent_loop.context_id,
        )

    # Cleanup: close logger to flush all events to disk
    logger.close()

    # Assert: log file exists and contains data
    events = _read_jsonl(log_path)
    assert events, "Expected monitoring events in JSONL log for full-system smoke test"

    # Filter events belonging to our fake episode
    ep_events = [
        e for e in events
        if e.get("payload", {}).get("episode_id") == agent_loop.episode_id
    ]
    assert ep_events, "Expected events tagged with the fake episode_id"

    # Extract event types and correlation_ids from those events
    types = {e["event_type"] for e in ep_events}
    corr_ids = {e.get("correlation_id") for e in ep_events}

    # We expect a coherent episode with:
    # - AGENT_PHASE_CHANGE
    # - PLAN_CREATED
    # - one or more PLAN_STEP_EXECUTED
    assert "AGENT_PHASE_CHANGE" in types
    assert "PLAN_CREATED" in types
    assert "PLAN_STEP_EXECUTED" in types

    # All events for this episode should share the same correlation_id
    assert corr_ids == {agent_loop.episode_id}, (
        "Expected all episode events to share a single correlation_id matching episode_id"
    )


```




---
### Logging behavior

- **All structured monitoring** → `monitoring.logger.log_event` → `JsonFileLogger` → `logs/monitoring/events.log`.
    
- **LLM-specific logs** → likely separate `logs/llm/*.json` for tokens, prompts, etc.
    
- Consumers:
    
    - `TuiDashboard`
        
    - Episode inspector
        
    - LLM log viewer
        
    - Your future self with `jq` and resentment.


```python
# path: src/monitoring/llm_logging.py

"""
LLM-specific logging utilities for GTNH_Agent (M2 + M9 bridge).

Structured monitoring events still use:
    monitoring.logger.log_event -> JsonFileLogger -> logs/monitoring/events.log

This module is specifically for:
    - Per-call LLM logs (planner / critic / scribe / error_model / etc.)
    - Written as individual JSON files under logs/llm/
    - Consumed by monitoring.tools.iter_llm_logs and human tools (jq, etc.)

Schema (one JSON per file):

{
  "ts": <float unix timestamp>,
  "role": "planner" | "critic" | "scribe" | "error_model" | ...,
  "model": "<model identifier>",
  "episode_id": "<episode-id or null>",
  "context_id": "<context-id or null>",
  "prompt": "<raw text prompt or short summary>",
  "response": "<raw text response or short summary>",
  "tokens_prompt": <int or null>,
  "tokens_completion": <int or null>,
  "meta": {
    ... arbitrary extra fields, e.g.:
    "temperature": 0.2,
    "top_p": 0.95,
    "latency_ms": 123.4,
    "error": null
  }
}

Notes:
- This is intentionally simple and file-based to stay fully offline.
- Filenames are unique based on timestamp + a random suffix.
"""

from __future__ import annotations  # allow forward references in type hints

import json                         # for writing JSON payloads
import time                         # for timestamps
import uuid                         # for unique filenames
from dataclasses import dataclass, asdict  # for LLMCallLog
from pathlib import Path            # for filesystem paths
from typing import Any, Dict, Optional     # for type hints


JsonDict = Dict[str, Any]


# ------------------------------------------------------------------------------
# Data structure representing a single LLM call
# ------------------------------------------------------------------------------

@dataclass
class LLMCallLog:
    """
    Structured record for one LLM interaction.

    This is the canonical schema for logs under logs/llm/*.json.
    """
    ts: float                        # unix timestamp when call completed
    role: str                        # "planner", "critic", "scribe", "error_model", etc.
    model: str                       # model identifier (e.g. "qwen2.5-coder-7b-instruct")
    episode_id: Optional[str]        # logical episode id, if available
    context_id: Optional[str]        # context / env id, if available
    prompt: str                      # text passed into the LLM (raw or summarized)
    response: str                    # text returned by the LLM (raw or summarized)
    tokens_prompt: Optional[int]     # number of prompt tokens, if known
    tokens_completion: Optional[int] # number of completion tokens, if known
    meta: JsonDict                   # free-form metadata (latency, error flags, etc.)

    def to_dict(self) -> JsonDict:
        """
        Convert the dataclass to a JSON-serializable dict.
        """
        return asdict(self)


# ------------------------------------------------------------------------------
# LLM log writer
# ------------------------------------------------------------------------------

class LLMLogWriter:
    """
    File-based LLM log writer.

    Responsibilities:
    - Ensure logs/llm directory exists.
    - Write one JSON file per LLM call using the LLMCallLog schema.
    - Keep the interface dead simple so any LLM client can call it.

    Typical usage:

        writer = LLMLogWriter()
        log = LLMCallLog(
            ts=time.time(),
            role="planner",
            model="qwen2.5-coder-7b-instruct",
            episode_id=episode_id,
            context_id=context_id,
            prompt=prompt_text,
            response=response_text,
            tokens_prompt=prompt_tokens,
            tokens_completion=completion_tokens,
            meta={"temperature": 0.2, "latency_ms": 150.0},
        )
        writer.write(log)
    """

    def __init__(self, log_dir: Path | None = None) -> None:
        # Default directory for LLM logs if not specified
        if log_dir is None:
            log_dir = Path("logs") / "llm"
        self._log_dir = log_dir
        # Ensure the directory exists
        self._log_dir.mkdir(parents=True, exist_ok=True)

    @property
    def log_dir(self) -> Path:
        """
        Return the directory where LLM logs are written.
        """
        return self._log_dir

    def write(self, call_log: LLMCallLog) -> Path:
        """
        Persist a single LLM call log as a JSON file.

        Returns:
            The full path to the written JSON file.
        """
        # Use timestamp + random suffix to avoid collisions
        ts_part = f"{call_log.ts:.6f}"
        role_part = call_log.role or "unknown"
        rand_part = uuid.uuid4().hex[:8]
        filename = f"{ts_part}_{role_part}_{rand_part}.json"
        path = self._log_dir / filename

        # Serialize dataclass to a dict then to JSON
        data = call_log.to_dict()

        # Write as pretty-printed JSON (small files, human-readable)
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, sort_keys=True)

        return path


# ------------------------------------------------------------------------------
# Convenience function for quick logging
# ------------------------------------------------------------------------------

def log_llm_call(
    role: str,
    model: str,
    prompt: str,
    response: str,
    episode_id: Optional[str] = None,
    context_id: Optional[str] = None,
    tokens_prompt: Optional[int] = None,
    tokens_completion: Optional[int] = None,
    meta: Optional[JsonDict] = None,
    log_dir: Path | None = None,
) -> Path:
    """
    One-shot helper to log an LLM call without manually constructing LLMCallLog.

    Arguments:
        role:
            Logical role of this LLM call, e.g. "planner", "critic", "scribe".
        model:
            Model identifier string.
        prompt:
            Prompt text (raw or summarized).
        response:
            Response text (raw or summarized).
        episode_id:
            Episode correlation id, if available.
        context_id:
            Context/environment id, if available.
        tokens_prompt:
            Number of prompt tokens, if known.
        tokens_completion:
            Number of completion tokens, if known.
        meta:
            Additional metadata such as latency, temperature, errors, etc.
        log_dir:
            Optional override for the logs directory (defaults to logs/llm).

    Returns:
        Path to the written JSON log file.
    """
    # Create an LLMCallLog instance with current timestamp and provided details
    call_log = LLMCallLog(
        ts=time.time(),
        role=role,
        model=model,
        episode_id=episode_id,
        context_id=context_id,
        prompt=prompt,
        response=response,
        tokens_prompt=tokens_prompt,
        tokens_completion=tokens_completion,
        meta=meta or {},
    )

    # Create a writer for the given directory and persist the log
    writer = LLMLogWriter(log_dir=log_dir)
    return writer.write(call_log)


```


---

### Failure points & mitigations (cross-phase)

- **Config / model mismatch (Phase 0 / 1)**
    
    - Mitigate with:
        
        - `validate_env.py`
            
        - Unit tests for `load_env_profile`
            
        - Early checks for model file presence, Minecraft host/port, etc.
            
- **LLM failures (Phase 1 / AgentLoop)**
    
    - Timeouts, OOM, garbage responses.
        
    - Use:
        
        - Error-model or retry logic in M2/M8.
            
        - Emit `LOG` events for LLM errors with subtypes like `"LLM_TIMEOUT"`, `"LLM_BAD_OUTPUT"`.
            
- **BotCore / Minecraft IO (Phase 2)**
    
    - Socket timeouts, desync, invalid action targets.
        
    - Emit:
        
        - `ACTION_EXECUTED` with `success=False`, `error="..."`.
            
        - Possibly `PLAN_FAILED` when repeatedly failing.
            
- **AgentLoop stepping (Phase 3)**
    
    - Any unhandled exception inside `agent_loop.step()`.
        
    - `safe_step_with_logging`:
        
        - Logs `"AGENT_STEP_EXCEPTION"` via M9.
            
        - Re-raises for controlled shutdown or recovery.
            
- **Monitoring & TUI overload**
    
    - High-volume event spam.
        
    - Mitigation:
        
        - Event bus is in-process and simple.
            
        - TUI samples latest state; worst case: slightly laggy HUD.
            
        - Logger drops writes silently on disk failure rather than killing the process.

```python
# path: src/runtime/failure_mitigation.py

"""
Cross-phase failure handling helpers for GTNH_Agent.

This module centralizes how we turn failures into structured monitoring
events for M9, matching the "Failure points & mitigations (cross-phase)"
design.

It DOES NOT try to detect failures itself. Instead, it provides small,
explicit helpers that other modules can call when they hit trouble.

Failure classes covered:

1) Config / model mismatch (Phase 0 / 1)
   - Env loader or validators can call:
       emit_config_error(...)
       emit_model_path_error(...)

2) LLM failures (Phase 1 / AgentLoop)
   - LLM wrappers or planner/critic code can call:
       emit_llm_failure(...)
   - Uses LOG events with subtypes like:
       "LLM_TIMEOUT", "LLM_OOM", "LLM_BAD_OUTPUT", "LLM_BACKEND_ERROR"

3) BotCore / Minecraft IO (Phase 2)
   - BotCore action wrappers can call:
       emit_action_failure(...)
   - Optionally escalate to plan failure:
       emit_plan_failed_due_to_actions(...)

4) AgentLoop stepping (Phase 3)
   - Already handled by runtime.error_handling.safe_step_with_logging()
   - That helper logs "AGENT_STEP_EXCEPTION" events.
"""

from __future__ import annotations  # forward type references in type hints

from typing import Any, Dict, Iterable, Optional  # type hints

from monitoring.bus import EventBus                    # event bus used across system
from monitoring.events import EventType                # monitoring event type enum
from monitoring.logger import log_event                # convenience helper for JSONL logging


JsonDict = Dict[str, Any]


# ------------------------------------------------------------------------------
# 1. Config / model mismatch (Phase 0 / 1)
# ------------------------------------------------------------------------------

def emit_config_error(
    bus: EventBus,
    message: str,
    *,
    env_profile_repr: Optional[str] = None,
    error_repr: Optional[str] = None,
) -> None:
    """
    Emit a LOG event for a general configuration error.

    Typical usage (Phase 0 / 1):

        try:
            env_profile = load_env_profile(...)
        except Exception as exc:
            emit_config_error(bus, "Failed to load env profile", error_repr=repr(exc))
            raise

    This does NOT exit the process by itself; caller decides whether to abort.
    """
    payload: JsonDict = {
        "subtype": "CONFIG_ERROR",
        "env_profile": env_profile_repr,
        "error": error_repr,
    }

    log_event(
        bus=bus,
        module="runtime.config",
        event_type=EventType.LOG,
        message=message,
        payload=payload,
        correlation_id=None,
    )


def emit_model_path_error(
    bus: EventBus,
    message: str,
    *,
    model_id: str,
    expected_path: str,
    error_repr: Optional[str] = None,
) -> None:
    """
    Emit a LOG event for a model path mismatch / missing file.

    Example usage:

        if not model_path.exists():
            emit_model_path_error(
                bus,
                "Model path does not exist",
                model_id=model_id,
                expected_path=str(model_path),
                error_repr=None,
            )
            raise FileNotFoundError(...)

    This corresponds to the "Config / model mismatch (Phase 0 / 1)" bullet.
    """
    payload: JsonDict = {
        "subtype": "MODEL_PATH_ERROR",
        "model_id": model_id,
        "expected_path": expected_path,
        "error": error_repr,
    }

    log_event(
        bus=bus,
        module="runtime.models",
        event_type=EventType.LOG,
        message=message,
        payload=payload,
        correlation_id=None,
    )


# ------------------------------------------------------------------------------
# 2. LLM failures (Phase 1 / AgentLoop)
# ------------------------------------------------------------------------------

def emit_llm_failure(
    bus: EventBus,
    *,
    subtype: str,
    role: str,
    model: str,
    episode_id: Optional[str],
    context_id: Optional[str],
    error_repr: str,
    meta: Optional[JsonDict] = None,
) -> None:
    """
    Emit a LOG event describing an LLM failure.

    Expected subtypes include (but are not limited to):
        - "LLM_TIMEOUT"
        - "LLM_OOM"
        - "LLM_BAD_OUTPUT"
        - "LLM_BACKEND_ERROR"

    Example usage in M2 / M8:

        try:
            result = planner_llm.call(prompt)
        except TimeoutError as exc:
            emit_llm_failure(
                bus,
                subtype="LLM_TIMEOUT",
                role="planner",
                model=planner_model_id,
                episode_id=episode_id,
                context_id=context_id,
                error_repr=repr(exc),
                meta={"timeout_s": timeout_value},
            )
            raise
    """
    payload: JsonDict = {
        "subtype": subtype,
        "llm_role": role,
        "model": model,
        "episode_id": episode_id,
        "context_id": context_id,
        "error": error_repr,
        "meta": meta or {},
    }

    log_event(
        bus=bus,
        module="runtime.llm",
        event_type=EventType.LOG,
        message=f"LLM failure ({subtype}) for role={role}, model={model}",
        payload=payload,
        correlation_id=episode_id,
    )


# ------------------------------------------------------------------------------
# 3. BotCore / Minecraft IO (Phase 2)
# ------------------------------------------------------------------------------

def emit_action_failure(
    bus: EventBus,
    *,
    episode_id: Optional[str],
    context_id: Optional[str],
    action_name: str,
    action_args: JsonDict,
    error_repr: str,
    module_name: str = "M6.bot_core",
) -> None:
    """
    Emit an ACTION_EXECUTED event for a failed action.

    This corresponds to "Emit ACTION_EXECUTED with success=False, error=...".

    Intended usage inside BotCore or its wrappers:

        try:
            res = self._do_click_block(...)
        except Exception as exc:
            emit_action_failure(
                bus=self.event_bus,
                episode_id=episode_id,
                context_id=context_id,
                action_name="click_block",
                action_args={"pos": pos.to_tuple()},
                error_repr=repr(exc),
            )
            raise

    Note:
    - success is explicitly False
    - error field includes a string representation of the cause
    """
    payload: JsonDict = {
        "episode_id": episode_id,
        "context_id": context_id,
        "action_name": action_name,
        "action_args": action_args,
        "success": False,
        "error": error_repr,
    }

    log_event(
        bus=bus,
        module=module_name,
        event_type=EventType.ACTION_EXECUTED,
        message=f"Action failed: {action_name}",
        payload=payload,
        correlation_id=episode_id,
    )


def emit_plan_failed_due_to_actions(
    bus: EventBus,
    *,
    episode_id: Optional[str],
    context_id: Optional[str],
    reason: str,
    failing_actions: Iterable[JsonDict],
    module_name: str = "M8.agent_loop",
) -> None:
    """
    Emit a PLAN_FAILED event when repeated action failures make
    continuing the plan pointless.

    This corresponds to:
        "Possibly PLAN_FAILED when repeatedly failing."

    `failing_actions` is a list (or other iterable) of summaries, e.g.:

        failing_actions = [
            {"action": "move_to", "error": "path_blocked"},
            {"action": "move_to", "error": "path_blocked"},
        ]

    Caller decides when to trigger this (e.g. after N retries).
    """
    payload: JsonDict = {
        "episode_id": episode_id,
        "context_id": context_id,
        "reason": reason,
        "failing_actions": list(failing_actions),
    }

    log_event(
        bus=bus,
        module=module_name,
        event_type=EventType.PLAN_FAILED,
        message=reason,
        payload=payload,
        correlation_id=episode_id,
    )


# ------------------------------------------------------------------------------
# 4. Monitoring & TUI overload
# ------------------------------------------------------------------------------

def emit_monitoring_overload_warning(
    bus: EventBus,
    *,
    approx_rate_hz: float,
    module_name: str = "M9.monitoring",
    context_id: Optional[str] = None,
) -> None:
    """
    Optional helper to log when you detect excessive event volume.

    This does NOT implement rate limiting by itself, but gives you
    a structured way to mark that you think you're spamming the bus/logs.

    Example usage:

        if events_per_second > 500:
            emit_monitoring_overload_warning(bus, approx_rate_hz=events_per_second)

    The design text notes:
    - Event bus is simple & in-process.
    - TUI samples latest state; worst case: slightly laggy.
    - Logger can drop events silently on disk failure.

    This helper just makes such conditions visible in the logs.
    """
    payload: JsonDict = {
        "subtype": "MONITORING_OVERLOAD",
        "approx_rate_hz": approx_rate_hz,
        "context_id": context_id,
    }

    log_event(
        bus=bus,
        module=module_name,
        event_type=EventType.LOG,
        message="High monitoring event rate detected",
        payload=payload,
        correlation_id=None,
    )

```

```python
# path: tests/test_failure_mitigation.py

"""
Unit tests for runtime.failure_mitigation helpers.

These tests verify that the helpers:
- Emit events with the correct event_type.
- Carry the expected subtype and payload fields.
"""

from __future__ import annotations

from typing import Any, Dict, List  # type hints

from monitoring.bus import EventBus          # event bus
from monitoring.events import EventType      # event types
from monitoring.events import MonitoringEvent  # event dataclass

from runtime.failure_mitigation import (
    emit_config_error,
    emit_model_path_error,
    emit_llm_failure,
    emit_action_failure,
    emit_plan_failed_due_to_actions,
    emit_monitoring_overload_warning,
)


def _capture_events(bus: EventBus) -> List[MonitoringEvent]:
    """
    Subscribe a collector to the bus and return the underlying list.
    """
    captured: List[MonitoringEvent] = []

    def subscriber(evt: MonitoringEvent) -> None:
        captured.append(evt)

    bus.subscribe(subscriber)
    return captured


def test_emit_config_error_and_model_path_error():
    bus = EventBus()
    captured = _capture_events(bus)

    emit_config_error(
        bus=bus,
        message="Failed to load env profile",
        env_profile_repr="EnvProfile(name='test')",
        error_repr="ValueError('oops')",
    )

    emit_model_path_error(
        bus=bus,
        message="Model path does not exist",
        model_id="planner-model",
        expected_path="/models/planner.bin",
        error_repr=None,
    )

    assert len(captured) == 2

    cfg_evt, model_evt = captured

    assert cfg_evt.event_type == EventType.LOG
    assert cfg_evt.payload["subtype"] == "CONFIG_ERROR"
    assert "EnvProfile(name='test')" in cfg_evt.payload["env_profile"]

    assert model_evt.event_type == EventType.LOG
    assert model_evt.payload["subtype"] == "MODEL_PATH_ERROR"
    assert model_evt.payload["model_id"] == "planner-model"
    assert model_evt.payload["expected_path"] == "/models/planner.bin"


def test_emit_llm_failure():
    bus = EventBus()
    captured = _capture_events(bus)

    emit_llm_failure(
        bus=bus,
        subtype="LLM_TIMEOUT",
        role="planner",
        model="test-model",
        episode_id="ep-123",
        context_id="ctx-1",
        error_repr="TimeoutError('boom')",
        meta={"timeout_s": 5.0},
    )

    assert len(captured) == 1
    evt = captured[0]

    assert evt.event_type == EventType.LOG
    assert evt.payload["subtype"] == "LLM_TIMEOUT"
    assert evt.payload["llm_role"] == "planner"
    assert evt.payload["model"] == "test-model"
    assert evt.payload["episode_id"] == "ep-123"
    assert evt.payload["context_id"] == "ctx-1"
    assert evt.payload["meta"]["timeout_s"] == 5.0
    assert evt.correlation_id == "ep-123"


def test_emit_action_failure_and_plan_failed_due_to_actions():
    bus = EventBus()
    captured = _capture_events(bus)

    emit_action_failure(
        bus=bus,
        episode_id="ep-abc",
        context_id="ctx-xyz",
        action_name="move_to",
        action_args={"pos": [0, 64, 0]},
        error_repr="IOError('no path')",
    )

    failing_actions: List[Dict[str, Any]] = [
        {"action": "move_to", "error": "path_blocked"},
        {"action": "move_to", "error": "path_blocked"},
    ]

    emit_plan_failed_due_to_actions(
        bus=bus,
        episode_id="ep-abc",
        context_id="ctx-xyz",
        reason="Too many failed movement attempts",
        failing_actions=failing_actions,
    )

    assert len(captured) == 2

    action_evt, plan_evt = captured

    # First event: ACTION_EXECUTED failure
    assert action_evt.event_type == EventType.ACTION_EXECUTED
    assert action_evt.payload["success"] is False
    assert action_evt.payload["action_name"] == "move_to"
    assert action_evt.payload["error"].startswith("IOError")

    # Second event: PLAN_FAILED escalation
    assert plan_evt.event_type == EventType.PLAN_FAILED
    assert plan_evt.payload["reason"] == "Too many failed movement attempts"
    assert len(plan_evt.payload["failing_actions"]) == 2
    assert plan_evt.correlation_id == "ep-abc"


def test_emit_monitoring_overload_warning():
    bus = EventBus()
    captured = _capture_events(bus)

    emit_monitoring_overload_warning(
        bus=bus,
        approx_rate_hz=123.45,
        context_id="ctx-overload",
    )

    assert len(captured) == 1
    evt = captured[0]

    assert evt.event_type == EventType.LOG
    assert evt.payload["subtype"] == "MONITORING_OVERLOAD"
    assert evt.payload["approx_rate_hz"] == 123.45
    assert evt.payload["context_id"] == "ctx-overload"

```



---

You now have a **multi-phase, multi-module architecture** that isn’t just vibes and markdown.

Phase 0 gives the world, Phase 1 gives the mind, Phase 2 gives the body, Phase 3 gives awareness & control.

The rest is “just” teaching it to build a Stargate without melting your base.