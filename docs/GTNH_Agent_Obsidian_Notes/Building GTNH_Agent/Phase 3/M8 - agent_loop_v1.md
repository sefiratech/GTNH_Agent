# M8 · `agent_loop_v1`

**Phase:** P3 – Agent Orchestration  
**Role:** Implement the core episode loop using `AgentRuntime` as the service boundary:  

1. Observe via BotCore (indirectly, through `AgentRuntime`)  
2. Encode world state (M7) and call planner (M2)  
3. Decompose planner output into skills (M5) and BotCore actions (M6)  
4. Execute and record `TraceStep`s into a `PlanTrace` (M7)  
5. Optionally run critic + virtue scoring (M2 + M4)  
6. Store an `Experience` for future learning (M10)  

M8 does **not** rewire semantics, LLM stack, or BotCore. It orchestrates them through `AgentRuntime` and owns the control logic around episodes, retries, and logging.

---

## 0. Purpose & High-Level Algorithm

**Purpose:**  
Provide the first full, end-to-end **agent loop** on top of the already integrated runtime (M0–M7). The loop operates at an **episode** level: one episode = one planner call, zero or more execution steps, optional critique, and an experience record.  

**High-level algorithm (v1):**

1. `runtime = build_agent_runtime(profile=...)`  
2. `loop = AgentLoop(runtime, config=...)`  
3. `EpisodeResult = loop.run_episode(episode_id=...)`  

Inside `run_episode`:

1. `plan = runtime.planner_tick(goal=...)`  
2. `obs = runtime.get_latest_planner_observation()`  
3. Initialize `PlanTrace` from plan + planner payload  
4. Execute plan steps via skills → BotCore actions (through runtime)  
5. Update `PlanTrace.steps` with `TraceStep`s  
6. Optionally:
   - Run critic over `PlanTrace`  
   - Run virtue scoring over plan / trace  
7. Wrap into `Experience` and push into `ExperienceBuffer`  

The loop is designed to evolve later into a **state machine** (`Idle → Planning → Executing → Recovering`), but v1 is implemented as a single episode-level call with clear internal stages.

---

## 1. Responsibilities & Boundaries

### 1.1 What M8 owns

M8 (`src/agent/loop.py` + `src/agent/experience.py`) owns:

- The **episode loop**:
  - Define `AgentLoop`, `AgentLoopConfig`, and `EpisodeResult`.
  - Orchestrate one end-to-end cycle: observe → plan → act → critique → log.
- The **link between runtime and higher-level evaluation**:
  - Use `AgentRuntime` for all environment-facing operations (planner, BotCore, semantics).
  - Convert plans to `PlanTrace` and `Experience` objects for later learning (M10).
- The **entry point for self-evaluation & retry**:
  - Add hooks for critic calls and retry policies (even if simple in v1).
- **Virtue & curriculum hooks**:
  - Accept an optional `virtue_engine` and (later) curriculum goal selection.
  - Annotate traces with virtue scores when available.
- **Experience buffering**:
  - Manage an in-memory `ExperienceBuffer` that stores episode outcomes for later processing.

### 1.2 What M8 does *not* do

M8 explicitly does **not**:

- Construct or manage:
  - BotCore clients
  - LLM backends
  - Semantics databases
  - Observation encoders  
  All of that is the job of `AgentRuntime` (M6 + M7 + M2 + M3).
- Contain GTNH-specific item IDs, machine names, or recipe logic (M3).
- Implement virtue lattice math (M4) beyond calling into an injected engine.
- Define or load skills from YAML (M5) beyond invoking a provided skill registry handle.
- Handle monitoring UI directly (M9) beyond emitting logs / events through shared logging utilities.

Think of M8 as the **conductor** for an already-tuned orchestra, not the person soldering instruments together.

---

## 2. High-Level Architecture

```text
+----------------------------+
|  app.runtime / CLI / TUI   |
|  (e.g. src/app/runtime.py) |
+--------------+-------------+
               |
               v
       +-----------------+
       |   AgentLoop     |
       | (M8, episodes)  |
       +--------+--------+
                |
       +--------+--------+
       |   AgentRuntime  |  (M6 + M7 + M2 + M3)
       |  runtime_m6_m7  |
       +--+-----+-----+--+
          |     |     |
          v     v     v
      BotCore  LLM  Semantics
       (M6)    (M2)   (M3)
          ^     ^
          |     |
          +-- Observation (M7)

         +----------------+
         | Experience     |
         |  Buffer (M8)   |
         +--------+-------+
                  |
                  v
        Learning / M10 (future)

```

Key points:

- **Single entrypoint:** `build_agent_runtime()` constructs `AgentRuntime`, which hides all lower-level modules. AgentLoop never re-implements this wiring.  
- **Observation flow:** BotCore (`observe()`) → M7 encoder → PlannerModel (M2). AgentLoop only sees the planner’s output + an `Observation` / planner payload.
- **Execution flow:** Planner plan → skill-level execution (M5, injected) → BotCore actions via runtime.
- **Experience flow:** `PlanTrace` + critic/virtue outputs → `ExperienceBuffer` → future M10 learning.

---

## 3. Core Types

M8 builds on types defined in `spec/` and `observation/trace_schema.py`. The important ones for this module are:

### 3.1 Episode & loop configuration

File: `src/agent/loop.py`

```python
# src/agent/loop.py
"""
Phase 3 preparation: AgentLoop scaffold.

Concept:
    M8 should NOT recreate wiring for:
      - observation
      - encoding
      - planner / critic model calls

Instead, it should treat AgentRuntime (M6 + M7 + M2) as a service:

    runtime = build_agent_runtime(...)
    loop = AgentLoop(runtime=runtime, virtue_engine=..., skills=...)
    result = loop.run_episode(...)

This module provides:
    - AgentLoopConfig: basic knobs for the loop
    - EpisodeResult: structured return value
    - AgentLoop: a thin orchestrator over AgentRuntime

The actual execution logic is intentionally minimal for now:
    - call planner_tick()
    - build a PlanTrace with zero steps (no executor yet)
    - optionally call critic at the loop level
    - optionally store the result in an ExperienceBuffer

Later, M8 can:
    - insert real action execution
    - populate TraceStep entries
    - update tech_state
    - add retry/self-eval logic
without changing the basic interface.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import logging
import time

from agent.runtime_m6_m7 import AgentRuntime
from agent.experience import Experience, ExperienceBuffer
from observation.trace_schema import PlanTrace
from observation.encoder import encode_for_critic
from spec.experience import CriticResult

logger = logging.getLogger(__name__)


@dataclass
class AgentLoopConfig:
    """
    Configuration knobs for the high-level agent loop.

    Fields:
        enable_critic:
            If True, run the critic on each episode's PlanTrace and attach
            the result to the experience. If False, critic is skipped.

        store_experiences:
            If True, append an Experience record to the ExperienceBuffer
            after each episode.

        max_planner_calls:
            For now, controls how many planner calls run per episode.
            With no executor, this is typically 1 (plan once, then stop).

        max_skill_steps:
            Soft cap on how many plan steps (skills) are executed per episode.

        fail_fast_on_invalid_plan:
            If True, raise an error when the planner returns an invalid shape
            (e.g. missing "steps"). If False, the loop may choose to recover.

        log_virtue_scores:
            Reserved for future virtue integration; when True, virtue scores
            should be attached to traces if a virtue engine is present.

        log_traces:
            Reserved for future trace logging controls (e.g. to disk/monitoring).
    """
    enable_critic: bool = True
    store_experiences: bool = True
    max_planner_calls: int = 1          # v1: single-plan episodes
    max_skill_steps: int = 16           # soft cap on steps from a plan
    fail_fast_on_invalid_plan: bool = True
    log_virtue_scores: bool = True
    log_traces: bool = True


@dataclass
class EpisodeResult:
    """
    Result of a single agent episode: one planner call
    plus zero or more execution steps and an optional critique.

    Fields:
        episode_id:
            Optional external episode identifier, for logging / correlation.

        plan:
            The planner output dict returned by AgentRuntime.planner_tick().

        trace:
            PlanTrace that wraps the plan plus any execution details
            (currently no steps are added; future M8 work will fill them).

        critic_result:
            Output from the critic model, if enable_critic is True and a
            critic is configured on the runtime. Otherwise None.

        started_at / finished_at:
            Timestamps (seconds since epoch) for rough timing.
    """
    episode_id: Optional[int]
    plan: Dict[str, Any]
    trace: PlanTrace
    critic_result: Optional[CriticResult]
    started_at: float
    finished_at: float


@dataclass
class AgentLoop:
    """
    High-level loop scaffold that wraps AgentRuntime.

    This is the main seam where Phase 3 (M8) will grow:

        - Just before planning: decide whether to reuse / adapt previous plan
        - After planning: execute actions, record TraceStep entries
        - After execution: run critic, update virtues/skills, store experiences

    For now, it wires just enough to:
        - validate Phase 0–2 integration
        - keep PlanTrace & critic at the loop level
        - keep experience logging optional and externalizable
    """
    runtime: AgentRuntime
    experience_buffer: ExperienceBuffer = field(default_factory=ExperienceBuffer)
    config: AgentLoopConfig = field(default_factory=AgentLoopConfig)
    virtue_engine: Optional[Any] = None   # reserved for M4/M8 integration
    skills: Optional[Any] = None          # reserved for M5/M8 integration

    def run_episode(self, episode_id: Optional[int] = None) -> EpisodeResult:
        """
        Run a minimal "episode":

            - Call planner_tick() once (or max_planner_calls times in future)
            - Build a PlanTrace with no execution steps yet
            - Optionally run the critic and store an Experience

        This is intentionally conservative so M8 can expand it without
        fighting baked-in assumptions.
        """
        started_at = time.time()

        # Phase 2 behavior: one planner call per episode.
        plan = self.runtime.planner_tick()
        obs = self.runtime.get_latest_planner_observation()

        # Try to grab a tech_state-like object from the runtime.
        # Different implementations may expose different attribute names
        # (e.g. current_tech_state vs tech_state). We don't enforce one yet.
        tech_state = getattr(self.runtime, "current_tech_state", None)
        if tech_state is None:
            tech_state = getattr(self.runtime, "tech_state", None)

        # Build a PlanTrace with no steps yet.
        # Future M8 work will attach real TraceStep entries.
        trace = PlanTrace(
            plan=plan,
            steps=[],
            tech_state=tech_state,
            context_id=self.runtime.config.context_id,
            planner_payload=obs.json_payload,
            virtue_scores={},  # virtues not yet integrated into the loop
        )

        critic_result: Optional[CriticResult] = None

        if self.config.enable_critic and getattr(self.runtime, "critic_model", None):
            try:
                critic_payload = encode_for_critic(trace)
                critic_result = self.runtime.critic_model.evaluate(critic_payload)
            except Exception as exc:  # defensive: critic must not crash the loop
                logger.warning("Critic evaluation failed: %r", exc)
                critic_result = None

        finished_at = time.time()

        result = EpisodeResult(
            episode_id=episode_id,
            plan=plan,
            trace=trace,
            critic_result=critic_result,
            started_at=started_at,
            finished_at=finished_at,
        )

        if self.config.store_experiences:
            experience = Experience(
                trace=trace,
                critic_result=critic_result,
                episode_id=episode_id,
                meta={
                    "duration_sec": finished_at - started_at,
                    "context_id": self.runtime.config.context_id,
                },
            )
            self.experience_buffer.add(experience)

        return result

```
### 3.2 Experience & buffer

File: `src/agent/experience.py` (already present, extended in M8)

```python
# src/agent/experience.py
"""
Phase 4 preparation: experience structures.

This module defines small, generic containers that can be used by
the agent loop (M8) and any future experience / memory system:

  - Experience: one episode's plan/trace/critic bundle, plus context
  - ExperienceBuffer: an in-memory collection with a simple API

Nothing here assumes a specific storage backend. You can later:
  - bolt on filesystem / DB logging
  - add sampling logic for training
  - add indexing for retrieval
without changing the M8 loop signature.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import time

from observation.trace_schema import PlanTrace
from spec.experience import CriticResult


@dataclass
class Experience:
    """
    Episode-level experience bundle suitable for M10 learning.

    Fields:
        episode_id:
            Optional id for grouping experiences by logical episode.

        trace:
            PlanTrace containing:
              - plan emitted by the planner
              - execution steps (may be empty at first)
              - tech_state, context_id, planner_payload, virtue_scores

        critic_result:
            Output from the critic model, if enabled. Typed via CriticResult
            so different critic backends can still share a common shape.

        env_profile_name:
            Name/identifier of the active environment profile
            (e.g. from env.yaml via AgentRuntime).

        context_id:
            High-level virtue / scenario context id associated with the
            episode (e.g. "lv_early_factory").

        tech_state_snapshot:
            Serialized snapshot of tech_state at the end of the episode,
            suitable for logging and learning.

        meta:
            Free-form metadata (profile name, seed, environment tags, duration,
            curriculum goal, etc.).

        timestamp:
            Seconds since epoch when the experience was recorded.
    """
    trace: PlanTrace
    critic_result: Optional[CriticResult]
    episode_id: Optional[int] = None
    env_profile_name: str = ""
    context_id: str = ""
    tech_state_snapshot: Dict[str, Any] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=lambda: time.time())


@dataclass
class ExperienceBuffer:
    """
    Simple append-only buffer for episodes.

    This is intentionally simple for Phase 3/4 prep:
      - append-only list
      - trivial retrieval API

    Later you can:
      - add persistence hooks (write to disk, DB, etc.)
      - add sampling strategies (for training)
      - add filters / indexes
    """
    _items: List[Experience] = field(default_factory=list)

    def add(self, experience: Experience) -> None:
        """Append a new experience to the buffer."""
        self._items.append(experience)

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self._items)

    def last(self) -> Optional[Experience]:
        """Return the most recent experience, or None if buffer is empty."""
        return self._items[-1] if self._items else None

    @property
    def experiences(self) -> List[Experience]:
        """
        Backwards-compatible view of the internal list.

        Older tests or code may still access .experiences directly.
        """
        return self._items

```

### 3.3 Virtue engine protocol (hook for M4)

```python
# AgentLoop interface / contract
# src/spec/agent_loop.py

from __future__ import annotations

from typing import Any, Dict, Mapping, Protocol

from observation.trace_schema import PlanTrace

from .types import WorldState
from .llm import PlannerModel, CriticModel
from .skills import SkillRegistry
from .bot_core import BotCore


class AgentLoop(Protocol):
    """High-level control loop for the GTNH Agent.

    This ties together:
    - BotCore (M6) for acting in the world
    - SkillRegistry (M5) for available behaviors
    - PlannerModel / CriticModel (M2) for plan generation & evaluation
    """

    # Structural dependencies (not required at runtime, but part of the design)
    bot: BotCore
    skills: SkillRegistry
    planner: PlannerModel
    critic: CriticModel

    def step(self) -> None:
        """
        Perform one full agent iteration:
        - observe world via BotCore.get_world_state()
        - encode world into an Observation (M7)
        - compute or reuse a plan using PlannerModel (M2)
        - select and execute skills (M5) via BotCore.execute_action()
        - evaluate / record experience (M10)
        """
        ...

    def set_goal(self, goal: str, context: Mapping[str, Any]) -> None:
        """Set or update the current top-level goal (e.g. 'establish LV steam power')."""
        ...

    def get_status(self) -> Mapping[str, Any]:
        """
        Provide a snapshot of:
        - current goal
        - current plan
        - last action/result
        - any error state
        - any high-level metrics (ticks alive, deaths, etc.)
        """
        ...


class VirtueEngine(Protocol):
    """
    Protocol for a virtue scoring engine.

    Implemented in M4 using virtues.lattice and virtues.metrics.
    M8 (AgentLoop) only uses this via dependency injection: it does not know
    how virtues are computed, it only receives scores over a PlanTrace.
    """

    def score_trace(self, trace: PlanTrace, context_id: str) -> Dict[str, float]:
        ...

```

M8 does not implement the engine; it just calls one if provided.

---

## 4. AgentLoop Class

File: `src/agent/loop.py` (this is the canonical M8 class)

### 4.1 Skeleton

```python
import time
from typing import Optional

from agent.runtime_m6_m7 import AgentRuntime
from agent.experience import Experience, ExperienceBuffer
from observation.trace_schema import PlanTrace, TraceStep
from observation.encoder import encode_for_critic
from spec.agent_loop import AgentLoopSpec  # high-level contract
from spec.experience import CriticResult


class AgentLoop(AgentLoopSpec):
    """
    Episode-level agent loop built on top of AgentRuntime.

    Responsibilities:
      - Call planner via runtime
      - Execute plan steps via injected skills / runtime
      - Build PlanTrace
      - Optionally call critic + virtues
      - Store experiences
    """

    def __init__(
        self,
        runtime: AgentRuntime,
        config: Optional[AgentLoopConfig] = None,
        virtue_engine: Optional[VirtueEngine] = None,
        experience_buffer: Optional[ExperienceBuffer] = None,
    ) -> None:
        self.runtime = runtime
        self.config = config or AgentLoopConfig()
        self.virtue_engine = virtue_engine
        self.experiences = experience_buffer or ExperienceBuffer()

    # ------------------------------------------------
    # Public API
    # ------------------------------------------------

    def run_episode(self, episode_id: Optional[int] = None) -> EpisodeResult:
        """
        Run a single episode:
          planner → executor → critic/virtue → experience buffer.
        """
        started_at = time.time()

        # 1. Plan using runtime (includes observe + encode + planner call)
        plan = self._call_planner_once()

        # 2. Build initial trace (no steps yet)
        trace = self._build_initial_trace(plan)

        # 3. Execute plan into TraceStep(s)
        self._execute_plan_into_trace(plan, trace)

        # 4. Critic + virtues
        critic_result = self._maybe_run_critic(trace)
        self._maybe_run_virtues(trace)

        finished_at = time.time()

        result = EpisodeResult(
            episode_id=episode_id,
            plan=plan,
            trace=trace,
            critic_result=critic_result,
            started_at=started_at,
            finished_at=finished_at,
        )

        # 5. Experience logging
        if self.config.store_experiences:
            self._store_experience(result)

        return result
```

---

## 5. Phase Logic (Internal Stages)

M8 reuses the old “phases” concept (`Idle`, `Planning`, `Executing`, `Recovering`) but normalizes it into **stages inside `run_episode`** instead of a long-lived tick-based state machine. The same mental model still applies, just scoped to one episode.

### 5.1 Planning stage

```python
    def _call_planner_once(self) -> dict:
        """
        Delegate planning to AgentRuntime.
        Runtime handles:
          - BotCore.observe()
          - encode_for_planner()
          - PlannerModel.call()
        """
        # In the future, a goal string and/or curriculum goal object
        # would be passed down here.
        plan = self.runtime.planner_tick()
        if self.config.fail_fast_on_invalid_plan:
            if not isinstance(plan, dict) or "steps" not in plan:
                raise ValueError("Planner returned invalid plan")
        return plan
```

### 5.2 Building the initial trace

```python
    def _build_initial_trace(self, plan: dict) -> PlanTrace:
        """
        Construct a PlanTrace with no steps yet, using the
        latest planner observation and tech state from runtime.
        """
        obs = self.runtime.get_latest_planner_observation()
        tech_state = self.runtime.get_tech_state()
        context_id = self.runtime.get_context_id()

        return PlanTrace(
            plan=plan,
            steps=[],
            tech_state=tech_state,
            planner_payload=obs.json_payload,
            context_id=context_id,
            virtue_scores={},
        )
```

### 5.3 Executing stage

Execution in v1 uses a **simple skill-executor seam**. The executor may live inside `AgentRuntime` or be passed in as part of `runtime` to keep M8 from touching BotCore directly. The exact code will match however `AgentRuntime` exposes “execute step” utilities. Conceptually:

```python
    def _execute_plan_into_trace(self, plan: dict, trace: PlanTrace) -> None:
        """
        Walk through the planner's steps and execute them via runtime.
        Record each low-level action as a TraceStep.
        """
        steps = plan.get("steps", [])
        max_steps = min(len(steps), self.config.max_skill_steps)

        for step_idx in range(max_steps):
            step_spec = steps[step_idx]
            # Delegate to runtime: resolve skill, generate actions, call BotCore
            action_results = self.runtime.execute_plan_step(step_spec, step_idx)

            # runtime returns a list[TraceStep] so M8 just extends the trace
            for ts in action_results:
                trace.steps.append(ts)

            # Simple fail-fast policy: if the last step failed, stop early
            if action_results and not action_results[-1].result.success:
                break
```

Notes:

- **No direct BotCore calls here**; this method assumes `runtime.execute_plan_step` wraps skill resolution + BotCore integration.
- This is where you later inject retry logic or hierarchical planning (e.g., tasks vs skills).

### 5.4 Critic stage (Recovering / self-eval hook)

```python
    def _maybe_run_critic(self, trace: PlanTrace) -> Optional[CriticResult]:
        if not self.config.enable_critic or not self.runtime.has_critic():
            return None

        try:
            payload = encode_for_critic(trace)
            critic = self.runtime.get_critic_model()
            result = critic.evaluate(payload)
            return result
        except Exception:
            # Critic is advisory; failures should not crash the loop
            return None
```

This is the seed of the **self-evaluation + retry** quality: later versions can inspect `CriticResult` and decide to rerun `planner_tick()` or adjust constraints.

### 5.5 Virtue scoring stage

```python
    def _maybe_run_virtues(self, trace: PlanTrace) -> None:
        if not self.config.log_virtue_scores or self.virtue_engine is None:
            return
        try:
            scores = self.virtue_engine.score_trace(
                trace=trace,
                context_id=trace.context_id,
            )
            trace.virtue_scores = scores
        except Exception:
            # Treat virtue scoring as best-effort telemetry
            pass
```

Virtue evaluation is intentionally kept at the **trace** level so that M4 remains independent of BotCore details.

### 5.6 Experience stage (logging & handoff to M10)

```python
    def _store_experience(self, result: EpisodeResult) -> None:
        tech_state_snapshot = dict(result.trace.tech_state.to_serializable())
        env_profile_name = self.runtime.get_env_profile_name()

        xp = Experience(
            episode_id=result.episode_id,
            trace=result.trace,
            critic_result=result.critic_result,
            env_profile_name=env_profile_name,
            context_id=result.trace.context_id,
            tech_state_snapshot=tech_state_snapshot,
            meta={
                "duration_sec": result.finished_at - result.started_at,
                "planner_model": self.runtime.get_planner_name(),
            },
        )
        self.experiences.add(xp)
```

This is the bridge from **control** (M8) to **learning** (M10).

---

## 6. Testing & Simulation

M8 is intentionally **offline-testable** with existing fakes:

- `tests/fakes/fake_bot_core.py`
- `tests/fakes/fake_llm_stack.py`
- `tests/fakes/fake_skills.py`
- `tests/test_agent_loop_stub.py`
- `tests/test_runtime_m6_m7_smoke.py`  

The goal is to validate the loop without running a real Minecraft server.

### 6.1 Runtime-driven tests (preferred path)

Instead of constructing BotCore/LLM/skills directly, M8 tests build a **fake or dummy AgentRuntime** that conforms to the real runtime API:

```python
from agent.loop import AgentLoop, AgentLoopConfig
from agent.experience import ExperienceBuffer
from tests.fakes.fake_runtime import FakeAgentRuntime   # small test helper


def test_agent_loop_episode_runs_end_to_end():
    runtime = FakeAgentRuntime()
    buffer = ExperienceBuffer()
    loop = AgentLoop(runtime=runtime, config=AgentLoopConfig(), experience_buffer=buffer)

    result = loop.run_episode(episode_id=1)

    assert result.plan  # planner was called
    assert len(result.trace.steps) > 0  # some execution happened
    assert len(buffer) == 1            # experience stored
```

Fake runtime responsibilities:

- `planner_tick()` returns a trivial plan with one or two steps.
- `get_latest_planner_observation()` returns a simple Observation.
- `execute_plan_step()` returns a small list of `TraceStep`s with `success=True`.

```python
# tests/fakes/fake_runtime.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from observation.trace_schema import PlanTrace, TraceStep
from spec.types import WorldState, Action, ActionResult


@dataclass
class _FakeObservation:
    """Minimal observation wrapper providing a json_payload field."""
    json_payload: Dict[str, Any]


@dataclass
class _FakeRuntimeConfig:
    """Minimal config object to satisfy AgentRuntime expectations."""
    context_id: str = "test_context"
    profile_name: str = "test_profile"


class FakeAgentRuntime:
    """
    Dummy AgentRuntime for M8 tests.

    Responsibilities:
      - Provide planner_tick() returning a tiny plan
      - Provide get_latest_planner_observation() for PlanTrace
      - Provide get_tech_state() and get_context_id()
      - Provide execute_plan_step() → list[TraceStep]
      - Omit critic_model so critic remains optional
    """

    def __init__(self) -> None:
        self.config = _FakeRuntimeConfig()
        self.planner_name = "fake_planner_v1"
        self._tick: int = 0
        self._position = {"x": 0.0, "y": 64.0, "z": 0.0}

    # ------------------------------------------------
    # Planner-facing API
    # ------------------------------------------------

    def planner_tick(self) -> Dict[str, Any]:
        """
        Return a trivial plan with a single step that moves the bot.
        """
        return {
            "steps": [
                {
                    "skill": "test_move",
                    "params": {"x": 1, "y": 64, "z": 0},
                }
            ]
        }

    def get_latest_planner_observation(self) -> _FakeObservation:
        """
        Minimal observation object with a json_payload dict.
        """
        self._tick += 1
        payload: Dict[str, Any] = {
            "tick": self._tick,
            "position": dict(self._position),
            "context_id": self.config.context_id,
        }
        return _FakeObservation(json_payload=payload)

    def get_tech_state(self) -> Dict[str, Any]:
        """
        Simple tech state stub for PlanTrace.tech_state.
        """
        return {"stage": "test_stage", "tier": "lv"}

    def get_context_id(self) -> str:
        """
        Context ID used by virtue scoring.
        """
        return self.config.context_id

    # ------------------------------------------------
    # Execution-facing API
    # ------------------------------------------------

    def execute_plan_step(
        self,
        step_spec: Dict[str, Any],
        step_idx: int,
    ) -> List[TraceStep]:
        """
        Execute a single plan step as one or more low-level actions.

        In this fake implementation, we:
          - build a single Action of type 'move_to'
          - update the internal position
          - wrap the result in a TraceStep
        """
        params = step_spec.get("params", {})
        target_x = float(params.get("x", self._position["x"]))
        target_y = float(params.get("y", self._position["y"]))
        target_z = float(params.get("z", self._position["z"]))

        world_before = self._make_world_state()
        action = Action(
            type="move_to",
            params={"x": target_x, "y": target_y, "z": target_z},
        )

        # Commit the move
        self._position["x"] = target_x
        self._position["y"] = target_y
        self._position["z"] = target_z

        result = ActionResult(
            success=True,
            error=None,
            details={"step_idx": step_idx},
        )
        world_after = self._make_world_state()

        trace_step = TraceStep(
            world_before=world_before,
            action=action,
            result=result,
            world_after=world_after,
            meta={
                "skill": step_spec.get("skill", "test_move"),
                "plan_step_idx": step_idx,
                "context_id": self.config.context_id,
            },
        )
        return [trace_step]

    # ------------------------------------------------
    # Optional critic API (not used in basic test)
    # ------------------------------------------------

    @property
    def critic_model(self) -> None:
        """
        No critic in the basic fake runtime; AgentLoop should handle None.
        """
        return None

    # ------------------------------------------------
    # Helpers
    # ------------------------------------------------

    def _make_world_state(self) -> WorldState:
        """
        Construct a minimal WorldState that satisfies TraceStep usage.
        """
        self._tick += 1
        return WorldState(
            tick=self._tick,
            position=dict(self._position),
            dimension="overworld",
            inventory=[],
            nearby_entities=[],
            blocks_of_interest=[],
            tech_state={},
            context={},
        )

```

### 6.2 Backwards-compat tests

Existing tests like `test_agent_loop_stub.py` should be updated rather than deleted:

- Preserve their expectations about:
  - `EpisodeResult` shape
  - `ExperienceBuffer` behavior
  - PlanTrace construction basics
- Extend them to assert:
  - `TraceStep`s are present after execution
  - Critic/virtue hooks do not crash the loop even if fakes are minimal.
```python
# tests/test_agent_loop_v1.py

from __future__ import annotations

from agent.loop import AgentLoop, AgentLoopConfig
from agent.experience import ExperienceBuffer
from tests.fakes.fake_runtime import FakeAgentRuntime


def test_agent_loop_episode_runs_end_to_end() -> None:
    """
    Basic integration test for M8:

      - Uses FakeAgentRuntime (no real Minecraft, LLM, or skills)
      - Verifies that:
          * planner was called
          * at least one TraceStep was produced
          * an Experience was stored in the buffer
    """
    runtime = FakeAgentRuntime()
    buffer = ExperienceBuffer()
    loop = AgentLoop(
        runtime=runtime,
        config=AgentLoopConfig(),
        experience_buffer=buffer,
    )

    result = loop.run_episode(episode_id=1)

    # Planner produced a plan
    assert isinstance(result.plan, dict)
    assert "steps" in result.plan
    assert len(result.plan["steps"]) > 0

    # Execution produced at least one trace step
    assert len(result.trace.steps) > 0

    # Experience was stored in the buffer
    assert len(buffer) == 1

    xp = buffer.last()
    assert xp is not None
    assert xp.episode_id == 1
    # Sanity check some metadata
    assert xp.env_profile_name == runtime.config.profile_name
    assert xp.context_id == result.trace.context_id

```

---

## 7. External Patterns Worth Mirroring

M8’s design is inspired by several external patterns, but implemented in a **pure-Python, config-driven** way:

- **Voyager / similar Minecraft agents**:
  - Episode-based loops over a learned skill library.
  - Replay buffers feeding into new skill generation.
- **Reflexion-style agents**:
  - Critic-based self-evaluation and retry loops.
- **RL frameworks (rollout/trace structure)**:
  - Clear distinction between observation, action, reward/metrics, and trace.

We steal the *ideas*, not the dependencies.

---

## 8. Completion Criteria for M8 (v1)

M8 counts as “v1 complete” when all of the following hold:

1. **Episode loop works with real runtime stubs**
   - `AgentLoop.run_episode()`:
     - Calls `runtime.planner_tick()`
     - Builds a `PlanTrace`
     - Executes at least one plan step via `runtime.execute_plan_step`
     - Returns a populated `EpisodeResult`.

2. **Trace logging is structurally sound**
   - `PlanTrace.steps` contains `TraceStep`s with:
     - `world_before`, `action`, `result`, `world_after`.
   - `planner_payload` is populated from `get_latest_planner_observation()`.

3. **Critic & virtue hooks are wired**
   - If a critic model exists, it is called via `encode_for_critic(trace)`.
   - If a virtue engine is provided, `trace.virtue_scores` is set.

4. **Experience buffering is active**
   - `ExperienceBuffer` gets one entry per episode when `store_experiences=True`.
   - Experiences include:
     - `episode_id`
     - `trace`
     - `critic_result`
     - basic metadata (duration, env profile).

5. **No low-level coupling**
   - M8 does not:
     - Import `bot_core.*` directly.
     - Import `llm_stack.*` directly.
     - Import `semantics.*` directly.
   - Instead, it uses the **public API of `AgentRuntime`** and the spec interfaces.

6. **Tests pass with fakes**
   - Updated `test_agent_loop_stub.py` (or `test_agent_loop_v1.py`) passes with:
     - Fake runtime
     - Fake virtue engine
     - Fake critic (optional).

When this is satisfied, M8 is a proper orchestration layer standing on top of M0–M7, ready to be connected to M9 (monitoring), M10 (learning), and M11 (curriculum) without ripping anything apart.
