## 1. What this Inter-Phase Integration should produce

By the end of Inter-Phase 0,1,2 you want:

1. A **single bootstrap path**:
```
# high level
env_profile  -> M0
core types   -> M1
llm stack    -> M2
semantics    -> M3
virtues      -> M4
skills       -> M5
botcore spec -> M6
encoder      -> M7
--------------------------------
-> AgentRuntime / AgentShell

```

A **single function** that other phases can call:
python:
```
from agent.bootstrap import build_agent_runtime

runtime = build_agent_runtime(profile="dev_local")
plan = runtime.planner_tick()

```

1. **Extension hooks** ready for:
    
    - Phase 3: agent loop / control (M8)
        
    - Phase 4: logging/experience/memory (whatever you stick there later)
        

Basically: everything before M8 should be callable as “a service” with one unified entrypoint.

---

## 2. Recommended integration artifacts

Here’s what I’d add now. Think of this as the Inter-Phase 0,1,2 deliverables.

### A. `src/agent/bootstrap.py`

Goal: wire **M0 → M7** into a single `build_agent_runtime()` call.

Shape:
```
# src/agent/bootstrap.py

from __future__ import annotations

from typing import Optional

from agent.logging_config import configure_logging
from agent.runtime_m6_m7 import AgentRuntime, AgentRuntimeConfig
from semantics.loader import load_semantics_db
from semantics.schema import TechState
from observation.testing import DummySemanticsDB, make_minimal_tech_state
from llm_stack.models import load_planner_model, load_critic_model
from env.loader import load_env_profile
from virtues.engine import VirtueEngine
from skills.registry import SkillRegistry
from bot_core.runtime import get_bot_core  # stub for now, real later


def build_agent_runtime(
    profile: str = "dev_local",
    use_dummy_semantics: bool = False,
) -> AgentRuntime:
    """
    Inter-Phase 0,1,2 bootstrap.

    Wires:
      - M0 env profiles
      - M1 core types (implicitly used by everyone)
      - M2 LLM stack (planner/critic loaders)
      - M3 semantics (TechState + SemanticsDB)
      - M4 virtue engine (initialized, even if not used yet)
      - M5 skill registry (initialized, even if M8 isn't calling it yet)
      - M6 BotCore (via get_bot_core)
      - M7 observation encoder (via AgentRuntime)
    """
    configure_logging()

    env = load_env_profile(profile)  # M0
    # you may already have this: env.models, env.minecraft, etc.

    # --- M3: semantics + tech state ---
    if use_dummy_semantics:
        semantics_db = DummySemanticsDB()
        tech_state: TechState = make_minimal_tech_state()
    else:
        semantics_db, tech_state = load_semantics_db(env)

    # --- M4: virtue lattice ---
    virtue_engine = VirtueEngine.from_config(env)  # not yet used by runtime
    # Keep reference for future M8 hook; don't need it in AgentRuntime yet.

    # --- M5: skills ---
    skills = SkillRegistry.from_config(env, semantics_db=semantics_db)
    # Also held for future M8; not injected into AgentRuntime yet.

    # --- M2: planner / critic models ---
    planner_model = load_planner_model(env)
    critic_model = load_critic_model(env)  # may return None while critic is stub

    # --- M6: BotCore ---
    bot_core = get_bot_core()

    # --- AgentRuntime (M6 + M7) ---
    runtime_config = AgentRuntimeConfig(
        context_id=env.context_id,       # from M4/M0 config
        initial_tech_state=tech_state,
    )

    runtime = AgentRuntime(
        bot_core=bot_core,
        semantics_db=semantics_db,
        planner_model=planner_model,
        critic_model=critic_model,
        config=runtime_config,
    )

    # You might want to return (runtime, virtue_engine, skills)
    # later for M8. For now, just runtime is fine.
    return runtime

```

This is your **Inter-Phase spine**: one function that stands on top of M0–M7.

You don’t have to implement every call exactly like that today (some modules are still partial), but that’s the _shape_ you want.

---

### B. `tools/agent_demo.py` (manual exercise of the pipeline)

Simple Phase 0–2 end-to-end demo, no M8:
```
# tools/agent_demo.py

from __future__ import annotations

from agent.bootstrap import build_agent_runtime
from agent.logging_config import configure_logging
import logging


def main() -> None:
    configure_logging(logging.DEBUG)

    runtime = build_agent_runtime(profile="dev_local", use_dummy_semantics=True)

    plan = runtime.planner_tick()
    obs = runtime.get_latest_planner_observation()

    print("=== Planner Observation (summary) ===")
    print(obs.text_summary)
    print()

    print("=== Planner Plan (dict) ===")
    print(plan)


if __name__ == "__main__":
    main()

```

You run:
bash:
```
python3 tools/agent_demo.py

```

If that prints something coherent, you have a working Phase 0–2 pipeline from config → semantics → observation → planner.

### C. Integration tests for the bootstrap

You’ve already got M7 tests. Now you add a tiny one:
```
# tests/test_phase012_bootstrap.py
from agent.bootstrap import build_agent_runtime
from spec.types import Observation

def test_phase012_bootstrap_planner_flow():
    runtime = build_agent_runtime()
    plan = runtime.planner_tick()
    obs = runtime.get_latest_planner_observation()

    assert isinstance(plan, dict)
    assert isinstance(obs, Observation)
    keys = obs.json_payload.keys()
    for k in [
        "tech_state",
        "agent",
        "inventory_summary",
        "machines_summary",
        "nearby_entities",
        "env_summary",
        "craftable_summary",
        "context_id",
        "text_summary",
    ]:
        assert k in keys

```

That’s your **Inter-Phase 0,1,2 contract**: if this dies, some module broke the chain.

---

## 3. “Preparations for Phases 3 & 4”

This is basically: **don’t paint yourself into a corner**.

While doing Inter-Phase 0,1,2, keep these seams explicit:

1. **AgentRuntime as a service**
    
    Phase 3 (M8) should _not_ recreate any wiring. It should only:
python:
```
runtime = build_agent_runtime(...)
loop = AgentLoop(runtime=runtime, virtue_engine=..., skills=...)
loop.run_episode(...)

```

1. **PlanTrace & Critic hooks**
    
    You already have PlanTrace & `encode_for_critic`. That’s where Phase 4 (experience / memory / logging) will latch on:
    
    - After each plan execution:
        
        - emit `PlanTrace`
            
        - pass to critic
            
        - log / store Experience
            
    
    Don’t hard-wire critiquing into low-level code; keep it at the “loop / M8” level.
    
2. **Logging / debug knobs stay local**
    
    All the logging we wired into M7 is gated on logging level. Phase 3 & 4 shouldn’t have to touch observation internals to turn tracing on/off; they should just adjust logging config and maybe set a few flags in env.


```
# src/agent/experience.py
"""
Phase 4 preparation: experience structures.

This module defines small, generic containers that can be used by
the agent loop (M8) and any future experience / memory system:

  - Experience: one episode's plan + trace + critic result
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


@dataclass
class Experience:
    """
    A single experience record, usually produced at the end of an episode.

    Fields:
        trace:
            PlanTrace containing:
              - plan emitted by the planner
              - execution steps (may be empty at first)
              - tech_state, context_id, planner_observation, virtue_scores

        critic_result:
            Output from the critic model, if enabled. Shape is intentionally
            loose (Dict[str, Any]) so different critic backends can be used.

        timestamp:
            Seconds since epoch when the experience was recorded.

        episode_id:
            Optional id for grouping experiences by logical episode.

        meta:
            Free-form metadata (profile name, seed, environment tags, etc.).
    """
    trace: PlanTrace
    critic_result: Optional[Dict[str, Any]]
    timestamp: float = field(default_factory=lambda: time.time())
    episode_id: Optional[int] = None
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperienceBuffer:
    """
    Minimal in-memory buffer for experiences.

    This is intentionally simple for Phase 3/4 prep:
      - append-only list
      - trivial retrieval API

    Later you can:
      - add persistence hooks (write to disk, DB, etc.)
      - add sampling strategies (for training)
      - add filters / indexes
    """
    experiences: List[Experience] = field(default_factory=list)

    def add(self, experience: Experience) -> None:
        """Append a new experience to the buffer."""
        self.experiences.append(experience)

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.experiences)

    def last(self) -> Optional[Experience]:
        """Return the most recent experience, or None if buffer is empty."""
        return self.experiences[-1] if self.experiences else None

```


```
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
    """
    enable_critic: bool = True
    store_experiences: bool = True
    max_planner_calls: int = 1


@dataclass
class EpisodeResult:
    """
    Summary of a single run_episode invocation.

    Fields:
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
    plan: Dict[str, Any]
    trace: PlanTrace
    critic_result: Optional[Dict[str, Any]]
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

        # Build a PlanTrace with no steps yet.
        # Future M8 work will attach real TraceStep entries.
        trace = PlanTrace(
            plan=plan,
            steps=[],
            tech_state=self.runtime.current_tech_state,
            context_id=self.runtime.config.context_id,
            planner_payload=obs.json_payload,
            virtue_scores={},  # virtues not yet integrated into the loop
        )

        critic_result: Optional[Dict[str, Any]] = None

        if self.config.enable_critic and getattr(self.runtime, "critic_model", None):
            try:
                critic_payload = encode_for_critic(trace)
                critic_result = self.runtime.critic_model.evaluate(critic_payload)
            except Exception as exc:  # defensive: critic must not crash the loop
                logger.warning("Critic evaluation failed: %r", exc)
                critic_result = None

        finished_at = time.time()

        result = EpisodeResult(
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

```
# tests/test_agent_loop_stub.py

from agent.bootstrap import build_agent_runtime
from agent.loop import AgentLoop, AgentLoopConfig
from agent.experience import ExperienceBuffer, Experience
from observation.trace_schema import PlanTrace


def test_agent_loop_run_episode_minimal():
    runtime = build_agent_runtime()
    buffer = ExperienceBuffer()
    loop = AgentLoop(
        runtime=runtime,
        experience_buffer=buffer,
        config=AgentLoopConfig(
            enable_critic=True,
            store_experiences=True,
            max_planner_calls=1,
        ),
    )

    result = loop.run_episode(episode_id=1)

    # Basic structural checks
    assert isinstance(result.plan, dict)
    assert isinstance(result.trace, PlanTrace)
    # Critic may be a dummy; just assert it's either dict or None
    assert result.critic_result is None or isinstance(result.critic_result, dict)

    # Experience logging should have recorded exactly one entry
    assert len(buffer) == 1
    exp = buffer.last()
    assert isinstance(exp, Experience)
    assert exp.episode_id == 1
    assert isinstance(exp.trace, PlanTrace)

```



---

## 4. So what _should_ you do next?

Given your pattern:

> “When I finish a phase, I integrate that phase with everything before it.”

You nailed Phase 2 (M6 & M7).  
Now the right move is:

- **Do Inter-Phase 0,1,2 now**, in this constrained sense:
    
    - Create `build_agent_runtime` (or equivalent).
        
    - Add a tiny demo script.
        
    - Add one integration test that proves the whole stack M0–M7 works together.
        

Don’t overreach into full “agent loop” territory yet. That belongs to M8 and the next integration round.

So no, you’re not overdoing it. This is exactly the moment to stitch 0–2 together so Future You doesn’t have to debug a 7-module integration _and_ a brand new agent loop at the same time.