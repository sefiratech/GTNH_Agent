Here’s your Phase 0 integration stitched together so the thing feels like _one_ system instead of two disjoint modules that pretend not to know each other.

---

## Phase 0 Integrated Architecture – M0 + M1

### 1. High-level data flow

**Startup path:**

1. `env.loader.load_environment()` (M0)
    
    - Reads `env.yaml`, `minecraft.yaml`, `models.yaml`, `hardware.yaml`
        
    - Builds an `EnvProfile` describing:
        
        - active profile name (`dev`, `cpu_only`, etc.)
            
        - bot mode (`forge_mod` vs `external_client`)
            
        - Minecraft connection (path, version, world name, connection mode)
            
        - model backend / model names (for M2 later)
            
        - hardware limits
            
2. This `EnvProfile` is handed to a **bootstrap** function that:
    
    - Chooses / constructs the concrete `BotCore` implementation (M6, later)
        
    - Chooses the AgentLoop implementation (M8, later)
        
    - Injects **spec** interfaces:
        
        - `BotCore` (body)
            
        - `SkillRegistry` (M5, later)
            
        - `PlannerModel` & `CriticModel` (M2, later)
            
        - `ExperienceRecorder` (M10, later)
            
3. The runtime loop then does:
```
load_environment()  →  create_bot_core(env)    →  create_agent_loop(...)
                     →  while running:
                           bot.tick()
                           agent.step()

```
Phase 0 doesn’t implement the real BotCore / Planner / Critic yet, but **the wiring + contracts** exist and are testable.

---

### 2. Module interaction map (Phase 0 only)

- **M0 · `env.loader`**
    
    - Exports: `EnvProfile`, `load_environment()`
        
    - Used by: runtime bootstrap
        
- **M1 · `spec.*`**
    
    - `spec.types` → `WorldState`, `Observation`, `Action`, `ActionResult`
        
    - `spec.bot_core` → `BotCore`
        
    - `spec.skills` → `Skill`, `SkillRegistry`
        
    - `spec.llm` → `PlannerModel`, `CodeModel`, `CriticModel`
        
    - `spec.agent_loop` → `AgentLoop`
        
    - `spec.experience` → `ExperienceRecorder`, `SkillLearner`
        
    - Used by: runtime bootstrap + future modules (M2, M5, M6, M8, M10)
        

So Phase 0 integration is basically:

> “EnvProfile in, AgentLoop wiring ready, fake pipeline test passing.”

---

## 3. Unified integration code (Phase 0 bootstrap)

This is a Phase 0 “app entrypoint” sketch that shows how M0 and M1 line up.  
It uses **placeholder implementations** where later modules will plug in concrete code.

You can drop this as something like:  
`src/app/runtime.py` (or `src/gtnh_agent/runtime.py` if you’re feeling fancy).
Python:
```
# src/app/runtime.py

from __future__ import annotations  # allow forward type hints

from dataclasses import dataclass   # for simple container types
from typing import Any, Dict        # generic dict typing

from env.loader import EnvProfile, load_environment  # M0: environment + config loader

from spec import (                  # M1: canonical interfaces
    AgentLoop,
    BotCore,
    SkillRegistry,
    PlannerModel,
    CriticModel,
    ExperienceRecorder,
    WorldState,
    Observation,
    Action,
    ActionResult,
)

# ------------------------------
# Phase 0: placeholder concrete types
# These are intentionally dumb. Real implementations arrive in M2, M5, M6, M8, M10.
# ------------------------------


@dataclass
class NoopSkillRegistry(SkillRegistry):
    """Temporary empty skill registry for Phase 0 wiring."""

    def list_skills(self) -> list[str]:  # type: ignore[override]
        return []  # no skills yet; M5 will replace this

    def get_skill(self, name: str):  # type: ignore[override]
        raise KeyError(f"Skill '{name}' not implemented in Phase 0")

    def describe_all(self) -> Dict[str, Dict[str, Any]]:  # type: ignore[override]
        return {}  # no metadata yet


@dataclass
class FakePlanner(PlannerModel):
    """Minimal planner stub for wiring tests and Phase 0 sanity."""

    def plan(  # type: ignore[override]
        self,
        observation: Observation,
        goal: str,
        skill_descriptions: Dict[str, Dict[str, Any]],
        constraints: Dict[str, Any],
    ) -> Dict[str, Any]:
        # For now, just return an empty plan; M2 will implement real logic.
        return {"steps": []}


@dataclass
class FakeCritic(CriticModel):
    """Minimal critic stub; just says 'ok' to everything for now."""

    def evaluate_plan(  # type: ignore[override]
        self,
        observation: Observation,
        plan: Dict[str, Any],
        virtue_scores: Dict[str, float],
    ) -> Dict[str, Any]:
        # No real virtue scoring in Phase 0; M4 + M2 will handle that.
        return {"ok": True, "score": 1.0}


@dataclass
class DummyExperienceRecorder(ExperienceRecorder):
    """No-op recorder; just keeps a counter for now."""

    steps_recorded: int = 0  # simple counter so you know calls are happening

    def record_step(  # type: ignore[override]
        self,
        world_before: WorldState,
        action: Action,
        result: ActionResult,
        world_after: WorldState,
        meta: Dict[str, Any],
    ) -> None:
        self.steps_recorded += 1  # bump count; real M10 will persist data

    def flush(self) -> None:  # type: ignore[override]
        # In Phase 0, nothing to flush.
        pass


@dataclass
class DummyBotCore(BotCore):
    """Very simple BotCore stub.

    Enough to prove that:
    - EnvProfile exists
    - WorldState can be created
    - AgentLoop can call into BotCore
    """

    env: EnvProfile           # the resolved environment settings
    tick_counter: int = 0     # pretend 'server tick' counter

    def connect(self) -> None:  # type: ignore[override]
        # In Phase 0, this might just log "connected".
        # Real M6 will actually attach to Forge / external client.
        print(f"[DummyBotCore] connect() using profile '{self.env.name}'")

    def disconnect(self) -> None:  # type: ignore[override]
        # Mirror of connect; in Phase 0 just for lifecycle tracing.
        print("[DummyBotCore] disconnect()")

    def get_world_state(self) -> WorldState:  # type: ignore[override]
        # Build a minimal WorldState; M6 + M3 will fill in real semantics.
        return WorldState(
            tick=self.tick_counter,                      # fake tick count
            position={"x": 0.0, "y": 64.0, "z": 0.0},   # spawn-ish position
            dimension="overworld",                      # default dimension
            inventory=[],                               # no items yet
            nearby_entities=[],                         # none tracked yet
            blocks_of_interest=[],                      # no special blocks yet
            tech_state={},                              # unknown tech state
            context={"profile": self.env.name},         # tag with active profile
        )

    def execute_action(self, action: Action) -> ActionResult:  # type: ignore[override]
        # Phase 0: accept any action, increment tick, pretend success.
        self.tick_counter += 1
        print(f"[DummyBotCore] execute_action(): {action.type} {action.params}")
        return ActionResult(
            success=True,
            error=None,
            details={"tick": self.tick_counter},
        )

    def tick(self) -> None:  # type: ignore[override]
        # Advance fake tick; real M6 would pump event loop, socket IO, etc.
        self.tick_counter += 1


@dataclass
class SimpleAgentLoop(AgentLoop):
    """Minimal AgentLoop implementation for Phase 0.

    Proves that:
    - BotCore, PlannerModel, SkillRegistry, CriticModel can compose
    - EnvProfile + spec.* are coherent
    """

    bot: BotCore                 # concrete BotCore implementation
    skills: SkillRegistry        # skill registry (empty for now)
    planner: PlannerModel        # planner (fake)
    critic: CriticModel          # critic (fake)
    recorder: ExperienceRecorder # experience recorder (no-op)

    _goal: str = "idle"          # current top-level goal
    _plan: Dict[str, Any] | None = None  # last plan (raw dict)
    _last_result: ActionResult | None = None  # last action result
    _error_state: str | None = None          # any error string

    def set_goal(self, goal: str, context: Dict[str, Any]) -> None:  # type: ignore[override]
        # Store goal and optionally use context later (e.g. tech phase).
        self._goal = goal
        # In Phase 0 we ignore context; M3/M4/M11 will care later.

    def get_status(self) -> Dict[str, Any]:  # type: ignore[override]
        # Provide a debug snapshot useful for logging / UI.
        return {
            "goal": self._goal,
            "plan": self._plan,
            "last_result": self._last_result,
            "error_state": self._error_state,
        }

    def step(self) -> None:  # type: ignore[override]
        # 1) Observe the world via BotCore.
        world_before = self.bot.get_world_state()

        # 2) Encode into an Observation (M7 will eventually own this logic).
        obs = Observation(
            json_payload={"tick": world_before.tick},  # trivial encoding
            text_summary=f"Tick {world_before.tick}",  # placeholder summary
        )

        # 3) Get a plan from the planner if we don't have one yet.
        if self._plan is None:
            self._plan = self.planner.plan(
                obs,
                self._goal,
                self.skills.describe_all(),
                constraints={},
            )

        # 4) Let the critic look at the plan (ignored in Phase 0).
        _critique = self.critic.evaluate_plan(
            obs,
            self._plan,
            virtue_scores={},
        )

        # 5) Execute each step by mapping skills → actions → BotCore.
        for step in self._plan.get("steps", []):
            skill_name = step["skill"]
            params = step.get("params", {})

            # Look up the Skill; will explode in Phase 0 if plan references any.
            skill = self.skills.get_skill(skill_name)
            actions = skill.suggest_actions(world_before, params)

            for action in actions:
                result = self.bot.execute_action(action)
                self._last_result = result

                # 6) Record experience for learning (no-op recorder in Phase 0).
                world_after = self.bot.get_world_state()
                self.recorder.record_step(
                    world_before=world_before,
                    action=action,
                    result=result,
                    world_after=world_after,
                    meta={"goal": self._goal},
                )

        # Note: in Phase 0 we keep reusing the same plan; M8 will add invalidation.


# ------------------------------
# Bootstrap / runtime glue for Phase 0
# ------------------------------


def create_phase0_agent(env: EnvProfile) -> tuple[BotCore, AgentLoop]:
    """Wire M0 (EnvProfile) + M1 (spec) into a runnable stub agent.

    Later phases will:
    - swap DummyBotCore for a real Forge/external client core (M6)
    - replace FakePlanner/FakeCritic with real LLM-backed models (M2)
    - swap NoopSkillRegistry with a populated registry (M5)
    - use a real ExperienceRecorder (M10)
    """
    bot = DummyBotCore(env=env)                      # body powered by EnvProfile
    skills = NoopSkillRegistry()                     # empty skill registry for now
    planner = FakePlanner()                          # trivial planner
    critic = FakeCritic()                            # trivial critic
    recorder = DummyExperienceRecorder()             # simple counter

    agent = SimpleAgentLoop(                         # assemble AgentLoop
        bot=bot,
        skills=skills,
        planner=planner,
        critic=critic,
        recorder=recorder,
    )
    return bot, agent


def main() -> None:
    """Entry point for a Phase 0 dry-run of the architecture."""
    # 1) Resolve environment from config (M0).
    env = load_environment()
    print(f"[main] Loaded environment profile: {env.name} (bot_mode={env.bot_mode})")

    # 2) Build BotCore + AgentLoop using that environment.
    bot, agent = create_phase0_agent(env)

    # 3) Connect the bot (no-op in Phase 0, but traces lifecycle).
    bot.connect()

    try:
        # 4) Set a simple goal to exercise the loop.
        agent.set_goal("phase0_sanity_check", context={})

        # 5) Run a few iterations to prove nothing explodes.
        for i in range(3):
            print(f"[main] --- Agent step {i} ---")
            bot.tick()   # advance BotCore time
            agent.step() # run one agent iteration
            print(f"[main] status = {agent.get_status()}")
    finally:
        # 6) Always disconnect cleanly.
        bot.disconnect()


if __name__ == "__main__":
    main()  # allow `python -m app.runtime` style execution

```

You won’t _keep_ all these stubs forever, but this file shows:

- How **M0 EnvProfile** flows into the **BotCore** and **AgentLoop**
    
- Where the LLM stack, skills, and learning will plug in
    
- How the event loop is structured (`tick` / `step`)
    

And you can run it _today_ without M2/M5/M6 being implemented yet.

---

## 4. Testing, logging & failure points

### Testing hooks

- **Unit tests (already existing):**
    
    - `tests/test_env_loader.py` → EnvProfile loading & validation (M0)
        
    - `tests/test_architecture_integration.py` → fake spec wiring (M1)
        
- **Optional new tests using this runtime:**
    
    - A tiny smoke test:
Python:
```
# tests/test_phase0_runtime.py

from env.loader import load_environment
from app.runtime import create_phase0_agent


def test_phase0_bootstrap_runs():
    env = load_environment()
    bot, agent = create_phase0_agent(env)

    bot.connect()
    agent.set_goal("test_goal", context={})
    bot.tick()
    agent.step()
    status = agent.get_status()
    bot.disconnect()

    assert status["goal"] == "test_goal"
    assert "plan" in status


```

- - This ensures **EnvProfile + spec wiring + bootstrap** all cohere.
        

### Logging

For now you’re spewing `print(...)` like an animal. When you stop prototyping:

- Replace `print` with Python’s `logging`:
    
    - Create a `logger = logging.getLogger("gtnh_agent.runtime")`
        
    - Use `logger.info()`, `logger.debug()`, `logger.warning()`, etc.
        
- This lets you:
    
    - Capture logs from CI
        
    - Turn on debug for just one module
        
    - Eventually feed logs into Scribe / ExperienceRecorder
        

Phase 0: prints are fine.  
Post-M2: use proper logging or it’s going to be a mess.

### Failure points to watch

At Phase 0, likely explosions:

1. **EnvProfile misconfig**
    
    - `load_environment()` raises:
        
        - `FileNotFoundError` for YAML
            
        - `ValueError` for invalid versions / bot_mode
            
    - Runtime should treat these as **fatal at startup**.
        
2. **SkillRegistry empty but planner tries to use skills**
    
    - In this stub, `NoopSkillRegistry.get_skill()` raises `KeyError`.
        
    - That’s intentional: it reminds you you’re running Phase 0 wiring, not real skills.
        
3. **LLM stack not wired yet**
    
    - `FakePlanner` returns `{"steps": []}`, so AgentLoop does nothing.
        
    - Once M2 lands, failures will include:
        
        - backend not reachable
            
        - model path wrong
            
        - VRAM issues
            
4. **BotCore assumptions**
    
    - `DummyBotCore` doesn’t use the real env.minecraft paths or connection methods.
        
    - When M6 swaps in a real core, failure modes become:
        
        - cannot connect to server / SP
            
        - world not found
            
        - version mismatch
            

The whole point of Phase 0 integration is to ensure:  
**when you plug in M2/M3/M4/M5/M6 later, they snap into a stable skeleton, not a ball of spaghetti.**

You’ve got that skeleton now.