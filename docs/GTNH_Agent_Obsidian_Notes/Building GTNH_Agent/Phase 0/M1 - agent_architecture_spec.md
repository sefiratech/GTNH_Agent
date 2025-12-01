**Purpose:**  
Unify Mineflayer + Voyager insights into a **single architecture spec**.

Overview:

- Extract from Mineflayer:
    
    - Bot lifecycle
        
    - World model
        
    - Pathfinding
        
    - Action abstraction
        
- Extract from Voyager:
    
    - Planner → Skill library → Execution loop
        
    - Reflection & learning
        
- **Dependencies:** `M0`
    
- **Difficulty:** ⭐⭐
    
- **Scalability notes:**
    
    - Produce one canonical architecture doc: diagrams + interfaces.
        
    - This is the contract everything else conforms to.


Details:

M1 is the “this is how the whole monster fits together” module. If you do this well, future-you suffers less. So let’s do it well.

---

# M1 · `agent_architecture_spec`

**Phase:** P0 – Foundations  
**Role:** Canonical architecture & interface spec for the GTNH Agent. No heavy logic, just contracts and diagrams.

M1 is a _design_ module: it defines **components**, **interfaces**, and **data flows** that other code modules must implement.

---

## 1. Responsibilities & Boundaries

### 1.1 What M1 owns

- The **conceptual architecture** of the agent, merging:
    
    - Mineflayer-style bot:
        
        - Bot lifecycle
            
        - World model
            
        - Pathfinding
            
        - Action abstraction
            
    - Voyager-style agent:
        
        - Planner → Skill library → Execution loop
            
        - Reflection & learning
            
- The **core interfaces** as Python protocols / abstract base classes:
    
    - `EnvProfile` usage (from M0)
        
    - `WorldState`, `Observation`, `Action`, `ActionResult`
        
    - `BotCore` interface
        
    - `Skill`, `SkillRegistry`
        
    - `PlannerModel`, `CodeModel`, `CriticModel`
        
    - `AgentLoop` interface / state machine outline
        
    - `ExperienceRecorder` for learning
        
- A **set of diagrams** & a markdown spec describing:
    
    - Data flow
        
    - Responsibility boundaries
        
    - What calls what
        

### 1.2 What M1 does _not_ do

- No actual pathfinding.
    
- No LLM calls.
    
- No real Minecraft integration.
    
- No learning logic.
    

It’s the blueprint; everything else is construction.

---

## 2. High-Level Architecture

### 2.1 Top-level data flow

+-------------------+         +-----------------+          +---------------------+
|    BotCore (M6)   |  ---> | Observation     |  ---> |  AgentLoop (M8)     |
| - connect            |         | Encoding (M7)   |         | - plan                      |
| - act()                  |         +-----------------+          | - select skills          |
+-------------------+                                                               |
											  v
										 +-------------+
	                                     |SkillRegistry (M5)
                                         | - skill defs
                                         +-------------+
                                                |
                                                v
                                        +------------------------+
                                        | LLM Stack (M2)          |
                                        | - PlanCodeModel      |
                                        | - ErrorModel              |
                                        | - ScribeModel            |
                                        +------------------------+
                                                |
                                                v
                                        +-------------------------+
                                        | VirtueLattice (M4)       |
                                        | WorldSemantics (M3)       |
                                        +-------------------------+
                                                  |
                                                  v
                                         +------------------------+
                                        | ExperienceRecorder   |
                                        | & SkillLearning (M10) |
                                         +------------------------+

M1’s job is to define the _shapes_ of these boxes and the contracts between them.

---

## 3. Core Interface Spec (Python)

Use `typing.Protocol` or abstract base classes; keep it light so everything else can depend on these without circular nonsense.

### 3.1 Basic types: observations, actions, results
Python:
```
# src/spec/types.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional


@dataclass
class WorldState:
    """Semantic snapshot of the world relevant to decision-making.

    This is not raw packets. It's a structured view that BotCore (M6) / world
    semantics (M3) produce for the AgentLoop and skills.
    """
    tick: int                                # current server tick number
    position: Mapping[str, float]           # {"x": ..., "y": ..., "z": ...}
    dimension: str                          # e.g. "overworld", "nether"
    inventory: List[Dict[str, Any]]         # list of item stacks
    nearby_entities: List[Dict[str, Any]]   # mobs, players, dropped items, etc.
    blocks_of_interest: List[Dict[str, Any]]  # ores, machines, machines, etc.
    tech_state: Dict[str, Any]              # inferred tech progression info
    context: Dict[str, Any]                 # misc extra metadata (phase, flags, etc.)


@dataclass
class Observation:
    """Compact encoding of WorldState for LLMs/tools.

    Produced by the observation encoder (M7).
    """
    json_payload: Dict[str, Any]            # LLM-ready JSON-like dict
    text_summary: str                       # optional textual summary for prompts


@dataclass
class Action:
    """Abstract action that AgentLoop can send to BotCore."""
    type: str                               # e.g. "move", "break_block", "use_item"
    params: Dict[str, Any]                  # parameters for the action


@dataclass
class ActionResult:
    """Result of executing an Action."""
    success: bool                           # did it work?
    error: Optional[str]                    # error message if not
    details: Dict[str, Any]                 # optional extra info (e.g. new position)

```

---

### 3.2 BotCore interface (Mineflayer-style body)
Python:
```
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

```

This is the Mineflayer-adjacent part: the “bot.”

---

### 3.3 Skill interface & registry (Voyager-style skills)
Python:
```
# src/spec/skills.py

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Protocol

from .types import WorldState, Action


class Skill(Protocol):
    """A reusable macro/behavior unit."""

    @property
    def name(self) -> str:
        """Unique name used in plans and registry."""
        ...

    def describe(self) -> Dict[str, Any]:
        """
        Return metadata for planners and UIs, for example:
        {
            "description": "Chop down a nearby tree and collect logs.",
            "params": {
                "min_logs": {"type": "int", "default": 8},
            },
            "preconditions": ["has_axe"],
            "effects": ["inventory.logs >= min_logs"],
        }
        """
        ...

    def suggest_actions(
        self,
        world: WorldState,
        params: Mapping[str, Any],
    ) -> List[Action]:
        """
        Given current world state and skill parameters,
        propose a sequence of Actions for BotCore to execute.

        This is where high-level intentions turn into concrete steps like:
        - move to position
        - break block
        - pick up item
        """
        ...


class SkillRegistry(Protocol):
    """Central registry of available skills."""

    def list_skills(self) -> List[str]:
        """Return all skill names."""
        ...

    def get_skill(self, name: str) -> Skill:
        """Fetch a skill by name; should raise if the skill is unknown."""
        ...

    def describe_all(self) -> Dict[str, Dict[str, Any]]:
        """
        Return metadata for all skills for use by planners and LLM prompts.

        Example shape:
        {
          "chop_tree": {...},
          "craft_planks": {...},
        }
        """
        ...

```

Skills are Voyager’s “code tools,” but abstracted into this spec.

---

### 3.4 LLM stack interfaces (planner, code, critic)
Python:
```
# src/spec/llm.py

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Protocol

from .types import Observation


class PlannerModel(Protocol):
    """Generates high-level plans from observations and goals."""

    def plan(
        self,
        observation: Observation,
        goal: str,
        skill_descriptions: Mapping[str, Dict[str, Any]],
        constraints: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """
        Return a structured plan, e.g.:

        {
          "steps": [
            {"skill": "chop_tree", "params": {...}},
            {"skill": "craft_planks", "params": {...}},
          ]
        }

        Concrete implementations (M2) can wrap this into a richer Plan
        dataclass if desired, but the contract here is JSON-like.
        """
        ...


class CodeModel(Protocol):
    """Generates or refines skill implementations or scripts."""

    def propose_skill_implementation(
        self,
        skill_spec: Mapping[str, Any],
        examples: List[Mapping[str, Any]],
    ) -> str:
        """
        Given a skill spec and example traces, return new/updated code as a string.

        M10 will own persistence / evaluation / rollout of this code.
        """
        ...


class CriticModel(Protocol):
    """Evaluates and reflects on plans or skill performance."""

    def evaluate_plan(
        self,
        observation: Observation,
        plan: Mapping[str, Any],
        virtue_scores: Mapping[str, float],
    ) -> Dict[str, Any]:
        """
        Return critique & suggested adjustments, e.g.:

        {
          "score": 0.82,
          "issues": ["too risky near lava", "ignores low food"],
          "suggested_changes": [...]
        }
        """
        ...


```

M1 only defines _interfaces_ here. M2 will implement them with local LLMs.

---

### 3.5 AgentLoop interface (Voyager-style control)
Python:
```
# src/spec/agent_loop.py

from __future__ import annotations

from typing import Any, Mapping, Protocol

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


```

Concrete implementations will later wire `PlannerModel`, `SkillRegistry`, `BotCore`, etc.

---

### 3.6 Experience & learning interfaces
Python:
```
# src/spec/experience.py

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Protocol

from .types import WorldState, Action, ActionResult


class ExperienceRecorder(Protocol):
    """Collects data about episodes for future learning."""

    def record_step(
        self,
        world_before: WorldState,
        action: Action,
        result: ActionResult,
        world_after: WorldState,
        meta: Mapping[str, Any],
    ) -> None:
        """Save a single transition."""
        ...

    def flush(self) -> None:
        """Force writing any buffered experience to disk or durable storage."""
        ...


class SkillLearner(Protocol):
    """Learns new skills or refines existing ones from recorded experience."""

    def propose_skill_updates(self) -> List[Dict[str, Any]]:
        """
        Analyze recorded experience and suggest:
        - new skills
        - updated skill implementations
        - deprecations

        Returned items are JSON-like dicts; a later module (M10)
        decides how to turn them into actual code / registry updates.
        """
        ...

```

M10 will implement these, but M1 specifies the interfaces.

3.7 \_\_init\_\_.py 
Python:
```
"""
Core interface specifications for the GTNH Agent.

This package defines stable contracts for:
- WorldState / Observation / Action types
- BotCore interface (the "body")
- Skill and SkillRegistry interfaces
- Planner / Code / Critic LLM interfaces
- AgentLoop interface (control logic)
- Experience recording / skill learning hooks

Everything else in the project is expected to import these abstractions,
not redefine them.
"""

from .types import WorldState, Observation, Action, ActionResult
from .bot_core import BotCore
from .skills import Skill, SkillRegistry
from .llm import PlannerModel, CodeModel, CriticModel
from .agent_loop import AgentLoop
from .experience import ExperienceRecorder, SkillLearner

__all__ = [
    "WorldState",
    "Observation",
    "Action",
    "ActionResult",
    "BotCore",
    "Skill",
    "SkillRegistry",
    "PlannerModel",
    "CodeModel",
    "CriticModel",
    "AgentLoop",
    "ExperienceRecorder",
    "SkillLearner",
]

```

---

## 4. System Diagrams

### 4.1 Component view

+---------------------+
| EnvProfile (M0)     |
+---------------------+
           |
           v
+---------------------+                        +---------------------+
| BotCore (M6)        |                          | LLM Stack (M2)      |
| - get_world_state() |                      | - PlannerModel      |
| - execute_action()  |                       | - CodeModel         |
+----------+----------+                        | - CriticModel       |
           |                                      +----------+----------+
           v                                                      |
+---------------------+                                        |
| ObservationEncoder  |                                   |
| (M7)                |                                                 |
+----------+----------+                                        |
           |                                                      |
           v                                                      v
      +-----------------------+               +--------------------+
      | AgentLoop (M8)        |               | VirtueLattice (M4)|
      | - set_goal()          |                    | WorldSemantics(M3)|
      | - step()              |                       +--------------------+
      +-----------------------+
                   |
                   v
          +-----------------------+
          | SkillRegistry (M5)    |
          +----------+------------+
                     |
                     v
             +--------------------+
             | ExperienceRecorder |
             | & SkillLearner     |
             |      (M10)         |
             +--------------------+

M1’s job is to define these **boxes and arrows** precisely, so implementation modules don’t fight over who owns what.

---

## 5. Local Testing / Simulation Strategy

M1 is mostly interfaces and docs, but you can still test it in useful ways.

### 5.1 Contract tests with fake implementations

You can create tiny, toy “fake” implementations to verify that interfaces compose logically.

Example: minimal fake BotCore + SkillRegistry + PlannerModel + AgentLoop.

Python:
```
# tests/test_architecture_integration.py

from __future__ import annotations

from typing import Any, Dict, List

from spec.types import WorldState, Action, ActionResult, Observation
from spec.bot_core import BotCore
from spec.skills import Skill, SkillRegistry
from spec.llm import PlannerModel, CriticModel
from spec.agent_loop import AgentLoop


class FakeBotCore(BotCore):
    def __init__(self) -> None:
        self._tick = 0
        self._pos: Dict[str, float] = {"x": 0.0, "y": 64.0, "z": 0.0}

    def connect(self) -> None:
        pass

    def disconnect(self) -> None:
        pass

    def get_world_state(self) -> WorldState:
        return WorldState(
            tick=self._tick,
            position=dict(self._pos),
            dimension="overworld",
            inventory=[],
            nearby_entities=[],
            blocks_of_interest=[],
            tech_state={},
            context={},
        )

    def execute_action(self, action: Action) -> ActionResult:
        if action.type == "move":
            dx = action.params.get("dx", 0)
            dz = action.params.get("dz", 0)
            self._pos["x"] += dx
            self._pos["z"] += dz
            self._tick += 1
            return ActionResult(
                success=True,
                error=None,
                details={"position": dict(self._pos)},
            )
        return ActionResult(
            success=False,
            error="unsupported_action",
            details={},
        )

    def tick(self) -> None:
        self._tick += 1


class FakeSkill(Skill):
    @property
    def name(self) -> str:
        return "move_forward"

    def describe(self) -> Dict[str, Any]:
        return {
            "description": "Move the agent forward by one block.",
            "params": {},
            "preconditions": [],
            "effects": ["position changes"],
        }

    def suggest_actions(self, world: WorldState, params: Dict[str, Any]) -> List[Action]:
        return [Action(type="move", params={"dx": 1, "dz": 0})]


class FakeSkillRegistry(SkillRegistry):
    def list_skills(self) -> List[str]:
        return ["move_forward"]

    def get_skill(self, name: str) -> Skill:
        if name != "move_forward":
            raise KeyError(name)
        return FakeSkill()

    def describe_all(self) -> Dict[str, Dict[str, Any]]:
        s = FakeSkill()
        return {s.name: s.describe()}


class FakePlanner(PlannerModel):
    def plan(
        self,
        observation: Observation,
        goal: str,
        skill_descriptions: Dict[str, Dict[str, Any]],
        constraints: Dict[str, Any],
    ) -> Dict[str, Any]:
        return {"steps": [{"skill": "move_forward", "params": {}}]}


class FakeCritic(CriticModel):
    def evaluate_plan(
        self,
        observation: Observation,
        plan: Dict[str, Any],
        virtue_scores: Dict[str, float],
    ) -> Dict[str, Any]:
        return {"ok": True, "score": 1.0}


class SimpleAgentLoop(AgentLoop):
    def __init__(
        self,
        bot: BotCore,
        skills: SkillRegistry,
        planner: PlannerModel,
        critic: CriticModel,
    ) -> None:
        self.bot = bot
        self.skills = skills
        self.planner = planner
        self.critic = critic
        self._goal = "test"
        self._last_plan: Dict[str, Any] = {}
        self._last_critique: Dict[str, Any] = {}

    def set_goal(self, goal: str, context: Dict[str, Any]) -> None:
        self._goal = goal

    def get_status(self) -> Dict[str, Any]:
        return {
            "goal": self._goal,
            "plan": self._last_plan,
            "critique": self._last_critique,
        }

    def step(self) -> None:
        world = self.bot.get_world_state()
        obs = Observation(json_payload={"tick": world.tick}, text_summary="test")

        plan = self.planner.plan(
            obs,
            self._goal,
            self.skills.describe_all(),
            constraints={},
        )
        self._last_plan = plan

        # Let the critic look at the plan (even if we ignore the result for now).
        self._last_critique = self.critic.evaluate_plan(
            obs,
            plan,
            virtue_scores={},
        )

        for step in plan["steps"]:
            skill = self.skills.get_skill(step["skill"])
            actions = skill.suggest_actions(world, step.get("params", {}))
            for action in actions:
                self.bot.execute_action(action)


def test_fake_architecture_round_trip():
    bot = FakeBotCore()
    skills = FakeSkillRegistry()
    planner = FakePlanner()
    critic = FakeCritic()
    agent = SimpleAgentLoop(bot, skills, planner, critic)

    agent.set_goal("go_somewhere", {})
    agent.step()
    status = agent.get_status()

    assert status["plan"]["steps"][0]["skill"] == "move_forward"
    assert status["critique"]["ok"] is True

```

This test is a sanity check: your **interfaces + flows** actually form a usable loop.

---

## 6. Libraries / APIs / Repos Worth Examining

Not for copy-paste, but for conceptual patterns.

- **Mineflayer (bot architecture, plugins, pathfinding):**
    
    - How they structure:
        
        - Bot object
            
        - Plugins
            
        - Inventory & world model
            
- **Voyager (agent & skill learning):**
    
    - Their:
        
        - Planner/skill/critic structure
            
        - How skills are stored as code snippets
            
        - Experience and reflection loop
            
- **Python libraries for interface hygiene:**
    
    - `typing.Protocol`, `abc` – for clean, mockable interfaces
        
    - `dataclasses` – for light, immutable-ish data containers
        
- **Patterns from RL frameworks (for inspiration only):**
    
    - OpenAI Gym / Gymnasium: environment interfaces (`reset`, `step`).
        
    - RLlib / similar: how they separate environment from policy.
        

You’re not implementing RL exactly, but their separation of concerns is worth imitating.

---

## 7. Completion Criteria for M1

M1 is “done enough” when:

1. **Canonical spec exists & is readable**
    
    - A markdown doc (e.g. `docs/architecture.md`) showing:
        
        - Component diagrams (like above)
            
        - Short descriptions of each interface
            
    - The diagrams match the actual Python interfaces.
        
2. **Interfaces live in one place**
    
    - All core interfaces defined under `src/spec/` (or similar).
        
    - Other modules depend on these interfaces, not ad hoc custom ones.
        
3. **Fake end-to-end test passes**
    
    - A toy implementation (like `FakeBotCore` + `SimpleAgentLoop`) composes without type or logic conflicts.
        
    - CI runs these tests on each commit that touches the spec.
        
4. **No Minecraft or LLM code inside M1**
    
    - Module contains _only_:
        
        - Type definitions
            
        - Protocols / ABCs
            
        - Architecture docs
            
    - That guarantees M1 doesn’t explode when you inevitably refactor implementation details.
        
5. **Stable enough to freeze**
    
    - You’re not constantly changing these interfaces every time you tweak a downstream implementation.
        
    - Minor revisions are expected, but the top-level contracts and data types are mostly stable.
        

Once those boxes are checked, M1 becomes the “law” the rest of the project obeys. And you stop arguing with yourself in three months about what `AgentLoop` is supposed to do.