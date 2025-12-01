## What “integrating Phase 0 and 1” _actually_ means

Right now:

- **Phase 0** gives you:
    
    - `env.yaml` / `models.yaml` / `hardware.yaml`
        
    - `EnvProfile` loader + `validate_env` stuff
        
    - “this is the runtime profile we’re pretending exists”
        
- **Phase 1** gives you:
    
    - `run_phase1_planning_episode(world, goal, virtue_context_id, *, episode_id=None)`
        
    - fake planner + fake virtue lattice
        
    - real semantics stub (M3) + real skills (M5)
        
    - offline integration tests
        

Integration at this point should _not_ mean:

- “Phase 1 reaches down into every P0 detail and starts depending on env internals.”
    

It _should_ mean:

> “Given an `EnvProfile` from P0, I can spin up a `WorldState` snapshot and run a Phase-1 planning episode in an automated way.”

That’s it. No heroic coupling.

---

## Minimal, sane integration plan

### 1. Add a tiny adapter: “P0 → WorldState”

Create:
python:
```
# src/integration/adapters/m0_env_to_world.py

from typing import Dict, Any

from spec.types import WorldState
from env_loader import EnvProfile  # whatever you called it in M0


def world_from_env_profile(profile: "EnvProfile") -> WorldState:
    """
    Phase 1 adapter: build a *planning* WorldState from an EnvProfile.

    This is intentionally lossy / approximate in Phase 1:
    - It does NOT attempt to describe actual chunk contents.
    - It encodes only what the planner / semantics care about at this phase:
      - selected env name
      - modpack / GTNH version
      - maybe tech_hint (if you added such a field)
      - world seed placeholder if available
    """
    data: Dict[str, Any] = {
        "env_name": profile.name,
        "modpack": getattr(profile, "modpack_id", None),
        "gtnh_version": getattr(profile, "gtnh_version", None),
        "world_seed": getattr(profile, "world_seed", None),
        "meta": {
            "profile_path": str(getattr(profile, "source_path", "")),
            "phase": "p1_offline",
        },
    }

    # WorldState will eventually be a real structured type.
    # For now, if it's a dataclass or pydantic model, just adapt to it.
    try:
        return WorldState.from_dict(data)  # type: ignore[attr-defined]
    except AttributeError:
        # If WorldState is just a typed alias for Dict[str, Any] in Phase 1,
        # this is enough.
        return WorldState(data)  # type: ignore[call-arg]

```

### 2. Add a small CLI harness for offline planning

Hook them together in a tool:
```
# src/cli/phase1_offline.py

import argparse
import json

from env_loader import load_env_profile  # M0
from integration.adapters.m0_env_to_world import world_from_env_profile
from integration.phase1_integration import run_phase1_planning_episode


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 1 offline planner (P0 env + P1 integration)."
    )
    parser.add_argument("--env", required=True, help="Env profile name (from env.yaml)")
    parser.add_argument("--goal", required=True, help="Natural language planning goal")
    parser.add_argument(
        "--virtue-context-id",
        default="lv_coke_ovens",
        help="Virtue context id (e.g. lv_coke_ovens, lv_resources)",
    )
    args = parser.parse_args()

    # P0: load env profile
    profile = load_env_profile(args.env)

    # Adapter: EnvProfile -> WorldState
    world = world_from_env_profile(profile)

    # P1: run planning episode
    best_plan, scores = run_phase1_planning_episode(
        world=world,
        goal=args.goal,
        virtue_context_id=args.virtue_context_id,
    )

    print(json.dumps(
        {
            "best_plan": best_plan,
            "scores": scores,
        },
        indent=2,
        sort_keys=True,
    ))


if __name__ == "__main__":
    main()

```

### 3. Add a tiny integration test for P0↔P1 wiring

Something like:
python:
```
# tests/test_p0_p1_env_bridge.py

from integration.adapters.m0_env_to_world import world_from_env_profile
from integration.phase1_integration import run_phase1_planning_episode

from env_loader import EnvProfile


class DummyEnvProfile(EnvProfile):  # or just a simple dummy if EnvProfile is a dataclass
    pass


def test_p0_env_to_p1_planning_bridge(monkeypatch):
    profile = DummyEnvProfile(
        name="test_env",
        # fill whatever is required by your real EnvProfile
    )

    world = world_from_env_profile(profile)

    # Fake world_state methods if needed
    if not hasattr(world, "to_summary_dict"):
        def _summary():
            return {"env_name": "test_env"}
        setattr(world, "to_summary_dict", _summary)

    best_plan, scores = run_phase1_planning_episode(
        world=world,
        goal="sanity check planning goal",
        virtue_context_id="lv_resources",
    )

    assert isinstance(best_plan, dict)
    assert isinstance(scores, dict)

```

Minimal test, big payoff: if you refactor either P0 or P1 and break the bridge, CI screams immediately.

---

Right now you want:

- P0 = “how we run this thing”
    
- P1 = “how the thinking loop works”
    
- A very thin wire between them, not a merged hairball.
    

So: add adapter + CLI + one test, and call P0↔P1 “integrated enough for this tech tier.”