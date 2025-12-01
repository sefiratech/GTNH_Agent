## 0. Mental model: 3 surfaces, not 100 connections

Think in terms of **interfaces**, not “all modules talking to each other.”

There are really just **three big integration surfaces** now:

1. **Upstream to M8 AgentLoop**
    
    - `curriculum.next_goal(...)`
        
    - `curriculum.get_skill_view_for_goal(goal)`
        
    - AgentLoop consumes both and just… obeys.
        
2. **Downstream from M8 into M10 (learning)**
    
    - Experiences → `ReplayBuffer.append_experience(...)`
        
    - M10 reads those via `ExperienceBuffer` and computes stats.
        
3. **M10 → M11 feedback**
    
    - `LearningManager.compute_skill_stats()`
        
    - `LearningManager.build_skill_view(include_candidates=...)`
        
    - Curriculum uses that to decide which skills are “stable vs exploratory.”
        

Everything else is flavor.
## 1. Pass A: Wire “objects” without caring about behavior

Goal of this pass: get the construction graph right with **no clever logic**.

Think: “can I construct everything and run a fake episode without exploding?”

### A.1. Build stack in one place (AgentController)

In your `AgentController` (or whatever your entrypoint is), do the boring DI work:

- Build **ReplayBuffer** (M10)
    
- Build **LearningManager** with that buffer + SkillRegistry
    
- Load **CurriculumConfig** + **CurriculumEngine**
    
- Build **CurriculumManager** with:
    
    - `learning_manager`
        
    - `curriculum_engine`
        
    - `SkillPolicy` (from config)
        
- Build **AgentLoop** with:
    
    - `runtime`
        
    - `planner`
        
    - `curriculum=CurriculumManager`
        
    - `replay_buffer=ReplayBuffer`
        
    - `skills=SkillRegistry`
        
    - `virtue_engine`, `world_model`, etc if you have them
        

At this stage, you **don’t care** if:

- LearningManager returns garbage stats
    
- CurriculumEngine picks a dumb goal  
    You just want object graphs and imports to be stable.
    

Run a tiny smoke script:
```bash
python tools/agent_demo.py --episodes 1 --fake-world

```
If it runs one episode without throwing, Pass A is done.

## Complete



## 2. Pass B: One-way data flow, no feedback loops yet

Now you turn on **data flow**, but not “learning drives behavior” yet.

### B.1. AgentLoop → ReplayBuffer → ExperienceBuffer

You already have:

- `build_experience_from_episode(...)`
    
- `replay_buffer.append_experience(experience)`
    

Make sure, in `AgentLoop.run_episode`:

- At the end of the episode, it always:
    
    - builds an `ExperienceEpisode` (or whatever you called it)
        
    - calls `ReplayBuffer.append_experience(...)`
        

Once that’s in:

- Write a tool: `tools/inspect_experiences.py`
    
    - load the JSONL
        
    - print count + last experience summary
        

Sanity checks:

- After 3 fake episodes, buffer has 3 lines.
    
- No JSON parse errors.
    
- The goal IDs / texts match what curriculum picked.
    

This locks in **M8 → M10 data path**.

### B.2. Curriculum.next_goal as sole authority

In `AgentLoop.run_episode`:

- Replace any remaining fallback “invent a goal string” logic with:
```python
tech_state = semantics.infer_tech_state_from_world(world_state, graph)
experience_summary = replay_buffer.summarize_recent(...)  # can be dumb v1
goal = curriculum.next_goal(tech_state, experience_summary)

```
For v1, `experience_summary` can literally be:
```python
{"episode_count": replay_buffer.count()}

```

Curriculum can ignore it for now. The important part:  
**AgentLoop no longer creates goals on its own.**

That’s Q1.5 satisfied at a functional level.

src/learning/buffer.py  
src/curriculum/engine.py  
src/curriculum/manager.py  
src/agent/loop.py  
tools/inspect_experiences.py

pytest tests/test_llm_role_boundaries.py -q
python3 tools/agent_demo.py --episodes 3
python3 tools/inspect_experiences.py --path path/to/replay.jsonl

observation/trace_schema.py
integration/episode_logging.py

## Complete

---

## 3. Pass C: “Safe vs exploratory” actually influencing behavior

Now you let M10/M11 _matter_.

You’ve already got the plumbing:

- `LearningManager.build_skill_view(include_candidates: bool) -> SkillView`
    
- `CurriculumManager.get_skill_view_for_goal(goal)`
    
- `AgentLoop._build_world_summary(skill_view=...)`
    
- `Dispatcher.filter_skills_for_planning(skill_view, registry)` behind the scenes
    

So Pass C is:

### C.1. Make LearningManager return something non-idiotic

You don’t need genius. You need **structure**.

For v1:

- `compute_skill_stats()`:
    
    - iterate over all experiences
        
    - count per-skill:
        
        - uses
            
        - successes
            
    - success rate = successes / uses
        
- `build_skill_view(include_candidates)`:
    
    - Ask `SkillRegistry.describe_all()` for all skill specs
        
    - `active_skills = [s for s in skills if spec.metadata.status == "stable" or "accepted"]`
        
    - `candidate_skills = [s for s in skills if spec.metadata.status == "candidate"]` if `include_candidates` else `[]`
        

That’s it. No thresholding, no retry budgets. Just map status → set.

### C.2. Curriculum chooses stable vs exploratory

In `CurriculumManager.get_skill_view_for_goal(goal)`:

- Decide a **usage_mode**:
```python
if goal.source == "curriculum" and goal.phase.startswith("P1"):
    usage_mode = SkillUsageMode.ALLOW_CANDIDATES
else:
    usage_mode = SkillUsageMode.STABLE_ONLY

```

Then:
```python
return learning_manager.build_skill_view(
    include_candidates=(usage_mode == SkillUsageMode.ALLOW_CANDIDATES)
)

```

Now “is this curriculum slice exploratory?” actually changes what the **planner sees**.

This is the **first real behavioral difference** introduced by M10/M11.

src/learning/manager.py  
src/curriculum/manager.py


pytest tests/test_llm_role_boundaries.py -q
python3 tools/agent_demo.py --episodes 3
python3 tools/inspect_experiences.py --path data/experiences/replay.jsonl
pytest tests/test_skill_learning_view.py -q
pytest tests/test_curriculum_skill_view_policy.py -q




---

## 4. How to keep it psychologically manageable

You’re not doing “the full integration” in one nightmare session. You’re doing:

1. **Object graph pass**  
    “Can I construct everything and run one episode with fakes?”
    
2. **Data plumbing pass**  
    “Does an episode produce replay-able experiences & goals only come from curriculum?”
    
3. **Policy pass**  
    “Does changing SkillUsageMode actually change the visible skills?”
    

That’s it. Everything else is “later-You’s problem.”

---

## 5. After that: testing “lab correctness”

Once those passes are done, _then_ you run the big test battery:

- **Structure tests**
    
    - `pytest tests/test_llm_role_boundaries.py`
        
    - small tests for ExperienceBuffer, LearningManager, CurriculumEngine
        
- **Golden-path integration**
    
    - a single integration test that:
        
        - builds fake runtime
            
        - builds full stack (M0–M11)
            
        - runs 1–2 episodes
            
        - asserts:
            
            - a goal came from curriculum
                
            - a plan got produced
                
            - at least one Experience hit the replay buffer
                

If that passes, you’ve basically got a **working lab agent**.


---

So yes, your roadmap concept is sane:

- **Now:** wire M10/M11 into the existing spine with 3 narrow integration passes.
    
- **Then:** trust-but-verify with small tests.
    
- **After:** Create an LLM pipeline using real LLMs, not fakes.

## See Next Section

---

## Golden Path Integration


## 3. Order of operations (so pytest doesn’t turn into a war crime)

To avoid the “pytest clusterfuck”:

1. **Keep existing unit tests green**
    
    - `test_llm_role_boundaries.py`
        
    - your SkillLearningManager / CurriculumManager tests
        
2. **Add small structure tests**
    
    - ExperienceBuffer basic I/O
        
    - CurriculumEngine simple next_goal test
        
3. **Only then** add the integration test:
    
    - Put it in `tests/test_lab_integration.py`
        
    - Use _fakes_ and _minimal real objects_
        
    - Don’t pull in real env.yaml / models.yaml / Minecraft configs
        

If that integration test passes, congrats: your stack, as wired, is coherent. Anything you break after that, you’ll know _where_ you broke it: at the surfaces, not in some mysterious edge of the codebase.

In human terms: you’ve got a lab rat that can walk on the treadmill and poop into the telemetry bucket. Intelligence can come later.



## 0. What “lab correctness” actually means

For this phase, “correct” does **not** mean:

- The bot plays GTNH well
    
- The curriculum is smart
    
- The skills are optimal
    

It means:

1. The “three surfaces” are wired the way you intended:
    
    - **Surface 1: M8 ← M11**
        
        - AgentLoop calls `curriculum.next_goal(...)`
            
        - AgentLoop calls `curriculum.get_skill_view_for_goal(...)`
            
    - **Surface 2: M8 → M10**
        
        - AgentLoop ends an episode and pushes an `Experience` into `ReplayBuffer`
            
    - **Surface 3: M10 → M11**
        
        - LearningManager reads from `ExperienceBuffer`
            
        - LearningManager builds a `SkillView`
            
        - Curriculum uses `SkillView` to shape what the planner sees
            
2. You can run a tiny fake-world episode and see those flows actually happen.
    

That’s it. No brain required. Just blood circulation.

---

## 1. Structure tests: hit the three surfaces separately

You already started some of this, but here’s how to think about it:

### 1.1 ExperienceBuffer: “can it store and read?”

Goal: prove we can write a couple of `Experience` objects and read them back without exploding.

Core checks:

- `append_experience(...)` writes JSONL lines.
    
- `load_all()` yields `Experience` objects.
    
- `count()` matches number of non-empty lines.
    

This confirms your **M8 → M10** file boundary is sane.

pytest tests/test_experience_buffer.py -q

## Complete


---

### 1.2 SkillLearningManager: “can it produce a reasonable SkillView?”

You already gave it smarter behavior: `compute_skill_stats`, `build_skill_view`.

For Pass C lab correctness, the important bits:

- With a fake registry that says  
    `status = "active"` / `"accepted"` / `"candidate"` / `"retired"`
    
- `build_skill_view(include_candidates=False)`  
    → active has stable skills, candidates list empty
    
- `build_skill_view(include_candidates=True)`  
    → same active, but now candidate skills appear
    

This validates the **M10 → M11 “what skills are safe / exploratory?”** surface.

You already have a test in that direction; that’s basically this.

test_skill_learning_view.py

## Complete

---

### 1.3 CurriculumEngine: “does tech_state → goal work?”

You just rewired it to your actual `PhaseConfig` / `PhaseGoal` schema. You want one small unit test that says:

- Given:
    
    - a `CurriculumConfig` with:
        
        - 2 phases
            
            - phase 1: `required_active="stone_age"`
                
            - phase 2: `required_active="steam_age"`
                
- And `TechState(active="stone_age", unlocked=[...])`
    
- And a dumb `WorldState` with empty `context`
    
- When you call `next_goal(tech_state=..., world=..., experience_summary=None)`
    
- Then:
    
    - you get a non-None `AgentGoal`
        
    - `.phase == "phase_1_id"` (whatever you called it)
        
    - `.source == "curriculum"`
        

That proves the **M3 → M11** semantic handoff is usable.


test_curriculum_engine_basic.py

## Complete

---

## 2. Golden-path integration test: fake world, real stack

This is the fun part. You tie things together just enough to see the full loop run.

Conceptually, you want a test named something like:
test_lab_integration_happy_path_creates_experience()

and it should do this:

### 2.1. Build a **fake runtime** (M6/M7 substitute)

You DO NOT want real Minecraft, real LLMs, real anything. Just enough to satisfy AgentLoop.

Make a `FakeRuntime` that:

- Has a `.config` with:
    
    - `default_goal = "[LAB] dummy goal"`
        
    - `phase = "demo_p0"`
        
- Has a `.get_latest_planner_observation()` method that returns some fake dict
    
- Has a `.get_world_summary()` method that returns a small dict like `{ "inventory": {}, "machines": [] }`
    
- Has no real `planner` → AgentLoop will use fallback plan
    

Something like:
```python
class FakeRuntime:
    def __init__(self):
        self.config = type("Cfg", (), {
            "default_goal": "[LAB] dummy goal",
            "phase": "demo_p0",
        })()
        self.latest_observation = {"placeholder": True}

    def get_latest_planner_observation(self):
        return self.latest_observation

    def get_world_summary(self):
        return {
            "inventory": {},
            "machines": [],
        }

```
This pretends to be the M6/M7 world.


pytest tests/test_lab_integration_happy_path.py -q

## Complete


---

### 2.2. Build a **tiny curriculum stack** (M11 + M10 wiring)

You don’t want to load actual YAML for the integration test; just build a small `CurriculumConfig` in code:

- 1 phase:
    
    - `id="P1_demo"`
        
    - `tech_targets.required_active="stone_age"`
        
    - `goals=[PhaseGoal(id="demo_core", description="Stabilize base")]`
        
    - `completion_conditions` empty → so it never completes
        
- Fake `TechState`:
    
    - `active="stone_age"`
        
    - `unlocked=["stone_age"]`
        

Then:

- Create a **real** `ExperienceBuffer` using `tmp_path / "replay.jsonl"`.
    
- Create a **real** `SkillLearningManager` but give it:
    
    - that `ExperienceBuffer`
        
    - a _dummy_ `SkillRegistry` that reports one active skill and one candidate
        
- Create a **real** `CurriculumEngine(config)` with that `CurriculumConfig`.
    
- Create a `SkillPolicy(usage_mode=STABLE_ONLY)`.
    

Finally:

- `CurriculumManager(learning_manager, engine, policy, strategy=None)`
    

So AgentLoop has a “real enough” curriculum stack.

test_lab_integration_skill_view.py

## Complete
---

### 2.3. Build AgentLoop with replay buffer

Now wire **real AgentLoop**:

- `runtime = FakeRuntime()`
    
- `loop = AgentLoop( runtime=runtime, planner=None, # fallback path curriculum=curriculum_manager, skills=None, monitor=None, replay_buffer=replay_buffer, )`
    

Use an `AgentLoopConfig` that:

- keeps `store_experiences=True`
    
- critic on/off doesn’t really matter, but you can leave it on; if critic is missing it just no-ops.







---

### 2.4. Run 1–2 episodes and assert the important bits

In the test:
result = loop.run_episode(episode_id=0)

Then assert:

1. **Goal came from curriculum:**
```python
assert result.plan["goal_text"] == "Stabilize base"
# or:
assert "demo" in result.plan["goal_id"]
# and
# you can inspect the AgentGoal via replay_experience.problem_signature if needed

```

- Or more directly: in `build_experience_from_episode`, the `goal.source` is set to `"curriculum"`. You can read back the last experience and assert that.
    
- **Planner produced _a plan_ shape:**
    
    With fallback planner and no skills, it’s fine if `steps == []`. “Plan got produced” at this stage just means:
```python
assert "goal_id" in result.plan
assert "tasks" in result.plan
assert "steps" in result.plan

```

At least one Experience hit the replay buffer:

After run_episode returns, check:
assert replay_buffer.count() >= 1

Optionally load the last experience and assert:
```python
last = list(replay_buffer.load_all())[-1]
assert last.goal.text == "Stabilize base"
assert last.final_outcome["success"] in (True, False)

```

If _those_ three things are true, then for this fake world:

- AgentLoop → CurriculumManager → CurriculumEngine produced a goal.
    
- AgentLoop ran its phases without exploding and logged a plan.
    
- AgentLoop → Experience builder → ExperienceBuffer → Replay JSONL worked.
    

That’s your **lab agent**: dumb as rocks but structurally alive.

---

## Final Integration Before Installing Live Features


Just run loop running the full pytest suite and fixing the errors until it's all green.


ERROR tests/test_q1_control_and_experience.py::test_experience_episode_roundtrip_has_pre_and_post_eval - TypeError: ExperienceBuffer.__init__() got an unexpected keyword argument '...

FAILED tests/test_agent_loop_stub.py::test_agent_loop_run_episode_minimal - RuntimeError: AgentLoop.run_episode: curriculum.next_goal is required in Pa...

FAILED tests/test_agent_loop_v1.py::test_agent_loop_episode_runs_end_to_end - RuntimeError: AgentLoop.run_episode: curriculum.next_goal is required in Pa...

FAILED tests/test_curriculum_engine_basic.py::test_curriculum_engine_selects_goal_by_tech_state - assert None is not None

FAILED tests/test_curriculum_engine_phase.py::test_phase_selection_basic_progression - TypeError: CurriculumEngine.view() takes 1 positional argument but 3 were g...

FAILED tests/test_curriculum_engine_phase.py::test_phase_completion_with_machines - TypeError: CurriculumEngine.view() takes 1 positional argument but 3 were g...

FAILED tests/test_curriculum_learning_integration.py::test_learning_triggers_only_after_enough_episodes - TypeError: CurriculumEngine.view() takes 1 positional argument but 3 were g...

FAILED tests/test_curriculum_projects.py::test_project_unlocks_after_required_phases - TypeError: CurriculumEngine.view() takes 1 positional argument but 3 were g...

FAILED tests/test_error_model_with_fake_backend.py::test_error_model_parses_fixed_json - TypeError: ErrorContext.__init__() got an unexpected keyword argument 'role'

FAILED tests/test_lab_integration_happy_path.py::test_lab_integration_happy_path_creates_experience - RuntimeError: AgentLoop.run_episode: curriculum returned no goal (None).

FAILED tests/test_lab_integration_skill_view.py::test_lab_integration_real_skill_view - RuntimeError: AgentLoop.run_episode: curriculum returned no goal (None).

FAILED tests/test_q1_control_and_experience.py::test_critic_and_error_model_responses_share_failure_shape - TypeError: CriticResponse.__init__() got an unexpected keyword argument 'cr...

FAILED tests/test_scribe_model_with_fake_backend.py::test_scribe_model_parses_fixed_json - TypeError: TraceSummaryRequest.__init__() got an unexpected keyword argumen...

FAILED tests/test_skill_learning_manager.py::test_skill_learning_manager_end_to_end - TypeError: ExperienceBuffer.__init__() got an unexpected keyword argument '...

FAILED tests/test_skill_registry.py::test_skill_registry_rejects_deprecated - KeyError: 'No active SkillSpec found for skill: old_skill'

FAILED tests/test_synthesizer.py::test_synthesizer_produces_candidate - AssertionError: assert 'candidate' == 'proposed'


---
## Meta-strategy

Short version so your brain doesn’t melt:

1. **A: Experience & Learning**
    
    - Fix `ExperienceBuffer` & `SkillLearningManager` expectations.
        
2. **B: Curriculum Engine & Phases**
    
    - Fix `CurriculumEngine.next_goal` & `view` & phase / project behavior.
        
3. **C: AgentLoop & Lab Integration**
    
    - Align `AgentLoop` + CurriculumManager + replay behavior with those lab tests.
        
4. **D: LLM / Critic / Error / Scribe Protocols**
    
    - Fix dataclass signatures & shape compat for those models.
        
5. **E: Skills & Synthesizer**
    
    - Clean up registry & candidate status semantics.
        

Use `pytest <file>::<test> -q` on each step, and only occasionally do a full `pytest -q` sweep to see what your last three “fixes” secretly broke.

Whenever you’re ready, pick the first test from Group A and dump the full traceback; we’ll start gutting things from there.

---

## Group A – Experience & Learning (shared plumbing)

These two are almost certainly the same underlying issue (`ExperienceBuffer` ctor & friends):

1. `tests/test_q1_control_and_experience.py::test_experience_episode_roundtrip_has_pre_and_post_eval`

`tests/test_skill_learning_manager.py::test_skill_learning_manager_end_to_end`


**Why first?**

- They exercise the **canonical `ExperienceBuffer` API** and how it’s used by higher layers.
    
- If the constructor signature or behavior is off, you’ll keep tripping over it in later curriculum / lab tests.
    
- Fixing these stabilizes the M8 → M10 “shared file boundary.”
    

**Move:**  
Run them one at a time:
pytest tests/test_q1_control_and_experience.py::test_experience_episode_roundtrip_has_pre_and_post_eval -q
# Pass

pytest tests/test_skill_learning_manager.py::test_skill_learning_manager_end_to_end -q

# Pass


Patch `ExperienceBuffer.__init__` / helpers until both are green.

## Group B – Curriculum engine & phase semantics

All the “CurriculumEngine.view / next_goal / phase logic” complaints live here:

3. `tests/test_curriculum_engine_basic.py::test_curriculum_engine_selects_goal_by_tech_state`
    
4. `tests/test_curriculum_engine_phase.py::test_phase_selection_basic_progression`
    
5. `tests/test_curriculum_engine_phase.py::test_phase_completion_with_machines`
    
6. `tests/test_curriculum_learning_integration.py::test_learning_triggers_only_after_enough_episodes`
    
7. `tests/test_curriculum_projects.py::test_project_unlocks_after_required_phases`
    

**Why second?**

- These lock in the **M3 → M11 semantic handoff** and the shape of `CurriculumEngine.view` & `next_goal`.
    
- A ton of later stuff (CurriculumManager, lab integration, even AgentLoop behavior) implicitly assumes these APIs.
    
- Right now you’ve got `view()` signature issues and `next_goal` returning `None` where tests expect a real goal.
    

**Move:**

In order:

pytest tests/test_curriculum_engine_basic.py::test_curriculum_engine_selects_goal_by_tech_state -q

# Pass

pytest tests/test_curriculum_engine_phase.py::test_phase_selection_basic_progression -q

# Pass

pytest tests/test_curriculum_engine_phase.py::test_phase_completion_with_machines -q

# Pass

pytest tests/test_curriculum_projects.py::test_project_unlocks_after_required_phases -q

# Pass

pytest tests/test_curriculum_learning_integration.py::test_learning_triggers_only_after_enough_episodes -q

# Pass

Get:

- `CurriculumEngine.next_goal(tech_state, world)` returning non-None when it should.
    
- `CurriculumEngine.view(tech_state, world)` matching what those tests expect.
    

Once these pass, the curriculum layer should be “structurally correct.”

---

## Group C – AgentLoop & lab integration (curriculum + replay + loop)

# Going forward I am pasting all tests in each group and correcting them at once.

These are all screaming about `curriculum.next_goal` and “no goal” cases:

8. `tests/test_agent_loop_stub.py::test_agent_loop_run_episode_minimal`
    
9. `tests/test_agent_loop_v1.py::test_agent_loop_episode_runs_end_to_end`
    
10. `tests/test_lab_integration_happy_path.py::test_lab_integration_happy_path_creates_experience`
    
11. `tests/test_lab_integration_skill_view.py::test_lab_integration_real_skill_view`
    

**Why third?**

- They sit on top of **Groups A + B**:
    
    - They assume `ExperienceBuffer` works.
        
    - They assume curriculum returns sensible goals.
        
- Fixing them earlier is pointless while curriculum still returns `None` or explodes.
    

**What’s likely wrong conceptually:**

- Your updated `AgentLoop.run_episode` logic about curriculum is slightly at odds with what the tests expect:
    
    - Some tests expect a **fallback goal** when curriculum is missing / not configured.
        
    - Some expect the **goal to come from curriculum** in lab setups (where you wired a tiny CurriculumManager).
        
- The RuntimeError `"curriculum.next_goal is required in Pass B."` should not exist anymore for the test stubs.
    

**Move:**

Run them after Groups A & B:
pytest tests/test_agent_loop_stub.py::test_agent_loop_run_episode_minimal -q
pytest tests/test_agent_loop_v1.py::test_agent_loop_episode_runs_end_to_end -q
pytest tests/test_lab_integration_happy_path.py::test_lab_integration_happy_path_creates_experience -q
pytest tests/test_lab_integration_skill_view.py::test_lab_integration_real_skill_view -q

#  Complete

Use these to:

- Align `AgentLoop` fallback behavior with tests (no RuntimeError for “no curriculum” cases).
    
- Confirm lab integration still produces experiences with goal.source == `"curriculum"` where appropriate.
    

---

## Group D – LLM / critic / error / scribe protocol shapes

These are pure schema / dataclass signature mismatches:

12. `tests/test_error_model_with_fake_backend.py::test_error_model_parses_fixed_json`
    
13. `tests/test_q1_control_and_experience.py::test_critic_and_error_model_responses_share_failure_shape`
    
14. `tests/test_scribe_model_with_fake_backend.py::test_scribe_model_parses_fixed_json`
    

**Why fourth?**

- These mostly hit `spec.llm` & whatever wrappers live around critic/error/scribe.
    
- They shouldn’t affect curriculum or AgentLoop logic, but they block full-suite green.
    
- Fixing them early doesn’t help the more central mechanics; now that the core is stable, you fix the façades.
    

**Move:**

Run them individually:
pytest tests/test_error_model_with_fake_backend.py::test_error_model_parses_fixed_json -q
pytest tests/test_q1_control_and_experience.py::test_critic_and_error_model_responses_share_failure_shape -q
pytest tests/test_scribe_model_with_fake_backend.py::test_scribe_model_parses_fixed_json -q

#  Complete

Then:

- Adjust dataclasses like `ErrorModelResponse`, `CriticResponse`, whatever `TraceSummaryRequest` equivalent is, to accept the fields tests are feeding (`role`, `critical`, etc.).
    
- Keep them backward compatible where possible (default fields, optional kwargs).
    

---

## Group E – Skills & synthesizer policy (local behavior)

These are leaf-node weirdos:

15. `tests/test_skill_registry.py::test_skill_registry_rejects_deprecated`
    
16. `tests/test_synthesizer.py::test_synthesizer_produces_candidate`
    

**Why last?**

- They’re localized:
    
    - `SkillRegistry` should throw on `"deprecated"` or similar.
        
    - `SkillSynthesizer` / `SkillCandidate` status expectations (`'candidate'` vs `'proposed'`).
        
- They don’t cascade to the loop, curriculum, or replay logic.
    
- Tweaking them earlier might mask higher-level issues by making tests pass for the wrong reason.
    

**Move:**
pytest tests/test_skill_registry.py::test_skill_registry_rejects_deprecated -q
pytest tests/test_synthesizer.py::test_synthesizer_produces_candidate -q

# 

Adjust:

- Registry behavior on deprecated skills to match test assumptions.
    
- Default `SkillCandidate.status` / YAML parsing so the test sees `'proposed'` (or adjust test if the new spec explicitly changed; but assume test is the contract).



