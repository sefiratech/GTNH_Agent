Short answer: yes, Phase 4 brain-fusion time. You’re wiring the “what should we learn next?” (M10) to the “where are we in the tech journey and what matters now?” (M11).

I’ll give you:

1. **Architecture outline & data flow**
    
2. **Event / trigger model**
    
3. **A cohesive Python orchestrator module** with comments on every line
    
4. **Notes on testing, logging, failure modes**
    

---

## 1. Architecture Outline: M10 ↔ M11

**Roles**

- **M11 (Curriculum)**
    
    - Owns _where we are_: current phase, phase goals, virtue multipliers
        
    - Provides:
        
        - `ActiveCurriculumView`
            
        - `phase_view.skill_focus` (`must_have`, `preferred`)
            
        - `unlocked_projects` (for Stargate & other long-horizon madness)
            
- **M10 (Skill Learning)**
    
    - Owns _how we improve skills over time_:
        
        - Experience buffer / replay store
            
        - Skill-learning cycles (`run_learning_cycle_for_goal` or similar)
            
        - Synthesizing / refining skill definitions
            
- **Glue Logic (Phase 4 integration)**
    
    - Reads curriculum → decides which skills to prioritize
        
    - Checks experience buffer → sees if we have enough data to train
        
    - Calls M10 when curriculum & data say “go”
        

---

## 2. Data Flow & Function Interactions

End-to-end loop (conceptual):

1. **M8 finishes an episode:**
    
    - Has `tech_state`, `world_state`, `episode_trace`
        
    - Stores `episode_trace` into M10’s replay buffer
        
2. **Coordinator asks M11:**
    
    - `curr_view = curriculum_engine.view(tech_state, world_state)`
        
    - Extract:
        
        - `phase_id = curr_view.phase_view.phase.id`
            
        - `skill_focus = curr_view.phase_view.skill_focus`
            
3. **Coordinator derives learning targets:**
    
    - From `skill_focus.must_have` and `skill_focus.preferred`
        
    - Possibly merged with defaults / global skill list
        
4. **Coordinator checks M10 buffer:**
    
    - For each target skill, query something like  
        `experience_count = replay_store.count_episodes(skill_name, context_id)`
        
    - If `experience_count >= min_episodes`, schedule a learning cycle
        
5. **M10 runs learning cycle per chosen skill:**
    
    - `learning_manager.run_learning_cycle_for_goal(...)`
        
    - Returns result: success / failure, updated skill metadata, etc.
        
6. **Coordinator logs results:**
    
    - For monitoring: which skills were trained, how often, with what outcome
        
    - Potentially surfaces this for dashboards / dev tools


```python
# path: src/runtime/curriculum_learning_coordinator.py

from __future__ import annotations  # allow forward-annotated types

from dataclasses import dataclass  # lightweight config holder
from typing import Any, Dict, List, Optional  # basic typing primitives

from semantics.schema import TechState  # M3: structured tech progression
from spec.types import WorldState       # M7: normalized world snapshot

from curriculum.engine import CurriculumEngine, ActiveCurriculumView  # M11: curriculum logic
from learning.manager import SkillLearningManager                     # M10: learning core
from learning.replay import EpisodeReplayStore                        # M10: replay buffer API


@dataclass
class CoordinatorConfig:
    """
    Configuration for curriculum-driven skill learning.
    """
    min_episodes_per_skill: int = 5          # threshold for "enough experience"
    max_skills_per_tick: int = 3             # cap scheduled skills per call
    context_prefix: str = "curriculum"       # prefix for context_id used by learning
    include_preferred_skills: bool = True    # whether to include preferred skills after must_have


class CurriculumLearningCoordinator:
    """
    Coordinator that implements the M10 ↔ M11 data flow:

      1. (External) M8 stores episode into replay buffer (not handled here).
      2. Ask M11 for current curriculum view.
      3. Derive learning targets from skill_focus.
      4. Check M10 replay buffer for experience counts.
      5. Run M10 learning cycles for eligible skills.
      6. Return a structured summary for logging / monitoring.
    """

    def __init__(
        self,
        *,
        curriculum_engine: CurriculumEngine,
        learning_manager: SkillLearningManager,
        replay_store: EpisodeReplayStore,
        config: Optional[CoordinatorConfig] = None,
    ) -> None:
        self._curriculum_engine = curriculum_engine             # M11 engine instance
        self._learning_manager = learning_manager               # M10 manager instance
        self._replay_store = replay_store                       # M10 replay buffer instance
        self._config = config or CoordinatorConfig()            # use provided config or defaults

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_episode(
        self,
        *,
        tech_state: TechState,
        world_state: WorldState,
        episode_trace: Dict[str, Any],
        episode_meta: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Main entrypoint called after an episode is finished and stored.

        Implements steps 2–6 of the data flow description.
        """
        # --------------------------------------------------------------
        # 2. Coordinator asks M11:
        #    - curr_view = curriculum_engine.view(tech_state, world_state)
        #    - phase_id, skill_focus, etc.
        # --------------------------------------------------------------
        curr_view: ActiveCurriculumView = self._curriculum_engine.view(
            tech_state,
            world_state,
        )
        phase_view = curr_view.phase_view                        # convenience alias for phase-level info

        phase_id = phase_view.phase.id                           # current phase id
        skill_focus = phase_view.skill_focus                     # dict: { "must_have": [...], "preferred": [...] }

        # --------------------------------------------------------------
        # 3. Coordinator derives learning targets:
        #    - from skill_focus.must_have / skill_focus.preferred
        # --------------------------------------------------------------
        targets = self._derive_learning_targets(skill_focus)     # ordered list of candidate skill names

        # apply per-tick cap to avoid scheduling too many skills at once
        targets = targets[: self._config.max_skills_per_tick]

        # build curriculum-aware context identifier used by M10
        context_id = self._build_context_id(
            curriculum_id=curr_view.curriculum_id,
            phase_id=phase_id,
        )

        # --------------------------------------------------------------
        # 4. Coordinator checks M10 buffer:
        #    - experience_count = replay_store.count_episodes(skill_name, context_id)
        #    - if >= min_episodes, schedule a learning cycle
        # --------------------------------------------------------------
        ready_skills: List[str] = []
        for skill_name in targets:
            count = self._replay_store.count_episodes(
                skill_name=skill_name,
                context_id=context_id,
            )
            if count >= self._config.min_episodes_per_skill:
                ready_skills.append(skill_name)

        # --------------------------------------------------------------
        # 5. M10 runs learning cycle per chosen skill:
        #    - learning_manager.run_learning_cycle_for_goal(...)
        # --------------------------------------------------------------
        learning_results: List[Dict[str, Any]] = []
        for skill_name in ready_skills:
            # derive a loose goal substring from the skill name
            goal_substring = skill_name.replace("_", " ")

            # call into M10’s learning manager
            result = self._learning_manager.run_learning_cycle_for_goal(
                goal_substring=goal_substring,
                target_skill_name=skill_name,
                context_id=context_id,
                tech_tier=tech_state.active,
                success_only=True,
                min_episodes=self._config.min_episodes_per_skill,
            )

            # normalize to a logging-friendly record
            learning_results.append(
                {
                    "skill_name": skill_name,
                    "context_id": context_id,
                    "tech_tier": tech_state.active,
                    "result": result,
                }
            )

        # --------------------------------------------------------------
        # 6. Coordinator logs results:
        #    - caller can log / store this summary for dashboards / dev tools
        # --------------------------------------------------------------
        summary: Dict[str, Any] = {
            "curriculum_id": curr_view.curriculum_id,
            "phase_id": phase_id,
            "phase_name": phase_view.phase.name,
            "episode_meta": episode_meta,
            "episode_trace_summary": episode_trace.get("summary"),  # optional short episode summary
            "targets_considered": targets,
            "skills_ready": ready_skills,
            "learning_results": learning_results,
            "unlocked_projects": [p.id for p in curr_view.unlocked_projects],
        }

        return summary

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _derive_learning_targets(
        self,
        skill_focus: Dict[str, List[str]],
    ) -> List[str]:
        """
        Derive an ordered list of skills to *try* to train based on skill_focus.

        Priority:
          1. must_have (in listed order)
          2. preferred (if config.include_preferred_skills is True)
        """
        must_have = list(skill_focus.get("must_have", []))
        preferred = list(skill_focus.get("preferred", []))

        if not self._config.include_preferred_skills:
            return must_have

        seen = set(must_have)
        ordered: List[str] = list(must_have)

        for name in preferred:
            if name not in seen:
                ordered.append(name)
                seen.add(name)

        return ordered

    def _build_context_id(self, *, curriculum_id: str, phase_id: str) -> str:
        """
        Build a context_id of the form:

          "{prefix}:{curriculum_id}:{phase_id}"
        """
        return f"{self._config.context_prefix}:{curriculum_id}:{phase_id}"

```




---

## 3. Event / Trigger Model

Typical triggers for M10 ← M11 integration:

- **Per N episodes in the current phase**  
    e.g., every 5 episodes in `steam_early`, attempt to refine `maintain_boilers`.
    
- **On phase completion**  
    When M11 says `phase_view.is_complete`, do a burst of learning on the phase’s core skills.
    
- **On project unlock**  
    When a long-horizon project unlocks (e.g. Stargate), prioritize the skills that will matter for that project.
    

In practice: the coordinator gets called from the main loop with `tech_state`, `world_state`, and “episode just ended”, then decides whether to trigger learning.


```python
# path: src/runtime/curriculum_learning_triggers.py

from __future__ import annotations  # allow forward references in type hints

from dataclasses import dataclass, field  # for simple config / state holders
from typing import Any, Dict, List, Set  # basic typing primitives

from semantics.schema import TechState  # M3: structured tech progression state
from spec.types import WorldState       # M7: normalized world snapshot

from curriculum.engine import CurriculumEngine  # M11: curriculum engine
from runtime.curriculum_learning_coordinator import (  # coordinator glues M10 ↔ M11
    CurriculumLearningCoordinator,
)
from curriculum.engine import ActiveCurriculumView  # for type hints only


@dataclass
class TriggerConfig:
    """
    Configuration for when to trigger curriculum-driven learning.

    This directly encodes the "Event / Trigger Model" for M10 ← M11.
    """
    episodes_per_phase: int = 5     # periodic trigger: every N episodes in the same phase
    enable_phase_completion: bool = True   # trigger burst on phase completion
    enable_project_unlock: bool = True     # trigger burst when a long-horizon project unlocks


@dataclass
class TriggerState:
    """
    Internal state for trigger decisions across episodes.
    """
    episodes_in_phase: Dict[str, int] = field(default_factory=dict)   # phase_id -> episode count
    phase_complete: Dict[str, bool] = field(default_factory=dict)     # phase_id -> last known is_complete flag
    seen_projects: Set[str] = field(default_factory=set)              # set of project ids already seen as unlocked


class CurriculumLearningTriggerManager:
    """
    Wraps CurriculumLearningCoordinator with an event / trigger model.

    Responsibilities:
      - Track how many episodes have run in each phase.
      - Detect phase completion transitions.
      - Detect new long-horizon project unlocks.
      - Decide when to call the underlying coordinator to actually run M10 learning.
    """

    def __init__(
        self,
        *,
        curriculum_engine: CurriculumEngine,
        coordinator: CurriculumLearningCoordinator,
        config: TriggerConfig | None = None,
    ) -> None:
        self._curriculum_engine = curriculum_engine               # M11 engine for phase / project view
        self._coordinator = coordinator                           # orchestration for M10 learning cycles
        self._config = config or TriggerConfig()                  # trigger configuration
        self._state = TriggerState()                              # mutable trigger state

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def handle_episode(
        self,
        *,
        tech_state: TechState,
        world_state: WorldState,
        episode_trace: Dict[str, Any],
        episode_meta: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Main entrypoint called after an episode ends.

        The AgentLoop (M8) should:
          1. Store the episode into the replay buffer (M10).
          2. Call this method with tech_state, world_state, and episode metadata.

        This method:
          - Queries M11 for current phase / projects.
          - Updates trigger state (episodes_in_phase, completion flags, project set).
          - Decides whether learning should be triggered.
          - If yes, calls the coordinator to perform learning and returns a rich summary.
          - If no, returns a "no-op" summary for logging.
        """
        # Get the current curriculum view from M11
        curr_view: ActiveCurriculumView = self._curriculum_engine.view(
            tech_state,
            world_state,
        )
        phase_view = curr_view.phase_view                         # convenience alias
        phase_id = phase_view.phase.id                            # current phase identifier
        is_complete = phase_view.is_complete                      # boolean completion flag
        unlocked_projects = [p.id for p in curr_view.unlocked_projects]  # list of unlocked project ids

        # Update counters for episodes-per-phase trigger
        episode_count = self._state.episodes_in_phase.get(phase_id, 0) + 1
        self._state.episodes_in_phase[phase_id] = episode_count

        # Determine whether each trigger type should fire
        triggers: List[str] = []                                  # collect which triggers fired this tick

        # 1) Per N episodes in the current phase
        if episode_count % self._config.episodes_per_phase == 0:
            triggers.append("periodic_phase_episode")

        # 2) On phase completion (rising edge only)
        if self._config.enable_phase_completion:
            was_complete = self._state.phase_complete.get(phase_id, False)
            if is_complete and not was_complete:
                triggers.append("phase_completion")
            self._state.phase_complete[phase_id] = is_complete

        # 3) On project unlock (newly unlocked projects)
        new_projects = set(unlocked_projects) - self._state.seen_projects
        if self._config.enable_project_unlock and new_projects:
            triggers.append("project_unlock")
            self._state.seen_projects.update(new_projects)

        # If nothing fired, return a no-op summary for monitoring / logs
        if not triggers:
            return {
                "triggered": False,
                "triggers": [],
                "curriculum_id": curr_view.curriculum_id,
                "phase_id": phase_id,
                "phase_name": phase_view.phase.name,
                "phase_is_complete": is_complete,
                "episode_meta": episode_meta,
                "unlocked_projects": unlocked_projects,
                "episodes_in_phase": episode_count,
                "coordinator_summary": None,
            }

        # At least one trigger fired: delegate to the coordinator to actually run learning
        coordinator_summary = self._coordinator.process_episode(
            tech_state=tech_state,
            world_state=world_state,
            episode_trace=episode_trace,
            episode_meta=episode_meta,
        )

        # Wrap everything into a unified summary
        return {
            "triggered": True,
            "triggers": triggers,
            "curriculum_id": curr_view.curriculum_id,
            "phase_id": phase_id,
            "phase_name": phase_view.phase.name,
            "phase_is_complete": is_complete,
            "episode_meta": episode_meta,
            "unlocked_projects": unlocked_projects,
            "episodes_in_phase": episode_count,
            "new_projects": list(new_projects),
            "coordinator_summary": coordinator_summary,
        }

```



---

## 4. Orchestrator Module (Phase 4 Integration)

New file, not overwriting anything:


```python

# path: src/runtime/phase4_curriculum_learning_orchestrator.py  # file location in repo

from __future__ import annotations  # enable future-style annotations

from dataclasses import dataclass  # for lightweight config containers
from typing import Dict, List, Any, Optional  # type hints for clarity

from semantics.schema import TechState  # M3: structured tech progression state
from spec.types import WorldState       # M7: normalized world snapshot type

from curriculum.engine import CurriculumEngine, ActiveCurriculumView  # M11 core
from learning.manager import SkillLearningManager                     # M10 core
from learning.replay import EpisodeReplayStore                        # M10 buffer


@dataclass  # simple config object instead of a mess of kwargs
class LearningScheduleConfig:
    """Configuration knobs for curriculum-driven learning scheduling."""  # docstring for humans
    min_episodes_per_skill: int = 5      # how many episodes before we try to train a skill
    max_skills_per_tick: int = 3         # prevent explosion: cap skills per integration step
    always_include_preferred: bool = True  # whether to include preferred skills after must_have
    context_prefix: str = "curriculum"   # prefix for learning context IDs


class CurriculumLearningOrchestrator:
    """
    Glue between M11 (curriculum) and M10 (skill learning).

    Responsibilities:
      - Query M11 for current phase + skill_focus.
      - Inspect M10 replay buffer for experience counts.
      - Decide which skills to train.
      - Trigger M10 learning cycles.
      - Return a structured summary for logging / monitoring.
    """  # long-form docstring so Future You remembers what this is for

    def __init__(
        self,
        *,
        curriculum_engine: CurriculumEngine,
        learning_manager: SkillLearningManager,
        replay_store: EpisodeReplayStore,
        config: Optional[LearningScheduleConfig] = None,
    ) -> None:
        """Initialize orchestrator with its dependencies."""  # docstring
        self._curriculum_engine = curriculum_engine          # hold M11 engine instance
        self._learning_manager = learning_manager            # hold M10 manager instance
        self._replay_store = replay_store                    # hold M10 replay buffer
        self._config = config or LearningScheduleConfig()    # use provided config or defaults

    # ------------------------------------------------------------------
    # Core entrypoint
    # ------------------------------------------------------------------

    def run_after_episode(
        self,
        *,
        tech_state: TechState,
        world: WorldState,
        episode_meta: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Main integration entrypoint called after an episode completes.

        Typical call site: M8's main loop, right after storing an episode.
        """  # docstring
        # Step 1: query curriculum for current phase + skill-focus
        curr_view = self._curriculum_engine.view(tech_state, world)  # get ActiveCurriculumView from M11
        phase_view = curr_view.phase_view                             # convenience alias for current phase view

        # Step 2: derive ordered skill targets for learning
        targets = self._select_learning_targets(phase_view.skill_focus)  # must_have then preferred

        # Enforce cap on skills per scheduling tick
        targets = targets[: self._config.max_skills_per_tick]         # slice to avoid over-scheduling

        # Step 3: compute context_id for this phase / curriculum
        context_id = self._build_context_id(curr_view, phase_view)    # context_id used by M10

        # Step 4: decide which skills actually have enough experience
        ready_skills = self._filter_skills_with_experience(           # choose only skills with enough episodes
            targets=targets,
            context_id=context_id,
        )

        # Step 5: trigger learning cycles for ready skills
        learning_results: List[Dict[str, Any]] = []                   # accumulate per-skill results
        for skill_name in ready_skills:                               # iterate through each chosen skill
            result = self._run_learning_cycle_for_skill(              # call into M10 manager
                skill_name=skill_name,
                tech_state=tech_state,
                context_id=context_id,
            )
            learning_results.append(result)                           # store result for logging / monitoring

        # Step 6: produce summary payload for logging / debug
        summary: Dict[str, Any] = {                                   # top-level summary dict
            "curriculum_id": curr_view.curriculum_id,                 # which curriculum we are using
            "phase_id": phase_view.phase.id,                          # current phase id
            "phase_name": phase_view.phase.name,                      # human-friendly phase name
            "episode_meta": episode_meta,                             # pass-through of episode metadata
            "targets_considered": targets,                            # which skills we considered for learning
            "skills_trained": [r["skill_name"] for r in learning_results],  # list of actually trained skills
            "learning_results": learning_results,                     # detailed per-skill results
            "unlocked_projects": [p.id for p in curr_view.unlocked_projects],  # long-horizon projects in scope
        }
        return summary                                                # caller can log or inspect this

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _select_learning_targets(
        self,
        skill_focus: Dict[str, List[str]],
    ) -> List[str]:
        """
        Turn skill_focus into an ordered list of skills to potentially train.

        Priority:
          1. must_have
          2. preferred (optional, toggled by config)
        """  # docstring
        must_have = list(skill_focus.get("must_have", []))            # copy must_have list to avoid mutating input
        preferred = list(skill_focus.get("preferred", []))            # copy preferred list

        if not self._config.always_include_preferred:                 # if config says "only must_have"
            return must_have                                          # return only must_have skills

        seen = set(must_have)                                         # track which skills we've already added
        ordered: List[str] = []                                       # final ordered list

        ordered.extend(must_have)                                     # ensure must_have come first
        for name in preferred:                                        # iterate through preferred list
            if name not in seen:                                      # avoid duplicates
                ordered.append(name)                                  # add new preferred skill
                seen.add(name)                                        # mark as seen

        return ordered                                                # return ordered skill names

    def _build_context_id(
        self,
        curr_view: ActiveCurriculumView,
        phase_view: Any,
    ) -> str:
        """
        Build a context_id used for learning, tying episodes to curriculum phase.

        Example: "curriculum:default_speedrun:steam_early"
        """  # docstring
        prefix = self._config.context_prefix                          # base prefix from config
        return f"{prefix}:{curr_view.curriculum_id}:{phase_view.phase.id}"  # formatted context_id string

    def _filter_skills_with_experience(
        self,
        *,
        targets: List[str],
        context_id: str,
    ) -> List[str]:
        """
        Filter skill targets down to those with enough experience in the replay store.
        """  # docstring
        ready: List[str] = []                                         # skills we will actually train
        for skill_name in targets:                                    # iterate through candidate skills
            count = self._replay_store.count_episodes(                # ask replay store how many episodes we have
                skill_name=skill_name,
                context_id=context_id,
            )
            if count >= self._config.min_episodes_per_skill:          # check against threshold
                ready.append(skill_name)                              # schedule for training
        return ready                                                  # return list of ready skills

    def _run_learning_cycle_for_skill(
        self,
        *,
        skill_name: str,
        tech_state: TechState,
        context_id: str,
    ) -> Dict[str, Any]:
        """
        Invoke a single learning cycle for a given skill via SkillLearningManager.

        Returns a dict that is safe to log / inspect.
        """  # docstring
        # Derive a loose natural-language goal from the skill name (simple heuristic)
        goal_substring = skill_name.replace("_", " ")                 # turn 'feed_coke_ovens' into 'feed coke ovens'

        # Call into M10 to perform the actual learning cycle
        result = self._learning_manager.run_learning_cycle_for_goal(  # delegate to learning manager
            goal_substring=goal_substring,                            # fuzzy goal description
            target_skill_name=skill_name,                             # canonical skill identifier
            context_id=context_id,                                    # curriculum-linked context
            tech_tier=tech_state.active,                              # current active tech tier
            success_only=True,                                        # use only successful episodes for learning
            min_episodes=self._config.min_episodes_per_skill,         # again enforce min episodes here
        )

        # Normalize into a logging-friendly dictionary
        return {
            "skill_name": skill_name,                                 # which skill was trained
            "context_id": context_id,                                 # where in the curriculum this occurred
            "tech_tier": tech_state.active,                           # tech tier during training
            "result": result,                                         # raw result from learning manager
        }


```

---

## 5. Testing, Logging & Failure Points

### Testing strategy

1. **Unit tests for the orchestrator**
    
    - Mock `CurriculumEngine`, `SkillLearningManager`, `EpisodeReplayStore`.
        
    - Verify:
        
        - `run_after_episode`:
            
            - Uses `skill_focus` order correctly.
                
            - Honors `min_episodes_per_skill` & `max_skills_per_tick`.
                
            - Passes correct `context_id` / `tech_tier` to learning manager.
                
        - No learning is triggered if replay counts are too low.
            
2. **Integration tests (lightweight)**
    
    - Use a small real curriculum YAML.
        
    - Use a fake replay store that you seed with episodes.
        
    - Assert learning is triggered only after enough episodes hit the buffer.
        
3. **Property-ish tests**
    
    - If you randomize `skill_focus` contents but keep counts, ensure:
        
        - No duplicate scheduled skills.
            
        - Changing `always_include_preferred` flips behavior predictably.


```python
# path: tests/test_curriculum_learning_orchestrator.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import pytest

from runtime.phase4_curriculum_learning_orchestrator import (
    CurriculumLearningOrchestrator,
    LearningScheduleConfig,
)


# -----------------------------
# Minimal fakes / stubs
# -----------------------------

@dataclass
class FakePhase:
    id: str
    name: str


@dataclass
class FakePhaseView:
    phase: FakePhase
    is_complete: bool
    active_goals: List[str]
    virtue_overrides: Dict[str, float]
    skill_focus: Dict[str, List[str]]


@dataclass
class FakeActiveCurriculumView:
    curriculum_id: str
    phase_view: FakePhaseView
    unlocked_projects: List[Any]


class FakeCurriculumEngine:
    def __init__(self, view: FakeActiveCurriculumView) -> None:
        self._view = view
        self.calls: List[Dict[str, Any]] = []

    def view(self, tech_state: Any, world: Any) -> FakeActiveCurriculumView:
        self.calls.append({"tech_state": tech_state, "world": world})
        return self._view


class FakeReplayStore:
    def __init__(self, counts: Dict[str, int]) -> None:
        self._counts = counts
        self.calls: List[Dict[str, Any]] = []

    def count_episodes(self, *, skill_name: str, context_id: str) -> int:
        self.calls.append({"skill_name": skill_name, "context_id": context_id})
        return self._counts.get(skill_name, 0)


class FakeLearningManager:
    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    def run_learning_cycle_for_goal(
        self,
        *,
        goal_substring: str,
        target_skill_name: str,
        context_id: str,
        tech_tier: str,
        success_only: bool,
        min_episodes: int,
    ) -> Dict[str, Any]:
        call = {
            "goal_substring": goal_substring,
            "target_skill_name": target_skill_name,
            "context_id": context_id,
            "tech_tier": tech_tier,
            "success_only": success_only,
            "min_episodes": min_episodes,
        }
        self.calls.append(call)
        return {"status": "ok", "echo": call}


# Simple tech/world stubs so we don't drag real schemas in if not needed
@dataclass
class FakeTechState:
    active: str
    unlocked: List[str]


@dataclass
class FakeWorldState:
    context: Dict[str, Any]


# -----------------------------
# Tests
# -----------------------------

def _make_orchestrator_for_order_test() -> tuple[
    CurriculumLearningOrchestrator,
    FakeLearningManager,
]:
    # Skill-focus with both must_have and preferred
    skill_focus = {
        "must_have": ["maintain_boilers", "feed_coke_ovens"],
        "preferred": ["chunk_mining", "optimize_routes"],
    }

    phase_view = FakePhaseView(
        phase=FakePhase(id="steam_early", name="Steam Early"),
        is_complete=False,
        active_goals=["Maintain stable steam power."],
        virtue_overrides={"Safety": 1.5},
        skill_focus=skill_focus,
    )
    curr_view = FakeActiveCurriculumView(
        curriculum_id="default_speedrun",
        phase_view=phase_view,
        unlocked_projects=[],
    )

    curriculum_engine = FakeCurriculumEngine(curr_view)
    replay_store = FakeReplayStore(
        counts={
            # All skills have plenty of experience
            "maintain_boilers": 10,
            "feed_coke_ovens": 10,
            "chunk_mining": 10,
            "optimize_routes": 10,
        }
    )
    learning_manager = FakeLearningManager()

    config = LearningScheduleConfig(
        min_episodes_per_skill=5,
        max_skills_per_tick=2,          # cap at 2 to test ordering
        always_include_preferred=True,
        context_prefix="curriculum",
    )

    orchestrator = CurriculumLearningOrchestrator(
        curriculum_engine=curriculum_engine,
        learning_manager=learning_manager,
        replay_store=replay_store,
        config=config,
    )
    return orchestrator, learning_manager


def test_run_after_episode_respects_order_and_max_skills() -> None:
    """
    Orchestrator should:
      - Respect skill_focus ordering (must_have first).
      - Enforce max_skills_per_tick.
      - Pass correct context_id and tech_tier to learning manager.
    """
    orchestrator, learning_manager = _make_orchestrator_for_order_test()

    tech_state = FakeTechState(active="steam_age", unlocked=[])
    world = FakeWorldState(context={})
    episode_meta = {"episode_id": "ep_001"}

    summary = orchestrator.run_after_episode(
        tech_state=tech_state,
        world=world,
        episode_meta=episode_meta,
    )

    # Only 2 skills trained due to max_skills_per_tick=2
    trained = summary["skills_trained"]
    assert trained == ["maintain_boilers", "feed_coke_ovens"]

    # Learning manager called twice, in the same order
    assert len(learning_manager.calls) == 2
    c1, c2 = learning_manager.calls

    # Check correct context_id and tech_tier passed
    assert c1["context_id"] == "curriculum:default_speedrun:steam_early"
    assert c2["context_id"] == "curriculum:default_speedrun:steam_early"
    assert c1["tech_tier"] == "steam_age"
    assert c2["tech_tier"] == "steam_age"

    # Check goal_substring heuristic
    assert c1["goal_substring"] == "maintain boilers"
    assert c2["goal_substring"] == "feed coke ovens"


def test_run_after_episode_no_learning_when_insufficient_experience() -> None:
    """
    If replay counts are below min_episodes_per_skill, no learning cycles should run.
    """
    skill_focus = {
        "must_have": ["maintain_boilers"],
        "preferred": ["feed_coke_ovens"],
    }
    phase_view = FakePhaseView(
        phase=FakePhase(id="steam_early", name="Steam Early"),
        is_complete=False,
        active_goals=["Maintain stable steam power."],
        virtue_overrides={},
        skill_focus=skill_focus,
    )
    curr_view = FakeActiveCurriculumView(
        curriculum_id="eco_factory",
        phase_view=phase_view,
        unlocked_projects=[],
    )

    curriculum_engine = FakeCurriculumEngine(curr_view)
    replay_store = FakeReplayStore(
        counts={
            # Not enough experience for either skill
            "maintain_boilers": 1,
            "feed_coke_ovens": 2,
        }
    )
    learning_manager = FakeLearningManager()

    config = LearningScheduleConfig(
        min_episodes_per_skill=5,   # require 5
        max_skills_per_tick=5,
        always_include_preferred=True,
        context_prefix="curriculum",
    )

    orchestrator = CurriculumLearningOrchestrator(
        curriculum_engine=curriculum_engine,
        learning_manager=learning_manager,
        replay_store=replay_store,
        config=config,
    )

    tech_state = FakeTechState(active="steam_age", unlocked=[])
    world = FakeWorldState(context={})
    episode_meta = {"episode_id": "ep_002"}

    summary = orchestrator.run_after_episode(
        tech_state=tech_state,
        world=world,
        episode_meta=episode_meta,
    )

    assert summary["skills_trained"] == []
    assert summary["learning_results"] == []
    assert learning_manager.calls == []


def test_select_learning_targets_respects_preferred_toggle() -> None:
    """
    Changing always_include_preferred should flip whether preferred skills are included.
    """
    skill_focus = {
        "must_have": ["core_skill"],
        "preferred": ["extra_skill_1", "extra_skill_2"],
    }
    phase_view = FakePhaseView(
        phase=FakePhase(id="phase_x", name="Phase X"),
        is_complete=False,
        active_goals=[],
        virtue_overrides={},
        skill_focus=skill_focus,
    )
    curr_view = FakeActiveCurriculumView(
        curriculum_id="test_curriculum",
        phase_view=phase_view,
        unlocked_projects=[],
    )

    curriculum_engine = FakeCurriculumEngine(curr_view)
    replay_store = FakeReplayStore(counts={})
    learning_manager = FakeLearningManager()

    # Config with preferred included
    config_yes = LearningScheduleConfig(
        min_episodes_per_skill=1,
        max_skills_per_tick=10,
        always_include_preferred=True,
    )
    orch_yes = CurriculumLearningOrchestrator(
        curriculum_engine=curriculum_engine,
        learning_manager=learning_manager,
        replay_store=replay_store,
        config=config_yes,
    )
    targets_yes = orch_yes._select_learning_targets(skill_focus)
    assert targets_yes == ["core_skill", "extra_skill_1", "extra_skill_2"]

    # Config with preferred excluded
    config_no = LearningScheduleConfig(
        min_episodes_per_skill=1,
        max_skills_per_tick=10,
        always_include_preferred=False,
    )
    orch_no = CurriculumLearningOrchestrator(
        curriculum_engine=curriculum_engine,
        learning_manager=learning_manager,
        replay_store=replay_store,
        config=config_no,
    )
    targets_no = orch_no._select_learning_targets(skill_focus)
    assert targets_no == ["core_skill"]

```


```python
# path: tests/test_curriculum_learning_integration.py

from __future__ import annotations

from pathlib import Path
from textwrap import dedent
from typing import Any, Dict, List

from curriculum.loader import load_curriculum
from curriculum.engine import CurriculumEngine
from runtime.phase4_curriculum_learning_orchestrator import (
    CurriculumLearningOrchestrator,
    LearningScheduleConfig,
)

from dataclasses import dataclass


# Minimal stubs for learning + replay

class FakeReplayStore:
    def __init__(self, counts: Dict[str, int]) -> None:
        self._counts = counts

    def count_episodes(self, *, skill_name: str, context_id: str) -> int:
        return self._counts.get((skill_name, context_id), 0)


class FakeLearningManager:
    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    def run_learning_cycle_for_goal(
        self,
        *,
        goal_substring: str,
        target_skill_name: str,
        context_id: str,
        tech_tier: str,
        success_only: bool,
        min_episodes: int,
    ) -> Dict[str, Any]:
        call = {
            "goal_substring": goal_substring,
            "target_skill_name": target_skill_name,
            "context_id": context_id,
            "tech_tier": tech_tier,
            "success_only": success_only,
            "min_episodes": min_episodes,
        }
        self.calls.append(call)
        return {"status": "ok"}


@dataclass
class FakeTechState:
    active: str
    unlocked: List[str]


@dataclass
class FakeWorldState:
    context: Dict[str, Any]


def _write_yaml(path: Path, text: str) -> None:
    normalized = dedent(text).lstrip("\n")
    path.write_text(normalized, encoding="utf-8")


def test_learning_triggers_only_after_enough_episodes(tmp_path: Path) -> None:
    """
    Integration-style test:

      - Uses a small real curriculum YAML.
      - Uses real CurriculumEngine.
      - Uses fake replay store seeded with varying episode counts.
      - Asserts learning only fires once episode count crosses threshold.
    """
    yaml_text = """
    id: "integration_curriculum"
    name: "Integration Curriculum"
    description: "Tiny curriculum for integration testing."
    phases:
      - id: "steam_early"
        name: "Steam Early"
        tech_targets:
          required_active: "steam_age"
          required_unlocked: []
        goals:
          - id: "g1"
            description: "Make steam."
        virtue_overrides: {}
        skill_focus:
          must_have:
            - "maintain_boilers"
          preferred:
            - "feed_coke_ovens"
        completion_conditions:
          tech_unlocked: []
          machines_present: []
    long_horizon_projects: []
    """
    cfg_path = tmp_path / "integration_curriculum.yaml"
    _write_yaml(cfg_path, yaml_text)

    curriculum_config = load_curriculum(cfg_path)
    engine = CurriculumEngine(curriculum_config)

    # context id the orchestrator will build:
    # "curriculum:integration_curriculum:steam_early"
    context_id = "curriculum:integration_curriculum:steam_early"

    # Replay store with counts below threshold at first
    replay_store = FakeReplayStore(
        counts={
            ("maintain_boilers", context_id): 3,  # below min=5
            ("feed_coke_ovens", context_id): 10,
        }
    )
    learning_manager = FakeLearningManager()

    config = LearningScheduleConfig(
        min_episodes_per_skill=5,
        max_skills_per_tick=5,
        always_include_preferred=True,
        context_prefix="curriculum",
    )

    orchestrator = CurriculumLearningOrchestrator(
        curriculum_engine=engine,
        learning_manager=learning_manager,
        replay_store=replay_store,
        config=config,
    )

    tech_state = FakeTechState(active="steam_age", unlocked=[])
    world_state = FakeWorldState(context={})
    episode_meta = {"episode_id": "ep_001"}

    # First run: maintain_boilers has only 3 episodes, so *no* skill should be ready
    summary1 = orchestrator.run_after_episode(
        tech_state=tech_state,
        world=world_state,
        episode_meta=episode_meta,
    )

    assert summary1["skills_trained"] == []
    assert learning_manager.calls == []

    # Now bump maintain_boilers above threshold
    replay_store._counts[("maintain_boilers", context_id)] = 6

    # Second run: both skills have enough episodes, so both should be trained
    summary2 = orchestrator.run_after_episode(
        tech_state=tech_state,
        world=world_state,
        episode_meta={"episode_id": "ep_002"},
    )

    assert summary2["skills_trained"] == ["maintain_boilers", "feed_coke_ovens"]
    assert len(learning_manager.calls) == 2
    assert learning_manager.calls[0]["target_skill_name"] == "maintain_boilers"
    assert learning_manager.calls[1]["target_skill_name"] == "feed_coke_ovens"

```


```python
# path: tests/test_curriculum_learning_properties.py

from __future__ import annotations

import random
from typing import Dict, List

from runtime.phase4_curriculum_learning_orchestrator import (
    CurriculumLearningOrchestrator,
    LearningScheduleConfig,
)

from dataclasses import dataclass


# Tiny fake engine/view just to reach _select_learning_targets nicely if needed

@dataclass
class FakePhase:
    id: str
    name: str


@dataclass
class FakePhaseView:
    phase: FakePhase
    is_complete: bool
    active_goals: List[str]
    virtue_overrides: Dict[str, float]
    skill_focus: Dict[str, List[str]]


@dataclass
class FakeActiveCurriculumView:
    curriculum_id: str
    phase_view: FakePhaseView
    unlocked_projects: List[object]


class FakeCurriculumEngine:
    def __init__(self, view: FakeActiveCurriculumView) -> None:
        self._view = view

    def view(self, *args, **kwargs) -> FakeActiveCurriculumView:
        return self._view


class FakeReplayStore:
    def count_episodes(self, *, skill_name: str, context_id: str) -> int:
        return 999  # always enough episodes for property tests


class FakeLearningManager:
    def __init__(self) -> None:
        self.calls: List[Dict] = []

    def run_learning_cycle_for_goal(self, **kwargs):
        self.calls.append(kwargs)
        return {"status": "ok"}


def test_select_learning_targets_no_duplicates_and_stable_behavior() -> None:
    """
    Property-ish test:

      - Randomize order of must_have / preferred.
      - Ensure no duplicates in targets.
      - Ensure must_have always come first.
      - Ensure toggling always_include_preferred removes preferred but keeps must_have.
    """
    must_have = ["skill_a", "skill_b", "skill_c"]
    preferred = ["skill_c", "skill_d", "skill_e"]  # deliberate overlap

    # Randomize internal order; logic should still behave
    random.shuffle(must_have)
    random.shuffle(preferred)

    skill_focus = {
        "must_have": must_have,
        "preferred": preferred,
    }

    phase_view = FakePhaseView(
        phase=FakePhase(id="p", name="Phase P"),
        is_complete=False,
        active_goals=[],
        virtue_overrides={},
        skill_focus=skill_focus,
    )
    curr_view = FakeActiveCurriculumView(
        curriculum_id="c",
        phase_view=phase_view,
        unlocked_projects=[],
    )

    engine = FakeCurriculumEngine(curr_view)
    replay_store = FakeReplayStore()
    learning_manager = FakeLearningManager()

    # Preferred included
    config_yes = LearningScheduleConfig(
        min_episodes_per_skill=1,
        max_skills_per_tick=10,
        always_include_preferred=True,
    )
    orch_yes = CurriculumLearningOrchestrator(
        curriculum_engine=engine,
        learning_manager=learning_manager,
        replay_store=replay_store,
        config=config_yes,
    )

    targets_yes = orch_yes._select_learning_targets(skill_focus)

    # Must_have must be prefix of targets_yes in the same relative order
    assert targets_yes[: len(must_have)] == must_have

    # No duplicates in the full list
    assert len(set(targets_yes)) == len(targets_yes)

    # Preferred-only set is (preferred - must_have)
    expected_tail = [s for s in preferred if s not in must_have]
    assert targets_yes[len(must_have) :] == expected_tail

    # Preferred excluded
    config_no = LearningScheduleConfig(
        min_episodes_per_skill=1,
        max_skills_per_tick=10,
        always_include_preferred=False,
    )
    orch_no = CurriculumLearningOrchestrator(
        curriculum_engine=engine,
        learning_manager=learning_manager,
        replay_store=replay_store,
        config=config_no,
    )

    targets_no = orch_no._select_learning_targets(skill_focus)

    # Only must_have should remain
    assert targets_no == must_have

```


---

### Logging

At minimum, log something like this per call to `run_after_episode`:

- Current `phase_id` / `phase_name`
    
- `targets_considered`
    
- `skills_trained`
    
- `unlocked_projects`
    
- Any error states from M10 (e.g. convergence issues, missing skill definitions)
    

Use structured logging (dicts / JSON) so you can filter:

- “Show me all times we tried to train `maintain_boilers` in `steam_early`.”
    
- “Show me learning attempts related to `stargate_project`.”
    

### Failure points & mitigation

- **Curriculum missing / malformed**
    
    - Loader will raise `FileNotFoundError` / `ValueError`.
        
    - Orchestrator can be constructed only after curriculum loads; fail fast on startup instead of mid-loop.
        
- **Replay store misconfigured**
    
    - `count_episodes` could raise if called with wrong context key.
        
    - Unit tests around `_filter_skills_with_experience` help catch API drift.
        
- **Learning manager blowups**
    
    - `run_learning_cycle_for_goal` could throw:
        
        - Skill not found
            
        - No episodes match filter
            
        - Internal model error
            
    - Wrap that call in `try/except` if you want the orchestrator to be resilient:
        
        - Log failure
            
        - Skip skill this tick, don’t crash whole loop
            
- **TechState / WorldState mismatch**
    
    - If you ever change `TechState` / `WorldState` schemas without updating curriculum logic, your tests will scream. Good.
        

---

So yes: this gives you a **coherent multi-module architecture** where:

- M11 decides _what matters now_,
    
- M10 decides _how to improve it_,
    
- And this orchestrator keeps them from stepping on each other like two drunk villagers in a one-block corridor.