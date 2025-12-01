# Phase Integration 1 · M2 + M3 + M4 + M5

**Phase:** P1 – Offline Core Pillars  
**Scope:** Integrate the LLM stack (M2), GTNH semantics (M3), virtue lattice (M4), and skill registry (M5) into a coherent architecture and data flow.

This document defines how these modules talk to each other, how data moves through a single “planning episode,” and where testing & logging should hook in. It is the glue between the module docs and the eventual agent loop (M8). fileciteturn5file0

---

## 1. High-Level Flow

One planning cycle in Phase 1 (no actual Minecraft IPC yet) looks like this:

1. **Inputs**
   - `WorldState` snapshot (fake or serialized)
   - `TechState` (from M3)
   - Curriculum / context ID (Phase / scenario label)
   - User or system goal (text)

2. **Metadata & Context Fetch**
   - **M3**: load semantics DB (items, blocks, tech graph)
   - **M5**: load skill specs & packs → available skills for this tech state
   - **M4**: load virtue lattice config
   - **M2**: select backend + prompt preset for planner / critic

3. **Planning**
   - M2 PLANNER LLM receives:
     - goal, world summary, tech state
     - skill metadata (from M5)
     - virtue-aware hints (from M4 config)
   - It returns one or more candidate **plans**:
     - A plan is a list/graph of steps, each step referencing **skill names + parameters**.

4. **Evaluation**
   - **M3**: annotate plan with world/tech implications (using semantics + skill preconditions/effects).
   - **M4**: compute plan features, then virtue scores per plan.
   - **M4**: select best plan (`compare_plans`) for this context.

5. **Output**
   - Best plan + scores + explanation
   - Plan still in **abstract skill form**, ready for M8 to later expand into `Action`s.

This integration is *offline-only*: no packets to Minecraft, no real-time loop, just clean data flow between M2–M5.

---

## 2. Module Roles in Integration

### 2.1 M2 – LLM Stack

**Responsibilities in Phase 1 integration:**

- Provide **planner** and **critic** call surfaces:
  - `planner.generate_plan(...)`
  - `critic.explain_scores(...)` (optional)
- Hide backend specifics (llama.cpp vs others).
- Apply prompt presets that inject:
  - skill metadata
  - virtue hints
  - world/tech summaries

**Inputs:**
- Text goal, world/tech summary, skill metadata JSON, virtue context ID.

**Outputs:**
- JSON-like plan(s):
  - `{"id": str, "steps": [{"skill": str, "params": {...}}, ...]}`

---

### 2.2 M3 – Semantics / Tech State

**Responsibilities in integration:**

- Provide `TechState` for the current scenario.
- Provide semantic DBs:
  - block/item groups
  - tech graph edges
  - constraints (e.g. “requires steam_age to use LV machines”).
- Provide helper functions used by M4 feature extraction and by sanity checks around skills.

**Key functions (conceptual):**

- `get_current_tech_state(world) -> TechState`
- `lookup_item_semantics(item_id) -> ItemSemantics`
- `estimate_tech_progress(plan, tech_state) -> TechDeltaEstimate`

M3 doesn’t pick plans, it just helps describe what a plan *does* to the world/tech layer.

---

### 2.3 M4 – Virtue Lattice

**Responsibilities in integration:**

- Provide **plan feature extraction**:
  - `summarize_plan(plan, world, tech_state, semantics_db, skill_metadata)`
- Provide **scoring & comparison**:
  - `score_plan(summary, config)`
  - `compare_plans(plans, context_id, config)`

**Inputs:**
- Plan objects from M2.
- Semantics from M3.
- Skill metadata from M5.
- Virtue lattice config.

**Outputs:**
- Scores per virtue.
- Pareto-ish decision of “best plan” for a given context. fileciteturn5file0

---

### 2.4 M5 – Skill Registry

**Responsibilities in integration:**

- Provide **metadata** about available skills:
  - `SkillSpec` objects (preconditions, effects, tags, params).
- Provide skill **visibility** filtered by tech state & curriculum:
  - `registry.describe_for_tech_state(tech_state, enabled_packs)`
- Future: used by M8 to expand skills into `Action`s.

**Inputs:**
- TechState (from M3).
- Enabled Skill Packs (from M11 / config).

**Outputs:**
- JSON-like skill descriptors for the planner and the virtue feature extractor.

---

## 3. Data Flow Diagram (Text)

One planning episode:

```text
Goal + World Snapshot
       │
       ▼
   [M3] Semantics / TechState
       │
       ├──► TechState
       └──► Semantics DB
       │
       ▼
   [M5] Skill Registry
       │
       └──► Skill Metadata (filtered by TechState + packs)
       │
       ▼
   [M4] Virtue Config + Feature Helpers
       │
       ▼
   [M2] Planner LLM
       │   (goal + world summary + skills + virtue hints)
       ▼
   Candidate Plans (skill-level)
       │
       ▼
   [M4] summarize_plan + compare_plans
       │
       └──► Best plan + scores
```

---

## 4. Integration Orchestrator (Python)

Below is a **single-file integration scaffold** showing how modules could be wired together for Phase 1. Every line is commented for clarity.
python:


```

# phase1_integration.py

from typing import Dict, Any, List, Tuple

# --- M1 / shared types -------------------------------------------------------

from spec.types import WorldState, TechState, Action   # core shared data models


# --- M2: LLM Stack interfaces ------------------------------------------------

from llm_stack.planner import PlannerBackend           # abstract planner interface
from llm_stack.backends import get_planner_backend     # backend selector (e.g. llama.cpp)


# --- M3: Semantics / Tech State ---------------------------------------------

from semantics.tech_state import get_current_tech_state     # derive TechState from WorldState
from semantics.db import load_semantics_db                  # items/blocks/tech graph


# --- M4: Virtue Lattice ------------------------------------------------------

from virtues.loader import load_virtue_config               # load lattice config
from virtues.features import summarize_plan                 # feature extraction from plan
from virtues.lattice import compare_plans                   # pick best plan based on virtues


# --- M5: Skill Registry ------------------------------------------------------

from skills.registry import get_global_skill_registry       # entrypoint to SkillRegistry
from skills.packs import load_all_skill_packs               # to know which packs exist


# --- Phase 1 Integration Context --------------------------------------------

def build_phase1_context(world: WorldState) -> Dict[str, Any]:
    '''
    Build all shared context objects needed for a single planning episode.

    This includes:
    - tech_state (M3)
    - semantics_db (M3)
    - virtue_config (M4)
    - skill_registry (M5)
    - skill_packs (M5 / config)
    - planner_backend (M2)
    '''
    tech_state: TechState = get_current_tech_state(world)          # derive TechState from WorldState
    semantics_db = load_semantics_db()                             # load GTNH semantics database
    virtue_config = load_virtue_config()                           # load virtue lattice config
    skill_registry = get_global_skill_registry()                   # global SkillRegistry instance
    skill_packs = load_all_skill_packs()                           # all defined Skill Packs
    planner: PlannerBackend = get_planner_backend("planner_default")  # select planner backend

    return {
        "tech_state": tech_state,                                  # current tech tier
        "semantics_db": semantics_db,                              # GTNH semantics snapshot
        "virtue_config": virtue_config,                            # lattice + weights
        "skill_registry": skill_registry,                          # registered skills
        "skill_packs": skill_packs,                                # available Skill Packs
        "planner": planner,                                        # planner LLM interface
    }


def compute_available_skills(context: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    '''
    Use M3 + M5 to derive the set of skills visible to the planner.

    - Reads tech_state from context
    - Enables Skill Packs whose requires_tech_state matches
      the current tech_state (for Phase 1)
    - Asks the SkillRegistry for a filtered view of skill metadata
    '''
    tech_state: TechState = context["tech_state"]                    # current TechState
    skill_registry = context["skill_registry"]                       # SkillRegistry instance
    skill_packs = context["skill_packs"]                             # mapping pack_name -> SkillPack

    # Determine which Skill Packs should be enabled at this tech_state
    enabled_pack_names: List[str] = [
        name
        for name, pack in skill_packs.items()
        if pack.requires_tech_state == tech_state.name               # compare tech_state name
    ]

    # Ask registry for a filtered description of skills
    skill_metadata: Dict[str, Dict[str, Any]] = skill_registry.describe_for_tech_state(
        tech_state=tech_state.name,                                  # string tech ID for registry
        enabled_pack_names=enabled_pack_names,                       # only these packs are active
    )

    return skill_metadata                                            # mapping skill_name -> metadata dict


def call_planner_llm(
    planner: PlannerBackend,
    goal: str,
    world: WorldState,
    tech_state: TechState,
    skill_metadata: Dict[str, Dict[str, Any]],
    virtue_context_id: str,
) -> List[Dict[str, Any]]:
    '''
    Call the planner LLM (M2) to generate candidate plans.

    The prompt should include:
    - The goal
    - A summarized view of the world and tech_state
    - A machine-readable list of available skills
    - A virtue context identifier for hinting (e.g. "lv_bootstrap")
    '''
    # Build a payload for the planner backend
    planner_input: Dict[str, Any] = {
        "goal": goal,                                      # planning objective text
        "world_summary": world.to_summary_dict(),          # compressed WorldState
        "tech_state": tech_state.name,                     # string representation of TechState
        "skills": skill_metadata,                          # metadata for each available skill
        "virtue_context_id": virtue_context_id,            # context name for virtue config
    }

    # Delegate to the planner backend
    planner_output: Dict[str, Any] = planner.generate_plan(planner_input)

    # Expect a list of plan dicts under the 'plans' key
    plans: List[Dict[str, Any]] = planner_output.get("plans", [])
    return plans                                            # return raw planner plans


def select_best_plan(
    plans: List[Dict[str, Any]],
    world: WorldState,
    context: Dict[str, Any],
    virtue_context_id: str,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    '''
    Use M3 + M4 to evaluate and select the best plan.

    Steps:
    - Summarize each plan into a PlanSummary using semantics + skill metadata
    - Score each summary via virtue lattice
    - Use compare_plans to choose the best one
    '''
    tech_state: TechState = context["tech_state"]                     # current TechState
    semantics_db = context["semantics_db"]                            # GTNH semantics DB
    virtue_config = context["virtue_config"]                          # virtue lattice config
    skill_registry = context["skill_registry"]                        # SkillRegistry instance

    # Get full skill metadata (unfiltered) for feature extraction
    all_skill_metadata: Dict[str, Dict[str, Any]] = skill_registry.describe_all()

    plan_summaries = []                                               # will hold PlanSummary objects
    for plan in plans:
        # summarize_plan turns a raw plan dict into a feature-rich PlanSummary
        summary = summarize_plan(
            plan=plan,                                                # raw plan from planner
            world=world,                                              # WorldState
            tech_state=tech_state,                                    # TechState
            semantics_db=semantics_db,                                # semantics reference
            skill_metadata=all_skill_metadata,                        # all SkillSpecs
        )
        plan_summaries.append(summary)                                # accumulate summary

    # Compare all plan summaries with the virtue lattice
    best_summary, scores = compare_plans(
        plans=plan_summaries,                                         # summaries to rank
        context_id=virtue_context_id,                                 # virtue context ID
        config=virtue_config,                                         # lattice configuration
    )

    # Find the raw plan that corresponds to the best summary
    best_plan_id = best_summary.id                                    # ID of best plan
    best_plan = next(p for p in plans if p.get("id") == best_plan_id) # locate raw plan

    return best_plan, scores                                          # return chosen plan + scores


def run_phase1_planning_episode(world: WorldState, goal: str, virtue_context_id: str):
    '''
    Top-level driver for a Phase 1 planning episode.

    This is the main entrypoint for integration tests:
    - builds context (M2, M3, M4, M5)
    - derives available skills
    - calls planner LLM to get candidate plans
    - runs virtue lattice to choose the best one
    - returns best plan + scores
    '''
    # Build shared integration context
    context = build_phase1_context(world)

    # Compute which skills are visible at this tech_state
    skill_metadata = compute_available_skills(context)

    # Extract planner backend and tech_state from context
    planner: PlannerBackend = context["planner"]
    tech_state: TechState = context["tech_state"]

    # Call planner to generate candidate plans for the given goal
    plans = call_planner_llm(
        planner=planner,                                              # planner backend
        goal=goal,                                                    # planning goal
        world=world,                                                  # current WorldState
        tech_state=tech_state,                                        # current TechState
        skill_metadata=skill_metadata,                                # visible skills
        virtue_context_id=virtue_context_id,                          # virtue scenario ID
    )

    # If no plans returned, treat this as a failure for now
    if not plans:
        raise RuntimeError("Planner returned no plans for goal: %r" % (goal,))

    # Use virtue lattice to select the best plan among candidates
    best_plan, scores = select_best_plan(
        plans=plans,                                                  # candidate plans
        world=world,                                                  # WorldState
        context=context,                                              # integration context
        virtue_context_id=virtue_context_id,                          # virtue scenario ID
    )

    return best_plan, scores                                          # output for tests / caller
```

---

## 5. Logging & Observability

For Phase 1 integration tests and future debugging, log at these points:

1. **Context Build**
   - Log `tech_state`, number of skills, number of packs loaded.
2. **Planner Call (M2)**
   - Log goal, virtue_context_id, and a hash of the world summary (never full spam by default).
   - Log number of candidate plans returned.
3. **Feature Extraction (M4)**
   - Log which skills show up in each plan.
   - Log any missing or unknown skills/features.
4. **Virtue Scoring**
   - Log per-plan virtue scores (debug level).
   - Log which plan was chosen and why (the top few features).
5. **Failures**
   - Planner returned no plans.
   - summarize_plan raises due to bad semantics or unknown skill.
   - compare_plans cannot choose (ties, invalid config).

Recommend a per-episode correlation ID so you can trace logs across modules.

## 5.1 `src/integration/episode_logging.py`

This is a thin wrapper around `logging` that gives you:

- a per-episode correlation ID
    
- clear methods for each integration step
    
- no magic frameworks, just Python logging done like an adult
python:
```
# src/integration/episode_logging.py

import logging
import uuid
from typing import Dict, Any, Iterable, List, Optional


LOGGER_NAME = "gtnh_agent.phase1"


def generate_correlation_id() -> str:
    """
    Generate a short correlation ID for a planning episode.

    Using UUID4 hex shortened to 8 chars is enough for logs and debugging.
    """
    return uuid.uuid4().hex[:8]


def get_base_logger() -> logging.Logger:
    """
    Get the base logger for Phase 1 integration.

    If no handlers are configured yet, attach a simple STDERR handler.
    The application can override this later with a richer logging config.
    """
    logger = logging.getLogger(LOGGER_NAME)

    if not logger.handlers:
        # Default: INFO level with simple formatting
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


class EpisodeLogger:
    """
    Structured logger for a single planning episode.

    Every log line includes the episode correlation ID, so you can trace
    context build -> planner call -> feature extraction -> scoring -> failures.
    """

    def __init__(self, episode_id: str, logger: Optional[logging.Logger] = None) -> None:
        self.episode_id = episode_id
        self._logger = logger or get_base_logger()

    # --- Internal helpers ----------------------------------------------------

    def _prefix(self, msg: str) -> str:
        """Prefix messages with the correlation ID."""
        return f"[episode={self.episode_id}] {msg}"

    # --- Context build -------------------------------------------------------

    def log_context_built(
        self,
        tech_state_name: str,
        num_skills: int,
        num_packs: int,
    ) -> None:
        """
        Log summary of the integration context for this episode.
        """
        self._logger.info(
            self._prefix(
                "context built: tech_state=%s, skills=%d, packs=%d",
            ),
            tech_state_name,
            num_skills,
            num_packs,
        )

    # --- Planner call --------------------------------------------------------

    def log_planner_request(
        self,
        goal: str,
        virtue_context_id: str,
        world_summary_hash: str,
        num_visible_skills: int,
    ) -> None:
        """
        Log high-level input info for the planner call.
        """
        self._logger.info(
            self._prefix(
                "planner request: goal=%r, virtue_context_id=%s, "
                "world_hash=%s, visible_skills=%d",
            ),
            goal,
            virtue_context_id,
            world_summary_hash,
            num_visible_skills,
        )

    def log_planner_response(
        self,
        num_plans: int,
    ) -> None:
        """
        Log how many candidate plans the planner returned.
        """
        self._logger.info(
            self._prefix("planner response: plans=%d"),
            num_plans,
        )

    # --- Feature extraction (virtue feature layer) ---------------------------

    def log_plan_features(
        self,
        plan_id: str,
        skill_names: Iterable[str],
        missing_skills: Optional[Iterable[str]] = None,
    ) -> None:
        """
        Log which skills appear in a plan summary and any unknown/missing ones.
        """
        skills_list = list(skill_names)
        self._logger.debug(
            self._prefix("plan features: plan_id=%s, skills=%s"),
            plan_id,
            skills_list,
        )

        if missing_skills:
            missing_list = list(missing_skills)
            if missing_list:
                self._logger.warning(
                    self._prefix("plan features: plan_id=%s, missing_skills=%s"),
                    plan_id,
                    missing_list,
                )

    # --- Virtue scoring ------------------------------------------------------

    def log_virtue_scores(
        self,
        plan_id: str,
        scores: Dict[str, Any],
    ) -> None:
        """
        Log per-plan virtue scores at debug level.
        """
        self._logger.debug(
            self._prefix("virtue scores: plan_id=%s, scores=%s"),
            plan_id,
            scores,
        )

    def log_selected_plan(
        self,
        plan_id: str,
        scores: Dict[str, Any],
    ) -> None:
        """
        Log which plan was selected as best and summarize its scores.
        """
        self._logger.info(
            self._prefix("selected plan: plan_id=%s, scores=%s"),
            plan_id,
            scores,
        )

    # --- Failures / anomalies -----------------------------------------------

    def log_no_plans(self, goal: str) -> None:
        """
        Planner returned no candidate plans.
        """
        self._logger.error(
            self._prefix("failure: planner returned no plans for goal=%r"),
            goal,
        )

    def log_exception(
        self,
        phase: str,
        exc: BaseException,
    ) -> None:
        """
        Log an exception that occurred during a specific phase (e.g. 'summarize_plan').
        """
        self._logger.exception(
            self._prefix("exception in phase=%s: %s"),
            phase,
            exc,
        )

```

## 5.2 Updated `src/integration/phase1_integration.py` with logging

Now we wire `EpisodeLogger` into the orchestrator so logs actually fire at the right spots.

This is a **full updated file**, not a diff. You can overwrite your current version.
python:
```
# src/integration/phase1_integration.py

"""
Phase 1 integration orchestrator.

Wires M2 (LLM stack), M3 (semantics), M4 (virtue lattice), and M5 (skill registry)
into a single entrypoint: run_phase1_planning_episode().

This layer is OFFLINE ONLY in Phase 1:
- no Minecraft IPC
- no real-time loop
- used for planning simulations and integration tests
"""

from typing import Dict, Any, List, Tuple, Optional

# --- M1 / shared types -------------------------------------------------------

from spec.types import WorldState, TechState, Action   # core shared data models


# --- M2: LLM Stack interfaces ------------------------------------------------

from llm_stack.planner import PlannerBackend           # abstract planner interface
from llm_stack.backends import get_planner_backend     # backend selector (e.g. llama.cpp)


# --- M3: Semantics / Tech State ---------------------------------------------

from semantics.tech_state import get_current_tech_state     # derive TechState from WorldState
from semantics.db import load_semantics_db                  # items/blocks/tech graph


# --- M4: Virtue Lattice ------------------------------------------------------

from virtues.loader import load_virtue_config               # load lattice config
from virtues.features import summarize_plan                 # feature extraction from plan
from virtues.lattice import compare_plans                   # pick best plan based on virtues


# --- M5: Skill Registry ------------------------------------------------------

from skills.registry import get_global_skill_registry       # entrypoint to SkillRegistry
from skills.packs import load_all_skill_packs               # to know which packs exist


# --- Logging / observability -------------------------------------------------

from integration.episode_logging import (                   # per-episode logging helpers
    EpisodeLogger,
    generate_correlation_id,
)


# --- Phase 1 Integration Context --------------------------------------------

def build_phase1_context(
    world: WorldState,
    episode_logger: Optional[EpisodeLogger] = None,
) -> Dict[str, Any]:
    """
    Build all shared context objects needed for a single planning episode.

    This includes:
    - tech_state (M3)
    - semantics_db (M3)
    - virtue_config (M4)
    - skill_registry (M5)
    - skill_packs (M5 / config)
    - planner_backend (M2)
    """
    tech_state: TechState = get_current_tech_state(world)          # derive TechState from WorldState
    semantics_db = load_semantics_db()                             # load GTNH semantics database
    virtue_config = load_virtue_config()                           # load virtue lattice config
    skill_registry = get_global_skill_registry()                   # global SkillRegistry instance
    skill_packs = load_all_skill_packs()                           # all defined Skill Packs
    planner: PlannerBackend = get_planner_backend("planner_default")  # select planner backend

    if episode_logger is not None:
        # Use describe_all() here just to count; visibility is handled later
        num_specs = len(skill_registry.describe_all())
        episode_logger.log_context_built(
            tech_state_name=tech_state.name,
            num_skills=num_specs,
            num_packs=len(skill_packs),
        )

    return {
        "tech_state": tech_state,                                  # current tech tier
        "semantics_db": semantics_db,                              # GTNH semantics snapshot
        "virtue_config": virtue_config,                            # lattice + weights
        "skill_registry": skill_registry,                          # registered skills
        "skill_packs": skill_packs,                                # available Skill Packs
        "planner": planner,                                        # planner LLM interface
        "logger": episode_logger,                                  # episode-level logger
    }


def compute_available_skills(context: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Use M3 + M5 to derive the set of skills visible to the planner.

    - Reads tech_state from context
    - Enables Skill Packs whose requires_tech_state matches
      the current tech_state (for Phase 1)
    - Asks the SkillRegistry for a filtered view of skill metadata
    """
    tech_state: TechState = context["tech_state"]                    # current TechState
    skill_registry = context["skill_registry"]                       # SkillRegistry instance
    skill_packs = context["skill_packs"]                             # mapping pack_name -> SkillPack

    # Determine which Skill Packs should be enabled at this tech_state
    enabled_pack_names: List[str] = [
        name
        for name, pack in skill_packs.items()
        if pack.requires_tech_state == tech_state.name               # compare tech_state name
    ]

    # Ask registry for a filtered description of skills
    skill_metadata: Dict[str, Dict[str, Any]] = skill_registry.describe_for_tech_state(
        tech_state=tech_state.name,                                  # string tech ID for registry
        enabled_pack_names=enabled_pack_names,                       # only these packs are active
    )

    return skill_metadata                                            # mapping skill_name -> metadata dict


def _hash_world_summary(world: WorldState) -> str:
    """
    Compute a cheap hash of the world summary for logging.

    We don't want to dump the entire world state into logs,
    so we hash its string representation.
    """
    try:
        summary = world.to_summary_dict()
    except AttributeError:
        # Fallback: best-effort repr if to_summary_dict is not implemented yet
        summary = repr(world)

    data = str(summary).encode("utf-8")
    # Simple non-cryptographic hash is fine here
    return f"{abs(hash(data)) & 0xFFFFFFFF:08x}"


def call_planner_llm(
    planner: PlannerBackend,
    goal: str,
    world: WorldState,
    tech_state: TechState,
    skill_metadata: Dict[str, Dict[str, Any]],
    virtue_context_id: str,
    episode_logger: Optional[EpisodeLogger] = None,
) -> List[Dict[str, Any]]:
    """
    Call the planner LLM (M2) to generate candidate plans.

    The prompt should include:
    - The goal
    - A summarized view of the world and tech_state
    - A machine-readable list of available skills
    - A virtue context identifier for hinting (e.g. "lv_bootstrap")
    """
    world_hash = _hash_world_summary(world)                          # small, log-safe hash

    if episode_logger is not None:
        episode_logger.log_planner_request(
            goal=goal,
            virtue_context_id=virtue_context_id,
            world_summary_hash=world_hash,
            num_visible_skills=len(skill_metadata),
        )

    # Build a payload for the planner backend
    try:
        world_summary = world.to_summary_dict()
    except AttributeError:
        # In early phases, you might not have a proper summary method yet
        world_summary = {"raw": repr(world)}

    planner_input: Dict[str, Any] = {
        "goal": goal,                                      # planning objective text
        "world_summary": world_summary,                    # compressed WorldState
        "tech_state": tech_state.name,                     # string representation of TechState
        "skills": skill_metadata,                          # metadata for each available skill
        "virtue_context_id": virtue_context_id,            # context name for virtue config
    }

    # Delegate to the planner backend
    planner_output: Dict[str, Any] = planner.generate_plan(planner_input)

    # Expect a list of plan dicts under the 'plans' key
    plans: List[Dict[str, Any]] = planner_output.get("plans", [])

    if episode_logger is not None:
        episode_logger.log_planner_response(num_plans=len(plans))

    return plans                                            # return raw planner plans


def select_best_plan(
    plans: List[Dict[str, Any]],
    world: WorldState,
    context: Dict[str, Any],
    virtue_context_id: str,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Use M3 + M4 to evaluate and select the best plan.

    Steps:
    - Summarize each plan into a PlanSummary using semantics + skill metadata
    - Score each summary via virtue lattice
    - Use compare_plans to choose the best one
    """
    tech_state: TechState = context["tech_state"]                     # current TechState
    semantics_db = context["semantics_db"]                            # GTNH semantics DB
    virtue_config = context["virtue_config"]                          # virtue lattice config
    skill_registry = context["skill_registry"]                        # SkillRegistry instance
    episode_logger: Optional[EpisodeLogger] = context.get("logger")   # may be None

    # Get full skill metadata (unfiltered) for feature extraction
    all_skill_metadata: Dict[str, Dict[str, Any]] = skill_registry.describe_all()

    plan_summaries = []                                               # will hold PlanSummary objects

    for plan in plans:
        plan_id = plan.get("id", "<unknown>")

        try:
            # summarize_plan turns a raw plan dict into a feature-rich PlanSummary
            summary = summarize_plan(
                plan=plan,                                            # raw plan from planner
                world=world,                                          # WorldState
                tech_state=tech_state,                                # TechState
                semantics_db=semantics_db,                            # semantics reference
                skill_metadata=all_skill_metadata,                    # all SkillSpecs
            )
        except Exception as exc:
            # Log and re-raise: this is a serious consistency error
            if episode_logger is not None:
                episode_logger.log_exception(phase="summarize_plan", exc=exc)
            raise

        # Optionally log which skills appear in this plan
        if episode_logger is not None:
            # Assume summary exposes a 'skill_names' attribute or similar;
            # adjust when you finalize the PlanSummary API.
            skill_names = getattr(summary, "skill_names", [])
            missing_skills = getattr(summary, "missing_skills", [])
            episode_logger.log_plan_features(
                plan_id=plan_id,
                skill_names=skill_names,
                missing_skills=missing_skills,
            )

        plan_summaries.append(summary)                                # accumulate summary

    # Compare all plan summaries with the virtue lattice
    try:
        best_summary, scores = compare_plans(
            plans=plan_summaries,                                     # summaries to rank
            context_id=virtue_context_id,                             # virtue context ID
            config=virtue_config,                                     # lattice configuration
        )
    except Exception as exc:
        if episode_logger is not None:
            episode_logger.log_exception(phase="compare_plans", exc=exc)
        raise

    # Log per-plan scores if available
    if episode_logger is not None:
        # If compare_plans returns a dict keyed by plan id or summary, adapt accordingly.
        # For now we only log the chosen plan's scores.
        episode_logger.log_virtue_scores(
            plan_id=getattr(best_summary, "id", "<unknown>"),
            scores=scores,
        )
        episode_logger.log_selected_plan(
            plan_id=getattr(best_summary, "id", "<unknown>"),
            scores=scores,
        )

    # Find the raw plan that corresponds to the best summary
    best_plan_id = best_summary.id                                    # ID of best plan
    best_plan = next(p for p in plans if p.get("id") == best_plan_id) # locate raw plan

    return best_plan, scores                                          # return chosen plan + scores


def run_phase1_planning_episode(
    world: WorldState,
    goal: str,
    virtue_context_id: str,
    episode_id: Optional[str] = None,
):
    """
    Top-level driver for a Phase 1 planning episode.

    This is the main entrypoint for integration tests:
    - builds context (M2, M3, M4, M5)
    - derives available skills
    - calls planner LLM to get candidate plans
    - runs virtue lattice to choose the best one
    - returns best plan + scores

    Logging:
    - Each call gets an episode-level correlation ID (episode_id)
    - All logs from this episode are tagged with that ID
    """
    # Build an episode logger
    if episode_id is None:
        episode_id = generate_correlation_id()
    episode_logger = EpisodeLogger(episode_id)

    # Build shared integration context
    context = build_phase1_context(world, episode_logger=episode_logger)

    # Compute which skills are visible at this tech_state
    skill_metadata = compute_available_skills(context)

    # Extract planner backend and tech_state from context
    planner: PlannerBackend = context["planner"]
    tech_state: TechState = context["tech_state"]

    # Call planner to generate candidate plans for the given goal
    plans = call_planner_llm(
        planner=planner,                                              # planner backend
        goal=goal,                                                    # planning goal
        world=world,                                                  # current WorldState
        tech_state=tech_state,                                        # current TechState
        skill_metadata=skill_metadata,                                # visible skills
        virtue_context_id=virtue_context_id,                          # virtue scenario ID
        episode_logger=episode_logger,                                # log request/response
    )

    # If no plans returned, treat this as a failure for now
    if not plans:
        episode_logger.log_no_plans(goal)
        raise RuntimeError("Planner returned no plans for goal: %r" % (goal,))

    # Use virtue lattice to select the best plan among candidates
    best_plan, scores = select_best_plan(
        plans=plans,                                                  # candidate plans
        world=world,                                                  # WorldState
        context=context,                                              # integration context
        virtue_context_id=virtue_context_id,                          # virtue scenario ID
    )

    return best_plan, scores                                          # output for tests / caller

```

## 5.3 What this gives you, practically

You now have:

- A dedicated **EpisodeLogger** that:
    
    - logs context build (tech_state, skills, packs)
        
    - logs planner request/response
        
    - logs feature extraction (skills per plan + missing skills)
        
    - logs virtue scores + chosen plan
        
    - logs failure cases and exceptions
        
- A **per-episode correlation ID** tied to every log line
    

Usage from tests or experiments is trivial:
python:
```
from integration.phase1_integration import run_phase1_planning_episode

best_plan, scores = run_phase1_planning_episode(
    world=my_fake_world,
    goal="reach LV steam infrastructure",
    virtue_context_id="lv_bootstrap",
)

```

Logs will show one neat trace per episode like:

- `[episode=3fa2c9d1] context built: ...`
    
- `[episode=3fa2c9d1] planner request: ...`
    
- `[episode=3fa2c9d1] planner response: plans=3`
    
- `[episode=3fa2c9d1] virtue scores: ...`
    
- `[episode=3fa2c9d1] selected plan: ...`
    

You now have actual observability instead of blind faith in a stack of YAML and LLMs. Which, frankly, is progress.



---

## 6. Failure Points & Mitigations

### 6.1 Missing or Invalid Skill Specs (M5)

- **Symptom:** planner references a skill not present in the registry, or summarize_plan fails.
- **Mitigation:**
  - Registry validation on startup (assert all skills in packs have specs & implementations).
  - Test: “all skill_packs reference existing skills” in `tests/test_skill_packs_integrity.py`.

### 6.2 Semantics Drift (M3)

- **Symptom:** virtue selection changes in unexpected ways after editing tech graph / semantics.
- **Mitigation:**
  - Stable integration tests for core LV scenarios, similar to the virtue lattice LV check. fileciteturn5file0
  - Snapshot-driven tests: freeze a world/tech/semantics fixture and run regression.

### 6.3 LLM Planner Instability (M2)

- **Symptom:** same input → wildly different plan counts or structures.
- **Mitigation:**
  - Use deterministic temperature for offline tests.
  - Keep strict JSON schemas in the planner output parser.
  - Add retry + “parse-only” mode to detect when the LLM hallucinated broken structures.

### 6.4 Virtue Config Changes (M4)

- **Symptom:** “good” plans start scoring worse due to changed weights or edges.
- **Mitigation:**
  - Keep versioned virtue configs.
  - Run integration tests (e.g. LV coke oven layout scenario) on every change. fileciteturn5file0

## 6.5 New submodule: `src/integration/validators/__init__.py`

Just to make it a proper package.
python:
```
# src/integration/validators/__init__.py

"""
Validation and regression helpers for Phase 1 integration.

This package collects small utilities that:
- verify Skill Pack / Skill Registry integrity (M5)
- support snapshot-style regression tests for semantics (M3)
- add guardrails for planner stability (M2)
- help track virtue config regressions (M4)
"""

from .skill_integrity import (
    SkillPackIntegrityResult,
    validate_skill_packs_against_registry,
)

```

## 2. Skill spec / pack integrity (6.1)

This one _is_ fully actionable with your current code.

### `src/integration/validators/skill_integrity.py`
python:
```
# src/integration/validators/skill_integrity.py

from dataclasses import dataclass
from typing import Dict, List, Set

from skills.registry import get_global_skill_registry
from skills.packs import load_all_skill_packs, SkillPack


@dataclass
class SkillPackIntegrityResult:
    """
    Result of validating Skill Packs against the Skill Registry.

    - missing_specs:
        skills referenced in packs that do NOT have a SkillSpec
    - missing_impls:
        skills referenced in packs that do have a spec, but are NOT registered
        as implementations in the SkillRegistry (no Python class registered)
    - unused_specs:
        skills that have a SkillSpec but are NOT referenced in any pack
        (not necessarily an error, but useful to see)
    """
    missing_specs: Dict[str, List[str]]     # pack_name -> [skill_name, ...]
    missing_impls: Dict[str, List[str]]     # pack_name -> [skill_name, ...]
    unused_specs: List[str]                 # spec names not referenced by any pack


def validate_skill_packs_against_registry() -> SkillPackIntegrityResult:
    """
    Validate that all Skill Packs reference existing SkillSpecs and implementations.

    Checks:
    - Every skill listed in pack.skills has a SkillSpec in the registry
    - Every skill listed in pack.skills has a registered implementation
      (i.e. appears in registry.list_skills())
    - Compute which SkillSpecs are never referenced by any pack

    This function DOES NOT raise; it returns a result object so tests or
    CLI wrappers can decide how strict to be.
    """
    registry = get_global_skill_registry()
    packs: Dict[str, SkillPack] = load_all_skill_packs()

    # Specs known to the registry
    spec_metadata = registry.describe_all()  # only active + registered
    # But we also want specs that might be inactive / candidate
    # by peeking into the internal _specs map if available.
    all_spec_names: Set[str] = set(getattr(registry, "_specs", {}).keys())
    if not all_spec_names:
        # Fallback: just use describe_all keys if _specs isn't accessible
        all_spec_names = set(spec_metadata.keys())

    # Implementations known to the registry
    impl_names: Set[str] = set(registry.list_skills())

    missing_specs: Dict[str, List[str]] = {}
    missing_impls: Dict[str, List[str]] = {}
    referenced_specs: Set[str] = set()

    for pack_name, pack in packs.items():
        for skill_name in pack.skills:
            referenced_specs.add(skill_name)

            # Spec missing?
            if skill_name not in all_spec_names:
                missing_specs.setdefault(pack_name, []).append(skill_name)
                # If spec is missing, impl check is meaningless for this skill
                continue

            # Implementation missing?
            if skill_name not in impl_names:
                missing_impls.setdefault(pack_name, []).append(skill_name)

    # Specs that are never referenced by any pack
    unused_specs = sorted(all_spec_names - referenced_specs)

    return SkillPackIntegrityResult(
        missing_specs=missing_specs,
        missing_impls=missing_impls,
        unused_specs=unused_specs,
    )


def assert_skill_packs_integrity(strict_unused: bool = False) -> None:
    """
    Convenience function that RAISES if integrity checks fail.

    - Always raises if any missing_specs or missing_impls are found.
    - Optionally raises if there are unused_specs and strict_unused=True.

    Intended for use in:
    - a CLI validation tool
    - pytest integration tests
    """
    result = validate_skill_packs_against_registry()

    errors: List[str] = []

    if result.missing_specs:
        for pack_name, skills in result.missing_specs.items():
            errors.append(
                f"SkillPack '{pack_name}' references skills with no SkillSpec: {skills}"
            )

    if result.missing_impls:
        for pack_name, skills in result.missing_impls.items():
            errors.append(
                f"SkillPack '{pack_name}' references skills with no implementation: {skills}"
            )

    if strict_unused and result.unused_specs:
        errors.append(
            f"Unused SkillSpecs (not referenced by any pack): {result.unused_specs}"
        )

    if errors:
        joined = "\n".join(errors)
        raise AssertionError(f"SkillPack integrity validation failed:\n{joined}")


if __name__ == "__main__":
    # Simple CLI entrypoint:
    # python -m integration.validators.skill_integrity
    result = validate_skill_packs_against_registry()
    if not result.missing_specs and not result.missing_impls:
        print("SkillPack integrity OK.")
        if result.unused_specs:
            print(f"Note: unused SkillSpecs: {result.unused_specs}")
    else:
        print("SkillPack integrity problems detected:")
        if result.missing_specs:
            print("  Missing specs:")
            for pack_name, skills in result.missing_specs.items():
                print(f"    {pack_name}: {skills}")
        if result.missing_impls:
            print("  Missing implementations:")
            for pack_name, skills in result.missing_impls.items():
                print(f"    {pack_name}: {skills}")
        raise SystemExit(1)

```

### `tests/test_skill_packs_integrity.py`

This is the specific test the doc mentions.
python:
```
# tests/test_skill_packs_integrity.py

from integration.validators.skill_integrity import assert_skill_packs_integrity


def test_skill_packs_reference_existing_skills() -> None:
    """
    Integration check for M5:

    - Every skill referenced in any SkillPack must have:
      - a SkillSpec defined, and
      - a registered implementation in the SkillRegistry.

    This test will fail if:
    - a pack references a skill with no YAML spec
    - a pack references a skill that has a spec but no Python implementation
    """
    # This will raise AssertionError if something is wrong.
    assert_skill_packs_integrity(strict_unused=False)

```

If this test fails right now because some pack points at an incomplete spec (e.g. `feed_coke_ovens`), that’s not the test’s fault. That’s the whole point: it surfaces the break.

## 3. Semantics drift / snapshots (6.2)

Here we can’t import your semantics or virtue code yet without guessing paths, but we _can_ give you generic helpers that your eventual tests can call.

### `src/integration/validators/semantics_snapshots.py`
python:
```
# src/integration/validators/semantics_snapshots.py

from dataclasses import dataclass
from typing import Any, Callable, Dict


@dataclass
class SemanticsSnapshot:
    """
    Lightweight container for a 'semantics view' of the world.

    You can define this however you like in tests:
    - tech_state: string ID (e.g. "steam_age", "lv")
    - key_counts: number of items/blocks in certain semantic groups
    - any other scalar / small-structure information
    """
    tech_state: str
    features: Dict[str, Any]


def take_semantics_snapshot(
    *,
    world: Any,
    get_tech_state: Callable[[Any], Any],
    build_features: Callable[[Any, Any], Dict[str, Any]],
) -> SemanticsSnapshot:
    """
    Take a snapshot of semantics relevant for regression testing.

    Args:
        world:
            A world-like object / fixture.
        get_tech_state:
            Function that maps the world to a TechState-like object
            (must have .name or be convertible to string).
        build_features:
            Function that takes (world, tech_state) and returns a dict
            of scalar-ish features to compare against baselines.

    Returns:
        SemanticsSnapshot holding tech_state.name and the feature dict.
    """
    tech_state_obj = get_tech_state(world)
    tech_state_name = getattr(tech_state_obj, "name", str(tech_state_obj))

    features = build_features(world, tech_state_obj)

    return SemanticsSnapshot(
        tech_state=tech_state_name,
        features=features,
    )


def compare_semantics_snapshots(
    current: SemanticsSnapshot,
    expected: SemanticsSnapshot,
) -> Dict[str, str]:
    """
    Compare two SemanticsSnapshot instances and return a dict of differences.

    The returned dict maps a feature name to a human-readable difference description.
    Empty dict means "no differences detected" (within the feature sets compared).
    """
    diffs: Dict[str, str] = {}

    if current.tech_state != expected.tech_state:
        diffs["tech_state"] = f"{current.tech_state!r} != {expected.tech_state!r}"

    # Compare overlapping feature keys only
    keys = set(current.features.keys()) | set(expected.features.keys())
    for key in sorted(keys):
        cv = current.features.get(key)
        ev = expected.features.get(key)
        if cv != ev:
            diffs[key] = f"{cv!r} != {ev!r}"

    return diffs

```

You’d use this in tests like:

- Build a “golden” expected snapshot (LV world, certain group counts).
    
- Compute current snapshot.
    
- Assert `compare_semantics_snapshots(...)` is empty.
    

No imports from `semantics.*` here; you pass in the callables.

## 4. Planner stability utilities (6.3)

Same game: helpers that wrap your planner backend once M2 is wired.

### `src/integration/validators/planner_guardrails.py`
python:
```
# src/integration/validators/planner_guardrails.py

from dataclasses import dataclass
from typing import Any, Callable, Dict, List


@dataclass
class PlannerRunResult:
    """Simple container for a single planner run."""
    plans: List[Dict[str, Any]]


def run_planner_deterministic(
    *,
    planner_call: Callable[[Dict[str, Any]], Dict[str, Any]],
    payload: Dict[str, Any],
) -> PlannerRunResult:
    """
    Run the planner backend in a deterministic configuration.

    The caller is responsible for:
    - setting temperature/top_p/etc. in the payload,
    - ensuring the backend is configured to honor those settings.

    This wrapper only normalizes the output format for testing.
    """
    output = planner_call(payload)
    plans = output.get("plans", [])
    if not isinstance(plans, list):
        raise TypeError(f"planner returned non-list 'plans': {type(plans)!r}")
    return PlannerRunResult(plans=plans)


def assert_planner_stable_across_runs(
    *,
    planner_call: Callable[[Dict[str, Any]], Dict[str, Any]],
    base_payload: Dict[str, Any],
    runs: int = 3,
) -> None:
    """
    Assert that the planner is reasonably stable given identical inputs.

    Compares:
    - plan count across runs
    - basic shape of the first plan's 'steps' list

    This is intentionally conservative: it doesn't assert full equality of
    every field, but it will catch obvious instability, such as:
    - run 1 returns 1 plan, run 2 returns 5 plans, etc.
    - steps missing in some runs.
    """
    results: List[PlannerRunResult] = []
    for _ in range(runs):
        results.append(run_planner_deterministic(planner_call=planner_call, payload=base_payload))

    first = results[0]
    first_count = len(first.plans)

    for idx, res in enumerate(results[1:], start=2):
        count = len(res.plans)
        if count != first_count:
            raise AssertionError(
                f"planner instability: run 1 had {first_count} plans, "
                f"run {idx} had {count} plans"
            )

    # If there are no plans, there's nothing further to compare.
    if first_count == 0:
        return

    # Compare basic shape of first plan's steps across runs
    def plan_signature(plan: Dict[str, Any]) -> Any:
        steps = plan.get("steps", [])
        if not isinstance(steps, list):
            return ("invalid", None)
        # Only consider the 'skill' key of each step for now
        return tuple(step.get("skill") for step in steps)

    first_sig = plan_signature(first.plans[0])

    for idx, res in enumerate(results[1:], start=2):
        sig = plan_signature(res.plans[0])
        if sig != first_sig:
            raise AssertionError(
                f"planner instability in steps: signature mismatch between "
                f"run 1 and run {idx}"
            )

```

## 5. Virtue config regression hooks (6.4)

Again, generic utilities that compare virtue-score outputs across versions.

### `src/integration/validators/virtue_snapshots.py`
python:
```
# src/integration/validators/virtue_snapshots.py

from dataclasses import dataclass
from typing import Any, Callable, Dict, List


@dataclass
class VirtueScoreSnapshot:
    """
    Snapshot of virtue scores for a given plan in a given context.

    - context_id: e.g. 'lv_bootstrap'
    - plan_id: arbitrary identifier for the plan being evaluated
    - scores: raw scores from the virtue lattice (any JSON-serializable dict)
    """
    context_id: str
    plan_id: str
    scores: Dict[str, Any]


def take_virtue_snapshot(
    *,
    plan_summary: Any,
    context_id: str,
    virtue_config: Any,
    score_fn: Callable[[Any, str, Any], Dict[str, Any]],
) -> VirtueScoreSnapshot:
    """
    Call a scoring function to produce a VirtueScoreSnapshot.

    Args:
        plan_summary:
            Whatever object your virtue lattice uses to represent a plan.
        context_id:
            Virtue context identifier (e.g. 'lv_bootstrap').
        virtue_config:
            Config object / data structure for the lattice.
        score_fn:
            Callable of the form score_fn(plan_summary, context_id, virtue_config)
            that returns a dict of scores.

    Returns:
        VirtueScoreSnapshot bundling the context, a plan_id, and the scores.
    """
    scores = score_fn(plan_summary, context_id, virtue_config)
    plan_id = getattr(plan_summary, "id", "<unknown>")

    return VirtueScoreSnapshot(
        context_id=context_id,
        plan_id=plan_id,
        scores=scores,
    )


def compare_virtue_snapshots(
    baseline: VirtueScoreSnapshot,
    current: VirtueScoreSnapshot,
    *,
    tolerances: Dict[str, float] | None = None,
) -> Dict[str, str]:
    """
    Compare two VirtueScoreSnapshot instances and return differences.

    For numeric scores, you can optionally specify absolute tolerances per-key
    in the 'tolerances' dict. Non-numeric values are compared for exact equality.

    Returns:
        Dict[key, description] of differences. Empty dict means no detected change
        beyond tolerances.
    """
    diffs: Dict[str, str] = {}

    if baseline.context_id != current.context_id:
        diffs["context_id"] = f"{baseline.context_id!r} != {current.context_id!r}"

    if baseline.plan_id != current.plan_id:
        diffs["plan_id"] = f"{baseline.plan_id!r} != {current.plan_id!r}"

    base_scores = baseline.scores
    curr_scores = current.scores
    keys = set(base_scores.keys()) | set(curr_scores.keys())

    tol = tolerances or {}

    for key in sorted(keys):
        bv = base_scores.get(key)
        cv = curr_scores.get(key)

        if isinstance(bv, (int, float)) and isinstance(cv, (int, float)):
            limit = tol.get(key, 0.0)
            if abs(bv - cv) > limit:
                diffs[key] = f"{cv} (current) vs {bv} (baseline) exceeds tol={limit}"
        else:
            if bv != cv:
                diffs[key] = f"{cv!r} != {bv!r}"

    return diffs

```

This is what you’ll use later for “LV coke oven layout regression” etc: store a baseline snapshot, recompute, compare with tolerances.

---

## 6. What’s actually “live” now

Right now, concretely:

- `integration.validators.skill_integrity` is fully wired to your existing M5 code and can be used **today**.
    
- `tests/test_skill_packs_integrity.py` will immediately tell you if your packs + specs + impls are lying to each other.
    
- `semantics_snapshots.py`, `planner_guardrails.py`, and `virtue_snapshots.py` are utility modules ready to be hooked into M2/M3/M4 once those solidify, without breaking anything now.
    

If you want to wire planner / semantics / virtue tests next, we’ll need to see your current `semantics.*`, `virtues.*`, and `llm_stack.*` so we can plug these helpers into real pytest files.



---

## 7. Testing Strategy for Phase 1 Integration

1. **Unit-Level**
   - M3: semantics helpers (`lookup_item_semantics`, `get_current_tech_state`).
   - M4: `summarize_plan` on small synthetic plans.
   - M5: registry `describe_for_tech_state` with fake packs.

2. **Integration-Level (Offline)**
   - Fake worlds, fake tech_state, small sets of skills & packs.
   - Deterministic planner backend stub:
     - Instead of calling a real LLM, use a fake backend that returns predefined plans for given goals.
   - Run `run_phase1_planning_episode` and assert:
     - At least one plan is returned.
     - Virtue lattice picks the expected plan in controlled scenarios.

3. **Regression Scenarios (LV First)**
   - Coke oven / boiler layout scenario from the virtue doc. fileciteturn5file0
   - Early LV resource-gathering plan vs. random wandering plan.
   - These become canonical “sanity checks” for the full M2–M5 integration.


what you’re getting:

- A small `integration.testing` submodule with fakes you can reuse
    
- A real **offline Phase 1 integration test** that:
    
    - boots `phase1_integration` against fake M2/M3/M4
        
    - uses `run_phase1_planning_episode`
        
    - checks the LV coke-oven scenario behavior
        

Everything is self-contained and doesn’t require your real `semantics.*`, `virtues.*`, or `llm_stack.*` to exist. The test builds fake modules in `sys.modules` before importing `integration.phase1_integration`.

## 1. Testing submodule skeleton

### `src/integration/testing/__init__.py`
python:
```
# src/integration/testing/__init__.py

"""
Helpers and fakes for Phase 1 integration testing.

This package provides:

- Fake world / tech state types
- Fake planner backend (deterministic, no real LLM)
- Fake virtue / semantics behavior

These are used to:
- exercise integration.phase1_integration.run_phase1_planning_episode()
- define repeatable offline regression scenarios (e.g. LV coke ovens)
"""

from .fakes import (
    FakeWorldState,
    FakeTechState,
    FakePlannerBackend,
    fake_get_current_tech_state,
    fake_load_semantics_db,
    fake_load_virtue_config,
    fake_summarize_plan,
    fake_compare_plans,
)

```

## 2. Fakes for world, planner, virtues, semantics

### `src/integration/testing/fakes.py`
python:
```
# src/integration/testing/fakes.py

from dataclasses import dataclass
from typing import Any, Dict, List


# --- Fake world / tech state -----------------------------------------------

@dataclass
class FakeWorldState:
    """
    Minimal stand-in for WorldState for Phase 1 tests.

    Only requirement:
    - implements to_summary_dict() for logging / planner payload.
    """
    label: str
    extra: Dict[str, Any] | None = None

    def to_summary_dict(self) -> Dict[str, Any]:
        data = {"label": self.label}
        if self.extra:
            data.update(self.extra)
        return data


@dataclass
class FakeTechState:
    """
    Minimal TechState stand-in.

    Only requirement:
    - has a .name attribute.
    """
    name: str


# --- Fake semantics (M3) ----------------------------------------------------

def fake_get_current_tech_state(world: Any) -> FakeTechState:
    """
    Derive a simple tech state from the fake world.

    For now:
    - if world.label contains 'lv' (case-insensitive), return 'lv'
    - else return 'unknown'
    """
    label = getattr(world, "label", "").lower()
    if "lv" in label:
        return FakeTechState(name="lv")
    return FakeTechState(name="unknown")


def fake_load_semantics_db() -> Dict[str, Any]:
    """
    Return a tiny 'semantics DB' object.

    This doesn't need to be rich for Phase 1; it's just a placeholder
    passed through to summarize_plan().
    """
    return {
        "version": 1,
        "groups": {
            "coke_oven_blocks": ["gtnh:coke_brick"],
            "boiler_blocks": ["gtnh:boiler_casing"],
        },
    }


# --- Fake virtue config (M4) ------------------------------------------------

def fake_load_virtue_config() -> Dict[str, Any]:
    """
    Tiny virtue configuration placeholder.

    In real code this would be your lattice definition; for tests, we just
    need something to pass around to fake_compare_plans().
    """
    return {
        "version": 1,
        "contexts": {
            "lv_coke_ovens": {
                "preferred_plan_ids": ["clustered_coke_ovens"],
            },
            "lv_resources": {
                "preferred_plan_ids": ["focused_resource_run"],
            },
        },
    }


# --- Fake planner backend (M2) ----------------------------------------------

class FakePlannerBackend:
    """
    Deterministic planner backend for Phase 1 offline tests.

    generate_plan() looks at the text goal and returns static plans.
    This lets you test integration without a real LLM.
    """

    def generate_plan(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        goal: str = payload.get("goal", "").lower()

        plans: List[Dict[str, Any]]
        if "coke" in goal and "boiler" in goal:
            # LV coke ovens + boiler layout scenario
            plans = [
                {
                    "id": "clustered_coke_ovens",
                    "steps": [
                        {"skill": "build_coke_oven", "params": {"count": 16}},
                        {"skill": "build_railcraft_boiler", "params": {"size": "max"}},
                        {"skill": "wire_limited_distance", "params": {"voltage": "lv"}},
                    ],
                },
                {
                    "id": "scattered_coke_ovens",
                    "steps": [
                        {"skill": "build_coke_oven", "params": {"count": 16}},
                        {"skill": "scatter_boilers", "params": {}},
                        {"skill": "wire_long_distance", "params": {"voltage": "lv"}},
                    ],
                },
            ]
        elif "resource" in goal:
            # Early LV resource-gathering scenario
            plans = [
                {
                    "id": "focused_resource_run",
                    "steps": [
                        {"skill": "chop_tree", "params": {"radius": 8}},
                        {"skill": "mine_ore_cluster", "params": {"veins": 2}},
                    ],
                },
                {
                    "id": "random_wandering",
                    "steps": [
                        {"skill": "random_walk", "params": {"steps": 500}},
                    ],
                },
            ]
        else:
            # Generic fallback: single boring plan
            plans = [
                {
                    "id": "baseline_plan",
                    "steps": [
                        {"skill": "do_nothing_useful", "params": {}},
                    ],
                }
            ]

        return {"plans": plans}


# --- Fake virtue feature layer (M4) -----------------------------------------

@dataclass
class FakePlanSummary:
    """
    Minimal stand-in for a PlanSummary object produced by summarize_plan().

    Fields:
    - id: identifier of the underlying plan
    - skill_names: list of skills present in the plan
    - missing_skills: skills referenced in steps but absent from metadata
    """
    id: str
    skill_names: List[str]
    missing_skills: List[str]


def fake_summarize_plan(
    plan: Dict[str, Any],
    world: Any,
    tech_state: Any,
    semantics_db: Dict[str, Any],
    skill_metadata: Dict[str, Dict[str, Any]],
) -> FakePlanSummary:
    """
    Convert a raw plan dict into a FakePlanSummary.

    This is intentionally simple:
    - skill_names: the 'skill' field from each step
    - missing_skills: those skills not present in skill_metadata
    """
    plan_id = plan.get("id", "<unknown>")
    steps = plan.get("steps", []) or []

    skill_names: List[str] = []
    missing_skills: List[str] = []

    for step in steps:
        skill_name = step.get("skill")
        if not skill_name:
            continue
        skill_names.append(skill_name)
        if skill_name not in skill_metadata:
            missing_skills.append(skill_name)

    # Deduplicate missing_skills while preserving order
    seen = set()
    unique_missing: List[str] = []
    for s in missing_skills:
        if s in seen:
            continue
        seen.add(s)
        unique_missing.append(s)

    return FakePlanSummary(
        id=plan_id,
        skill_names=skill_names,
        missing_skills=unique_missing,
    )


def fake_compare_plans(
    plans: List[FakePlanSummary],
    context_id: str,
    config: Dict[str, Any],
):
    """
    Choose the "best" plan according to a tiny fake virtue policy.

    Policy:
    - If the virtue context has 'preferred_plan_ids', pick the first plan whose id
      appears there (if any).
    - Otherwise, pick the plan with the fewest missing_skills.
    - Scores dict is a simple heuristic containing:
      - 'chosen': chosen plan id
      - 'missing_skills': len(missing_skills)
      - 'skill_count': len(skill_names)
    """
    ctx_cfg = (config.get("contexts", {}) or {}).get(context_id, {}) or {}
    preferred_ids = ctx_cfg.get("preferred_plan_ids", []) or []

    best = None

    # First, look for an explicitly preferred id
    for p in plans:
        if p.id in preferred_ids:
            best = p
            break

    # If none explicitly preferred, pick the plan with least missing_skills
    if best is None:
        best = min(plans, key=lambda p: (len(p.missing_skills), -len(p.skill_names)))

    scores = {
        "chosen": best.id,
        "missing_skills": len(best.missing_skills),
        "skill_count": len(best.skill_names),
    }

    return best, scores

```

These fakes are:

- Deterministic
    
- Dead simple
    
- Just structured enough to let Phase 1 integration behave like a “real” episode.
    

---

## 3. Offline Phase 1 integration test

Now for the fun part: an actual pytest that:

- Installs fake modules into `sys.modules` **before** importing your integration orchestrator
    
- Runs `run_phase1_planning_episode` with the LV coke-oven scenario
    
- Asserts:
    
    - A plan is returned
        
    - The “clustered_coke_ovens” plan is chosen for that context
        
    - Scores is a dict (basic sanity)
        

### `tests/test_phase1_integration_offline.py`
python:
```
# tests/test_phase1_integration_offline.py

import sys
import types

from integration.testing import (
    FakeWorldState,
    FakePlannerBackend,
    fake_get_current_tech_state,
    fake_load_semantics_db,
    fake_load_virtue_config,
    fake_summarize_plan,
    fake_compare_plans,
)


def _install_fake_modules() -> None:
    """
    Install fake M2/M3/M4 modules into sys.modules so that
    integration.phase1_integration can import them without the real
    implementations existing yet.

    This lets us exercise the integration layer in pure offline mode.
    """

    # --- llm_stack.planner ---------------------------------------------------
    llm_planner_mod = types.ModuleType("llm_stack.planner")
    llm_planner_mod.PlannerBackend = FakePlannerBackend  # type alias for tests
    sys.modules["llm_stack.planner"] = llm_planner_mod

    # --- llm_stack.backends ---------------------------------------------------
    llm_backends_mod = types.ModuleType("llm_stack.backends")

    def get_planner_backend(name: str):
        # We ignore 'name' in this fake and always return a new FakePlannerBackend.
        return FakePlannerBackend()

    llm_backends_mod.get_planner_backend = get_planner_backend
    sys.modules["llm_stack.backends"] = llm_backends_mod

    # --- semantics.tech_state ------------------------------------------------
    semantics_tech_state = types.ModuleType("semantics.tech_state")
    semantics_tech_state.get_current_tech_state = fake_get_current_tech_state
    sys.modules["semantics.tech_state"] = semantics_tech_state

    # --- semantics.db --------------------------------------------------------
    semantics_db = types.ModuleType("semantics.db")
    semantics_db.load_semantics_db = fake_load_semantics_db
    sys.modules["semantics.db"] = semantics_db

    # --- virtues.loader ------------------------------------------------------
    virtues_loader = types.ModuleType("virtues.loader")
    virtues_loader.load_virtue_config = fake_load_virtue_config
    sys.modules["virtues.loader"] = virtues_loader

    # --- virtues.features ----------------------------------------------------
    virtues_features = types.ModuleType("virtues.features")
    virtues_features.summarize_plan = fake_summarize_plan
    sys.modules["virtues.features"] = virtues_features

    # --- virtues.lattice -----------------------------------------------------
    virtues_lattice = types.ModuleType("virtues.lattice")
    virtues_lattice.compare_plans = fake_compare_plans
    sys.modules["virtues.lattice"] = virtues_lattice


# Install fake modules BEFORE importing the integration orchestrator
_install_fake_modules()

from integration import phase1_integration as p1  # noqa: E402  (import after fakes)


def test_phase1_integration_lv_coke_oven_scenario() -> None:
    """
    Offline integration test for Phase 1.

    Scenario:
    - LV coke oven + boiler layout design
    - Fake planner returns both 'clustered' and 'scattered' plans
    - Fake virtue lattice prefers 'clustered_coke_ovens' in context 'lv_coke_ovens'

    Expectations:
    - run_phase1_planning_episode returns at least one plan
    - the chosen plan is the clustered coke-oven layout
    - virtue scores are returned as a dict
    """
    # Build a fake LV world
    world = FakeWorldState(label="lv_test_world")

    goal = "Design optimal LV coke oven and boiler layout"
    virtue_context_id = "lv_coke_ovens"

    best_plan, scores = p1.run_phase1_planning_episode(
        world=world,
        goal=goal,
        virtue_context_id=virtue_context_id,
    )

    # Basic sanity: a plan dictionary is returned
    assert isinstance(best_plan, dict)
    assert "id" in best_plan

    # For this scenario, fake_compare_plans should prefer the 'clustered' layout
    assert best_plan["id"] == "clustered_coke_ovens"

    # Scores should be a dict with at least the keys we defined in fake_compare_plans
    assert isinstance(scores, dict)
    assert scores.get("chosen") == "clustered_coke_ovens"
    assert "missing_skills" in scores
    assert "skill_count" in scores


def test_phase1_integration_lv_resource_scenario() -> None:
    """
    Offline integration test for early LV resource scenario.

    Scenario:
    - Goal mentions resources, triggering the resource plan set in FakePlannerBackend
    - Virtue context 'lv_resources' prefers 'focused_resource_run'
    """
    world = FakeWorldState(label="lv_resource_world")

    goal = "Plan an efficient early LV resource run (ores, logs)"
    virtue_context_id = "lv_resources"

    best_plan, scores = p1.run_phase1_planning_episode(
        world=world,
        goal=goal,
        virtue_context_id=virtue_context_id,
    )

    assert isinstance(best_plan, dict)
    assert best_plan["id"] == "focused_resource_run"

    assert isinstance(scores, dict)
    assert scores.get("chosen") == "focused_resource_run"

```

This test file gives you exactly what the section 7 bullets describe:

1. **Integration-level (offline)** test using:
    
    - fake world
        
    - fake tech_state
        
    - fake semantics
        
    - fake virtue lattice
        
    - fake planner backend
        
    - `run_phase1_planning_episode`
        
2. Two **regression scenarios**:
    
    - LV coke ovens / boiler layout
        
    - LV resource run vs wandering
        

No real LLMs were harmed in the making of this test.

---

## 4. How this fits the original bullets

- **Unit-level**
    
    - You already have:
        
        - M5: registry / packs unit tests (`test_skill_registry`, `test_skill_packs`).
            
    - When M3 / M4 implementations stabilize, you can add:
        
        - tests using `integration.validators.semantics_snapshots`
            
        - tests using `integration.validators.virtue_snapshots`
            
    - The fakes here don’t interfere with that.
        
- **Integration-level (offline)**
    
    - `test_phase1_integration_offline.py` is exactly this, wired to `run_phase1_planning_episode`.
        
- **Regression scenarios**
    
    - LV coke oven / boiler layout scenario: `test_phase1_integration_lv_coke_oven_scenario`
        
    - Early LV resource-gathering vs wandering: `test_phase1_integration_lv_resource_scenario`
        

Now when you run:
bash:
```
python3 -m pytest -q

```

you’re not just checking YAML syntax and registry behavior; you’re checking that the whole Phase 1 pipeline can:

- Build context
    
- Expose skills
    
- Call planner
    
- Run virtue selection
    
- Produce a stable, virtuous plan for specific LV scenarios
    

Which is exactly the point of this phase, give or take your addiction to coke ovens.


---

## 8. Summary

Phase 1 integration wires together:

- **M2** to generate candidate plans from goals and metadata.
- **M3** to interpret world & tech contexts.
- **M4** to impose your virtue lattice on decisions.
- **M5** to expose a clean, gated skill surface.

The `run_phase1_planning_episode` function is the central entrypoint for:

- Offline simulations
- Integration tests
- Future M8 agent loop embedding

Once this layer is stable, you can safely build M6–M8 on top without constantly rewriting the glue.


## Final Steps
## 1. Lock in the Phase 1 contract

These are basically “don’t let Future You wreck this accidentally.”

- **Freeze the public API of `run_phase1_planning_episode`**
    
    - Keep signature: `(world: WorldState, goal: str, virtue_context_id: str)`
        
    - If you add parameters later, make them keyword-only with safe defaults.

## A. Freeze the `run_phase1_planning_episode` API

You already _have_ the function in `src/integration/phase1_integration.py`. The “freeze” part just means:

1. **Keep this signature exactly:**
python:
```
def run_phase1_planning_episode(
    world: WorldState,
    goal: str,
    virtue_context_id: str,
):
    ...

```
If one day you need more knobs (like `max_plans`, `temperature`, etc.), add them as **optional keyword-only args** with defaults, _without_ moving or changing the first three:
python:
```
def run_phase1_planning_episode(
    world: WorldState,
    goal: str,
    virtue_context_id: str,
    *,
    max_plans: int = 5,
    temperature: float = 0.1,
):
    ...

```
So what you **do now**:

- Open `phase1_integration.py`
    
- Make sure the function has that 3-arg signature
    
- Add a clear docstring so Future You doesn’t “just tweak it a bit.”
    

Example docstring you can paste:
python:
```
def run_phase1_planning_episode(
    world: WorldState,
    goal: str,
    virtue_context_id: str,
):
    """
    Phase 1 integration entrypoint (M2–M5, offline only).

    Contract:
    - Inputs:
      - world: normalized WorldState snapshot
      - goal: natural-language planning objective
      - virtue_context_id: key into virtue config (e.g. "lv_coke_ovens")

    - Behavior:
      - Uses FAKE planner backend (deterministic, offline)
      - Uses FAKE virtue lattice / summarizer
      - Uses REAL semantics (M3) and REAL skills / registry / packs (M5)

    - Outputs:
      - (best_plan, scores)
        - best_plan: chosen plan dict from planner output
        - scores: per-plan virtue scores or summary metadata

    This API is considered stable for Phase 1. If you add parameters,
    make them keyword-only with safe defaults.
    """
    ...

```
That’s it. You’ve “frozen” the contract by:

- Fixing the signature
    
- Writing down what callers can rely on
    

No extra code gymnastics required.


---

## B. Document the integration assumptions in the Phase 1 md

You already appended that nice summary at the end of the Phase 1 Integration Module. Now you just need a tiny explicit section in the md that says:

> “Here’s what Phase 1 _actually_ uses under the hood.”

Add a section like this to your **Phase Integration 1** md (probably near the top or just above the testing section):


- **Document the integration in your Phase 1 md** (just a quick note, not a thesis):
    
    - Where planner backend is swapped to fake vs real
        
    - That Phase 1 assumes:
        
        - Fake planner (deterministic)
            
        - Fake virtue lattice
            
        - Real skills / registry / packs
            

You’ve basically done this conceptually already, but making it explicit keeps Phase 2 from mutating this file into a hydra.

markdown:
```
## Phase 1 Integration Contract

**Entry point**

- `integration.phase1_integration.run_phase1_planning_episode(world, goal, virtue_context_id)`

**Assumptions (Phase 1 only)**

- Planner (M2):
  - Uses a **fake / deterministic planner backend** (`FakePlannerBackend` from `integration.testing.fakes`)
  - No real LLM calls in Phase 1

- Virtue lattice (M4):
  - Uses **fake summarizer and comparer** from `integration.testing.fakes`
  - Real virtue configs can come later; Phase 1 just cares about the shape

- Semantics (M3):
  - Uses the **real semantics DB** and tech-state inference

- Skills (M5):
  - Uses **real SkillSpecs** (YAML), Skill Packs, and Python implementations
  - Registry + integrity tests must pass

**Stability**

- The `run_phase1_planning_episode(world, goal, virtue_context_id)` signature is considered stable for Phase 1.
- Future phases may add keyword-only parameters, but the three positional arguments must remain compatible.

```

You don’t have to write an essay; just clearly spell out:

- Which parts are fake vs real
    
- What that function is supposed to be, long-term

---

## 2. Tighten the switches between “fake” and “real”

Right now, you’ve hard-wired fakes for Phase 1. That’s good for tests, but you want a clean escape hatch later.

Minimal thing to do:

- In `phase1_integration.py`, add a tiny note / TODO next to imports:
python:
```
# Phase 1: we intentionally use fakes; Phase 2 will introduce a runtime switch.
from integration.testing.fakes import (
    fake_load_virtue_config as load_virtue_config,
    fake_summarize_plan as summarize_plan,
    fake_compare_plans as compare_plans,
)

```
No code change required, just mark it as intentional so you don’t forget why the hell integration is calling `integration.testing` in six months.

---

## 3. Sanity CLIs / scripts (optional but nice)

If you want to _use_ Phase 1 instead of just testing it:

- Add a tiny driver in `tools/phase1_demo.py`:
python:
```
# tools/phase1_demo.py

from spec.types import WorldState
from integration.phase1_integration import run_phase1_planning_episode


def main() -> None:
    world = WorldState(label="demo_world")  # however your fake/real world is built
    goal = "Design optimal LV coke oven and boiler layout"
    virtue_context_id = "lv_coke_ovens"

    best_plan, scores = run_phase1_planning_episode(
        world=world,
        goal=goal,
        virtue_context_id=virtue_context_id,
    )

    print("Best plan:", best_plan)
    print("Virtue scores:", scores)


if __name__ == "__main__":
    main()

```

Not required, but it’s helpful to have _one_ entrypoint you can run without pytest.

---

## 4. Coverage of “break glass” conditions

You _already_ log:

- no plans
    
- summarize_plan explosion
    
- compare_plans failure
    

If you want Phase 1 to be really solid, add **one** more test that asserts we don’t silently swallow failures:

- A test that sets fake planner to return `plans=[]` and asserts `run_phase1_planning_episode` raises `RuntimeError`.
    

But honestly, this is extra credit. Functionally you’re done.

tests/test_phase1_breakglass_no_plans.py
python:
```
import pytest

from integration.phase1_integration import run_phase1_planning_episode
from integration.testing.fakes import FakeWorldState, FakePlannerBackend

# Monkeypatch slot: the integration layer will call get_planner_backend()
# so we override it to return a FakePlannerBackend that produces zero plans.


class ZeroPlanBackend(FakePlannerBackend):
    def generate_plan(self, payload):
        return {"plans": []}


def test_phase1_no_plans_raises(monkeypatch):
    # Force the planner backend to be our zero-plan version
    monkeypatch.setattr(
        "integration.phase1_integration.get_planner_backend",
        lambda name: ZeroPlanBackend(),
    )

    world = FakeWorldState(label="empty_world")

    with pytest.raises(RuntimeError):
        run_phase1_planning_episode(
            world=world,
            goal="something impossible",
            virtue_context_id="lv_test",
        )

```

- Phase-1 integration is stable
    
- Contracts are pinned
    
- Fail-state behavior is guaranteed
    
- You’re ready for Phase 2 without the whole module mutating into a soup monster
    

Done.