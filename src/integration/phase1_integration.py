# src/integration/phase1_integration.py

# --- M4: Virtue Lattice ------------------------------------------------------
# Phase 1: we intentionally use fakes; Phase 2 will introduce a runtime switch.
from integration.testing.fakes import (  # type: ignore
    fake_load_virtue_config as load_virtue_config,
    fake_summarize_plan as summarize_plan,
    fake_compare_plans as compare_plans,
)

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

from typing import Any as _AnyAlias

from spec.types import WorldState  # real type defined in your codebase

# For now, keep TechState / Action as loose aliases so we don't depend
# on spec.types exporting them yet.
TechState = _AnyAlias
Action = _AnyAlias


# --- M2: LLM Stack interfaces ------------------------------------------------
# Try to use the real LLM stack; fall back to fakes if not available.

try:
    from llm_stack.planner import PlannerBackend           # type: ignore
    from llm_stack.backends import get_planner_backend     # type: ignore
except ImportError:
    # Phase 1 / tests: use deterministic fake backend
    from integration.testing.fakes import FakePlannerBackend as PlannerBackend  # type: ignore

    def get_planner_backend(name: str) -> PlannerBackend:
        return PlannerBackend()


# --- M3: Semantics / Tech State ---------------------------------------------

try:
    from semantics.tech_state import get_current_tech_state     # type: ignore
except ImportError:
    # Phase 1 / tests: fallback to fake tech-state inference
    from integration.testing.fakes import fake_get_current_tech_state as get_current_tech_state  # type: ignore

try:
    from semantics.db import load_semantics_db                  # type: ignore
except ImportError:
    from integration.testing.fakes import fake_load_semantics_db as load_semantics_db  # type: ignore


# --- M4: Virtue Lattice (real-path version kept for future phases) ----------
"""
try:
    from virtues.loader import load_virtue_config               # type: ignore
except ImportError:
    from integration.testing.fakes import fake_load_virtue_config as load_virtue_config  # type: ignore

try:
    from virtues.features import summarize_plan                 # type: ignore
except ImportError:
    from integration.testing.fakes import fake_summarize_plan as summarize_plan  # type: ignore

try:
    from virtues.lattice import compare_plans                   # type: ignore
except ImportError:
    from integration.testing.fakes import fake_compare_plans as compare_plans  # type: ignore
"""

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
        tech_name = getattr(tech_state, "name", str(tech_state))
        episode_logger.log_context_built(
            tech_state_name=tech_name,
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

    tech_name = getattr(tech_state, "name", str(tech_state))

    # Determine which Skill Packs should be enabled at this tech_state
    enabled_pack_names: List[str] = [
        name
        for name, pack in skill_packs.items()
        if pack.requires_tech_state == tech_name                     # compare tech_state name
    ]

    # Ask registry for a filtered description of skills
    skill_metadata: Dict[str, Dict[str, Any]] = skill_registry.describe_for_tech_state(
        tech_state=tech_name,                                        # string tech ID for registry
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
        tech_name = getattr(tech_state, "name", str(tech_state))
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
        world_summary = {"raw": repr(world)}

    tech_name = getattr(tech_state, "name", str(tech_state))

    planner_input: Dict[str, Any] = {
        "goal": goal,                                      # planning objective text
        "world_summary": world_summary,                    # compressed WorldState
        "tech_state": tech_name,                           # string representation of TechState
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
            summary = summarize_plan(
                plan=plan,                                            # raw plan from planner
                world=world,                                          # WorldState
                tech_state=tech_state,                                # TechState-like
                semantics_db=semantics_db,                            # semantics reference
                skill_metadata=all_skill_metadata,                    # all SkillSpecs
            )
        except Exception as exc:
            if episode_logger is not None:
                episode_logger.log_exception(phase="summarize_plan", exc=exc)
            raise

        if episode_logger is not None:
            skill_names = getattr(summary, "skill_names", [])
            missing_skills = getattr(summary, "missing_skills", [])
            episode_logger.log_plan_features(
                plan_id=plan_id,
                skill_names=skill_names,
                missing_skills=missing_skills,
            )

        plan_summaries.append(summary)

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

    if episode_logger is not None:
        episode_logger.log_virtue_scores(
            plan_id=getattr(best_summary, "id", "<unknown>"),
            scores=scores,
        )
        episode_logger.log_selected_plan(
            plan_id=getattr(best_summary, "id", "<unknown>"),
            scores=scores,
        )

    best_plan_id = getattr(best_summary, "id", None)
    if best_plan_id is None:
        # Fallback: assume first plan if summary lacks id
        best_plan = plans[0]
    else:
        best_plan = next(p for p in plans if p.get("id") == best_plan_id)

    return best_plan, scores


def run_phase1_planning_episode(
    world: WorldState,
    goal: str,
    virtue_context_id: str,
    *,
    episode_id: Optional[str] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Phase 1 integration entrypoint (M2–M5, offline only).

    Contract (Phase 1):

    - Inputs:
      - world: normalized WorldState snapshot
      - goal: natural-language planning objective
      - virtue_context_id: key into virtue config
        (e.g. "lv_coke_ovens", "lv_resources")

    - Behavior:
      - Uses FAKE planner backend (deterministic, offline) from integration.testing.fakes
      - Uses FAKE virtue summarizer / lattice (summarize_plan, compare_plans)
      - Uses REAL semantics (M3) where available
      - Uses REAL skills / SkillSpecs / SkillPacks / SkillRegistry (M5)
      - Runs a single planning episode:
        - build context
        - compute visible skills for current tech_state
        - call planner to get candidate plans
        - score and select best plan via virtue lattice

    - Outputs:
      - (best_plan, scores)
        - best_plan: chosen plan dict from planner output
        - scores: per-plan virtue scores or summary metadata (backend-dependent)

    Stability:

    - The (world, goal, virtue_context_id) signature is considered stable for Phase 1.
    - Additional parameters (like episode_id) must be keyword-only with safe defaults.

    Logging:

    - Each call gets an episode-level correlation ID (episode_id).
    - All logs from this episode are tagged with that ID.
    """
    if episode_id is None:
        episode_id = generate_correlation_id()
    episode_logger = EpisodeLogger(episode_id)

    # Build shared integration context (M2, M3, M4, M5)
    context = build_phase1_context(world, episode_logger=episode_logger)

    # Compute which skills are visible at this tech_state
    skill_metadata = compute_available_skills(context)

    # Extract planner backend and tech_state from context
    planner: PlannerBackend = context["planner"]
    tech_state: TechState = context["tech_state"]

    # Call planner to generate candidate plans for the given goal
    plans = call_planner_llm(
        planner=planner,
        goal=goal,
        world=world,
        tech_state=tech_state,
        skill_metadata=skill_metadata,
        virtue_context_id=virtue_context_id,
        episode_logger=episode_logger,
    )

    # If no plans returned, treat this as a failure for now
    if not plans:
        episode_logger.log_no_plans(goal)
        raise RuntimeError("Planner returned no plans for goal: %r" % (goal,))

    # Use virtue lattice to select the best plan among candidates
    best_plan, scores = select_best_plan(
        plans=plans,
        world=world,
        context=context,
        virtue_context_id=virtue_context_id,
    )

    return best_plan, scores

