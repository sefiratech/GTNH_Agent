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

