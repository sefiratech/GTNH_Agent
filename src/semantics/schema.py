# TechState, SemanticsDB, and related types
# src/semantics/schema.py

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional


# ---------------------------------------------------------------------------
# Core semantic entities
# ---------------------------------------------------------------------------

@dataclass
class BlockInfo:
    """
    Semantic info for a block id/variant.

    This is the in-memory view of one block entry resolved from gtnh_blocks.yaml:
    - block_id: canonical block id ("gregtech:gt.blockmachines")
    - variant: specific variant key ("basic_machine_lv") or None for default
    - category: primary semantic category ("gt_machine", "ore", "log", etc.)
    - tags: auxiliary tags ("machine", "lv", "overworld", "multiblock_part", ...)
    - attributes: extra properties (tier, material, harvest level, etc.)
    """
    block_id: str
    variant: Optional[str] = None
    category: str = "unknown"
    tags: List[str] = field(default_factory=list)
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ItemInfo:
    """
    Semantic info for an item id/variant.

    Mirrors gtnh_items.yaml:
    - item_id: canonical item id ("gregtech:gt.metaitem.01")
    - variant: sub-key ("plate.copper", "circuit.lv", etc.) or None for default
    - category: semantic category ("plate", "dust", "circuit", "ingot", ...)
    - material: base material when applicable ("copper", "iron", ...)
    - tags: auxiliary tags ("metal", "lv", "quest-item", "fuel", ...)
    - attributes: free-form bag (tier, stack limits, fuel value, etc.)
    """
    item_id: str
    variant: Optional[str] = None
    category: str = "unknown"
    material: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    attributes: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Tech progression types
# ---------------------------------------------------------------------------

@dataclass
class TechState:
    """
    Current inferred GTNH tech progression.

    - active: current main tech state id ("steam_age", "lv_electric", ...)
    - unlocked: all tech state ids inferred as unlocked (DAG closure)
    - tags: semantic tags for the current overall situation ("early-game", "lv")
    - virtue_profile: optional name linking into virtues.yaml profile
    - evidence: structured details used for inference (seen machines/items/flags)
    """
    active: str
    unlocked: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    virtue_profile: Optional[str] = None
    evidence: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TechTarget:
    """
    Recommended next tech step(s).

    Typically derived from gtnh_tech_graph.yaml:

    - id: target tech state id ("lv_electric", "mv_electric", ...)
    - reason: short explanation of why this target is useful now
    - prerequisites_missing: which tech states still need to be unlocked
    - difficulty_score: relative estimate (higher = harder)
    - expected_benefits: human-readable list of what this unlock gives
      (e.g. ["LV ore doubling", "access to basic circuits"])
    """
    id: str
    reason: str
    prerequisites_missing: List[str] = field(default_factory=list)
    difficulty_score: float = 1.0
    expected_benefits: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Recipe / crafting types
# ---------------------------------------------------------------------------

@dataclass
class CraftResult:
    """
    Represents one result entry from a recipe IO section.

    - item_id: canonical item id
    - variant: variant key or None
    - count: quantity produced/consumed
    """
    item_id: str
    variant: Optional[str]
    count: int


@dataclass
class CraftOption:
    """
    A craftable option given current inventory/machines.

    This is a *derived* view built from gtnh_recipes.json plus current state:

    - recipe_id: id from gtnh_recipes.json ("gt:steam_macerator_copper_ore")
    - primary_output: main deterministic output of the recipe
    - secondary_outputs: byproducts / extra outputs (if any)
    - machine_id: machine required to execute this recipe (if applicable)
    - tier: semantic tier label ("steam", "lv", "mv", ...) if known
    - limiting_resource: description of what will run out first
      (e.g. "copper_ore", "steam", "eu/t capacity")
    - notes: free-form explanation for debugging / planner traces
    """
    recipe_id: str
    primary_output: CraftResult
    secondary_outputs: List[CraftResult] = field(default_factory=list)
    machine_id: Optional[str] = None
    tier: Optional[str] = None
    limiting_resource: Optional[str] = None
    notes: str = ""

