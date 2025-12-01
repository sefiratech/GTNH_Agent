**Purpose:**  
Define GTNH tech + world understanding as **data + logic**.

Overview:

- Data layer (config files):
    
    - Block categories (ores, machines, cables, etc.)
        
    - Item categories (plates, circuits, tools)
        
    - Tech states & prereqs (LV steam, MV, etc.)
        
- Logic layer (Python):
    
    - `infer_tech_state(inventory, machines)`
        
    - `suggest_next_targets(tech_state)`
        
    - `craftable_items(inventory, known_recipes)`
        
- **Dependencies:** `M1`
    
- **Difficulty:** ⭐⭐⭐⭐
    
- **Scalability/perf:**
    
    - Keep recipes & categories in JSON/YAML, not code.
        
    - Cache derived graphs (like tech dependency DAGs).


Details:

M3 is where “a bunch of NBT blobs” turns into “the agent actually knows what the hell a LV machine is.” So yes, this one matters.

---

# M3 · `world_semantics_gtnh`

**Phase:** P1 – Offline Core Pillars  
**Role:** Turn raw items/blocks/machines into **semantic GTNH knowledge**: tech progression, categories, and craftability.

Dependencies:

- **M1**: for `WorldState` type and general architecture contracts
    
- **M0**: indirectly (for knowing it’s GTNH 2.8.1, but that’s baked into data)
    

---

## 1. Responsibilities & Boundaries

### 1.1 What M3 owns

**Data (config, not code):**

- Block categories
    
    - `ore`, `gt_machine`, `gt_cable`, `multiblock_part`, `decoration`, etc.
        
- Item categories
    
    - `plate`, `dust`, `ingot`, `circuit`, `tool`, `fluid_cell`, etc.
        
- Tech states & prereqs
    
    - LV steam, LV electric, MV, etc, with graph of dependencies.
        
- Recipe and machine capabilities
    
    - Which machines can craft what; what tiers they require.
        

**Logic (pure Python):**

- `infer_tech_state(inventory, machines)`
    
- `suggest_next_targets(tech_state)`
    
- `craftable_items(inventory, known_recipes, machines)`
    
- plus helper categorization functions.
    

### 1.2 What M3 does _not_ do

- No actual **world scanning** (bot core provides that).
    
- No pathfinding, no placement.
    
- No LLM calls.
    
- No persistence of agent experience (that’s M10).
    

This is a **semantic oracle**, not the controller.

---

## 2. Directory & File Layout

Proposed structure:

```
gtnh_agent/
  config/
    gtnh_blocks.yaml       # block categories & metadata
    gtnh_items.yaml        # item categories & material info
    gtnh_tech_graph.yaml   # tech states & dependencies
    gtnh_recipes.json      # (optional) flattened or exported recipe info
  src/
    semantics/
      __init__.py
      schema.py            # dataclasses for tech state, targets, etc.
      loader.py            # load YAML/JSON into in-memory structures
      tech_state.py        # infer_tech_state, suggest_next_targets
      crafting.py          # craftable_items and helpers
      categorize.py        # item/block categorization helpers

```

You can expand as GTNH insanity escalates, but this is a sane core.

---

## 3. Data Layer Design

### 3.1 Block categories: `config/gtnh_blocks.yaml`

Example:
Yaml:
```
blocks:
  "gregtech:gt.blockmachines":
    default_category: "gt_machine"
    variants:
      "basic_machine_lv": { category: "gt_machine", tier: "lv" }
      "basic_machine_mv": { category: "gt_machine", tier: "mv" }

  "gregtech:gt.blockores":
    default_category: "ore"
    variants:
      "copper_ore": { material: "copper" }
      "iron_ore": { material: "iron" }

  "minecraft:log":
    default_category: "log"

```

3.2 Item categories: `config/gtnh_items.yaml`
Yaml:
```
items:
  "gregtech:gt.metaitem.01":
    variants:
      "plate.copper": { category: "plate", material: "copper" }
      "dust.copper":  { category: "dust",  material: "copper" }
      "circuit.lv":   { category: "circuit", tier: "lv" }

  "minecraft:iron_ingot":
    default_category: "ingot"
    material: "iron"

```

### 3.3 Tech graph: `config/gtnh_tech_graph.yaml`

Minimal LV-upwards example:
Yaml:
```
tech_states:
  "stone_age":
    description: "No machines, stone tools only."
    prerequisites: []
    unlocks:
      - "steam_age"

  "steam_age":
    description: "Basic steam machines, bronze tools."
    prerequisites: ["stone_age"]
    unlocks:
      - "lv_electric"

  "lv_electric":
    description: "LV power, LV machines, basic circuits."
    prerequisites: ["steam_age"]
    unlocks:
      - "mv_electric"

  "mv_electric":
    description: "MV machines, better circuits."
    prerequisites: ["lv_electric"]
    unlocks: []

```

### 3.4 Recipes: `config/gtnh_recipes.json`

Flattened recipes (can be exported from NEI/other tooling later):
JSON:
```
{
  "recipes": [
    {
      "id": "gt:steam_macerator_copper_ore",
      "type": "machine",
      "machine": "steam_macerator",
      "input": [{"item": "gregtech:gt.blockores", "variant": "copper_ore"}],
      "output": [{"item": "gregtech:gt.metaitem.01", "variant": "dust.copper", "count": 2}],
      "tier": "steam"
    }
  ]
}

```

## 4. Logic Layer: Core Types

### 4.1 Semantic types: `src/semantics/schema.py`
Python:
```
# src/semantics/schema.py

from dataclasses import dataclass                # simple data containers
from typing import Dict, Any, List, Optional     # type hints for flexible structures


@dataclass
class BlockInfo:
    """Semantic info for a block id/variant."""
    block_id: str                               # e.g. "gregtech:gt.blockmachines"
    variant: Optional[str]                      # e.g. "basic_machine_lv"
    category: str                               # e.g. "gt_machine", "ore"
    attributes: Dict[str, Any]                  # arbitrary attributes (tier, material, etc.)


@dataclass
class ItemInfo:
    """Semantic info for an item id/variant."""
    item_id: str                                # e.g. "gregtech:gt.metaitem.01"
    variant: Optional[str]                      # e.g. "plate.copper"
    category: str                               # e.g. "plate", "dust"
    material: Optional[str]                     # e.g. "copper"
    attributes: Dict[str, Any]                  # extra info (tier, stack limits, etc.)


@dataclass
class TechState:
    """Current inferred GTNH tech progression."""
    unlocked: List[str]                         # list of tech state ids (e.g. ["stone_age", "steam_age"])
    active: str                                 # current main state (e.g. "lv_electric")
    evidence: Dict[str, Any]                    # evidence used for inference (machines, items, etc.)


@dataclass
class TechTarget:
    """Recommended next tech step(s)."""
    id: str                                     # target tech state id (e.g. "lv_electric")
    reason: str                                 # why this target is useful
    prerequisites_missing: List[str]            # which states are still missing
    difficulty_score: float                     # relative difficulty estimate


@dataclass
class CraftOption:
    """A craftable item/result given current inventory/machines."""
    recipe_id: str                              # id from gtnh_recipes
    output_item: str                            # item id or variant
    output_count: int
    limiting_resource: Optional[str]            # description of what will run out first
    notes: str                                  # explanation for debugging

```

---

## 5. Logic: Loading & Categorization

### 5.1 Config loader: `src/semantics/loader.py`
Python:
```
# src/semantics/loader.py

from pathlib import Path                       # filesystem path handling
from typing import Dict, Any                   # type hints
import yaml                                    # parse YAML
import json                                    # parse JSON

from .schema import BlockInfo, ItemInfo       # our semantic dataclasses


CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"
# points at gtnh_agent/config/


def _load_yaml(name: str) -> Dict[str, Any]:
    """Load a YAML config under the config directory."""
    path = CONFIG_DIR / name                           # build full path
    with path.open("r", encoding="utf-8") as f:        # open file in text mode
        return yaml.safe_load(f)                       # parse YAML into dict


def _load_json(name: str) -> Dict[str, Any]:
    """Load a JSON config under the config directory."""
    path = CONFIG_DIR / name                           # build full path
    with path.open("r", encoding="utf-8") as f:        # open file in text mode
        return json.load(f)                            # parse JSON into dict


class SemanticsDB:
    """In-memory semantic database for GTNH blocks/items/recipes."""

    def __init__(self) -> None:
        # load base configs
        blocks_cfg = _load_yaml("gtnh_blocks.yaml")    # all block category data
        items_cfg = _load_yaml("gtnh_items.yaml")      # all item category data
        recipes_cfg = _load_json("gtnh_recipes.json")  # optional recipe data

        self._block_index: Dict[str, Dict[str, Any]] = blocks_cfg["blocks"]
        self._item_index: Dict[str, Dict[str, Any]] = items_cfg["items"]
        self._recipes = recipes_cfg.get("recipes", [])

    def get_block_info(self, block_id: str, variant: str | None) -> BlockInfo:
        """Return semantic info for a given block id/variant."""
        # look up the base entry for this block id
        entry = self._block_index.get(block_id, {})
        # get default category if present, else "unknown"
        default_category = entry.get("default_category", "unknown")
        variants = entry.get("variants", {})
        # choose variant-specific data if available
        if variant and variant in variants:
            v = variants[variant]
            category = v.get("category", default_category)
            attributes = {k: v for k, v in v.items() if k != "category"}
        else:
            category = default_category
            attributes = {k: v for k, v in entry.items() if k not in ("default_category", "variants")}
        # return a BlockInfo dataclass instance
        return BlockInfo(
            block_id=block_id,
            variant=variant,
            category=category,
            attributes=attributes,
        )

    def get_item_info(self, item_id: str, variant: str | None) -> ItemInfo:
        """Return semantic info for a given item id/variant."""
        # look up the base entry for this item id
        entry = self._item_index.get(item_id, {})
        default_category = entry.get("default_category", "unknown")
        material = entry.get("material")                # may be None
        variants = entry.get("variants", {})
        if variant and variant in variants:
            v = variants[variant]
            category = v.get("category", default_category)
            material = v.get("material", material)
            attributes = {k: v for k, v in v.items() if k not in ("category", "material")}
        else:
            category = default_category
            attributes = {
                k: v for k, v in entry.items()
                if k not in ("default_category", "variants", "material")
            }

        return ItemInfo(
            item_id=item_id,
            variant=variant,
            category=category,
            material=material,
            attributes=attributes,
        )

    @property
    def recipes(self) -> list[dict[str, Any]]:
        """Return the raw recipes list."""
        return self._recipes

```

---

## 6. Logic: Tech State & Progression

### 6.1 Tech graph loader: `src/semantics/tech_state.py`
Python:
```
# src/semantics/tech_state.py

from dataclasses import dataclass                # for small helper dataclasses
from typing import Dict, Any, List, Set          # type hints for collections
from pathlib import Path                         # file path handling
import yaml                                      # YAML parsing
import networkx as nx                            # DAG handling

from spec.types import WorldState                # from M1
from .schema import TechState, TechTarget        # our semantic types


CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"


@dataclass
class TechNode:
    """Metadata for a single tech node."""
    id: str                                      # tech state id
    description: str                             # human-readable description
    prerequisites: List[str]                     # list of prerequisite tech ids
    unlocks: List[str]                           # list of tech ids this unlocks


class TechGraph:
    """DAG representation of GTNH tech progression."""

    def __init__(self) -> None:
        # load YAML definition of tech graph
        raw = self._load_graph_cfg()
        self._nodes: Dict[str, TechNode] = {}
        self._graph = nx.DiGraph()               # directed graph for prereq relationships

        # populate nodes and edges based on config
        for tech_id, data in raw["tech_states"].items():
            node = TechNode(
                id=tech_id,
                description=data.get("description", ""),
                prerequisites=data.get("prerequisites", []),
                unlocks=data.get("unlocks", []),
            )
            self._nodes[tech_id] = node
            self._graph.add_node(tech_id)        # register node in graph
        # add edges from prerequisites to tech_id
        for tech_id, node in self._nodes.items():
            for prereq in node.prerequisites:
                self._graph.add_edge(prereq, tech_id)

        # ensure graph has no cycles
        if not nx.is_directed_acyclic_graph(self._graph):
            raise ValueError("Tech graph must be a DAG")

    def _load_graph_cfg(self) -> Dict[str, Any]:
        """Load tech graph YAML."""
        path = CONFIG_DIR / "gtnh_tech_graph.yaml"
        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def all_nodes(self) -> List[str]:
        """Return all tech ids."""
        return list(self._nodes.keys())

    def prerequisites_of(self, tech_id: str) -> List[str]:
        """Return direct prerequisites for a tech id."""
        return self._nodes[tech_id].prerequisites

    def successors_of(self, tech_id: str) -> List[str]:
        """Return tech states unlocked by this one."""
        return self._nodes[tech_id].unlocks

    def topological_order(self) -> List[str]:
        """Return a topologically sorted list of tech ids."""
        return list(nx.topological_sort(self._graph))


def infer_tech_state_from_world(world: WorldState) -> TechState:
    """
    Infer the tech state from current machines & inventory.
    This is a heuristic function that inspects machines in world.context, inventory items, etc.
    """
    # For now, assume world.context contains a summary like:
    # {"machines": [{"type": "steam_macerator"}, {"type": "lv_macerator"}]}
    machines = world.context.get("machines", [])
    inventory = world.inventory

    unlocked: Set[str] = set()
    evidence: Dict[str, Any] = {"machines": machines, "inventory": inventory}

    # naive heuristic: presence of certain machines implies unlocked states
    machine_types = {m.get("type") for m in machines}

    if machine_types:
        unlocked.add("stone_age")          # if you have machines, you've passed stone age

    if any("steam" in (m.get("type") or "") for m in machines):
        unlocked.add("steam_age")

    if any("lv_" in (m.get("type") or "") for m in machines):
        unlocked.add("lv_electric")

    if any("mv_" in (m.get("type") or "") for m in machines):
        unlocked.add("mv_electric")

    # pick the highest-priority active state by some simple ordering
    order = ["stone_age", "steam_age", "lv_electric", "mv_electric"]
    active = "stone_age"
    for t in order:
        if t in unlocked:
            active = t

    return TechState(
        unlocked=list(unlocked),
        active=active,
        evidence=evidence,
    )


def suggest_next_targets(tech_state: TechState, graph: TechGraph) -> List[TechTarget]:
    """Given current tech_state, suggest next reachable tech targets."""
    unlocked = set(tech_state.unlocked)
    suggestions: List[TechTarget] = []

    # iterate over all possible tech states in topological order
    for tech_id in graph.topological_order():
        if tech_id in unlocked:
            # already unlocked, skip
            continue
        prereqs = set(graph.prerequisites_of(tech_id))
        missing = sorted(prereqs - unlocked)
        if missing:
            # can't unlock yet, but track prerequisites you still need
            difficulty = float(len(missing))
            reason = f"Requires {', '.join(missing)}"
        else:
            difficulty = 1.0
            reason = "All prerequisites satisfied."

        # consider as a potential target if it's 1 step ahead or fewer than N missing
        suggestions.append(
            TechTarget(
                id=tech_id,
                reason=reason,
                prerequisites_missing=missing,
                difficulty_score=difficulty,
            )
        )

    # sort suggestions by difficulty (easier first)
    suggestions.sort(key=lambda t: t.difficulty_score)
    return suggestions

```

This is intentionally simple; you’ll refine heuristics as you get more GTNH-specific.

---

## 7. Logic: Craftability

### 7.1 Craft function: `src/semantics/crafting.py`
Python:
```
# src/semantics/crafting.py

from typing import Dict, Any, List                 # type hints
from collections import Counter                    # count items efficiently

from spec.types import WorldState                  # world representation from M1
from .schema import CraftOption                    # craftability result
from .loader import SemanticsDB                    # DB with recipes/index


def inventory_to_counts(inventory: List[Dict[str, Any]]) -> Counter:
    """
    Convert inventory list into a Counter keyed by (item_id, variant).
    Each entry in inventory is expected to have fields like:
    {"item_id": "gregtech:gt.metaitem.01", "variant": "plate.copper", "count": 32}
    """
    counts: Counter = Counter()
    for stack in inventory:
        key = (stack["item_id"], stack.get("variant"))
        counts[key] += int(stack.get("count", 0))
    return counts


def craftable_items(world: WorldState, db: SemanticsDB) -> List[CraftOption]:
    """
    Determine which recipes are craftable given current inventory and machines.
    This is a coarse, static analysis: it does not simulate time or location.
    """
    inv_counts = inventory_to_counts(world.inventory)
    machines = world.context.get("machines", [])

    # gather machine types for fast membership tests
    machine_types = {m.get("type") for m in machines}

    options: List[CraftOption] = []

    # iterate through all recipes in the DB
    for recipe in db.recipes:
        # check machine requirement if any
        required_machine = recipe.get("machine")
        if required_machine and required_machine not in machine_types:
            # can't run this recipe, missing machine
            continue

        # check all inputs are available in required quantities
        inputs = recipe.get("input", [])
        limiting_resource = None
        missing = False

        for inp in inputs:
            key = (inp["item"], inp.get("variant"))
            required = int(inp.get("count", 1))
            available = inv_counts.get(key, 0)
            if available < required:
                missing = True
                limiting_resource = f"Missing {required - available}x {key}"
                break

        if missing:
            # skip this recipe, not enough materials
            continue

        # choose first output as primary for now
        output = recipe.get("output", [])[0]
        recipe_id = recipe.get("id", "unknown")
        options.append(
            CraftOption(
                recipe_id=recipe_id,
                output_item=output["item"],
                output_count=output.get("count", 1),
                limiting_resource=limiting_resource,
                notes=f"Machine: {required_machine or 'none'}",
            )
        )

    return options

```

---

## 8. Integrating with `WorldState`

M1’s `WorldState` already exists. M3’s logic expects certain structure in:

- `world.inventory`: list of `{item_id, variant?, count}`
    
- `world.context["machines"]`: list of `{type: str, tier?: str, id?: str}`
    

So upstream modules (BotCore / M7) must normalize to this schema.

---

## 9. Testing & Simulation

M3 is **highly testable** without Minecraft.

### 9.1 Unit tests for categorization

- Build a minimal `gtnh_blocks.yaml` & `gtnh_items.yaml` in test directory.
    
- Point loader at test config via env var or temp directory.
    
- Assert:
Python:
```
db = SemanticsDB()
info = db.get_item_info("gregtech:gt.metaitem.01", "plate.copper")
assert info.category == "plate"
assert info.material == "copper"

```

### 9.2 Tech inference tests

Create dummy `WorldState` objects:
Python:
```
from spec.types import WorldState
from semantics.tech_state import infer_tech_state_from_world, TechGraph, suggest_next_targets

world = WorldState(
    tick=0,
    position={"x": 0, "y": 64, "z": 0},
    dimension="overworld",
    inventory=[],
    nearby_entities=[],
    blocks_of_interest=[],
    tech_state={},
    context={"machines": [{"type": "steam_macerator"}]},
)
state = infer_tech_state_from_world(world)
graph = TechGraph()
targets = suggest_next_targets(state, graph)
# assert that lv_electric eventually shows up as a target

```

### 9.3 Craftability tests

- Use a tiny recipes JSON:
    
    - One recipe for turning `ore` into `dust` with `steam_macerator`.
        
- Build a fake `WorldState` inventory containing enough ore.
    
- Check `craftable_items` returns that recipe as craftable.
    

### 9.4 Performance / caching

- On load, M3 builds:
    
    - `SemanticsDB` (items, blocks, recipes)
        
    - `TechGraph` (DAG)
        
- In tests, time these operations:
    
    - Ensure they’re “once per process” (module-level singleton or DI).
        
- For real usage, you’ll maintain a global instance or an injected singleton.
    

---

## 10. Libraries / Repos to Look At

You’re not copying them, just stealing patterns.

- **YAML & config:**
    
    - `pyyaml` – already used.
        
- **Graph / DAG:**
    
    - `networkx` – for tech graph, topological sorts, reachability.
        
- **Minecraft / GTNH refs (for data export later):**
    
    - GTNH repo / scripts that export recipes.
        
    - Any NEI dump tools / CraftTweaker script exporters.
        
- **General patterns:**
    
    - Any game that uses separate data files for items/tech trees (Factorio modding, etc.) has similar ideas.
        

---

## 11. Completion Criteria for M3

M3 is “functional” when all of this is true:

1. **Data is externalized**
    
    - `gtnh_blocks.yaml`, `gtnh_items.yaml`, `gtnh_tech_graph.yaml` exist and load without exploding.
        
    - Adding or changing categories / tech states does **not** require code changes.
        
2. **SemanticsDB works**
    
    - `SemanticsDB().get_item_info()` and `.get_block_info()` return meaningful `ItemInfo`/`BlockInfo` for at least:
        
        - Early LV tech items (plates, cables, circuits)
            
        - Basic machines (steam age + LV)
            
3. **Tech inference works at LV level**
    
    - `infer_tech_state_from_world` correctly identifies at least:
        
        - `stone_age`
            
        - `steam_age`
            
        - `lv_electric`
            
    - `suggest_next_targets` returns reasonable next steps given these.
        
4. **Craftability function is sane**
    
    - `craftable_items` returns a non-empty list for a test world with:
        
        - A configured machine
            
        - Sufficient input materials
            
    - Correctly filters out recipes missing machines or materials.
        
5. **Performance is acceptable**
    
    - Initial load of `SemanticsDB` + `TechGraph` finishes quickly (sub-second-ish for small data).
        
    - Subsequent calls (`get_item_info`, `infer_tech_state`, `craftable_items`) are in the “just Python function call” range.
        
6. **Test suite passes**
    
    - Unit tests verify:
        
        - Categorization
            
        - Tech inference
            
        - Suggestion logic
            
        - Craftability
            
    - CI runs them whenever config or semantics code changes.
        

Once this is nailed down, the agent can stop treating GTNH as “some cubes and pain” and start seeing it as a structured tech progression it can plan over. Which is the whole point.

## 12. Just Kidding - That was just the skeleton. Now to fill in the actual semantics

So you want the modpack to cough up its own ontology instead of you hand-authoring all of GTNH. Finally, a survival instinct.

We’ll do **Option A: runtime-assisted ingestion**, and we’ll structure it so:

- You run _one_ script in GTNH_Agent.
    
- It slurps pre-dumped JSON from the modpack.
    
- It writes `*.generated.yaml` / `*.generated.json`.
    
- M3 merges those with your hand-authored YAML.
    

I’ll give you:

1. The **expected raw dump formats** (what your MC-side script/mod should output)
    
2. A **loader upgrade** to support base + generated configs
    
3. A **semantics ingestion module** (`scripts/ingest_gtnh_semantics.py`)
    
4. Minimal plumbing in `src/semantics/ingest/__init__.py` (so it’s a real module)
    

Ordered from least dependent to most dependent.

---

## 1. Define the raw formats you’ll dump from GTNH

These live on the GTNH side. You’ll dump them into something like:

- `config/raw/registry_items.json`
    
- `config/raw/registry_blocks.json`
    
- `config/raw/recipes_raw.json`
    

### 1.1 `registry_items.json`

Expected shape:
jsonc:
```
{
  "items": [
    {
      "id": "minecraft:iron_ingot",
      "display_name": "Iron Ingot",
      "category_hint": "ingot",          // optional
      "material_hint": "iron",           // optional
      "tags": ["forge:ingots/iron"]      // optional
    },
    {
      "id": "gregtech:gt.metaitem.01",
      "variant": "plate.copper",         // optional: meta/variant key
      "display_name": "Copper Plate",
      "category_hint": "plate",
      "material_hint": "copper",
      "tags": ["gt:plate", "material:copper"]
    }
  ]
}

```

We treat:

- `id` = canonical item id
    
- `variant` = optional semantic/meta key (for GregTech-style meta-items)
    
- `category_hint` & `material_hint` = soft annotations you can compute however you like in the mod
    

---

### 1.2 `registry_blocks.json`

Expected shape:
jsonc:
```
{
  "blocks": [
    {
      "id": "minecraft:stone",
      "display_name": "Stone",
      "category_hint": "stone",
      "tags": ["minecraft:stone"]
    },
    {
      "id": "gregtech:gt.blockmachines",
      "variant": "basic_machine_lv",
      "display_name": "Basic Machine LV",
      "category_hint": "gt_machine",
      "tier_hint": "lv",
      "tags": ["gt:machine", "tier:lv"]
    }
  ]
}

```

Again:

- `id` = canonical block id
    
- `variant` = meta/variant key (GT machine meta-blocks)
    
- `category_hint`, `tier_hint` = optional, “best guess from mod side”
    

---

### 1.3 `recipes_raw.json`

Expected shape:
jsonc:
```
{
  "recipes": [
    {
      "id": "gt:steam_macerator_copper_ore",
      "type": "machine",
      "machine": "gregtech:steam_macerator",
      "tier": "steam",
      "inputs": [
        {
          "item": "gregtech:gt.blockores",
          "variant": "copper_ore",
          "count": 1
        }
      ],
      "outputs": [
        {
          "item": "gregtech:gt.metaitem.01",
          "variant": "dust.copper",
          "count": 2
        }
      ],
      "byproducts": []
    }
  ]
}

```

We’ll transform this into the canonical M3 format:
jsonc:
```
{
  "recipes": [
    {
      "id": "...",
      "type": "machine",
      "machine": "gregtech:steam_macerator",
      "tier": "steam",
      "io": {
        "inputs": [...],
        "outputs": [...],
        "byproducts": [...]
      }
    }
  ]
}

```

Your MC-side tooling just needs to spit out the raw version.

---

## 2. Upgrade `SemanticsDB` loader to support base + generated configs

We’ll extend `src/semantics/loader.py` so it:

- Loads:
    
    - `gtnh_blocks.yaml` (hand-authored)
        
    - `gtnh_blocks.generated.yaml` (auto-ingested, optional)
        
    - `gtnh_items.yaml`
        
    - `gtnh_items.generated.yaml`
        
    - `gtnh_recipes.json`
        
    - `gtnh_recipes.generated.json`
        
- Merges generated into base with the rule: **hand-authored wins**.
    

### 2.1 Full updated `src/semantics/loader.py`

Drop this in as a complete replacement:
Python:
```
# src/semantics/loader.py
"""
Semantic config loader for GTNH blocks/items/recipes.

Responsibility:
  - Load YAML/JSON config files from CONFIG_DIR
  - Merge hand-authored and generated configs:
      * gtnh_blocks.yaml              (base)
      * gtnh_blocks.generated.yaml    (auto-ingested, optional)
      * gtnh_items.yaml               (base)
      * gtnh_items.generated.yaml     (auto-ingested, optional)
      * gtnh_recipes.json             (base)
      * gtnh_recipes.generated.json   (auto-ingested, optional)
  - Provide a simple in-memory index for:
      * blocks
      * items
      * recipes
  - Map those into BlockInfo / ItemInfo dataclasses.
"""

from pathlib import Path
from typing import Dict, Any, List

import json
import yaml

from .schema import BlockInfo, ItemInfo


# Default config directory; tests monkeypatch this to point at a temp dir.
CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"


# ---------------------------------------------------------------------------
# Low-level loaders
# ---------------------------------------------------------------------------

def _load_yaml(name: str) -> Dict[str, Any]:
    """
    Load a YAML config file from CONFIG_DIR and return it as a dict.

    Raises FileNotFoundError if the file does not exist.
    """
    path = CONFIG_DIR / name
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML config {path} must be a mapping at top level.")
    return data


def _load_yaml_optional(name: str) -> Dict[str, Any]:
    """
    Load a YAML config file if it exists, otherwise return {}.
    """
    path = CONFIG_DIR / name
    if not path.exists():
        return {}
    return _load_yaml(name)


def _load_json(name: str) -> Dict[str, Any]:
    """
    Load a JSON config file from CONFIG_DIR and return it as a dict.

    Raises FileNotFoundError if the file does not exist.
    """
    path = CONFIG_DIR / name
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"JSON config {path} must be a mapping at top level.")
    return data


def _load_json_optional(name: str) -> Dict[str, Any]:
    """
    Load a JSON config file if it exists, otherwise return {}.
    """
    path = CONFIG_DIR / name
    if not path.exists():
        return {}
    return _load_json(name)


# ---------------------------------------------------------------------------
# Merge helpers
# ---------------------------------------------------------------------------

def _merge_mapping_with_precedence(
    base: Dict[str, Any],
    generated: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Merge two top-level mappings for items/blocks.

    - generated: auto-ingested data
    - base:      hand-authored data

    For a given key (item_id/block_id):
      - if present in base → base entry wins entirely
      - else → generated entry is used
    """
    result: Dict[str, Any] = dict(generated or {})
    result.update(base or {})
    return result


def _merge_recipe_lists(
    base_recipes: List[Dict[str, Any]],
    generated_recipes: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Merge recipe lists by recipe id.

    - generated recipes are loaded first
    - base recipes override any recipe with the same id

    This lets hand-authored recipes correct or replace ingested ones.
    """
    by_id: Dict[str, Dict[str, Any]] = {}

    for r in generated_recipes or []:
        rid = r.get("id")
        if isinstance(rid, str):
            by_id[rid] = r

    for r in base_recipes or []:
        rid = r.get("id")
        if isinstance(rid, str):
            by_id[rid] = r

    return list(by_id.values())


# ---------------------------------------------------------------------------
# SemanticsDB
# ---------------------------------------------------------------------------

class SemanticsDB:
    """
    In-memory semantic database for GTNH blocks/items/recipes.

    Config expectations:

      gtnh_blocks.yaml / gtnh_blocks.generated.yaml:
        blocks:
          "mod:block":
            default_category: "something"
            # optional extra attributes at top level
            variants:
              "variant_key":
                category: "..."
                # extra attributes per variant

      gtnh_items.yaml / gtnh_items.generated.yaml:
        items:
          "mod:item":
            default_category: "..."
            material: "iron"        # optional default material
            variants:
              "plate.copper":
                category: "plate"
                material: "copper"

    Recipes are read from:
      - gtnh_recipes.json
      - gtnh_recipes.generated.json
    and merged by id, with base recipes overriding generated ones.
    """

    def __init__(self) -> None:
        # Blocks: merge generated + base, base wins
        blocks_base = _load_yaml_optional("gtnh_blocks.yaml")
        blocks_gen = _load_yaml_optional("gtnh_blocks.generated.yaml")
        blocks_merged = _merge_mapping_with_precedence(
            base=blocks_base.get("blocks", {}) if isinstance(blocks_base, dict) else {},
            generated=blocks_gen.get("blocks", {}) if isinstance(blocks_gen, dict) else {},
        )

        # Items: merge generated + base, base wins
        items_base = _load_yaml_optional("gtnh_items.yaml")
        items_gen = _load_yaml_optional("gtnh_items.generated.yaml")
        items_merged = _merge_mapping_with_precedence(
            base=items_base.get("items", {}) if isinstance(items_base, dict) else {},
            generated=items_gen.get("items", {}) if isinstance(items_gen, dict) else {},
        )

        # Recipes: merge base + generated by id, base wins
        recipes_base = _load_json_optional("gtnh_recipes.json")
        recipes_gen = _load_json_optional("gtnh_recipes.generated.json")

        base_list = recipes_base.get("recipes", []) if isinstance(recipes_base, dict) else []
        gen_list = recipes_gen.get("recipes", []) if isinstance(recipes_gen, dict) else []

        recipes_merged = _merge_recipe_lists(base_list, gen_list)

        self._block_index: Dict[str, Dict[str, Any]] = blocks_merged
        self._item_index: Dict[str, Dict[str, Any]] = items_merged
        self._recipes: List[Dict[str, Any]] = recipes_merged

    # ------------------------------------------------------------------
    # Block lookup
    # ------------------------------------------------------------------

    def get_block_info(self, block_id: str, variant: str | None) -> BlockInfo:
        """
        Return semantic info for a given block id/variant.

        If the block id is unknown, category defaults to "unknown" and
        attributes are an empty dict.
        """
        entry = self._block_index.get(block_id, {})

        # Default category can be provided either as "default_category"
        # or as a shorthand "category" at top level.
        default_category = entry.get(
            "default_category",
            entry.get("category", "unknown"),
        )

        variants = entry.get("variants", {}) or {}

        if variant and variant in variants:
            v = variants[variant]
            category = v.get("category", default_category)
            attributes = {
                k: v for k, v in v.items()
                if k != "category"
            }
        else:
            category = default_category
            attributes = {
                k: v
                for k, v in entry.items()
                if k not in ("default_category", "category", "variants")
            }

        return BlockInfo(
            block_id=block_id,
            variant=variant,
            category=category,
            attributes=attributes,
        )

    # ------------------------------------------------------------------
    # Item lookup
    # ------------------------------------------------------------------

    def get_item_info(self, item_id: str, variant: str | None) -> ItemInfo:
        """
        Return semantic info for a given item id/variant.

        If the item id is unknown, category defaults to "unknown" and
        material defaults to None.
        """
        entry = self._item_index.get(item_id, {})

        # Default category: support both "default_category" and "category"
        default_category = entry.get(
            "default_category",
            entry.get("category", "unknown"),
        )
        # Default material at top level (optional)
        material = entry.get("material")

        variants = entry.get("variants", {}) or {}

        if variant and variant in variants:
            v = variants[variant]
            category = v.get("category", default_category)
            # Variant may override material
            material = v.get("material", material)
            attributes = {
                k: v
                for k, v in v.items()
                if k not in ("category", "material")
            }
        else:
            category = default_category
            attributes = {
                k: v
                for k, v in entry.items()
                if k not in ("default_category", "category", "variants", "material")
            }

        return ItemInfo(
            item_id=item_id,
            variant=variant,
            category=category,
            material=material,
            attributes=attributes,
        )

    # ------------------------------------------------------------------
    # Recipes
    # ------------------------------------------------------------------

    @property
    def recipes(self) -> List[Dict[str, Any]]:
        """Return the raw recipes list."""
        return self._recipes

```

## 3. Add the ingestion module

We’ll keep this lightweight:

- A tiny package: `src/semantics/ingest/__init__.py`
    
- A script: `scripts/ingest_gtnh_semantics.py`
    

### 3.1 Package marker: `src/semantics/ingest/__init__.py`

Create this file (empty is fine, but let’s be slightly informative):
Python:
```
# src/semantics/ingest/__init__.py
"""
Runtime-assisted ingestion utilities for GTNH semantics.

These tools are *offline* helpers that:
  - read raw dumps from a running GTNH instance
  - normalize them into:
      * gtnh_items.generated.yaml
      * gtnh_blocks.generated.yaml
      * gtnh_recipes.generated.json
for use by SemanticsDB at runtime.

They are not imported in hot paths; they exist for build/maintenance workflows.
"""

```

### 3.2 Ingestion script: `scripts/ingest_gtnh_semantics.py`

This is the main “run once after dumping” tool.

It will:

- Read raw JSON dumps from a directory (default: `config/raw`)
    
- Generate:
    
    - `config/gtnh_items.generated.yaml`
        
    - `config/gtnh_blocks.generated.yaml`
        
    - `config/gtnh_recipes.generated.json`
        

Here’s the full script:
Python:
```
# scripts/ingest_gtnh_semantics.py

"""
Ingest raw GTNH modpack dumps into M3 semantics configs.

Usage (from project root):

    (.venv) python3 scripts/ingest_gtnh_semantics.py \
        --config-dir config \
        --raw-dir config/raw

Expected raw files (JSON, produced by a GTNH-side mod/script):

  - {raw_dir}/registry_items.json
      {
        "items": [
          {
            "id": "minecraft:iron_ingot",
            "display_name": "Iron Ingot",
            "category_hint": "ingot",        # optional
            "material_hint": "iron",         # optional
            "tags": ["forge:ingots/iron"]    # optional
          },
          {
            "id": "gregtech:gt.metaitem.01",
            "variant": "plate.copper",       # optional
            "display_name": "Copper Plate",
            "category_hint": "plate",
            "material_hint": "copper",
            "tags": ["gt:plate", "material:copper"]
          }
        ]
      }

  - {raw_dir}/registry_blocks.json
      {
        "blocks": [
          {
            "id": "minecraft:stone",
            "display_name": "Stone",
            "category_hint": "stone",
            "tags": ["minecraft:stone"]
          },
          {
            "id": "gregtech:gt.blockmachines",
            "variant": "basic_machine_lv",
            "display_name": "Basic Machine LV",
            "category_hint": "gt_machine",
            "tier_hint": "lv",
            "tags": ["gt:machine", "tier:lv"]
          }
        ]
      }

  - {raw_dir}/recipes_raw.json
      {
        "recipes": [
          {
            "id": "gt:steam_macerator_copper_ore",
            "type": "machine",
            "machine": "gregtech:steam_macerator",
            "tier": "steam",
            "inputs": [
              { "item": "gregtech:gt.blockores", "variant": "copper_ore", "count": 1 }
            ],
            "outputs": [
              { "item": "gregtech:gt.metaitem.01", "variant": "dust.copper", "count": 2 }
            ],
            "byproducts": []
          }
        ]
      }

Output (into {config_dir}):

  - gtnh_items.generated.yaml
  - gtnh_blocks.generated.yaml
  - gtnh_recipes.generated.json

These are *merged* with hand-authored configs at runtime by SemanticsDB.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import yaml


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required raw dump: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Raw JSON {path} must be an object at top level.")
    return data


# ---------------------------------------------------------------------------
# Items
# ---------------------------------------------------------------------------

def build_items_generated(raw_items: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build gtnh_items.generated.yaml structure from registry_items.json.

    Output structure:

      items:
        "mod:item":
          default_category: "..."
          material: "iron"
          variants:
            "plate.copper":
              category: "plate"
              material: "copper"
              tags: [...]
    """
    items_section: Dict[str, Any] = {}
    items = raw_items.get("items", []) or []
    if not isinstance(items, list):
        raise ValueError("registry_items.json: 'items' must be a list.")

    for entry in items:
        if not isinstance(entry, dict):
            continue

        item_id = entry.get("id")
        if not isinstance(item_id, str):
            continue

        variant = entry.get("variant")
        cat_hint = entry.get("category_hint")
        mat_hint = entry.get("material_hint")
        tags = entry.get("tags") or []

        if not isinstance(tags, list):
            tags = [tags]

        item_cfg = items_section.setdefault(item_id, {})

        if variant:
            # Ensure variants dict exists
            variants = item_cfg.setdefault("variants", {})
            v_cfg = variants.setdefault(variant, {})

            if cat_hint:
                v_cfg.setdefault("category", cat_hint)
            if mat_hint:
                v_cfg.setdefault("material", mat_hint)
            if tags:
                v_cfg.setdefault("tags", tags)
        else:
            # Top-level (non-variant) item
            if cat_hint:
                item_cfg.setdefault("default_category", cat_hint)
            if mat_hint:
                item_cfg.setdefault("material", mat_hint)
            if tags:
                item_cfg.setdefault("tags", tags)

    return {"items": items_section}


# ---------------------------------------------------------------------------
# Blocks
# ---------------------------------------------------------------------------

def build_blocks_generated(raw_blocks: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build gtnh_blocks.generated.yaml structure from registry_blocks.json.

    Output structure:

      blocks:
        "mod:block":
          default_category: "stone"
          variants:
            "basic_machine_lv":
              category: "gt_machine"
              tier: "lv"
              tags: [...]
    """
    blocks_section: Dict[str, Any] = {}
    blocks = raw_blocks.get("blocks", []) or []
    if not isinstance(blocks, list):
        raise ValueError("registry_blocks.json: 'blocks' must be a list.")

    for entry in blocks:
        if not isinstance(entry, dict):
            continue

        block_id = entry.get("id")
        if not isinstance(block_id, str):
            continue

        variant = entry.get("variant")
        cat_hint = entry.get("category_hint")
        tier_hint = entry.get("tier_hint")
        tags = entry.get("tags") or []

        if not isinstance(tags, list):
            tags = [tags]

        block_cfg = blocks_section.setdefault(block_id, {})

        if variant:
            variants = block_cfg.setdefault("variants", {})
            v_cfg = variants.setdefault(variant, {})

            if cat_hint:
                v_cfg.setdefault("category", cat_hint)
            if tier_hint:
                v_cfg.setdefault("tier", tier_hint)
            if tags:
                v_cfg.setdefault("tags", tags)
        else:
            if cat_hint:
                block_cfg.setdefault("default_category", cat_hint)
            if tags:
                block_cfg.setdefault("tags", tags)

    return {"blocks": blocks_section}


# ---------------------------------------------------------------------------
# Recipes
# ---------------------------------------------------------------------------

def build_recipes_generated(raw_recipes: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build gtnh_recipes.generated.json structure from recipes_raw.json.

    Input recipe entry (example):

      {
        "id": "gt:steam_macerator_copper_ore",
        "type": "machine",
        "machine": "gregtech:steam_macerator",
        "tier": "steam",
        "inputs": [...],
        "outputs": [...],
        "byproducts": [...]
      }

    Output recipe entry:

      {
        "id": "gt:steam_macerator_copper_ore",
        "type": "machine",
        "machine": "gregtech:steam_macerator",
        "tier": "steam",
        "io": {
          "inputs": [...],
          "outputs": [...],
          "byproducts": [...]
        }
      }
    """
    recipes_out: List[Dict[str, Any]] = []
    recipes_in = raw_recipes.get("recipes", []) or []
    if not isinstance(recipes_in, list):
        raise ValueError("recipes_raw.json: 'recipes' must be a list.")

    for r in recipes_in:
        if not isinstance(r, dict):
            continue

        rid = r.get("id")
        if not isinstance(rid, str):
            continue

        r_type = r.get("type", "crafting")
        machine = r.get("machine")
        tier = r.get("tier")

        inputs = r.get("inputs", []) or []
        outputs = r.get("outputs", []) or []
        byproducts = r.get("byproducts", []) or []

        recipes_out.append(
            {
                "id": rid,
                "type": r_type,
                "machine": machine,
                "tier": tier,
                "io": {
                    "inputs": inputs,
                    "outputs": outputs,
                    "byproducts": byproducts,
                },
            }
        )

    return {"recipes": recipes_out}


# ---------------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------------

def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Ingest raw GTNH dumps into generated semantics configs."
    )
    parser.add_argument(
        "--config-dir",
        type=str,
        default="config",
        help="Path to GTNH_Agent config directory (default: config/).",
    )
    parser.add_argument(
        "--raw-dir",
        type=str,
        default=None,
        help="Path to directory containing raw modpack dumps "
             "(default: <config-dir>/raw).",
    )

    args = parser.parse_args(argv)

    config_dir = Path(args.config_dir).resolve()
    raw_dir = Path(args.raw_dir).resolve() if args.raw_dir else (config_dir / "raw")

    if not config_dir.exists():
        raise SystemExit(f"Config directory does not exist: {config_dir}")
    if not raw_dir.exists():
        raise SystemExit(f"Raw dump directory does not exist: {raw_dir}")

    # Load raw JSONs
    items_raw_path = raw_dir / "registry_items.json"
    blocks_raw_path = raw_dir / "registry_blocks.json"
    recipes_raw_path = raw_dir / "recipes_raw.json"

    raw_items = _load_json(items_raw_path)
    raw_blocks = _load_json(blocks_raw_path)
    raw_recipes = _load_json(recipes_raw_path)

    # Build generated structures
    items_generated = build_items_generated(raw_items)
    blocks_generated = build_blocks_generated(raw_blocks)
    recipes_generated = build_recipes_generated(raw_recipes)

    # Write outputs
    items_out_path = config_dir / "gtnh_items.generated.yaml"
    blocks_out_path = config_dir / "gtnh_blocks.generated.yaml"
    recipes_out_path = config_dir / "gtnh_recipes.generated.json"

    items_out_path.write_text(
        yaml.safe_dump(items_generated, sort_keys=True),
        encoding="utf-8",
    )
    blocks_out_path.write_text(
        yaml.safe_dump(blocks_generated, sort_keys=True),
        encoding="utf-8",
    )
    recipes_out_path.write_text(
        json.dumps(recipes_generated, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print(f"[ingest] Wrote {items_out_path}")
    print(f"[ingest] Wrote {blocks_out_path}")
    print(f"[ingest] Wrote {recipes_out_path}")


if __name__ == "__main__":
    main()

```

## 4. How this fits into your workflow

### Step-by-step, in sane order:

1. **Add the loader upgrade**
    
    - Replace `src/semantics/loader.py` with the updated version above.
        
    - Run `pytest` to confirm all existing tests still pass.
        
2. **Add the ingestion package & script**
    
    - Create `src/semantics/ingest/__init__.py` as given.
        
    - Add `scripts/ingest_gtnh_semantics.py`.
        
3. **On the GTNH side, make your runtime dumps**
    
    - Use a tiny helper mod, CraftTweaker script, or whatever:
        
        - Dump items → `config/raw/registry_items.json`
            
        - Dump blocks → `config/raw/registry_blocks.json`
            
        - Dump recipes → `config/raw/recipes_raw.json`
            
    - Just match the raw schemas we defined. You can extend with more hints; the script will ignore unknown keys.
        
4. **In GTNH_Agent, run the ingestion script**
    

From your project root:
bash:
```
(.venv) python3 scripts/ingest_gtnh_semantics.py \
    --config-dir config \
    --raw-dir config/raw

```
That will spit out:

- `config/gtnh_items.generated.yaml`
    
- `config/gtnh_blocks.generated.yaml`
    
- `config/gtnh_recipes.generated.json`
    

5. **Run tests again**
bash:
```
python3 -m pytest -q

```

If anything breaks, it’s because your raw dumps are malformed, not because M3 is fragile.

6. **At runtime**, M3 does:
    

- Load base + generated configs via `SemanticsDB`
    
- Combine them with “hand-authored wins” semantics
    
- Feed the agent & planners a **complete, data-driven** GTNH semantics layer
    

You now have:

- A clean M3 module that doesn’t depend on you manually mimicking NEI
    
- A repeatable ingestion pipeline you can re-run after modpack updates
    
- A clear boundary:
    
    - Modpack dumps raw data
        
    - `ingest_gtnh_semantics.py` normalizes it
        
    - `SemanticsDB` serves it
        

Which is about as civilized as you’re going to get in GTNH.