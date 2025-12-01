# src/skills/packs.py

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import yaml


# Directory containing skill pack definitions:
# gtnh_agent/config/skill_packs/*.yaml
PACKS_DIR = Path(__file__).resolve().parents[2] / "config" / "skill_packs"


@dataclass
class SkillPack:
    """
    Declarative definition of a Skill Pack.

    Each pack:
    - is identified by name
    - requires a specific tech_state to be considered active
    - can be tagged for grouping / curriculum
    - lists the skill names it enables
    """
    name: str
    requires_tech_state: str
    tags: List[str]
    skills: List[str]


def _load_yaml(path: Path) -> dict:
    """
    Load a YAML file into a dict.

    Returns an empty dict if the file is empty, rather than None.
    """
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def load_skill_pack_from_file(path: Path) -> SkillPack:
    """
    Parse a single skill pack YAML file into a SkillPack dataclass.
    """
    raw = _load_yaml(path)

    name = raw["name"]
    requires_tech_state = raw["requires_tech_state"]
    tags = raw.get("tags", []) or []
    skills = raw.get("skills", []) or []

    if not skills:
        raise ValueError(f"Skill pack '{name}' in {path} has no skills defined.")

    return SkillPack(
        name=name,
        requires_tech_state=requires_tech_state,
        tags=tags,
        skills=skills,
    )


def load_all_skill_packs(packs_dir: Optional[Path] = None) -> Dict[str, SkillPack]:
    """
    Load all skill packs from the given directory (or PACKS_DIR by default).

    Returns:
        Dict[str, SkillPack]: mapping pack_name -> SkillPack
    """
    base_dir = packs_dir or PACKS_DIR
    if not base_dir.exists():
        raise FileNotFoundError(f"Skill packs directory does not exist: {base_dir}")

    packs: Dict[str, SkillPack] = {}
    for path in sorted(base_dir.glob("*.yaml")):
        pack = load_skill_pack_from_file(path)
        if pack.name in packs:
            raise ValueError(
                f"Duplicate skill pack name '{pack.name}' in '{path}'."
            )
        packs[pack.name] = pack
    return packs


def get_active_skill_names(
    tech_state: str,
    enabled_pack_names: List[str],
    packs: Optional[Dict[str, SkillPack]] = None,
) -> List[str]:
    """
    Determine which skills are available for a given tech_state and set of enabled packs.

    Args:
        tech_state:
            Current tech_state identifier (e.g. "lv", "steam_age", "mv").
        enabled_pack_names:
            Names of packs that the curriculum/runtime has chosen to enable.
            (M11 decides this; M5 just honors it.)
        packs:
            Optional preloaded mapping of pack_name -> SkillPack.
            If not provided, load_all_skill_packs() is used.

    Returns:
        A de-duplicated list of skill names that:
        - belong to at least one enabled pack, and
        - have a pack.requires_tech_state that matches the given tech_state.

    Notes:
        For now, requires_tech_state is treated as an exact match.
        You can later generalize this to ranges (>= lv, etc.) if TechState
        evolves into an ordered lattice instead of strings.
    """
    all_packs = packs or load_all_skill_packs()

    active_skills: List[str] = []

    for pack_name in enabled_pack_names:
        pack = all_packs.get(pack_name)
        if pack is None:
            # Silently skip unknown pack names; upstream should handle validation.
            continue

        if pack.requires_tech_state == tech_state:
            active_skills.extend(pack.skills)

    # Deduplicate while preserving order
    seen = set()
    unique_skills: List[str] = []
    for skill_name in active_skills:
        if skill_name in seen:
            continue
        seen.add(skill_name)
        unique_skills.append(skill_name)

    return unique_skills

