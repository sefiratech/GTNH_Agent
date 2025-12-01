# src/skills/base/basic_crafting.py

"""
BasicCraftingSkill

Generic crafting helper for early game:
- chooses a recipe by ID
- emits a crafting action with bounded repetitions
"""

from typing import Dict, Any, List

from skills.registry import register_skill, SkillImplBase
from spec.types import WorldState, Action


@register_skill
class BasicCraftingSkill(SkillImplBase):
    """
    Implementation of the 'basic_crafting' skill.

    Metadata comes from config/skills/basic_crafting.yaml.
    """

    skill_name = "basic_crafting"

    def suggest_actions(self, world: WorldState, params: Dict[str, Any]) -> List[Action]:
        recipe_id = params.get("recipe_id", "unknown_recipe")
        max_crafts = int(params.get("max_crafts", 4))
        use_crafting_table = bool(params.get("use_crafting_table", True))

        actions: List[Action] = []

        if use_crafting_table:
            actions.append(
                Action(
                    type="move_to_tagged_area",
                    params={"tag": "crafting_area"},
                )
            )

        actions.append(
            Action(
                type="craft_recipe",
                params={
                    "recipe_id": recipe_id,
                    "max_crafts": max_crafts,
                    "use_crafting_table": use_crafting_table,
                },
            )
        )

        return actions

