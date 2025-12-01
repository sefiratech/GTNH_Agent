from typing import Dict, Any, List

from skills.registry import register_skill, SkillImplBase
from spec.types import WorldState, Action


@register_skill
class PlantSaplingSkill(SkillImplBase):

    skill_name = "plant_sapling"

    def suggest_actions(self, world: WorldState, params: Dict[str, Any]) -> List[Action]:
        sapling_item = params.get("sapling_item", "minecraft:sapling")
        radius = int(params.get("radius", 8))
        max_saplings = int(params.get("max_saplings", 32))
        avoid_water = bool(params.get("avoid_water", True))

        return [
            Action(
                type="move_to_tagged_area",
                params={"tag": "tree_farm_area"},
            ),
            Action(
                type="plant_saplings_in_radius",
                params={
                    "item": sapling_item,
                    "radius": radius,
                    "max_saplings": max_saplings,
                    "avoid_water": avoid_water,
                },
            ),
        ]

