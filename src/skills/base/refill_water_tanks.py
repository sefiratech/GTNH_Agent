# src/skills/base/refill_water_tanks.py

"""
RefillWaterTanksSkill

Keeps a set of water tanks topped up for boilers and other machines.
"""

from typing import Dict, Any, List

from skills.registry import register_skill, SkillImplBase
from spec.types import WorldState, Action


@register_skill
class RefillWaterTanksSkill(SkillImplBase):
    """
    Implementation of the 'refill_water_tanks' skill.

    Metadata comes from config/skills/refill_water_tanks.yaml.
    """

    skill_name = "refill_water_tanks"

    def suggest_actions(self, world: WorldState, params: Dict[str, Any]) -> List[Action]:
        max_tanks = int(params.get("max_tanks", 4))
        water_source = params.get("water_source", "infinite_water")
        min_fill_fraction = float(params.get("min_fill_fraction", 0.5))

        return [
            Action(
                type="move_to_tagged_area",
                params={"tag": "water_tank_cluster"},
            ),
            Action(
                type="refill_tanks_from_source",
                params={
                    "max_tanks": max_tanks,
                    "water_source": water_source,
                    "min_fill_fraction": min_fill_fraction,
                },
            ),
        ]

