# src/skills/base/feed_steam_boiler.py

"""
FeedSteamBoilerSkill

Maintain a steam boiler by feeding fuel and water to keep pressure up.
"""

from typing import Dict, Any, List

from skills.registry import register_skill, SkillImplBase
from spec.types import WorldState, Action


@register_skill
class FeedSteamBoilerSkill(SkillImplBase):
    """
    Implementation of the 'feed_steam_boiler' skill.

    Metadata comes from config/skills/feed_steam_boiler.yaml.
    """

    skill_name = "feed_steam_boiler"

    def suggest_actions(self, world: WorldState, params: Dict[str, Any]) -> List[Action]:
        fuel_item = params.get("fuel_item", "minecraft:coal")
        water_source = params.get("water_source", "adjacent_tank")
        target_pressure = float(params.get("target_pressure", 0.8))

        return [
            Action(
                type="move_to_tagged_area",
                params={"tag": "steam_boiler"},
            ),
            Action(
                type="feed_boiler_fuel",
                params={
                    "fuel_item": fuel_item,
                    "target_pressure": target_pressure,
                },
            ),
            Action(
                type="feed_boiler_water",
                params={
                    "water_source": water_source,
                    "target_pressure": target_pressure,
                },
            ),
        ]

