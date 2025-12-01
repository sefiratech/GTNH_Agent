# src/skills/base/maintain_coke_ovens.py

"""
MaintainCokeOvensSkill

Higher-level maintenance loop over a coke-oven bank:
- keep them fed
- clear outputs regularly
"""

from typing import Dict, Any, List

from skills.registry import register_skill, SkillImplBase
from spec.types import WorldState, Action


@register_skill
class MaintainCokeOvensSkill(SkillImplBase):
    """
    Implementation of the 'maintain_coke_ovens' skill.

    Metadata comes from config/skills/maintain_coke_ovens.yaml.
    """

    skill_name = "maintain_coke_ovens"

    def suggest_actions(self, world: WorldState, params: Dict[str, Any]) -> List[Action]:
        input_item = params.get("input_item", "minecraft:coal")
        max_ovens = int(params.get("max_ovens", 16))
        collect_outputs = bool(params.get("collect_outputs", True))

        actions: List[Action] = [
            Action(
                type="move_to_tagged_area",
                params={"tag": "coke_oven_bank"},
            ),
            Action(
                type="feed_multiblock_group",
                params={
                    "group_tag": "coke_oven",
                    "input_item": input_item,
                    "max_structures": max_ovens,
                },
            ),
        ]

        if collect_outputs:
            actions.append(
                Action(
                    type="collect_outputs_from_group",
                    params={
                        "group_tag": "coke_oven",
                        "items": [
                            "minecraft:coal_coke",
                            "gtnh:creosote_bucket",
                        ],
                    },
                )
            )

        return actions

