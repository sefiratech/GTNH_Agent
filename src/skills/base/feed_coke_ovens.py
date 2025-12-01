from typing import Dict, Any, List

from skills.registry import register_skill, SkillImplBase
from spec.types import WorldState, Action


@register_skill
class FeedCokeOvensSkill(SkillImplBase):

    skill_name = "feed_coke_ovens"

    def suggest_actions(self, world: WorldState, params: Dict[str, Any]) -> List[Action]:
        input_item = params.get("input_item", "minecraft:coal")
        max_ovens = int(params.get("max_ovens", 16))
        pull_outputs = bool(params.get("pull_outputs", True))

        actions = [
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

        if pull_outputs:
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

