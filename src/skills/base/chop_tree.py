# src/skills/base/chop_tree.py

from typing import Dict, Any, List

from skills.registry import register_skill, SkillImplBase   # ← IMPORTANT
from spec.types import WorldState, Action


@register_skill
class ChopTreeSkill(SkillImplBase):
    """
    Implementation of the 'chop_tree' skill.

    Must inherit from SkillImplBase to get:
    - .describe()
    - .name
    - spec storage
    """

    skill_name = "chop_tree"

    def suggest_actions(self, world: WorldState, params: Dict[str, Any]) -> List[Action]:
        radius = int(params.get("radius", 8))

        return [
            Action(
                type="move_to_tagged_area",
                params={"tag": "tree_area"},
            ),
            Action(
                type="break_blocks_in_radius",
                params={
                    "block_tag": "log",
                    "radius": radius,
                    "max_blocks": 128,
                },
            ),
        ]

