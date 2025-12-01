# src/skills/base/__init__.py

"""
Base skill implementations for GTNH_Agent.

Importing this module ensures that all @register_skill decorators run and
that the SkillRegistry sees the implementations.
"""

from .chop_tree import ChopTreeSkill  # noqa: F401
from .plant_sapling import PlantSaplingSkill  # noqa: F401
from .feed_coke_ovens import FeedCokeOvensSkill  # noqa: F401
from .basic_crafting import BasicCraftingSkill  # noqa: F401
from .feed_steam_boiler import FeedSteamBoilerSkill  # noqa: F401
from .maintain_coke_ovens import MaintainCokeOvensSkill  # noqa: F401
from .refill_water_tanks import RefillWaterTanksSkill  # noqa: F401

