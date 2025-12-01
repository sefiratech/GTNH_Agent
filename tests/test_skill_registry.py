# tests/test_skill_registry.py

from typing import Dict, Any, List

import pytest

from skills import registry as skills_registry
from skills.registry import SkillRegistry, SkillImplBase, register_skill
from skills.schema import (
    SkillSpec,
    SkillPreconditions,
    SkillEffects,
    SkillMetrics,
    ParamSpec,
)
from skills.packs import SkillPack


# --- Helpers to build dummy SkillSpec ---

def make_dummy_spec(
    name: str = "dummy_skill",
    status: str = "active",
) -> SkillSpec:
    pre = SkillPreconditions(
        tech_states_any_of=["stone_age"],
        required_tools=[],
        dimension_allowlist=["overworld"],
        semantic_tags_any_of=[],
        extra={},
    )
    eff = SkillEffects(
        inventory_delta={},
        tech_delta={},
        tags=[],
        extra={},
    )
    metrics = SkillMetrics(
        success_rate=None,
        avg_cost=None,
    )
    params = {
        "foo": ParamSpec(
            name="foo",
            type="int",
            default=1,
            description="dummy param",
        )
    }
    return SkillSpec(
        name=name,
        version=1,
        status=status,
        origin="manual",
        description="dummy",
        params=params,
        preconditions=pre,
        effects=eff,
        tags=[],
        metrics=metrics,
    )


# --- Basic registry behavior ---


def test_skill_registry_registration(monkeypatch) -> None:
    # Monkeypatch load_all_skill_specs to only return our dummy spec
    from skills import loader as skills_loader

    def fake_load_all_skill_specs():
        return {"dummy_skill": make_dummy_spec()}

    monkeypatch.setattr(skills_loader, "load_all_skill_specs", fake_load_all_skill_specs)

    reg = SkillRegistry()

    class DummySkill(SkillImplBase):
        skill_name = "dummy_skill"

        def suggest_actions(self, world, params: Dict[str, Any]) -> List[Any]:
            return []

    # Should register cleanly
    reg.register(DummySkill(make_dummy_spec()))
    assert "dummy_skill" in reg.list_skills()

    descs = reg.describe_all()
    assert "dummy_skill" in descs
    assert descs["dummy_skill"]["description"] == "dummy"
    assert descs["dummy_skill"]["status"] == "active"


def test_skill_registry_rejects_deprecated(monkeypatch) -> None:
    from skills import loader as skills_loader

    def fake_load_all_skill_specs():
        return {"old_skill": make_dummy_spec(name="old_skill", status="deprecated")}

    monkeypatch.setattr(skills_loader, "load_all_skill_specs", fake_load_all_skill_specs)

    reg = SkillRegistry()

    class OldSkill(SkillImplBase):
        skill_name = "old_skill"

        def suggest_actions(self, world, params: Dict[str, Any]) -> List[Any]:
            return []

    with pytest.raises(ValueError):
        reg.register(OldSkill(make_dummy_spec(name="old_skill", status="deprecated")))


# --- Pack-gated describe_for_tech_state ---


def test_describe_for_tech_state_uses_skill_packs(monkeypatch) -> None:
    # Specs
    from skills import loader as skills_loader

    def fake_load_all_skill_specs():
        return {"dummy_skill": make_dummy_spec(name="dummy_skill", status="active")}

    monkeypatch.setattr(skills_loader, "load_all_skill_specs", fake_load_all_skill_specs)

    # Packs
    from skills import packs as skills_packs

    def fake_load_all_skill_packs():
        return {
            "lv_core": SkillPack(
                name="lv_core",
                requires_tech_state="lv",
                tags=["core"],
                skills=["dummy_skill"],
            ),
            "steam_age": SkillPack(
                name="steam_age",
                requires_tech_state="steam_age",
                tags=["power"],
                skills=["other_skill"],
            ),
        }

    monkeypatch.setattr(
        skills_packs, "load_all_skill_packs", fake_load_all_skill_packs
    )

    # Registry + skill impl
    reg = SkillRegistry()

    class DummySkill(SkillImplBase):
        skill_name = "dummy_skill"

        def suggest_actions(self, world, params: Dict[str, Any]) -> List[Any]:
            return []

    reg.register(DummySkill(make_dummy_spec(name="dummy_skill", status="active")))

    # Tech state lv, pack lv_core enabled
    descs_lv = reg.describe_for_tech_state(
        tech_state="lv",
        enabled_pack_names=["lv_core"],
    )
    assert "dummy_skill" in descs_lv

    # Tech state steam_age, lv_core still "enabled" but should not match on tech_state
    descs_steam = reg.describe_for_tech_state(
        tech_state="steam_age",
        enabled_pack_names=["lv_core"],
    )
    assert "dummy_skill" not in descs_steam


# --- Decorator integration ---


def test_register_skill_decorator(monkeypatch) -> None:
    # Fake specs so registry knows about 'decorator_skill'
    from skills import loader as skills_loader

    def fake_load_all_skill_specs():
        return {"decorator_skill": make_dummy_spec(name="decorator_skill")}

    monkeypatch.setattr(skills_loader, "load_all_skill_specs", fake_load_all_skill_specs)

    # Reset global registry to a fresh instance
    skills_registry._GLOBAL_REGISTRY = None

    # When this class is defined, @register_skill should:
    # - look up the spec
    # - construct an instance
    # - register it in the global registry
    @register_skill
    class DecoratorSkill(SkillImplBase):
        skill_name = "decorator_skill"

        def suggest_actions(self, world, params: Dict[str, Any]) -> List[Any]:
            return []

    reg = skills_registry.get_global_skill_registry()
    assert "decorator_skill" in reg.list_skills()

    descs = reg.describe_all()
    assert "decorator_skill" in descs
    assert descs["decorator_skill"]["name"] == "decorator_skill"

