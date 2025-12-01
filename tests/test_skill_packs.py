# tests/test_skill_packs.py

from pathlib import Path

from skills.packs import (
    SkillPack,
    load_skill_pack_from_file,
    load_all_skill_packs,
    get_active_skill_names,
)


def test_load_skill_pack_from_file(tmp_path: Path) -> None:
    yaml_content = """
name: "lv_core"
requires_tech_state: "lv"
tags:
  - "core"
  - "early_progression"
skills:
  - chop_tree
  - feed_coke_ovens
"""
    path = tmp_path / "lv_core.yaml"
    path.write_text(yaml_content, encoding="utf-8")

    pack = load_skill_pack_from_file(path)

    assert isinstance(pack, SkillPack)
    assert pack.name == "lv_core"
    assert pack.requires_tech_state == "lv"
    assert "core" in pack.tags
    assert "chop_tree" in pack.skills
    assert "feed_coke_ovens" in pack.skills


def test_load_all_skill_packs_duplicate_name_raises(tmp_path: Path) -> None:
    yaml_a = """
name: "lv_core"
requires_tech_state: "lv"
skills: ["chop_tree"]
"""
    yaml_b = """
name: "lv_core"
requires_tech_state: "lv"
skills: ["feed_coke_ovens"]
"""

    path_a = tmp_path / "a.yaml"
    path_b = tmp_path / "b.yaml"
    path_a.write_text(yaml_a, encoding="utf-8")
    path_b.write_text(yaml_b, encoding="utf-8")

    try:
        load_all_skill_packs(packs_dir=tmp_path)
    except ValueError as e:
        assert "Duplicate skill pack name 'lv_core'" in str(e)
    else:
        assert False, "Expected ValueError for duplicate skill pack names"


def test_get_active_skill_names_filters_by_tech_state() -> None:
    # Build fake packs in memory instead of hitting the filesystem
    packs = {
        "lv_core": SkillPack(
            name="lv_core",
            requires_tech_state="lv",
            tags=["core"],
            skills=["chop_tree", "feed_coke_ovens"],
        ),
        "steam_age": SkillPack(
            name="steam_age",
            requires_tech_state="steam_age",
            tags=["power"],
            skills=["feed_steam_boiler"],
        ),
    }

    # Tech state = lv, only lv_core enabled
    skills_lv = get_active_skill_names(
        tech_state="lv",
        enabled_pack_names=["lv_core"],
        packs=packs,
    )
    assert "chop_tree" in skills_lv
    assert "feed_coke_ovens" in skills_lv
    assert "feed_steam_boiler" not in skills_lv

    # Tech state = steam_age, both packs "enabled", but lv_core should not match
    skills_steam = get_active_skill_names(
        tech_state="steam_age",
        enabled_pack_names=["lv_core", "steam_age"],
        packs=packs,
    )
    assert "feed_steam_boiler" in skills_steam
    assert "chop_tree" not in skills_steam

