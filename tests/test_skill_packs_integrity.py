# tests/test_skill_packs_integrity.py

from integration.validators.skill_integrity import assert_skill_packs_integrity


def test_skill_packs_reference_existing_skills() -> None:
    """
    Integration check for M5:

    - Every skill referenced in any SkillPack must have:
      - a SkillSpec defined, and
      - a registered implementation in the SkillRegistry.

    This test will fail if:
    - a pack references a skill with no YAML spec
    - a pack references a skill that has a spec but no Python implementation
    """
    # This will raise AssertionError if something is wrong.
    assert_skill_packs_integrity(strict_unused=False)
