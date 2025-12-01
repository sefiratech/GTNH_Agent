# src/integration/validators/skill_integrity.py

from dataclasses import dataclass
from typing import Dict, List, Set

from skills.registry import get_global_skill_registry
from skills.packs import load_all_skill_packs, SkillPack


@dataclass
class SkillPackIntegrityResult:
    """
    Result of validating Skill Packs against the Skill Registry.

    - missing_specs:
        skills referenced in packs that do NOT have a SkillSpec
    - missing_impls:
        skills referenced in packs that do have a spec, but are NOT registered
        as implementations in the SkillRegistry (no Python class registered)
    - unused_specs:
        skills that have a SkillSpec but are NOT referenced in any pack
        (not necessarily an error, but useful to see)
    """
    missing_specs: Dict[str, List[str]]     # pack_name -> [skill_name, ...]
    missing_impls: Dict[str, List[str]]     # pack_name -> [skill_name, ...]
    unused_specs: List[str]                 # spec names not referenced by any pack


def validate_skill_packs_against_registry() -> SkillPackIntegrityResult:
    """
    Validate that all Skill Packs reference existing SkillSpecs and implementations.

    Checks:
    - Every skill listed in pack.skills has a SkillSpec in the registry
    - Every skill listed in pack.skills has a registered implementation
      (i.e. appears in registry.list_skills())
    - Compute which SkillSpecs are never referenced by any pack

    This function DOES NOT raise; it returns a result object so tests or
    CLI wrappers can decide how strict to be.
    """
    registry = get_global_skill_registry()
    packs: Dict[str, SkillPack] = load_all_skill_packs()

    # Specs known to the registry
    spec_metadata = registry.describe_all()  # only active + registered
    # But we also want specs that might be inactive / candidate
    # by peeking into the internal _specs map if available.
    all_spec_names: Set[str] = set(getattr(registry, "_specs", {}).keys())
    if not all_spec_names:
        # Fallback: just use describe_all keys if _specs isn't accessible
        all_spec_names = set(spec_metadata.keys())

    # Implementations known to the registry
    impl_names: Set[str] = set(registry.list_skills())

    missing_specs: Dict[str, List[str]] = {}
    missing_impls: Dict[str, List[str]] = {}
    referenced_specs: Set[str] = set()

    for pack_name, pack in packs.items():
        for skill_name in pack.skills:
            referenced_specs.add(skill_name)

            # Spec missing?
            if skill_name not in all_spec_names:
                missing_specs.setdefault(pack_name, []).append(skill_name)
                # If spec is missing, impl check is meaningless for this skill
                continue

            # Implementation missing?
            if skill_name not in impl_names:
                missing_impls.setdefault(pack_name, []).append(skill_name)

    # Specs that are never referenced by any pack
    unused_specs = sorted(all_spec_names - referenced_specs)

    return SkillPackIntegrityResult(
        missing_specs=missing_specs,
        missing_impls=missing_impls,
        unused_specs=unused_specs,
    )


def assert_skill_packs_integrity(strict_unused: bool = False) -> None:
    """
    Convenience function that RAISES if integrity checks fail.

    - Always raises if any missing_specs or missing_impls are found.
    - Optionally raises if there are unused_specs and strict_unused=True.

    Intended for use in:
    - a CLI validation tool
    - pytest integration tests
    """
    result = validate_skill_packs_against_registry()

    errors: List[str] = []

    if result.missing_specs:
        for pack_name, skills in result.missing_specs.items():
            errors.append(
                f"SkillPack '{pack_name}' references skills with no SkillSpec: {skills}"
            )

    if result.missing_impls:
        for pack_name, skills in result.missing_impls.items():
            errors.append(
                f"SkillPack '{pack_name}' references skills with no implementation: {skills}"
            )

    if strict_unused and result.unused_specs:
        errors.append(
            f"Unused SkillSpecs (not referenced by any pack): {result.unused_specs}"
        )

    if errors:
        joined = "\n".join(errors)
        raise AssertionError(f"SkillPack integrity validation failed:\n{joined}")


if __name__ == "__main__":
    # Simple CLI entrypoint:
    # python -m integration.validators.skill_integrity
    result = validate_skill_packs_against_registry()
    if not result.missing_specs and not result.missing_impls:
        print("SkillPack integrity OK.")
        if result.unused_specs:
            print(f"Note: unused SkillSpecs: {result.unused_specs}")
    else:
        print("SkillPack integrity problems detected:")
        if result.missing_specs:
            print("  Missing specs:")
            for pack_name, skills in result.missing_specs.items():
                print(f"    {pack_name}: {skills}")
        if result.missing_impls:
            print("  Missing implementations:")
            for pack_name, skills in result.missing_impls.items():
                print(f"    {pack_name}: {skills}")
        raise SystemExit(1)
