from dataclasses import dataclass
from typing import Dict, Any, List, Type, Optional, Iterable

from spec.skills import Skill as SkillProtocol, SkillRegistry as SkillRegistryProtocol
from spec.types import WorldState, Action

from .schema import SkillSpec
from . import loader as skills_loader
from . import packs as skills_packs


@dataclass(frozen=True)
class SkillVersionHandle:
    """
    Lightweight handle for a specific skill version.

    This is what M10 / learning code should pass around instead of
    directly mutating SkillSpec objects or YAML.

    Fields:
    - name:   logical skill name (e.g. "chop_tree")
    - version: version identifier (stringified)
    - status:  "active" | "candidate" | "deprecated"
    - origin:  "hand_authored" | "auto_synthesized"
    - spec:    the underlying SkillSpec object
    """
    name: str
    version: str
    status: str
    origin: str
    spec: SkillSpec

    @property
    def is_active(self) -> bool:
        return self.status == "active"

    @property
    def is_candidate(self) -> bool:
        return self.status == "candidate"

    @property
    def is_deprecated(self) -> bool:
        return self.status == "deprecated"


class SkillImplBase(SkillProtocol):
    """
    Base class for skill implementations.

    Subclasses must:
    - set `skill_name` to match a SkillSpec.name
    - implement `suggest_actions(world, params)`
    """

    skill_name: str = ""  # override in subclasses

    def __init__(self, spec: SkillSpec) -> None:
        # store the spec for use in describe()
        self._spec = spec

    @property
    def name(self) -> str:
        """Return the skill name defined by the spec."""
        return self._spec.name

    @property
    def spec(self) -> SkillSpec:
        """Expose the bound SkillSpec (useful for debugging / learning)."""
        return self._spec

    def describe(self) -> Dict[str, Any]:
        """
        Return metadata for planners and critics.

        Includes:
        - name, description
        - version, status, origin
        - parameter specs
        - preconditions
        - effects
        - tags
        - metrics (for M10 / learning)
        """
        meta = self._spec.metadata
        return {
            "name": self._spec.name,
            "description": self._spec.description,
            "version": meta.version,
            "status": meta.status,
            "origin": meta.origin,
            "params": {
                pname: {
                    "type": ps.type,
                    "default": ps.default,
                    "description": ps.description,
                }
                for pname, ps in self._spec.params.items()
            },
            "preconditions": {
                "tech_states_any_of": self._spec.preconditions.tech_states_any_of,
                "required_tools": self._spec.preconditions.required_tools,
                "dimension_allowlist": self._spec.preconditions.dimension_allowlist,
                "semantic_tags_any_of": self._spec.preconditions.semantic_tags_any_of,
                "extra": self._spec.preconditions.extra,
            },
            "effects": {
                "inventory_delta": self._spec.effects.inventory_delta,
                "tech_delta": self._spec.effects.tech_delta,
                "tags": self._spec.effects.tags,
                "extra": self._spec.effects.extra,
            },
            "tags": self._spec.tags,
            "metrics": {
                "success_rate": meta.metrics.success_rate,
                "avg_cost": meta.metrics.avg_cost,
                "avg_risk": meta.metrics.avg_risk,
                "last_used_at": meta.metrics.last_used_at,
            },
        }

    def suggest_actions(
        self,
        world: WorldState,
        params: Dict[str, Any],
    ) -> List[Action]:
        """
        This method must be implemented by subclasses.

        Implementations convert a high-level skill invocation into a sequence
        of low-level Actions suitable for the agent loop / Minecraft IPC layer.
        """
        raise NotImplementedError("SkillImplBase subclasses must override suggest_actions()")


class SkillRegistry(SkillRegistryProtocol):
    """
    Concrete SkillRegistry implementation backed by YAML specs and Python classes.

    Responsibilities:
    - load SkillSpec metadata from config/skills/*.yaml
    - register concrete Skill implementations
    - provide filtered views (all active skills, or by tech_state + SkillPack)
    - own all version-aware read/write operations for skills (Q1.2)
    """

    def __init__(self) -> None:
        # raw loader gives us a flat map, but we normalize to:
        #   name -> (version_str -> SkillSpec)
        raw_specs: Dict[str, SkillSpec] = skills_loader.load_all_skill_specs()

        # All specs (including deprecated), grouped by logical name.
        self._specs_by_name: Dict[str, Dict[str, SkillSpec]] = {}
        for _key, spec in raw_specs.items():
            name = spec.name
            version_key = self._normalize_version(spec.metadata.version)
            versions = self._specs_by_name.setdefault(name, {})
            versions[version_key] = spec

        # name -> *current active* SkillSpec (highest active version)
        # This preserves backwards compatibility for existing call sites.
        self._specs: Dict[str, SkillSpec] = {}
        for name, versions in self._specs_by_name.items():
            active_specs = [
                s for s in versions.values() if s.metadata.status == "active"
            ]
            if active_specs:
                latest_active = max(
                    active_specs,
                    key=lambda s: self._version_sort_key(s.metadata.version),
                )
                self._specs[name] = latest_active
            else:
                # Fallback: pick the highest non-deprecated version if any exist
                non_deprecated = [
                    s for s in versions.values() if s.metadata.status != "deprecated"
                ]
                if non_deprecated:
                    fallback = max(
                        non_deprecated,
                        key=lambda s: self._version_sort_key(s.metadata.version),
                    )
                    self._specs[name] = fallback

        # name -> SkillImplBase (Python implementation, always bound to "current" active spec)
        self._skills: Dict[str, SkillImplBase] = {}

    # --- Internal helpers ---

    @staticmethod
    def _normalize_version(version: Any) -> str:
        """Normalize version field to a string key."""
        return str(version)

    @staticmethod
    def _version_sort_key(version: Any) -> Any:
        """
        Sort key for version values.

        Tries to treat purely integer versions numerically,
        otherwise falls back to string comparison.
        """
        if isinstance(version, int):
            return (0, version)
        try:
            as_int = int(str(version))
            return (0, as_int)
        except (TypeError, ValueError):
            return (1, str(version))

    def _get_versions_for_name(self, skill_name: str) -> Dict[str, SkillSpec]:
        try:
            return self._specs_by_name[skill_name]
        except KeyError:
            raise KeyError(f"No SkillSpecs found for skill '{skill_name}'")

    def _build_handle(self, spec: SkillSpec) -> SkillVersionHandle:
        meta = spec.metadata
        return SkillVersionHandle(
            name=spec.name,
            version=self._normalize_version(meta.version),
            status=meta.status,
            origin=meta.origin,
            spec=spec,
        )

    def _recompute_active_for(self, skill_name: str) -> None:
        """
        Recompute the active spec for a given skill name
        and update the compatibility map `self._specs`.
        """
        versions = self._get_versions_for_name(skill_name)
        active_specs = [
            s for s in versions.values() if s.metadata.status == "active"
        ]
        if active_specs:
            latest_active = max(
                active_specs,
                key=lambda s: self._version_sort_key(s.metadata.version),
            )
            self._specs[skill_name] = latest_active
        else:
            # If nothing is active, try to pick a non-deprecated fallback.
            non_deprecated = [
                s for s in versions.values() if s.metadata.status != "deprecated"
            ]
            if non_deprecated:
                fallback = max(
                    non_deprecated,
                    key=lambda s: self._version_sort_key(s.metadata.version),
                )
                self._specs[skill_name] = fallback
            elif skill_name in self._specs:
                # No viable versions remain.
                del self._specs[skill_name]

    # --- Registration ---

    def register(self, skill: SkillImplBase) -> None:
        """
        Register a single skill implementation instance.

        Enforces:
        - a spec must exist for the skill name
        - no duplicate registrations
        - cannot register explicitly deprecated skills

        Contract detail for tests:
        - Attempting to register a skill whose only spec(s) are deprecated
          MUST raise ValueError, not KeyError.
        """
        name = skill.name

        # Check if we know this skill at all (including deprecated versions).
        versions = self._specs_by_name.get(name, {})

        if name not in self._specs:
            # Known skill name, but no active spec.
            if versions:
                # If any known version is explicitly deprecated, treat this
                # as "known but forbidden" and raise ValueError.
                if any(s.metadata.status == "deprecated" for s in versions.values()):
                    raise ValueError(f"Cannot register deprecated skill: {name}")
            # Otherwise it's just unknown / misconfigured.
            raise KeyError(f"No active SkillSpec found for skill: {name}")

        spec = self._specs[name]

        # Extra paranoia: even if it's in _specs, don't allow deprecated.
        if spec.metadata.status == "deprecated":
            raise ValueError(f"Cannot register deprecated skill: {name}")

        # Duplicate registration: configuration error, not "missing key".
        if name in self._skills:
            raise ValueError(f"Skill already registered: {name}")

        # Also respect an explicitly deprecated spec bound to the impl, if present.
        impl_spec = getattr(skill, "spec", None)
        if impl_spec is not None and getattr(impl_spec.metadata, "status", None) == "deprecated":
            raise ValueError(f"Cannot register deprecated skill: {name}")

        self._skills[name] = skill

    # --- Query methods ---

    def list_skills(self) -> List[str]:
        """
        Return names of registered skills that are not deprecated.

        This list ignores skills without implementations and those whose
        current active SkillSpec.metadata.status is 'deprecated'.
        """
        result: List[str] = []
        for name in self._skills.keys():
            spec = self._specs.get(name)
            if spec is None:
                continue
            if spec.metadata.status == "deprecated":
                continue
            result.append(name)
        return result

    def get_skill(self, name: str) -> SkillProtocol:
        """Return the skill implementation by name."""
        if name not in self._skills:
            raise KeyError(name)
        return self._skills[name]

    def describe_all(self) -> Dict[str, Dict[str, Any]]:
        """
        Return metadata for all non-deprecated, registered skills.

        This is the generic view used when you don't care about tech_state
        or Skill Packs (e.g. debugging, tooling).
        """
        descs: Dict[str, Dict[str, Any]] = {}
        for name in self.list_skills():
            skill = self._skills[name]
            descs[name] = skill.describe()
        return descs

    def describe_for_tech_state(
        self,
        tech_state: str,
        enabled_pack_names: Iterable[str],
    ) -> Dict[str, Dict[str, Any]]:
        """
        Return metadata for skills that are both:

        - registered and non-deprecated, and
        - included in at least one active Skill Pack for this tech_state.

        This is the primary view used by Planner / Critic / Curriculum systems.
        """
        packs = skills_packs.load_all_skill_packs()
        allowed_names = set(
            skills_packs.get_active_skill_names(
                tech_state=tech_state,
                enabled_pack_names=list(enabled_pack_names),
                packs=packs,
            )
        )

        descs: Dict[str, Dict[str, Any]] = {}
        for name in self.list_skills():
            if name not in allowed_names:
                continue
            skill = self._skills[name]
            descs[name] = skill.describe()
        return descs

    # --- Evolution / versioning API for M10 (Q1.2) ---

    def get_latest_skill(self, skill_name: str) -> SkillVersionHandle:
        """
        Return the latest *active* version of a skill.

        Rules:
        - Only considers specs with status == "active"
        - Among them, picks the highest version (according to _version_sort_key)
        """
        versions = self._get_versions_for_name(skill_name)
        active_specs = [
            s for s in versions.values() if s.metadata.status == "active"
        ]
        if not active_specs:
            raise KeyError(
                f"No active versions found for skill '{skill_name}'"
            )
        latest = max(
            active_specs,
            key=lambda s: self._version_sort_key(s.metadata.version),
        )
        return self._build_handle(latest)

    def list_skill_versions(self, skill_name: str) -> List[SkillVersionHandle]:
        """
        List all known versions of a skill, sorted by version ascending.

        Includes active, candidate, and deprecated versions.
        """
        versions = self._get_versions_for_name(skill_name)
        specs_sorted = sorted(
            versions.values(),
            key=lambda s: self._version_sort_key(s.metadata.version),
        )
        return [self._build_handle(spec) for spec in specs_sorted]

    def register_skill_candidate(self, candidate_spec: SkillSpec) -> SkillVersionHandle:
        """
        Register a new candidate skill version (from M10 / learning).

        This does NOT touch Python implementations; it only mutates the
        registry's in-memory SkillSpec tables.

        Enforced invariants:
        - status is forced to "candidate"
        - origin is forced to "auto_synthesized"
        """
        name = candidate_spec.name
        versions = self._specs_by_name.setdefault(name, {})

        # Normalize metadata first
        candidate_spec.metadata.status = "candidate"
        candidate_spec.metadata.origin = "auto_synthesized"

        version_key = self._normalize_version(candidate_spec.metadata.version)
        versions[version_key] = candidate_spec

        # Do NOT update self._specs[name] yet; that happens on promotion.
        return self._build_handle(candidate_spec)

    def mark_skill_version_deprecated(self, skill_version: "SkillVersionHandle") -> SkillVersionHandle:
        """
        Mark a specific skill version as deprecated.

        The caller must pass a SkillVersionHandle obtained from this registry.
        """
        name = skill_version.name
        versions = self._get_versions_for_name(name)

        version_key = self._normalize_version(skill_version.version)
        if version_key not in versions:
            raise KeyError(
                f"Version '{skill_version.version}' not found for skill '{name}'"
            )

        spec = versions[version_key]
        spec.metadata.status = "deprecated"

        # If this was the active version, recompute active.
        self._recompute_active_for(name)

        return self._build_handle(spec)

    def promote_skill_candidate(self, skill_version: "SkillVersionHandle") -> SkillVersionHandle:
        """
        Promote a candidate version to active.

        Behavior:
        - requires current status == "candidate"
        - sets that version's status to "active"
        - any previously active versions for this skill are flipped to "deprecated"
        - updates the compatibility map `self._specs[name]`
        """
        name = skill_version.name
        versions = self._get_versions_for_name(name)

        version_key = self._normalize_version(skill_version.version)
        if version_key not in versions:
            raise KeyError(
                f"Version '{skill_version.version}' not found for skill '{name}'"
            )

        spec = versions[version_key]
        if spec.metadata.status != "candidate":
            raise ValueError(
                f"Cannot promote non-candidate version '{skill_version.version}' "
                f"for skill '{name}' (status={spec.metadata.status!r})"
            )

        # Demote any existing active versions.
        for other_spec in versions.values():
            if other_spec is spec:
                continue
            if other_spec.metadata.status == "active":
                other_spec.metadata.status = "deprecated"

        # Promote this one.
        spec.metadata.status = "active"

        # Recompute active mapping for backwards-compat callers.
        self._recompute_active_for(name)

        return self._build_handle(spec)

    # --- Legacy evolution hook (still useful for direct spec replacement) ---

    def update_spec(self, spec: SkillSpec) -> None:
        """
        Replace or insert a SkillSpec.

        Intended for use by M10 (learning/evolution):

        - updating metadata.status (candidate -> active / deprecated)
        - updating metadata.metrics (success_rate, avg_cost, avg_risk, last_used_at)
        - updating descriptions / tags over time
        """
        name = spec.name
        version_key = self._normalize_version(spec.metadata.version)

        versions = self._specs_by_name.setdefault(name, {})
        versions[version_key] = spec

        # If this spec is active, or if there is no better active, recompute.
        self._recompute_active_for(name)


# Global registry instance (simple singleton pattern).
_GLOBAL_REGISTRY: Optional[SkillRegistry] = None


def get_global_skill_registry() -> SkillRegistry:
    """
    Lazy-load and return the global SkillRegistry instance.

    This is a convenience for modules that don't use explicit dependency injection.
    """
    global _GLOBAL_REGISTRY
    if _GLOBAL_REGISTRY is None:
        _GLOBAL_REGISTRY = SkillRegistry()
    return _GLOBAL_REGISTRY


def register_skill(cls: Type[SkillImplBase]) -> Type[SkillImplBase]:
    """
    Class decorator to register a skill implementation with the global registry.

    Usage:

        @register_skill
        class ChopTreeSkill(SkillImplBase):
            skill_name = "chop_tree"
            ...

    When the module is imported, this will:
    - look up the corresponding *active* SkillSpec by skill_name
    - instantiate the skill class with that spec
    - register it in the global registry
    """
    registry = get_global_skill_registry()

    skill_name = getattr(cls, "skill_name", "")
    if not skill_name:
        raise ValueError(f"Skill class {cls.__name__} must define skill_name")

    # Bind implementation to the latest active version of the spec.
    handle = registry.get_latest_skill(skill_name)
    instance = cls(handle.spec)
    registry.register(instance)

    return cls

