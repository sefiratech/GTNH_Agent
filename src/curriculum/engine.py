# path: src/curriculum/engine.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Dict, Any

from semantics.schema import TechState
from spec.agent_loop import AgentGoal

from .schema import (
    CurriculumConfig,
    PhaseConfig,
    PhaseGoal,
    PhaseTechTargets,
    PhaseSkillFocus,
    LongHorizonProject,
)


@dataclass
class ActivePhaseView:
    """
    Runtime view of the active curriculum phase.

    Contains:
      - the PhaseConfig
      - list of human-facing goal descriptions
      - phase-level completion flag
      - virtue_overrides and skill_focus for this phase
    """
    phase: PhaseConfig
    active_goals: List[str]
    is_complete: bool
    virtue_overrides: Dict[str, float]
    # Exposed as a plain dict so Phase 4 orchestrator can .get("must_have")
    skill_focus: Dict[str, List[str]]


@dataclass
class ActiveCurriculumView:
    """
    Runtime view of the whole curriculum in the current context.

    Fields expected by tests + orchestrator:

      - curriculum_id:
            Top-level curriculum id (from CurriculumConfig.id).

      - phase_view:
            ActivePhaseView describing the currently relevant phase
            given the TechState.

      - is_complete:
            True if *all* phases have their completion_conditions
            satisfied in the current context.

      - unlocked_projects:
            List of LongHorizonProject objects whose at least one stage
            has all depends_on_phases satisfied by the set of completed
            phases (see tests in test_curriculum_projects.py).
    """
    curriculum_id: str
    phase_view: ActivePhaseView
    is_complete: bool
    unlocked_projects: List[LongHorizonProject]


class CurriculumEngine:
    """
    Q1.5 – Curriculum-driven goal selection engine.

    Responsibilities:
      - Scan phases in order.
      - Check each phase's tech_targets against the current TechState.
      - Within the matching phase, select the next goal that is:
          * entry-condition satisfied
          * exit-condition NOT satisfied
      - Expose an ActiveCurriculumView / ActivePhaseView for other modules.

    This becomes the primary source of AgentGoal for the planner and
    skill-learning orchestrator.
    """

    def __init__(self, config: CurriculumConfig) -> None:
        self._config = config

    # ----------------------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------------------

    def _phase_matches_tech(self, tech_targets: PhaseTechTargets, tech: TechState) -> bool:
        """
        Minimal phase-gating rule:

        - required_active must equal TechState.active (if set)
        - required_unlocked must be a subset of TechState.unlocked
        """
        if tech_targets.required_active and tech_targets.required_active != tech.active:
            return False

        if tech_targets.required_unlocked:
            if not set(tech_targets.required_unlocked).issubset(set(tech.unlocked)):
                return False

        return True

    def _check_entry_conditions(self, phase: PhaseConfig, tech: TechState, world: Any) -> bool:
        """
        Placeholder for future per-goal entry conditions.

        For now, phases have only tech_targets; individual PhaseGoal has no
        extra entry/exit conditions beyond "phase is active".
        """
        # If we ever extend PhaseGoal with entry conditions, they get checked here.
        return True

    def _check_exit_conditions(self, phase: PhaseConfig, tech: TechState, world: Any) -> bool:
        """
        Phase-level completion conditions.

        Mirrors PhaseCompletionConditions from schema, but we keep the logic
        intentionally loose so the YAML can evolve.

        IMPORTANT (for goal selection / phase_view):
        - A phase with *no* completion conditions is treated as *incomplete*.
        - Only when at least one condition is specified AND all specified
          conditions are satisfied do we treat the phase as complete.
        """
        cond = phase.completion_conditions

        has_any_condition = bool(cond.tech_unlocked) or bool(cond.machines_present)
        if not has_any_condition:
            # No explicit completion conditions => phase not complete yet
            # from the perspective of next_goal / phase_view.
            return False

        # tech_unlocked
        if cond.tech_unlocked:
            if not set(cond.tech_unlocked).issubset(set(tech.unlocked)):
                return False

        # machines_present
        ctx = getattr(world, "context", {}) or {}
        machines = ctx.get("machines", [])
        for req in cond.machines_present:
            mtype = req.get("type")
            min_count = int(req.get("min_count", 1))
            if not mtype:
                continue
            actual = sum(
                1 for m in machines
                if isinstance(m, dict) and m.get("type") == mtype
            )
            if actual < min_count:
                return False

        return True

    def _compute_completed_phases(self, tech_state: TechState, world: Any) -> List[str]:
        """
        Return the list of phase IDs whose completion_conditions are satisfied.

        Semantics for *projects* (see test_curriculum_projects.py):

        - A phase is considered "complete" if its completion_conditions are
          satisfied.
        - A phase with *no* completion conditions is treated as *trivially
          complete* for project-unlock logic.

        This intentionally differs from _check_exit_conditions, which is more
        conservative for goal-selection.
        """
        completed: List[str] = []
        for phase in self._config.phases:
            cond = phase.completion_conditions
            has_any_condition = bool(cond.tech_unlocked) or bool(cond.machines_present)

            if not has_any_condition:
                # For project semantics, empty conditions are vacuously satisfied.
                completed.append(phase.id)
                continue

            if self._check_exit_conditions(phase, tech_state, world):
                completed.append(phase.id)

        return completed

    def _compute_unlocked_projects(
        self,
        tech_state: TechState,
        world: Any,
    ) -> List[LongHorizonProject]:
        """
        Determine which long-horizon projects are unlocked.

        Semantics from tests:

          * A phase is "complete" if its completion_conditions are satisfied
            (or empty, which we treat as trivially satisfied here).
          * A project unlocks if ANY of its stages has all depends_on_phases
            contained in the completed phase set.
        """
        completed_phases = set(self._compute_completed_phases(tech_state, world))
        unlocked: List[LongHorizonProject] = []

        for project in self._config.long_horizon_projects:
            for stage in project.stages:
                if all(pid in completed_phases for pid in stage.depends_on_phases):
                    unlocked.append(project)
                    break  # one unlocked stage is enough

        return unlocked

    # ----------------------------------------------------------------------
    # NEXT GOAL SELECTION (Q1.5)
    # ----------------------------------------------------------------------

    def next_goal(
        self,
        *,
        tech_state: TechState,
        world: Any,
        experience_summary: Optional[Dict[str, Any]] = None,
    ) -> Optional[AgentGoal]:
        """
        Return the next incomplete goal from the first phase whose tech_targets
        match the current TechState.

        Selection rule:
          - Find the first phase where _phase_matches_tech(...) is True.
          - If that phase is already complete (completion_conditions satisfied),
            skip it and continue.
          - Within that phase, pick the first goal whose entry_conditions
            (currently trivial) are satisfied.

        Returns None if no suitable goal exists.
        """
        for phase in self._config.phases:
            if not self._phase_matches_tech(phase.tech_targets, tech_state):
                continue

            # Phase already "complete"? Skip it.
            if self._check_exit_conditions(phase, tech_state, world):
                continue

            # In this simple schema, goals themselves are not individually
            # completed; phase-level completion covers them. So we just pick
            # the first goal where entry conditions are okay.
            for goal in phase.goals:
                if not self._check_entry_conditions(phase, tech_state, world):
                    continue

                # This is the next goal.
                return AgentGoal(
                    id=f"{phase.id}:{goal.id}",
                    text=goal.description,
                    phase=phase.id,
                    source="curriculum",
                )

        return None  # no matching phase / goal

    # ----------------------------------------------------------------------
    # PHASE / CURRICULUM VIEW (for planners / dashboards / orchestrators)
    # ----------------------------------------------------------------------

    def view(
        self,
        tech_state: TechState,
        world: Any,
    ) -> ActiveCurriculumView:
        """
        Produce a curriculum snapshot for the current context.

        Tests expect:

          - engine.view(tech_state, world) to accept positional args.
          - Returned object to have:
              * .curriculum_id
              * .phase_view.phase.id
              * .phase_view.is_complete
              * .phase_view.active_goals (list of descriptions)
              * .phase_view.skill_focus as a dict:
                    { "must_have": [...], "preferred": [...] }
              * .unlocked_projects (list[LongHorizonProject])
        """
        # 1) Pick active phase via tech_targets
        matching_phase: Optional[PhaseConfig] = None
        for phase in self._config.phases:
            if self._phase_matches_tech(phase.tech_targets, tech_state):
                matching_phase = phase
                break

        if matching_phase is None:
            # No matching phase; pick the first one as a fallback snapshot
            if not self._config.phases:
                raise ValueError("CurriculumConfig has no phases.")
            matching_phase = self._config.phases[0]

        # 2) Phase-level completion in this context (conservative semantics)
        phase_is_complete = self._check_exit_conditions(matching_phase, tech_state, world)

        # 3) Active goals = all goal descriptions for this phase.
        active_goals: List[str] = [
            goal.description for goal in matching_phase.goals
        ]

        # 3b) Normalize PhaseSkillFocus → plain dict so callers can .get(...)
        sf: PhaseSkillFocus = matching_phase.skill_focus
        skill_focus_dict: Dict[str, List[str]] = {
            "must_have": list(sf.must_have),
            "preferred": list(sf.preferred),
        }

        phase_view = ActivePhaseView(
            phase=matching_phase,
            active_goals=active_goals,
            is_complete=phase_is_complete,
            virtue_overrides=matching_phase.virtue_overrides,
            skill_focus=skill_focus_dict,
        )

        # 4) Curriculum-level completion: all phases complete?
        all_complete = all(
            self._check_exit_conditions(phase, tech_state, world)
            for phase in self._config.phases
        )

        # 5) Long-horizon project unlocks
        unlocked_projects = self._compute_unlocked_projects(tech_state, world)

        return ActiveCurriculumView(
            curriculum_id=self._config.id,
            phase_view=phase_view,
            is_complete=all_complete,
            unlocked_projects=unlocked_projects,
        )

