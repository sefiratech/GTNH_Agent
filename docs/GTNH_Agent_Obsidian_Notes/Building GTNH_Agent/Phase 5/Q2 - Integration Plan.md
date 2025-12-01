
## Q1.1 – Self-Evaluation & Retry Loop

### 1. Goals

- Add a **structured self-evaluation loop** in M8 that:
    
    - Evaluates plans _before_ execution (using CriticModel + virtues).
        
    - Decides whether to **retry**, **adjust**, or **abandon** a plan.
        
    - Evaluates outcomes _after_ execution (using ErrorModel + Scribe).
        
- Introduce clear data structures and APIs so:
    
    - M2 (LLM stack), M4 (virtues), M9 (monitoring), and M10 (learning) can all hook into the loop cleanly.
        
- Force clean role separation:
    
    - Planner proposes.
        
    - Critic evaluates pre-execution.
        
    - ErrorModel diagnoses post-failure.
        
    - Scribe summarizes.
        

Primary module:

- M8 (`agent loop`)
    

Supporting modules:

- M2 (`llm_stack`)
    
- M4 (`virtue lattice`)
    
- M9 (`monitoring`)
    
- M10 (`experience / learning`)
    

---

### 2. Core Data Structures (M8 + spec)

**Files:**

- `src/agent/loop.py`
    
- `src/agent/schema.py` or `src/spec/agent_loop.py`

---
### 3. New Agent Loop Functions (M8)

**File:**

- `src/agent/loop.py`
    


```python
# path: src/agent/loop.py
"""
Phase 3+8: AgentLoop with self-evaluation + optional retry scaffold.

Concept:
    M8 should NOT recreate wiring for:
      - observation
      - encoding
      - planner / critic model calls

Instead, it should treat AgentRuntime (M6 + M7 + M2) as a service:

    runtime = build_agent_runtime(...)
    loop = AgentLoop(runtime=runtime, virtue_engine=..., skills=...)
    result = loop.run_episode(...)

This module provides:
    - AgentLoopConfig: basic knobs for the loop
    - EpisodeResult: structured return value
    - AgentLoop: a thin orchestrator over AgentRuntime

Execution logic (current version):
    - propose a plan via planner_tick()
    - build a PlanTrace
    - run pre-execution self-evaluation (critic + virtues)
    - optional retry loop around planning
    - execute accepted plan
    - run post-execution outcome evaluation (ErrorModel if available)
    - store an Experience in the ExperienceBuffer

Later, M8 can:
    - add richer RetryPolicy configuration
    - wire a dedicated ErrorModel encoder
    - insert more detailed TraceStep semantics
    - update tech_state
without changing the basic interface.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import logging
import time

from agent.runtime_m6_m7 import AgentRuntime
from agent.experience import Experience, ExperienceBuffer
from observation.trace_schema import PlanTrace, TraceStep
from observation.encoder import encode_for_critic
from spec.agent_loop import (
    VirtueEngine,
    PlanAttempt,
    PlanEvaluation,
    RetryPolicy,
)
from spec.experience import CriticResult

logger = logging.getLogger(__name__)


@dataclass
class AgentLoopConfig:
    """
    Configuration knobs for the high-level agent loop.

    Fields:
        enable_critic:
            If True, run the critic on each episode's PlanTrace and attach
            the result to the experience. If False, critic is skipped.

        enable_retry_loop:
            If True, the loop will run a simple pre-execution retry policy
            using Critic / virtues before committing to execution.

        store_experiences:
            If True, append an Experience record to the ExperienceBuffer
            after each episode.

        max_planner_calls:
            For now, controls how many planner calls run per episode. With
            retry disabled, this is typically 1 (plan once, then stop).

        max_skill_steps:
            Soft cap on how many plan steps (skills) are executed per episode.

        fail_fast_on_invalid_plan:
            If True, raise an error when the planner returns an invalid shape
            (e.g. missing "steps"). If False, the loop may choose to recover.

        log_virtue_scores:
            When True, virtue scores should be attached to traces if a virtue
            engine is present.

        log_traces:
            Reserved for future trace logging controls (e.g. to disk/monitoring).

        max_retries:
            Maximum number of times the loop will attempt to re-plan before
            giving up. Only used if enable_retry_loop is True.

        retryable_failure_types:
            If non-empty, only these failure_type values (from Critic/Error)
            are considered eligible for retry.

        abort_on_severity:
            If severity from Critic/Error is in this list, the loop will not
            retry and will instead abandon the plan.
    """
    enable_critic: bool = True
    enable_retry_loop: bool = False
    store_experiences: bool = True
    max_planner_calls: int = 1          # v1: single-plan episodes
    max_skill_steps: int = 16           # soft cap on steps from a plan
    fail_fast_on_invalid_plan: bool = True
    log_virtue_scores: bool = True
    log_traces: bool = True

    max_retries: int = 0
    retryable_failure_types: List[str] = field(default_factory=list)
    abort_on_severity: List[str] = field(default_factory=lambda: ["high"])


@dataclass
class EpisodeResult:
    """
    Result of a single agent episode: one or more planning attempts plus
    zero or more execution steps and evaluation data.

    Fields:
        episode_id:
            Optional external episode identifier, for logging / correlation.

        plan:
            The planner output dict returned by the *accepted* planning attempt.
            If all attempts were abandoned, this will be the last attempted plan.

        trace:
            PlanTrace that wraps the plan plus any execution details.

        critic_result:
            Output from the critic model (pre-execution), if enable_critic is
            True and a critic is configured on the runtime. Otherwise None.

        started_at / finished_at:
            Timestamps (seconds since epoch) for rough timing.

        plan_evaluation:
            PlanEvaluation struct representing the pre-execution evaluation
            for the accepted plan attempt, if any.

        outcome_evaluation:
            PlanEvaluation struct representing the post-execution outcome
            evaluation, if any (may be None if execution was skipped).
    """
    episode_id: Optional[int]
    plan: Dict[str, Any]
    trace: PlanTrace
    critic_result: Optional[CriticResult]
    started_at: float
    finished_at: float = 0.0
    plan_evaluation: Optional[PlanEvaluation] = None
    outcome_evaluation: Optional[PlanEvaluation] = None


@dataclass
class AgentLoop:
    """
    Episode-level agent loop built on top of AgentRuntime.

    Responsibilities:
      - Call planner via runtime
      - Optionally run a pre-execution self-evaluation + retry loop
      - Execute plan steps via injected skills / runtime
      - Build PlanTrace
      - Optionally call critic + virtues
      - Run a post-execution outcome evaluation (ErrorModel when present)
      - Store experiences

    This class is intentionally conservative:
        - It pushes as much "intelligence" as possible into AgentRuntime,
          VirtueEngine, and the LLM Stack.
        - It keeps the orchestration explicit and debuggable.
    """
    runtime: AgentRuntime
    experience_buffer: ExperienceBuffer = field(default_factory=ExperienceBuffer)
    config: AgentLoopConfig = field(default_factory=AgentLoopConfig)
    virtue_engine: Optional[VirtueEngine] = None   # M4 integration via protocol
    skills: Optional[Any] = None                   # reserved for M5/M8 integration

    # ------------------------------------------------
    # Public API
    # ------------------------------------------------

    def run_episode(self, episode_id: Optional[int] = None) -> EpisodeResult:
        """
        Run a single episode:
          planner → self-eval (+optional retry) → executor → outcome eval → experience buffer.
        """
        started_at = time.time()
        retry_policy = self._build_retry_policy()
        attempt_index = 0

        accepted_trace: Optional[PlanTrace] = None
        accepted_plan: Optional[Dict[str, Any]] = None
        accepted_plan_eval: Optional[PlanEvaluation] = None
        critic_result: Optional[CriticResult] = None

        # ------------------------------------------------
        # Planning + pre-execution evaluation + optional retry loop
        # ------------------------------------------------
        while True:
            if self.config.max_planner_calls > 0 and attempt_index >= self.config.max_planner_calls:
                logger.info("AgentLoop reached max_planner_calls=%d; stopping planning attempts.", self.config.max_planner_calls)
                break

            plan = self._call_planner_once()
            trace = self._build_initial_trace(plan)

            plan_eval, plan_critic_result = self._evaluate_plan(trace, attempt_index, retry_policy)
            critic_result = plan_critic_result

            # Decide whether to retry based on evaluation + policy
            if self._maybe_retry_plan(plan_eval, retry_policy):
                logger.info(
                    "AgentLoop retrying plan (attempt=%d, plan_id=%s, failure_type=%r, severity=%r)",
                    attempt_index,
                    plan_eval.plan_id,
                    plan_eval.failure_type,
                    plan_eval.severity,
                )
                attempt_index += 1
                if retry_policy.retry_budget > 0:
                    retry_policy.retry_budget -= 1
                continue

            # Plan accepted
            accepted_plan = plan
            accepted_trace = trace
            accepted_plan_eval = plan_eval
            break

        # Fallback: if nothing was accepted, keep the last attempt (if any)
        if accepted_plan is None or accepted_trace is None:
            logger.warning("AgentLoop did not accept any plan; using last attempted plan/trace.")
            if "plan" not in locals() or "trace" not in locals():
                # This really should not happen, but keep it safe.
                dummy_plan: Dict[str, Any] = {}
                dummy_trace = self._build_initial_trace(dummy_plan)
                accepted_plan, accepted_trace = dummy_plan, dummy_trace
            else:
                accepted_plan, accepted_trace = plan, trace  # type: ignore[assignment]

        # ------------------------------------------------
        # Execution + post-execution outcome evaluation
        # ------------------------------------------------
        self._execute_plan_into_trace(accepted_plan, accepted_trace)
        outcome_eval = self._evaluate_outcome(accepted_trace, attempt_index)

        finished_at = time.time()

        result = EpisodeResult(
            episode_id=episode_id,
            plan=accepted_plan,
            trace=accepted_trace,
            critic_result=critic_result,
            started_at=started_at,
            finished_at=finished_at,
            plan_evaluation=accepted_plan_eval,
            outcome_evaluation=outcome_eval,
        )

        # ------------------------------------------------
        # Experience logging
        # ------------------------------------------------
        if self.config.store_experiences:
            self._store_experience(result)

        return result

    # ------------------------------------------------
    # Internal helpers: planning + trace construction
    # ------------------------------------------------

    def _call_planner_once(self) -> Dict[str, Any]:
        """
        Delegate planning to AgentRuntime.

        Runtime handles:
          - BotCore.observe()
          - encode_for_planner()
          - PlannerModel.call()
        """
        plan = self.runtime.planner_tick()
        if self.config.fail_fast_on_invalid_plan:
            if not isinstance(plan, dict) or "steps" not in plan:
                raise ValueError("Planner returned invalid plan")
        return plan

    def _build_initial_trace(self, plan: Dict[str, Any]) -> PlanTrace:
        """
        Construct a PlanTrace with no steps yet, using the
        latest planner observation and tech state from runtime.
        """
        obs = self.runtime.get_latest_planner_observation()

        # Try runtime helpers first, then fall back to attributes
        get_tech_state = getattr(self.runtime, "get_tech_state", None)
        if callable(get_tech_state):
            tech_state = get_tech_state()
        else:
            tech_state = getattr(self.runtime, "current_tech_state", None)
            if tech_state is None:
                tech_state = getattr(self.runtime, "tech_state", None)

        get_context_id = getattr(self.runtime, "get_context_id", None)
        if callable(get_context_id):
            context_id = get_context_id()
        else:
            context_id = getattr(self.runtime.config, "context_id", "")

        return PlanTrace(
            plan=plan,
            steps=[],
            tech_state=tech_state,
            planner_payload=obs.json_payload,
            context_id=context_id,
            virtue_scores={},
        )

    # ------------------------------------------------
    # Internal helpers: self-evaluation + retry
    # ------------------------------------------------

    def _build_retry_policy(self) -> RetryPolicy:
        """
        Construct a RetryPolicy instance based on AgentLoopConfig.

        If retry is disabled or max_retries <= 0, the policy will effectively
        disable retries.
        """
        if not self.config.enable_retry_loop or self.config.max_retries <= 0:
            return RetryPolicy(
                max_retries=0,
                retry_budget=0,
                retryable_failure_types=[],
                abort_on_severity=[],
            )

        return RetryPolicy(
            max_retries=self.config.max_retries,
            retry_budget=self.config.max_retries,
            retryable_failure_types=list(self.config.retryable_failure_types),
            abort_on_severity=list(self.config.abort_on_severity),
        )

    def _evaluate_plan(
        self,
        trace: PlanTrace,
        attempt_index: int,
        retry_policy: RetryPolicy,
    ) -> Tuple[PlanEvaluation, Optional[CriticResult]]:
        """
        Run pre-execution evaluation for a planning attempt:

        - Optionally run CriticModel over the encoded PlanTrace
        - Optionally compute virtue scores
        - Construct a PlanEvaluation struct

        This does not decide retry itself; that is delegated to _maybe_retry_plan.
        """
        critic_result = self._maybe_run_critic(trace)
        self._maybe_run_virtues(trace)

        plan_dict = trace.plan or {}
        plan_id = (
            str(plan_dict.get("id"))
            if "id" in plan_dict
            else str(plan_dict.get("plan_id"))
            if "plan_id" in plan_dict
            else f"{trace.context_id or 'ctx'}-attempt-{attempt_index}"
        )

        virtue_scores: Dict[str, float] = trace.virtue_scores or {}

        critic_payload: Dict[str, Any] = {}
        failure_type: Optional[str] = None
        severity: Optional[str] = None
        fix_suggestions: Optional[List[str]] = None

        if critic_result is not None:
            if isinstance(critic_result, dict):
                critic_payload = dict(critic_result)
                failure_type = critic_payload.get("failure_type")
                severity = critic_payload.get("severity")
                fs = critic_payload.get("fix_suggestions")
                if isinstance(fs, list):
                    fix_suggestions = fs
            else:
                # best-effort: introspect attributes if present
                critic_payload = {"raw": critic_result}
                failure_type = getattr(critic_result, "failure_type", None)
                severity = getattr(critic_result, "severity", None)
                fs = getattr(critic_result, "fix_suggestions", None)
                if isinstance(fs, list):
                    fix_suggestions = fs

        evaluation = PlanEvaluation(
            plan_id=plan_id,
            attempt_index=attempt_index,
            virtue_scores=virtue_scores,
            critic_feedback=critic_payload,
            failure_type=failure_type,
            severity=severity,
            fix_suggestions=fix_suggestions,
        )
        return evaluation, critic_result

    def _maybe_retry_plan(self, evaluation: PlanEvaluation, policy: RetryPolicy) -> bool:
        """
        Decide whether to retry planning based on the evaluation + policy.

        This is intentionally conservative and can be refined later. For now:
            - If retry is disabled or budget exhausted → no retry
            - If severity is in abort_on_severity → no retry
            - If retryable_failure_types is non-empty and failure_type not in it → no retry
            - Otherwise, if failure_type is present and budget remains → retry
        """
        if not self.config.enable_retry_loop:
            return False

        if policy.max_retries <= 0 or policy.retry_budget <= 0:
            return False

        if evaluation.severity and evaluation.severity in policy.abort_on_severity:
            return False

        if policy.retryable_failure_types:
            if not evaluation.failure_type:
                return False
            if evaluation.failure_type not in policy.retryable_failure_types:
                return False

        if not evaluation.failure_type:
            # No explicit failure signaled; treat as acceptable.
            return False

        return True

    # ------------------------------------------------
    # Internal helpers: execution + postmortem
    # ------------------------------------------------

    def _execute_plan_into_trace(self, plan: Dict[str, Any], trace: PlanTrace) -> None:
        """
        Walk through the planner's steps and execute them via runtime.
        Record each low-level action as a TraceStep.

        v1 delegates the actual execution semantics to AgentRuntime:
        runtime.execute_plan_step(step_spec, step_idx) -> list[TraceStep]
        """
        steps: List[Dict[str, Any]] = plan.get("steps", [])
        max_steps = min(len(steps), self.config.max_skill_steps)

        for step_idx in range(max_steps):
            step_spec = steps[step_idx]

            # Delegate to runtime: resolve skill, generate actions, call BotCore
            execute_step = getattr(self.runtime, "execute_plan_step", None)
            if not callable(execute_step):
                # No executor wired yet; keep the trace empty and bail.
                logger.debug("AgentRuntime has no execute_plan_step; skipping execution.")
                break

            action_results: List[TraceStep] = execute_step(step_spec, step_idx)

            # runtime returns a list[TraceStep] so M8 just extends the trace
            for ts in action_results:
                trace.steps.append(ts)

            # Simple fail-fast policy: if the last step failed, stop early
            if action_results and not action_results[-1].result.success:
                break

    def _evaluate_outcome(self, trace: PlanTrace, attempt_index: int) -> PlanEvaluation:
        """
        Post-execution outcome evaluation.

        v1 behavior:
          - If an ErrorModel is present on the runtime and exposes .evaluate(),
            call it with the same encoded payload used for the critic.
          - Reuse virtue scores already attached to the trace (if any).
          - Construct a PlanEvaluation as a postmortem snapshot.

        This is intentionally minimal; a dedicated error encoder and richer
        schema can be added later without changing the AgentLoop interface.
        """
        plan_dict = trace.plan or {}
        plan_id = (
            str(plan_dict.get("id"))
            if "id" in plan_dict
            else str(plan_dict.get("plan_id"))
            if "plan_id" in plan_dict
            else f"{trace.context_id or 'ctx'}-attempt-{attempt_index}"
        )

        virtue_scores: Dict[str, float] = trace.virtue_scores or {}

        error_model = getattr(self.runtime, "error_model", None)
        error_payload: Dict[str, Any] = {}
        failure_type: Optional[str] = None
        severity: Optional[str] = None
        fix_suggestions: Optional[List[str]] = None

        if error_model is not None and callable(getattr(error_model, "evaluate", None)):
            try:
                encoded = encode_for_critic(trace)
                error_result = error_model.evaluate(encoded)
                if isinstance(error_result, dict):
                    error_payload = dict(error_result)
                    failure_type = error_payload.get("failure_type")
                    severity = error_payload.get("severity")
                    fs = error_payload.get("fix_suggestions")
                    if isinstance(fs, list):
                        fix_suggestions = fs
                else:
                    error_payload = {"raw": error_result}
                    failure_type = getattr(error_result, "failure_type", None)
                    severity = getattr(error_result, "severity", None)
                    fs = getattr(error_result, "fix_suggestions", None)
                    if isinstance(fs, list):
                        fix_suggestions = fs
            except Exception as exc:
                logger.warning("ErrorModel evaluation failed: %r", exc)

        evaluation = PlanEvaluation(
            plan_id=plan_id,
            attempt_index=attempt_index,
            virtue_scores=virtue_scores,
            critic_feedback=error_payload,
            failure_type=failure_type,
            severity=severity,
            fix_suggestions=fix_suggestions,
        )
        return evaluation

    # ------------------------------------------------
    # Internal helpers: critic, virtues, experience
    # ------------------------------------------------

    def _maybe_run_critic(self, trace: PlanTrace) -> Optional[CriticResult]:
        """
        Optionally run the critic on the trace.

        Critic is advisory; failures must not crash the loop.
        """
        if not self.config.enable_critic:
            return None

        critic_model = getattr(self.runtime, "critic_model", None)
        if critic_model is None:
            return None

        try:
            critic_payload = encode_for_critic(trace)
            result = critic_model.evaluate(critic_payload)
            return result
        except Exception as exc:
            logger.warning("Critic evaluation failed: %r", exc)
            return None

    def _maybe_run_virtues(self, trace: PlanTrace) -> None:
        """
        Optionally compute virtue scores over the trace.

        This is a best-effort telemetry hook; failures are ignored.
        """
        if not self.config.log_virtue_scores or self.virtue_engine is None:
            return

        try:
            scores = self.virtue_engine.score_trace(
                trace=trace,
                context_id=trace.context_id,
            )
            trace.virtue_scores = scores
        except Exception as exc:
            logger.warning("Virtue scoring failed: %r", exc)

    def _store_experience(self, result: EpisodeResult) -> None:
        """
        Convert an EpisodeResult into an Experience and append to the buffer.
        """
        trace = result.trace

        # Snapshot tech state in a defensive way
        ts = trace.tech_state
        if ts is None:
            tech_state_snapshot: Dict[str, Any] = {}
        elif hasattr(ts, "to_serializable"):
            tech_state_snapshot = ts.to_serializable()  # type: ignore[assignment]
        elif isinstance(ts, dict):
            tech_state_snapshot = dict(ts)
        else:
            tech_state_snapshot = {"raw": ts}

        # Try to extract env profile name; fall back to empty string
        env_profile_name = getattr(self.runtime.config, "profile_name", "")

        xp = Experience(
            trace=trace,
            critic_result=result.critic_result,
            episode_id=result.episode_id,
            env_profile_name=env_profile_name,
            context_id=trace.context_id,
            tech_state_snapshot=tech_state_snapshot,
            meta={
                "duration_sec": result.finished_at - result.started_at,
                "planner_model": getattr(self.runtime, "planner_name", ""),
                "plan_evaluation": {
                    "pre": result.plan_evaluation,
                    "post": result.outcome_evaluation,
                },
            },
        )
        self.experience_buffer.add(xp)

```



```python
# path: src/spec/agent_loop.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Protocol

from observation.trace_schema import PlanTrace

from .types import WorldState
from .llm import PlannerModel, CriticModel
from .skills import SkillRegistry
from .bot_core import BotCore


@dataclass
class PlanAttempt:
    """
    Spec-level representation of a planning attempt.

    This is intentionally decoupled from the underlying planner's raw output
    so that M8/M10 can reason about attempts in a uniform way, regardless
    of how the LLM stack formats its responses.
    """
    plan_id: str
    goal: Any  # typically AgentGoal (see spec.types or spec.agent_loop extensions)
    task_plan: Any  # typically TaskPlan (see spec.skills / spec.agent_loop)
    created_at: float
    attempt_index: int = 0


@dataclass
class PlanEvaluation:
    """
    Spec-level representation of plan evaluation results.

    Used both for:
        - pre-execution evaluation (CriticModel + virtues)
        - post-execution outcome evaluation (ErrorModel, if present)
    """
    plan_id: str
    attempt_index: int
    virtue_scores: Dict[str, float]
    critic_feedback: Dict[str, Any]
    failure_type: Optional[str] = None
    severity: Optional[str] = None
    fix_suggestions: Optional[List[str]] = None


@dataclass
class RetryPolicy:
    """
    Policy controlling how the agent retries plan generation.

    Fields:
        max_retries:
            Hard cap on how many planning attempts the loop will make.

        retry_budget:
            Mutable remaining retry count; decremented by the loop.

        retryable_failure_types:
            If non-empty, only failure_type values in this list are eligible
            for retry (e.g. "plan_quality", "missing_detail").

        abort_on_severity:
            If severity is in this list, the loop should not retry and instead
            abandon the plan (e.g. "high", "fatal").
    """
    max_retries: int
    retry_budget: int
    retryable_failure_types: List[str] = field(default_factory=list)
    abort_on_severity: List[str] = field(default_factory=list)


class AgentLoop(Protocol):
    """High-level control loop for the GTNH Agent.

    This ties together:
    - BotCore (M6) for acting in the world
    - SkillRegistry (M5) for available behaviors
    - PlannerModel / CriticModel (M2) for plan generation & evaluation
    """

    # Structural dependencies (not required at runtime, but part of the design)
    bot: BotCore
    skills: SkillRegistry
    planner: PlannerModel
    critic: CriticModel

    def step(self) -> None:
        """
        Perform one full agent iteration:
        - observe world via BotCore.get_world_state()
        - encode world into an Observation (M7)
        - compute or reuse a plan using PlannerModel (M2)
        - select and execute skills (M5) via BotCore.execute_action()
        - evaluate / record experience (M10)
        """
        ...

    def set_goal(self, goal: str, context: Mapping[str, Any]) -> None:
        """Set or update the current top-level goal (e.g. 'establish LV steam power')."""
        ...

    def get_status(self) -> Mapping[str, Any]:
        """
        Provide a snapshot of:
        - current goal
        - current plan
        - last action/result
        - any error state
        - any high-level metrics (ticks alive, deaths, etc.)
        """
        ...


class VirtueEngine(Protocol):
    """
    Protocol for a virtue scoring engine.

    Implemented in M4 using virtues.lattice and virtues.metrics.
    M8 (AgentLoop) only uses this via dependency injection: it does not know
    how virtues are computed, it only receives scores over a PlanTrace.
    """

    def score_trace(self, trace: PlanTrace, context_id: str) -> Dict[str, float]:
        ...

```





---

### 4. LLM Stack Requirements (M2)

**Files:**

- `src/llm_stack/plan_code.py`
    
- `src/llm_stack/critic.py`
    
- `src/llm_stack/error_model.py`
    
- `src/spec/llm.py`
    

#### 4.1 Planner

- `call_planner(goal: AgentGoal, world_summary) -> TaskPlan`
    
    - Returns structured plan only.
        
    - No critique, no retry logic embedded.
        

```python
# path: src/llm_stack/plan_code.py

from __future__ import annotations

import json
import logging
from typing import Dict, Any, List

from spec.llm import PlanCodeModel
from spec.types import Observation
from .schema import (
    PlanRequest,
    PlanResponse,
    PlanStep,
    SkillImplRequest,
    SkillImplResponse,
)
from .backend import LLMBackend
from .presets import RolePreset
from .json_utils import load_json_or_none
from .log_files import log_llm_call  # <-- new import


logger = logging.getLogger(__name__)


class PlanCodeModelImpl(PlanCodeModel):
    """PlanCodeModel backed by a single local LLM with role presets.

    Responsibilities:
        - Planning:
            Convert a goal + observation + skills + constraints into a
            structured plan object with "steps".
        - Codegen:
            Propose Python implementations for skills.

    Notes on Q1.1 / Q1.3 alignment:
        - `plan(...)` is the low-level primitive used by the current runtime.
        - `call_planner(goal, world_summary)` is the higher-level planner-facing
          API used by hierarchical planning code. It returns a structured plan
          only (no critique / retry logic embedded).
    """

    def __init__(self, backend: LLMBackend, preset: RolePreset) -> None:
        self._backend = backend
        self._preset = preset

    # -------------------------------------------------------------------------
    # High-level planner entrypoint (Q1.1 / Q1.3)
    # -------------------------------------------------------------------------

    def call_planner(
        self,
        goal: Any,
        world_summary: Dict[str, Any],
    ) -> Dict[str, Any]:
        """High-level planner entrypoint for hierarchical planning.

        Expected usage (conceptual, not enforced here):

            - `goal` is an AgentGoal-like object, with at least a `.text`
              or `.description` attribute. If it's a plain string, that is
              used directly as the goal text.

            - `world_summary` is a dict providing:
                - "observation": Observation
                - "skills": Dict[str, Dict[str, Any]]
                - "constraints": Dict[str, Any]

        This method:
            - extracts a goal string from `goal`
            - pulls the Observation + skills + constraints from world_summary
            - delegates to `plan(...)`
            - returns the structured plan dict produced by the low-level API

        It returns a JSON-like dict with:
            {
                "steps": [
                    {"skill": "...", "params": {...}},
                    ...
                ],
                "notes": "optional notes",
                "raw_text": "<raw LLM response>",
                # on error:
                # "error": "...",
            }

        No critique or retry logic is embedded here; self-evaluation and
        retry live in the AgentLoop / Critic / ErrorModel layers.
        """
        # Derive goal text
        if isinstance(goal, str):
            goal_text = goal
        else:
            goal_text = getattr(goal, "text", None) or getattr(goal, "description", None)
            if not goal_text:
                goal_text = str(goal)

        observation = world_summary.get("observation")
        skills = world_summary.get("skills", {})
        constraints = world_summary.get("constraints", {})

        if not isinstance(observation, Observation):
            raise TypeError(
                "world_summary['observation'] must be an Observation; "
                f"got {type(observation)!r}"
            )

        return self.plan(
            observation=observation,
            goal=goal_text,
            skill_descriptions=skills,
            constraints=constraints,
        )

    # -------------------------------------------------------------------------
    # Planning (low-level primitive used by current runtime)
    # -------------------------------------------------------------------------

    def _build_plan_prompt(self, req: PlanRequest) -> str:
        skills_json = json.dumps(req.skills, indent=2)
        obs_json = json.dumps(req.observation, indent=2)
        constraints_json = json.dumps(req.constraints, indent=2)

        prompt = f"""
You are the planning and coding brain for a Minecraft GTNH automation agent.

Your ONLY job now is to produce a JSON object describing a plan.
Do NOT write explanations, commentary, Markdown, or any text before or after the JSON.
The FIRST non-whitespace character in your reply MUST be '{{'.
The LAST non-whitespace character in your reply MUST be '}}'.

Keep the plan concise. Prefer 1–4 steps unless constraints require more.

Observation:
{obs_json}

Goal:
{req.goal}

Available skills (with descriptions and parameters):
{skills_json}

Constraints:
{constraints_json}

Return JSON exactly in this structure:
{{
  "steps": [
    {{"skill": "str", "params": {{ ... }} }},
    ...
  ],
  "notes": "optional free-form explanation"
}}
"""
        return prompt.strip()

    def _extract_json_object(self, raw: str) -> str:
        """
        Try to salvage a JSON object from a noisy LLM reply.

        Strategy:
        - Strip whitespace
        - Find first '{' and last '}' and take that slice
        - If that fails, just return the original raw text
        """
        if not raw:
            return raw

        trimmed = raw.strip()
        first = trimmed.find("{")
        last = trimmed.rfind("}")

        if first == -1 or last == -1 or last <= first:
            # No obvious JSON object; give the caller the original
            return trimmed

        candidate = trimmed[first : last + 1]
        logger.debug("PlanCodeModel extracted JSON candidate: %s", candidate)
        return candidate

    def plan(
        self,
        observation: Observation,
        goal: str,
        skill_descriptions: Dict[str, Dict[str, Any]],
        constraints: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Low-level planning primitive.

        Inputs:
            observation:
                Observation object with json_payload + text_summary.

            goal:
                Text description of the immediate objective.

            skill_descriptions:
                Mapping from skill name → metadata dict. Used by the prompt
                to tell the model what tools it can call.

            constraints:
                Arbitrary constraint dict (e.g. time, risk level, resource
                hints, etc.).

        Returns:
            JSON-like dict with keys:
                - "steps": list of {"skill": str, "params": dict}
                - "notes": optional free text
                - "raw_text": raw LLM output
                - "error": present if JSON parsing failed
        """
        obs_dict = {
            "json_payload": observation.json_payload,
            "text_summary": observation.text_summary,
        }

        req = PlanRequest(
            observation=obs_dict,
            goal=goal,
            skills=skill_descriptions,
            constraints=constraints,
        )
        prompt = self._build_plan_prompt(req)
        logger.debug("PlanCodeModel plan prompt: %s", prompt)

        # IMPORTANT: respect the preset; no hidden clamp here.
        max_tokens = self._preset.max_tokens

        raw = self._backend.generate(
            prompt,
            max_tokens=max_tokens,
            temperature=self._preset.temperature,
            stop=self._preset.stop,
            system_prompt=self._preset.system_prompt,
        )
        logger.debug("PlanCodeModel plan raw output: %s", raw)

        # File-level log for replay / forensics
        log_llm_call(
            role="plan_code",
            operation="plan",
            prompt=prompt,
            raw_response=raw,
            extra={
                "goal": goal,
                "constraints": constraints,
                "skill_count": len(skill_descriptions),
            },
        )

        # Try to clean up any extra junk around the JSON
        candidate = self._extract_json_object(raw)

        # Use shared JSON util so parsing behavior is consistent across modules
        data, err = load_json_or_none(candidate, context="PlanCodeModel.plan")
        if data is None:
            # Don't hard-crash here; surface the raw text so higher layers
            # (or ErrorModel) can decide what to do.
            logger.error("PlanCodeModel JSON decode error: %s", err)
            return {
                "steps": [],
                "notes": "json_parse_error",
                "raw_text": raw,
                "error": err,
            }

        steps = [
            PlanStep(skill=step["skill"], params=step.get("params", {}))
            for step in data.get("steps", [])
        ]
        resp = PlanResponse(
            steps=steps,
            notes=data.get("notes", ""),
            raw_text=raw,
        )
        return {
            "steps": [s.__dict__ for s in resp.steps],
            "notes": resp.notes,
            "raw_text": resp.raw_text,
        }

    # -------------------------------------------------------------------------
    # Codegen
    # -------------------------------------------------------------------------

    def _build_code_prompt(self, req: SkillImplRequest) -> str:
        name = req.skill_spec.get("name", "unknown_skill")
        description = req.skill_spec.get("description", "")
        params = req.skill_spec.get("params", {})

        examples_text = ""
        for ex in req.examples:
            examples_text += f"\nExample trace:\n{json.dumps(ex, indent=2)}\n"

        prompt = f"""
You are generating Python code for a skill in a Minecraft GTNH agent.

Skill name: {name}
Description: {description}
Parameters: {json.dumps(params, indent=2)}
{examples_text}

Write a Python function body that, given a WorldState and params dict,
returns a list of Action objects to achieve the skill.

Return ONLY Python code, without backticks or explanations.
"""
        return prompt.strip()

    def propose_skill_implementation(
        self,
        skill_spec: Dict[str, Any],
        examples: List[Dict[str, Any]],
    ) -> str:
        req = SkillImplRequest(skill_spec=skill_spec, examples=examples)
        prompt = self._build_code_prompt(req)
        logger.debug("PlanCodeModel codegen prompt: %s", prompt)

        raw = self._backend.generate(
            prompt,
            max_tokens=self._preset.max_tokens,
            temperature=min(self._preset.temperature, 0.2),
            stop=None,
            system_prompt=self._preset.system_prompt,
        )
        logger.debug("PlanCodeModel codegen raw output: %s", raw)

        # File-level log for replay / forensics
        log_llm_call(
            role="plan_code",
            operation="codegen",
            prompt=prompt,
            raw_response=raw,
            extra={
                "skill_name": skill_spec.get("name"),
            },
        )

        resp = SkillImplResponse(code=raw, notes="", raw_text=raw)
        return resp.code

```


#### 4.2 CriticModel

New or completed adapter in `critic.py`:

```python
# path: src/llm_stack/critic.py

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from spec.llm import CriticModel
from .backend import LLMBackend
from .presets import RolePreset
from .json_utils import load_json_or_none
from .log_files import log_llm_call

logger = logging.getLogger(__name__)


class CriticModelImpl(CriticModel):
    """CriticModel backed by a single local LLM with role presets.

    Responsibilities:
        - Pre-execution critique of candidate plans.
        - Post-execution diagnosis of failed or suboptimal episodes
          (same schema, different context).

    Contract:
        - `evaluate(payload)` is the generic entrypoint used by M8:
            payload should contain:
                {
                    "plan": {...},             # structured plan dict
                    "observation": {...},      # world view / trace summary
                    "virtue_scores": {...},    # optional, may be empty
                    "context": {...},          # optional metadata
                }

          It returns a dict:
                {
                    "failure_type": Optional[str],
                    "severity": Optional[str],
                    "fix_suggestions": List[str],
                    "notes": str,
                }
          plus optional fields:
                "raw_text", "error"

        - `call_critic(plan, world_summary, virtues_hint)` is a convenience
          wrapper for higher-level callers that don't want to assemble the
          payload by hand.
    """

    def __init__(self, backend: LLMBackend, preset: RolePreset) -> None:
        self._backend = backend
        self._preset = preset

    # -------------------------------------------------------------------------
    # High-level helper
    # -------------------------------------------------------------------------

    def call_critic(
        self,
        plan: Dict[str, Any],
        world_summary: Dict[str, Any],
        virtues_hint: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """Convenience wrapper to critique a plan with world / virtue context.

        Args:
            plan:
                Structured plan dict, typically:
                    {"steps": [...], "notes": "...", ...}

            world_summary:
                Dict with keys like:
                    "observation": {...},  # text + json summary
                    "context": {...},      # optional metadata

            virtues_hint:
                Optional virtue score dict that can help the critic reason
                about tradeoffs, e.g. {"prudence": 0.7, "courage": 0.4, ...}.

        Returns:
            Critic response dict as described in the class docstring.
        """
        payload: Dict[str, Any] = {
            "plan": plan,
            "observation": world_summary.get("observation", {}),
            "context": world_summary.get("context", {}),
        }
        if virtues_hint is not None:
            payload["virtue_scores"] = virtues_hint

        return self.evaluate(payload)

    # -------------------------------------------------------------------------
    # Core API used by AgentLoop / encode_for_critic
    # -------------------------------------------------------------------------

    def _build_critic_prompt(self, payload: Dict[str, Any]) -> str:
        plan = payload.get("plan", {}) or {}
        observation = payload.get("observation", {}) or {}
        virtue_scores = payload.get("virtue_scores", {}) or {}
        context = payload.get("context", {}) or {}

        plan_json = json.dumps(plan, indent=2)
        obs_json = json.dumps(observation, indent=2)
        virtues_json = json.dumps(virtue_scores, indent=2)
        ctx_json = json.dumps(context, indent=2)

        prompt = f"""
You are a safety and quality critic for a Minecraft GTNH automation agent.

Your ONLY job now is to evaluate a candidate plan and provide structured
feedback on risks, failure modes, and possible improvements.

You MUST return a single JSON object and nothing else.
The FIRST non-whitespace character in your reply MUST be '{{'.
The LAST non-whitespace character in your reply MUST be '}}'.

The JSON MUST have this structure:

{{
  "failure_type": "str or null",
  "severity": "low" | "medium" | "high" | null,
  "fix_suggestions": ["list", "of", "concrete", "improvements"],
  "notes": "free-form explanation"
}}

Use these semantics:
- failure_type:
    - null if the plan looks acceptable
    - "plan_quality" if steps are unclear, missing, or contradictory
    - "missing_prereq" if prerequisites are absent
    - "resource_risk" if it likely fails due to resources
    - "safety_risk" if it risks killing the player or corrupting the world
- severity:
    - "low": minor issues, plan is probably fine
    - "medium": significant risk or inefficiency
    - "high": likely to fail or cause serious problems
- fix_suggestions:
    - 1–4 specific, actionable improvements (or empty list if none)
- notes:
    - brief explanation of your reasoning

Plan under review:
{plan_json}

Observation / world summary:
{obs_json}

Virtue scores (higher is better):
{virtues_json}

Context:
{ctx_json}
"""
        return prompt.strip()

    def _extract_json_object(self, raw: str) -> str:
        """
        Try to salvage a JSON object from a noisy LLM reply.

        Strategy:
        - Strip whitespace
        - Find first '{' and last '}' and take that slice
        - If that fails, just return the original raw text
        """
        if not raw:
            return raw

        trimmed = raw.strip()
        first = trimmed.find("{")
        last = trimmed.rfind("}")

        if first == -1 or last == -1 or last <= first:
            return trimmed

        candidate = trimmed[first : last + 1]
        logger.debug("CriticModel extracted JSON candidate: %s", candidate)
        return candidate

    def evaluate(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a plan + context payload and return structured feedback.

        This is the entrypoint expected by AgentLoop._maybe_run_critic:

            critic_payload = encode_for_critic(trace)
            result = critic_model.evaluate(critic_payload)

        Args:
            payload:
                Dict containing at least:
                    "plan": {...},
                    "observation": {...},
                and optionally:
                    "virtue_scores": {...},
                    "context": {...}.

        Returns:
            Dict with keys:
                - "failure_type": Optional[str]
                - "severity": Optional[str]
                - "fix_suggestions": List[str]
                - "notes": str
                - "raw_text": raw LLM output
              plus optional:
                - "error": str, if parsing failed
        """
        prompt = self._build_critic_prompt(payload)
        logger.debug("CriticModel evaluate prompt: %s", prompt)

        # Be slightly conservative on temperature for critique.
        max_tokens = self._preset.max_tokens
        temperature = min(self._preset.temperature, 0.4)

        raw = self._backend.generate(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=self._preset.stop,
            system_prompt=self._preset.system_prompt,
        )
        logger.debug("CriticModel evaluate raw output: %s", raw)

        # File-level log for replay / forensics
        log_llm_call(
            role="critic",
            operation="evaluate",
            prompt=prompt,
            raw_response=raw,
            extra={
                "has_plan": "plan" in payload,
                "has_observation": "observation" in payload,
            },
        )

        candidate = self._extract_json_object(raw)
        data, err = load_json_or_none(candidate, context="CriticModel.evaluate")

        if data is None:
            logger.error("CriticModel JSON decode error: %s", err)
            # Return a best-effort fallback result so the caller can
            # still treat this like a CriticResponse.
            return {
                "failure_type": None,
                "severity": None,
                "fix_suggestions": [],
                "notes": "json_parse_error",
                "raw_text": raw,
                "error": err,
            }

        # Normalize fields and fill in defaults
        failure_type = data.get("failure_type")
        severity = data.get("severity")
        fix_suggestions = data.get("fix_suggestions", [])
        notes = data.get("notes", "")

        if not isinstance(fix_suggestions, list):
            fix_suggestions = [str(fix_suggestions)]

        result: Dict[str, Any] = {
            "failure_type": failure_type,
            "severity": severity,
            "fix_suggestions": [str(s) for s in fix_suggestions],
            "notes": str(notes),
            "raw_text": raw,
        }
        return result

```

#### 4.3 ErrorModel

- `call_error_model(episode_trace, plan) -> dict` with a compatible schema:
    
    - Same fields as CriticModel where possible:
        
        - `failure_type`
            
        - `severity`
            
        - `fix_suggestions`
            
- Used **only** after execution failures in `evaluate_outcome`.
    

```python
# path: src/llm_stack/error_model.py

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from spec.llm import ErrorModel as ErrorModelInterface
from .schema import ErrorContext, ErrorAnalysis
from .backend import LLMBackend
from .presets import RolePreset
from .json_utils import load_json_or_none
from .log_files import log_llm_call


logger = logging.getLogger(__name__)


class ErrorModelImpl(ErrorModelInterface):
    """ErrorModel backed by the same local LLM, low temperature.

    Responsibilities (Q1-aligned):

      1. Post-execution outcome analysis for plans (agent-level):
         - `evaluate(payload)`:
             payload typically contains:
                 {
                     "plan": {...},
                     "observation": {...},
                     "virtue_scores": {...},   # optional
                     "trace": {...},           # optional episode summary
                     "context": {...},         # optional metadata
                 }
             returns a dict with:
                 {
                     "failure_type": Optional[str],
                     "severity": Optional[str],
                     "fix_suggestions": List[str],
                     "notes": str,
                     "raw_text": str,
                     "error": Optional[str],   # if parsing failed
                 }

         - `call_error_model(episode_trace, plan)`:
             convenience wrapper that assembles a payload and calls `evaluate`.

      2. LLM-call error analysis (legacy / infra-level):
         - `analyze_failure(ctx: ErrorContext) -> ErrorAnalysis`
             unchanged in spirit:
                 {
                   "classification": "short_label",
                   "summary": "short explanation",
                   "suggested_fix": {...},
                   "retry_advised": bool,
                   "raw_text": str,
                 }

    The two paths are deliberately separate:
      - AgentLoop uses `evaluate` / `call_error_model`.
      - Lower-level LLM infra may still use `analyze_failure`.
    """

    def __init__(self, backend: LLMBackend, preset: RolePreset) -> None:
        self._backend = backend
        self._preset = preset

    # -------------------------------------------------------------------------
    # Q1: Agent-level outcome evaluation
    # -------------------------------------------------------------------------

    def call_error_model(
        self,
        episode_trace: Any,
        plan: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Convenience wrapper for agent-level outcome evaluation.

        Args:
            episode_trace:
                Typically a PlanTrace or a dict-like representation of an
                episode. We only make best-effort use of it here; the caller
                can pre-encode a more detailed payload if desired.

            plan:
                Structured plan dict (same schema as CriticModel sees).

        Returns:
            ErrorModel-style result dict with keys:
                - failure_type
                - severity
                - fix_suggestions
                - notes
                - raw_text
                - error (optional)
        """
        # Best-effort extraction of observation / trace info.
        if isinstance(episode_trace, dict):
            observation = episode_trace.get("observation", {})
            trace_info = episode_trace
        else:
            # Try some common attributes; if they don't exist, fall back.
            observation = getattr(episode_trace, "observation", None)
            if observation is None and hasattr(episode_trace, "planner_payload"):
                # Observations often encoded as json_payload in trace.
                observation = {
                    "json_payload": getattr(episode_trace, "planner_payload", None),
                    "text_summary": getattr(episode_trace, "text_summary", ""),
                }
            trace_info = getattr(episode_trace, "to_serializable", None)
            if callable(trace_info):
                trace_info = trace_info()
            elif not isinstance(trace_info, dict):
                trace_info = {"raw": str(episode_trace)}

        payload: Dict[str, Any] = {
            "plan": plan,
            "observation": observation or {},
            "trace": trace_info,
            "context": {"source": "agent_outcome"},
        }
        return self.evaluate(payload)

    def _build_outcome_prompt(self, payload: Dict[str, Any]) -> str:
        plan = payload.get("plan", {}) or {}
        observation = payload.get("observation", {}) or {}
        trace = payload.get("trace", {}) or {}
        virtue_scores = payload.get("virtue_scores", {}) or {}
        context = payload.get("context", {}) or {}

        plan_json = json.dumps(plan, indent=2)
        obs_json = json.dumps(observation, indent=2)
        trace_json = json.dumps(trace, indent=2)
        virtues_json = json.dumps(virtue_scores, indent=2)
        ctx_json = json.dumps(context, indent=2)

        prompt = f"""
You are a failure analyst for a Minecraft GTNH automation agent.

The plan has ALREADY BEEN EXECUTED. Your job is to analyze what went wrong
(or what nearly went wrong) and provide structured guidance for improving
future plans and skills.

You MUST return a single JSON object and nothing else.
The FIRST non-whitespace character in your reply MUST be '{{'.
The LAST non-whitespace character in your reply MUST be '}}'.

The JSON MUST have this structure:

{{
  "failure_type": "str or null",
  "severity": "low" | "medium" | "high" | null,
  "fix_suggestions": ["list", "of", "concrete", "improvements"],
  "notes": "free-form explanation"
}}

Use these semantics:
- failure_type:
    - null if execution was successful enough
    - "execution_error" if steps failed at runtime (blocked, exceptions, etc.)
    - "missing_prereq" if runtime lacked required resources/tech
    - "resource_exhaustion" if resources ran out or would soon
    - "safety_risk" if it caused or risked serious harm (deaths, explosions)
    - "plan_mismatch" if execution diverged from plan in a problematic way
- severity:
    - "low": minor issues, easy to patch
    - "medium": noticeable failures, some impact
    - "high": serious or repeated failure; plan/skill needs revision
- fix_suggestions:
    - 1–4 specific, actionable changes (e.g. new prerequisites, checks,
      or alternative strategies)
- notes:
    - brief explanation of what happened and why

Plan that was executed:
{plan_json}

Observation / world summary at planning time:
{obs_json}

Execution trace (summarized):
{trace_json}

Virtue scores (if any, higher is better):
{virtues_json}

Context:
{ctx_json}
"""
        return prompt.strip()

    def _extract_json_object(self, raw: str) -> str:
        """
        Try to salvage a JSON object from a noisy LLM reply.

        Strategy:
        - Strip whitespace
        - Find first '{' and last '}' and take that slice
        - If that fails, just return the original raw text
        """
        if not raw:
            return raw

        trimmed = raw.strip()
        first = trimmed.find("{")
        last = trimmed.rfind("}")

        if first == -1 or last == -1 or last <= first:
            return trimmed

        candidate = trimmed[first : last + 1]
        logger.debug("ErrorModel extracted JSON candidate: %s", candidate)
        return candidate

    def evaluate(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Agent-level outcome evaluation entrypoint.

        This mirrors CriticModel.evaluate's contract where possible, but is
        intended for post-execution analysis.

        Args:
            payload:
                Dict containing at least:
                    "plan": {...},
                    "observation": {...},
                and optionally:
                    "trace": {...},
                    "virtue_scores": {...},
                    "context": {...}.

        Returns:
            Dict with keys:
                - "failure_type": Optional[str]
                - "severity": Optional[str]
                - "fix_suggestions": List[str]
                - "notes": str
                - "raw_text": raw LLM output
              plus optional:
                - "error": str, if parsing failed
        """
        prompt = self._build_outcome_prompt(payload)
        logger.debug("ErrorModel evaluate prompt: %s", prompt)

        max_tokens = self._preset.max_tokens
        temperature = min(self._preset.temperature, 0.3)

        raw = self._backend.generate(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=self._preset.stop,
            system_prompt=self._preset.system_prompt,
        )
        logger.debug("ErrorModel evaluate raw output: %s", raw)

        log_llm_call(
            role="error_model",
            operation="evaluate_outcome",
            prompt=prompt,
            raw_response=raw,
            extra={
                "has_plan": "plan" in payload,
                "has_observation": "observation" in payload,
            },
        )

        candidate = self._extract_json_object(raw)
        data, err = load_json_or_none(candidate, context="ErrorModel.evaluate")

        if data is None:
            logger.error("ErrorModel JSON decode error: %s", err)
            return {
                "failure_type": None,
                "severity": None,
                "fix_suggestions": [],
                "notes": "json_parse_error",
                "raw_text": raw,
                "error": err,
            }

        failure_type = data.get("failure_type")
        severity = data.get("severity")
        fix_suggestions = data.get("fix_suggestions", [])
        notes = data.get("notes", "")

        if not isinstance(fix_suggestions, list):
            fix_suggestions = [str(fix_suggestions)]

        return {
            "failure_type": failure_type,
            "severity": severity,
            "fix_suggestions": [str(s) for s in fix_suggestions],
            "notes": str(notes),
            "raw_text": raw,
        }

    # -------------------------------------------------------------------------
    # Legacy LLM-call failure analysis (infra-level)
    # -------------------------------------------------------------------------

    def _build_prompt(self, ctx: ErrorContext) -> str:
        """Legacy prompt for analyzing LLM call failures.

        This is kept separate from the agent-level outcome evaluation.
        """
        ctx_json = json.dumps(
            {
                "role": ctx.role,
                "operation": ctx.operation,
                "prompt": ctx.prompt,
                "raw_response": ctx.raw_response,
                "error_type": ctx.error_type,
                "metadata": ctx.metadata,
            },
            indent=2,
        )

        prompt = f"""
You are an error analyst for an LLM-based GTNH agent.

Here is the context of a failure:
{ctx_json}

Explain:
1. What likely went wrong (classification).
2. A brief human-readable summary.
3. A JSON object suggesting how to fix or retry.

Respond ONLY with valid JSON:
{{
  "classification": "short_label",
  "summary": "short explanation",
  "suggested_fix": {{ ... }},
  "retry_advised": true or false
}}
"""
        return prompt.strip()

    def analyze_failure(self, ctx: ErrorContext) -> ErrorAnalysis:
        """Legacy API: analyze a single LLM call failure.

        This is kept for compatibility with tooling that inspects backend
        errors and wants a richer ErrorAnalysis object.
        """
        prompt = self._build_prompt(ctx)
        logger.debug("ErrorModel prompt: %s", prompt)

        raw = self._backend.generate(
            prompt,
            max_tokens=self._preset.max_tokens,
            temperature=self._preset.temperature,
            stop=self._preset.stop,
            system_prompt=self._preset.system_prompt,
        )
        logger.debug("ErrorModel raw output: %s", raw)

        # File-level log for replay / forensics
        log_llm_call(
            role="error_model",
            operation="analyze_failure",
            prompt=prompt,
            raw_response=raw,
            extra={
                "error_type": ctx.error_type,
                "source_role": ctx.role,
                "source_operation": ctx.operation,
            },
        )

        candidate = self._extract_json_object(raw)
        data, err = load_json_or_none(candidate, context="ErrorModel.analyze_failure")
        if data is None:
            # If the error model itself returns garbage, we still give callers
            # a structured ErrorAnalysis describing that situation.
            logger.error("ErrorModel JSON decode error: %s", err)
            return ErrorAnalysis(
                classification="json_parse_error",
                summary=f"Failed to parse ErrorModel JSON: {err}",
                suggested_fix={},
                retry_advised=False,
                raw_text=raw,
            )

        return ErrorAnalysis(
            classification=data.get("classification", "unknown"),
            summary=data.get("summary", ""),
            suggested_fix=data.get("suggested_fix", {}),
            retry_advised=bool(data.get("retry_advised", False)),
            raw_text=raw,
        )

```

#### 4.4 Spec

In `src/spec/llm.py`, define:

- `CriticResponse`
    
- `ErrorModelResponse`
    

With shared fields for `failure_type`, `severity`, `fix_suggestions`.


```python
# path: src/spec/llm.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Dict, Any, List, Optional

from .types import Observation


# ---------------------------------------------------------------------------
# Shared response shapes for critic / error model
# ---------------------------------------------------------------------------

@dataclass
class CriticResponse:
    """
    Structured response from the CriticModel.

    Implementations (e.g. llm_stack.critic.CriticModelImpl) are free to
    return a plain dict at runtime, but this dataclass defines the canonical
    shape used by higher-level components (AgentLoop, monitoring, etc.).
    """
    failure_type: Optional[str]
    severity: Optional[str]
    fix_suggestions: List[str]
    notes: str
    raw_text: str
    error: Optional[str] = None


@dataclass
class ErrorModelResponse:
    """
    Structured response from the ErrorModel when analyzing post-execution
    outcomes.

    This mirrors CriticResponse where possible so that AgentLoop and
    learning/monitoring code can treat the two models symmetrically.
    """
    failure_type: Optional[str]
    severity: Optional[str]
    fix_suggestions: List[str]
    notes: str
    raw_text: str
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Legacy / M1 interfaces (Planner / Code / Critic)
# ---------------------------------------------------------------------------

class PlannerModel(Protocol):
    """High-level planner interface (M1 legacy)."""

    def plan(
        self,
        observation: Observation,
        goal: str,
        skill_descriptions: Dict[str, Dict[str, Any]],
        constraints: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Given an observation and goal, return a structured plan.

        The exact structure is left to the implementation, but typically:
        {
          "steps": [...],
          "raw_text": "...",
          ...
        }
        """
        ...


class CodeModel(Protocol):
    """Code / skill implementation model interface (M1 legacy)."""

    def propose_skill_implementation(
        self,
        skill_spec: Dict[str, Any],
        examples: List[Dict[str, Any]],
    ) -> str:
        """
        Given a skill specification and optional example traces,
        return Python code implementing the skill.
        """
        ...


class CriticModel(Protocol):
    """
    Plan / behavior critic interface.

    NOTE:
        The older M1-style interface used `evaluate_plan(observation, plan, virtues)`.
        Q1 upgrades this to a more general payload-based API:

            critic_payload = {
                "plan": {...},
                "observation": {...},
                "virtue_scores": {...},
                "context": {...},
            }
            result: Dict[str, Any] = critic.evaluate(critic_payload)

        Implementations may also expose a convenience helper `call_critic(...)`
        but the primary contract is `evaluate(payload) -> Dict[str, Any]`
        with the CriticResponse shape.
    """

    def evaluate(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate a plan + context payload and return structured feedback.

        Payload MUST contain at least:
            - "plan": dict
            - "observation": dict

        It MAY also contain:
            - "virtue_scores": dict[str, float]
            - "context": dict

        Implementations are expected to return a dict matching CriticResponse
        fields where possible:
            {
              "failure_type": Optional[str],
              "severity": Optional[str],
              "fix_suggestions": List[str],
              "notes": str,
              "raw_text": str,
              # optional:
              "error": str,
            }
        """
        ...

    def call_critic(
        self,
        plan: Dict[str, Any],
        world_summary: Dict[str, Any],
        virtues_hint: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Optional convenience helper for higher-level callers that prefer to
        pass `plan` + `world_summary` (+ optional virtue scores) instead of
        assembling the full payload by hand.

        Implementations SHOULD internally call `evaluate(...)`.
        """
        ...


# ---------------------------------------------------------------------------
# M2 interfaces (single-model multi-role: PlanCode / Error / Scribe)
# ---------------------------------------------------------------------------

class PlanCodeModel(Protocol):
    """
    Unified planning + codegen interface over a single underlying model.

    Implemented by llm_stack.plan_code.PlanCodeModelImpl.
    """

    def plan(
        self,
        observation: Observation,
        goal: str,
        skill_descriptions: Dict[str, Dict[str, Any]],
        constraints: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Produce a structured plan dict.

        M2 convention (PlanCodeModelImpl):
        {
          "steps": [
            {"skill": "name", "params": {...}},
            ...
          ],
          "notes": "text",
          "raw_text": "raw LLM output"
        }
        """
        ...

    def propose_skill_implementation(
        self,
        skill_spec: Dict[str, Any],
        examples: List[Dict[str, Any]],
    ) -> str:
        """
        Produce Python code implementing the given skill.

        Implementations are free to return additional metadata elsewhere,
        but this interface returns just the code string.
        """
        ...


class ErrorModel(Protocol):
    """
    Error analysis / recovery interface.

    Q1 distinguishes two major responsibilities:

        1) Agent-level outcome evaluation (post-execution):
            - `evaluate(payload) -> Dict[str, Any]`
            - `call_error_model(episode_trace, plan) -> Dict[str, Any]`

        2) Low-level LLM-call failure analysis (infra / tooling):
            - `analyze_failure(ctx) -> ErrorAnalysis`

    To avoid circular imports, we type the arguments as Any. Concrete
    dataclasses live in llm_stack.schema:
      - ErrorContext
      - ErrorAnalysis
    """

    def evaluate(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a plan + outcome payload and return structured feedback.

        Payload SHOULD contain:
            - "plan": dict
            - "observation": dict

        It MAY also contain:
            - "trace": dict or serializable object
            - "virtue_scores": dict[str, float]
            - "context": dict

        Implementations are expected to return a dict matching
        ErrorModelResponse fields where possible:
            {
              "failure_type": Optional[str],
              "severity": Optional[str],
              "fix_suggestions": List[str],
              "notes": str,
              "raw_text": str,
              # optional:
              "error": str,
            }
        """
        ...

    def call_error_model(self, episode_trace: Any, plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optional convenience helper for agent-level callers.

        Implementations SHOULD:
            - extract observation / trace details from `episode_trace`
            - build a payload
            - delegate to `evaluate(payload)`
        """
        ...

    def analyze_failure(self, ctx: Any) -> Any:
        """
        Analyze a low-level failure context (typically from LLM infra) and
        return structured analysis describing:

        - classification (short label)
        - summary (human-readable)
        - suggested_fix (machine-usable hints)
        - retry_advised (bool)

        The concrete types are defined in llm_stack.schema as:
            - ErrorContext
            - ErrorAnalysis
        """
        ...


class ScribeModel(Protocol):
    """
    Log/trace summarization interface.

    Concrete request/response types live in llm_stack.schema:
      - TraceSummaryRequest
      - TraceSummaryResponse
    """

    def summarize_trace(self, req: Any) -> Any:
        """
        Summarize a trace/log into a compressed representation suitable for:

        - human documentation
        - long-term context storage
        - debugging

        Implementations typically return:
          TraceSummaryResponse(summary, keywords, suggested_tags, raw_text)
        """
        ...

```



### 5. Virtue Integration (M4)

**Files:**

- `src/virtues/evaluator.py` (or similar)
    
- `src/virtues/schema.py`

```python
# path: src/virtues/evaluator.py

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from .schema import VirtueConfig, PlanSummary, PlanScore

logger = logging.getLogger(__name__)

try:
    # Preferred: a single scoring entrypoint defined in metrics.py
    from .metrics import compute_plan_score  # type: ignore
except Exception:  # pragma: no cover - defensive import
    compute_plan_score = None  # type: ignore[misc]


def _coerce_plan_summary(plan: Any, plan_id_fallback: str = "plan") -> PlanSummary:
    """
    Best-effort conversion of an arbitrary object into a PlanSummary.

    This is intentionally forgiving so that M8 / M3 / M11 can pass in either:
      - a PlanSummary instance (ideal)
      - a dict with numeric feature fields
      - a dict-like object with .get()

    Any missing numeric fields default to 0.0, and context_features defaults
    to an empty dict.
    """
    if isinstance(plan, PlanSummary):
        return plan

    if not isinstance(plan, dict):
        # Try to treat it like an object with attributes or .__dict__
        if hasattr(plan, "__dict__"):
            plan = dict(plan.__dict__)
        else:
            logger.warning(
                "virtues._coerce_plan_summary received unsupported plan type %r; "
                "using zeroed PlanSummary.",
                type(plan),
            )
            plan = {}

    # Allow id under multiple common keys
    plan_id = (
        str(plan.get("id"))
        if plan.get("id") is not None
        else str(plan.get("plan_id"))
        if plan.get("plan_id") is not None
        else plan_id_fallback
    )

    def _num(key: str) -> float:
        v = plan.get(key, 0.0)
        try:
            return float(v)
        except Exception:
            return 0.0

    context_features = plan.get("context_features") or {}
    if not isinstance(context_features, dict):
        context_features = {}

    return PlanSummary(
        id=plan_id,
        time_cost=_num("time_cost"),
        resource_cost=_num("resource_cost"),
        risk_level=_num("risk_level"),
        pollution_level=_num("pollution_level"),
        infra_reuse_score=_num("infra_reuse_score"),
        infra_impact_score=_num("infra_impact_score"),
        novelty_score=_num("novelty_score"),
        aesthetic_score=_num("aesthetic_score"),
        complexity_score=_num("complexity_score"),
        tech_progress_score=_num("tech_progress_score"),
        stability_score=_num("stability_score"),
        reversibility_score=_num("reversibility_score"),
        context_features=context_features,
    )


def _score_with_builtin_logic(
    plan_summary: PlanSummary,
    virtue_config: VirtueConfig,
    context_id: str,
) -> PlanScore:
    """
    Extremely simple fallback scoring logic used if virtues.metrics
    does not expose `compute_plan_score`.

    This is NOT meant to be clever; it's just a reasonable default so that
    callers always get something structured back.

    Strategy:
        - overall_score: heuristic combining tech_progress, stability,
          reversibility, inverse risk/resource/pollution.
        - derived_virtues: a single "prudence" and "ambition" as examples.
        - allowed: True unless risk_level is extremely high.
    """
    # Heuristic weights; these can be tuned or replaced by metrics.compute_plan_score
    tech = plan_summary.tech_progress_score
    stability = plan_summary.stability_score
    reversibility = plan_summary.reversibility_score
    risk = plan_summary.risk_level
    resource = plan_summary.resource_cost
    pollution = plan_summary.pollution_level

    # Normalize a bit so it doesn't go completely insane
    overall = (
        1.5 * tech
        + 1.0 * stability
        + 0.5 * reversibility
        - 1.0 * risk
        - 0.5 * resource
        - 0.5 * pollution
    )

    derived_virtues: Dict[str, float] = {
        "prudence": max(0.0, 1.0 - risk / 10.0),
        "ambition": max(0.0, min(1.0, tech / 10.0)),
    }

    allowed = risk < 8.0
    disallowed_reason: Optional[str] = None
    if not allowed:
        disallowed_reason = "risk_level_too_high"

    # Node-level detail is not computed here; we keep it minimal.
    node_scores: Dict[str, Any] = {}

    return PlanScore(
        plan_id=plan_summary.id,
        context_id=context_id,
        node_scores=node_scores,               # type: ignore[arg-type]
        derived_virtues=derived_virtues,
        overall_score=overall,
        allowed=allowed,
        disallowed_reason=disallowed_reason,
    )


def evaluate_plan_with_virtues(
    plan: Any,
    world_state: Any,
    virtue_config: VirtueConfig,
    context_id: str = "default",
) -> Dict[str, float]:
    """
    High-level entrypoint for virtue evaluation used by M8 / AgentLoop.

    This function is intentionally simple from the caller's perspective:

        scores = evaluate_plan_with_virtues(plan, world_state, virtue_config)

    Inputs:
        plan:
            Either:
              - a PlanSummary instance (preferred), or
              - a dict with numeric feature fields as defined in PlanSummary.

        world_state:
            Reserved for future use (e.g. tech band, biome, danger level).
            The current implementation does not inspect it directly; any
            required world-derived features should already be baked into
            PlanSummary.context_features by M3 / semantics.

        virtue_config:
            Loaded VirtueConfig describing the lattice, contexts, and derived
            virtues.

        context_id:
            Which VirtueContext to use (e.g. "eco_factory", "speedrun").
            If not found in virtue_config.contexts, a default weighting is
            assumed by the scoring function.

    Returns:
        Dict[str, float] mapping:
            virtue_id -> score

        Typically this is:
            - `PlanScore.derived_virtues`
        but it may also include an "overall" entry when convenient.
    """
    del world_state  # currently unused; kept for signature stability

    plan_summary = _coerce_plan_summary(plan)

    # Preferred path: delegate to virtues.metrics.compute_plan_score
    if compute_plan_score is not None:
        try:
            plan_score: PlanScore = compute_plan_score(
                plan_summary=plan_summary,
                virtue_config=virtue_config,
                context_id=context_id,
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(
                "virtues.evaluate_plan_with_virtues: compute_plan_score failed: %r; "
                "falling back to builtin heuristic.",
                exc,
            )
            plan_score = _score_with_builtin_logic(plan_summary, virtue_config, context_id)
    else:
        plan_score = _score_with_builtin_logic(plan_summary, virtue_config, context_id)

    scores: Dict[str, float] = dict(plan_score.derived_virtues)
    # Optionally expose overall score under a standard key
    scores.setdefault("overall", plan_score.overall_score)
    return scores

```

```python
# path: src/virtues/schema.py

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class VirtueNode:
    id: str
    label: str
    role: str
    tier: str
    pillar: str


@dataclass
class VirtueEdge:
    source: str
    target: str


@dataclass
class VirtueContext:
    id: str
    description: str
    node_weights: Dict[str, float]
    hard_constraints: List[Dict[str, str]]  # simple condition refs for now


@dataclass
class DerivedVirtueSpec:
    id: str
    from_nodes: Dict[str, float]
    description: str


@dataclass
class VirtueConfig:
    """
    Canonical virtue configuration loaded from YAML.

    Fields:
        nodes:
            Mapping from node_id → VirtueNode, representing the base virtue
            lattice (e.g. the 10 sefirot plus any auxiliary nodes).

        edges:
            Directed edges in the lattice, typically used for propagation.

        features:
            Raw mapping from YAML describing how plan features map to nodes.
            Concrete interpretation lives in virtues.metrics.

        contexts:
            Named VirtueContext objects describing different weighting schemes
            (e.g. "factory_safety", "speedrun", "eco_factory").

        derived_virtues:
            Aggregate virtue definitions (e.g. "prudence" from multiple nodes).
    """
    nodes: Dict[str, VirtueNode]
    edges: List[VirtueEdge]
    features: Dict[str, Dict]          # raw mapping from YAML
    contexts: Dict[str, VirtueContext]
    derived_virtues: Dict[str, DerivedVirtueSpec]


@dataclass
class PlanSummary:
    """Compact numeric representation of a plan, post-semantics."""
    id: str
    time_cost: float
    resource_cost: float
    risk_level: float
    pollution_level: float
    infra_reuse_score: float
    infra_impact_score: float
    novelty_score: float
    aesthetic_score: float
    complexity_score: float
    tech_progress_score: float
    stability_score: float
    reversibility_score: float
    # extra context features for constraints / context node, etc.
    context_features: Dict[str, float]


@dataclass
class NodeScore:
    node_id: str
    raw: float
    propagated: float
    rationale: str


@dataclass
class PlanScore:
    """
    Full virtue evaluation for a single plan in a specific context.

    Higher-level modules (AgentLoop, curriculum, learning) usually only
    care about:
        - derived_virtues: dict[str, float]
        - overall_score: float
        - allowed / disallowed_reason
    """
    plan_id: str
    context_id: str
    node_scores: Dict[str, NodeScore]
    derived_virtues: Dict[str, float]
    overall_score: float
    allowed: bool
    disallowed_reason: Optional[str]

```

### 6. Monitoring & Events (M9)

**Files:**

- `src/monitoring/events.py`
    
- `src/monitoring/integration.py`
    
In M8:

- Emit `PlanEvaluated` after each `evaluate_plan`.
    
- Emit `PlanRetried` when `maybe_retry_plan` returns True.
    
- Emit `PlanAbandoned` when evaluation decides not to retry.
    
- Emit `PlanOutcomeEvaluated` at the end of `evaluate_outcome`.
    

These feed dashboards / logs and later can feed learning.


```python
# path: src/monitoring/events.py
"""
Event and command schemas for M9 – monitoring_and_tools.

This module defines:
- MonitoringEvent (structured system events)
- EventType enum
- ControlCommandType enum
- ControlCommand for human/system-issued controls

All events are JSON-serializable via `.to_dict()` and are intended
for use with monitoring.bus.EventBus and monitoring.logger.JsonFileLogger.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, asdict
from enum import Enum, auto
from typing import Any, Dict, Optional


# ============================================================
# Event Types
# ============================================================

class EventType(Enum):
    """Typed monitoring events emitted throughout the agent system."""

    # Agent lifecycle phases (Idle, Planning, Executing, Recovering)
    AGENT_PHASE_CHANGE = auto()

    # Planner & plan structure
    PLAN_CREATED = auto()
    PLAN_STEP_EXECUTED = auto()
    PLAN_FAILED = auto()

    # Self-evaluation + retry loop (Q1.1)
    PLAN_EVALUATED = auto()          # pre-execution evaluation for an attempt
    PLAN_RETRIED = auto()            # a plan attempt was rejected and retried
    PLAN_ABANDONED = auto()          # plan was abandoned (no further retries)
    PLAN_OUTCOME_EVALUATED = auto()  # post-execution outcome evaluation

    # Low-level action execution (BotCore)
    ACTION_EXECUTED = auto()

    # World / semantics updates
    TECH_STATE_UPDATED = auto()

    # Virtue engine scoring
    VIRTUE_SCORES = auto()

    # Critic analysis
    CRITIC_RESULT = auto()

    # Full state snapshot (rare, expensive)
    SNAPSHOT = auto()

    # Control surface events
    CONTROL_COMMAND = auto()

    # Generic log messages
    LOG = auto()


# ============================================================
# Monitoring Event Structure
# ============================================================

@dataclass
class MonitoringEvent:
    """
    Runtime event emitted by the agent, planner, critic, BotCore,
    observation pipeline, or the control surface.

    All fields must be JSON-safe.
    """

    ts: float                   # UNIX timestamp (seconds)
    module: str                 # Source module string ("M8.agent_loop", "bot_core", etc.)
    event_type: EventType       # Enum describing the event class
    message: str                # Short human-readable description
    payload: Dict[str, Any]     # Structured data (plan, action, tech state, scores)
    correlation_id: Optional[str] = None  # Used for grouping events per plan/episode

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a JSON-safe dict for loggers."""
        data = asdict(self)
        data["event_type"] = self.event_type.name  # store name, not enum
        return data


# ============================================================
# Control Commands
# ============================================================

class ControlCommandType(Enum):
    """
    Commands that humans or automated tools can send to control the agent loop.
    """

    PAUSE = auto()          # Freeze AgentLoop
    RESUME = auto()         # Unpause loop
    SINGLE_STEP = auto()    # Execute exactly one planning/execution iteration
    CANCEL_PLAN = auto()    # Drop current plan
    SET_GOAL = auto()       # Set a new goal in the AgentLoop
    DUMP_STATE = auto()     # Emit a full snapshot event


@dataclass
class ControlCommand:
    """
    Represents an external command for the agent.

    Sent through EventBus.publish_command(), then interpreted by
    monitoring.controller.AgentController.
    """

    cmd: ControlCommandType             # The specific command
    args: Dict[str, Any]                # Additional arguments for command execution

    @staticmethod
    def pause() -> "ControlCommand":
        return ControlCommand(ControlCommandType.PAUSE, {})

    @staticmethod
    def resume() -> "ControlCommand":
        return ControlCommand(ControlCommandType.RESUME, {})

    @staticmethod
    def single_step() -> "ControlCommand":
        return ControlCommand(ControlCommandType.SINGLE_STEP, {})

    @staticmethod
    def cancel_plan() -> "ControlCommand":
        return ControlCommand(ControlCommandType.CANCEL_PLAN, {})

    @staticmethod
    def set_goal(goal: str) -> "ControlCommand":
        return ControlCommand(ControlCommandType.SET_GOAL, {"goal": goal})

    @staticmethod
    def dump_state() -> "ControlCommand":
        return ControlCommand(ControlCommandType.DUMP_STATE, {})

```




```python
# path: src/monitoring/integration.py
"""
Integration helpers for M9 – monitoring_and_tools.

This module provides convenience functions for emitting well-structured
MonitoringEvents from:

- M8 AgentLoop (phases, plans, steps, critic, virtues, experiences)
- M7 Observation encoding (planner payloads, trace structure)
- M6 BotCore (observations, actions)
- M3/M4 Semantics & Virtues (TechState inference, virtue scoring)

All functions are thin wrappers around monitoring.logger.log_event
and enforce consistent payload shapes across the codebase.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .bus import EventBus
from .events import EventType
from .logger import log_event


JsonDict = Dict[str, Any]


# ============================================================
# M8 – AgentLoop integration
# ============================================================

def emit_agent_phase_change(
    bus: EventBus,
    phase: str,
    episode_id: Optional[str] = None,
    context_id: Optional[str] = None,
) -> None:
    """
    Emit an AGENT_PHASE_CHANGE event when AgentLoop changes phase.

    Expected phases: "IDLE", "PLANNING", "EXECUTING", "RECOVERING", etc.
    """
    payload: JsonDict = {
        "phase": phase,
        "episode_id": episode_id,
        "context_id": context_id,
    }
    log_event(
        bus=bus,
        module="M8.agent_loop",
        event_type=EventType.AGENT_PHASE_CHANGE,
        message=f"Agent phase changed to {phase}",
        payload=payload,
        correlation_id=episode_id,
    )


def emit_plan_created(
    bus: EventBus,
    plan: JsonDict,
    goal: Optional[str],
    episode_id: str,
    context_id: Optional[str] = None,
) -> None:
    """
    Emit a PLAN_CREATED event when the planner returns a plan.

    `plan` is expected to be the raw planner output (JSON-like).
    """
    steps = plan.get("steps") or []
    payload: JsonDict = {
        "plan": plan,
        "goal": goal or "",
        "step_count": len(steps),
        "episode_id": episode_id,
        "context_id": context_id,
    }
    log_event(
        bus=bus,
        module="M8.agent_loop",
        event_type=EventType.PLAN_CREATED,
        message="New plan created",
        payload=payload,
        correlation_id=episode_id,
    )


def emit_plan_step_executed(
    bus: EventBus,
    episode_id: str,
    step_index: int,
    step_spec: JsonDict,
    trace_step_result: JsonDict,
    context_id: Optional[str] = None,
) -> None:
    """
    Emit a PLAN_STEP_EXECUTED event when a single plan step has been run.

    `step_spec` is the planner's step spec.
    `trace_step_result` should be a JSON-like representation of TraceStep.
    """
    payload: JsonDict = {
        "episode_id": episode_id,
        "context_id": context_id,
        "step_index": step_index,
        "step_spec": step_spec,
        "trace_step": trace_step_result,
    }
    log_event(
        bus=bus,
        module="M8.agent_loop",
        event_type=EventType.PLAN_STEP_EXECUTED,
        message=f"Plan step executed (index={step_index})",
        payload=payload,
        correlation_id=episode_id,
    )


def emit_plan_failed(
    bus: EventBus,
    episode_id: str,
    reason: str,
    failing_step_index: Optional[int] = None,
    context_id: Optional[str] = None,
) -> None:
    """
    Emit a PLAN_FAILED event when AgentLoop aborts execution of a plan.
    """
    payload: JsonDict = {
        "episode_id": episode_id,
        "context_id": context_id,
        "reason": reason,
        "step_index": failing_step_index,
    }
    log_event(
        bus=bus,
        module="M8.agent_loop",
        event_type=EventType.PLAN_FAILED,
        message=f"Plan failed: {reason}",
        payload=payload,
        correlation_id=episode_id,
    )


def emit_plan_evaluated(
    bus: EventBus,
    episode_id: str,
    plan_id: str,
    attempt_index: int,
    virtue_scores: Dict[str, float],
    failure_type: Optional[str],
    severity: Optional[str],
    context_id: Optional[str] = None,
) -> None:
    """
    Emit a PLAN_EVALUATED event after a pre-execution evaluation of a plan
    attempt (Critic + virtues).

    This should be called once per call to AgentLoop._evaluate_plan.
    """
    payload: JsonDict = {
        "episode_id": episode_id,
        "context_id": context_id,
        "plan_id": plan_id,
        "attempt_index": attempt_index,
        "virtue_scores": virtue_scores,
        "failure_type": failure_type,
        "severity": severity,
    }
    log_event(
        bus=bus,
        module="M8.agent_loop",
        event_type=EventType.PLAN_EVALUATED,
        message=f"Plan evaluated (attempt={attempt_index}, plan_id={plan_id})",
        payload=payload,
        correlation_id=episode_id,
    )


def emit_plan_retried(
    bus: EventBus,
    episode_id: str,
    plan_id: str,
    attempt_index: int,
    reason: str,
    remaining_budget: int,
    context_id: Optional[str] = None,
) -> None:
    """
    Emit a PLAN_RETRIED event when the self-eval loop decides to retry
    planning instead of executing the current attempt.
    """
    payload: JsonDict = {
        "episode_id": episode_id,
        "context_id": context_id,
        "plan_id": plan_id,
        "attempt_index": attempt_index,
        "reason": reason,
        "remaining_retry_budget": remaining_budget,
    }
    log_event(
        bus=bus,
        module="M8.agent_loop",
        event_type=EventType.PLAN_RETRIED,
        message=f"Plan retry scheduled (attempt={attempt_index}, plan_id={plan_id})",
        payload=payload,
        correlation_id=episode_id,
    )


def emit_plan_abandoned(
    bus: EventBus,
    episode_id: str,
    plan_id: str,
    attempt_index: int,
    reason: str,
    context_id: Optional[str] = None,
) -> None:
    """
    Emit a PLAN_ABANDONED event when the self-eval loop decides that
    planning should stop without executing the current attempt.
    """
    payload: JsonDict = {
        "episode_id": episode_id,
        "context_id": context_id,
        "plan_id": plan_id,
        "attempt_index": attempt_index,
        "reason": reason,
    }
    log_event(
        bus=bus,
        module="M8.agent_loop",
        event_type=EventType.PLAN_ABANDONED,
        message=f"Plan abandoned (attempt={attempt_index}, plan_id={plan_id})",
        payload=payload,
        correlation_id=episode_id,
    )


def emit_plan_outcome_evaluated(
    bus: EventBus,
    episode_id: str,
    plan_id: str,
    attempt_index: int,
    outcome: Dict[str, Any],
    context_id: Optional[str] = None,
) -> None:
    """
    Emit a PLAN_OUTCOME_EVALUATED event at the end of post-execution
    outcome evaluation (ErrorModel).

    `outcome` is expected to follow the ErrorModelResponse/CriticResponse-like
    dict shape:
        {
          "failure_type": ...,
          "severity": ...,
          "fix_suggestions": [...],
          "notes": "...",
          "raw_text": "...",
          ...
        }
    """
    payload: JsonDict = {
        "episode_id": episode_id,
        "context_id": context_id,
        "plan_id": plan_id,
        "attempt_index": attempt_index,
        "outcome": outcome,
    }
    log_event(
        bus=bus,
        module="M8.agent_loop",
        event_type=EventType.PLAN_OUTCOME_EVALUATED,
        message=f"Plan outcome evaluated (attempt={attempt_index}, plan_id={plan_id})",
        payload=payload,
        correlation_id=episode_id,
    )


def emit_critic_result(
    bus: EventBus,
    episode_id: str,
    critic_result: JsonDict,
    context_id: Optional[str] = None,
) -> None:
    """
    Emit a CRITIC_RESULT event after the critic evaluates the plan/trace.
    """
    payload: JsonDict = {
        "episode_id": episode_id,
        "context_id": context_id,
        "critic_result": critic_result,
    }
    log_event(
        bus=bus,
        module="M8.agent_loop",
        event_type=EventType.CRITIC_RESULT,
        message="Critic evaluation completed",
        payload=payload,
        correlation_id=episode_id,
    )


def emit_virtue_scores(
    bus: EventBus,
    episode_id: str,
    scores: Dict[str, Any],
    context_id: Optional[str] = None,
) -> None:
    """
    Emit a VIRTUE_SCORES event when VirtueEngine returns scores for a trace.
    """
    payload: JsonDict = {
        "episode_id": episode_id,
        "context_id": context_id,
        "scores": scores,
    }
    log_event(
        bus=bus,
        module="M8.agent_loop",
        event_type=EventType.VIRTUE_SCORES,
        message="Virtue scores computed for episode",
        payload=payload,
        correlation_id=episode_id,
    )


def emit_experience_recorded(
    bus: EventBus,
    experience_id: str,
    episode_id: str,
    env_profile_name: str,
    context_id: Optional[str],
    meta: JsonDict,
) -> None:
    """
    Emit a LOG event marking that an Experience has been added to the buffer.

    We don't define a dedicated EventType for this (yet); instead we rely on
    LOG + subtype for compatibility with the existing EventType enum.
    """
    payload: JsonDict = {
        "subtype": "EXPERIENCE_RECORDED",
        "experience_id": experience_id,
        "episode_id": episode_id,
        "env_profile_name": env_profile_name,
        "context_id": context_id,
        "meta": meta,
    }
    log_event(
        bus=bus,
        module="M8.agent_loop",
        event_type=EventType.LOG,
        message="Experience recorded",
        payload=payload,
        correlation_id=episode_id,
    )


# ============================================================
# M7 – Observation Encoding integration
# ============================================================

def emit_planner_observation_snapshot(
    bus: EventBus,
    episode_id: str,
    planner_payload: JsonDict,
    tech_state: Optional[JsonDict],
    context_id: Optional[str] = None,
) -> None:
    """
    Emit a SNAPSHOT event for the planner-side observation payload.

    Intended to be called just after the observation encoder builds
    the planner input payload.
    """
    payload: JsonDict = {
        "episode_id": episode_id,
        "context_id": context_id,
        "planner_payload": planner_payload,
        "tech_state": tech_state or {},
    }
    log_event(
        bus=bus,
        module="M7.observation",
        event_type=EventType.SNAPSHOT,
        message="Planner observation snapshot",
        payload=payload,
        correlation_id=episode_id,
    )


def emit_trace_structure_snapshot(
    bus: EventBus,
    episode_id: str,
    trace_summary: JsonDict,
    context_id: Optional[str] = None,
) -> None:
    """
    Emit a SNAPSHOT event describing the overall structure of a PlanTrace.

    `trace_summary` should be a compressed/summary representation of:
    - steps count
    - key timestamps
    - failure flags
    - any other relevant structural info
    """
    payload: JsonDict = {
        "episode_id": episode_id,
        "context_id": context_id,
        "trace": trace_summary,
    }
    log_event(
        bus=bus,
        module="M7.observation",
        event_type=EventType.SNAPSHOT,
        message="Trace structure snapshot",
        payload=payload,
        correlation_id=episode_id,
    )


# ============================================================
# M6 – BotCore integration
# ============================================================

def emit_observation_metadata(
    bus: EventBus,
    world_meta: JsonDict,
    context_id: Optional[str] = None,
) -> None:
    """
    Emit a LOG event summarizing the latest RawWorldSnapshot / WorldState.

    `world_meta` is expected to contain:
    - dimension
    - position
    - loaded_chunk_count
    - entity_counts, etc.
    """
    payload: JsonDict = {
        "subtype": "OBSERVATION_METADATA",
        "world": world_meta,
        "context_id": context_id,
    }
    log_event(
        bus=bus,
        module="M6.bot_core",
        event_type=EventType.LOG,
        message="Observation metadata",
        payload=payload,
        correlation_id=None,
    )


def emit_action_executed_from_botcore(
    bus: EventBus,
    action_type: str,
    params: JsonDict,
    success: bool,
    error: Optional[str],
    context_id: Optional[str] = None,
) -> None:
    """
    Emit an ACTION_EXECUTED event from BotCore when an action completes.

    This is lower-level than PLAN_STEP_EXECUTED and is focused on the
    BotCore action primitives.
    """
    payload: JsonDict = {
        "context_id": context_id,
        "action_type": action_type,
        "params": params,
        "success": success,
        "error": error,
    }
    log_event(
        bus=bus,
        module="M6.bot_core",
        event_type=EventType.ACTION_EXECUTED,
        message=f"Action executed: {action_type}",
        payload=payload,
        correlation_id=None,
    )


def emit_action_failure_short(
    bus: EventBus,
    action_type: str,
    reason: str,
    context_id: Optional[str] = None,
) -> None:
    """
    Emit a PLAN_FAILED-style LOG event for low-level action failures that
    don't map cleanly to a specific episode/step.
    """
    payload: JsonDict = {
        "subtype": "ACTION_FAILURE",
        "action_type": action_type,
        "reason": reason,
        "context_id": context_id,
    }
    log_event(
        bus=bus,
        module="M6.bot_core",
        event_type=EventType.LOG,
        message=f"Action failure: {action_type}",
        payload=payload,
        correlation_id=None,
    )


# ============================================================
# M3 & M4 – Semantics & Virtues integration
# ============================================================

def emit_tech_state_updated(
    bus: EventBus,
    tech_state_dict: JsonDict,
    episode_id: Optional[str] = None,
    context_id: Optional[str] = None,
) -> None:
    """
    Emit a TECH_STATE_UPDATED event when semantics.tech_state infers a new
    TechState for the agent.
    """
    payload: JsonDict = {
        "tech_state": tech_state_dict,
        "episode_id": episode_id,
        "context_id": context_id,
    }
    log_event(
        bus=bus,
        module="M3.semantics.tech_state",
        event_type=EventType.TECH_STATE_UPDATED,
        message="Tech state inferred/updated",
        payload=payload,
        correlation_id=episode_id,
    )


def emit_virtue_scores_from_engine(
    bus: EventBus,
    scores: Dict[str, Any],
    trace_meta: JsonDict,
    episode_id: Optional[str] = None,
    context_id: Optional[str] = None,
) -> None:
    """
    Emit a VIRTUE_SCORES event specifically from virtues.lattice / metrics,
    including trace metadata if available.
    """
    payload: JsonDict = {
        "scores": scores,
        "trace_meta": trace_meta,
        "episode_id": episode_id,
        "context_id": context_id,
    }
    log_event(
        bus=bus,
        module="M4.virtues",
        event_type=EventType.VIRTUE_SCORES,
        message="Virtue scores computed",
        payload=payload,
        correlation_id=episode_id,
    )

```


### 7. Experience & Learning Hooks (M10)

**Files:**

- `src/learning/buffer.py`
    
- `src/learning/schema.py`
    
- `src/learning/manager.py`
    

At the end of an episode, right after `evaluate_outcome`:

- Build an `Experience` object (ties into Q1.4):
    
    - Include:
        
        - `goal`
            
        - `plan`
            
        - `pre_eval` + `post_eval`
            
        - `final_outcome`
            
        - `virtue_scores`
            
        - `failure_type` / `severity`
            
- Call `append_experience(experience)`.
    

Later, M10 can:

- Use `failure_type` & `virtue_scores` to:
    
    - Improve planning prompts.
        
    - Adjust retry policies.
        
    - Update skill metrics (Quality 2).


```python
# path: src/learning/buffer.py

"""
M10 Experience Buffer

JSONL-backed storage for ExperienceEpisode objects.

Responsibilities:
- Persist each episode from M8 as one JSON object per line.
- Provide streaming access to raw dicts (for quick analysis).
- Provide typed access to ExperienceEpisode objects.
- Expose basic filter helpers by goal, success, skill usage, and tech tier.

This module does NOT:
- Decide when episodes are created (that’s AgentLoop + EpisodeRecorder).
- Perform clustering, synthesis, or evaluation (see synthesizer/evaluator/manager).

Q1 alignment:
- Each stored episode corresponds to a rich Experience that includes:
    - goal
    - plan
    - pre_eval / post_eval (critic + error model, etc.)
    - final_outcome
    - virtue_scores
    - failure_type / severity
  plus tech_state, trace, success, and metadata.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional

from .schema import ExperienceEpisode, EpisodeId, SkillName


# Type aliases for serializer hooks
TechStateToDict = Callable[[Any], Dict[str, Any]]
TechStateFromDict = Callable[[Dict[str, Any]], Any]
TraceToDict = Callable[[Any], Dict[str, Any]]
TraceFromDict = Callable[[Dict[str, Any]], Any]


class ExperienceBuffer:
    """
    JSONL-backed experience buffer.

    Each line in the file is a single JSON object representing one episode:
    {
      "id": "...",
      "goal": "...",
      "plan": {...},
      "pre_eval": {...},
      "post_eval": {...},
      "final_outcome": {...},
      "failure_type": "str or null",
      "severity": "low" | "medium" | "high" | null,
      "tech_state": {...},
      "trace": {...},
      "virtue_scores": {...},
      "success": true/false,
      "metadata": {...}
    }

    TechState and PlanTrace serialization is controlled by injected
    serializer/deserializer functions. This keeps M10 decoupled from the
    exact implementation details in M3/M7.
    """

    def __init__(
        self,
        path: Path,
        *,
        tech_state_to_dict: TechStateToDict,
        tech_state_from_dict: TechStateFromDict,
        trace_to_dict: TraceToDict,
        trace_from_dict: TraceFromDict,
    ) -> None:
        """
        Initialize an ExperienceBuffer backed by a JSONL file.

        Parameters
        ----------
        path:
            File path where episodes are stored.
        tech_state_to_dict:
            Function converting TechState -> dict.
        tech_state_from_dict:
            Function converting dict -> TechState.
        trace_to_dict:
            Function converting PlanTrace -> dict.
        trace_from_dict:
            Function converting dict -> PlanTrace.
        """
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if not self._path.exists():
            self._path.write_text("", encoding="utf-8")

        self._tech_state_to_dict = tech_state_to_dict
        self._tech_state_from_dict = tech_state_from_dict
        self._trace_to_dict = trace_to_dict
        self._trace_from_dict = trace_from_dict

    # ------------------------------------------------------------------
    # Core I/O
    # ------------------------------------------------------------------

    def append(self, episode: ExperienceEpisode) -> None:
        """
        Append one ExperienceEpisode to the buffer.

        Uses the serializer hooks to encode TechState and PlanTrace.
        """
        data = episode.to_dict(
            tech_state_to_dict=self._tech_state_to_dict,
            plan_trace_to_dict=self._trace_to_dict,
        )
        with self._path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")

    def append_experience(self, experience: ExperienceEpisode) -> None:
        """
        Q1-friendly alias used by AgentLoop / integration code.

        This is the canonical call-site for:
            - build ExperienceEpisode from:
                goal, plan, pre_eval, post_eval, final_outcome,
                virtue_scores, failure_type, severity, tech_state, trace, success
            - append_experience(experience)
        """
        self.append(experience)

    def load_all_raw(self) -> Iterator[Dict[str, Any]]:
        """
        Stream all stored episodes as raw dicts.

        This is useful for quick analysis or when you only care about
        a few fields and want to avoid reconstructing full objects.
        """
        with self._path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    # corrupted line; skip but don't blow up the whole buffer
                    continue

    def load_all(self) -> Iterator[ExperienceEpisode]:
        """
        Stream all stored episodes as ExperienceEpisode objects.

        Uses the deserializer hooks to reconstruct TechState and PlanTrace.
        """
        for raw in self.load_all_raw():
            yield ExperienceEpisode.from_dict(
                raw,
                tech_state_from_dict=self._tech_state_from_dict,
                plan_trace_from_dict=self._trace_from_dict,
            )

    # ------------------------------------------------------------------
    # Filter helpers (operate on raw dicts for efficiency)
    # ------------------------------------------------------------------

    def iter_by_goal_substring(
        self,
        substring: str,
        *,
        case_sensitive: bool = False,
    ) -> Iterator[Dict[str, Any]]:
        """
        Iterate over raw episodes whose goal contains a given substring.
        """
        if not case_sensitive:
            substring = substring.lower()

        for ep in self.load_all_raw():
            goal = ep.get("goal", "")
            target = goal if case_sensitive else goal.lower()
            if substring in target:
                yield ep

    def iter_by_success(
        self,
        success: bool,
    ) -> Iterator[Dict[str, Any]]:
        """
        Iterate over raw episodes filtered by success flag.
        """
        for ep in self.load_all_raw():
            if bool(ep.get("success")) is success:
                yield ep

    def iter_by_skill_usage(
        self,
        skill_name: SkillName,
    ) -> Iterator[Dict[str, Any]]:
        """
        Iterate over raw episodes where the given skill appears in the trace.

        Assumes the serialized trace has the structure:
        trace: {
          "steps": [
            {
              "meta": {
                "skill": "<skill_name>",
                ...
              },
              ...
            },
            ...
          ],
          ...
        }
        """
        for ep in self.load_all_raw():
            trace = ep.get("trace") or {}
            steps = trace.get("steps") or []
            used_here = any(
                isinstance(step, dict)
                and isinstance(step.get("meta"), dict)
                and step["meta"].get("skill") == skill_name
                for step in steps
            )
            if used_here:
                yield ep

    def iter_by_tech_tier(
        self,
        active_tier: str,
    ) -> Iterator[Dict[str, Any]]:
        """
        Iterate over raw episodes whose TechState active tier matches.

        Assumes serialized TechState has the structure:
        tech_state: {
          "active": "<tier>",
          ...
        }
        """
        for ep in self.load_all_raw():
            tech_state = ep.get("tech_state") or {}
            if tech_state.get("active") == active_tier:
                yield ep

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def count(self) -> int:
        """
        Return an approximate count of episodes (number of non-empty lines).

        This is O(n) over the file and is meant for diagnostics, not tight loops.
        """
        n = 0
        with self._path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    n += 1
        return n

    def path(self) -> Path:
        """
        Return the underlying file path. Mostly for tooling and tests.
        """
        return self._path

```


```python
# path: src/learning/schema.py

"""
Experience & skill-learning schema for M10.

This module defines the core data structures used by the learning layer:
- ExperienceEpisode: a single plan/episode as seen by M10
- SkillPerformanceStats: aggregated metrics for a single skill
- SkillCandidate: a proposed new or refined skill

The design assumes:
- TechState comes from `semantics.schema.TechState`
- PlanTrace comes from `observation.trace_schema.PlanTrace`
- Virtue scores are produced by M4 (virtues.*)

Q1 alignment:
- ExperienceEpisode now captures the richer Experience object requested by Q1:
    - goal
    - plan
    - pre_eval (pre-execution self-eval / critic)
    - post_eval (post-execution error model, etc.)
    - final_outcome (episode-level outcome summary)
    - virtue_scores
    - failure_type / severity
  in addition to tech_state, trace, success, and metadata.

All structures are:
- Plain dataclasses for easy testing
- JSON-friendly via `to_dict()`
- Reconstructable via `from_dict()` with optional adapters
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict, is_dataclass
from typing import Any, Dict, List, Optional, Literal, Callable, TypeVar, Generic
from datetime import datetime

# External schemas from existing modules
from semantics.schema import TechState          # M3 tech progression representation
from observation.trace_schema import PlanTrace  # M7 execution trace representation


# ---------------------------------------------------------------------------
# Type aliases & small helpers
# ---------------------------------------------------------------------------

VirtueScores = Dict[str, float]
EpisodeId = str
CandidateId = str
SkillName = str

SkillCandidateStatus = Literal[
    "proposed",    # just synthesized by M10
    "evaluating",  # being A/B tested or under review
    "accepted",    # promoted into main registry
    "rejected",    # discarded / archived
]

T = TypeVar("T")


def _maybe_to_dict(obj: Any) -> Any:
    """
    Best-effort conversion of arbitrary objects into JSON-friendly structures.

    Priority:
    1. If object has `to_dict()`, call it.
    2. If it's a dataclass, use `asdict()`.
    3. If it has __dict__, return that.
    4. Otherwise, return as-is (caller must ensure JSON-serializable).
    """
    if obj is None:
        return None

    to_dict = getattr(obj, "to_dict", None)
    if callable(to_dict):
        return to_dict()

    if is_dataclass(obj):
        return asdict(obj)

    if hasattr(obj, "__dict__"):
        return dict(obj.__dict__)

    return obj


def _datetime_to_iso(dt: Optional[datetime]) -> Optional[str]:
    if dt is None:
        return None
    return dt.isoformat()


def _datetime_from_iso(value: Optional[str]) -> Optional[datetime]:
    if value is None:
        return None
    return datetime.fromisoformat(value)


# ---------------------------------------------------------------------------
# Core episode schema
# ---------------------------------------------------------------------------

@dataclass
class EpisodeMetadata:
    """
    Auxiliary metadata for an episode.

    This intentionally mirrors what M9 / runtime already know:
    - timestamps
    - environment / profile IDs
    - tags (e.g., curriculum slice, skill pack, scenario)
    """
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    env_profile_id: Optional[str] = None       # link back to EnvProfile / env.yaml
    curriculum_id: Optional[str] = None        # which curriculum unit (M11) this came from
    skill_pack_id: Optional[str] = None        # which skill pack was active (e.g. "steam_age")
    tags: List[str] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "start_time": _datetime_to_iso(self.start_time),
            "end_time": _datetime_to_iso(self.end_time),
            "env_profile_id": self.env_profile_id,
            "curriculum_id": self.curriculum_id,
            "skill_pack_id": self.skill_pack_id,
            "tags": list(self.tags),
            "extra": dict(self.extra),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EpisodeMetadata":
        return cls(
            start_time=_datetime_from_iso(data.get("start_time")),
            end_time=_datetime_from_iso(data.get("end_time")),
            env_profile_id=data.get("env_profile_id"),
            curriculum_id=data.get("curriculum_id"),
            skill_pack_id=data.get("skill_pack_id"),
            tags=list(data.get("tags") or []),
            extra=dict(data.get("extra") or {}),
        )


@dataclass
class ExperienceEpisode:
    """
    A single learning episode derived from one plan execution in M8.

    This is the primary unit of experience that M10 consumes.

    Q1 Experience fields:
        - goal: text of the top-level goal for this episode
        - plan: structured plan dict from the planner
        - pre_eval: pre-execution evaluation (Critic / virtues, etc.)
        - post_eval: post-execution evaluation (ErrorModel, etc.)
        - final_outcome: high-level outcome summary for the episode
        - virtue_scores: dict[str, float] from VirtueEngine
        - failure_type / severity: rolled-up labels for easy filtering

    Plus:
        - tech_state: TechState at (or near) execution
        - trace: full PlanTrace for the episode
        - success: coarse success flag
        - metadata: EpisodeMetadata (timestamps, curriculum, tags, etc.)
    """
    id: EpisodeId
    goal: str
    plan: Dict[str, Any]
    pre_eval: Dict[str, Any]
    post_eval: Dict[str, Any]
    final_outcome: Dict[str, Any]
    tech_state: TechState
    trace: PlanTrace
    virtue_scores: VirtueScores
    success: bool
    failure_type: Optional[str] = None
    severity: Optional[str] = None
    metadata: EpisodeMetadata = field(default_factory=EpisodeMetadata)

    # ------------------------------------------------------------------
    # Serialization helpers
    # ------------------------------------------------------------------

    def to_dict(
        self,
        *,
        tech_state_to_dict: Optional[Callable[[TechState], Dict[str, Any]]] = None,
        plan_trace_to_dict: Optional[Callable[[PlanTrace], Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Convert this episode into a JSON-serializable dict.

        Callers can override how TechState and PlanTrace are serialized by
        passing `tech_state_to_dict` and `plan_trace_to_dict`.
        """
        if tech_state_to_dict is None:
            tech_state_to_dict = _maybe_to_dict  # type: ignore[assignment]
        if plan_trace_to_dict is None:
            plan_trace_to_dict = _maybe_to_dict  # type: ignore[assignment]

        return {
            "id": self.id,
            "goal": self.goal,
            "plan": _maybe_to_dict(self.plan),
            "pre_eval": _maybe_to_dict(self.pre_eval),
            "post_eval": _maybe_to_dict(self.post_eval),
            "final_outcome": _maybe_to_dict(self.final_outcome),
            "tech_state": tech_state_to_dict(self.tech_state),
            "trace": plan_trace_to_dict(self.trace),
            "virtue_scores": dict(self.virtue_scores),
            "success": self.success,
            "failure_type": self.failure_type,
            "severity": self.severity,
            "metadata": self.metadata.to_dict(),
        }

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
        *,
        tech_state_from_dict: Callable[[Dict[str, Any]], TechState],
        plan_trace_from_dict: Callable[[Dict[str, Any]], PlanTrace],
    ) -> "ExperienceEpisode":
        """
        Reconstruct an ExperienceEpisode from a dict.

        The caller must provide `tech_state_from_dict` and `plan_trace_from_dict`
        because M10 does not own those implementations.
        """
        return cls(
            id=data["id"],
            goal=data["goal"],
            plan=dict(data.get("plan") or {}),
            pre_eval=dict(data.get("pre_eval") or {}),
            post_eval=dict(data.get("post_eval") or {}),
            final_outcome=dict(data.get("final_outcome") or {}),
            tech_state=tech_state_from_dict(data["tech_state"]),
            trace=plan_trace_from_dict(data["trace"]),
            virtue_scores=dict(data.get("virtue_scores") or {}),
            success=bool(data["success"]),
            failure_type=data.get("failure_type"),
            severity=data.get("severity"),
            metadata=EpisodeMetadata.from_dict(data.get("metadata") or {}),
        )


# ---------------------------------------------------------------------------
# Aggregated performance metrics
# ---------------------------------------------------------------------------

@dataclass
class SkillPerformanceStats:
    """
    Aggregated statistics for a single skill across many episodes.

    This struct is computed by the evaluator and is used both for:
    - baseline skills (current registry entries)
    - candidate skills (after experimental rollout)
    """
    skill_name: SkillName
    uses: int
    success_rate: float
    avg_time: float
    avg_resource_cost: float
    avg_virtue_scores: VirtueScores = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "skill_name": self.skill_name,
            "uses": int(self.uses),
            "success_rate": float(self.success_rate),
            "avg_time": float(self.avg_time),
            "avg_resource_cost": float(self.avg_resource_cost),
            "avg_virtue_scores": dict(self.avg_virtue_scores),
        }

    @classmethod
    def zero(cls, skill_name: SkillName) -> "SkillPerformanceStats":
        """Convenience constructor for 'no data yet' stats."""
        return cls(
            skill_name=skill_name,
            uses=0,
            success_rate=0.0,
            avg_time=0.0,
            avg_resource_cost=0.0,
            avg_virtue_scores={},
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SkillPerformanceStats":
        return cls(
            skill_name=data["skill_name"],
            uses=int(data.get("uses", 0)),
            success_rate=float(data.get("success_rate", 0.0)),
            avg_time=float(data.get("avg_time", 0.0)),
            avg_resource_cost=float(data.get("avg_resource_cost", 0.0)),
            avg_virtue_scores=dict(data.get("avg_virtue_scores") or {}),
        )


# ---------------------------------------------------------------------------
# Skill candidates
# ---------------------------------------------------------------------------

@dataclass
class SkillCandidate:
    """
    A proposed new or refined skill produced by M10.

    This struct is the bridge between:
    - LLM-based synthesis (spec_yaml + impl_code + rationale)
    - evaluation (metrics_before / metrics_after)
    - lifecycle management (status)
    """
    id: CandidateId
    base_skill_name: Optional[SkillName]          # None if brand new, else refined skill
    spec_yaml: str                                # full YAML for SkillSpec
    impl_code: str                                # Python implementation stub
    rationale: str                                # explanation from synthesizer
    status: SkillCandidateStatus                  # lifecycle state

    metrics_before: Optional[SkillPerformanceStats] = None
    metrics_after: Optional[SkillPerformanceStats] = None

    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    tags: List[str] = field(default_factory=list)  # arbitrary labels (e.g., "steam_age", "LV")
    extra: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Lifecycle helpers
    # ------------------------------------------------------------------

    def mark_status(self, new_status: SkillCandidateStatus) -> None:
        """Update candidate status and bump updated_at."""
        self.status = new_status
        self.updated_at = datetime.utcnow()

    def touch(self) -> None:
        """Bump updated_at without changing status."""
        self.updated_at = datetime.utcnow()

    # ------------------------------------------------------------------
    # Serialization helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "base_skill_name": self.base_skill_name,
            "spec_yaml": self.spec_yaml,
            "impl_code": self.impl_code,
            "rationale": self.rationale,
            "status": self.status,
            "metrics_before": (
                self.metrics_before.to_dict() if self.metrics_before else None
            ),
            "metrics_after": (
                self.metrics_after.to_dict() if self.metrics_after else None
            ),
            "created_at": _datetime_to_iso(self.created_at),
            "updated_at": _datetime_to_iso(self.updated_at),
            "tags": list(self.tags),
            "extra": dict(self.extra),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SkillCandidate":
        metrics_before_raw = data.get("metrics_before")
        metrics_after_raw = data.get("metrics_after")

        return cls(
            id=data["id"],
            base_skill_name=data.get("base_skill_name"),
            spec_yaml=data.get("spec_yaml", ""),
            impl_code=data.get("impl_code", ""),
            rationale=data.get("rationale", ""),
            status=data.get("status", "proposed"),
            metrics_before=(
                SkillPerformanceStats.from_dict(metrics_before_raw)
                if metrics_before_raw
                else None
            ),
            metrics_after=(
                SkillPerformanceStats.from_dict(metrics_after_raw)
                if metrics_after_raw
                else None
            ),
            created_at=_datetime_from_iso(data.get("created_at")),
            updated_at=_datetime_from_iso(data.get("updated_at")),
            tags=list(data.get("tags") or []),
            extra=dict(data.get("extra") or {}),
        )


# ---------------------------------------------------------------------------
# Public exports
# ---------------------------------------------------------------------------

__all__ = [
    "VirtueScores",
    "EpisodeId",
    "CandidateId",
    "SkillName",
    "SkillCandidateStatus",
    "EpisodeMetadata",
    "ExperienceEpisode",
    "SkillPerformanceStats",
    "SkillCandidate",
]

```


```python
# path: src/learning/manager.py

"""
M10 SkillLearningManager

High-level orchestrator for skill learning:

Responsibilities
----------------
- Read episodes from ExperienceBuffer
- Filter/cluster by goal substring or skill pattern
- Invoke SkillSynthesizer for candidate creation
- Run SkillEvaluator for baseline metrics & recommendation
- Persist candidate artifacts under config/skills_candidates/
- Expose a simple "learning cycle" API that M11 or tools can call

Q1 context:
- ExperienceEpisode now includes pre_eval/post_eval/final_outcome/failure_type/severity.
  This manager doesn't strictly need those fields yet, but they are available for
  future learning policies (e.g., weighting failures differently, conditioning
  synthesis on specific failure_types, etc.).
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from .buffer import ExperienceBuffer
from .schema import ExperienceEpisode, SkillCandidate, SkillPerformanceStats, SkillName
from .synthesizer import SkillSynthesizer
from .evaluator import SkillEvaluator

from skills.registry import SkillRegistry  # M5 skill registry


class SkillLearningManager:
    """
    High-level orchestrator for M10 skill learning.

    Typical usage (offline or scheduled):

        manager = SkillLearningManager(
            buffer=experience_buffer,
            synthesizer=skill_synthesizer,
            evaluator=skill_evaluator,
            skills=skill_registry,
            candidates_dir=Path("config/skills_candidates"),
        )

        result = manager.run_learning_cycle_for_goal(
            goal_substring="maintain coke ovens",
            target_skill_name="maintain_coke_ovens",
            context_id="steam_age",
            tech_tier="steam_age",
        )

    """

    def __init__(
        self,
        buffer: ExperienceBuffer,
        synthesizer: SkillSynthesizer,
        evaluator: SkillEvaluator,
        skills: SkillRegistry,
        candidates_dir: Path,
        *,
        semantics_db: Any | None = None,
    ) -> None:
        """
        Parameters
        ----------
        buffer:
            ExperienceBuffer instance for reading episodes.
        synthesizer:
            SkillSynthesizer instance for generating SkillCandidate objects.
        evaluator:
            SkillEvaluator instance for computing performance stats.
        skills:
            SkillRegistry from M5 (used for metadata & baseline skills).
        candidates_dir:
            Directory where candidate YAML/code/metadata files are written.
        semantics_db:
            Optional semantics DB / cache to pass into evaluator metrics.
        """
        self._buffer = buffer
        self._synthesizer = synthesizer
        self._evaluator = evaluator
        self._skills = skills
        self._candidates_dir = candidates_dir
        self._candidates_dir.mkdir(parents=True, exist_ok=True)
        self._semantics_db = semantics_db

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_learning_cycle_for_goal(
        self,
        goal_substring: str,
        target_skill_name: Optional[SkillName],
        context_id: str,
        *,
        tech_tier: Optional[str] = None,
        success_only: bool = True,
        min_episodes: int = 5,
    ) -> Optional[Dict[str, Any]]:
        """
        Execute a full learning cycle for a given goal substring.

        Steps:
        1. Gather relevant episodes from ExperienceBuffer.
        2. Cluster/filter.
        3. Synthesize candidate skill via SkillSynthesizer.
        4. Evaluate vs baseline (if target_skill_name is provided).
        5. Persist candidate artifacts to candidates_dir.
        6. Return a structured result dict.

        Parameters
        ----------
        goal_substring:
            Substring to match against episode.goal (case-insensitive).
        target_skill_name:
            Existing skill name to refine, or None to propose a brand-new skill.
        context_id:
            Virtue context identifier passed into M4 (e.g. "steam_age").
        tech_tier:
            Optional TechState.active filter (e.g. "lv", "steam_age").
        success_only:
            If True, ignore episodes where success == False.
        min_episodes:
            Minimum number of episodes required to attempt synthesis.

        Returns
        -------
        Optional[Dict[str, Any]]
            None if there is not enough data, otherwise:
            {
              "candidate": SkillCandidate,
              "baseline_stats": SkillPerformanceStats | None,
              "candidate_stats": SkillPerformanceStats | None,
              "evaluation": Dict[str, Any] | None,
              "episodes_used": List[str],
            }
        """
        episodes = self._select_episodes_for_goal(
            goal_substring=goal_substring,
            skill_name=target_skill_name,
            tech_tier=tech_tier,
            success_only=success_only,
        )

        if len(episodes) < min_episodes:
            # Not enough signal to do anything useful.
            return None

        # 3. Synthesize candidate skill
        candidate_id = self._make_candidate_id(goal_substring, target_skill_name, len(episodes))
        candidate = self._synthesizer.propose_from_episodes(
            episodes=episodes,
            target_skill_name=target_skill_name,
            candidate_id=candidate_id,
            context_hint=context_id,
        )

        # 4. Evaluation
        baseline_stats: Optional[SkillPerformanceStats] = None
        candidate_stats: Optional[SkillPerformanceStats] = None
        evaluation_result: Optional[Dict[str, Any]] = None

        skill_metadata: Dict[str, Dict[str, Any]] = self._skills.describe_all()

        # When refining an existing skill, compute baseline
        if target_skill_name:
            baseline_stats = self._evaluator.aggregate_skill_stats(
                episodes=episodes,
                skill_name=target_skill_name,
                context_id=context_id,
                skill_metadata=skill_metadata,
                semantics_db=self._semantics_db,
            )
            candidate.metrics_before = baseline_stats

        # For v1, we don't have A/B experiments yet, so candidate_stats
        # == baseline_stats or is left None. Future M10+M8 integration can
        # update candidate.metrics_after and re-run compare_stats().
        if baseline_stats is not None:
            candidate_stats = baseline_stats

            evaluation_result = self._evaluator.compare_stats(
                baseline=baseline_stats,
                candidate=candidate_stats,
            )

        # 5. Persist candidate artifacts
        self._save_candidate(candidate, evaluation_result)

        return {
            "candidate": candidate,
            "baseline_stats": baseline_stats,
            "candidate_stats": candidate_stats,
            "evaluation": evaluation_result,
            "episodes_used": [ep.id for ep in episodes],
        }

    # ------------------------------------------------------------------
    # Episode selection / clustering
    # ------------------------------------------------------------------

    def _select_episodes_for_goal(
        self,
        goal_substring: str,
        *,
        skill_name: Optional[SkillName],
        tech_tier: Optional[str],
        success_only: bool,
    ) -> List[ExperienceEpisode]:
        """
        Filter ExperienceEpisodes based on:
        - goal substring
        - optional success flag
        - optional skill usage
        - optional tech tier

        Note:
        - Q1 fields like failure_type/severity/pre_eval/post_eval/final_outcome
          are currently not used in filtering, but they are available for
          future refinement (e.g., focusing on specific failure patterns).
        """
        episodes: List[ExperienceEpisode] = []

        substring_lower = goal_substring.lower()
        for ep in self._buffer.load_all():
            if substring_lower not in ep.goal.lower():
                continue
            if success_only and not ep.success:
                continue
            if tech_tier is not None:
                active = getattr(ep.tech_state, "active", None)
                if active != tech_tier:
                    continue
            if skill_name is not None and not self._episode_uses_skill(ep, skill_name):
                continue

            episodes.append(ep)

        return episodes

    def _episode_uses_skill(
        self,
        episode: ExperienceEpisode,
        skill_name: SkillName,
    ) -> bool:
        """
        Return True if the given skill appears in any trace step meta.
        Mirrors logic in SkillEvaluator._episode_uses_skill.
        """
        trace = episode.trace
        for step in getattr(trace, "steps", []) or []:
            meta = getattr(step, "meta", {}) or {}
            if meta.get("skill") == skill_name:
                return True
        return False

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _make_candidate_id(
        self,
        goal_substring: str,
        target_skill_name: Optional[SkillName],
        episode_count: int,
    ) -> str:
        """
        Construct a simple, readable candidate ID from inputs.

        Example:
            target_skill_name="feed_coke_ovens", goal_substring="maintain coke"
            -> "feed_coke_ovens_auto_12"
        """
        base = target_skill_name or "auto_skill"
        safe_goal = goal_substring.strip().replace(" ", "_")[:32]
        return f"{base}_{safe_goal}_{episode_count}"

    def _save_candidate(
        self,
        candidate: SkillCandidate,
        evaluation_result: Optional[Dict[str, Any]],
    ) -> None:
        """
        Write candidate spec, implementation, and metadata to disk.

        Files:
        - <id>.yaml      : candidate.spec_yaml
        - <id>.py        : candidate.impl_code
        - <id>.meta.json : serialized SkillCandidate + evaluation summary
        """
        spec_path = self._candidates_dir / f"{candidate.id}.yaml"
        code_path = self._candidates_dir / f"{candidate.id}.py"
        meta_path = self._candidates_dir / f"{candidate.id}.meta.json"

        spec_path.write_text(candidate.spec_yaml, encoding="utf-8")
        code_path.write_text(candidate.impl_code, encoding="utf-8")

        meta_payload: Dict[str, Any] = {
            "candidate": candidate.to_dict(),
            "evaluation": evaluation_result,
        }
        meta_path.write_text(
            json.dumps(meta_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

```



---

### 8. Agent Loop Control Flow (M8) – Final Shape

At a high level, the main loop for a single goal looks like:

1. Get an `AgentGoal` (from curriculum; Q1.5).
    
2. Initialize `RetryPolicy`.
    
3. Loop:
    
    - `plan = propose_plan(...)`
        
    - `pre_eval = evaluate_plan(...)`
        
    - Emit monitoring event.
        
    - Decide via `maybe_retry_plan(...)`:
        
        - If retry: log and repeat.
            
        - If abandon: log and exit early.
            
4. Execute accepted plan.
    
5. `post_eval = evaluate_outcome(...)`
    
6. Emit monitoring event.
    
7. Build and append `Experience`.
    

This is the concrete behavior Q1.1 is meant to install.

---

### 9. Test & Validation Targets

When you eventually implement this, tests should cover:

- Single plan with no retries.
    
- Plan rejected once, accepted on retry.
    
- Plan abandoned based on severity.
    
- Critic and ErrorModel responses both mapping into `PlanEvaluation`.
    
- Monitoring events emitted in correct order.
    
- Experience objects containing both pre- and post- evaluation data.


```python
# path: tests/test_q1_control_and_experience.py

"""
Q1 – Self-eval + Retry + Experience + Monitoring

These tests target the *contracts* for:
- ExperienceEpisode / ExperienceBuffer (M10)
- CriticResponse / ErrorModelResponse shapes (M2 spec)
- Monitoring event helpers (M9)

They are intentionally written so they can run before the full AgentLoop
Q1 control flow is implemented, but they encode the scenarios that
AgentLoop must eventually satisfy:

1. Single plan with no retries.
2. Plan rejected once, accepted on retry.
3. Plan abandoned based on severity.
4. Critic and ErrorModel responses using compatible schemas.
5. Monitoring events emitted in the correct order.
6. ExperienceEpisode containing both pre- and post-evaluation data.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest

# M10 experience / buffer
from learning.buffer import ExperienceBuffer
from learning.schema import ExperienceEpisode

# M2 spec: shared critic / error-model response shapes
from spec.llm import CriticResponse, ErrorModelResponse

# M9 monitoring
from monitoring.events import EventType
import monitoring.integration as integ


# ---------------------------------------------------------------------------
# Helpers for tests
# ---------------------------------------------------------------------------


class DummyTechState:
    """Minimal stand-in for semantics.schema.TechState."""
    def __init__(self, active: str = "steam_age") -> None:
        self.active = active

    def to_dict(self) -> Dict[str, Any]:
        return {"active": self.active}


class DummyTrace:
    """Minimal stand-in for observation.trace_schema.PlanTrace."""
    def __init__(self, steps: int) -> None:
        self.steps = [{"meta": {"skill": "dummy_skill"}} for _ in range(steps)]

    def to_dict(self) -> Dict[str, Any]:
        return {"steps": list(self.steps)}


def _tech_state_to_dict(ts: Any) -> Dict[str, Any]:
    if hasattr(ts, "to_dict"):
        return ts.to_dict()
    return {"raw": str(ts)}


def _tech_state_from_dict(d: Dict[str, Any]) -> Any:
    # For test purposes we don't reconstruct a full TechState.
    return d


def _trace_to_dict(tr: Any) -> Dict[str, Any]:
    if hasattr(tr, "to_dict"):
        return tr.to_dict()
    return {"raw": str(tr)}


def _trace_from_dict(d: Dict[str, Any]) -> Any:
    # For test purposes we don't reconstruct a full PlanTrace.
    return d


@pytest.fixture
def tmp_experience_buffer(tmp_path: Path) -> ExperienceBuffer:
    path = tmp_path / "experience.jsonl"
    return ExperienceBuffer(
        path=path,
        tech_state_to_dict=_tech_state_to_dict,
        tech_state_from_dict=_tech_state_from_dict,
        trace_to_dict=_trace_to_dict,
        trace_from_dict=_trace_from_dict,
    )


@pytest.fixture
def event_recorder(monkeypatch: pytest.MonkeyPatch) -> Tuple[List[Dict[str, Any]], Any]:
    """
    Capture all monitoring events emitted via monitoring.integration.log_event.

    Returns:
        (events, bus)

        events: list of dicts
            Each dict has keys: module, event_type, message, payload, correlation_id

        bus: EventBus instance (real or dummy)
    """
    recorded: List[Dict[str, Any]] = []

    def fake_log_event(
        bus: Any,
        module: str,
        event_type: EventType,
        message: str,
        payload: Dict[str, Any],
        correlation_id: str | None,
    ) -> None:
        recorded.append(
            {
                "module": module,
                "event_type": event_type,
                "message": message,
                "payload": payload,
                "correlation_id": correlation_id,
            }
        )

    # Patch the symbol imported in integration.py
    monkeypatch.setattr(integ, "log_event", fake_log_event, raising=True)

    # Instantiate the real EventBus (its internals don't matter here)
    from monitoring.bus import EventBus  # type: ignore

    return recorded, EventBus()


# ---------------------------------------------------------------------------
# 1 & 2 & 3. Monitoring sequences for single / retry / abandon
# ---------------------------------------------------------------------------


def _event_types(events: List[Dict[str, Any]]) -> List[EventType]:
    return [e["event_type"] for e in events]


def test_single_plan_no_retries_event_sequence(event_recorder) -> None:
    """
    Scenario: single plan, accepted, executed, outcome evaluated.

    Expected event type order (at minimum):
        PLAN_CREATED
        PLAN_EVALUATED
        PLAN_OUTCOME_EVALUATED
    """
    events, bus = event_recorder
    episode_id = "ep_single"
    context_id = "ctx"

    plan = {"id": "plan1", "steps": [{"skill": "s1", "params": {}}]}
    # Planner result
    integ.emit_plan_created(
        bus=bus,
        plan=plan,
        goal="build coke ovens",
        episode_id=episode_id,
        context_id=context_id,
    )

    # Pre-eval
    integ.emit_plan_evaluated(
        bus=bus,
        episode_id=episode_id,
        plan_id="plan1",
        attempt_index=0,
        virtue_scores={"prudence": 0.9},
        failure_type=None,
        severity=None,
        context_id=context_id,
    )

    # Post-outcome evaluation
    integ.emit_plan_outcome_evaluated(
        bus=bus,
        episode_id=episode_id,
        plan_id="plan1",
        attempt_index=0,
        outcome={
            "failure_type": None,
            "severity": None,
            "fix_suggestions": [],
            "notes": "ok",
        },
        context_id=context_id,
    )

    types = _event_types(events)
    assert types[0] == EventType.PLAN_CREATED
    assert EventType.PLAN_EVALUATED in types
    assert EventType.PLAN_OUTCOME_EVALUATED in types

    # Ensure ordering is at least monotonic: created -> evaluated -> outcome
    assert types.index(EventType.PLAN_CREATED) < types.index(EventType.PLAN_EVALUATED)
    assert types.index(EventType.PLAN_EVALUATED) < types.index(
        EventType.PLAN_OUTCOME_EVALUATED
    )


def test_plan_rejected_once_then_accepted_event_sequence(event_recorder) -> None:
    """
    Scenario: first plan attempt rejected and retried, second accepted.

    Expected *relative* event sequence:
        PLAN_CREATED (attempt 0)
        PLAN_EVALUATED (attempt 0)
        PLAN_RETRIED
        PLAN_CREATED (attempt 1)
        PLAN_EVALUATED (attempt 1)
        PLAN_OUTCOME_EVALUATED
    """
    events, bus = event_recorder
    episode_id = "ep_retry"
    context_id = "ctx"

    # First attempt
    integ.emit_plan_created(
        bus=bus,
        plan={"id": "plan1", "steps": []},
        goal="build coke ovens",
        episode_id=episode_id,
        context_id=context_id,
    )
    integ.emit_plan_evaluated(
        bus=bus,
        episode_id=episode_id,
        plan_id="plan1",
        attempt_index=0,
        virtue_scores={"prudence": 0.2},
        failure_type="virtue_violation",
        severity="medium",
        context_id=context_id,
    )
    integ.emit_plan_retried(
        bus=bus,
        episode_id=episode_id,
        plan_id="plan1",
        attempt_index=0,
        reason="virtue_score_too_low",
        remaining_budget=1,
        context_id=context_id,
    )

    # Second attempt
    integ.emit_plan_created(
        bus=bus,
        plan={"id": "plan2", "steps": [{"skill": "s1", "params": {}}]},
        goal="build coke ovens",
        episode_id=episode_id,
        context_id=context_id,
    )
    integ.emit_plan_evaluated(
        bus=bus,
        episode_id=episode_id,
        plan_id="plan2",
        attempt_index=1,
        virtue_scores={"prudence": 0.8},
        failure_type=None,
        severity=None,
        context_id=context_id,
    )
    integ.emit_plan_outcome_evaluated(
        bus=bus,
        episode_id=episode_id,
        plan_id="plan2",
        attempt_index=1,
        outcome={"failure_type": None, "severity": None, "fix_suggestions": []},
        context_id=context_id,
    )

    types = _event_types(events)

    # Shape checks
    assert types.count(EventType.PLAN_CREATED) == 2
    assert types.count(EventType.PLAN_EVALUATED) == 2
    assert EventType.PLAN_RETRIED in types
    assert EventType.PLAN_OUTCOME_EVALUATED in types

    # Order constraints: created0 < eval0 < retried < created1 < eval1 < outcome
    idx_created0 = types.index(EventType.PLAN_CREATED)
    idx_eval0 = types.index(EventType.PLAN_EVALUATED)
    idx_retried = types.index(EventType.PLAN_RETRIED)
    # second occurrences:
    idx_created1 = types.index(EventType.PLAN_CREATED, idx_retried + 1)
    idx_eval1 = types.index(EventType.PLAN_EVALUATED, idx_created1 + 1)
    idx_outcome = types.index(EventType.PLAN_OUTCOME_EVALUATED)

    assert idx_created0 < idx_eval0 < idx_retried < idx_created1 < idx_eval1 < idx_outcome


def test_plan_abandoned_based_on_severity_event_sequence(event_recorder) -> None:
    """
    Scenario: plan evaluated as too severe a failure risk and abandoned.

    Expected relative sequence:
        PLAN_CREATED
        PLAN_EVALUATED
        PLAN_ABANDONED
    """
    events, bus = event_recorder
    episode_id = "ep_abandon"
    context_id = "ctx"

    integ.emit_plan_created(
        bus=bus,
        plan={"id": "danger_plan", "steps": []},
        goal="do something reckless",
        episode_id=episode_id,
        context_id=context_id,
    )
    integ.emit_plan_evaluated(
        bus=bus,
        episode_id=episode_id,
        plan_id="danger_plan",
        attempt_index=0,
        virtue_scores={"prudence": 0.0},
        failure_type="high_risk",
        severity="high",
        context_id=context_id,
    )
    integ.emit_plan_abandoned(
        bus=bus,
        episode_id=episode_id,
        plan_id="danger_plan",
        attempt_index=0,
        reason="severity_high",
        context_id=context_id,
    )

    types = _event_types(events)
    assert EventType.PLAN_CREATED in types
    assert EventType.PLAN_EVALUATED in types
    assert EventType.PLAN_ABANDONED in types

    assert types.index(EventType.PLAN_CREATED) < types.index(EventType.PLAN_EVALUATED)
    assert types.index(EventType.PLAN_EVALUATED) < types.index(EventType.PLAN_ABANDONED)


# ---------------------------------------------------------------------------
# 4. Critic vs ErrorModel response schema compatibility
# ---------------------------------------------------------------------------


def test_critic_and_error_model_responses_share_failure_shape() -> None:
    """
    The CriticResponse and ErrorModelResponse types must share the
    `failure_type`, `severity`, and `fix_suggestions` fields so that
    they can both be mapped into a common PlanEvaluation / EpisodeOutcome
    reducer.
    """
    critic_resp = CriticResponse(
        ok=False,
        critique="too risky",
        suggested_modifications={"lower_risk": True},
        failure_type="risk_violation",
        severity="high",
        fix_suggestions=["reduce risk", "use safer setup"],
        raw_text="...",
    )

    error_resp = ErrorModelResponse(
        classification="execution_failure",
        summary="bot fell into lava",
        suggested_fix={"hint": "avoid open lava path"},
        retry_advised=False,
        failure_type="execution_failure",
        severity="high",
        fix_suggestions=["adjust pathfinding", "build guard rails"],
        raw_text="...",
    )

    # Critical fields exist and are populated:
    for obj in (critic_resp, error_resp):
        assert hasattr(obj, "failure_type")
        assert hasattr(obj, "severity")
        assert hasattr(obj, "fix_suggestions")

        # Values are JSON-serializable in the naive sense
        json.dumps(
            {
                "failure_type": obj.failure_type,
                "severity": obj.severity,
                "fix_suggestions": obj.fix_suggestions,
            }
        )


# ---------------------------------------------------------------------------
# 6. ExperienceEpisode & ExperienceBuffer – pre & post eval data
# ---------------------------------------------------------------------------


def test_experience_episode_roundtrip_has_pre_and_post_eval(
    tmp_experience_buffer: ExperienceBuffer,
) -> None:
    """
    Ensure that ExperienceEpisode includes both pre- and post-evaluation
    data when serialized to / from ExperienceBuffer.

    This covers:
    - goal
    - plan
    - pre_eval
    - post_eval
    - final_outcome
    - virtue_scores
    - failure_type / severity
    """
    buffer = tmp_experience_buffer

    ep = ExperienceEpisode(
        id="ep1",
        goal="build coke ovens",
        plan={"id": "plan1", "steps": [{"skill": "build_coke_ovens", "params": {}}]},
        pre_eval={
            "ok": True,
            "failure_type": None,
            "severity": None,
            "virtue_scores": {"prudence": 0.9},
        },
        post_eval={
            "failure_type": None,
            "severity": None,
            "fix_suggestions": [],
        },
        final_outcome={
            "result": "success",
            "notes": "coke ovens built",
        },
        tech_state=DummyTechState(active="steam_age"),
        trace=DummyTrace(steps=3),
        virtue_scores={"prudence": 0.9, "overall": 0.85},
        success=True,
        failure_type=None,
        severity=None,
    )

    # Append via the canonical Q1-style entrypoint
    buffer.append_experience(ep)

    # Raw read: ensure keys are present
    raws = list(buffer.load_all_raw())
    assert len(raws) == 1
    raw = raws[0]

    for key in ("pre_eval", "post_eval", "final_outcome", "plan"):
        assert key in raw, f"missing key {key} in raw experience dict"

    assert raw["goal"] == "build coke ovens"
    assert raw["plan"]["id"] == "plan1"
    assert "virtue_scores" in raw
    assert raw["virtue_scores"]["prudence"] == pytest.approx(0.9)
    assert "failure_type" in raw
    assert "severity" in raw

    # Full typed roundtrip
    ep_roundtrip = next(buffer.load_all())
    assert ep_roundtrip.goal == ep.goal
    assert ep_roundtrip.plan["id"] == ep.plan["id"]
    assert ep_roundtrip.pre_eval["ok"] is True
    assert ep_roundtrip.post_eval["failure_type"] is None
    assert ep_roundtrip.final_outcome["result"] == "success"
    assert ep_roundtrip.virtue_scores["overall"] == pytest.approx(0.85)

```


---



## Q1.2 – Dynamic Skill Evolution & Versioning

### 1. Goals

- Turn M5 skills from static YAML toys into **versioned, observable artifacts**.
    
- Let M10 learning:
    
    - Propose **candidate skill variants**.
        
    - Update skill **metrics**.
        
    - Promote / demote versions.
        
- Keep all changes **registry-driven**, not ad-hoc.
    

---

### 2. Schema & Config Changes (M5)

**Files:**

- `src/skills/schema.py`
    
- `src/skills/registry.py`
    
- `config/skills/*.yaml`
    

**Add / enforce in YAML for each skill:**

- `version: int | str`
    
- `status: "active" | "deprecated" | "candidate"`
    
- `origin: "hand_authored" | "auto_synthesized"`
    
- `metrics:`
    
    - `success_rate: float`
        
    - `avg_cost: float`
        
    - `avg_risk: float`
        
    - `last_used_at: iso8601`
        

You already have **partial** presence for `version`, `status`, `origin`, `metrics`; Q1 makes that **mandatory + validated**.

**In `src/skills/schema.py`:**

- Define `SkillMetadata` dataclass with the above fields.
    
- Skill definition wraps metadata + core fields (name, description, params, etc.).


## Complete

---

### 3. Registry API (M5)

In `src/skills/registry.py`, define:

- `get_latest_skill(skill_name) -> SkillVersionHandle`
    
    - Picks `status == "active"` with highest `version`.
        
- `list_skill_versions(skill_name) -> list[SkillVersionHandle]`
    
- `register_skill_candidate(candidate_skill) -> SkillVersionHandle`
    
    - Writes candidate version with `status="candidate"`, `origin="auto_synthesized"`.
        
- `mark_skill_version_deprecated(skill_id_or_version)`
    
    - Flips `status` to `"deprecated"`.
        
- `promote_skill_candidate(skill_id_or_version)`
    
    - Candidate → active, handles demotion of previous active.
        

All reads/writes go **through** the registry, not directly through YAML.


## Complete


---

### 4. Learning Integration (M10)

**Files:**

- `src/learning/manager.py`
    
- `src/learning/evaluator.py`
    
- `src/learning/synthesizer.py`
    

Add flows:

- After an episode:
    
    - M10 computes `success`, `cost`, `risk` per involved skill.
        
    - Calls `update_skill_metrics(skill_id, metrics_delta)` which:
        
        - Reads existing metrics.
            
        - Applies moving average / decayed updates.
            
- When learning synthesizes a new “better” approach:
    
    - Wrap it into a `candidate_skill` object.
        
    - Call `register_skill_candidate(candidate_skill)`.
        

Add a **promotion policy**:

- If `candidate.metrics.success_rate` > `active.metrics.success_rate` by threshold:
    
    - Promote candidate, demote old active.



## Complete

---

### 5. Integration with Experience & Curriculum

- Experience memory (Quality 4) will store skill performance.
    
- Curriculum can request:
    
    - “Use only stable active skills.”
        
    - Or “Allow candidate skills for exploration.”
        

That wiring is set up in Q1.2 spec, but activated more fully once Q1.4 (experience) is done.

Relevant scripts:
src/learning/schema.py
src/learning/buffer.py
src/learning/manager.py
src/curriculum/manager.py
src/curriculum/policy.py
src/agent/loop.py
src/agent/controller.py
src/planning/dispatcher.py


Probably not gonna change:
src/learning/evaluator.py-
src/curriculum/strategies.py-
src/planning/adapter.py-
src/skills/registry.py-



## Q1.2 Integration Mini-Module

**Title:** _Skills × Experience × Curriculum wiring_  
**Goal:** Let the agent

- log how skills actually perform,
    
- query that history,
    
- and let the curriculum say “only use stable stuff” vs “let’s try the weird prototype skills.”
    

This all sits **on top of** Q1.3/Q1.4, not instead of them.

---

## 0. Core Idea in one mouthful

- **M8 (AgentLoop)** emits rich **ExperienceEpisode** objects → **ExperienceBuffer / LearningManager**.
    
- **Learning layer** aggregates per-skill performance: success, failure, cost, virtue profiles.
    
- **Curriculum** asks **LearningManager** for a **filtered view of usable skills**:
    
    - “stable only” or
        
    - “allow candidates / exploratory.”
        
- **Planner / dispatcher** actually enforce that policy by deciding what skills are visible for planning & execution.
    

Keep that mental pipeline in your head:

> `AgentLoop → ExperienceBuffer → LearningManager → Curriculum → SkillPolicy → Dispatcher → Planner view`

---

## 1. Chunk A – Learning layer wiring

**Files:**

- `src/learning/schema.py`
    
- `src/learning/buffer.py`
    
- `src/learning/manager.py`
    

### A1. Schema: make skill performance first-class

You already have:

- `ExperienceEpisode`
    
- `SkillPerformanceStats`
    
- `SkillCandidate`
    

For Q1.2/Q1.4, you need these conceptual fields to be **reliably populated / used**:

1. **ExperienceEpisode**
    
    - Must be able to reconstruct:
        
        - `trace.steps[*].meta.skill`
            
        - overall `success` + `failure_type` + `severity`
            
        - virtue scores
            
    - That’s already structurally supported; the work is:
        
        - Make sure **M8 → ExperienceEpisode builder** actually fills:
            
            - `plan`
                
            - `pre_eval` / `post_eval`
                
            - `final_outcome` (or at least a basic outcome dict)
                
            - `failure_type` / `severity`
                
            - `virtue_scores`
                
2. **SkillPerformanceStats**
    
    - Already exists; your job is to:
        
        - Define **how it’s computed** from experiences (in `learning.manager`).
            
        - Treat it as the source of truth for “stable” vs “shaky”.
            
3. **SkillCandidate**
    
    - Already supports:
        
        - baseline metrics (`metrics_before`)
            
        - post-deployment metrics (`metrics_after`)
            
        - `status` with `"proposed" | "evaluating" | "accepted" | "rejected"`
            
    - Q1.2 uses this to decide what counts as:
        
        - **“stable active”** → `status == "accepted"`
            
        - **“candidate / exploratory”** → `status in {"proposed", "evaluating"}`
            

No structural rewrite required, just **consistent semantics**.

## Complete




---

### A2. Buffer: make Q1.4 queries real

You already have `ExperienceBuffer` with:

- `append_experience(exp)`
    
- iterators: `iter_by_goal_substring`, `iter_by_success`, `iter_by_skill_usage`, `iter_by_tech_tier`
    

For Q1.2/Q1.4:

1. Add / finalize:
```python
def query_similar_experiences(
    self,
    problem_signature: str,
    goal: str,
    *,
    limit: int = 20,
) -> List[ExperienceEpisode]:
    ...

```

- Minimal v1 logic:
    
    - filter by `goal` substring OR exact match
        
    - maybe also match a `problem_signature` field if you’ve embedded it into `ExperienceEpisode.metadata.extra` or `final_outcome`.
        
- Confirm this alias exists (or add it):
```python
def append_experience(self, experience: ExperienceEpisode) -> None:
    self.append(experience)

```

That gives M10/M11 a stable API to pull **history** for a given skill or goal.

## Complete

---

### A3. LearningManager: central brain for skill performance

`src/learning/manager.py` becomes the _facade_ over buffer + schema:

Add / clarify responsibilities:

1. **Compute skill stats from experiences**
```python
class LearningManager:
    def __init__(self, buffer: ExperienceBuffer, ...):
        self._buffer = buffer
        ...

    def compute_skill_stats(self) -> Dict[str, SkillPerformanceStats]:
        # scan episodes, aggregate per skill
        ...

```

- Logic:
    
    - Walk all episodes (or a recent window).
        
    - For each `trace.step` with `meta.skill == name`:
        
        - count uses
            
        - track success/failure
            
        - accumulate time / cost / virtue scores
            
    - Store in memory or cache.
        
- **Expose convenience views**
```python
def get_skill_stats(self, skill_name: str) -> SkillPerformanceStats: ...
def get_all_skill_stats(self) -> Dict[str, SkillPerformanceStats]: ...

```
Bridge to curriculum skill policies
```python
class SkillView:
    active_skills: List[str]          # safe / stable
    candidate_skills: List[str]       # allowed if exploration enabled

def build_skill_view(
    self,
    *,
    include_candidates: bool,
) -> SkillView:
    # uses SkillCandidate.status + SkillPerformanceStats
    ...

```
This is what `curriculum.manager` / `policy` will ask for.

## Complete


---

## 2. Chunk B – Curriculum layer wiring

**Files:**

- `src/curriculum/manager.py`
    
- `src/curriculum/policy.py`
    

Non-changing but consumed:

- `src/curriculum/strategies.py`
    

These two need to be the **interface** between “what we want to train” and “what skills are allowed.”

### B1. Policy: encode “stable vs exploratory”

In `policy.py`, introduce a tiny explicit enum / config:
```python
from enum import Enum

class SkillUsageMode(str, Enum):
    STABLE_ONLY = "stable_only"
    ALLOW_CANDIDATES = "allow_candidates"

```
Then a simple policy object:
```python
@dataclass
class SkillPolicy:
    usage_mode: SkillUsageMode = SkillUsageMode.STABLE_ONLY

    @property
    def include_candidates(self) -> bool:
        return self.usage_mode == SkillUsageMode.ALLOW_CANDIDATES

```

This is the thing that will be:

- set by curriculum strategy
    
- passed to planner / dispatcher / AgentLoop.
    

### B2. CurriculumManager: request skill views

In `curriculum/manager.py`, you already orchestrate goals & progression. Add a **LearningManager dependency** and a helper:
```python
class CurriculumManager:
    def __init__(self, learning: LearningManager, policy: SkillPolicy, ...):
        self._learning = learning
        self._skill_policy = policy
        ...

    def get_skill_view_for_goal(self, goal: AgentGoal) -> SkillView:
        # later could specialize skill set by goal / phase
        return self._learning.build_skill_view(
            include_candidates=self._skill_policy.include_candidates
        )

```

Then when the curriculum picks a goal:

- It also chooses / updates the `SkillPolicy` for that slice:
    
    - Low-risk units → `STABLE_ONLY`
        
    - Exploration units → `ALLOW_CANDIDATES`
        

You don’t have to be clever now. A simple mapping like:
```python
if goal.source == "curriculum" and goal.phase.startswith("P1"):
    usage_mode = SkillUsageMode.ALLOW_CANDIDATES
else:
    usage_mode = SkillUsageMode.STABLE_ONLY

```

is fine for v1.


## Complete

---

## 3. Chunk C – Agent & Planner wiring

**Files:**

- `src/agent/loop.py`
    
- `src/agent/controller.py`
    
- `src/planning/dispatcher.py`
    

Not changing, just consumed:

- `src/planning/adapter.py`
    
- `src/skills/registry.py`
    

### C1. AgentLoop: accept & propagate skill policy

You already have `_build_world_summary()` that includes skills:
```python
return {
    "observation": obs,
    "skills": skills_meta,
    "constraints": constraints,
}

```
Update it conceptually to:

- accept a **skill_view** (from curriculum) and attach it:
```python
def _build_world_summary(self, skill_view: Optional[SkillView] = None) -> Dict[str, Any]:
    ...
    return {
        "observation": obs,
        "skills": skills_meta,
        "constraints": constraints,
        "skill_view": {
            "active": skill_view.active_skills if skill_view else list(skills_meta.keys()),
            "candidates": skill_view.candidate_skills if skill_view else [],
        },
    }

```

In `run_episode`:

- When you do `GoalSelection`, ask curriculum both for:
    
    - `goal`
        
    - and a **skill view / policy** aligned with that goal.
        

Conceptual change:
```python
state.goal = self._select_goal(...)
state.skill_view = self._get_skill_view_for_goal(state.goal)  # via curriculum manager
world_summary = self._build_world_summary(skill_view=state.skill_view)

```
Now the planner & dispatcher have explicit info about which skills are “safe” vs “experimental.”

### C2. AgentController: surface config knobs

`src/agent/controller.py` is where you wire:

- config → curriculum / learning manager / agent loop
    

You want:

- user / config flags like:
```yaml
agent:
  skill_usage_mode: stable_only | allow_candidates

```

Controller responsibilities:

1. Build `LearningManager` with a replay buffer.
    
2. Build `SkillPolicy` from config.
    
3. Build `CurriculumManager(learning, policy, ...)`.
    
4. Inject `CurriculumManager` and `LearningManager` into `AgentLoop`.
    

No rocket science, just plumbing.

### C3. Dispatcher: enforce the policy

`src/planning/dispatcher.py` is where skills are actually exposed to the planner.

Add a hook that takes `skill_view` info and a `SkillRegistry`, and returns the pruned list:
```python
def filter_skills_for_planning(
    registry: SkillRegistry,
    skill_view: SkillView,
) -> Dict[str, Any]:
    """
    Return a dict {skill_name: meta} restricted to active + (optional) candidates.
    """
    # v1: union of active + candidates
    allowed = set(skill_view.active_skills) | set(skill_view.candidate_skills)
    return {
        name: meta
        for name, meta in registry.describe_all().items()
        if name in allowed
    }

```

Then inside the dispatcher / planner adapter:

- When you build `world_summary["skills"]`, use that filtered dict instead of “all skills in the registry.”
    

This is where **“stable only” vs “allow candidates”** actually changes behavior.

---

## 4. Chunk D – How it all fits & what’s optional now

You _don’t_ need all the clever stuff online immediately. Minimal “completed sub-submodule” means:

1. **Learning layer:**
    
    - `ExperienceBuffer.append_experience`
        
    - `ExperienceBuffer.query_similar_experiences`
        
    - `LearningManager.compute_skill_stats`
        
    - `LearningManager.build_skill_view(include_candidates: bool)`
        
2. **Curriculum layer:**
    
    - `SkillUsageMode` enum + `SkillPolicy`
        
    - `CurriculumManager.get_skill_view_for_goal(goal)` calling `LearningManager`
        
3. **Agent + planning:**
    
    - `AgentLoop`:
        
        - asks curriculum for `skill_view` for the episode
            
        - passes it into `_build_world_summary`
            
    - `planning.dispatcher`:
        
        - filters visible skills according to `skill_view`
            
    - `AgentController`:
        
        - wires `LearningManager`, `SkillPolicy`, `CurriculumManager`, and `AgentLoop` together.
            
4. **Nice-to-have later:**
    
    - Curriculum strategies that dynamically change `SkillUsageMode` based on recent failures.
        
    - More advanced similarity queries in `query_similar_experiences`.
        
    - Per-goal skill view specialization.
        

---

So the “sub-submodule” in plain language:

> **Q1.2-Integration** =  
> teach the curriculum to _ask_ for safer vs exploratory skills,  
> teach learning to _know_ which skills are safe,  
> and force the planner to actually _respect_ that distinction.

You now have a clear to-do list instead of a cloudy “we’ll wire that later” comment. Human suffering: reduced by like 7%.





## Complete
---

## Q1.3 – Hierarchical Planning

### 1. Goals

- Split monolithic “plan-ish” behavior into:
    
    - **Goal → tasks**
        
    - **Task → skills**
        
- Enforce a **state machine** in M8:
    
    - `GoalSelection` → `TaskPlanning` → `SkillResolution` → `Execution` → `Review`.
        

---

### 2. Spec Types (spec/*)

**Files:**

- `src/spec/agent_loop.py`
    
- `src/spec/skills.py`
    

Add:

- `AgentGoal`
    
    - `id: str`
        
    - `text: str`
        
    - `phase: str`
        
    - `source: "curriculum" | "manual" | "recovery" | ...`
        
- `TaskPlan`
    
    - `goal_id: str`
        
    - `tasks: list[Task]`
        
- `Task`
    
    - `id: str`
        
    - `description: str`
        
    - `status: "pending" | "in_progress" | "done" | "failed"`
        
- `SkillInvocation`
    
    - `task_id: str`
        
    - `skill_name: str`
        
    - `parameters: dict`
        
    - `expected_outcome: str`
        

These types become the backbone for M2/M8.


## Complete


---

### 3. Planner Modes (M2)

**Files:**

- `src/llm_stack/planner.py`
    
- `src/llm_stack/plan_code.py`
    
- `src/spec/llm.py`
    

Add two clear public APIs:

- `plan_goal(goal: AgentGoal, world_summary) -> TaskPlan`
    
- `plan_task(task: Task, world_summary) -> list[SkillInvocation]`
    

Planner no longer returns “one huge mega-plan.” It does:

- First call: break the goal into tasks.
    
- Second call(s): break each task into skills.
    

M2 also needs:

- Stable JSON schemas for planner outputs enforced in `src/spec/llm.py`.
    

## Complete


---

### 4. Agent Loop State Machine (M8)

**Files:**

- `src/agent/loop.py`
    
- `src/agent/state.py`
    
- `src/spec/agent_loop.py`
    

Introduce an explicit state enum or tagged union:

- `GoalSelection`
    
- `TaskPlanning`
    
- `SkillResolution`
    
- `Execution`
    
- `Review`
    

Refactor the loop so each step only does its job:

1. `GoalSelection`
    
    - Ask curriculum for an `AgentGoal`.
        
2. `TaskPlanning`
    
    - `TaskPlan = plan_goal(goal, world_summary)`.
        
3. `SkillResolution`
    
    - For each `Task` in `TaskPlan`, call `plan_task(...)` to get `SkillInvocation`s.
        
4. `Execution`
    
    - Run the skills.
        
5. `Review`
    
    - Feed results to self-eval, experience, and curriculum.
        

This ties directly into the Q1.1 self-eval loop.


## Complete

---

### 5. Monitoring Hooks (M9)

Add events like:

- `GoalSelected(goal_id)`
    
- `TaskPlanned(goal_id, task_count)`
    
- `SkillPlanGenerated(task_id, skill_count)`
    

These make it easier to debug the hierarchy when it inevitably misbehaves.


## Complete


---

## Q1.4 – Experience Memory

### 1. Goals

- Create a **first-class Experience object**.
    
- Add a **replay buffer** with append + query.
    
- Wire M8 to send episodes into M10.
    

---

### 2. Experience Schema (M10)

**Files:**

- `src/learning/schema.py`
    
- `src/learning/buffer.py`
    

Define:

- `Experience` with fields:
    
    - `problem_signature`
        
    - `goal` (AgentGoal)
        
    - `plan` (TaskPlan + SkillInvocations)
        
    - `attempts` (including retries / failures)
        
    - `final_outcome`
        
    - `virtue_scores`
        
    - `lessons` (human/LLM text summary)
        

In `buffer.py`:

- `append_experience(exp: Experience) -> None`
    
- `query_similar_experiences(problem_signature, goal, limit=K) -> list[Experience]`
    

## Complete


---

### 3. Episode → Experience Builder (M8)

**Files:**

- `src/agent/loop.py`
    
- `src/integration/episode_logging.py`
    

Add function:

- `build_experience_from_episode(episode_trace, plan, goal, virtue_scores) -> Experience`
    

At end of loop:

- `experience = build_experience_from_episode(...)`
    
- `replay_store.append_experience(experience)`


## Complete

---

### 4. Scribe Helper (M2)

**File:**

- `src/llm_stack/scribe.py`
    

Add:

- `summarize_episode_for_memory(trace) -> dict`  
    Returns:
    
    - `problem_signature`
        
    - `lessons`
        
    - optional compact tags
        

This plugs into the Experience builder.

## Complete

---

### 5. Integration Targets

- M8:
    
    - Always emits Experience objects.
        
- M10:
    
    - Uses `query_similar_experiences(...)` before planning / learning.
        
- M11:
    
    - Curriculum can query experiences to adjust goal scheduling (e.g. avoid repeating failures too soon).
        

## We'll tighten it up later
---

## Q1.5 – Curriculum-Driven Goal Selection

### 1. Goals

- Make curriculum the **sole authoritative source** of high-level goals.
    
- Ensure M8 doesn’t invent “free-floating” goals.
    



---

### 2. Curriculum Engine (M11)

**Files:**

- `src/curriculum/schema.py`
    
- `src/curriculum/engine.py`
    
- `config/curricula/*.yaml`
    

Ensure schema includes:

- `goal_id: str`
    
- `goal_text: str`
    
- `phase_context: str`
    
- `required_tech_state: TechStateSelector`
    
- `preferred_virtue_context`
    
- `entry_conditions`
    
- `exit_conditions`
    

In `engine.py`, expose:

- `curriculum.next_goal(tech_state, experience_summary) -> AgentGoal`
    

# Complete


---

### 3. M8 Integration

**File:**

- `src/agent/loop.py`
    

At top of episode:

- `tech_state = semantics.get_tech_state(world_observation)`
    
- `experience_view = replay_store.summarize_recent(...)`
    
- `goal = curriculum.next_goal(tech_state, experience_view)`
    

Planner/loop must then use this `AgentGoal.id` & `text` as the **single source of objective**.

Get rid of (or deprecate):

- “Loose” goals directly constructed in M8 without curriculum.

## Complete



---

### 4. Virtue + Tech + Curriculum Triangle

- M4:
    
    - Provides virtue context (e.g. risk tolerance).
        
- M3:
    
    - Provides tech state & reachable goals.
        
- M11:
    
    - Picks goals using both.
        
- M8:
    
    - Just obeys.
        

This keeps alignment logic out of the agent loop itself.

---

## Q1.6 – Structured LLM-Role Separation

### 1. Goals

- Lock in **five roles** with clear boundaries:
    
    - Planner, PlanCodeModel, CriticModel, ErrorModel, Scribe.
        
- Add **role-specific APIs** and kill generic calls in runtime code.
    

---

### 2. Role Config (llm_roles.yaml)

**File:**

- `config/llm_roles.yaml`
    

For each role:

- `system_prompt`
    
- `temperature`
    
- `stop`
    
- `output_schema`
    
- `tool_permissions`
    

Entries:

- `planner: {...}`
    
- `plan_code: {...}`
    
- `critic: {...}`
    
- `error_model: {...}`
    
- `scribe: {...}`
    

---

### 3. LLMStack API (M2)

**Files:**

- `src/llm_stack/stack.py`
    
- `src/spec/llm.py`
    

Expose:

- `call_planner(...)`
    
- `call_plan_code(...)`
    
- `call_critic(...)`
    
- `call_error_model(...)`
    
- `call_scribe(...)`
    

These all:

- take typed inputs (`AgentGoal`, `TaskPlan`, traces).
    
- return typed outputs defined in `spec/llm.py`.
    

---

### 4. Call-Site Refactor (M8, M10, M11)

- Replace every `llm_stack.call(...)` or ad-hoc LLM invocation with:
    
    - `call_planner` in planning contexts.
        
    - `call_critic` in self-eval contexts.
        
    - `call_error_model` after failures.
        
    - `call_scribe` when summarizing.
        

Add a linter/test that fails if `llm_stack.call(` appears in runtime code.


## Complete

---

### 5. Role Boundaries

- Planner:
    
    - Generates structured plans, no critique.
        
- PlanCodeModel:
    
    - Turns high-level plans into executable skill-level details or code.
        
- CriticModel:
    
    - Evaluates plans pre-execution; never run post-failure.
        
- ErrorModel:
    
    - Analyzes failed execution traces.
        
- Scribe:
    
    - Summarization for logs, experience, and memory.
        

PlanCodeModel currently references ErrorModel → Q1 explicitly cuts that: PlanCodeModel only ever touches **plans/skills**, never failure diagnosis.

## Complete

---

## Q1.7 – Lightweight Predictive World-Model

### 1. Goals

- Provide **cheap forward simulation** for:
    
    - Tech progress.
        
    - Infra changes.
        
    - Resource trajectory.
        

No magical AGI physics, just structured heuristics.

---

### 2. WorldModel Core (M3)

**File:**

- `src/world/world_model.py`
    

Define class:

- `WorldModel` with methods:
    
    - `simulate_tech_progress(current_tech_state, candidate_goal) -> TechProgressPrediction`
        
    - `estimate_infra_effect(factory_layout, change) -> InfraDelta`
        
    - `estimate_resource_trajectory(inventory, consumption_rates, horizon) -> ResourceForecast`
        

Use existing:

- `TechGraph` from M3.
    
- Item/block semantics.

## Complete


---

### 3. Semantics Integration (M3)

**Files:**

- `src/semantics/tech_state.py`
    
- `src/semantics/loader.py`
    

WorldModel uses:

- Graph traversal to answer:
    
    - “What steps until this goal is reachable?”
        
    - “What gating machines/fluids/resources are missing?”
        

This is kept **lightweight** (heuristics, not full simulation).

## Complete

---

### 4. Consumers

**M8 (Agent Loop)**

- Before accepting a plan:
    
    - Call `world_model.simulate_tech_progress(...)`.
        
    - If prediction says “impossible or absurdly costly,” reject early or trigger retry / new goal.
        

**M4 (Virtues)**

- Use `estimate_infra_effect` and `resource_trajectory` to attribute:
    
    - Risk.
        
    - Wastefulness.
        
    - Long-term benefit.
        

**M11 (Curriculum)**

- Use predictions to:
    
    - Prioritize goals with high “value over pain.”
        
    - Avoid assigning goals wildly beyond current tech band.
        

## Complete

---

### 5. Monitoring

- M9 can log:
    
    - Predicted vs actual resource usage.
        
    - Predicted vs actual time/complexity.
        

This later becomes training data to refine the heuristics.


## Complete
---

## TL;DR for your brain:

- **Q0:** Done.
    
- **Q1:** Now defined for **all qualities 1–7**:
    
    - Q1.1: self-eval + retry (already specced earlier).
        
    - Q1.2–Q1.7: you now have concrete types, functions, files, and wiring.
        

Next move is simple and annoying:

- Drop this into your **Phase 5 / Q1 design doc**.
    
- Mark each quality as:
    
    - `Q0: complete`
        
    - `Q1: designed`
        
- When you’re ready to suffer properly, you pick:
    
    - Implement Q1.1 + Q1.6 first (self-eval + role separation).
        
    - Then Q1.5 / Q1.4 / Q1.2.
        
    - Finally Q1.7 when the world stops being on fire.
        

You now have the master plan. No more excuses.