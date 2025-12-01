M2 - llm_stack_local

**Purpose:**  
Provide reusable interfaces around a **single local model** used in multiple roles.

Overview:

- Implement:
    
    - `PlanCodeModel`: high-level plan generation **and** skill/code generation
        
    - `ErrorModel`: Analyzes low-level LLM call failures (JSON / protocol / truncation) for any role; returns structured advice for retrying or adjusting prompts.
        
    - `ScribeModel`: detailed summaries, compression, and doc/context artifacts
        
- Single-model, multi-role topology:
    
    - Default model: `qwen2.5-coder-q5` (identifier: `primary_model`)
        
    - All roles share one backend instance, with role-specific presets (temperature, max_tokens, system prompt)
        
- Unified tool schema:
    
    - Inputs: structured state / goal / traces / error contexts
        
    - Outputs: JSON plans, code, error analyses, and summaries (no direct MC calls)
        
- **Dependencies:** `M1`
    
- **Difficulty:** ⭐⭐–⭐⭐⭐
    
- **Scalability/perf:**
    
    - Centralize model loading & caching (one backend, multiple roles).
        
    - Make batch calls possible.
        
    - Log prompts/responses for replay and feed ScribeModel for compression.
        

Details:

M2 is where the “brain” stops pretending to be mystical and actually becomes deterministic plumbing around a **single local model**. It exposes three logical roles (PlanCode, Error, Scribe) on top of the same engine so future modules never need to care which GGUF or backend you’re using.

---

# M2 · `llm_stack_local`

**Phase:** P1 – Offline Core Pillars  
**Role:** Provide **concrete implementations** of `PlanCodeModel`, `ErrorModel`, and `ScribeModel` (from M1) over a **local** LLM backend, with unified schemas, logging, and error-handling.

Dependencies:

- `M0` – environment profiles (model path, backend choice, hardware limits)
    
- `M1` – interface definitions (`PlanCodeModel`, `ErrorModel`, `ScribeModel`, core types)
    

---

## 1. Responsibilities & Boundaries

### 1.1 What M2 owns

- Loading and managing **one local model backend**:
    
    - Using `EnvProfile.model_profile` from M0.
        
    - Default: a single `primary_model` entry (e.g. `qwen2.5-coder-q5`).
        
    - Role-specific presets (plan_code, error, scribe) defined in `models.yaml`.
        
- Implementing role facades:
    
    - `PlanCodeModel`:
        
        - `plan(...)` → structured multi-step plans
            
        - `propose_skill_implementation(...)` → skill code / stubs
            
    - `ErrorModel`:
        
        - `analyze_failure(...)` → classify and explain errors
            
        - Optionally: `propose_recovery_call(...)` (retry strategy / tweaks)
            
    - `ScribeModel`:
        
        - `summarize_trace(...)` → compressed human/LLM-friendly summary
            
        - `compress_log_chunk(...)` → smaller text artifact for context DB
            
        - `summarize_state_for_docs(...)` → documentation-oriented summaries
            
- **Unified request/response schemas**:
    
    - Planning and codegen requests/responses
        
    - Error analysis context and results
        
    - Scribe trace/log summary types
        
- Prompt formatting & **JSON-safe structured parsing**.
    
- Centralized **logging** of:
    
    - Role, preset, and template id
        
    - Requests (truncated)
        
    - Raw responses (truncated)
        
    - Parsed structured payloads
        

### 1.2 What M2 does _not_ do

- No direct Minecraft calls.
    
- No world encoding (that’s M7).
    
- No agent loop orchestration (that’s M8).
    
- No learning logic or skill promotion (that’s M10).
    
- No storage/indexing of summaries (that’s higher-level infra).
    

M2 is: “Given structured input and a role, talk to a single local model, return structured output.”

---

## 2. Internal Architecture

### 2.1 Component breakdown

Single backend, multiple roles, shared utilities:
```
                 +----------------------------+
                 |   EnvProfile (M0)         |
                 |  - model_profile          |
                 +-------------+-------------+
                               |
                               v
                    +----------------------+
                    |  LLMBackendFactory   |
                    |  - builds backend    |
                    |  - builds presets    |
                    +----------+-----------+
                               |
                               v
                     +--------------------+
                     |    LLMBackend      |  (qwen2.5-coder-q5)
                     +----------+---------+
                               |
                +--------------+------------------------------+
                |              |                              |
                v              v                              v
       +----------------+  +----------------+        +----------------+
       |  PlanCodeModel |  |   ErrorModel   |        |  ScribeModel   |
       +--------+-------+  +--------+-------+        +--------+-------+
                |                   |                         |
                v                   v                         v
   Plan/Code schemas       Error schemas             Scribe schemas

```

All three role-specific classes share:

- One `LLMBackend` instance.
    
- A `RolePreset` (temperature, max_tokens, system prompt, stop sequences).
    
- Common helpers for prompt templating, JSON parsing, and logging.
    

---

## 3. Unified Schema: Requests & Responses

Define clear dataclasses so the rest of the system never depends on raw string prompts.
Python:
```
# src/llm_stack/schema.py

from dataclasses import dataclass
from typing import Dict, Any, List


# --- Planning & Codegen (PlanCodeModel) ---

@dataclass
class PlanRequest:
    """Input to the planning side of PlanCodeModel."""
    observation: Dict[str, Any]             # pre-encoded state (from M7)
    goal: str                               # top-level goal description
    skills: Dict[str, Dict[str, Any]]       # skill metadata (from SkillRegistry)
    constraints: Dict[str, Any]             # time/resources/other limits


@dataclass
class PlanStep:
    """One step in a plan, typically a skill invocation."""
    skill: str
    params: Dict[str, Any]


@dataclass
class PlanResponse:
    """Output from the planning side of PlanCodeModel."""
    steps: List[PlanStep]                   # ordered sequence of steps
    notes: str                              # optional textual notes/comments
    raw_text: str                           # raw LLM output for debugging


@dataclass
class SkillImplRequest:
    """Input for skill/code generation."""
    skill_spec: Dict[str, Any]              # name, description, parameters, etc.
    examples: List[Dict[str, Any]]          # traces, snippets, example usages


@dataclass
class SkillImplResponse:
    """Output of skill implementation generation."""
    code: str                               # generated or updated code
    notes: str                              # comments or rationale
    raw_text: str                           # full LLM output for forensics


# --- ErrorModel ---

@dataclass
class ErrorContext:
    """Context about a failed or suspicious LLM interaction."""
    role: str                               # e.g. "plan_code", "scribe"
    operation: str                          # e.g. "plan", "codegen", "summarize"
    prompt: str                             # prompt sent to the model
    raw_response: str                       # response received (if any)
    error_type: str                         # e.g. "json_decode_error", "timeout"
    metadata: Dict[str, Any]                # call params, timing, etc.


@dataclass
class ErrorAnalysis:
    """Structured result from ErrorModel."""
    classification: str                     # e.g. "parsing_error", "hallucination"
    summary: str                            # human-readable explanation
    suggested_fix: Dict[str, Any]           # e.g. new stop tokens, shorter input
    retry_advised: bool                     # whether a retry is recommended
    raw_text: str                           # raw LLM output


# --- ScribeModel ---

@dataclass
class TraceSummaryRequest:
    """Input for log/trace summarization."""
    trace: Dict[str, Any]                   # plan + actions + outcomes, etc.
    purpose: str                            # "human_doc", "context_chunk", "debug"


@dataclass
class TraceSummaryResponse:
    """Output from ScribeModel."""
    summary: str                            # compressed narrative summary
    keywords: List[str]                     # keywords for indexing
    suggested_tags: List[str]               # tags/categories (e.g. "LV", "boilers")
    raw_text: str                           # raw LLM output


```

These schemas are **internal to M2** but can be used by M8/M9/M10 for clarity and type safety.

---

## 4. Backend Abstraction

### 4.1 Backend interface
Python:
```
# src/llm_stack/backend.py

from typing import Protocol, List, Optional


class LLMBackend(Protocol):
    """Simple interface around a local text generation backend."""

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int,
        temperature: float,
        stop: Optional[List[str]] = None,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Generate a text completion for the given prompt."""
        ...


```

4.2 Role presets
Python:
```
# src/llm_stack/presets.py

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class RolePreset:
    """Configuration for a logical LLM role."""
    name: str
    temperature: float
    max_tokens: int
    system_prompt: Optional[str] = None
    stop: Optional[List[str]] = None


```

4.3 Example: llama.cpp-based backend (for Qwen GGUF)
Python:
```
# src/llm_stack/backend_llamacpp.py

from typing import List, Optional
from pathlib import Path
from llama_cpp import Llama

from .backend import LLMBackend


class LlamaCppBackend(LLMBackend):
    """LLMBackend implementation using llama.cpp local inference."""

    def __init__(
        self,
        model_path: str,
        context_length: int = 8192,
        n_gpu_layers: int = 35,
    ) -> None:
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(path)

        self._llm = Llama(
            model_path=str(path),
            n_ctx=context_length,
            n_gpu_layers=n_gpu_layers,
        )

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int,
        temperature: float,
        stop: Optional[List[str]] = None,
        system_prompt: Optional[str] = None,
    ) -> str:
        full_prompt = (
            prompt
            if system_prompt is None
            else f"{system_prompt}\n\n{prompt}"
        )

        output = self._llm(
            full_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop or [],
        )
        text = output["choices"][0]["text"]
        return text.strip()

```

Later you can implement a dedicated Qwen backend; the interface stays the same.

---

## 5. Role-specific Implementations

All role implementations live in `src/llm_stack/` and conform to `spec.llm` interfaces from M1.

### 5.1 PlanCodeModel implementation

Implements both planning and codegen using a shared backend + role preset.
Python:
```
# src/llm_stack/plan_code.py

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
    """PlanCodeModel backed by a single local LLM with role presets."""

    def __init__(self, backend: LLMBackend, preset: RolePreset) -> None:
        self._backend = backend
        self._preset = preset

    # --- Planning ---

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

    # --- Codegen ---

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

### 5.2 ErrorModel implementation

Takes an `ErrorContext` and returns `ErrorAnalysis`. Used by higher layers when something goes wrong.
Python:
```
# src/llm_stack/error_model.py

from __future__ import annotations

import json
import logging

from spec.llm import ErrorModel as ErrorModelInterface
from .schema import ErrorContext, ErrorAnalysis
from .backend import LLMBackend
from .presets import RolePreset
from .json_utils import load_json_or_none
from .log_files import log_llm_call


logger = logging.getLogger(__name__)


class ErrorModelImpl(ErrorModelInterface):
    """ErrorModel backed by the same local LLM, low temperature."""

    def __init__(self, backend: LLMBackend, preset: RolePreset) -> None:
        self._backend = backend
        self._preset = preset

    def _build_prompt(self, ctx: ErrorContext) -> str:
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

        data, err = load_json_or_none(raw, context="ErrorModel.analyze_failure")
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

### 5.3 ScribeModel implementation

Summarizes traces/logs for human docs and long-term context.
Python:
```
# src/llm_stack/scribe.py

from __future__ import annotations

import json
import logging

from spec.llm import ScribeModel as ScribeModelInterface
from .schema import TraceSummaryRequest, TraceSummaryResponse
from .backend import LLMBackend
from .presets import RolePreset
from .json_utils import load_json_or_none
from .log_files import log_llm_call


logger = logging.getLogger(__name__)


class ScribeModelImpl(ScribeModelInterface):
    """ScribeModel for summarizing traces and logs."""

    def __init__(self, backend: LLMBackend, preset: RolePreset) -> None:
        self._backend = backend
        self._preset = preset

    def _build_prompt(self, req: TraceSummaryRequest) -> str:
        trace_json = json.dumps(req.trace, indent=2)

        prompt = f"""
You are a scribe for a Minecraft GTNH automation agent.

Summarize the following trace for purpose: "{req.purpose}"

Trace:
{trace_json}

Produce a concise summary plus keywords and tags.

Respond ONLY with valid JSON:
{{
  "summary": "text",
  "keywords": ["...", "..."],
  "suggested_tags": ["...", "..."]
}}
"""
        return prompt.strip()

    def summarize_trace(self, req: TraceSummaryRequest) -> TraceSummaryResponse:
        prompt = self._build_prompt(req)
        logger.debug("ScribeModel prompt: %s", prompt)

        raw = self._backend.generate(
            prompt,
            max_tokens=self._preset.max_tokens,
            temperature=self._preset.temperature,
            stop=self._preset.stop,
            system_prompt=self._preset.system_prompt,
        )
        logger.debug("ScribeModel raw output: %s", raw)

        # File-level log for replay / forensics
        log_llm_call(
            role="scribe",
            operation="summarize_trace",
            prompt=prompt,
            raw_response=raw,
            extra={
                "purpose": req.purpose,
            },
        )

        data, err = load_json_or_none(raw, context="ScribeModel.summarize_trace")
        if data is None:
            # Again, don't hard-crash; give a structured, degraded response.
            logger.error("ScribeModel JSON decode error: %s", err)
            return TraceSummaryResponse(
                summary="json_parse_error",
                keywords=[],
                suggested_tags=[],
                raw_text=raw,
            )

        return TraceSummaryResponse(
            summary=data.get("summary", ""),
            keywords=data.get("keywords", []),
            suggested_tags=data.get("suggested_tags", []),
            raw_text=raw,
        )



```
## 6. Orchestration: LLMStack Manager

`LLMStack` wires the environment config, backend, presets, and role facades into one object.
Python:
```
# src/llm_stack/stack.py

from env.loader import load_environment

from spec.llm import PlanCodeModel, ErrorModel, ScribeModel

from .backend_llamacpp import LlamaCppBackend
from .presets import RolePreset
from .plan_code import PlanCodeModelImpl
from .error_model import ErrorModelImpl
from .scribe import ScribeModelImpl


class LLMStack:
    """Convenience holder for all role-specific facades over a single backend."""

    def __init__(self) -> None:
        # Load environment profile (M0)
        env = load_environment()
        mp = env.model_profile  # ModelProfile(name, backend, models)

        if mp.backend != "llama_cpp":
            raise NotImplementedError(f"Unsupported backend: {mp.backend}")

        # ---- Choose primary model config ----
        # Prefer a "primary" entry if present; otherwise just take the first.
        if "primary" in mp.models:
            primary_cfg = mp.models["primary"]
        else:
            # fallback: first entry in the models dict
            primary_cfg = next(iter(mp.models.values()))

        # Shared backend for all roles
        self._backend = LlamaCppBackend(
            model_path=primary_cfg.path,
            context_length=primary_cfg.context_length,
        )

        # ---- Role presets (hardcoded for now; can be moved to YAML later) ----

        self._plan_code_preset = RolePreset(
            name="plan_code",
            temperature=0.2,
            max_tokens=768,
            system_prompt=(
                "You are the planning and coding brain for a Minecraft GTNH "
                "automation agent. You strictly follow the requested JSON or "
                "code-only output format."
            ),
            stop=None,
        )

        self._error_preset = RolePreset(
            name="error",
            temperature=0.0,
            max_tokens=512,
            system_prompt=(
                "You are a precise error analyst for an LLM stack. You only "
                "respond with valid JSON describing the error and suggested fix."
            ),
            stop=None,
        )

        self._scribe_preset = RolePreset(
            name="scribe",
            temperature=0.3,
            max_tokens=1024,
            system_prompt=(
                "You are a concise technical scribe for a Minecraft GTNH agent. "
                "You summarize traces and produce JSON with summary, keywords, and tags."
            ),
            stop=None,
        )

        # ---- Instantiate facades ----
        self._plan_code = PlanCodeModelImpl(self._backend, self._plan_code_preset)
        self._error = ErrorModelImpl(self._backend, self._error_preset)
        self._scribe = ScribeModelImpl(self._backend, self._scribe_preset)

    # ---- Public accessors ----

    @property
    def plan_code(self) -> PlanCodeModel:
        return self._plan_code

    @property
    def error(self) -> ErrorModel:
        return self._error

    @property
    def scribe(self) -> ScribeModel:
        return self._scribe


```

Everything else in the system just calls `LLMStack().plan_code`, `LLMStack().error`, `LLMStack().scribe`.

---

## 7. Local Testing / Simulation Strategy

M2 must be testable without real Minecraft or real models.

### 7.1 Fake backend
Python:
```
# tests/test_llm_stack_fake_backend.py

from typing import List, Optional

from llm_stack.backend import LLMBackend
from llm_stack.presets import RolePreset
from llm_stack.plan_code import PlanCodeModelImpl
from spec.types import Observation


class FakeBackend(LLMBackend):
    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int,
        temperature: float,
        stop: Optional[List[str]] = None,
        system_prompt: Optional[str] = None,
    ) -> str:
        # Return a fixed JSON plan regardless of input
        return '{"steps": [{"skill": "test_skill", "params": {"x": 1}}], "notes": "ok"}'


def test_plan_code_planner_with_fake_backend():
    backend = FakeBackend()
    preset = RolePreset(
        name="plan_code",
        temperature=0.2,
        max_tokens=512,
    )
    model = PlanCodeModelImpl(backend, preset)

    obs = Observation(
        json_payload={"foo": "bar"},
        text_summary="test observation",
    )

    plan = model.plan(
        observation=obs,
        goal="do something testy",
        skill_descriptions={"test_skill": {"description": "dummy"}},
        constraints={},
    )

    assert plan["steps"][0]["skill"] == "test_skill"
    assert plan["steps"][0]["params"]["x"] == 1


```

Similar fake-backend tests for `ErrorModelImpl` and `ScribeModelImpl`.

### 7.2 Real-model smoke test

Once `qwen2.5-coder-q5` is installed and configured:

- Configure `primary_model` and role presets in `models.yaml`.
    
- Run a small script to test `LLMStack().plan_code.plan(...)`.
    
- Verify:
    
    - It returns a dict with `steps` and `notes`.
        
    - No crashes, sane latency.
        

### 7.3 Logging & replay

- Configure logger to JSON lines.
    
- For each M2 call, log:
    
    - role
        
    - operation
        
    - truncated prompt
        
    - truncated raw response
        
    - timing & token metadata (if available)
        
- Later, build a replay script to:
    
    - Re-run stored prompts against updated models for regression testing.
        

---

## 8. Libraries, APIs, Repos to Examine

- **Local inference backends:**
    
    - `llama_cpp_python` for GGUF (initial target).
        
    - Custom Qwen backends as needed.
        
- **Structured prompting & parsing:**
    
    - Repos demonstrating tool-calling / JSON-mode patterns.
        
- **Logging:**
    
    - Python `logging` with a JSON formatter.
        
    - Optional: `structlog` for structured logs.
        

---

## 9. Completion Criteria for M2

M2 is “done enough” when:

1. **Interfaces implemented**
    
    - `PlanCodeModelImpl` fully implements `PlanCodeModel`.
        
    - `ErrorModelImpl` implements `ErrorModel`.
        
    - `ScribeModelImpl` implements `ScribeModel`.
        
2. **Single-model, multi-role topology**
    
    - Default config uses **one** model (e.g. `primary` → Qwen 2.5 Coder) for all roles.
        
    - Role differences (temperature, max_tokens, system prompts, stop sequences) are handled by `RolePreset`s, not separate model loads.
        
3. **Config-driven behavior**
    
    - `LLMStack` reads `model_profiles` from `models.yaml` via `EnvProfile.model_profile`.
        
    - Switching models or changing temps / max_tokens / context length is done in config only (no code changes required).
        
4. **ErrorModel integrated**
    
    - Callers can pass parse failures and malformed outputs into `ErrorModelImpl` via `ErrorContext`.
        
    - `ErrorAnalysis` is structured enough to drive concrete actions:
        
        - retry vs. no-retry
            
        - suggested prompt/parameter adjustments
            
        - basic classification of failure type.
            
5. **ScribeModel usable**
    
    - `ScribeModelImpl.summarize_trace(...)` returns a `TraceSummaryResponse` with:
        
        - non-empty `summary`
            
        - list-valued `keywords`
            
        - list-valued `suggested_tags`
            
    - Output is stable and JSON-structured enough to be:
        
        - stored in a context DB
            
        - used in human-facing documentation.
            
6. **Tests with fake backend pass in CI**
    
    - PlanCode behavior is tested with a `FakeBackend` that returns fixed JSON.
        
    - ErrorModel behavior is tested with a `FakeBackend` that returns fixed error JSON.
        
    - ScribeModel behavior is tested with a `FakeBackend` that returns fixed summary JSON.
        
    - These tests run in CI **without** requiring local models or llama.cpp.
        
7. **Real-model smoke tests pass (manual)**
    
    - On a machine with the configured Qwen model installed:
        
        - `scripts/smoke_llm_stack.py` calls `LLMStack().plan_code.plan(...)` and returns a valid plan dict with `steps` and `notes`.
            
        - `scripts/smoke_error_model.py` returns a plausible `ErrorAnalysis`.
            
        - `scripts/smoke_scribe_model.py` returns a coherent `TraceSummaryResponse`.
            
    - No hard crashes; latency is acceptable for offline planning / analysis.
        
8. **Logging wired**
    
    - PlanCode, ErrorModel, and ScribeModel log:
        
        - prompts at debug level
            
        - raw model outputs at debug level (optionally truncated).
            
    - Log format is machine-readable (e.g. line-oriented JSON or easily parseable text) so future replay / regression tools can consume it.
        
9. **Timeouts and resource limits are enforced**
    
    - Long-running calls are bounded at the caller level (e.g. max_tokens kept modest for planning, error analysis, and summarization).
        
    - Context length is respected via config (`context_length` in `models.yaml`), matching the backend settings.
        

Once these conditions are met, M2 provides a solid, production-grade **local LLM plumbing layer**:

- Single Qwen brain
    
- Three clean roles
    
- Config-driven behavior
    
- Tested with and without real models
    

At that point, any further polishing belongs to higher-level modules (M3/M4/M5/M8), not the LLM stack itself.