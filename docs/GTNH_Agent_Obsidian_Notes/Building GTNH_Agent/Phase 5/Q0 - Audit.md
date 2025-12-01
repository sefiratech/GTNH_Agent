## Q0 – Audit-only Module Workflow

### 0. Scope & Non-Goals

**Scope**

- Audit all **7 qualities** across the actual codebase.
    
- Fill the **Audit Matrix** with real status per bullet (`missing | stub | partial | complete`).
    
- Identify **collisions** where multiple modules own the same concern.
    
- Tighten the **LLM role model boundaries** on paper (Planner / Critic / ErrorModel / Scribe / PlanCodeModel), including scheduling rules.
    

**Strict Non-Goals**

- No refactors.
    
- No renames.
    
- No “quick fixes.”
    
- Q0 produces **truth**, not changes.
    

---

## 1. Artifacts Q0 Must Produce

Q0 is done when these exist and are filled, not when your feelings say “good enough.”

1. **Audit Matrix status table**
    
    - One row per **quality 1–7**.
        
    - Columns at least:
        
        - `M2`, `M3`, `M4`, `M5`, `M6`, `M7`, `M8`, `M9`, `M10`, `M11`, `runtime/*`, `spec/*`
            
        - Each cell: `missing | stub | partial | complete` + _1–2 line note_.
            
2. **Collision log**
    
    - Short list of “ownership conflicts,” e.g.:
        
        - “Self-eval logic duplicated in `agent_loop.loop` and `llm_stack.plan_code`.”
            
        - “Goal selection partly in curriculum engine, partly in ad-hoc prompts.”
            
3. **Role boundary spec (LLM roles)**
    
    - One page defining:
        
        - **Responsibilities** for: `Planner`, `PlanCodeModel`, `CriticModel`, `ErrorModel`, `Scribe`.
            
        - **Inputs/outputs** per role (schemas).
            
        - **Scheduling rules** (who calls what, in what order, under which conditions).
            
    - Includes status: `CriticModel` = `missing/stub/partial/complete`.
        
4. **Per-quality mini-summary**
    
    - For each quality 1–7:
        
        - “What exists.”
            
        - “What is missing.”
            
        - “Where it lives (modules/files).”
            
        - “Any surprises / landmines.”
            

---

## 2. Global Q0 Loop

This is the main loop you run for each of the **7 qualities**.

For each quality `q` in `[1..7]`:

1. **Clarify ground truth from the checklist**
    
    - Rephrase the checklist into 3 pieces:
        
        - Required **types / schema** (dataclasses, fields).
            
        - Required **functions / behaviors**.
            
        - Required **integration points** (who calls who).
            
    - Put this at the top of that quality’s section in the Audit Matrix.
        
2. **Code walk**
    
    - Walk the modules listed in the quality’s “Target modules”:
        
        - Open relevant files in:
            
            - Primary module (e.g. `M8` for self-eval).
                
            - Secondary modules (e.g. `M2`, `M9`).
                
            - Any `spec/*` or `runtime/*` called out.
                
    - For each checklist bullet:
        
        - Answer exactly: **present? where? how complete?**
            
3. **Status labeling**
    
    - For every bullet in that quality’s checklist:
        
        - Mark `missing | stub | partial | complete`.
            
    - Collapse bullet-level statuses up into the Matrix cell per module:
        
        - E.g. if most bullets present but some missing: `partial` + 1–2 lines.
            
4. **Collisions pass**
    
    - While auditing, note any of:
        
        - Same concern implemented in 2+ modules.
            
        - Same event / schema defined twice.
            
        - LLM role doing work that belongs to a different role.
            
    - Add a collision line:
        
        - `Quality 1 / Self-eval: M2.ErrorModel doing plan evaluation that belongs in CriticModel.`
            
5. **Per-quality mini-summary**
    
    - End each quality with:
        
        - **Status**: “Mostly stub,” “Half wired,” “Basically working but messy,” etc.
            
        - **Top 2–3 blockers** for making it real in Q1.
            
    - Log this where the Audit Matrix can reference it.
        

Once all 7 qualities have gone through this loop, you freeze Q0 and only then move to integration planning.


## Script 1: Q0 Auto Scanner (types/functions/config presence → Matrix hints)

This one walks the tree and checks for required types / functions / strings in the right modules, then outputs a markdown summary and a JSON dump you can fold into the Audit Matrix.
```python
# path: tools/audit/q0_auto_scan.py
#!/usr/bin/env python3
"""
Q0 Auto Scanner

Semi-automated helper for Phase 5 / Q0 audit.

- Encodes expectations for each quality (1–7) as symbol checks.
- Scans code/config to see which symbols are present and where.
- Heuristically assigns module-level status:
    missing | stub | partial | complete
  based purely on presence coverage (not semantic correctness).
- Emits:
    - Markdown summary to stdout
    - JSON dump to tools/audit/q0_auto_scan_output.json

This does NOT replace the human Q0 pass; it just pre-fills a bunch
of "present? where?" so you don't have to manually grep everything.
"""

import ast
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]


Kind = Literal["class", "function", "dataclass", "event", "config_key", "string"]


@dataclass
class Expectation:
    quality_id: int                # 1..7
    quality_name: str              # e.g. "Self-evaluation + Retry"
    module: str                    # e.g. "M8", "M2", "M9"
    kind: Kind                     # how to search
    name: str                      # symbol or key
    search_roots: List[str]        # relative dirs from repo root
    file_glob: str = "*.py"        # override for yaml/json
    note: str = ""                 # human note for context


@dataclass
class Finding:
    expectation: Expectation
    present: bool
    locations: List[str]


def _iter_files(root: Path, glob_pattern: str) -> List[Path]:
    return [p for p in root.rglob(glob_pattern) if p.is_file()]


def _scan_py_for_symbol(files: List[Path], kind: Kind, name: str) -> List[str]:
    locations: List[str] = []
    for f in files:
        try:
            src = f.read_text(encoding="utf-8")
        except Exception:
            continue
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue

        found = False

        if kind in ("class", "dataclass"):
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef) and node.name == name:
                    found = True
                    break

        if kind == "function":
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name == name:
                    found = True
                    break

        if kind == "string":
            if name in src:
                found = True

        if kind == "event":
            # Heuristic: look for class/enum entries or string constants
            if name in src:
                found = True

        if found:
            locations.append(str(f.relative_to(REPO_ROOT)))
    return locations


def _scan_text_for_key(files: List[Path], name: str) -> List[str]:
    locations: List[str] = []
    key_pattern = re.compile(rf"\b{name}\b")
    for f in files:
        try:
            text = f.read_text(encoding="utf-8")
        except Exception:
            continue
        if key_pattern.search(text):
            locations.append(str(f.relative_to(REPO_ROOT)))
    return locations


def run_expectation_scan(exp: Expectation) -> Finding:
    locations: List[str] = []

    # Resolve search roots
    root_paths = [REPO_ROOT / r for r in exp.search_roots]
    all_files: List[Path] = []
    for rp in root_paths:
        if rp.exists():
            all_files.extend(_iter_files(rp, exp.file_glob))

    if exp.kind in ("class", "function", "dataclass", "event", "string"):
        locations = _scan_py_for_symbol(all_files, exp.kind, exp.name)
    elif exp.kind == "config_key":
        locations = _scan_text_for_key(all_files, exp.name)
    else:
        locations = []

    return Finding(expectation=exp, present=bool(locations), locations=locations)


def _status_from_ratio(present: int, total: int) -> str:
    if total == 0:
        return "missing"
    ratio = present / total
    if ratio == 0:
        return "missing"
    if ratio < 0.34:
        return "stub"
    if ratio < 1.0:
        return "partial"
    return "complete"


def aggregate_findings(findings: List[Finding]) -> Dict[int, Dict[str, Dict[str, object]]]:
    """
    Returns:
        {quality_id: {module: {"present": int, "total": int, "status": str, "details": [...]}}}
    """
    agg: Dict[int, Dict[str, Dict[str, object]]] = {}

    for f in findings:
        qid = f.expectation.quality_id
        mod = f.expectation.module
        agg.setdefault(qid, {})
        if mod not in agg[qid]:
            agg[qid][mod] = {
                "present": 0,
                "total": 0,
                "details": [],
            }
        cell = agg[qid][mod]
        cell["total"] += 1
        if f.present:
            cell["present"] += 1
            detail = f"{f.expectation.kind} '{f.expectation.name}' FOUND in {', '.join(f.locations[:3])}"  # noqa: E501
        else:
            detail = f"{f.expectation.kind} '{f.expectation.name}' MISSING (expected in {f.expectation.search_roots})"  # noqa: E501
        if f.expectation.note:
            detail += f" – {f.expectation.note}"
        cell["details"].append(detail)

    # assign status
    for qid, modules in agg.items():
        for mod, cell in modules.items():
            cell["status"] = _status_from_ratio(cell["present"], cell["total"])
    return agg


def print_markdown_summary(agg: Dict[int, Dict[str, Dict[str, object]]], qualities_map: Dict[int, str]) -> None:
    # Determine all modules seen
    modules: List[str] = sorted({m for modules in agg.values() for m in modules.keys()})

    print("# Q0 Auto-scan Summary\n")
    print("> This is heuristic: it only checks symbol presence, not correctness.\n")

    # One table per quality
    for qid in sorted(agg.keys()):
        qname = qualities_map.get(qid, f"Quality {qid}")
        print(f"## Quality {qid}: {qname}\n")
        print("| Module | Status | Present / Total | Notes (auto-generated) |")
        print("|--------|--------|-----------------|-------------------------|")
        modules_for_q = agg[qid]
        for mod in modules:
            if mod not in modules_for_q:
                continue
            cell = modules_for_q[mod]
            status = cell["status"]
            present = cell["present"]
            total = cell["total"]
            # show only first 2 notes inline to keep it readable
            notes = cell["details"][:2]
            notes_str = "<br>".join(notes)
            print(f"| {mod} | `{status}` | {present} / {total} | {notes_str} |")
        print()


def dump_json(findings: List[Finding], agg: Dict[int, Dict[str, Dict[str, object]]]) -> None:
    output = {
        "findings": [
            {
                "expectation": asdict(f.expectation),
                "present": f.present,
                "locations": f.locations,
            }
            for f in findings
        ],
        "aggregate": agg,
    }
    out_path = REPO_ROOT / "tools" / "audit" / "q0_auto_scan_output.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\nWrote detailed JSON to {out_path.relative_to(REPO_ROOT)}")


def build_expectations() -> Tuple[List[Expectation], Dict[int, str]]:
    """
    Encodes a *minimal but useful* subset of expectations.
    You can expand this as Q0 matures.
    """
    qualities: Dict[int, str] = {
        1: "Self-evaluation + Retry Loop",
        2: "Dynamic Skill Evolution & Versioning",
        3: "Hierarchical Planning",
        4: "Experience Memory",
        5: "Curriculum-driven Goal Selection",
        6: "Structured LLM-role Separation",
        7: "Lightweight Predictive World-model",
    }

    exp: List[Expectation] = []

    # --- Quality 1: Self-eval + retry (M8, M2, M9) ---

    exp.extend([
        # M8 agent loop steps
        Expectation(1, qualities[1], "M8", "function", "propose_plan",
                    ["src/agent_loop"], "*.py", "Agent loop planning step"),
        Expectation(1, qualities[1], "M8", "function", "evaluate_plan",
                    ["src/agent_loop"], "*.py", "Agent loop evaluation step"),
        Expectation(1, qualities[1], "M8", "function", "maybe_retry_plan",
                    ["src/agent_loop"], "*.py", "Retry gate based on policy"),
        Expectation(1, qualities[1], "M8", "function", "evaluate_outcome",
                    ["src/agent_loop"], "*.py", "End-of-episode evaluation"),
        # Dataclasses (schema)
        Expectation(1, qualities[1], "M8", "class", "PlanAttempt",
                    ["src/agent_loop", "src/spec"], "*.py", "Dataclass for a single plan attempt"),
        Expectation(1, qualities[1], "M8", "class", "PlanEvaluation",
                    ["src/agent_loop", "src/spec"], "*.py", "Dataclass for plan evaluation result"),
        Expectation(1, qualities[1], "M8", "class", "RetryPolicy",
                    ["src/agent_loop", "src/spec"], "*.py", "Dataclass for retry policy"),
        # Virtue-based evaluation helper
        Expectation(1, qualities[1], "M4", "function", "evaluate_plan_with_virtues",
                    ["src/virtues", "src/integration"], "*.py", "Virtue-aware plan evaluation"),
        # M2 critic/error structured outputs
        Expectation(1, qualities[1], "M2", "string", "failure_type",
                    ["src/llm_stack"], "*.py", "Error/Critic response field"),
        Expectation(1, qualities[1], "M2", "string", "severity",
                    ["src/llm_stack"], "*.py", "Error/Critic response field"),
        Expectation(1, qualities[1], "M2", "string", "fix_suggestions",
                    ["src/llm_stack"], "*.py", "Error/Critic response field"),
        # M9 events
        Expectation(1, qualities[1], "M9", "event", "PlanEvaluated",
                    ["src/monitoring"], "*.py", "Monitoring event"),
        Expectation(1, qualities[1], "M9", "event", "PlanRetried",
                    ["src/monitoring"], "*.py", "Monitoring event"),
        Expectation(1, qualities[1], "M9", "event", "PlanAbandoned",
                    ["src/monitoring"], "*.py", "Monitoring event"),
    ])

    # --- Quality 2: Skill evolution & versioning (M5, M10) ---

    exp.extend([
        # Skill schema fields as config keys
        Expectation(2, qualities[2], "M5", "config_key", "version",
                    ["config/skills"], "*.yaml", "Skill version field in YAML"),
        Expectation(2, qualities[2], "M5", "config_key", "status",
                    ["config/skills"], "*.yaml", "Skill status field in YAML"),
        Expectation(2, qualities[2], "M5", "config_key", "origin",
                    ["config/skills"], "*.yaml", "Skill origin field in YAML"),
        Expectation(2, qualities[2], "M5", "config_key", "metrics",
                    ["config/skills"], "*.yaml", "Metrics struct in skill YAML"),
        # Registry functions
        Expectation(2, qualities[2], "M5", "function", "get_latest_skill",
                    ["src/skills"], "*.py", "Registry API for latest skill version"),
        Expectation(2, qualities[2], "M5", "function", "list_skill_versions",
                    ["src/skills"], "*.py", "Registry API for all skill versions"),
        Expectation(2, qualities[2], "M5", "function", "register_skill_candidate",
                    ["src/skills"], "*.py", "Registry API to register candidate skills"),
        Expectation(2, qualities[2], "M5", "function", "mark_skill_version_deprecated",
                    ["src/skills"], "*.py", "Registry API for deprecating versions"),
        # M10 integration hints
        Expectation(2, qualities[2], "M10", "string", "SkillLearningManager",
                    ["src/learning"], "*.py", "Learning manager for skills"),
        Expectation(2, qualities[2], "M10", "string", "create_candidate_skill",
                    ["src/learning"], "*.py", "Any method that creates candidate skills"),
        Expectation(2, qualities[2], "M10", "string", "update_skill_metrics",
                    ["src/learning"], "*.py", "Any method that updates metrics"),
    ])

    # --- Quality 3: Hierarchical planning (M8, M2, spec) ---

    exp.extend([
        # Spec layer types
        Expectation(3, qualities[3], "spec/*", "class", "AgentGoal",
                    ["src/spec"], "*.py", "Spec: Agent goal type"),
        Expectation(3, qualities[3], "spec/*", "class", "TaskPlan",
                    ["src/spec"], "*.py", "Spec: Task-level plan"),
        Expectation(3, qualities[3], "spec/*", "class", "SkillInvocation",
                    ["src/spec"], "*.py", "Spec: Skill invocation entry"),
        # Planner modes
        Expectation(3, qualities[3], "M2", "function", "plan_goal",
                    ["src/llm_stack"], "*.py", "Planner mode: goal → tasks"),
        Expectation(3, qualities[3], "M2", "function", "plan_task",
                    ["src/llm_stack"], "*.py", "Planner mode: task → skills"),
        # State machine labels
        Expectation(3, qualities[3], "M8", "string", "GoalSelection",
                    ["src/agent_loop"], "*.py", "Agent loop state name"),
        Expectation(3, qualities[3], "M8", "string", "TaskPlanning",
                    ["src/agent_loop"], "*.py", "Agent loop state name"),
        Expectation(3, qualities[3], "M8", "string", "SkillResolution",
                    ["src/agent_loop"], "*.py", "Agent loop state name"),
        Expectation(3, qualities[3], "M8", "string", "Execution",
                    ["src/agent_loop"], "*.py", "Agent loop state name"),
        Expectation(3, qualities[3], "M8", "string", "Review",
                    ["src/agent_loop"], "*.py", "Agent loop state name"),
    ])

    # --- Quality 4: Experience memory (M10, M8, Scribe, M9) ---

    exp.extend([
        # Experience schema fields
        Expectation(4, qualities[4], "M10", "config_key", "problem_signature",
                    ["src/learning", "tests"], "*.py", "Experience field"),
        Expectation(4, qualities[4], "M10", "config_key", "final_outcome",
                    ["src/learning", "tests"], "*.py", "Experience field"),
        Expectation(4, qualities[4], "M10", "config_key", "virtue_scores",
                    ["src/learning", "tests"], "*.py", "Experience field"),
        Expectation(4, qualities[4], "M10", "config_key", "lessons",
                    ["src/learning", "tests"], "*.py", "Experience field"),
        # Buffer API
        Expectation(4, qualities[4], "M10", "function", "append_experience",
                    ["src/learning"], "*.py", "Replay / experience buffer append"),
        Expectation(4, qualities[4], "M10", "function", "query_similar_experiences",
                    ["src/learning"], "*.py", "Replay query function"),
        # M8 hooks
        Expectation(4, qualities[4], "M8", "function", "build_experience_from_episode",
                    ["src/agent_loop", "src/integration"], "*.py", "Mapper from episode → experience"),
        Expectation(4, qualities[4], "M8", "string", "append_experience",
                    ["src/agent_loop", "src/integration"], "*.py", "Call site into buffer"),
        # Scribe helper
        Expectation(4, qualities[4], "M2", "function", "summarize_episode_for_memory",
                    ["src/llm_stack"], "*.py", "Scribe helper for memory"),
    ])

    # --- Quality 5: Curriculum-driven goal selection (M11 + M8/M3/M4) ---

    exp.extend([
        # Curriculum engine outputs
        Expectation(5, qualities[5], "M11", "class", "CurriculumConfig",
                    ["src/curriculum"], "*.py", "Core curriculum schema"),
        Expectation(5, qualities[5], "M11", "class", "PhaseConfig",
                    ["src/curriculum"], "*.py", "Per-phase config"),
        Expectation(5, qualities[5], "M11", "function", "view",
                    ["src/curriculum"], "*.py", "Curriculum.view(...) API"),
        # Metadata fields
        Expectation(5, qualities[5], "M11", "config_key", "required_tech_state",
                    ["config/curricula", "src/curriculum"], "*.yaml", "Tech state requirement"),
        Expectation(5, qualities[5], "M11", "config_key", "preferred_virtue_context",
                    ["config/curricula", "src/curriculum"], "*.yaml", "Virtue context key"),
        Expectation(5, qualities[5], "M11", "config_key", "entry_conditions",
                    ["config/curricula", "src/curriculum"], "*.yaml", "Phase entry conditions"),
        Expectation(5, qualities[5], "M11", "config_key", "exit_conditions",
                    ["config/curricula", "src/curriculum"], "*.yaml", "Phase exit conditions"),
        # M8 top-of-loop goal selection
        Expectation(5, qualities[5], "M8", "string", "curriculum.next_goal",
                    ["src/agent_loop", "src/curriculum"], "*.py", "Curriculum is goal source"),
    ])

    # --- Quality 6: LLM-role separation (M2 + config + spec) ---

    exp.extend([
        # Role config in llm_roles.yaml
        Expectation(6, qualities[6], "M2", "config_key", "planner",
                    ["config"], "llm_roles.yaml", "Planner role definition"),
        Expectation(6, qualities[6], "M2", "config_key", "critic",
                    ["config"], "llm_roles.yaml", "Critic role definition"),
        Expectation(6, qualities[6], "M2", "config_key", "error_model",
                    ["config"], "llm_roles.yaml", "Error model role"),
        Expectation(6, qualities[6], "M2", "config_key", "scribe",
                    ["config"], "llm_roles.yaml", "Scribe role"),
        # LLMStack APIs
        Expectation(6, qualities[6], "M2", "function", "call_planner",
                    ["src/llm_stack"], "*.py", "Role-specific planner call"),
        Expectation(6, qualities[6], "M2", "function", "call_critic",
                    ["src/llm_stack"], "*.py", "Role-specific critic call"),
        Expectation(6, qualities[6], "M2", "function", "call_error_model",
                    ["src/llm_stack"], "*.py", "Role-specific error model call"),
        Expectation(6, qualities[6], "M2", "function", "call_scribe",
                    ["src/llm_stack"], "*.py", "Role-specific scribe call"),
        # Generic calls (we want to know they exist)
        Expectation(6, qualities[6], "runtime/*", "string", "llm_stack.call(",
                    ["src"], "*.py", "Generic LLM calls (should be phased out)"),
    ])

    # --- Quality 7: Predictive world-model (M3 + consumers) ---

    exp.extend([
        Expectation(7, qualities[7], "M3", "class", "WorldModel",
                    ["src/world", "src/semantics"], "*.py", "World model main class"),
        Expectation(7, qualities[7], "M3", "function", "simulate_tech_progress",
                    ["src/world", "src/semantics"], "*.py", "Tech progress simulation"),
        Expectation(7, qualities[7], "M3", "function", "estimate_infra_effect",
                    ["src/world", "src/semantics"], "*.py", "Infra effect estimation"),
        Expectation(7, qualities[7], "M3", "function", "estimate_resource_trajectory",
                    ["src/world", "src/semantics"], "*.py", "Resource trajectory estimation"),
        # Consumers
        Expectation(7, qualities[7], "M8", "string", "simulate_tech_progress",
                    ["src/agent_loop"], "*.py", "Agent loop using world-model"),
        Expectation(7, qualities[7], "M4", "string", "estimate_infra_effect",
                    ["src/virtues"], "*.py", "Virtue scoring using world-model"),
        Expectation(7, qualities[7], "M11", "string", "estimate_resource_trajectory",
                    ["src/curriculum"], "*.py", "Curriculum using world-model"),
    ])

    return exp, qualities


def main() -> None:
    expectations, qualities = build_expectations()
    findings: List[Finding] = []
    for e in expectations:
        findings.append(run_expectation_scan(e))
    agg = aggregate_findings(findings)
    print_markdown_summary(agg, qualities)
    dump_json(findings, agg)


if __name__ == "__main__":
    main()

```


**What this automates for you**

- “Clarify ground truth” → partially automated via structured expectations.
    
- “Code walk” → replaced by AST / text scanning for symbol presence.
    
- “Status labeling” → auto-calculated coverage per module/quality.
    
- You still have to:
    
    - Check if the implementations are _actually_ wired correctly.
        
    - Refine statuses when presence ≠ correctness.
        
    - Fill in collision log & mini-summaries.




## Script 2: LLM Role Usage Reporter (helps with Quality 1 & 6 + collisions)

This one gives you a quick view of where you’re still using generic `llm_stack.call(...)`, where roles bleed, and where ErrorModel / CriticModel are mentioned.
```python
# path: tools/audit/llm_role_usage_report.py
#!/usr/bin/env python3
"""
LLM Role Usage Reporter

Helper for:
- Quality 1 (Self-eval + retry)
- Quality 6 (Structured LLM-role separation)
- Collision detection between Planner / Critic / ErrorModel / Scribe

Scans the repo for:
- Generic vs role-specific LLMStack calls
- Cross-role references (e.g. ErrorModel imported/used in planner)
- Places where CriticModel is referenced or missing

Outputs a markdown report to stdout.
"""

from pathlib import Path
import re
from typing import Dict, List, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]


ROLE_CALL_PATTERNS = {
    "generic_call": re.compile(r"llm_stack\.call\("),
    "call_planner": re.compile(r"llm_stack\.call_planner\("),
    "call_critic": re.compile(r"llm_stack\.call_critic\("),
    "call_error_model": re.compile(r"llm_stack\.call_error_model\("),
    "call_scribe": re.compile(r"llm_stack\.call_scribe\("),
}

ROLE_NAME_PATTERNS = {
    "Planner": re.compile(r"\bPlanner\b"),
    "PlanCodeModel": re.compile(r"\bPlanCodeModel\b"),
    "CriticModel": re.compile(r"\bCriticModel\b"),
    "ErrorModel": re.compile(r"\bErrorModel\b"),
    "Scribe": re.compile(r"\bScribe\b"),
}


def iter_py_files() -> List[Path]:
    return [p for p in REPO_ROOT.rglob("*.py") if p.is_file()]


def scan_llm_calls(files: List[Path]) -> Dict[str, Dict[str, List[int]]]:
    """
    Returns:
        {relative_path: {pattern_name: [line_numbers...]}}
    """
    results: Dict[str, Dict[str, List[int]]] = {}
    for f in files:
        try:
            lines = f.read_text(encoding="utf-8").splitlines()
        except Exception:
            continue

        rel = str(f.relative_to(REPO_ROOT))
        file_hits: Dict[str, List[int]] = {}

        for lineno, line in enumerate(lines, start=1):
            for tag, pattern in ROLE_CALL_PATTERNS.items():
                if pattern.search(line):
                    file_hits.setdefault(tag, []).append(lineno)

        if file_hits:
            results[rel] = file_hits
    return results


def scan_role_name_collisions(files: List[Path]) -> Dict[str, Dict[str, List[int]]]:
    """
    Looks for references to role names anywhere, useful for collisions:
    - ErrorModel in planner
    - CriticModel missing entirely
    """
    results: Dict[str, Dict[str, List[int]]] = {}
    for f in files:
        try:
            lines = f.read_text(encoding="utf-8").splitlines()
        except Exception:
            continue
        rel = str(f.relative_to(REPO_ROOT))
        file_hits: Dict[str, List[int]] = {}
        for lineno, line in enumerate(lines, start=1):
            for tag, pattern in ROLE_NAME_PATTERNS.items():
                if pattern.search(line):
                    file_hits.setdefault(tag, []).append(lineno)
        if file_hits:
            results[rel] = file_hits
    return results


def summarize_llm_calls(call_hits: Dict[str, Dict[str, List[int]]]) -> None:
    print("# LLM Role Usage Report\n")

    print("## Files using generic vs role-specific calls\n")
    print("| File | generic `llm_stack.call` | call_planner | call_critic | call_error_model | call_scribe |")
    print("|------|--------------------------|--------------|-------------|------------------|------------|")

    for rel, hits in sorted(call_hits.items()):
        generic = len(hits.get("generic_call", []))
        planner = len(hits.get("call_planner", []))
        critic = len(hits.get("call_critic", []))
        err = len(hits.get("call_error_model", []))
        scribe = len(hits.get("call_scribe", []))
        print(f"| {rel} | {generic} | {planner} | {critic} | {err} | {scribe} |")

    print("\n> Any non-zero count in `generic llm_stack.call` is a Q1 refactor target for Quality 6.\n")


def summarize_role_collisions(role_hits: Dict[str, Dict[str, List[int]]]) -> None:
    print("## Role name references (collision hints)\n")
    print("| File | Planner | PlanCodeModel | CriticModel | ErrorModel | Scribe |")
    print("|------|---------|--------------|-------------|------------|--------|")

    for rel, hits in sorted(role_hits.items()):
        planner = len(hits.get("Planner", []))
        plan_code = len(hits.get("PlanCodeModel", []))
        critic = len(hits.get("CriticModel", []))
        err = len(hits.get("ErrorModel", []))
        scribe = len(hits.get("Scribe", []))
        print(f"| {rel} | {planner} | {plan_code} | {critic} | {err} | {scribe} |")

    print(
        "\n> Use this to spot collisions: "
        "e.g. ErrorModel referenced inside planner modules, or CriticModel never appearing anywhere."
    )


def main() -> None:
    files = iter_py_files()
    call_hits = scan_llm_calls(files)
    role_hits = scan_role_name_collisions(files)
    summarize_llm_calls(call_hits)
    summarize_role_collisions(role_hits)


if __name__ == "__main__":
    main()

```
**What this automates for you**

- Quickly lists where generic calls still exist.
    
- Shows where each role name appears across the repo.
    
- Makes it trivial to spot:
    
    - planner files that mention ErrorModel,
        
    - total absence of CriticModel,
        
    - or random code calling generic `llm_stack.call`.

### What _can’t_ be automated (and you have to do manually)

- Whether `evaluate_plan` is actually separate from `propose_plan` in behavior, not just as a function name.
    
- Whether the retry policy is sane versus cargo cult.
    
- Whether your world-model actually encodes _useful_ predictions instead of decorative math.
    
- Whether Curriculum is truly the single source of goal selection instead of “plus vibes.”
    

But with these scripts, you at least don’t have to manually grep every symbol like it’s 1998.

---

## 3. Quality-Specific Workflows

Now the zoomed-in workflows for each quality, using your checklists.

---

### Q0.1 – Self-Evaluation + Retry Loop (Quality 1)

**Target modules:** `M8`, `M2`, `M9`

**Checklist recap**

- M8:
    
    - Steps: `propose_plan`, `evaluate_plan`, `maybe_retry`, `evaluate_outcome`.
        
    - Dataclasses: `PlanAttempt`, `PlanEvaluation`, `RetryPolicy`.
        
    - Functions: `evaluate_plan_with_virtues`, `maybe_retry_plan`, `postmortem_plan_failure`.
        
- M2:
    
    - `CriticModel` & `ErrorModel` return:
        
        - `failure_type`, `severity`, `fix_suggestions`.
            
- M9:
    
    - Events: `PlanEvaluated`, `PlanRetried`, `PlanAbandoned`.
        

**Workflow**

1. **M8 loop audit**
    
    - Open `src/agent_loop/loop.py`, `schema.py`, `state.py`.
        
    - Locate:
        
        - Where the plan is produced.
            
        - Where it is evaluated (virtue scores / critic).
            
        - Where retries are decided.
            
        - Where outcome is reflected at the end of the episode.
            
    - For each of:
        
        - `propose_plan`
            
        - `evaluate_plan`
            
        - `maybe_retry`
            
        - `evaluate_outcome`
            
        - Mark: does this exist as a **separate step** or is it implicit / fused?
            
2. **Dataclass audit**
    
    - Search in `agent_loop/schema.py`, `spec/agent_loop.py`, `spec/experience.py` for:
        
        - `PlanAttempt`
            
        - `PlanEvaluation`
            
        - `RetryPolicy`
            
    - Note:
        
        - Exact name used (close enough is fine for Q0).
            
        - Missing fields vs checklist.
            
3. **Function audit**
    
    - Search in `agent_loop` + `virtues` + `integration/validators` for:
        
        - `evaluate_plan_with_virtues`
            
        - `maybe_retry_plan`
            
        - `postmortem_plan_failure`
            
    - Mark each as `missing/stub/partial/complete`.
        
4. **M2 role boundary audit (PlanCodeModel, ErrorModel, CriticModel, Scribe)**
    
    - Open:
        
        - `src/llm_stack/planner.py`
            
        - `src/llm_stack/plan_code.py`
            
        - `src/llm_stack/critic.py`
            
        - `src/llm_stack/error_model.py`
            
        - `src/llm_stack/scribe.py`
            
        - `src/llm_stack/schema.py`
            
        - `config/llm_roles.yaml`
            
    - For **PlanCodeModel / Planner**:
        
        - What do they return now? Does it already encode partial evaluation?
            
    - For **ErrorModel**:
        
        - Confirm output schema has, or can be mapped to:
            
            - `failure_type`, `severity`, `fix_suggestions`.
                
    - For **CriticModel**:
        
        - Status: is there a real `CriticModel` adapter, or just tests / stubs?
            
    - Log where plan evaluation logic currently lives:
        
        - In planner?
            
        - In error model?
            
        - In M8 loop?
            
        - Collision notes here are mandatory.
            
5. **M9 events audit**
    
    - Open `src/monitoring/events.py`, `integration.py`, `bus.py`.
        
    - Find definitions (or absence) of:
        
        - `PlanEvaluated`
            
        - `PlanRetried`
            
        - `PlanAbandoned`
            
    - Check if these events are:
        
        - Defined only.
            
        - Defined + emitted.
            
        - Defined + emitted + consumed by dashboard/logs.
            
6. **Update Matrix + role spec**
    
    - Mark statuses for M8, M2, M9 cells under Quality 1.
        
    - In the **Role boundary spec** doc:
        
        - Write down current reality:
            
            - Which model is doing what piece of self-eval.
                
        - Flag obvious boundary violations:
            
            - e.g. planner doing critique, error model doing plan scoring.
                

---

### Q0.2 – Dynamic Skill Evolution & Versioning (Quality 2)

**Target modules:** `M5`, `M10`

**Checklist recap**

- Skill schema (M5) fields:
    
    - `version`, `status`, `origin`, `metrics{success_rate, avg_cost, avg_risk, last_used_at}`.
        
- Registry functions:
    
    - `get_latest_skill`, `list_skill_versions`, `register_skill_candidate`, `mark_skill_version_deprecated`.
        
- M10 integration:
    
    - Learning output can:
        
        - Create new candidate skill version.
            
        - Update metrics.
            
        - Promote/demote versions.
            

**Workflow**

1. **Skill schema audit**
    
    - Open:
        
        - `src/skills/schema.py`
            
        - `config/skills/*.yaml`
            
    - Check for each field in schema & in YAML:
        
        - `version`
            
        - `status`
            
        - `origin`
            
        - `metrics` subfields
            
    - Label each field: `missing | stub | partial | complete`.
        
2. **Registry audit**
    
    - Open `src/skills/registry.py`.
        
    - Check functions:
        
        - Existence and signature of the four registry functions.
            
    - If they’re partially there (e.g. a simpler `get_skill` that doesn’t handle versions), mark `partial` and note behavior.
        
3. **Learning integration audit**
    
    - Open:
        
        - `src/learning/manager.py`
            
        - `src/learning/evaluator.py`
            
        - `src/learning/synthesizer.py`
            
    - Look for flows:
        
        - From replay buffer → skill metrics update.
            
        - From synthesized outcome → new skill candidate.
            
    - Check if any code even _mentions_:
        
        - `status`, `version`, `origin`, `metrics`.
            
    - Log whether versioning is:
        
        - Planned in comments only.
            
        - Semi-implemented.
            
        - Actually wired end-to-end.
            
4. **Update Matrix**
    
    - Fill M5 and M10 cells under Quality 2 + a 1–2 line note.
        

---

### Q0.3 – Hierarchical Planning (Quality 3)

**Target modules:** `M8`, `M2`, `spec/*`

**Checklist recap**

- Spec defines: `AgentGoal`, `TaskPlan`, `SkillInvocation`.
    
- M2 planner modes:
    
    - `plan_goal → tasks`
        
    - `plan_task → skills`
        
- M8 loop state machine:
    
    - `GoalSelection`, `TaskPlanning`, `SkillResolution`, `Execution`, `Review`.
        

**Workflow**

1. **Spec audit**
    
    - Open:
        
        - `src/spec/agent_loop.py`
            
        - `src/spec/skills.py`
            
        - `src/spec/types.py`
            
    - Confirm presence + shape of:
        
        - `AgentGoal`
            
        - `TaskPlan`
            
        - `SkillInvocation`
            
    - Record: where these are only “suggested” vs. used concretely.
        
2. **Planner modes audit (M2)**
    
    - Open:
        
        - `src/llm_stack/planner.py`
            
        - `src/llm_stack/plan_code.py`
            
        - `src/llm_stack/presets.py`
            
    - Check:
        
        - Are there distinct presets / APIs for:
            
            - goal → tasks
                
            - task → skills
                
        - Or is everything one giant mega-plan?
            
3. **M8 state machine audit**
    
    - Inspect `src/agent_loop/loop.py` and `state.py`:
        
        - Does the loop explicitly model:
            
            - `GoalSelection`
                
            - `TaskPlanning`
                
            - `SkillResolution`
                
            - `Execution`
                
            - `Review`
                
        - Or is it an unstructured linear pipeline?
            
    - Note any “state-ish” enums or dataclasses.
        
4. **Update Matrix + collisions**
    
    - This is prime territory for collisions between:
        
        - Curriculum-driven goals vs. ad-hoc planner prompts.
            
    - Log them.
        

---

### Q0.4 – Experience Memory (Quality 4)

**Target modules:** `M10`, hooks in `M8`, Scribe, `M9`

**Checklist recap**

- Experience schema:
    
    - `problem_signature, goal, plan, attempts, final_outcome, virtue_scores, lessons`.
        
- Buffer:
    
    - `append_experience`, `query_similar_experiences`.
        
- M8:
    
    - `build_experience_from_episode(...)`, `replay_store.append_experience(...)`.
        
- Scribe/M2:
    
    - `summarize_episode_for_memory(trace) -> (...)`.
        

**Workflow**

1. **Schema & buffer audit**
    
    - Open:
        
        - `src/learning/buffer.py`
            
        - `src/learning/schema.py`
            
    - Map actual experience fields against target schema.
        
    - Check buffer methods:
        
        - `append_experience`
            
        - Any function that looks like `query_similar_*`.
            
2. **M8 hook audit**
    
    - Open `src/agent_loop/loop.py`, `integration/episode_logging.py`.
        
    - Find:
        
        - Where episodes are summarized and written.
            
        - Whether there’s a distinct “experience” abstraction or just raw traces.
            
3. **Scribe/M2 audit**
    
    - Open:
        
        - `src/llm_stack/scribe.py`
            
        - Any helper in `scribe`/`json_utils` for “memory / replay”.
            
    - Check for:
        
        - Function that matches or approximates
            
            - `summarize_episode_for_memory(trace) -> (problem_signature, lessons, ...)`.
                
    - If missing, mark `missing` and note where summarization currently happens instead.
        
4. **M9 audit (optional but smart)**
    
    - Check if monitoring stores anything that could be reused as experience snapshots.
        
5. **Update Matrix**
    

---

### Q0.5 – Curriculum-Driven Goal Selection (Quality 5)

**Target modules:** `M11`, integrated with `M8`, `M3`, `M4`

**Checklist recap**

- Curriculum engine outputs:
    
    - `AgentGoal` or equivalent: `goal_text`, `goal_id`, `phase_context`.
        
- Schema metadata:
    
    - `required_tech_state`, `preferred_virtue_context`, `entry_conditions`, `exit_conditions`.
        
- M8:
    
    - Top of loop calls something like:
        
        - `goal = curriculum.next_goal(tech_state, experience_memory?)`
            
    - Planner uses this goal as primary objective.
        

**Workflow**

1. **Curriculum engine audit (M11)**
    
    - Open:
        
        - `src/curriculum/schema.py`
            
        - `src/curriculum/engine.py`
            
    - Check:
        
        - Do phases/goals expose enough to become an `AgentGoal`?
            
        - Are virtue overrides + tech requirements clearly encoded?
            
2. **Metadata audit**
    
    - Compare schema to checklist:
        
        - `required_tech_state`
            
        - `preferred_virtue_context`
            
        - `entry_conditions`
            
        - `exit_conditions`
            
    - Mark each field: `missing/stub/partial/complete`.
        
3. **M8 integration audit**
    
    - Open `src/curriculum/integration_agent_loop.py` and `src/agent_loop/loop.py`.
        
    - Trace:
        
        - How does the agent currently choose goals?
            
        - Is curriculum the source of truth, or is there any ad-hoc goal prompting?
            
    - Note any place where `tech_state` + curriculum view are **ignored**.
        
4. **Update Matrix + collision log**
    
    - Collisions almost guaranteed between:
        
        - Curriculum goals vs. M8 local “task prompts”.
            

---

### Q0.6 – Structured LLM-Role Separation (Quality 6)

**Target modules:** `M2`, `config/llm_roles.yaml`, `spec/llm.py`

This is where the **CriticModel** subtask lives.

**Checklist recap**

- `config/llm_roles.yaml` defines for each role:
    
    - system prompt, temperature, stop tokens, output schema, tool permissions.
        
- `LLMStack` provides:
    
    - `call_planner`, `call_critic`, `call_scribe`, `call_error_model`, etc.
        
- All call sites in M8/M10/M11 go through these role-specific calls.
    

**Workflow**

1. **Role config audit**
    
    - Open `config/llm_roles.yaml`.
        
    - For each role:
        
        - Planner, PlanCodeModel (if separate), Critic, ErrorModel, Scribe.
            
    - Check presence of:
        
        - system prompt
            
        - temperature
            
        - stop tokens
            
        - output schema
            
        - tool permissions
            
    - Mark missing bits per role.
        
2. **LLMStack API audit**
    
    - Open:
        
        - `src/llm_stack/stack.py`
            
        - `src/llm_stack/schema.py`
            
        - `src/spec/llm.py`
            
    - Confirm:
        
        - Do we have explicit `call_planner`, `call_critic`, `call_scribe`, `call_error_model`, etc.?
            
        - Or are some still generic `call(role_id=...)` patterns?
            
3. **Call-site audit (M8/M10/M11)**
    
    - Grep mentally / mechanically for:
        
        - `llm_stack.call(`
            
    - Check call sites in:
        
        - `agent_loop/loop.py`
            
        - `learning/*`
            
        - `curriculum/*`
            
    - For each:
        
        - Note whether it uses **role-specific wrapper** or generic calls.
            
    - Record offenders as Q1 refactor targets.
        
4. **CriticModel subtask**
    
    - In the **Role boundary spec** doc:
        
        - Document current status of Critic:
            
            - config present?
                
            - adapter file present?
                
            - used anywhere?
                
        - Define desired scheduling in words:
            
            - Planner proposes.
                
            - Critic evaluates plan.
                
            - ErrorModel only runs **after** execution failure.
                
        - Mark in Matrix:
            
            - Quality 1 / M2 (self-eval).
                
            - Quality 6 / M2 (role separation).
                
5. **Update Matrix + collisions**
    
    - Any place where:
        
        - ErrorModel is used as Critic.
            
        - Planner both plans and critiques.
            
        - Scribe does non-summarization tasks.
            
    - Add to collision log.
        

---

### Q0.7 – Lightweight Predictive World-Model (Quality 7)

**Target modules:** `M3` (world_model), consumers `M4`, `M8`, `M11`

**Checklist recap**

- World model provides:
    
    - `simulate_tech_progress(...)`
        
    - `estimate_infra_effect(...)`
        
    - `estimate_resource_trajectory(...)`
        
- It uses:
    
    - `TechGraph`
        
    - semantics (items/blocks).
        
- Integration:
    
    - M8: early plan pruning.
        
    - M4: risk/throughput for virtue scoring.
        
    - M11: weighing big long-horizon goals.
        

**Workflow**

1. **World model presence audit**
    
    - Look for:
        
        - `src/world/world_model.py` or equivalent.
            
    - If non-existent:
        
        - Mark Quality 7 as globally `missing` with a clear note.
            
    - If exists:
        
        - Check for the three main functions or equivalents.
            
2. **M3 integration audit**
    
    - Open:
        
        - `src/semantics/tech_state.py`
            
        - `src/semantics/crafting.py`
            
        - `src/semantics/loader.py`
            
    - Note:
        
        - Any existing functions that approximate “prediction” semantics.
            
3. **Consumer audit**
    
    - Scan:
        
        - `virtues/*`
            
        - `agent_loop/*`
            
        - `curriculum/*`
            
    - Look for calls to world-model-ish functions.
        
    - Most likely conclusion: “slot reserved, not yet wired.”
        
4. **Update Matrix**
    

---

## 4. Q0 Exit Criteria

You’re done with Q0 when:

1. **Audit Matrix**:
    
    - Every Quality × Module cell has:
        
        - A status (`missing/stub/partial/complete`).
            
        - A one-line note if anything non-trivial is going on.
            
2. **Collision log**:
    
    - Contains all discovered ambiguous ownerships.
        
3. **Role boundary spec**:
    
    - Documents actual state of Planner / PlanCodeModel / Critic / ErrorModel / Scribe.
        
    - Clarifies CriticModel’s intended role & its current implementation status.
        
4. **Per-quality mini-summaries**:
    
    - For all 7 qualities, each has:
        
        - “Where we are.”
            
        - “Main gaps.”
            
        - “Most obvious next moves for Q1.”
            

Then Q1 can start **implementing** instead of guessing.

There you go: Q0 as an actual module workflow instead of a vibe.