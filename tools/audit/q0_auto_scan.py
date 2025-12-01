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
