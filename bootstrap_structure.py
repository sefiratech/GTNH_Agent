#!/usr/bin/env python3
"""
Bootstrap the GTNH Agent repo structure.

Creates directories and stub files for:
- config/
- src/ (all modules)
- tests/ (including fakes)
- pyproject.toml (if missing, with a minimal skeleton)

Safe to re-run: it will NOT overwrite existing files.
"""

from pathlib import Path  # convenient, cross-platform path handling

# Root of the repo: this script should reside in gtnh_agent/
ROOT = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Directories to create
# ---------------------------------------------------------------------------

dirs = [
    # Config directories
    ROOT / "config",
    ROOT / "config" / "curricula",
    ROOT / "config" / "skills",
    ROOT / "config" / "skills_candidates",
    ROOT / "config" / "tools",

    # Source root
    ROOT / "src",

    # Core packages under src/
    ROOT / "src" / "agent_loop",
    ROOT / "src" / "bot_core",
    ROOT / "src" / "bot_core" / "nav",
    ROOT / "src" / "bot_core" / "net",
    ROOT / "src" / "env",
    ROOT / "src" / "llm_stack",
    ROOT / "src" / "monitoring",
    ROOT / "src" / "observation",
    ROOT / "src" / "semantics",
    ROOT / "src" / "skills",
    ROOT / "src" / "skills" / "base",
    ROOT / "src" / "spec",
    ROOT / "src" / "learning",
    ROOT / "src" / "virtues",
    ROOT / "src" / "curriculum",

    # Tests
    ROOT / "tests",
    ROOT / "tests" / "fakes",
]

# ---------------------------------------------------------------------------
# Files to create (if not present)
#   key   = Path
#   value = initial file content
# ---------------------------------------------------------------------------

files = {
    # ----------------- config root -----------------
    ROOT / "config" / "env.yaml": "# core environment config (runtime mode, paths, logging)\n",
    ROOT / "config" / "minecraft.yaml": "# minecraft/forge/gtnh profiles (host, port, saves, etc.)\n",
    ROOT / "config" / "models.yaml": "# local model definitions (planner, codegen, critic)\n",
    ROOT / "config" / "hardware.yaml": "# hardware limits (max tokens, concurrent LLM calls, etc.)\n",
    ROOT / "config" / "gtnh_blocks.yaml": "# GTNH block categories & metadata\n",
    ROOT / "config" / "gtnh_items.yaml": "# GTNH item categories & materials\n",
    ROOT / "config" / "gtnh_tech_graph.yaml": "# tech states & dependencies DAG\n",
    ROOT / "config" / "gtnh_recipes.json": "{\n  \"_comment\": \"optional flattened recipes for fast lookup\"\n}\n",
    ROOT / "config" / "virtues.yaml": "# virtue lattice definitions & base weights\n",

    # curricula
    ROOT / "config" / "curricula" / "default_speedrun.yaml": "# default speedrun curriculum\n",
    ROOT / "config" / "curricula" / "eco_factory.yaml": "# eco factory curriculum\n",
    ROOT / "config" / "curricula" / "aesthetic_megabase.yaml": "# aesthetic megabase curriculum\n",

    # skills (specs only; implementations live in src/skills/base/)
    ROOT / "config" / "skills" / "chop_tree.yaml": "# spec for chop_tree skill\n",
    ROOT / "config" / "skills" / "plant_sapling.yaml": "# spec for plant_sapling skill\n",
    ROOT / "config" / "skills" / "feed_coke_ovens.yaml": "# spec for feed_coke_ovens skill\n",

    # tools
    ROOT / "config" / "tools" / "validate_env.py": "def main() -> None:\n    # TODO: validate env/hardware/model configs\n    pass\n\n\nif __name__ == \"__main__\":\n    main()\n",
    ROOT / "config" / "tools" / "print_env.py": "def main() -> None:\n    # TODO: pretty-print active configuration\n    pass\n\n\nif __name__ == \"__main__\":\n    main()\n",

    # ----------------- src packages: __init__.py -----------------
    ROOT / "src" / "__init__.py": "# src package root\n",
    ROOT / "src" / "agent_loop" / "__init__.py": "# agent_loop package\n",
    ROOT / "src" / "bot_core" / "__init__.py": "# bot_core package\n",
    ROOT / "src" / "bot_core" / "nav" / "__init__.py": "# bot_core.nav package\n",
    ROOT / "src" / "bot_core" / "net" / "__init__.py": "# bot_core.net package\n",
    ROOT / "src" / "env" / "__init__.py": "# env package\n",
    ROOT / "src" / "llm_stack" / "__init__.py": "# llm_stack package\n",
    ROOT / "src" / "monitoring" / "__init__.py": "# monitoring package\n",
    ROOT / "src" / "observation" / "__init__.py": "# observation package\n",
    ROOT / "src" / "semantics" / "__init__.py": "# semantics package\n",
    ROOT / "src" / "skills" / "__init__.py": "# skills package\n",
    ROOT / "src" / "skills" / "base" / "__init__.py": "# skills.base package\n",
    ROOT / "src" / "spec" / "__init__.py": "# spec package\n",
    ROOT / "src" / "learning" / "__init__.py": "# learning package\n",
    ROOT / "src" / "virtues" / "__init__.py": "# virtues package\n",
    ROOT / "src" / "curriculum" / "__init__.py": "# curriculum package\n",

    # ----------------- src: core modules -----------------
    # agent_loop
    ROOT / "src" / "agent_loop" / "state.py": "# AgentPhase enum and agent loop state definitions\n",
    ROOT / "src" / "agent_loop" / "schema.py": "# AgentConfig, PlanState, and related dataclasses\n",
    ROOT / "src" / "agent_loop" / "loop.py": "# AgentLoopV1 implementation (observe → plan → act → recover)\n",

    # bot_core
    ROOT / "src" / "bot_core" / "core.py": "# BotCoreImpl: high-level interface for connecting and executing actions\n",
    ROOT / "src" / "bot_core" / "snapshot.py": "# RawWorldSnapshot + conversions to WorldState\n",
    ROOT / "src" / "bot_core" / "world_tracker.py": "# track chunks/entities and build snapshots from packets\n",
    ROOT / "src" / "bot_core" / "nav" / "grid.py": "# navigation grid abstraction based on world blocks\n",
    ROOT / "src" / "bot_core" / "nav" / "pathfinder.py": "# A* (or similar) pathfinding over NavGrid\n",
    ROOT / "src" / "bot_core" / "nav" / "mover.py": "# convert paths into low-level movement actions\n",
    ROOT / "src" / "bot_core" / "actions.py": "# ActionExecutor: map high-level Actions to protocol operations\n",
    ROOT / "src" / "bot_core" / "net" / "client.py": "# external protocol client for MC 1.7.10\n",
    ROOT / "src" / "bot_core" / "net" / "ipc.py": "# IPC bridge to in-process Forge mod (if used)\n",

    # env
    ROOT / "src" / "env" / "schema.py": "# EnvProfile, MinecraftProfile, ModelProfile dataclasses\n",
    ROOT / "src" / "env" / "loader.py": "# load environment config (env.yaml, minecraft.yaml, models.yaml, hardware.yaml)\n",

    # llm_stack
    ROOT / "src" / "llm_stack" / "schema.py": "# request/response types for planner, codegen, critic\n",
    ROOT / "src" / "llm_stack" / "backend.py": "# abstract backend interface for local LLM engines\n",
    ROOT / "src" / "llm_stack" / "backend_llamacpp.py": "# llama.cpp-specific backend implementation\n",
    ROOT / "src" / "llm_stack" / "planner.py": "# PlannerModel wrapper logic\n",
    ROOT / "src" / "llm_stack" / "codegen.py": "# CodeModel wrapper logic\n",
    ROOT / "src" / "llm_stack" / "critic.py": "# CriticModel wrapper logic\n",
    ROOT / "src" / "llm_stack" / "stack.py": "# LLMStack aggregator (planner, codegen, critic)\n",

    # monitoring
    ROOT / "src" / "monitoring" / "events.py": "# MonitoringEvent, EventType, ControlCommand definitions\n",
    ROOT / "src" / "monitoring" / "logger.py": "# JSON logger subscribing to EventBus\n",
    ROOT / "src" / "monitoring" / "bus.py": "# EventBus for monitoring events and control commands\n",
    ROOT / "src" / "monitoring" / "controller.py": "# AgentController linking control commands to AgentLoopV1\n",
    ROOT / "src" / "monitoring" / "dashboard_tui.py": "# rich-based TUI dashboard\n",

    # observation
    ROOT / "src" / "observation" / "schema.py": "# PlannerEncoding, CriticEncoding dataclasses\n",
    ROOT / "src" / "observation" / "encoder.py": "# encode_for_planner / encode_for_critic implementations\n",
    ROOT / "src" / "observation" / "trace_schema.py": "# TraceStep, PlanTrace definitions\n",

    # semantics
    ROOT / "src" / "semantics" / "schema.py": "# TechState, SemanticsDB, and related types\n",
    ROOT / "src" / "semantics" / "loader.py": "# load GTNH block/item/tech/recipe config into SemanticsDB\n",
    ROOT / "src" / "semantics" / "tech_state.py": "# infer_tech_state, suggest_next_targets\n",
    ROOT / "src" / "semantics" / "crafting.py": "# craftable_items and crafting helpers\n",
    ROOT / "src" / "semantics" / "categorize.py": "# item/block category helpers\n",

    # skills
    ROOT / "src" / "skills" / "schema.py": "# SkillSpec, ParamSpec, Preconditions, Effects\n",
    ROOT / "src" / "skills" / "registry.py": "# SkillRegistry implementation and decorators\n",
    ROOT / "src" / "skills" / "base" / "chop_tree.py": "# concrete chop_tree skill implementation stub\n",
    ROOT / "src" / "skills" / "base" / "feed_coke_ovens.py": "# concrete feed_coke_ovens skill implementation stub\n",

    # spec
    ROOT / "src" / "spec" / "types.py": "# core shared types: WorldState, Action, ActionResult, Observation\n",
    ROOT / "src" / "spec" / "bot_core.py": "# BotCore interface definition\n",
    ROOT / "src" / "spec" / "skills.py": "# SkillImplBase and skill-related interfaces\n",
    ROOT / "src" / "spec" / "llm.py": "# PlannerModel, CodeModel, CriticModel interfaces\n",
    ROOT / "src" / "spec" / "agent_loop.py": "# AgentLoop interface / contract\n",
    ROOT / "src" / "spec" / "experience.py": "# shared experience/trace interfaces (if needed)\n",

    # learning
    ROOT / "src" / "learning" / "schema.py": "# ExperienceEpisode, SkillPerformanceStats, SkillCandidate, etc.\n",
    ROOT / "src" / "learning" / "buffer.py": "# ExperienceBuffer implementation\n",
    ROOT / "src" / "learning" / "synthesizer.py": "# SkillSynthesizer using CodeModel\n",
    ROOT / "src" / "learning" / "evaluator.py": "# SkillEvaluator for metrics and comparisons\n",
    ROOT / "src" / "learning" / "manager.py": "# SkillLearningManager orchestration\n",

    # virtues
    ROOT / "src" / "virtues" / "schema.py": "# virtue nodes, contexts, metric schema\n",
    ROOT / "src" / "virtues" / "loader.py": "# load virtues.yaml into VirtueConfig\n",
    ROOT / "src" / "virtues" / "metrics.py": "# map plan+world into raw metrics\n",
    ROOT / "src" / "virtues" / "lattice.py": "# score_plan, compare_plans, weight merging\n",

    # curriculum
    ROOT / "src" / "curriculum" / "schema.py": "# PhaseConfig, CurriculumConfig, LongHorizonProject\n",
    ROOT / "src" / "curriculum" / "loader.py": "# load curriculum YAMLs from config/curricula\n",
    ROOT / "src" / "curriculum" / "engine.py": "# CurriculumEngine: select phase/goals/overrides\n",

    # ----------------- tests -----------------
    ROOT / "tests" / "__init__.py": "# tests package\n",
    ROOT / "tests" / "test_architecture_integration.py": "# high-level integration tests placeholder\n",
    ROOT / "tests" / "test_llm_stack_fake_backend.py": "# tests for llm_stack with fake backend\n",
    ROOT / "tests" / "test_observation_planner_encoding.py": "# tests for encode_for_planner\n",
    ROOT / "tests" / "test_observation_critic_encoding.py": "# tests for encode_for_critic\n",
    ROOT / "tests" / "test_agent_loop_v1.py": "# tests for AgentLoopV1 with fakes\n",

    ROOT / "tests" / "fakes" / "__init__.py": "# tests.fakes package\n",
    ROOT / "tests" / "fakes" / "fake_bot_core.py": "# fake BotCoreImpl for tests\n",
    ROOT / "tests" / "fakes" / "fake_llm_stack.py": "# fake LLMStack for tests\n",
    ROOT / "tests" / "fakes" / "fake_skills.py": "# fake skill implementations for tests\n",
}

# ---------------------------------------------------------------------------
# Optional: pyproject.toml stub (if you haven't created one yet)
# ---------------------------------------------------------------------------

pyproject_path = ROOT / "pyproject.toml"

pyproject_minimal = """[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "gtnh-agent"
version = "0.1.0"
description = "LLM-based GTNH agent."
requires-python = ">=3.10"
license = { text = "MIT" }

[project.dependencies]
# runtime deps go here

"""


def main() -> None:
    # Create all directories
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)

    # Create all files if they don't already exist
    for path, content in files.items():
        if not path.exists():
            path.write_text(content, encoding="utf-8")

    # Create minimal pyproject.toml if missing
    if not pyproject_path.exists():
        pyproject_path.write_text(pyproject_minimal, encoding="utf-8")


if __name__ == "__main__":
    main()
