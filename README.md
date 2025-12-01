GTNH Agent – Modular LLM-Driven Cognitive Architecture for Complex Sandbox Automation

A fully modular, test-driven agent architecture designed to operate inside deeply complex, stateful sandbox worlds.
Originally built to automate GregTech: New Horizons, but structured as a general cognitive framework with pluggable semantics, skills, curricula, and LLM reasoning layers.

This project implements a complete end-to-end agent:

Observation → semantic normalization

World modeling & tech-state inference

LLM planning (planner / scribe / critic roles)

Skill execution pipelines

Curriculum-driven long-horizon learning

Experience memory

Monitoring + introspection tooling

Fully isolated, test-driven integration harness

You get a reusable architecture, not a hardcoded game bot.

Core Features
1. Modular Cognitive Stack

Each subsystem is isolated and swappable:

Observation pipeline (M7): encodes raw snapshots into structured planner/critic inputs

LLM stack (M6): planner, scribe, critic, error-model

Skill system (M5): declarative skills, versioning, registry, loader, integrity checks

Semantics engine (M3): items, recipes, tech-graph inference, categorization

Curriculum engine (M11 precursor): tasks, strategies, progression models

Agent Loop (M8): unified observe → plan → execute → summarize loop

Virtue lattice (M4): safety/quality alignment constraints on planning

Runtime (Phase 1 & 2): orchestration + integration logic

Monitoring layer (M9): event bus, TUI dashboard, audit tools

Everything is config-driven and environment-agnostic.

2. Two Execution Backends
Forge Mod Backend

Best for full sandbox introspection:

Direct access to world state

Rich entity + block + inventory snapshots

Minimal packet ambiguity

External Client Backend

Simpler debugging:

Runs as an external bot

No modified client needed

Limited data but easier networking

Backends are swappable via env.yaml.

3. Test-Covered Architecture

Over 120+ tests, including:

Core logic

LLM stack contracts

Observation encoding

Planning integration

Skill registry + loading

Curriculum policies

Tech-state inference

Integration smoke tests

World-model invariants

CI runs the full suite (minus optional llama.cpp tests).

4. Data-Driven Everything

The system loads:

Skills

Skill packs

Tech graph

Recipes

Environment profiles

LLM role configurations

Virtue constraints

…from YAML and JSON configs, keeping code clean and behavior declarative.

5. Reusable Beyond GTNH

Although this implementation ships with GTNH semantics, the architecture is:

domain-agnostic

environment-pluggable

driven by config + semantics adapters

designed to support any complex crafting or stateful environment

A future version will extract the reusable core into a standalone framework.

Project Structure (High-Level)

src/
  agent/          Core agent runtime + loop
  bot_core/       Movement, navigation, world snapshots, IPC/net
  observation/    Snapshot encoding → LLM-ready structures
  semantics/      Items, recipes, tech graph, inference
  llm_stack/      Planner / critic / scribe roles + error model
  skills/         Skill definitions, loader, registry, versioning
  curriculum/     Long-horizon planning + strategies
  virtues/        Alignment constraints, evaluator
  runtime/        Orchestration layers for integration
  monitoring/     Event bus, dashboards, tools
  world/          Predictive world-model placeholder

config/           YAML configs for skills, env, models, curricula
scripts/          Ingestion tools, generators, smoke tests
tests/            Full integration + unit test suite
tools/            Audit utilities, demos, CLI helpers

Running the Agent (Offline Mode)

Create a venv:
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt  # or use pyproject.toml

Run a full offline step:
python -m scripts.demo_offline_agent_step

Run full test suite:
pytest -q


Why This Project Matters

This repository demonstrates:

advanced modular architecture

LLM-tooling integration

curriculum-based reasoning

deep configuration-driven design

automated testing at scale

domain semantics ingestion + normalization

agent alignment via virtue constraints

reproducible orchestration and monitoring

It is a serious cognitive framework, not a toy bot.


License

MIT License (see LICENSE file).


Future Work

Extract a true general-purpose agent framework

Add persistent world-model learning

Integrate curriculum-based self-improvement

Add skill evolution + autonomous refinement

Externalize planning trace visualizations

Expand semantics adapters beyond GTNH
