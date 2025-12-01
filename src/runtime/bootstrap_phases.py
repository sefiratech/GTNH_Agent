# path: src/runtime/bootstrap_phases.py

"""
Bootstrap the full GTNH_Agent stack across Phases 0–3.

Phases:
- Phase 0: M0 (environment), M1 (validation/tools)
- Phase 1: M2 (LLM stack), M3 (semantics), M4 (virtues), M5 (skills)
- Phase 2: M6 (BotCore), M7 (observation encoding)
- Phase 3: M8 (AgentLoop), M9 (monitoring & tools)
"""

from __future__ import annotations  # forward references in type hints

import threading                    # for TUI thread
import time                         # for main loop pacing
from pathlib import Path            # for config & log paths
from typing import Tuple            # for clearer return annotations

# ---- Phase 0 imports (environment + validation) ------------------------------

from env_loader.core import load_env_profile         # M0: load + validate env.yaml/hardware.yaml/etc.
from tools.validate_env import validate_environment  # M1: sanity checks on paths / versions

# ---- Phase 1 imports (LLM stack, semantics, virtues, skills) -----------------

from llm_stack.builder import build_llm_stack        # M2: planner/critic/scribe/error-model clients
from semantics.tech_state import (                   # M3: semantics + tech graph
    load_semantic_models,
    TechStateEngine,
)
from virtues.engine import VirtueEngine              # M4: virtue lattice + scoring rules
from skills.registry import SkillRegistry            # M5: skill definitions + curriculum link

# ---- Phase 2 imports (BotCore + observation encoding) ------------------------

from bot_core.core import BotCore                    # M6: Minecraft IO & action engine
from observation.encoder import ObservationEncoder   # M7: planner/critic payload builder

# ---- Phase 3 imports (AgentLoop + monitoring) --------------------------------

from monitoring.bus import EventBus, default_bus     # M9: event bus
from monitoring.logger import JsonFileLogger         # M9: JSONL logger
from monitoring.controller import AgentController    # M9: control surface
from monitoring.dashboard_tui import TuiDashboard    # M9: HUD
from monitoring import integration as mon_int        # M9: integration helpers (emit_*)

from agent_loop.loop import AgentLoopV1              # M8: core agent loop

from runtime.error_handling import safe_step_with_logging  # runtime helper for step exceptions


# ------------------------------------------------------------------------------
# Phase 0 – environment & validation
# ------------------------------------------------------------------------------

def bootstrap_phase0(config_root: Path) -> "EnvProfile":
    """
    Phase 0:
    - Load environment configs (env.yaml, hardware.yaml, minecraft.yaml, models.yaml).
    - Perform validation to ensure everything is runnable before continuing.
    """
    env_profile = load_env_profile(config_root=config_root)
    validate_environment(env_profile)
    return env_profile


# ------------------------------------------------------------------------------
# Phase 1 – LLM stack, semantics, virtues, skills
# ------------------------------------------------------------------------------

def bootstrap_phase1(
    env_profile: "EnvProfile",
) -> Tuple["LlmStack", TechStateEngine, VirtueEngine, SkillRegistry]:
    """
    Phase 1:
    - Build LLM stack (planner, critic, scribe, etc.) using env/model config.
    - Load GTNH semantics & tech graph.
    - Initialize virtue engine with lattice + scoring rules.
    - Initialize skills registry + curricula.
    """
    # Build the LLM stack (M2) based on env + models configuration
    llm_stack = build_llm_stack(env_profile=env_profile)

    # Load semantics (M3): items, blocks, recipes, tech graph, etc.
    semantics_models = load_semantic_models(config_root=env_profile.config_root)

    # Create a TechStateEngine which can infer current tech tier / missing unlocks
    tech_state_engine = TechStateEngine(semantics_models=semantics_models)

    # Initialize virtue engine (M4) with virtue lattice & rules
    virtue_engine = VirtueEngine(config_root=env_profile.config_root)

    # Build skills registry (M5) from curricula + skill definitions
    skill_registry = SkillRegistry(config_root=env_profile.config_root)

    return llm_stack, tech_state_engine, virtue_engine, skill_registry


# ------------------------------------------------------------------------------
# Phase 2 – BotCore + observation encoding
# ------------------------------------------------------------------------------

def bootstrap_phase2(
    env_profile: "EnvProfile",
    tech_state_engine: TechStateEngine,
) -> Tuple[BotCore, ObservationEncoder]:
    """
    Phase 2:
    - Create BotCore (M6) wired to the Minecraft instance.
    - Create an ObservationEncoder (M7) that converts world state into planner/critic payloads.
    """
    bot_core = BotCore(
        mc_host=env_profile.minecraft.host,
        mc_port=env_profile.minecraft.port,
        profile=env_profile.minecraft,
    )

    obs_encoder = ObservationEncoder(
        tech_state_engine=tech_state_engine,
        semantics_models=tech_state_engine.semantics_models,
    )

    return bot_core, obs_encoder


# ------------------------------------------------------------------------------
# Phase 3 – monitoring stack (M9) and AgentLoop (M8)
# ------------------------------------------------------------------------------

def build_monitoring_stack(
    use_default_bus: bool = True,
    log_path: Path | None = None,
) -> Tuple[EventBus, JsonFileLogger, TuiDashboard]:
    """
    Phase 3 (M9):
    - Build EventBus and JsonFileLogger.
    - Optionally reuse default_bus so external tools can attach.
    - Start TUI dashboard in background.
    """
    if use_default_bus:
        bus = default_bus
    else:
        bus = EventBus()

    if log_path is None:
        log_path = Path("logs") / "monitoring" / "events.log"

    logger = JsonFileLogger(path=log_path, bus=bus)

    dashboard = TuiDashboard(bus)
    t = threading.Thread(
        target=lambda: dashboard.run(refresh_per_second=4.0),
        name="TuiDashboardThread",
    )
    t.daemon = True
    t.start()

    return bus, logger, dashboard


def bootstrap_phase3(
    env_profile: "EnvProfile",
    llm_stack: "LlmStack",
    tech_state_engine: TechStateEngine,
    virtue_engine: VirtueEngine,
    skill_registry: SkillRegistry,
    bot_core: BotCore,
    obs_encoder: ObservationEncoder,
    bus: EventBus,
) -> Tuple[AgentLoopV1, AgentController]:
    """
    Phase 3:
    - Build the AgentLoop (M8) with all its dependencies.
    - Wrap it in an AgentController (M9) to handle control commands.
    """
    agent_loop = AgentLoopV1(
        env_profile=env_profile,              # Phase 0: environment config
        llm_stack=llm_stack,                  # Phase 1: LLM clients
        tech_state_engine=tech_state_engine,  # Phase 1: semantics/tech inference
        virtue_engine=virtue_engine,          # Phase 1: virtue scoring
        skill_registry=skill_registry,        # Phase 1: skills/curricula
        bot_core=bot_core,                    # Phase 2: body
        obs_encoder=obs_encoder,              # Phase 2: perception / planner payloads
        event_bus=bus,                        # Phase 3: monitoring bus
        mon_integration=mon_int,              # Phase 3: emit_* helpers
    )

    controller = AgentController(agent=agent_loop, bus=bus)
    return agent_loop, controller


# ------------------------------------------------------------------------------
# Unified runtime main
# ------------------------------------------------------------------------------

def run_full_system(config_root: Path) -> None:
    """
    Bring up the entire GTNH_Agent system across Phases 0–3 and run the main loop.

    High-level:
    - Phase 0: load & validate environment.
    - Phase 1: build thinking stack (LLM, semantics, virtues, skills).
    - Phase 2: build body + perception (BotCore + ObservationEncoder).
    - Phase 3: build monitoring + AgentLoop + controller + TUI.
    """
    # Phase 0: environment & validation
    env_profile = bootstrap_phase0(config_root=config_root)

    # Phase 1: LLM stack, semantics, virtues, skills
    llm_stack, tech_state_engine, virtue_engine, skill_registry = bootstrap_phase1(env_profile)

    # Phase 2: BotCore + observation encoding
    bot_core, obs_encoder = bootstrap_phase2(env_profile, tech_state_engine)

    # Phase 3: monitoring stack (EventBus, logger, TUI)
    bus, logger, _dashboard = build_monitoring_stack(
        use_default_bus=True,  # share bus with monitoring CLI tools
        log_path=None,         # default logs/monitoring/events.log
    )

    # Phase 3: AgentLoop + AgentController
    agent_loop, controller = bootstrap_phase3(
        env_profile=env_profile,
        llm_stack=llm_stack,
        tech_state_engine=tech_state_engine,
        virtue_engine=virtue_engine,
        skill_registry=skill_registry,
        bot_core=bot_core,
        obs_encoder=obs_encoder,
        bus=bus,
    )

    # Main runtime loop
    try:
        while True:
            safe_step_with_logging(
                controller=controller,
                bus=bus,
                episode_id=None,   # TODO: thread real episode_id from agent_loop
                context_id=None,   # TODO: thread context_id from env/profile
            )
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Shutting down GTNH_Agent full system...")
    finally:
        logger.close()


if __name__ == "__main__":
    run_full_system(config_root=Path("config"))

