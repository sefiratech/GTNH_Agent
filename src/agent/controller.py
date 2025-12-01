from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from learning.buffer import ExperienceBuffer
from learning.manager import SkillLearningManager
from curriculum.manager import CurriculumManager
from curriculum.policy import SkillPolicy, SkillUsageMode
from agent.loop import AgentLoop, AgentLoopConfig
from agent.runtime_m6_m7 import AgentRuntime
from planning.dispatcher import PlanningDispatcher


@dataclass
class ControllerConfig:
    """
    Configuration for AgentController wiring.

    Fields
    ------
    replay_path:
        JSONL file where high-level Experience episodes are persisted
        for M10/M11 learning.

    skill_usage_mode:
        Initial skill usage policy for curriculum:
          - "stable_only"
          - "allow_candidates"
        String or SkillUsageMode enum; invalid values fall back to STABLE_ONLY.
    """
    replay_path: str = "data/replay/experience.jsonl"
    skill_usage_mode: str | SkillUsageMode = "stable_only"


class AgentController:
    """
    Orchestrates runtime, curriculum, learning, replay buffer, and AgentLoop.

    This is intentionally thin: it wires together objects that are constructed
    elsewhere (skills registry, synthesizer, evaluator, curriculum loader).

    The main responsibility here is:
      - build the shared ExperienceBuffer (M10)
      - build SkillLearningManager (M10)
      - build CurriculumManager (M11) with a SkillPolicy
      - build AgentRuntime (M6+M7)
      - build PlanningDispatcher (planner façade)
      - build AgentLoop (M8) with replay_buffer attached
    """

    def __init__(
        self,
        config: ControllerConfig,
        *,
        skills: Any,
        synthesizer: Any,
        evaluator: Any,
        curriculum_loader: Any,
        loop_config: Optional[AgentLoopConfig] = None,
        runtime_kwargs: Optional[dict[str, Any]] = None,
    ) -> None:
        """
        Parameters
        ----------
        config:
            ControllerConfig instance (replay path, skill usage mode).
        skills:
            SkillRegistry instance (M5) used by runtime, planner, learning.
        synthesizer:
            SkillSynthesizer instance (M10) for skill candidate generation.
        evaluator:
            SkillEvaluator instance (M10) for candidate evaluation / stats.
        curriculum_loader:
            Object responsible for loading curriculum configs / engines;
            passed through to CurriculumManager.
        loop_config:
            Optional AgentLoopConfig to override default loop behavior.
        runtime_kwargs:
            Optional extra kwargs forwarded to AgentRuntime constructor
            (e.g. semantics/world_model injection).
        """
        self.config = config

        # Normalize skill usage mode to enum with safe fallback.
        usage_raw = config.skill_usage_mode
        if isinstance(usage_raw, SkillUsageMode):
            usage_mode = usage_raw
        else:
            try:
                usage_mode = SkillUsageMode(str(usage_raw))
            except ValueError:
                usage_mode = SkillUsageMode.STABLE_ONLY

        # Replay buffer (M10) – JSONL-backed Experience storage.
        replay_path = Path(config.replay_path)
        self.replay_buffer = ExperienceBuffer(replay_path)

        # Learning manager (M10) – facade over replay buffer + skills.
        self.learning_manager = SkillLearningManager(
            buffer=self.replay_buffer,
            synthesizer=synthesizer,
            evaluator=evaluator,
            skills=skills,
            candidates_dir=Path("config/skills_candidates"),
        )

        # Skill policy (M11) – stable-only vs candidate-inclusive usage.
        self.skill_policy = SkillPolicy(usage_mode=usage_mode)

        # Curriculum manager (M11) – bridge between curriculum engine and learning.
        self.curriculum = CurriculumManager(
            learning=self.learning_manager,
            policy=self.skill_policy,
            loader=curriculum_loader,
        )

        # Runtime (M6+M7) – bot core + observation pipeline + tools.
        runtime_kwargs = runtime_kwargs or {}
        self.runtime = AgentRuntime(skills=skills, **runtime_kwargs)

        # Planner dispatcher – filters skills and adapts to planner API.
        self.dispatcher = PlanningDispatcher(skills=skills)

        # AgentLoop (M8) – hierarchical planning, execution, and self-eval.
        loop_cfg = loop_config or AgentLoopConfig()
        self.loop = AgentLoop(
            runtime=self.runtime,
            planner=self.dispatcher,
            curriculum=self.curriculum,
            skills=skills,
            replay_buffer=self.replay_buffer,
            config=loop_cfg,
        )

    # ------------------------------------------------------------------
    # Public control surface
    # ------------------------------------------------------------------

    def run_episode(self, episode_id: Optional[int] = None):
        """
        Run a single episode through the AgentLoop and return its EpisodeResult.
        """
        return self.loop.run_episode(episode_id=episode_id)

    def run_episodes(self, num_episodes: int) -> list[Any]:
        """
        Run N episodes sequentially.

        Returns a list of EpisodeResult objects (one per episode).
        """
        results: list[Any] = []
        for i in range(num_episodes):
            result = self.loop.run_episode(episode_id=i)
            results.append(result)
        return results

