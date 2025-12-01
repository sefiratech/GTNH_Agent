# src/curriculum/strategies.py

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional, Any

from spec.agent_loop import AgentGoal


class CurriculumStrategy(ABC):
    """
    Base class for curriculum strategies.

    A strategy decides:
      - which goal to select next
      - how to update after each episode
    """

    @abstractmethod
    def select_next_goal(self) -> Optional[AgentGoal]:
        raise NotImplementedError

    def on_episode_complete(
        self,
        *,
        goal: AgentGoal,
        task_plan: Any,
        episode_result: Any
    ) -> None:
        """
        Optional callback after each episode. Override if needed.
        """
        return


class StaticGoalStrategy(CurriculumStrategy):
    """
    Dumb baseline:
      - Always return the same goal.
      - Good for testing runtime.
    """

    def __init__(self, goal: AgentGoal):
        self._goal = goal

    def select_next_goal(self) -> Optional[AgentGoal]:
        return self._goal


class SequentialListStrategy(CurriculumStrategy):
    """
    Simple for v1:
      - Walk a static list of goals in order.
      - Loop forever or stop at end.
    """

    def __init__(self, goals: list[AgentGoal], *, loop: bool = True) -> None:
        self._goals = goals
        self._i = 0
        self._loop = loop

    def select_next_goal(self) -> Optional[AgentGoal]:
        if not self._goals:
            return None

        goal = self._goals[self._i]

        if self._i < len(self._goals) - 1:
            self._i += 1
        elif self._loop:
            self._i = 0
        else:
            # one-shot curriculum
            self._i += 1
            if self._i >= len(self._goals):
                return None

        return goal

    def on_episode_complete(self, *, goal, task_plan, episode_result) -> None:
        # v1 doesn’t need anything, but the hook is here.
        pass

