"""The hierarchical agent: a high-level controller choosing (skill, clip) or completion,
and the shared skill library executing the choice in the world.

The agent takes one decision and executes one skill at a time. The caller owns the
evaluation protocol around it: budgets, termination, grading and logging.
"""

from __future__ import annotations

from .controllers.base import Controller, ControllerOutput, DecisionContext
from .decision import Decision
from .interface import World
from .skills import SkillLibrary, SkillOutcome


class HierarchicalAgent:
    def __init__(self, controller: Controller, skills: SkillLibrary):
        self.controller = controller
        self.skills = skills

    @property
    def name(self) -> str:
        return self.controller.name

    def reset(self) -> None:
        self.controller.reset()

    def decide(self, world: World, decision_index: int, used: int, budget: int) -> ControllerOutput:
        return self.controller.decide(DecisionContext(decision_index, used, budget, world))

    def act(self, decision: Decision, world: World) -> SkillOutcome:
        outcome = self.skills.execute(decision, world)
        self.controller.executed(decision)
        return outcome
