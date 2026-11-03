"""Common high-level controller interface."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from ..decision import Decision
from ..interface import World


@dataclass
class DecisionContext:
    decision_index: int
    used: int  # physical skill invocations so far
    budget: int
    # Controllers read only what their information boundary allows: the VLM controller renders
    # views; the cable state and the grasp query are privileged (reference, true-progress diagnostic).
    world: World


@dataclass
class ControllerOutput:
    decision: Decision | None
    failure: str | None = None  # interface failure category; None on a valid decision
    info: dict = field(default_factory=dict)


class Controller(Protocol):
    name: str

    def reset(self) -> None: ...
    def decide(self, ctx: DecisionContext) -> ControllerOutput: ...
    def executed(self, decision: Decision) -> None: ...
