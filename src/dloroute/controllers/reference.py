"""Ground-truth state reference controller (protocol Section 7).

Deterministic state machine over the world's cable state (ground truth in
simulation), following the commanded route order. It issues the same
Insert/Pull/Flatten commands as the VLM, with identical primitive semantics, and
never moves particles, resets the scene or uses an extra skill. Predicates and
thresholds are reconstruction choices to be frozen on development scenes.

Rules v4 (2026-11-03, after the Pull semantics of R6b), applied in order at each
decision (k = currently valid ordered prefix, target = the first required clip
outside it, prev = the last clip inside it). Like the paper's symbolic planner they
choose Insert or Pull from the head position and Flatten after three failures:
 0. The requested route is complete (exactly the requested transits) -> done.
 1. The head passed prev but protrudes no more than the success margin ->
    Insert(prev), to finish the passage (a Pull would grasp beside prev).
 2. The head is in the target's tunnel or inside its insertion region:
    a. after ``max_failed_inserts`` consecutive Inserts at the target without
       progress -> Flatten(target) if its head grasp clears every clip, else
       Pull(target) if that grasp clears; this resets the count;
    b. otherwise Insert(target).
 3. Otherwise (the head is out of Insert's reach, e.g. just out of prev) ->
    Pull(target) if its grasp clears every clip, else Flatten(target) if the
    head grasp clears, else Insert(target).
The grasp checks use the world's grasp-collision query (privileged geometry, R14);
the skills themselves never check (R15). Wrong threading beyond the valid prefix
is handled by the same rules: the target stays the first unsatisfied clip, so a
regression returns the controller to the earlier unsatisfied clip.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


from ..decision import Decision
from ..interface import CableState, ClipSpec, SceneSpec
from ..skills import SkillParams, pull_grasp_index
from ..state import head_metrics, in_insertion_region, route_status
from .base import ControllerOutput, DecisionContext


@dataclass(frozen=True)
class ReferenceParams:
    max_failed_inserts: int = 3
    success_margin: float = 0.02


class ReferenceController:
    name = "reference"

    def __init__(self, scene: SceneSpec, route: list[str], skills: SkillParams,
                 params: ReferenceParams = ReferenceParams(), render_images: bool = True):
        self.clips = {c.clip_id: c for c in scene.clips}
        self.route = route
        self.sp = skills
        self.p = params
        self.render_images = render_images  # identical observation logging for both controllers
        self.reset()

    def reset(self) -> None:
        self.failed_inserts = 0
        self.last_target: str | None = None
        self.last_decision: Decision | None = None
        self.last_prefix = 0
        self.crop_clip = self.route[0]

    # --- controller interface ------------------------------------------------
    def decide(self, ctx: DecisionContext) -> ControllerOutput:
        info: dict = {}
        if self.render_images:
            v = ctx.world.views(self.crop_clip)
            info.update({"crop": v.info, "images": {"full": v.full, "crop": v.crop}})
        decision, rule = self.choose(ctx.world.state(), ctx.world.grasp_collides)
        info["rule"] = rule
        return ControllerOutput(decision, None, info)

    def executed(self, decision: Decision) -> None:
        self.crop_clip = decision.target_clip

    # --- decision ------------------------------------------------------------
    def choose(self, state: CableState, grasp_collides: Callable[[int], bool]) -> tuple[Decision, str]:
        """Return the decision and the rule that produced it (for the trace)."""
        points = state.points
        rs = route_status(state, self.route)
        k = rs.valid_prefix
        if rs.complete:
            return self._emit(Decision(None, None, True), "0-complete")
        target_id = self.route[k]
        prev = self.last_decision
        progressed = k > self.last_prefix
        if target_id != self.last_target or progressed or prev is None or prev.skill != "Insert":
            self.failed_inserts = 0
        elif prev.target_clip == target_id:
            self.failed_inserts += 1
        self.last_target, self.last_prefix = target_id, k

        prev_id = self.route[k - 1] if k > 0 else None
        if prev_id is not None:
            m_prev = head_metrics(self.clips[prev_id], state)
            if m_prev.rope_out and m_prev.signed_dist <= self.p.success_margin:
                return self._emit(Decision("Insert", prev_id, False), "1-finish-prev")
        target = self.clips[target_id]
        m = head_metrics(target, state)
        pull_clear = not grasp_collides(pull_grasp_index(points, self.sp))
        head_clear = not grasp_collides(0)  # Flatten's first grasp is the head
        if m.rope_in or in_insertion_region(target, points[0], self.sp.insertion_region):
            if self.failed_inserts >= self.p.max_failed_inserts:
                if head_clear:
                    return self._emit(Decision("Flatten", target_id, False), "2a-retry-flatten")
                if pull_clear:
                    return self._emit(Decision("Pull", target_id, False), "2a-retry-pull")
            return self._emit(Decision("Insert", target_id, False), "2b-insert")
        if pull_clear:
            return self._emit(Decision("Pull", target_id, False), "3-pull")
        if head_clear:
            return self._emit(Decision("Flatten", target_id, False), "3-flatten")
        return self._emit(Decision("Insert", target_id, False), "3-insert")

    def _emit(self, d: Decision, rule: str) -> tuple[Decision, str]:
        self.last_decision = d
        return d, rule
