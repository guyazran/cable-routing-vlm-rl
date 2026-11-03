"""VLM controller (protocol Section 7).

The VLM selects skill and target, monitors progress from images and its own
action history, and decides when to stop. Nothing advances automatically on
simulator success: the controller reads only rendered views. In the separately
labelled true-progress diagnostic (Section 12) a compact progress summary
derived from the cable state is appended to the prompt; it never prescribes a
skill.
"""

from __future__ import annotations

from ..decision import Decision
from ..planner.client import PlannerClient
from ..planner.prompts import PlannerState, PromptBuilder
from ..state import route_status
from .base import ControllerOutput, DecisionContext


def progress_summary(route: list[str], valid_prefix: int, complete: bool) -> str:
    done = route[:valid_prefix]
    nxt = route[valid_prefix] if valid_prefix < len(route) else None
    return (f"Currently valid ordered prefix: {', '.join(done) if done else '(none)'}.\n"
            f"First unsatisfied requested clip: {nxt or '(none)'}.\n"
            f"All requested threading currently valid and nothing else threaded: {'yes' if complete else 'no'}.")


class VLMController:
    def __init__(self, client: PlannerClient, prompts: PromptBuilder, route: list[str],
                 all_clips: list[str], episode_seed: int, privileged_progress: bool = False):
        self.name = "vlm+progress" if privileged_progress else "vlm"
        self.client = client
        self.prompts = prompts
        self.route = route
        self.all_clips = all_clips
        self.episode_seed = episode_seed
        self.privileged = privileged_progress
        self.reset()

    def reset(self) -> None:
        self.state = PlannerState(crop_clip=self.route[0])

    def decide(self, ctx: DecisionContext) -> ControllerOutput:
        v = ctx.world.views(self.state.crop_clip)
        summary = None
        if self.privileged:
            rs = route_status(ctx.world.state(), self.route)
            summary = progress_summary(self.route, rs.valid_prefix, rs.complete)
        task = self.prompts.task_text(self.route, self.all_clips, self.state, summary)
        messages = self.prompts.messages(task, v.full, v.crop)
        seed = self.episode_seed * 10_000 + 2 * ctx.decision_index
        rec = self.client.decide(messages, self.all_clips, seed)
        info = {"query": rec.to_dict(), "task_text": task, "crop": v.info,
                "consecutive_inserts": self.state.consecutive_inserts,
                "images": {"full": v.full, "crop": v.crop}}
        return ControllerOutput(rec.decision, rec.failure, info)

    def executed(self, decision: Decision) -> None:
        self.state.record(decision.skill, decision.target_clip)
