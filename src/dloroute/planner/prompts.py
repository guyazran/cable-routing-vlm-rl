"""Planner request assembly (protocol Section 6).

Memory policy: every decision is one fresh request holding the system prompt, the
fixed in-context examples, the task, the complete structured action history and
only the CURRENT full view and crop. Earlier images and earlier reasoning are
never sent back (preserve_thinking=false). Simulator success labels, true
progress and the reference controller's decisions are never included, except in
the separately labelled true-progress diagnostic (``privileged_progress``).
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

PROMPT_DIR = Path(__file__).resolve().parents[3] / "prompts"


def png_data_url(img: np.ndarray | Image.Image) -> str:
    im = img if isinstance(img, Image.Image) else Image.fromarray(np.asarray(img))
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


@dataclass
class HistoryEntry:
    step: int
    skill: str
    target_clip: str

    def render(self) -> str:
        return f"{self.step}. {self.skill}({self.target_clip})"


@dataclass
class PlannerState:
    """Bookkeeping visible to the planner. Contains no geometric success signal."""

    history: list[HistoryEntry] = field(default_factory=list)
    consecutive_inserts: int = 0
    counter_target: str | None = None
    crop_clip: str = "C01"

    def record(self, skill: str, target: str) -> None:
        self.history.append(HistoryEntry(len(self.history) + 1, skill, target))
        if skill == "Insert" and target == self.counter_target:
            self.consecutive_inserts += 1
        elif skill == "Insert":
            self.consecutive_inserts, self.counter_target = 1, target
        else:
            self.consecutive_inserts, self.counter_target = 0, None
        self.crop_clip = target


@dataclass(frozen=True)
class PromptConfig:
    local_horizon: int = 7
    max_consecutive_inserts: int = 3
    allow_missing_examples: bool = False  # only for serving smoke tests


class PromptBuilder:
    def __init__(self, cfg: PromptConfig = PromptConfig(), prompt_dir: Path = PROMPT_DIR):
        self.cfg = cfg
        self.dir = prompt_dir
        self.system = (prompt_dir / "system.md").read_text().format(
            local_horizon=cfg.local_horizon, max_consecutive_inserts=cfg.max_consecutive_inserts)
        self.task_tmpl = (prompt_dir / "task.md").read_text()
        self.examples = json.loads((prompt_dir / "examples.json").read_text())
        self._example_parts = self._load_example_parts()

    def _load_example_parts(self) -> list[dict]:
        parts: list[dict] = [{"type": "text", "text": "## In-context examples"}]
        for ex in self.examples["image_examples"]:
            path = self.dir / ex["image"]
            if path.exists():
                parts.append({"type": "image_url", "image_url": {"url": png_data_url(Image.open(path).convert("RGB"))}})
            elif not self.cfg.allow_missing_examples:
                raise FileNotFoundError(f"in-context example image missing: {path} (render it from a development scene)")
            parts.append({"type": "text", "text": ex["text"]})
        for text in self.examples["verbal_examples"]:
            parts.append({"type": "text", "text": text})
        return parts

    def version_hash(self) -> str:
        h = hashlib.sha256()
        for name in ("system.md", "task.md", "examples.json"):
            h.update((self.dir / name).read_bytes())
        for ex in self.examples["image_examples"]:
            p = self.dir / ex["image"]
            if p.exists():
                h.update(p.read_bytes())
        return h.hexdigest()[:16]

    def task_text(self, route: list[str], all_clips: list[str], state: PlannerState,
                  privileged_progress: str | None = None) -> str:
        others = [c for c in all_clips if c not in route]
        if others:
            scope = (f"Only the {len(route)} clips listed above are required. The other visible clips "
                     f"({', '.join(others)}) are obstacles outside the requested route: do not thread them. "
                     f"The task is complete once the listed clips are threaded in order, not when all visible clips are.")
        else:
            scope = "All visible clips are part of the requested route."
        history = "\n".join(h.render() for h in state.history) or "(none yet)"
        text = self.task_tmpl.format(
            route_list=" -> ".join(route), scope_note=scope, history=history,
            consecutive_inserts=state.consecutive_inserts, counter_target=state.counter_target or "-",
            crop_clip=state.crop_clip)
        if privileged_progress is not None:
            text += ("\n## Simulator progress summary (privileged diagnostic information)\n"
                     + privileged_progress + "\n")
        return text

    def messages(self, task_text: str, scene_img, crop_img) -> list[dict]:
        user = list(self._example_parts) + [
            {"type": "text", "text": task_text},
            {"type": "image_url", "image_url": {"url": png_data_url(scene_img)}},
            {"type": "image_url", "image_url": {"url": png_data_url(crop_img)}},
            {"type": "text", "text": "Reason about the current state, then give your decision as the final JSON object."},
        ]
        return [{"role": "system", "content": self.system}, {"role": "user", "content": user}]
