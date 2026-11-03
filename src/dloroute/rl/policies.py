"""Insertion policies: one observation (rl.observation) -> one action in [-1, 1]^7
(decoded by skills.decode_insert_action).

* :func:`load_sac_policy` - the frozen SAC checkpoint (deterministic actor mean).
* :class:`HeuristicInsertPolicy` - the paper's heuristic baseline, expressed in the
  same action space: place the head straight in front of the clip, then regrasp
  further along the cable and push it along the clip centreline. Used as the
  optional Table I baseline and to smoke-test the pipeline before a checkpoint
  exists; never a substitute for the learned policy in the primary study.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .observation import ObsSpec

Policy = Callable[[np.ndarray], np.ndarray]


class SACPolicy:
    """The frozen SAC actor. Calling it gives the deterministic action, tanh(mean)."""

    def __init__(self, model):
        self.model = model

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        action, _ = self.model.predict(obs, deterministic=True)
        return np.asarray(action, dtype=float)


def load_sac_policy(path: str, device: str = "cpu") -> SACPolicy:
    from stable_baselines3 import SAC

    return SACPolicy(SAC.load(path, device=device))


@dataclass
class HeuristicInsertPolicy:
    """Two-phase heuristic. Positions are in the clip frame (obs / pos_scale undone).

    Phase "place": grasp the head and set it ``stage`` in front of the entry face
    on the axis. Phase "push" (once the head is staged): grasp the point ``push_grasp`` behind the head
    and move it from behind to ``finger_clearance`` before the entry face, which
    pushes the head through by roughly push_grasp - finger_clearance.
    """

    spec: ObsSpec = ObsSpec()
    clip_depth: float = 0.015  # unscaled
    region: float = 0.16
    stage: float = 0.03
    push_grasp_index: int = 3  # 7.5 cm behind the head
    finger_clearance: float = 0.014
    staged_tol: float = 0.015

    def _encode(self, idx: int, vias: list[tuple[float, float, float]], entry: float) -> np.ndarray:
        a = [2 * idx / (self.spec.n_points - 1) - 1]
        for x, y, yaw in vias:
            a += [2 * (1 - (entry - x) / self.region) - 1, y / (self.region / 2), yaw / (np.pi / 2)]
        return np.clip(np.array(a), -1, 1)

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        n = self.spec.n_points
        pts = obs[: 3 * n].reshape(n, 3) * self.spec.pos_scale
        rope_in = obs[-1] > 0.5
        entry = -self.clip_depth * float(obs[-2]) / 2  # obs[-2] is the clip scale
        head = pts[0]
        # The paper's heuristic is open-loop: place the head, then regrasp and push.
        # Push as soon as the head sits at the staging point (or is already inside).
        staged = abs(head[1]) < self.staged_tol and entry - self.stage - self.staged_tol < head[0] < entry
        if staged or rope_in:
            x_end = entry - self.finger_clearance
            x_start = max(x_end - 0.06, entry - self.region)
            return self._encode(self.push_grasp_index, [(x_start, 0.0, 0.0), (x_end, 0.0, 0.0)], entry)
        x = entry - self.stage
        return self._encode(0, [(x - 0.03, 0.0, 0.0), (x, 0.0, 0.0)], entry)
