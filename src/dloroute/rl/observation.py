"""Insertion-policy observation.

Conventions resolved here (the paper leaves them open; see reports/deviation_log.md R1):

* Observation points are K = 16 centreline points sampled every ``stride`` = 2
  points (2.5 cm) from the head tip, p_0 = head. The sampling is defined by arc
  length from the head, so it does not depend on the total cable length.
* Observation vector (all in the target clip's local frame, positions divided by
  ``pos_scale``): K x (x, y, z), clip scale, rope_in -> 3K + 2 = 50 values. The
  clip pose is the identity in its own frame and is therefore not repeated.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..interface import ClipSpec


@dataclass(frozen=True)
class ObsSpec:
    n_points: int = 16
    stride: int = 2
    pos_scale: float = 0.16

    @property
    def dim(self) -> int:
        return 3 * self.n_points + 2

    def point_indices(self) -> np.ndarray:
        return np.arange(self.n_points) * self.stride


def observation(points: np.ndarray, target: ClipSpec, rope_in: bool, spec: ObsSpec = ObsSpec()) -> np.ndarray:
    local = target.pose.to_local(points[spec.point_indices()])
    return np.concatenate([local.ravel() / spec.pos_scale, [target.scale, float(rope_in)]]).astype(np.float32)
