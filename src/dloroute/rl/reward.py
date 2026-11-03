"""Insertion reward (paper Eq. 1-4) and local insertion success.

Conventions resolved here (see reports/deviation_log.md R3, R4):

* rope_in: head inside the target tunnel (entered forward, not yet out).
  rope_out: the transit nearest the head is a forward pass of the target clip.
* d_floor / d_ceil: planar distances from the head to the entry-face / exit-face
  centres. R (the region of the indicator) is the 0.16 m x 0.16 m insertion
  square in front of the entry face.
* ||p_{i+3} - p_i||_y is the absolute difference of local y coordinates, using
  sampled points p_0..p_6.
* Success: rope_out and signed distance > 2 cm after release and settling.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..interface import ClipSpec
from ..state import HeadMetrics, in_insertion_region
from .observation import ObsSpec


@dataclass(frozen=True)
class RewardParams:
    """Eq. 1-4 coefficients from the paper."""

    collision: float = -2.0  # beta
    horizon: float = -0.001  # gamma (per-step episode-length penalty)
    flat: float = 0.5  # eta
    insertion_bonus: float = 0.5
    success_margin: float = 0.02


@dataclass
class RewardTerms:
    insertion: float
    collision: float
    horizon: float
    dist: float
    flat: float

    @property
    def total(self) -> float:
        return self.insertion + self.collision + self.horizon + self.dist + self.flat


def reward(points: np.ndarray, target: ClipSpec, m: HeadMetrics, collided: bool, t: int,
           spec: ObsSpec = ObsSpec(), rp: RewardParams = RewardParams(), region: float = 0.16) -> RewardTerms:
    if m.rope_in and not m.rope_out:
        r_dist = 10.0 * m.d_floor
    elif m.rope_out:
        r_dist = 20.0 * m.d_ceil
    else:
        inside = in_insertion_region(target, points[0], region)
        r_dist = float(inside) / (4.0 + 80.0 * m.d_floor)
    p = target.pose.to_local(points[spec.point_indices()[:7]])
    dev = sum(abs(p[i + 3, 1] - p[i, 1]) for i in range(4))
    r_flat = 1.0 / (1.0 + (1000.0 / 3.0) * dev)
    return RewardTerms(
        insertion=rp.insertion_bonus * (float(m.rope_in) + float(m.rope_out)),
        collision=rp.collision * float(collided),
        horizon=rp.horizon * t,
        dist=r_dist,
        flat=rp.flat * r_flat,
    )


def insertion_success(m: HeadMetrics, rp: RewardParams = RewardParams()) -> bool:
    return m.rope_out and m.signed_dist > rp.success_margin
